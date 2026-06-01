"""
Sasin Lecture Capture — FastAPI Transcription Server v3
- Groq STT API with persistent HTTP/2 keep-alive (warm-up)
- Deepgram fallback
- Gemini Vision for AI slide analysis
Receives audio chunks, transcribes via Groq Whisper-large-v3, returns text.
Stores transcripts to disk for later retrieval.
"""
import os
import json
import asyncio
import logging
import base64
import tempfile
import subprocess
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import httpx

app = FastAPI(title="Sasin Lecture Capture")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Config
TRANSCRIPTS_DIR = Path("/root/lecture_transcripts")
TRANSCRIPTS_DIR.mkdir(exist_ok=True)

GROQ_API_KEY = os.environ["GROQ_API_KEY"]
GROQ_TRANSCRIBE_URL = "https://api.groq.com/openai/v1/audio/transcriptions"
GROQ_MODEL = "whisper-large-v3"

DEEPGRAM_API_KEY = os.environ["DEEPGRAM_API_KEY"]
DEEPGRAM_TRANSCRIBE_URL = "https://api.deepgram.com/v1/listen?model=nova-2&smart_format=true"

GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]
GEMINI_VISION_MODEL = "gemini-2.5-flash"
GEMINI_VISION_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_VISION_MODEL}:generateContent"

TRANSCRIBE_TIMEOUT = 25

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("capture")

# Persistent HTTP/2 Client
_http_client: httpx.AsyncClient | None = None


class SaveRequest(BaseModel):
    transcript: str
    word_count: int
    duration: int
    chunks: int
    timestamp: str
    tags: str = ""


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "engine": "groq",
        "model": GROQ_MODEL,
        "fallback": "deepgram/nova-2",
        "transcripts_count": len(list(TRANSCRIPTS_DIR.glob("*.json"))),
    }


@app.post("/transcribe")
async def transcribe(audio: UploadFile = File(...), chunk_index: int = Form(0)):
    if not audio.filename:
        raise HTTPException(400, "No audio file provided")

    suffix = Path(audio.filename).suffix or ".webm"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        raw_audio = await audio.read()
        tmp.write(raw_audio)
        tmp_path = tmp.name

    wav_path = tmp_path
    try:
        if suffix not in (".wav",):
            wav_path = tmp_path + ".wav"
            loop = asyncio.get_event_loop()
            await asyncio.wait_for(
                loop.run_in_executor(None, _convert_audio, tmp_path, wav_path),
                timeout=TRANSCRIBE_TIMEOUT,
            )

        text = await asyncio.wait_for(
            _transcribe_with_fallback(wav_path),
            timeout=TRANSCRIBE_TIMEOUT,
        )

        return {
            "text": text,
            "chunk_index": chunk_index,
            "engine": getattr(_transcribe_with_fallback, "_last_engine", "groq"),
        }

    except asyncio.TimeoutError:
        logger.warning(f"Chunk {chunk_index} timed out after {TRANSCRIBE_TIMEOUT}s")
        return {"text": "", "chunk_index": chunk_index, "error": "timeout"}
    except Exception as e:
        logger.error(f"Chunk {chunk_index} failed: {e}")
        return {"text": "", "chunk_index": chunk_index, "error": str(e)[:200]}

    finally:
        for p in [tmp_path, wav_path]:
            if os.path.exists(p):
                try:
                    os.unlink(p)
                except OSError:
                    pass


def _convert_audio(src: str, dst: str):
    subprocess.run(
        ["ffmpeg", "-y", "-i", src, "-ar", "16000", "-ac", "1", "-f", "wav", dst],
        capture_output=True,
        timeout=TRANSCRIBE_TIMEOUT,
    )


async def _groq_transcribe(wav_path: str) -> str:
    global _http_client
    if _http_client is None:
        _http_client = httpx.AsyncClient(
            http2=True,
            timeout=httpx.Timeout(30.0),
            limits=httpx.Limits(max_keepalive_connections=5, max_connections=20),
        )

    with open(wav_path, "rb") as f:
        files = {"file": (os.path.basename(wav_path), f, "audio/wav")}
        data = {"model": GROQ_MODEL, "response_format": "json"}

        resp = await _http_client.post(
            GROQ_TRANSCRIBE_URL,
            headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
            files=files,
            data=data,
        )

    if resp.status_code != 200:
        logger.error(f"Groq API error {resp.status_code}: {resp.text[:300]}")
        return ""

    result = resp.json()
    return result.get("text", "").strip()


async def _deepgram_transcribe(wav_path: str) -> str:
    global _http_client
    if _http_client is None:
        _http_client = httpx.AsyncClient(
            http2=True,
            timeout=httpx.Timeout(30.0),
            limits=httpx.Limits(max_keepalive_connections=5, max_connections=20),
        )

    with open(wav_path, "rb") as f:
        resp = await _http_client.post(
            DEEPGRAM_TRANSCRIBE_URL,
            headers={
                "Authorization": f"Token {DEEPGRAM_API_KEY}",
                "Content-Type": "audio/wav",
            },
            content=f.read(),
        )

    if resp.status_code != 200:
        logger.error(f"Deepgram API error {resp.status_code}: {resp.text[:300]}")
        return ""

    result = resp.json()
    try:
        return result["results"]["channels"][0]["alternatives"][0]["transcript"].strip()
    except (KeyError, IndexError):
        return ""


async def _transcribe_with_fallback(wav_path: str) -> str:
    try:
        text = await _groq_transcribe(wav_path)
        if text:
            _transcribe_with_fallback._last_engine = "groq"
            return text
    except Exception as e:
        logger.warning(f"Groq failed, trying Deepgram: {e}")

    try:
        text = await _deepgram_transcribe(wav_path)
        if text:
            _transcribe_with_fallback._last_engine = "deepgram"
            return text
    except Exception as e:
        logger.error(f"Deepgram also failed: {e}")

    _transcribe_with_fallback._last_engine = "none"
    return ""


@app.post("/save")
async def save_transcript(req: SaveRequest):
    filename = f"lecture_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    filepath = TRANSCRIPTS_DIR / filename

    data = {
        "timestamp": req.timestamp,
        "duration_seconds": req.duration,
        "word_count": req.word_count,
        "chunks": req.chunks,
        "transcript": req.transcript,
        "engine": "groq",
        "tags": req.tags,
    }

    with open(filepath, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    txt_path = filepath.with_suffix(".txt")
    with open(txt_path, "w") as f:
        f.write(req.transcript)

    ingest_result = None
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            files = {"file": (f"{filename}.txt", req.transcript.encode("utf-8"), "text/plain")}
            resp = await client.post(
                "http://localhost:8899/ingest",
                files=files,
                data={"file_type": "text", "tags": req.tags or "lecture-transcript"},
            )
            if resp.status_code in (200, 201):
                ingest_result = resp.json()
    except Exception:
        pass

    return {
        "saved": str(filepath),
        "text_file": str(txt_path),
        "deep_tutor_ingested": ingest_result is not None,
        "ingest_result": ingest_result,
    }


# ─── AI Slide Analysis (Gemini Vision) ───

SLIDE_ANALYSIS_PROMPT = """Analyze this lecture slide. Return a JSON object with:
- "text": all visible text extracted from the slide
- "questions": array of questions found on the slide (empty array if none)
- "answers": array of concise answers to each question (empty array if none)
- "summary": one-sentence summary of the slide content

Keep answers under 2 sentences each. Use the same language as the slide."""


@app.post("/analyze-slide")
async def analyze_slide(image: UploadFile = File(...)):
    global _http_client
    if _http_client is None:
        _http_client = httpx.AsyncClient(
            http2=True,
            timeout=httpx.Timeout(30.0),
            limits=httpx.Limits(max_keepalive_connections=5, max_connections=20),
        )

    image_bytes = await image.read()
    mime_type = image.content_type or "image/jpeg"
    img_b64 = base64.b64encode(image_bytes).decode()

    payload = {
        "contents": [{
            "parts": [
                {"text": SLIDE_ANALYSIS_PROMPT},
                {"inline_data": {"mime_type": mime_type, "data": img_b64}}
            ]
        }]
    }

    try:
        resp = await _http_client.post(
            f"{GEMINI_VISION_URL}?key={GEMINI_API_KEY}",
            json=payload,
            timeout=15.0,
        )

        if resp.status_code != 200:
            logger.error(f"Gemini API error {resp.status_code}: {resp.text[:300]}")
            return {"error": f"AI analysis failed (HTTP {resp.status_code})", "text": "", "questions": [], "answers": [], "summary": ""}

        result = resp.json()
        raw_text = result["candidates"][0]["content"]["parts"][0]["text"]

        try:
            clean = raw_text.strip()
            if clean.startswith("```"):
                clean = clean.split("\n", 1)[1].rsplit("\n```", 1)[0] if "```" in clean else clean
            parsed = json.loads(clean)
            return {
                "text": parsed.get("text", ""),
                "questions": parsed.get("questions", []),
                "answers": parsed.get("answers", []),
                "summary": parsed.get("summary", ""),
            }
        except (json.JSONDecodeError, KeyError):
            return {"text": raw_text, "questions": [], "answers": [], "summary": ""}

    except Exception as e:
        logger.error(f"Slide analysis failed: {e}")
        return {"error": str(e)[:200], "text": "", "questions": [], "answers": [], "summary": ""}


@app.get("/sessions")
async def list_sessions():
    sessions = []
    for f in sorted(TRANSCRIPTS_DIR.glob("*.json"), reverse=True):
        with open(f) as fp:
            data = json.load(fp)
        sessions.append({
            "filename": f.name,
            "timestamp": data.get("timestamp", ""),
            "duration_seconds": data.get("duration_seconds", 0),
            "word_count": data.get("word_count", 0),
            "chunks": data.get("chunks", 0),
            "engine": data.get("engine", "local"),
        })
    return {"sessions": sessions}


@app.get("/sessions/{filename}")
async def get_session(filename: str):
    filepath = TRANSCRIPTS_DIR / filename
    if not filepath.exists():
        raise HTTPException(404, "Session not found")
    with open(filepath) as f:
        return json.load(f)


# Serve capture UI
_HTML_PATH = "/root/capture-index.html"
if os.path.exists(_HTML_PATH):
    from fastapi.responses import FileResponse

    @app.get("/")
    async def serve_capture():
        return FileResponse(_HTML_PATH)


@app.on_event("startup")
async def startup():
    global _http_client
    _http_client = httpx.AsyncClient(
        http2=True,
        timeout=httpx.Timeout(30.0),
        limits=httpx.Limits(max_keepalive_connections=5, max_connections=20),
    )

    import io as _io, wave as _wave
    sample_rate = 16000
    num_samples = int(sample_rate * 0.1)
    buf = _io.BytesIO()
    with _wave.open(buf, "w") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(b"\x00\x00" * num_samples)
    silent_wav = buf.getvalue()

    print(f"Warming up Groq connection ({GROQ_MODEL})...")
    try:
        resp = await _http_client.post(
            GROQ_TRANSCRIBE_URL,
            headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
            files={"file": ("silent.wav", silent_wav, "audio/wav")},
            data={"model": GROQ_MODEL, "response_format": "json"},
            timeout=15.0,
        )
        if resp.status_code == 200:
            print(f"  ✅ Groq connection warmed up ({resp.elapsed.total_seconds():.2f}s)")
        else:
            print(f"  ⚠️  Groq warm-up returned {resp.status_code}")
    except Exception as e:
        print(f"  ⚠️  Groq warm-up failed: {e}")

    print("Warming up Deepgram connection (nova-2)...")
    try:
        resp = await _http_client.post(
            DEEPGRAM_TRANSCRIBE_URL,
            headers={"Authorization": f"Token {DEEPGRAM_API_KEY}", "Content-Type": "audio/wav"},
            content=silent_wav,
            timeout=15.0,
        )
        if resp.status_code == 200:
            print(f"  ✅ Deepgram connection warmed up ({resp.elapsed.total_seconds():.2f}s)")
        else:
            print(f"  ⚠️  Deepgram warm-up returned {resp.status_code}")
    except Exception as e:
        print(f"  ⚠️  Deepgram warm-up failed: {e}")

    print(f"Capture server ready (engine: groq/{GROQ_MODEL}, fallback: deepgram/nova-2)")


@app.on_event("shutdown")
async def shutdown():
    global _http_client
    if _http_client:
        await _http_client.aclose()
        _http_client = None


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8898))
    uvicorn.run(app, host="0.0.0.0", port=port)
