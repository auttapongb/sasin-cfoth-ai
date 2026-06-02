"""
Sasin Lecture Capture — FastAPI Server v4
Two-Box Architecture:
  Box 1: Raw transcript (STT) + unanalyzed slides
  Box 2: AI Co-Learner insights (slide analysis, thinking out loud, answers)

Features:
  - Groq STT with Deepgram fallback
  - Gemini Vision for slide analysis → routed to Box 1 or 2
  - AI Co-Learner agent (DeepSeek) that studies alongside
  - SQLite persistence for all entries
  - Timestamp-based linking between boxes
  - Session management (save/load)
"""

import os
import json
import asyncio
import sqlite3
import logging
import base64
import tempfile
import subprocess
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse
from pydantic import BaseModel
import uvicorn
import httpx

app = FastAPI(title="Sasin Lecture Capture v4")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Config ──
TRANSCRIPTS_DIR = Path("/root/lecture_transcripts")
TRANSCRIPTS_DIR.mkdir(exist_ok=True)

DB_PATH = Path("/root/sasin-capture.db")

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_TRANSCRIBE_URL = "https://api.groq.com/openai/v1/audio/transcriptions"
GROQ_MODEL = "whisper-large-v3"

DEEPGRAM_API_KEY = os.environ.get("DEEPGRAM_API_KEY", "")
DEEPGRAM_TRANSCRIBE_URL = "https://api.deepgram.com/v1/listen?model=nova-2&smart_format=true"

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL = "gemini-2.5-flash"
GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"

# AI Co-Learner: uses DeepSeek direct API
COLEARNER_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
COLEARNER_MODEL = os.environ.get("COLEARNER_MODEL", "deepseek-chat")
COLEARNER_URL = os.environ.get("COLEARNER_URL", "https://api.deepseek.com/v1/chat/completions")

TRANSCRIBE_TIMEOUT = 25
COLEARNER_TIMEOUT = 15
_last_engine_used = "groq"
_last_colearner_time = 0.0
COLEARNER_COOLDOWN_SEC = 20  # server-side cooldown

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("capture")

_http_client: httpx.AsyncClient | None = None


# ═══════════════════════════════════════════
# DATABASE
# ═══════════════════════════════════════════

def init_db():
    """Create tables if they don't exist."""
    with sqlite3.connect(str(DB_PATH)) as conn:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                title TEXT DEFAULT '',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                tags TEXT DEFAULT '',
                status TEXT DEFAULT 'active'
            );

            CREATE TABLE IF NOT EXISTS entries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                box TEXT NOT NULL CHECK(box IN ('transcript','colearner','slide')),
                content TEXT NOT NULL,
                content_type TEXT DEFAULT 'text',
                timestamp_iso TEXT NOT NULL,
                elapsed_sec REAL NOT NULL DEFAULT 0,
                linked_entry_id INTEGER,
                metadata TEXT DEFAULT '{}',
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                FOREIGN KEY (session_id) REFERENCES sessions(id),
                FOREIGN KEY (linked_entry_id) REFERENCES entries(id)
            );

            CREATE INDEX IF NOT EXISTS idx_entries_session ON entries(session_id);
            CREATE INDEX IF NOT EXISTS idx_entries_box ON entries(session_id, box);
            CREATE INDEX IF NOT EXISTS idx_entries_timestamp ON entries(session_id, timestamp_iso);
        """)
        conn.commit()

init_db()


def db_execute(sql: str, params: tuple = ()) -> int:
    """Execute SQL and return lastrowid."""
    with sqlite3.connect(str(DB_PATH)) as conn:
        conn.execute("PRAGMA journal_mode=WAL")
        cur = conn.execute(sql, params)
        conn.commit()
        return cur.lastrowid


def db_fetch(sql: str, params: tuple = ()) -> list[dict]:
    """Fetch rows as dicts."""
    with sqlite3.connect(str(DB_PATH)) as conn:
        conn.row_factory = sqlite3.Row
        return [dict(row) for row in conn.execute(sql, params).fetchall()]


def create_session(session_id: str | None = None) -> str:
    sid = session_id or datetime.now(timezone.utc).strftime("session_%Y%m%d_%H%M%S_%f")
    now = datetime.now(timezone.utc).isoformat()
    existing = db_fetch("SELECT id FROM sessions WHERE id = ?", (sid,))
    if not existing:
        db_execute(
            "INSERT INTO sessions (id, created_at, updated_at) VALUES (?, ?, ?)",
            (sid, now, now),
        )
    return sid


def add_entry(session_id: str, box: str, content: str, content_type: str = "text",
              timestamp_iso: str = "", elapsed_sec: float = 0,
              linked_entry_id: int | None = None, metadata: dict | None = None) -> int:
    if not timestamp_iso:
        timestamp_iso = datetime.now(timezone.utc).isoformat()
    # Auto-create session if needed
    create_session(session_id)
    entry_id = db_execute(
        """INSERT INTO entries (session_id, box, content, content_type, timestamp_iso,
           elapsed_sec, linked_entry_id, metadata)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (session_id, box, content, content_type, timestamp_iso, elapsed_sec,
         linked_entry_id, json.dumps(metadata or {})),
    )
    # Update session timestamp
    db_execute(
        "UPDATE sessions SET updated_at = ? WHERE id = ?",
        (datetime.now(timezone.utc).isoformat(), session_id),
    )
    return entry_id


def get_entries(session_id: str, box: str | None = None) -> list[dict]:
    if box:
        return db_fetch(
            "SELECT * FROM entries WHERE session_id = ? AND box = ? ORDER BY id ASC",
            (session_id, box),
        )
    return db_fetch(
        "SELECT * FROM entries WHERE session_id = ? ORDER BY id ASC",
        (session_id,),
    )


# ═══════════════════════════════════════════
# MODELS
# ═══════════════════════════════════════════

class SaveRequest(BaseModel):
    transcript: str
    word_count: int
    duration: int
    chunks: int
    timestamp: str
    tags: str = ""


class EntryRequest(BaseModel):
    session_id: str
    box: str
    content: str
    content_type: str = "text"
    timestamp_iso: str = ""
    elapsed_sec: float = 0
    linked_entry_id: int | None = None
    metadata: dict | None = None


class CoLearnerRequest(BaseModel):
    session_id: str
    transcript_chunk: str
    slide_context: str = ""
    elapsed_sec: float = 0


# ═══════════════════════════════════════════
# ENDPOINTS
# ═══════════════════════════════════════════

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "engine": "groq",
        "model": GROQ_MODEL,
        "fallback": "deepgram/nova-2",
        "colearner_model": COLEARNER_MODEL,
        "db_size_mb": round(DB_PATH.stat().st_size / 1024**2, 2) if DB_PATH.exists() else 0,
    }


# ── STT Transcription (Box 1) ──

@app.post("/transcribe")
async def transcribe(audio: UploadFile = File(...), chunk_index: int = Form(0),
                     session_id: str = Form(""), elapsed_sec: float = Form(0)):
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

        now_iso = datetime.now(timezone.utc).isoformat()

        # Save to Box 1 if we have a session
        entry_id = None
        if session_id and text.strip():
            entry_id = add_entry(
                session_id=session_id,
                box="transcript",
                content=text.strip(),
                content_type="text",
                timestamp_iso=now_iso,
                elapsed_sec=elapsed_sec,
                metadata={"chunk_index": chunk_index, "engine": _last_engine_used},
            )

        return {
            "text": text,
            "chunk_index": chunk_index,
            "engine": _last_engine_used,
            "timestamp_iso": now_iso,
            "entry_id": entry_id,
            "elapsed_sec": elapsed_sec,
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
            http2=True, timeout=httpx.Timeout(30.0),
            limits=httpx.Limits(max_keepalive_connections=5, max_connections=20),
        )
    with open(wav_path, "rb") as f:
        resp = await _http_client.post(
            GROQ_TRANSCRIBE_URL,
            headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
            files={"file": (os.path.basename(wav_path), f, "audio/wav")},
            data={"model": GROQ_MODEL, "response_format": "json"},
        )
    if resp.status_code != 200:
        logger.error(f"Groq API error {resp.status_code}: {resp.text[:300]}")
        return ""
    return resp.json().get("text", "").strip()


async def _deepgram_transcribe(wav_path: str) -> str:
    global _http_client
    if _http_client is None:
        _http_client = httpx.AsyncClient(
            http2=True, timeout=httpx.Timeout(30.0),
            limits=httpx.Limits(max_keepalive_connections=5, max_connections=20),
        )
    with open(wav_path, "rb") as f:
        resp = await _http_client.post(
            DEEPGRAM_TRANSCRIBE_URL,
            headers={"Authorization": f"Token {DEEPGRAM_API_KEY}", "Content-Type": "audio/wav"},
            content=f.read(),
        )
    if resp.status_code != 200:
        logger.error(f"Deepgram API error {resp.status_code}: {resp.text[:300]}")
        return ""
    try:
        return resp.json()["results"]["channels"][0]["alternatives"][0]["transcript"].strip()
    except (KeyError, IndexError):
        return ""


async def _transcribe_with_fallback(wav_path: str) -> str:
    global _last_engine_used
    try:
        text = await _groq_transcribe(wav_path)
        if text:
            _last_engine_used = "groq"
            return text
    except Exception as e:
        logger.warning(f"Groq failed, trying Deepgram: {e}")
    try:
        text = await _deepgram_transcribe(wav_path)
        if text:
            _last_engine_used = "deepgram"
            return text
    except Exception as e:
        logger.error(f"Deepgram also failed: {e}")
    _last_engine_used = "none"
    return ""


# ── Save Transcript ──

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
    return {"saved": str(filepath), "text_file": str(txt_path)}


# ── Slide Analysis → routes to Box 1 or Box 2 ──

def _cleanup_old_slides(slide_dir: Path, keep: int = 100):
    """Keep only the most recent N slide files."""
    try:
        files = sorted(slide_dir.glob("*.jpg"), key=lambda p: p.stat().st_mtime, reverse=True)
        for old in files[keep:]:
            old.unlink(missing_ok=True)
    except Exception:
        pass


SLIDE_ANALYSIS_PROMPT = """Analyze this lecture slide. Return a JSON object with exactly these keys:
- "has_useful_info": true if the slide contains educational content (text, diagrams, charts, formulas, tables), false if it's blank, blurry, or just a title slide with no substance
- "text": all visible text extracted from the slide
- "questions": array of questions found on the slide (empty array if none)
- "answers": array of concise answers to each question (empty array if none)
- "summary": one-sentence summary of the slide content
- "explanation": a 2-3 sentence explanation of the key concept on this slide (leave empty if has_useful_info is false)

Keep answers under 2 sentences each. Use the same language as the slide."""


@app.post("/analyze-slide")
async def analyze_slide(
    image: UploadFile = File(...),
    session_id: str = Form(""),
    elapsed_sec: float = Form(0),
):
    global _http_client
    if _http_client is None:
        _http_client = httpx.AsyncClient(
            http2=True, timeout=httpx.Timeout(30.0),
            limits=httpx.Limits(max_keepalive_connections=5, max_connections=20),
        )

    image_bytes = await image.read()
    mime_type = image.content_type or "image/jpeg"
    img_b64 = base64.b64encode(image_bytes).decode()
    now_iso = datetime.now(timezone.utc).isoformat()

    # Save image for retrieval
    img_hash = hashlib.md5(image_bytes).hexdigest()[:12]
    img_dir = TRANSCRIPTS_DIR / "slides"
    img_dir.mkdir(exist_ok=True)
    img_path = img_dir / f"{img_hash}_{datetime.now().strftime('%H%M%S')}.jpg"
    with open(img_path, "wb") as f:
        f.write(image_bytes)
    # Cleanup: keep only last 100 slides
    _cleanup_old_slides(img_dir, keep=100)

    payload = {
        "contents": [{
            "parts": [
                {"text": SLIDE_ANALYSIS_PROMPT},
                {"inline_data": {"mime_type": mime_type, "data": img_b64}},
            ]
        }]
    }

    try:
        resp = await _http_client.post(
            f"{GEMINI_URL}?key={GEMINI_API_KEY}",
            json=payload,
            timeout=15.0,
        )
        if resp.status_code != 200:
            logger.error(f"Gemini API error {resp.status_code}: {resp.text[:300]}")
            return await _handle_slide_fallback(session_id, now_iso, elapsed_sec, img_path, img_hash)

        result = resp.json()
        raw_text = result["candidates"][0]["content"]["parts"][0]["text"]
        clean = raw_text.strip()
        if clean.startswith("```"):
            clean = clean.split("\n", 1)[1].rsplit("\n```", 1)[0] if "```" in clean else clean

        try:
            parsed = json.loads(clean)
        except json.JSONDecodeError:
            return await _handle_slide_fallback(session_id, now_iso, elapsed_sec, img_path, img_hash)

        has_useful = parsed.get("has_useful_info", bool(parsed.get("text", "").strip()))

        if has_useful and parsed.get("text", "").strip():
            # BOX 2: AI Co-Learner insight
            content_json = {
                "text": parsed.get("text", ""),
                "questions": parsed.get("questions", []),
                "answers": parsed.get("answers", []),
                "summary": parsed.get("summary", ""),
                "explanation": parsed.get("explanation", ""),
            }
            entry_id = None
            if session_id:
                entry_id = add_entry(
                    session_id=session_id,
                    box="colearner",
                    content=json.dumps(content_json),
                    content_type="slide_analysis",
                    timestamp_iso=now_iso,
                    elapsed_sec=elapsed_sec,
                    metadata={"image_hash": img_hash, "slide_text": parsed.get("text", "")[:500]},
                )
            return {
                "box": "colearner",
                "analysis": content_json,
                "entry_id": entry_id,
                "timestamp_iso": now_iso,
                "image_url": f"/slides/{img_path.name}",
            }
        else:
            # BOX 1: No useful info — return raw image entry
            return await _handle_slide_fallback(session_id, now_iso, elapsed_sec, img_path, img_hash)

    except Exception as e:
        logger.error(f"Slide analysis failed: {e}")
        return await _handle_slide_fallback(session_id, now_iso, elapsed_sec, img_path, img_hash)


async def _handle_slide_fallback(session_id: str, now_iso: str, elapsed_sec: float,
                                 img_path: Path, img_hash: str) -> dict:
    """Slide couldn't be analyzed → goes to Box 1 as raw image."""
    entry_id = None
    if session_id:
        entry_id = add_entry(
            session_id=session_id,
            box="transcript",
            content=f"[Slide captured at {datetime.now().strftime('%H:%M:%S')}]",
            content_type="slide_image",
            timestamp_iso=now_iso,
            elapsed_sec=elapsed_sec,
            metadata={"image_hash": img_hash, "image_path": str(img_path)},
        )
    return {
        "box": "transcript",
        "analysis": {"text": "", "questions": [], "answers": [], "summary": "", "explanation": ""},
        "entry_id": entry_id,
        "timestamp_iso": now_iso,
        "image_url": f"/slides/{img_path.name}",
        "note": "No useful info detected — slide added to transcript panel",
    }


@app.get("/slides/{filename}")
async def get_slide(filename: str):
    path = TRANSCRIPTS_DIR / "slides" / filename
    if not path.exists():
        raise HTTPException(404, "Slide not found")
    return FileResponse(path)


# ── AI Co-Learner Agent (Box 2) ──

COLEARNER_SYSTEM_PROMPT = """You are an AI study partner attending a lecture alongside a student. Your job is to "think out loud" in Box 2 of a learning interface while the raw transcript appears in Box 1.

Given a transcript chunk and optional slide context, respond in ONE of these modes:
1. EXPLAIN — If the chunk introduces a new concept, explain it clearly in 1-3 sentences
2. QUESTION — If something seems unclear or worth exploring deeper, pose a thought-provoking question
3. CONNECT — If you can connect this to something previously discussed or a real-world application, do so
4. CLARIFY — If the transcript is vague, suggest what the lecturer might mean
5. INSIGHT — Add extra context, a mnemonic, or a key takeaway

RULES:
- Be concise: 1-3 sentences max
- Use the same language as the transcript
- Sound like a smart study buddy, not a textbook
- If the transcript is empty or just filler words, respond with an empty string ""
- If there's slide context, prioritize explaining the slide content
- NEVER repeat back the transcript — always add value
- NEVER say "the transcript says..." — just give the insight directly

Format your response as plain text, no JSON, no markdown."""


@app.post("/co-learner")
async def co_learner(req: CoLearnerRequest):
    """AI Co-Learner thinks out loud about transcript and slide context."""
    if not COLEARNER_API_KEY:
        return {"insight": "", "error": "Co-learner API not configured"}

    # Server-side cooldown
    global _last_colearner_time
    import time as _time
    now = _time.time()
    if now - _last_colearner_time < COLEARNER_COOLDOWN_SEC:
        return {"insight": "", "reason": "cooldown"}
    _last_colearner_time = now

    transcript = req.transcript_chunk.strip()
    slide = req.slide_context.strip()

    if not transcript and not slide:
        return {"insight": "", "reason": "no input"}

    user_message = ""
    if slide:
        user_message = f"SLIDE CONTENT: {slide}\n\n"
    if transcript:
        user_message += f"TRANSCRIPT: {transcript}"
    if not user_message:
        return {"insight": "", "reason": "empty after processing"}

    global _http_client
    if _http_client is None:
        _http_client = httpx.AsyncClient(
            http2=True, timeout=httpx.Timeout(30.0),
            limits=httpx.Limits(max_keepalive_connections=5, max_connections=20),
        )

    now_iso = datetime.now(timezone.utc).isoformat()

    try:
        resp = await _http_client.post(
            COLEARNER_URL,
            headers={
                "Authorization": f"Bearer {COLEARNER_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": COLEARNER_MODEL,
                "messages": [
                    {"role": "system", "content": COLEARNER_SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
                "max_tokens": 200,
                "temperature": 0.7,
            },
            timeout=COLEARNER_TIMEOUT,
        )

        if resp.status_code != 200:
            logger.error(f"Co-learner API error {resp.status_code}: {resp.text[:300]}")
            return {"insight": "", "error": f"API error {resp.status_code}"}

        data = resp.json()
        insight = data["choices"][0]["message"]["content"].strip()

        # Remove quotes, "the transcript says...", etc.
        if insight.startswith('"') and insight.endswith('"'):
            insight = insight[1:-1]
        if insight.lower().startswith("the transcript"):
            insight = insight.split(". ", 1)[-1] if ". " in insight else insight

        # Save to Box 2
        entry_id = None
        if req.session_id and insight:
            entry_id = add_entry(
                session_id=req.session_id,
                box="colearner",
                content=insight,
                content_type="insight",
                timestamp_iso=now_iso,
                elapsed_sec=req.elapsed_sec,
                metadata={"transcript_chunk": transcript[:200], "slide_context": slide[:200]},
            )

        return {
            "insight": insight,
            "entry_id": entry_id,
            "timestamp_iso": now_iso,
            "model": COLEARNER_MODEL,
        }

    except asyncio.TimeoutError:
        return {"insight": "", "error": "timeout"}
    except Exception as e:
        logger.error(f"Co-learner error: {e}")
        return {"insight": "", "error": str(e)[:200]}


# ── Entry Management ──

@app.post("/entries")
async def save_entry(req: EntryRequest):
    """Save a manual entry to a box."""
    sid = create_session(req.session_id)
    entry_id = add_entry(
        session_id=sid,
        box=req.box,
        content=req.content,
        content_type=req.content_type,
        timestamp_iso=req.timestamp_iso,
        elapsed_sec=req.elapsed_sec,
        linked_entry_id=req.linked_entry_id,
        metadata=req.metadata,
    )
    return {"entry_id": entry_id, "session_id": sid}


@app.get("/entries/{session_id}")
async def load_entries(session_id: str, box: str | None = None):
    """Load all entries for a session, optionally filtered by box."""
    entries = get_entries(session_id, box)
    return {"session_id": session_id, "count": len(entries), "entries": entries}


# ── Session Management ──

@app.get("/sessions")
async def list_sessions():
    sessions = db_fetch("SELECT * FROM sessions ORDER BY updated_at DESC LIMIT 50")
    for s in sessions:
        s["entry_count"] = db_fetch(
            "SELECT COUNT(*) as cnt FROM entries WHERE session_id = ?", (s["id"],)
        )[0]["cnt"]
    return {"sessions": sessions}


@app.post("/sessions")
async def new_session(title: str = Form(""), tags: str = Form("")):
    sid = create_session()
    if title:
        db_execute("UPDATE sessions SET title = ?, tags = ? WHERE id = ?", (title, tags, sid))
    return {"session_id": sid, "title": title}


@app.get("/sessions/{session_id}")
async def get_session_detail(session_id: str):
    session = db_fetch("SELECT * FROM sessions WHERE id = ?", (session_id,))
    if not session:
        raise HTTPException(404, "Session not found")
    entries = get_entries(session_id)
    return {"session": session[0], "entry_count": len(entries), "entries": entries}


# ── Legacy endpoints ──

@app.get("/sessions_legacy")
async def list_sessions_legacy():
    sessions = []
    for f in sorted(TRANSCRIPTS_DIR.glob("*.json"), reverse=True):
        with open(f) as fp:
            data = json.load(fp)
        sessions.append({
            "filename": f.name,
            "timestamp": data.get("timestamp", ""),
            "duration_seconds": data.get("duration_seconds", 0),
            "word_count": data.get("word_count", 0),
        })
    return {"sessions": sessions}


# ── Serve Frontend ──

_HTML_PATH = Path("/docker/hermes-bot/data/sasin-cfoth-ai/capture/index.html")

@app.get("/")
async def serve_frontend():
    if _HTML_PATH.exists():
        return FileResponse(str(_HTML_PATH))
    return HTMLResponse("<h1>Capture server running. Frontend not deployed yet.</h1>")


# ── Lifecycle ──

@app.on_event("startup")
async def startup():
    global _http_client
    _http_client = httpx.AsyncClient(
        http2=True, timeout=httpx.Timeout(30.0),
        limits=httpx.Limits(max_keepalive_connections=5, max_connections=20),
    )
    # Warm up STT
    import io as _io, wave as _wave
    buf = _io.BytesIO()
    with _wave.open(buf, "w") as wav:
        wav.setnchannels(1); wav.setsampwidth(2); wav.setframerate(16000)
        wav.writeframes(b"\x00\x00" * 1600)
    silent = buf.getvalue()
    try:
        resp = await _http_client.post(
            GROQ_TRANSCRIBE_URL,
            headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
            files={"file": ("s.wav", silent, "audio/wav")},
            data={"model": GROQ_MODEL, "response_format": "json"},
            timeout=15.0,
        )
        print(f"  ✅ Groq warm ({resp.elapsed.total_seconds():.2f}s)")
    except Exception as e:
        print(f"  ⚠️  Groq warm failed: {e}")
    try:
        resp = await _http_client.post(
            DEEPGRAM_TRANSCRIBE_URL,
            headers={"Authorization": f"Token {DEEPGRAM_API_KEY}", "Content-Type": "audio/wav"},
            content=silent, timeout=15.0,
        )
        print(f"  ✅ Deepgram warm ({resp.elapsed.total_seconds():.2f}s)")
    except Exception as e:
        print(f"  ⚠️  Deepgram warm failed: {e}")

    init_db()
    missing = []
    if not GROQ_API_KEY: missing.append("GROQ_API_KEY")
    if not DEEPGRAM_API_KEY: missing.append("DEEPGRAM_API_KEY")
    if not GEMINI_API_KEY: missing.append("GEMINI_API_KEY")
    if missing:
        print(f"  WARNING: Missing API keys: {chr(44).join(missing)} - some features disabled")
    print(f"Capture v4 ready — Groq STT + Gemini Vision + AI Co-Learner ({COLEARNER_MODEL})")
    print(f"DB: {DB_PATH} ({DB_PATH.stat().st_size} bytes)")


@app.on_event("shutdown")
async def shutdown():
    global _http_client
    if _http_client:
        await _http_client.aclose()
        _http_client = None


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8898))
    uvicorn.run(app, host="0.0.0.0", port=port)
