"""
Sasin Lecture Capture — FastAPI Transcription Server
Receives audio chunks, transcribes with faster-whisper, returns text.
Stores transcripts to disk for later retrieval.
"""
import os
import json
import tempfile
import subprocess
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

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
MODEL_SIZE = "small"  # small = ~500MB, balanced speed/accuracy. Options: tiny, base, small, medium

# Lazy load model
_model = None

def get_model():
    global _model
    if _model is None:
        from faster_whisper import WhisperModel
        _model = WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8")
    return _model


class SaveRequest(BaseModel):
    transcript: str
    word_count: int
    duration: int
    chunks: int
    timestamp: str


@app.get("/health")
async def health():
    return {"status": "ok", "model": MODEL_SIZE, "transcripts_count": len(list(TRANSCRIPTS_DIR.glob("*.json")))}


@app.post("/transcribe")
async def transcribe(audio: UploadFile = File(...), chunk_index: int = Form(0)):
    """Receive an audio chunk, transcribe it, return text."""
    if not audio.filename:
        raise HTTPException(400, "No audio file provided")
    
    # Save to temp file
    suffix = Path(audio.filename).suffix or ".webm"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await audio.read()
        tmp.write(content)
        tmp_path = tmp.name
    
    try:
        # Convert to WAV if needed (faster-whisper needs WAV or raw audio)
        wav_path = tmp_path
        if suffix not in (".wav",):
            wav_path = tmp_path + ".wav"
            subprocess.run(
                ["ffmpeg", "-y", "-i", tmp_path, "-ar", "16000", "-ac", "1", "-f", "wav", wav_path],
                capture_output=True, timeout=30
            )
        
        # Transcribe
        model = get_model()
        segments, info = model.transcribe(wav_path, beam_size=2, language="en")
        
        # Collect text
        text_parts = []
        for segment in segments:
            text_parts.append(segment.text.strip())
        
        text = " ".join(text_parts)
        
        return {
            "text": text,
            "chunk_index": chunk_index,
            "language": info.language,
            "language_probability": info.language_probability
        }
    
    except Exception as e:
        # Fallback: return empty rather than fail
        return {
            "text": "",
            "chunk_index": chunk_index,
            "error": str(e)
        }
    
    finally:
        # Cleanup
        for p in [tmp_path, wav_path if wav_path != tmp_path else None]:
            if p and os.path.exists(p):
                try:
                    os.unlink(p)
                except:
                    pass


@app.post("/save")
async def save_transcript(req: SaveRequest):
    """Save the full transcript to disk."""
    filename = f"lecture_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    filepath = TRANSCRIPTS_DIR / filename
    
    data = {
        "timestamp": req.timestamp,
        "duration_seconds": req.duration,
        "word_count": req.word_count,
        "chunks": req.chunks,
        "transcript": req.transcript
    }
    
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    # Also save as plain text
    txt_path = filepath.with_suffix(".txt")
    with open(txt_path, "w") as f:
        f.write(req.transcript)
    
    return {"saved": str(filepath), "text_file": str(txt_path)}


@app.get("/sessions")
async def list_sessions():
    """List all saved lecture transcripts."""
    sessions = []
    for f in sorted(TRANSCRIPTS_DIR.glob("*.json"), reverse=True):
        with open(f) as fp:
            data = json.load(fp)
        sessions.append({
            "filename": f.name,
            "timestamp": data.get("timestamp", ""),
            "duration_seconds": data.get("duration_seconds", 0),
            "word_count": data.get("word_count", 0),
            "chunks": data.get("chunks", 0)
        })
    return {"sessions": sessions}


@app.get("/sessions/{filename}")
async def get_session(filename: str):
    """Get a specific transcript."""
    filepath = TRANSCRIPTS_DIR / filename
    if not filepath.exists():
        raise HTTPException(404, "Session not found")
    with open(filepath) as f:
        return json.load(f)


if __name__ == "__main__":
    # Warm up model on startup
    print(f"Loading faster-whisper model '{MODEL_SIZE}'...")
    get_model()
    print("Model loaded. Starting server on port 8898...")
    uvicorn.run(app, host="0.0.0.0", port=8898)