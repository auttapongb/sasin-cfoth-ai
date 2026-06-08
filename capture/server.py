"""
Sasin Lecture Capture — FastAPI Server v5
Two-Box Architecture:
  Box 1: Raw transcript (STT) + unanalyzed slides
  Box 2: AI Co-Learner insights (slide analysis, thinking out loud, answers)

Features:
  - faster-whisper local STT (primary, offline, fast)
  - Groq STT with Deepgram fallback (cloud backup, switchable)
  - Audio enhancement via noisereduce (denoising)
  - Gemini Vision for slide analysis → routed to Box 1 or 2
  - AI Co-Learner agent (DeepSeek) that studies alongside
  - SQLite persistence for all entries
  - Timestamp-based linking between boxes
  - Session management (save/load)

Engine control:
  TRANSCRIBE_ENGINE=faster_whisper (default) — local, offline, fast
  TRANSCRIBE_ENGINE=groq — Groq cloud API (needs internet, higher accuracy)
  TRANSCRIBE_ENGINE=auto — try faster_whisper first, fall back to Groq
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
import shutil
import wave
import edge_tts
import numpy as np
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env")

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse, Response, RedirectResponse
from pydantic import BaseModel, field_validator
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

AUDIO_DIR = Path("/root/lecture_audio")
AUDIO_DIR.mkdir(exist_ok=True)

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
AUDIO_ENHANCE_ENABLED = os.environ.get("AUDIO_ENHANCE", "true").lower() == "true"
TRANSCRIBE_ENGINE = os.environ.get("TRANSCRIBE_ENGINE", "faster_whisper")  # faster_whisper | groq | auto
FASTER_WHISPER_MODEL = os.environ.get("FASTER_WHISPER_MODEL", "tiny.en")  # default: fastest
_current_fw_model = FASTER_WHISPER_MODEL  # runtime-switchable
_last_engine_used = "groq"
_last_colearner_time = 0.0
_webm_headers: dict[str, bytes] = {}  # session_id -> cached WebM header bytes
COLEARNER_COOLDOWN_SEC = 20  # server-side cooldown

# ── Lazy-loaded models ──
_faster_whisper_model = None  # loaded on first use
_noisereduce = None

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("capture")

_http_client: httpx.AsyncClient | None = None
_llm_semaphore = None  # lazily initialized in _call_llm


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

            -- v5: Learning Toolkit tables

            CREATE TABLE IF NOT EXISTS flashcards (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                front TEXT NOT NULL,
                back TEXT NOT NULL,
                source TEXT DEFAULT '',
                fsrs_state TEXT NOT NULL DEFAULT '{"stability":0,"difficulty":0,"elapsed_days":0,"scheduled_days":0,"reps":0,"lapses":0,"state":0,"last_review":null}',
                next_review TEXT NOT NULL DEFAULT (datetime('now')),
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                FOREIGN KEY (session_id) REFERENCES sessions(id)
            );
            CREATE INDEX IF NOT EXISTS idx_flashcards_next ON flashcards(next_review);
            CREATE INDEX IF NOT EXISTS idx_flashcards_session ON flashcards(session_id);

            CREATE TABLE IF NOT EXISTS quizzes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                title TEXT NOT NULL DEFAULT '',
                questions TEXT NOT NULL DEFAULT '[]',
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                FOREIGN KEY (session_id) REFERENCES sessions(id)
            );
            CREATE INDEX IF NOT EXISTS idx_quizzes_session ON quizzes(session_id);

            CREATE TABLE IF NOT EXISTS quiz_attempts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                quiz_id INTEGER NOT NULL,
                answers TEXT NOT NULL DEFAULT '[]',
                score REAL NOT NULL DEFAULT 0,
                total INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                FOREIGN KEY (quiz_id) REFERENCES quizzes(id)
            );

            CREATE TABLE IF NOT EXISTS briefs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                topic TEXT NOT NULL DEFAULT '',
                content TEXT NOT NULL DEFAULT '',
                frameworks TEXT NOT NULL DEFAULT '[]',
                key_concepts TEXT NOT NULL DEFAULT '[]',
                discussion_questions TEXT NOT NULL DEFAULT '[]',
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                FOREIGN KEY (session_id) REFERENCES sessions(id)
            );
            CREATE INDEX IF NOT EXISTS idx_briefs_session ON briefs(session_id);

            CREATE TABLE IF NOT EXISTS case_studies (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                company TEXT NOT NULL DEFAULT '',
                industry TEXT NOT NULL DEFAULT '',
                step INTEGER NOT NULL DEFAULT 1,
                analysis TEXT NOT NULL DEFAULT '',
                frameworks_applied TEXT NOT NULL DEFAULT '[]',
                recommendations TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                updated_at TEXT NOT NULL DEFAULT (datetime('now')),
                FOREIGN KEY (session_id) REFERENCES sessions(id)
            );

            CREATE TABLE IF NOT EXISTS audio_overviews (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                title TEXT NOT NULL DEFAULT '',
                script TEXT NOT NULL DEFAULT '',
                audio_path TEXT NOT NULL DEFAULT '',
                style TEXT NOT NULL DEFAULT 'podcast',
                duration_sec REAL NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                FOREIGN KEY (session_id) REFERENCES sessions(id)
            );

            CREATE TABLE IF NOT EXISTS action_items (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                text TEXT NOT NULL,
                speaker TEXT DEFAULT '',
                assignee TEXT DEFAULT '',
                deadline TEXT DEFAULT '',
                source_entry_id INTEGER,
                status TEXT NOT NULL DEFAULT 'open' CHECK(status IN ('open','done','archived')),
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                FOREIGN KEY (session_id) REFERENCES sessions(id),
                FOREIGN KEY (source_entry_id) REFERENCES entries(id)
            );
            CREATE INDEX IF NOT EXISTS idx_actions_session ON action_items(session_id);
            CREATE INDEX IF NOT EXISTS idx_actions_status ON action_items(status);

            CREATE TABLE IF NOT EXISTS class_templates (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                structure TEXT NOT NULL DEFAULT '{}',
                is_default INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            );
        """)
        conn.commit()

init_db()


def db_execute(sql: str, params: tuple = ()) -> int:
    """Execute SQL and return lastrowid."""
    conn = _get_db_conn()
    cur = conn.execute(sql, params)
    conn.commit()
    return cur.lastrowid


def db_fetch(sql: str, params: tuple = ()) -> list[dict]:
    """Fetch rows as dicts."""
    conn = _get_db_conn()
    return [dict(row) for row in conn.execute(sql, params).fetchall()]


import threading as _threading
_db_local = _threading.local()

def _get_db_conn():
    """Thread-local SQLite connection with WAL mode."""
    if not hasattr(_db_local, 'conn') or _db_local.conn is None:
        conn = sqlite3.connect(str(DB_PATH))
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.row_factory = sqlite3.Row
        _db_local.conn = conn
    return _db_local.conn


# Seed default class templates
_default_templates = [
    ("Lecture", {"sections": ["objectives", "key_concepts", "frameworks", "examples", "questions", "summary"]}),
    ("Case Study", {"sections": ["company_background", "industry_context", "problem_statement", "framework_application", "alternatives", "recommendation", "implementation_risks"]}),
    ("Guest Speaker", {"sections": ["speaker_bio", "key_insights", "industry_trends", "career_advice", "Q&A_highlights", "action_items"]}),
    ("Group Meeting", {"sections": ["attendees", "agenda", "decisions", "action_items", "next_steps", "blockers"]}),
    ("Exam Review", {"sections": ["topics_covered", "key_formulas", "practice_questions", "common_mistakes", "memory_aids", "study_priorities"]}),
]
for name, structure in _default_templates:
    db_execute(
        "INSERT OR IGNORE INTO class_templates (name, structure, is_default) VALUES (?, ?, 1)",
        (name, json.dumps(structure)),
    )


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


VALID_BOXES = {'transcript', 'colearner', 'slide', 'notes', 'mindmap', 'action_items', 'case_analysis'}

class EntryRequest(BaseModel):
    session_id: str
    box: str
    content: str
    content_type: str = "text"
    timestamp_iso: str = ""
    elapsed_sec: float = 0
    linked_entry_id: int | None = None
    metadata: dict | None = None

    @field_validator('box')
    @classmethod
    def validate_box(cls, v: str) -> str:
        if v not in VALID_BOXES:
            raise ValueError(f"box must be one of {sorted(VALID_BOXES)}, got '{v}'")
        return v


class CoLearnerRequest(BaseModel):
    session_id: str
    transcript_chunk: str
    slide_context: str = ""
    elapsed_sec: float = 0


# ── v5: New Models ──

class BriefGenerateRequest(BaseModel):
    session_id: str = ""
    topic: str = ""
    template_id: int | None = None

class FlashcardGenerateRequest(BaseModel):
    session_id: str
    count: int = 10

class FlashcardReviewRequest(BaseModel):
    rating: int  # 1=Again, 2=Hard, 3=Good, 4=Easy

class QuizGenerateRequest(BaseModel):
    session_id: str
    qtype: str = "mcq"  # mcq, short, mixed
    count: int = 10
    topic: str = ""

class QuizSubmitRequest(BaseModel):
    answers: list[dict]  # [{question_idx: 0, answer: "B"}, ...]

class AudioGenerateRequest(BaseModel):
    session_id: str
    style: str = "podcast"  # podcast, summary, lecture

class CaseAnalyzeRequest(BaseModel):
    session_id: str
    company: str = ""
    industry: str = ""

class CaseStepRequest(BaseModel):
    step_num: int
    question: str = ""

class TemplateCreateRequest(BaseModel):
    name: str
    structure: dict

class ActionItemsExtractRequest(BaseModel):
    session_id: str

class PresentGenerateRequest(BaseModel):
    session_id: str
    style: str = "executive"  # executive, academic, pitch

class ResearchRequest(BaseModel):
    query: str
    session_id: str = ""


# ═══════════════════════════════════════════
# ENDPOINTS
# ═══════════════════════════════════════════

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "engine": TRANSCRIBE_ENGINE,
        "faster_whisper_model": _current_fw_model,
        "groq_model": GROQ_MODEL,
    }


@app.get("/engine/status")
async def engine_status():
    """Return current STT engine config for UI display."""
    return {
        "engine": TRANSCRIBE_ENGINE,
        "last_used": _last_engine_used,
        "audio_enhance": AUDIO_ENHANCE_ENABLED,
        "faster_whisper_model": _current_fw_model,
        "groq_model": GROQ_MODEL,
    }


# ── TTS (Text-to-Speech) ──

class TTSRequest(BaseModel):
    text: str
    voice: str = "en-US-AriaNeural"

TTS_DIR = Path("/root/tts_cache")
TTS_DIR.mkdir(exist_ok=True)

@app.post("/tts")
async def text_to_speech(req: TTSRequest):
    """Convert text to speech using Microsoft Edge TTS (free, no API key)."""
    text_hash = hashlib.md5(req.text.encode()).hexdigest()[:12]
    cache_path = TTS_DIR / f"tts_{text_hash}_{req.voice}.mp3"
    if cache_path.exists():
        return FileResponse(cache_path, media_type="audio/mpeg",
                           headers={"X-TTS-Cached": "true"})
    try:
        communicate = edge_tts.Communicate(req.text, req.voice)
        await communicate.save(str(cache_path))
        return FileResponse(cache_path, media_type="audio/mpeg")
    except Exception as e:
        logger.error(f"TTS error: {e}")
        return JSONResponse({"error": str(e)[:200]}, status_code=500)


@app.post("/engine/set")
async def engine_set(engine: str = Query(...)):
    """Update TRANSCRIBE_ENGINE env var (runtime only, not persisted)."""
    global TRANSCRIBE_ENGINE
    engine = engine.lower()
    if engine not in ("faster_whisper", "groq", "auto"):
        raise HTTPException(400, f"Invalid engine: {engine}. Use faster_whisper, groq, or auto")
    TRANSCRIBE_ENGINE = engine
    return {"engine": TRANSCRIBE_ENGINE, "message": f"Switched to {engine} (runtime only — restart to persist)"}


@app.get("/model/status")
async def model_status():
    """Return current faster-whisper model."""
    return {
        "model": _current_fw_model,
        "default": FASTER_WHISPER_MODEL,
        "available": ["tiny.en", "base.en", "small.en"],
    }


@app.post("/model/set")
async def model_set(model: str = Query(...)):
    """Switch faster-whisper model at runtime (tiny.en ↔ base.en)."""
    global _current_fw_model, _faster_whisper_model
    model = model.lower()
    if model not in ("tiny.en", "base.en", "small.en"):
        raise HTTPException(400, f"Invalid model: {model}. Use tiny.en, base.en, or small.en")
    _current_fw_model = model
    _faster_whisper_model = None  # force reload on next use
    return {"model": _current_fw_model, "message": f"Switched to {model} (reloads on next transcription)"}


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
            # WebM header caching: some browsers send headerless chunks after first
            # Save first chunk's header bytes and prepend to subsequent chunks
            if raw_audio[:4] == b"\x1a\x45\xdf\xa3":  # EBML header magic
                _webm_headers[session_id] = raw_audio[:2000]  # cache header
            elif session_id in _webm_headers and len(raw_audio) > 100:
                # Prepend cached header to headerless chunk
                raw_audio = _webm_headers[session_id] + raw_audio
                with open(tmp_path, "wb") as fh:
                    fh.write(raw_audio)
            wav_path = tmp_path + ".wav"
            loop = asyncio.get_event_loop()
            logger.debug(f"Converting chunk {chunk_index}: {len(raw_audio)} bytes (suffix={suffix})")
            success = await asyncio.wait_for(
                loop.run_in_executor(None, _convert_audio, tmp_path, wav_path),
                timeout=TRANSCRIBE_TIMEOUT,
            )
            if not success:
                return {"text": "", "chunk_index": chunk_index, "error": "audio_conversion_failed", "elapsed_sec": elapsed_sec}

        # ── Audio Enhancement ──
        wav_path = _enhance_audio(wav_path)

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

            # Save original audio (WAV) for this session — save each chunk
            session_audio_dir = AUDIO_DIR / session_id
            session_audio_dir.mkdir(exist_ok=True)
            chunk_audio_path = session_audio_dir / f"chunk_{chunk_index:04d}.wav"
            shutil.copy2(wav_path, chunk_audio_path)

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



# ── Audio Enhancement ──

def _enhance_audio(wav_path: str) -> str:
    """Apply noise reduction to WAV file. Returns path to enhanced file."""
    global _noisereduce
    if not AUDIO_ENHANCE_ENABLED:
        return wav_path

    try:
        if _noisereduce is None:
            import noisereduce as nr
            _noisereduce = nr

        # Read WAV
        with wave.open(wav_path, 'rb') as wf:
            n_channels = wf.getnchannels()
            sample_width = wf.getsampwidth()
            framerate = wf.getframerate()
            n_frames = wf.getnframes()
            audio_data = wf.readframes(n_frames)

        # Convert to numpy
        audio = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
        if n_channels > 1:
            audio = audio.reshape(-1, n_channels).mean(axis=1)  # mono

        # Stationary noise reduction (fast, deterministic)
        enhanced = _noisereduce.reduce_noise(
            y=audio, sr=framerate,
            stationary=True, prop_decrease=0.85,
            n_fft=512, win_length=512, hop_length=128
        )

        # Convert back to int16
        enhanced_int16 = (enhanced * 32768.0).clip(-32768, 32767).astype(np.int16)

        # Write enhanced file
        enhanced_path = wav_path + ".enhanced.wav"
        with wave.open(enhanced_path, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(framerate)
            wf.writeframes(enhanced_int16.tobytes())

        logger.info(f"Audio enhanced: {wav_path} → {enhanced_path}")
        return enhanced_path

    except Exception as e:
        logger.warning(f"Audio enhancement skipped: {e}")
        return wav_path


# ── faster-whisper Transcription ──

def _load_faster_whisper():
    """Lazy-load faster-whisper model. Reloads if model was switched."""
    global _faster_whisper_model, _current_fw_model
    if _faster_whisper_model is None or getattr(_faster_whisper_model, '_model_name', '') != _current_fw_model:
        from faster_whisper import WhisperModel
        # Clear old model
        _faster_whisper_model = None
        _faster_whisper_model = WhisperModel(
            _current_fw_model,
            device="cpu",
            compute_type="int8",
            cpu_threads=4,
            num_workers=2,
        )
        _faster_whisper_model._model_name = _current_fw_model  # track which model is loaded
        logger.info(f"faster-whisper loaded: {_current_fw_model}")
    return _faster_whisper_model


async def _faster_whisper_transcribe(wav_path: str) -> str:
    """Transcribe using local faster-whisper."""
    loop = asyncio.get_event_loop()

    def _run():
        model = _load_faster_whisper()
        segments, _ = model.transcribe(wav_path, beam_size=5, language="en")
        return " ".join(seg.text.strip() for seg in segments)

    try:
        text = await asyncio.wait_for(
            loop.run_in_executor(None, _run),
            timeout=TRANSCRIBE_TIMEOUT,
        )
        return text.strip()
    except asyncio.TimeoutError:
        logger.warning("faster-whisper timed out")
        return ""
    except Exception as e:
        logger.warning(f"faster-whisper failed: {e}")
        return ""


def _convert_audio(src: str, dst: str) -> bool:
    result = subprocess.run(
        ["ffmpeg", "-y", "-i", src, "-ar", "16000", "-ac", "1", "-f", "wav", dst],
        capture_output=True,
        timeout=TRANSCRIBE_TIMEOUT,
    )
    if result.returncode != 0:
        stderr_tail = result.stderr.decode(errors="replace")[-500:] if result.stderr else "(no stderr)"
        logger.warning(f"ffmpeg failed (rc={result.returncode}) on {os.path.basename(src)}: {stderr_tail}")
        return False
    if not os.path.exists(dst) or os.path.getsize(dst) == 0:
        logger.warning(f"ffmpeg produced no output: {dst} missing or empty (src was {os.path.getsize(src)} bytes)")
        return False
    return True


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

    engine = TRANSCRIBE_ENGINE.lower()

    # ── faster_whisper (local, primary) ──
    if engine in ("faster_whisper", "auto"):
        try:
            text = await _faster_whisper_transcribe(wav_path)
            if text:
                _last_engine_used = "faster_whisper"
                return text
        except Exception as e:
            logger.warning(f"faster-whisper failed: {e}")

    # ── Groq (cloud, fallback or primary) ──
    if engine == "auto" or engine == "groq":
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
    # Also save to pipeline watch directory for auto-ingest
    pipeline_dir = Path("/docker/hermes-bot/data/sasin-cfoth-ai/capture/transcripts")
    pipeline_dir.mkdir(parents=True, exist_ok=True)
    pipeline_path = pipeline_dir / txt_path.name
    with open(pipeline_path, "w") as f:
        f.write(req.transcript)
    return {"saved": str(filepath), "text_file": str(txt_path), "pipeline_copy": str(pipeline_path)}


# ── Audio Download ──

@app.get("/audio/download/{session_id}")
async def download_session_audio(session_id: str):
    """Download the full recording for a session (concatenated WAV)."""
    session_dir = AUDIO_DIR / session_id
    if not session_dir.exists():
        raise HTTPException(404, f"No audio found for session {session_id}")

    chunks = sorted(session_dir.glob("chunk_*.wav"))
    if not chunks:
        raise HTTPException(404, f"No audio chunks for session {session_id}")

    # If only one chunk, serve directly
    if len(chunks) == 1:
        return FileResponse(str(chunks[0]), media_type="audio/wav",
                            filename=f"{session_id}.wav")

    # Concatenate WAVs (simple: strip headers, concatenate data, write new header)
    merged_path = session_dir / "merged.wav"
    # Read first chunk to get WAV params
    with open(chunks[0], "rb") as f:
        header = f.read(44)  # Standard WAV header is 44 bytes

    total_data_size = 0
    data_chunks = []
    for cp in chunks:
        with open(cp, "rb") as f:
            f.seek(44)
            data = f.read()
            total_data_size += len(data)
            data_chunks.append(data)

    # Write merged WAV
    with open(merged_path, "wb") as out:
        out.write(header[:4])    # "RIFF"
        out.write((36 + total_data_size).to_bytes(4, "little"))  # ChunkSize
        out.write(header[8:40])  # fmt subchunk
        out.write(total_data_size.to_bytes(4, "little"))  # Subchunk2Size
        out.write(b"".join(data_chunks))

    return FileResponse(str(merged_path), media_type="audio/wav",
                        filename=f"{session_id}.wav")


@app.get("/audio/{session_id}/info")
async def audio_info(session_id: str):
    """List audio chunks available for a session."""
    session_dir = AUDIO_DIR / session_id
    if not session_dir.exists():
        return {"session_id": session_id, "chunks": [], "total_chunks": 0}

    chunks = sorted(session_dir.glob("chunk_*.wav"))
    total_size = sum(c.stat().st_size for c in chunks)
    return {
        "session_id": session_id,
        "chunks": [c.name for c in chunks],
        "total_chunks": len(chunks),
        "total_size_bytes": total_size,
        "total_size_mb": round(total_size / 1024**2, 2),
    }


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
    sessions = db_fetch("""
        SELECT s.*,
               COALESCE((SELECT COUNT(*) FROM entries e WHERE e.session_id = s.id), 0) as entry_count,
               (SELECT e2.content FROM entries e2 WHERE e2.session_id = s.id 
                AND e2.box = 'transcript' AND e2.content_type != 'slide_image' 
                ORDER BY e2.id ASC LIMIT 1) as preview
        FROM sessions s ORDER BY s.updated_at DESC LIMIT 500
    """)
    for s in sessions:
        if s.get("preview"):
            s["preview"] = s["preview"][:120]
    return {"sessions": sessions}


@app.post("/sessions")
async def new_session(title: str = Form(""), tags: str = Form("")):
    sid = create_session()
    if not title:
        # Auto-generate timestamped title for same-day disambiguation
        title = datetime.now().strftime("Lecture — %b %d, %Y %I:%M %p")
    db_execute("UPDATE sessions SET title = ?, tags = ? WHERE id = ?", (title, tags, sid))
    return {"session_id": sid, "title": title}


@app.get("/sessions/{session_id}")
async def get_session_detail(session_id: str):
    session = db_fetch("SELECT * FROM sessions WHERE id = ?", (session_id,))
    if not session:
        raise HTTPException(404, "Session not found")
    entries = get_entries(session_id)
    return {"session": session[0], "entry_count": len(entries), "entries": entries}


@app.post("/sessions/{session_id}/rename")
async def rename_session(session_id: str, title: str = Form("")):
    session = db_fetch("SELECT id FROM sessions WHERE id = ?", (session_id,))
    if not session:
        raise HTTPException(404, "Session not found")
    db_execute("UPDATE sessions SET title = ?, updated_at = ? WHERE id = ?",
               (title, datetime.now(timezone.utc).isoformat(), session_id))
    return {"session_id": session_id, "title": title}

@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    """Cascade delete a session and all associated data."""
    existing = db_fetch("SELECT id FROM sessions WHERE id = ?", (session_id,))
    if not existing:
        raise HTTPException(404, "Session not found")
    tables = ["entries", "flashcards", "briefs", "quizzes", "quiz_attempts", 
              "audio_overviews", "case_studies", "action_items"]
    deleted = {}
    for table in tables:
        try:
            count = db_fetch(f"SELECT COUNT(*) as c FROM {table} WHERE session_id = ?", (session_id,))[0]["c"]
            if count > 0:
                db_execute(f"DELETE FROM {table} WHERE session_id = ?", (session_id,))
                deleted[table] = count
        except Exception:
            pass
    db_execute("DELETE FROM sessions WHERE id = ?", (session_id,))
    return {"session_id": session_id, "deleted": deleted, "total_items": sum(deleted.values())}

@app.delete("/templates/{template_id}")
async def delete_template(template_id: int):
    existing = db_fetch("SELECT id FROM class_templates WHERE id = ? AND is_default = 0", (template_id,))
    if not existing:
        raise HTTPException(404, "Template not found")
    db_execute("DELETE FROM class_templates WHERE id = ? AND is_default = 0", (template_id,))
    return {"deleted": template_id}


@app.get("/sessions/{session_id}/download")
async def download_session(session_id: str):
    """Download full session as plain text — Box 1 transcript + Box 2 insights."""
    session = db_fetch("SELECT * FROM sessions WHERE id = ?", (session_id,))
    if not session:
        raise HTTPException(404, "Session not found")
    entries = get_entries(session_id)
    
    lines = []
    lines.append(f"Session: {session[0]['title'] or session_id}")
    lines.append(f"Date: {session[0]['created_at']}")
    lines.append(f"Entries: {len(entries)}")
    lines.append("=" * 50)
    lines.append("")
    
    for e in entries:
        ts = e.get('timestamp_iso', '')[:19] if e.get('timestamp_iso') else ''
        box = e.get('box', '')
        ctype = e.get('content_type', '')
        content = e.get('content', '')
        
        if box == 'transcript':
            if ctype == 'slide_image':
                lines.append(f"[{ts}] 📸 Slide captured")
            else:
                lines.append(f"[{ts}] 🎙️ {content}")
        elif box == 'colearner':
            if ctype == 'slide_analysis':
                try:
                    import json as _json
                    parsed = _json.loads(content)
                    lines.append(f"[{ts}] 🔍 Slide Analysis:")
                    if parsed.get('summary'):
                        lines.append(f"    Summary: {parsed['summary']}")
                    if parsed.get('explanation'):
                        lines.append(f"    {parsed['explanation']}")
                    if parsed.get('text'):
                        lines.append(f"    Text: {parsed['text'][:200]}")
                except Exception:
                    lines.append(f"[{ts}] 🔍 Slide Analysis: {content[:200]}")
            else:
                lines.append(f"[{ts}] 💡 {content}")
        lines.append("")
    
    text = "\n".join(lines)
    
    # Use a safe filename
    safe_title = "".join(c for c in (session[0]['title'] or 'session') if c.isalnum() or c in ' -_')
    filename = f"{safe_title or 'session'}.txt"
    
    from fastapi.responses import Response
    return Response(
        content=text,
        media_type="text/plain",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'}
    )


# ── AI Chat (session-aware + KB) ──

CHAT_SYSTEM_PROMPT = """You are an AI Learning Assistant for a student in the Sasin School of Management EMBA program (Class of 2026).

YOUR ROLE:
- Help this student learn, understand frameworks, prepare for cases, and connect concepts
- You have access to their personal Knowledge Base of uploaded lecture PDFs, slides, transcripts
- Prioritize information from THEIR materials over general knowledge
- When the answer is in their KB, cite specific files/concepts from it
- When it's not in their materials, say so clearly and offer to help if they upload relevant content

TONE:
- Graduate-level academic, but approachable
- Concise and structured (use bullet points for frameworks/lists)
- Challenge the student with follow-up questions when appropriate
- Match the student's language (English or Thai)

CAPABILITIES:
- You have access to the LIVE lecture transcript if provided — reference specific quotes
- Explain business frameworks (Porter, SWOT, PESTLE, BCG, Blue Ocean, etc.)
- Connect lecture concepts across sessions
- Help prepare for case discussions and exams
- Summarize uploaded PDFs and slides
- Generate practice questions from study materials

FORMAT:
- Keep first response under 4 sentences unless detail is requested
- Use markdown-style formatting (bold for key terms, bullet points for lists)
- Always ground answers in the student's provided materials when possible"""

class ChatRequest(BaseModel):
    session_id: str = ""
    question: str
    transcript: str = ""

@app.post("/chat")
async def chat(req: ChatRequest):
    """Answer questions using current session context + KB."""
    if not req.question.strip():
        return {"answer": "Please ask a question about your lectures, frameworks, or study materials.", "sources": []}
    
    # Gather session context
    context_parts = []
    if req.session_id:
        entries = get_entries(req.session_id)
        transcript_entries = [e for e in entries if e["box"] == "transcript" and e.get("content_type") != "slide_image"]
        insight_entries = [e for e in entries if e["box"] == "colearner"]
        
        if req.transcript:
            # Live transcript from DOM (during recording)
            context_parts.append(f"LIVE TRANSCRIPT (currently recording):\n{req.transcript[-3000:]}")
        elif transcript_entries:
            # Saved transcript from DB
            transcript_text = " ".join(e["content"] for e in transcript_entries[-10:])
            context_parts.append(f"RECENT TRANSCRIPT:\n{transcript_text[-2000:]}")
        
        if insight_entries:
            insight_text = " ".join(e["content"] for e in insight_entries[-5:])
            context_parts.append(f"AI INSIGHTS:\n{insight_text[-1000:]}")
    
    # Fetch real KB file list + try to find relevant file content
    kb_context = ""
    try:
        global _http_client
        if _http_client is None:
            _http_client = httpx.AsyncClient(timeout=httpx.Timeout(15.0))
        
        # Get file list
        kb_resp = await _http_client.get("http://localhost:8001/api/v1/knowledge/emba-2026/files", timeout=5.0)
        if kb_resp.status_code == 200:
            data = kb_resp.json()
            files = data.get('files', data) if isinstance(data, dict) else data
            if isinstance(files, list) and len(files) > 0:
                # Build file list
                file_names = []
                for f in files[:30]:
                    name = f["name"] if isinstance(f, dict) else str(f)
                    size = f.get("size", 0) if isinstance(f, dict) else 0
                    file_names.append(f"{name} ({size//1024}KB)" if size > 1024 else name)
                
                # Try to find relevant files by keyword match on question
                question_lower = req.question.lower()
                relevant_files = []
                for f in files[:30]:
                    name = f["name"] if isinstance(f, dict) else str(f)
                    name_lower = name.lower()
                    # Simple keyword match
                    keywords = question_lower.split()
                    matches = sum(1 for kw in keywords if len(kw) > 2 and kw in name_lower)
                    if matches >= 2 or any(kw in name_lower for kw in keywords if len(kw) > 4):
                        relevant_files.append(name)
                
                # Fetch content of top 3 relevant files
                kb_content = ""
                for rf_name in relevant_files[:3]:
                    try:
                        content_resp = await _http_client.get(
                            f"http://localhost:8001/api/v1/knowledge/emba-2026/files/{rf_name}",
                            timeout=3.0
                        )
                        if content_resp.status_code == 200:
                            content = content_resp.text[:1500]  # limit per file
                            kb_content += f"\n--- {rf_name} ---\n{content}\n"
                    except Exception:
                        pass
                
                kb_context = f"KNOWLEDGE BASE ({len(files)} docs available):\n"
                kb_context += "Files: " + ", ".join(file_names[:15])
                if len(file_names) > 15:
                    kb_context += f" ... and {len(file_names)-15} more"
                if kb_content:
                    kb_context += f"\n\nRELEVANT CONTENT FROM YOUR MATERIALS:\n{kb_content[:3000]}"
    except Exception as e:
        logger.warning(f"KB fetch failed: {e}")
    
    if not kb_context:
        kb_context = "No files in knowledge base yet. Upload lecture PDFs or slides via 📄 button or 🧠 2nd Brain."
    context_parts.append(kb_context)
    
    context = "\n\n---\n\n".join(context_parts) if context_parts else "No session context available yet. Start recording or upload materials."
    
    # Call DeepSeek
    if not COLEARNER_API_KEY:
        return {"answer": "Chat API not configured. Set DEEPSEEK_API_KEY.", "error": "no_key"}
    
    try:
        resp = await _http_client.post(
            COLEARNER_URL,
            headers={"Authorization": f"Bearer {COLEARNER_API_KEY}", "Content-Type": "application/json"},
            json={
                "model": COLEARNER_MODEL,
                "messages": [
                    {"role": "system", "content": CHAT_SYSTEM_PROMPT},
                    {"role": "user", "content": f"CONTEXT:\n{context}\n\nQUESTION: {req.question}"}
                ],
                "max_tokens": 400,
                "temperature": 0.5,
            },
            timeout=20.0,
        )
        if resp.status_code != 200:
            return {"answer": f"AI error: {resp.status_code}", "error": str(resp.status_code)}
        
        data = resp.json()
        answer = data["choices"][0]["message"]["content"].strip()
        return {"answer": answer, "session_id": req.session_id, "model": COLEARNER_MODEL}
    
    except Exception as e:
        return {"answer": f"Error: {str(e)[:200]}", "error": str(e)[:200]}


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


# ═══════════════════════════════════════════════════════
# v5: Learning Toolkit — Shared Helpers
# ═══════════════════════════════════════════════════════

async def _call_llm(system_prompt: str, user_message: str, max_tokens: int = 500, temperature: float = 0.5, json_mode: bool = False) -> str:
    """Shared LLM caller using DeepSeek — concurrency-limited."""
    global _http_client, _llm_semaphore
    if _http_client is None:
        _http_client = httpx.AsyncClient(timeout=httpx.Timeout(30.0))
    if _llm_semaphore is None:
        _llm_semaphore = asyncio.Semaphore(2)
    async with _llm_semaphore:
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_message}]
        body = {"model": COLEARNER_MODEL, "messages": messages, "max_tokens": max_tokens, "temperature": temperature}
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        resp = await _http_client.post(COLEARNER_URL, headers={"Authorization": f"Bearer {COLEARNER_API_KEY}", "Content-Type": "application/json"}, json=body, timeout=30.0)
        if resp.status_code != 200:
            raise HTTPException(502, f"LLM error: {resp.status_code}")
        return resp.json()["choices"][0]["message"]["content"].strip()


async def _call_llm_stream(system_prompt: str, user_message: str, max_tokens: int = 500, temperature: float = 0.5):
    """Streaming LLM caller — yields SSE chunks as tokens arrive.
    Usage: StreamingResponse(_call_llm_stream(...), media_type="text/event-stream")
    """
    global _http_client, _llm_semaphore
    if _http_client is None:
        _http_client = httpx.AsyncClient(timeout=httpx.Timeout(30.0))
    if _llm_semaphore is None:
        _llm_semaphore = asyncio.Semaphore(2)
    async with _llm_semaphore:
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_message}]
        body = {"model": COLEARNER_MODEL, "messages": messages, "max_tokens": max_tokens,
                "temperature": temperature, "stream": True}
        accumulated = []
        async with _http_client.stream("POST", COLEARNER_URL,
                                        headers={"Authorization": f"Bearer {COLEARNER_API_KEY}",
                                                 "Content-Type": "application/json"},
                                        json=body, timeout=60.0) as resp:
            if resp.status_code != 200:
                yield f"data: {json.dumps({'error': f'LLM error: {resp.status_code}'})}\n\n"
                return
            async for line in resp.aiter_lines():
                if line.startswith("data: "):
                    data_str = line[6:]
                    if data_str == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data_str)
                        delta = chunk.get("choices", [{}])[0].get("delta", {})
                        token = delta.get("content", "")
                        if token:
                            accumulated.append(token)
                            yield f"data: {json.dumps({'token': token})}\n\n"
                    except json.JSONDecodeError:
                        continue
        yield f"data: {json.dumps({'done': True, 'full_text': ''.join(accumulated)})}\n\n"


def _build_session_text(session_id: str, max_chars: int = 6000) -> str:
    """Build combined transcript + insights text from a session."""
    entries = get_entries(session_id)
    parts = []
    for e in entries:
        if e["box"] == "transcript" and e.get("content_type") != "slide_image":
            parts.append(e["content"])
        elif e["box"] == "colearner":
            c = e["content"]
            try:
                parsed = json.loads(c)
                if parsed.get("summary"):
                    parts.append(parsed["summary"])
                if parsed.get("explanation"):
                    parts.append(parsed["explanation"])
            except Exception:
                parts.append(c[:300])
    text = " ".join(parts)
    return text[-max_chars:] if len(text) > max_chars else text


# ═══════════════════════════════════════════════════════
# v5: 1. Executive Briefs
# ═══════════════════════════════════════════════════════

BRIEF_SYSTEM = """You generate pre-class executive briefs for an EMBA student at Sasin School of Management.
Given lecture transcript content or a topic, produce a 1-page brief with these sections using markdown:

## Executive Summary
2-3 sentences capturing the essence.

## Key Frameworks
Detected business frameworks (Porter, SWOT, PESTLE, BCG, Blue Ocean, etc.) with 1-line explanation of how each applies.

## Must-Know Concepts
3-5 bullet points of the most important concepts.

## Discussion Questions
3 thought-provoking questions to prepare for class discussion.

## Real-World Connection
1-2 sentence connection to current business events or cases.

Be concise. Total output under 400 words. Use the same language as the source material."""


@app.post("/briefs/generate")
async def generate_brief(req: BriefGenerateRequest, stream: bool = Query(False)):
    text = ""
    if req.session_id:
        text = _build_session_text(req.session_id)
    if not text.strip() and not req.topic:
        raise HTTPException(400, "No session content or topic provided")
    if not COLEARNER_API_KEY:
        raise HTTPException(503, "LLM API not configured")
    user_msg = f"TOPIC: {req.topic}\n\nSOURCE MATERIAL:\n{text}" if req.topic else f"SOURCE MATERIAL:\n{text}"
    if stream:
        return StreamingResponse(
            _call_llm_stream(BRIEF_SYSTEM, user_msg, max_tokens=800, temperature=0.4),
            media_type="text/event-stream",
            headers={"X-Stream-Format": "sse", "X-Endpoint": "briefs"}
        )
    try:
        content = await _call_llm(BRIEF_SYSTEM, user_msg, max_tokens=800, temperature=0.4)
    except Exception as e:
        raise HTTPException(502, f"Brief generation failed: {e}")
    # Extract frameworks using fixed detector
    frameworks = _detect_frameworks(content)
    # Extract discussion questions
    dq = []
    for line in content.split("\n"):
        stripped = line.strip()
        if stripped.startswith(("1.", "2.", "3.", "- ", "* ")) and "?" in stripped:
            dq.append(stripped.lstrip("1234567890.- *"))
    brief_id = db_execute(
        "INSERT INTO briefs (session_id, topic, content, frameworks, discussion_questions) VALUES (?, ?, ?, ?, ?)",
        (req.session_id or "", req.topic, content, json.dumps(frameworks), json.dumps(dq)),
    )
    return {"id": brief_id, "topic": req.topic, "content": content, "frameworks": frameworks, "discussion_questions": dq}


@app.get("/briefs")
async def list_briefs():
    briefs = db_fetch("SELECT id, session_id, topic, frameworks, created_at FROM briefs ORDER BY created_at DESC LIMIT 50")
    for b in briefs:
        b["frameworks"] = json.loads(b.get("frameworks", "[]")) if isinstance(b.get("frameworks"), str) else b.get("frameworks", [])
    return {"briefs": briefs}


@app.get("/briefs/{brief_id}")
async def get_brief(brief_id: int):
    brief = db_fetch("SELECT * FROM briefs WHERE id = ?", (brief_id,))
    if not brief:
        raise HTTPException(404, "Brief not found")
    b = brief[0]
    for field in ["frameworks", "key_concepts", "discussion_questions"]:
        b[field] = json.loads(b.get(field, "[]")) if isinstance(b.get(field), str) else b.get(field, [])
    return b


@app.get("/briefs/{brief_id}/download")
async def download_brief(brief_id: int):
    brief = db_fetch("SELECT * FROM briefs WHERE id = ?", (brief_id,))
    if not brief:
        raise HTTPException(404, "Brief not found")
    b = brief[0]
    from fastapi.responses import Response
    return Response(content=b["content"], media_type="text/markdown",
                    headers={"Content-Disposition": f'attachment; filename="brief_{brief_id}.md"'})

@app.delete("/briefs/{brief_id}")
async def delete_brief(brief_id: int):
    existing = db_fetch("SELECT id FROM briefs WHERE id = ?", (brief_id,))
    if not existing:
        raise HTTPException(404, "Brief not found")
    db_execute("DELETE FROM briefs WHERE id = ?", (brief_id,))
    return {"deleted": brief_id}


# ═══════════════════════════════════════════════════════
# v5: 2. Spaced Repetition Flashcards (FSRS)
# ═══════════════════════════════════════════════════════

FLASHCARD_SYSTEM = """You generate study flashcards from lecture content for an EMBA student.
Given the transcript text, extract 10 key concept pairs as flashcards.
Return a JSON object: {"cards": [{"front": "concept or question", "back": "explanation or answer", "source": "brief quote from transcript"}]}
Make fronts concise questions or concept names. Make backs clear 1-2 sentence explanations.
Focus on frameworks, formulas, definitions, and key insights."""


@app.post("/flashcards/generate")
async def generate_flashcards(req: FlashcardGenerateRequest):
    text = _build_session_text(req.session_id)
    if not text.strip():
        raise HTTPException(400, "No session content")
    if not COLEARNER_API_KEY:
        raise HTTPException(503, "LLM API not configured")
    try:
        raw = await _call_llm(FLASHCARD_SYSTEM, f"Generate {req.count} flashcards from:\n\n{text}", max_tokens=1500, temperature=0.5, json_mode=True)
        data = json.loads(raw)
        cards = data.get("cards", [])
    except Exception as e:
        raise HTTPException(502, f"Flashcard generation failed: {e}")
    created = []
    for c in cards[:req.count]:
        fid = db_execute(
            "INSERT INTO flashcards (session_id, front, back, source) VALUES (?, ?, ?, ?)",
            (req.session_id, c.get("front", ""), c.get("back", ""), c.get("source", "")),
        )
        created.append({"id": fid, "front": c.get("front", ""), "back": c.get("back", ""), "source": c.get("source", "")})
    return {"cards": created, "total": len(created)}


@app.get("/flashcards/due")
async def get_due_flashcards():
    cards = db_fetch(
        "SELECT id, front, back, source, fsrs_state, next_review FROM flashcards WHERE next_review <= datetime('now') ORDER BY next_review ASC LIMIT 50"
    )
    for c in cards:
        c["fsrs_state"] = json.loads(c["fsrs_state"]) if isinstance(c.get("fsrs_state"), str) else c.get("fsrs_state", {})
    return {"cards": cards, "total_due": len(cards)}


@app.post("/flashcards/{card_id}/review")
async def review_flashcard(card_id: int, req: FlashcardReviewRequest):
    card = db_fetch("SELECT * FROM flashcards WHERE id = ?", (card_id,))
    if not card:
        raise HTTPException(404, "Card not found")
    c = card[0]
    state = json.loads(c["fsrs_state"]) if isinstance(c.get("fsrs_state"), str) else c.get("fsrs_state", {})
    rating = req.rating  # 1=Again, 2=Hard, 3=Good, 4=Easy
    # Simple FSRS-inspired scheduling
    if rating <= 2:
        state["lapses"] = state.get("lapses", 0) + 1
        state["reps"] = 0
        interval = 1  # review again tomorrow
    else:
        state["reps"] = state.get("reps", 0) + 1
        if state["reps"] == 1:
            interval = 1
        elif state["reps"] == 2:
            interval = 3
        else:
            interval = min(int(state.get("stability", 1) * 1.5 * (1 + 0.2 * (rating - 3))), 90)
    state["stability"] = max(1, interval)
    state["difficulty"] = max(1, min(10, state.get("difficulty", 5) + (3 - rating)))
    state["last_review"] = datetime.now(timezone.utc).isoformat()
    state["scheduled_days"] = interval
    next_review = (datetime.now(timezone.utc) + __import__("datetime").timedelta(days=interval)).isoformat()
    db_execute(
        "UPDATE flashcards SET fsrs_state = ?, next_review = ? WHERE id = ?",
        (json.dumps(state), next_review, card_id),
    )
    return {"id": card_id, "next_review": next_review, "state": state, "interval_days": interval}


@app.get("/flashcards/stats")
async def flashcard_stats():
    total = db_fetch("SELECT COUNT(*) as c FROM flashcards")[0]["c"]
    due = db_fetch("SELECT COUNT(*) as c FROM flashcards WHERE next_review <= datetime('now')")[0]["c"]
    mastered = db_fetch("SELECT COUNT(*) as c FROM flashcards WHERE json_extract(fsrs_state, '$.reps') >= 3")[0]["c"]
    return {"total": total, "due": due, "mastered": mastered}


@app.delete("/flashcards/{card_id}")
async def delete_flashcard(card_id: int):
    existing = db_fetch("SELECT id FROM flashcards WHERE id = ?", (card_id,))
    if not existing:
        raise HTTPException(404, "Flashcard not found")
    db_execute("DELETE FROM flashcards WHERE id = ?", (card_id,))
    return {"deleted": card_id}


# ═══════════════════════════════════════════════════════
# v5: 3. AI Quiz Generator
# ═══════════════════════════════════════════════════════

QUIZ_SYSTEM = """You generate quiz questions from lecture content for an EMBA student.
Return a JSON object: {"title": "Quiz Title", "questions": [{"q": "question text", "choices": ["A) ...", "B) ...", "C) ...", "D) ..."], "answer": "B", "explanation": "why this is correct"}]}
Make questions test understanding, not just recall. Include scenario-based questions.
For mcq type: include 4 choices, indicate correct answer letter. For short type: just q + answer + explanation."""


@app.post("/quizzes/generate")
async def generate_quiz(req: QuizGenerateRequest):
    text = _build_session_text(req.session_id)
    if not text.strip():
        raise HTTPException(400, "No session content")
    if not COLEARNER_API_KEY:
        raise HTTPException(503, "LLM API not configured")
    qtype = req.qtype if req.qtype in ("mcq", "short", "mixed") else "mcq"
    try:
        raw = await _call_llm(QUIZ_SYSTEM, f"Generate {req.count} {qtype} questions{f' about {req.topic}' if req.topic else ''} from:\n\n{text}", max_tokens=2000, temperature=0.6, json_mode=True)
        data = json.loads(raw)
    except Exception as e:
        raise HTTPException(502, f"Quiz generation failed: {e}")
    quiz_id = db_execute(
        "INSERT INTO quizzes (session_id, title, questions) VALUES (?, ?, ?)",
        (req.session_id, data.get("title", "Quiz"), json.dumps(data.get("questions", []))),
    )
    return {"id": quiz_id, "title": data.get("title", "Quiz"), "question_count": len(data.get("questions", [])), "questions": data.get("questions", [])}


@app.get("/quizzes")
async def list_quizzes():
    quizzes = db_fetch("SELECT id, session_id, title, created_at FROM quizzes ORDER BY created_at DESC LIMIT 50")
    for q in quizzes:
        questions = json.loads(db_fetch("SELECT questions FROM quizzes WHERE id = ?", (q["id"],))[0]["questions"]) if db_fetch("SELECT questions FROM quizzes WHERE id = ?", (q["id"],)) else []
        q["question_count"] = len(questions) if isinstance(questions, list) else 0
    return {"quizzes": quizzes}


@app.get("/quizzes/{quiz_id}")
async def get_quiz(quiz_id: int):
    quiz = db_fetch("SELECT * FROM quizzes WHERE id = ?", (quiz_id,))
    if not quiz:
        raise HTTPException(404, "Quiz not found")
    q = quiz[0]
    questions = json.loads(q["questions"]) if isinstance(q.get("questions"), str) else q.get("questions", [])
    # Strip answers from response — client should NOT see correct answers before submitting
    safe_questions = []
    for qq in questions:
        safe = {k: v for k, v in qq.items() if k != "answer"}
        safe_questions.append(safe)
    q["questions"] = safe_questions
    return q

@app.delete("/quizzes/{quiz_id}")
async def delete_quiz(quiz_id: int):
    existing = db_fetch("SELECT id FROM quizzes WHERE id = ?", (quiz_id,))
    if not existing:
        raise HTTPException(404, "Quiz not found")
    db_execute("DELETE FROM quiz_attempts WHERE quiz_id = ?", (quiz_id,))
    db_execute("DELETE FROM quizzes WHERE id = ?", (quiz_id,))
    return {"deleted": quiz_id}


@app.post("/quizzes/{quiz_id}/submit")
async def submit_quiz(quiz_id: int, req: QuizSubmitRequest):
    quiz = db_fetch("SELECT * FROM quizzes WHERE id = ?", (quiz_id,))
    if not quiz:
        raise HTTPException(404, "Quiz not found")
    questions = json.loads(quiz[0]["questions"]) if isinstance(quiz[0].get("questions"), str) else quiz[0].get("questions", [])
    correct = 0
    corrections = []
    for a in req.answers:
        idx = a.get("question_idx", -1)
        user_answer = a.get("answer", "").strip().upper()
        if 0 <= idx < len(questions):
            q = questions[idx]
            expected = q.get("answer", "").strip().upper()
            is_correct = user_answer == expected
            if is_correct:
                correct += 1
            else:
                corrections.append({"question_idx": idx, "your_answer": user_answer, "correct_answer": expected, "explanation": q.get("explanation", "")})
    score = round(correct / len(questions) * 100, 1) if questions else 0
    db_execute("INSERT INTO quiz_attempts (quiz_id, answers, score, total) VALUES (?, ?, ?, ?)", (quiz_id, json.dumps(req.answers), score, len(questions)))
    return {"score": score, "correct": correct, "total": len(questions), "corrections": corrections}


# ═══════════════════════════════════════════════════════
# v5: 4. Audio Overviews (AI Podcast)
# ═══════════════════════════════════════════════════════

AUDIO_SCRIPT_SYSTEM = """You write podcast scripts from lecture content for an EMBA student.
Create a natural conversation between two hosts: HOST_A (curious, asks questions) and HOST_B (knowledgeable, explains concepts).
The podcast should cover the key points from the source material in an engaging 5-10 minute format.

Format EXACTLY like this:
HOST_A: [dialogue]
HOST_B: [dialogue]
... alternating ...

Keep each speaker turn to 1-3 sentences. Total turns: 10-16.
Start with HOST_A introducing the topic. End with HOST_B giving a key takeaway."""


@app.post("/audio/generate")
async def generate_audio_overview(req: AudioGenerateRequest):
    text = _build_session_text(req.session_id)
    if not text.strip():
        raise HTTPException(400, "No content found for this session. Record a lecture or add entries first, then generate audio.")
    if not COLEARNER_API_KEY:
        raise HTTPException(503, "LLM API not configured")
    style = req.style if req.style in ("podcast", "summary", "lecture") else "podcast"
    try:
        script = await _call_llm(AUDIO_SCRIPT_SYSTEM, f"Create a {style} from:\n\n{text}", max_tokens=1200, temperature=0.7)
    except Exception as e:
        raise HTTPException(502, f"Script generation failed: {e}")
    lines = [l.strip() for l in script.split("\n") if l.strip().startswith(("HOST_A:", "HOST_B:"))]
    title = f"Lecture Overview - {datetime.now().strftime('%b %d, %Y')}"
    overview_id = db_execute(
        "INSERT INTO audio_overviews (session_id, title, script, style) VALUES (?, ?, ?, ?)",
        (req.session_id, title, json.dumps(lines), style),
    )
    return {"id": overview_id, "title": title, "script": lines, "turns": len(lines), "audio_available": False, "note": "TTS synthesis requires server-side audio generation. Script ready for playback."}


@app.get("/audio")
async def list_audio_overviews():
    overviews = db_fetch("SELECT id, session_id, title, style, duration_sec, created_at FROM audio_overviews ORDER BY created_at DESC LIMIT 30")
    return {"overviews": overviews}


@app.get("/audio/{overview_id}")
async def get_audio_overview(overview_id: int):
    ov = db_fetch("SELECT * FROM audio_overviews WHERE id = ?", (overview_id,))
    if not ov:
        raise HTTPException(404, "Overview not found")
    o = ov[0]
    o["script"] = json.loads(o["script"]) if isinstance(o.get("script"), str) else o.get("script", [])
    return o

@app.delete("/audio/{overview_id}")
async def delete_audio_overview(overview_id: int):
    existing = db_fetch("SELECT id FROM audio_overviews WHERE id = ?", (overview_id,))
    if not existing:
        raise HTTPException(404, "Overview not found")
    db_execute("DELETE FROM audio_overviews WHERE id = ?", (overview_id,))
    return {"deleted": overview_id}


# ═══════════════════════════════════════════════════════
# v5: 5. Case Study Analyzer
# ═══════════════════════════════════════════════════════

CASE_SYSTEM = """You are a case study analysis assistant for an EMBA student at Sasin School of Management.
Given a business case, analyze it through a structured 5-step framework:

Step 1 - SITUATION: Company background, industry context, key players
Step 2 - PROBLEM: Core problem statement, symptoms vs root causes
Step 3 - FRAMEWORKS: Apply 2-3 relevant frameworks (Porter, SWOT, PESTLE, BCG, VRIO, Value Chain, Blue Ocean, etc.) with specific evidence from the case
Step 4 - ALTERNATIVES: 3-4 viable strategic options with pros/cons for each
Step 5 - RECOMMENDATION: Your recommended course of action with implementation steps and risk mitigation

Be specific. Cite frameworks by name. Use data from the case when available."""


@app.post("/cases/analyze")
async def analyze_case(req: CaseAnalyzeRequest):
    text = _build_session_text(req.session_id)
    if not text.strip() and not req.company:
        raise HTTPException(400, "No session content or company provided")
    if not COLEARNER_API_KEY:
        raise HTTPException(503, "LLM API not configured")
    user_msg = f"COMPANY: {req.company}\nINDUSTRY: {req.industry}\n\nCASE CONTENT:\n{text}"
    try:
        analysis = await _call_llm(CASE_SYSTEM, user_msg, max_tokens=2000, temperature=0.5)
    except Exception as e:
        raise HTTPException(502, f"Case analysis failed: {e}")
    # Detect frameworks applied
    frameworks = _detect_frameworks(analysis)
    case_id = db_execute(
        "INSERT INTO case_studies (session_id, company, industry, step, analysis, frameworks_applied) VALUES (?, ?, ?, 5, ?, ?)",
        (req.session_id, req.company, req.industry, analysis, json.dumps(frameworks)),
    )
    return {"id": case_id, "company": req.company, "analysis": analysis, "frameworks": frameworks, "steps_completed": 5}


@app.get("/cases")
async def list_cases():
    cases = db_fetch("SELECT id, session_id, company, industry, step, created_at FROM case_studies ORDER BY created_at DESC LIMIT 30")
    return {"cases": cases}


@app.get("/cases/{case_id}")
async def get_case(case_id: int):
    case = db_fetch("SELECT * FROM case_studies WHERE id = ?", (case_id,))
    if not case:
        raise HTTPException(404, "Case not found")
    c = case[0]
    c["frameworks_applied"] = json.loads(c["frameworks_applied"]) if isinstance(c.get("frameworks_applied"), str) else c.get("frameworks_applied", [])
    return c


@app.get("/cases/{case_id}/export")
async def export_case(case_id: int):
    case = db_fetch("SELECT * FROM case_studies WHERE id = ?", (case_id,))
    if not case:
        raise HTTPException(404, "Case not found")
    c = case[0]
    from fastapi.responses import Response
    return Response(content=c["analysis"], media_type="text/markdown",
                    headers={"Content-Disposition": f"attachment; filename=case_{c['company'] or case_id}.md"})

@app.delete("/cases/{case_id}")
async def delete_case(case_id: int):
    existing = db_fetch("SELECT id FROM case_studies WHERE id = ?", (case_id,))
    if not existing:
        raise HTTPException(404, "Case not found")
    db_execute("DELETE FROM case_studies WHERE id = ?", (case_id,))
    return {"deleted": case_id}

# Fix: detect frameworks in markdown-formatted text (handles **bold**)
def _detect_frameworks(text: str) -> list:
    """Detect business frameworks in text, handling markdown formatting and smart quotes."""
    import re as _re
    clean = _re.sub(r'\*\*([^*]+)\*\*', r'\1', text)  # strip bold
    clean = _re.sub(r'\*([^*]+)\*', r'\1', clean)     # strip italic
    # Normalize smart quotes to straight quotes
    clean = clean.replace('\u2018', "'").replace('\u2019', "'").replace('\u201c', '"').replace('\u201d', '"')
    frameworks = []
    for fw in ["Porter's Five Forces", "SWOT", "PESTLE", "BCG Matrix", "Blue Ocean", 
               "Value Chain", "VRIO", "Balanced Scorecard", "Ansoff", "Core Competency",
               "Resource-Based View", "Design Thinking", "Lean Startup", "Business Model Canvas"]:
        if fw.lower() in clean.lower():
            frameworks.append(fw)
    return frameworks


# ═══════════════════════════════════════════════════════
# v5: 6. Class Templates
# ═══════════════════════════════════════════════════════

@app.get("/templates")
async def list_templates():
    templates = db_fetch("SELECT * FROM class_templates ORDER BY is_default DESC, name ASC")
    for t in templates:
        t["structure"] = json.loads(t["structure"]) if isinstance(t.get("structure"), str) else t.get("structure", {})
    return {"templates": templates}


@app.post("/templates")
async def create_template(req: TemplateCreateRequest):
    tid = db_execute(
        "INSERT OR IGNORE INTO class_templates (name, structure) VALUES (?, ?)",
        (req.name, json.dumps(req.structure)),
    )
    return {"id": tid, "name": req.name}


@app.post("/templates/apply")
async def apply_template(session_id: str = Form(...), template_id: int = Form(...)):
    template = db_fetch("SELECT * FROM class_templates WHERE id = ?", (template_id,))
    if not template:
        raise HTTPException(404, "Template not found")
    t = template[0]
    structure = json.loads(t["structure"]) if isinstance(t.get("structure"), str) else t.get("structure", {})
    sections = structure.get("sections", [])
    entries = get_entries(session_id)
    if not entries:
        raise HTTPException(400, "No entries in session")
    # Build formatted output from session entries mapped to template sections
    text = _build_session_text(session_id, 4000)
    formatted = f"# {t['name']}: {sections[0] if sections else 'Notes'}\n\n"
    if text:
        formatted += text
    formatted += f"\n\n---\n*Generated from template '{t['name']}' | {len(sections)} sections*"
    return {"template_name": t["name"], "sections": sections, "formatted": formatted}


# ═══════════════════════════════════════════════════════
# v5: 7. Action Items
# ═══════════════════════════════════════════════════════

@app.post("/sessions/{session_id}/action-items/extract")
async def extract_action_items(session_id: str):
    text = _build_session_text(session_id)
    if not text.strip():
        raise HTTPException(400, "No session content")
    if not COLEARNER_API_KEY:
        raise HTTPException(503, "LLM API not configured")
    prompt = """Extract action items, decisions, and deadlines from this transcript.
Return a JSON object: {"items": [{"text": "what needs to be done", "speaker": "who mentioned it if identifiable", "deadline": "when it's due if mentioned"}]}"""
    try:
        raw = await _call_llm(prompt, f"Extract action items from:\n\n{text}", max_tokens=800, temperature=0.3, json_mode=True)
        data = json.loads(raw)
        items = data.get("items", [])
    except Exception as e:
        raise HTTPException(502, f"Action item extraction failed: {e}")
    created = []
    for item in items:
        aid = db_execute(
            "INSERT INTO action_items (session_id, text, speaker, deadline) VALUES (?, ?, ?, ?)",
            (session_id, item.get("text", ""), item.get("speaker", ""), item.get("deadline", "")),
        )
        created.append({"id": aid, "text": item.get("text", ""), "speaker": item.get("speaker", ""), "deadline": item.get("deadline", "")})
    return {"items": created, "total": len(created)}


@app.get("/sessions/{session_id}/action-items")
async def list_action_items(session_id: str):
    items = db_fetch("SELECT * FROM action_items WHERE session_id = ? ORDER BY status, created_at DESC", (session_id,))
    return {"session_id": session_id, "items": items, "total": len(items)}


@app.patch("/action-items/{item_id}")
async def update_action_item(item_id: int, status: str = Form(...)):
    if status not in ("open", "done", "archived"):
        raise HTTPException(400, "Status must be: open, done, or archived")
    db_execute("UPDATE action_items SET status = ? WHERE id = ?", (status, item_id))
    return {"id": item_id, "status": status}

@app.delete("/action-items/{item_id}")
async def delete_action_item(item_id: int):
    existing = db_fetch("SELECT id FROM action_items WHERE id = ?", (item_id,))
    if not existing:
        raise HTTPException(404, "Action item not found")
    db_execute("DELETE FROM action_items WHERE id = ?", (item_id,))
    return {"deleted": item_id}


# ═══════════════════════════════════════════════════════
# v5: 8. Collaboration
# ═══════════════════════════════════════════════════════

@app.post("/sessions/{session_id}/share")
async def share_session(session_id: str):
    token = hashlib.sha256(f"{session_id}_{datetime.now().isoformat()}".encode()).hexdigest()[:16]
    db_execute("UPDATE sessions SET tags = ? WHERE id = ?", (f"shared:{token}", session_id))
    return {"share_token": token, "url": f"/shared/{token}"}


@app.get("/shared/{token}")
async def get_shared_session(token: str):
    session = db_fetch("SELECT * FROM sessions WHERE tags LIKE ?", (f"%shared:{token}%",))
    if not session:
        raise HTTPException(404, "Shared session not found")
    entries = get_entries(session[0]["id"])
    return {"session": session[0], "entry_count": len(entries), "entries": entries}


# ═══════════════════════════════════════════════════════
# v5: 9. Mind Map Generator
# ═══════════════════════════════════════════════════════

MINDMAP_SYSTEM = """Generate a mind map from lecture content. Return JSON:
{"central": "Main Topic", "nodes": [{"id": "1", "label": "Subtopic", "parent": "0"}, ...], "edges": [{"from": "0", "to": "1", "label": "relationship"}, ...]}
Node "0" is the central topic. Max 20 nodes. Group concepts logically."""


@app.get("/mindmap/{session_id}")
async def generate_mindmap(session_id: str):
    text = _build_session_text(session_id)
    if not text.strip():
        raise HTTPException(400, "No session content")
    if not COLEARNER_API_KEY:
        raise HTTPException(503, "LLM API not configured")
    try:
        raw = await _call_llm(MINDMAP_SYSTEM, f"Create mind map from:\n\n{text}", max_tokens=1000, temperature=0.4, json_mode=True)
        data = json.loads(raw)
    except Exception as e:
        raise HTTPException(502, f"Mind map generation failed: {e}")
    return data


# ═══════════════════════════════════════════════════════
# v5: 10. Schedule Planner
# ═══════════════════════════════════════════════════════

@app.get("/schedule/due")
async def schedule_due():
    due_cards = db_fetch("SELECT COUNT(*) as c FROM flashcards WHERE next_review <= datetime('now')")[0]["c"]
    due_actions = db_fetch("SELECT COUNT(*) as c FROM action_items WHERE status = 'open'")[0]["c"]
    recent_quizzes = db_fetch("SELECT COUNT(*) as c FROM quizzes WHERE created_at > datetime('now', '-7 days')")[0]["c"]
    return {"flashcards_due": due_cards, "open_action_items": due_actions, "quizzes_this_week": recent_quizzes}


@app.get("/schedule/calendar")
async def schedule_calendar(days: int = 7):
    from datetime import timedelta
    now = datetime.now(timezone.utc)
    calendar = []
    for i in range(days):
        day = now + timedelta(days=i)
        day_str = day.strftime("%Y-%m-%d")
        next_str = (day + timedelta(days=1)).strftime("%Y-%m-%d")
        cards = db_fetch(
            "SELECT COUNT(*) as c FROM flashcards WHERE next_review >= ? AND next_review < ?",
            (day_str, next_str),
        )[0]["c"]
        calendar.append({"date": day_str, "flashcards_due": cards})
    return {"calendar": calendar, "days": days}


# ═══════════════════════════════════════════════════════
# v5: 11. Presentation Generator
# ═══════════════════════════════════════════════════════

PRESENT_SYSTEM = """Generate presentation slides from lecture content. Return JSON:
{"title": "Presentation Title", "slides": [{"title": "Slide Title", "bullets": ["point 1", "point 2", "point 3"], "notes": "speaker notes"}, ...]}
Create 5-10 slides. Bullets should be concise. Notes should help the presenter elaborate."""


@app.post("/present/generate")
async def generate_presentation(req: PresentGenerateRequest):
    text = _build_session_text(req.session_id)
    if not text.strip():
        raise HTTPException(400, "No session content")
    if not COLEARNER_API_KEY:
        raise HTTPException(503, "LLM API not configured")
    try:
        raw = await _call_llm(PRESENT_SYSTEM, f"Generate {req.style} presentation from:\n\n{text}", max_tokens=1500, temperature=0.5, json_mode=True)
        data = json.loads(raw)
    except Exception as e:
        raise HTTPException(502, f"Presentation generation failed: {e}")
    return data


# ═══════════════════════════════════════════════════════
# v5: 12. Web Research (Perplexity-style)
# ═══════════════════════════════════════════════════════

RESEARCH_SYSTEM = """You research business topics for an EMBA student. Given a query, provide:
1. Key findings (2-3 bullet points)
2. Recent developments (last 2 years if known)
3. Relevance to business strategy
4. Recommended frameworks to apply
Be concise — under 250 words. If you don't know something recent, say so and provide timeless principles instead."""


@app.post("/research/search")
async def research_search(req: ResearchRequest):
    if not COLEARNER_API_KEY:
        raise HTTPException(503, "LLM API not configured")
    context = ""
    if req.session_id:
        context = _build_session_text(req.session_id, 2000)
    user_msg = f"SESSION CONTEXT:\n{context}\n\nRESEARCH QUERY: {req.query}" if context else f"RESEARCH QUERY: {req.query}"
    try:
        result = await _call_llm(RESEARCH_SYSTEM, user_msg, max_tokens=600, temperature=0.4)
    except Exception as e:
        raise HTTPException(502, f"Research failed: {e}")
    return {"query": req.query, "result": result}


# ═══════════════════════════════════════════════════════
# v5: Enhanced Chat with Source Citations
# ═══════════════════════════════════════════════════════

@app.post("/chat/v2")
async def chat_v2(req: ChatRequest, stream: bool = Query(False)):
    """Enhanced chat with mandatory source citations."""
    if not req.question.strip():
        return {"answer": "Please ask a question about your lectures, frameworks, or study materials.", "sources": []}
    context_parts = []
    sources = []
    if req.session_id:
        entries = get_entries(req.session_id)
        transcript_entries = [e for e in entries if e["box"] == "transcript" and e.get("content_type") != "slide_image"]
        if transcript_entries:
            transcript_text = " ".join(e["content"] for e in transcript_entries[-10:])
            context_parts.append(f"RECENT TRANSCRIPT:\n{transcript_text[-2000:]}")
        insight_entries = [e for e in entries if e["box"] == "colearner"]
        if insight_entries:
            insight_text = " ".join(e["content"] for e in insight_entries[-5:])
            context_parts.append(f"AI INSIGHTS:\n{insight_text[-1000:]}")
    # Fetch KB
    kb_context = ""
    try:
        global _http_client
        if _http_client is None:
            _http_client = httpx.AsyncClient(timeout=httpx.Timeout(15.0))
        kb_resp = await _http_client.get("http://localhost:8001/api/v1/knowledge/emba-2026/files", timeout=5.0)
        if kb_resp.status_code == 200:
            data = kb_resp.json()
            files = data.get('files', data) if isinstance(data, dict) else data
            if isinstance(files, list) and len(files) > 0:
                question_lower = req.question.lower()
                relevant_files = []
                for f in files[:30]:
                    name = f["name"] if isinstance(f, dict) else str(f)
                    name_lower = name.lower()
                    keywords = question_lower.split()
                    matches = sum(1 for kw in keywords if len(kw) > 2 and kw in name_lower)
                    if matches >= 2 or any(kw in name_lower for kw in keywords if len(kw) > 4):
                        relevant_files.append(name)
                for rf_name in relevant_files[:3]:
                    try:
                        content_resp = await _http_client.get(f"http://localhost:8001/api/v1/knowledge/emba-2026/files/{rf_name}", timeout=3.0)
                        if content_resp.status_code == 200:
                            content = content_resp.text[:1500]
                            kb_context += f"\n--- {rf_name} ---\n{content}\n"
                            sources.append({"file": rf_name, "preview": content[:200]})
                    except Exception:
                        pass
                if kb_context:
                    context_parts.append(f"RELEVANT KNOWLEDGE BASE CONTENT:{kb_context[:3000]}")
    except Exception as e:
        logger.warning(f"KB fetch failed: {e}")
    if not context_parts:
        context = "No session context or knowledge base yet. The user hasn't captured a lecture or uploaded materials. Encourage them to start a capture or upload PDFs, but still answer their question helpfully with general knowledge."
    else:
        context = "\n\n---\n\n".join(context_parts)
    system = CHAT_SYSTEM_PROMPT + "\n\nCRITICAL: You MUST cite specific sources when answering. If using KB content, mention the filename. If using transcript, mention 'from your lecture'. Format citations as **Source: filename** or **[Lecture Transcript]**."
    if stream:
        return StreamingResponse(
            _call_llm_stream(system, f"CONTEXT:\n{context}\n\nQUESTION: {req.question}", max_tokens=500, temperature=0.5),
            media_type="text/event-stream",
            headers={"X-Stream-Format": "sse", "X-Endpoint": "chat"}
        )
    try:
        resp = await _http_client.post(
            COLEARNER_URL,
            headers={"Authorization": f"Bearer {COLEARNER_API_KEY}", "Content-Type": "application/json"},
            json={"model": COLEARNER_MODEL, "messages": [{"role": "system", "content": system}, {"role": "user", "content": f"CONTEXT:\n{context}\n\nQUESTION: {req.question}"}], "max_tokens": 500, "temperature": 0.5},
            timeout=25.0,
        )
        if resp.status_code != 200:
            return {"answer": f"AI error: {resp.status_code}", "error": str(resp.status_code)}
        data = resp.json()
        answer = data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return {"answer": f"Error: {str(e)[:200]}", "error": str(e)[:200]}
    return {"answer": answer, "session_id": req.session_id, "model": COLEARNER_MODEL, "sources": sources}


# ── Serve Frontend ──

_CAPTURE_HTML = Path("/data/sasin-cfoth-ai/capture/index.html")
if not _CAPTURE_HTML.exists():
    _CAPTURE_HTML = Path("/docker/hermes-bot/data/sasin-cfoth-ai/capture/index.html")
_HUB_HTML = Path("/data/sasin-cfoth-ai/index.html")


@app.get("/")
async def serve_frontend(request: Request):
    host = request.headers.get("host", "")
    # sasin.cfoth.ai → hub page; capture.sasin.cfoth.ai → capture app
    if "sasin.cfoth.ai" in host and "capture" not in host:
        if _HUB_HTML.exists():
            return FileResponse(str(_HUB_HTML))
    if _CAPTURE_HTML.exists():
        return FileResponse(str(_CAPTURE_HTML))
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



# ═══════════════════════════════════════════════════════════════
# Second Brain Knowledge Graph Proxy
# ═══════════════════════════════════════════════════════════════

async def _get_http_client():
    """Lazy-init httpx client shared across proxy endpoints."""
    global _http_client
    if _http_client is None:
        _http_client = httpx.AsyncClient(
            http2=True, timeout=httpx.Timeout(30.0),
            limits=httpx.Limits(max_keepalive_connections=10, max_connections=30),
        )
    return _http_client


@app.get("/second-brain", response_class=HTMLResponse)
async def serve_second_brain():
    """Serve the interactive knowledge graph visualization."""
    graph_html = Path("/data/emba-second-brain/index.html")
    if graph_html.exists():
        # Read and fix fetch URLs to use the proxy prefix
        html = graph_html.read_text()
        html = html.replace("fetch('/graph'", "fetch('/second-brain/graph'")
        html = html.replace('fetch("/graph"', 'fetch("/second-brain/graph"')
        html = html.replace("fetch('/documents'", "fetch('/second-brain/documents'")
        html = html.replace('fetch("/documents"', 'fetch("/second-brain/documents"')
        html = html.replace("fetch('/sync'", "fetch('/second-brain/sync'")
        html = html.replace('fetch("/sync"', 'fetch("/second-brain/sync"')
        html = html.replace("fetch('/drive-file/'", "fetch('/second-brain/drive-file/'")
        html = html.replace('fetch("/drive-file/"', 'fetch("/second-brain/drive-file/"')
        html = html.replace("fetch('/doc/'", "fetch('/second-brain/doc/'")
        html = html.replace('fetch("/doc/"', 'fetch("/second-brain/doc/"')
        html = html.replace("fetch('/search", "fetch('/second-brain/search")
        html = html.replace('fetch("/search', 'fetch("/second-brain/search')
        html = html.replace("fetch('/concept/", "fetch('/second-brain/concept/")
        html = html.replace('fetch("/concept/', 'fetch("/second-brain/concept/')
        return HTMLResponse(html)
    return RedirectResponse("https://brain.cfoth.ai/", status_code=302)


@app.get("/second-brain/graph")
async def proxy_graph():
    """Proxy graph API from local Second Brain server with graceful fallback."""
    try:
        client = await _get_http_client()
        resp = await client.get("http://localhost:8400/graph", timeout=5.0)
        return JSONResponse(resp.json())
    except Exception:
        return JSONResponse({"nodes": [], "links": [], "updated": "unavailable", "note": "Second Brain server offline. Graph data embedded in page."})


@app.get("/second-brain/search")
async def proxy_search(q: str = Query(...)):
    """Proxy search API from local Second Brain server."""
    resp = await _http_client.get(f"http://localhost:8400/search?q={q}", timeout=10.0)
    return JSONResponse(resp.json())


@app.get("/second-brain/concept/{name}")
async def proxy_concept(name: str):
    """Proxy concept detail API."""
    resp = await _http_client.get(f"http://localhost:8400/concept/{name}", timeout=10.0)
    return JSONResponse(resp.json())


@app.get("/second-brain/download/{kb_name}/{filename:path}")
async def proxy_download(kb_name: str, filename: str):
    """Proxy file download from Second Brain → DeepTutor KB."""
    client = await _get_http_client()
    url = f"http://localhost:8400/download/{kb_name}/{filename}"
    resp = await client.get(url, timeout=10.0)
    if resp.status_code != 200:
        raise HTTPException(resp.status_code, f"File not found: {resp.status_code}")
    content_type = resp.headers.get("content-type", "application/octet-stream")
    return Response(content=resp.content, media_type=content_type,
                   headers={"Content-Disposition": f'attachment; filename="{filename}"'})


@app.get("/second-brain/doc/{name:path}")
async def proxy_doc(name: str):
    """Proxy document detail API."""
    resp = await _http_client.get(f"http://localhost:8400/doc/{name}", timeout=10.0)
    return JSONResponse(resp.json())


@app.get("/second-brain/stats")
async def proxy_stats():
    """Proxy stats API."""
    resp = await _http_client.get("http://localhost:8400/stats", timeout=10.0)
    return JSONResponse(resp.json())


@app.delete("/second-brain/doc/{name:path}")
async def proxy_delete_doc(name: str):
    """Proxy delete document from Second Brain."""
    resp = await _http_client.delete(f"http://localhost:8400/doc/{name}", timeout=15.0)
    return JSONResponse(resp.json(), status_code=resp.status_code)


@app.get("/second-brain/documents")
async def proxy_documents():
    """Proxy documents list from Second Brain."""
    resp = await _http_client.get("http://localhost:8400/documents", timeout=10.0)
    return JSONResponse(resp.json())


@app.post("/second-brain/sync")
async def proxy_sync():
    """Proxy manual sync trigger to Second Brain."""
    resp = await _http_client.post("http://localhost:8400/sync", timeout=120.0)
    return JSONResponse(resp.json())


@app.delete("/second-brain/drive-file/{drive_id}")
async def proxy_delete_drive_file(drive_id: str):
    """Proxy drive file removal from Second Brain sync state."""
    resp = await _http_client.delete(f"http://localhost:8400/drive-file/{drive_id}", timeout=10.0)
    return JSONResponse(resp.json(), status_code=resp.status_code)


@app.get("/second-brain/calendar")
async def proxy_calendar():
    """Proxy calendar events from Second Brain server."""
    try:
        resp = await _http_client.get("http://localhost:8400/calendar", timeout=10.0)
        return JSONResponse(resp.json())
    except Exception:
        return JSONResponse({"events": [], "total": 0, "synced_at": None, "note": "Calendar server unavailable"})


@app.on_event("startup")
async def startup():
    global _http_client
    if _http_client is None:
        _http_client = httpx.AsyncClient(
            http2=True, timeout=httpx.Timeout(30.0),
            limits=httpx.Limits(max_keepalive_connections=10, max_connections=30),
        )
        logger.info("httpx client initialized")


@app.on_event("shutdown")
async def shutdown():
    global _http_client
    if _http_client:
        await _http_client.aclose()
        _http_client = None


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8896))
    uvicorn.run(app, host="0.0.0.0", port=port, reload=False)
