"""
Second Brain — Unified Ingestion Pipeline
Accepts PDFs, images, audio → extracts text → feeds to DeepTutor Knowledge
"""
import os
import json
import base64
import tempfile
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Optional

import httpx
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import uvicorn

app = FastAPI(title="Sasin Second Brain")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Config
OPENROUTER_KEY = "sk-or-v1-093cb90db13c72b4ebad936adec7e07923cef1f92be366b30397ea2cc7017e8"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
DEEPTUTOR_URL = "http://localhost:8001"  # internal Docker network
KNOWLEDGE_BASE = "emba-2026"
INGEST_DIR = Path("/root/second_brain")
INGEST_DIR.mkdir(exist_ok=True)
(INGEST_DIR / "processed").mkdir(exist_ok=True)


# ─── DeepTutor Knowledge Base Setup ─────────────────────────────────────────

async def ensure_knowledge_base():
    """Create the EMBA-2026 knowledge base if it doesn't exist."""
    async with httpx.AsyncClient(timeout=30) as client:
        # Check if KB exists
        resp = await client.get(f"{DEEPTUTOR_URL}/api/v1/knowledge/list")
        if resp.status_code == 200:
            data = resp.json()
            # Response could be {"knowledge_bases": [...]} or [...]
            if isinstance(data, dict):
                kbs = data.get("knowledge_bases", [])
            else:
                kbs = data
            # Extract names
            kb_names = []
            for kb in kbs:
                if isinstance(kb, dict):
                    kb_names.append(kb.get("name", ""))
                elif isinstance(kb, str):
                    kb_names.append(kb)
            if KNOWLEDGE_BASE in kb_names:
                return True
        
        # Try to create it
        try:
            resp = await client.post(
                f"{DEEPTUTOR_URL}/api/v1/knowledge/create",
                data={"name": KNOWLEDGE_BASE, "description": "Sasin EMBA 2026 — All lectures, slides, notes, and materials"}
            )
            return resp.status_code in (200, 201)
        except:
            return False


async def upload_to_deeptutor(file_path: Path, original_name: str) -> dict:
    """Upload a processed file to DeepTutor Knowledge base."""
    try:
        async with httpx.AsyncClient(timeout=60) as client:
            with open(file_path, "rb") as f:
                resp = await client.post(
                    f"{DEEPTUTOR_URL}/api/v1/knowledge/{KNOWLEDGE_BASE}/upload",
                    files={"files": (original_name, f, "application/octet-stream")}
                )
            return {"status": resp.status_code, "detail": resp.text[:200]}
    except Exception as e:
        return {"status": 0, "detail": str(e)[:200]}


# ─── Image OCR via Vision LLM ───────────────────────────────────────────────

async def ocr_image(image_path: Path) -> str:
    """Extract text from an image using OpenRouter vision model."""
    # Read and encode image
    with open(image_path, "rb") as f:
        image_bytes = f.read()
    
    suffix = image_path.suffix.lower()
    mime_map = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png",
                ".webp": "image/webp", ".gif": "image/gif", ".bmp": "image/bmp"}
    mime_type = mime_map.get(suffix, "image/png")
    
    image_b64 = base64.b64encode(image_bytes).decode()
    data_url = f"data:{mime_type};base64,{image_b64}"
    
    try:
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(
                OPENROUTER_URL,
                headers={
                    "Authorization": f"Bearer {OPENROUTER_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "openai/gpt-5",
                    "messages": [{
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": data_url, "detail": "high"}},
                            {"type": "text", "text": "Extract ALL text visible in this image. Include slide titles, bullet points, labels, diagram captions, and handwritten notes. Output only the extracted text, preserving structure. Do not add commentary."}
                        ]
                    }],
                    "max_tokens": 2000
                }
            )
            
            if resp.status_code == 200:
                data = resp.json()
                return data["choices"][0]["message"]["content"].strip()
            elif resp.status_code == 401:
                raise Exception("OpenRouter API key is invalid or expired. Image saved but not OCR'd — update the key in second_brain.py.")
            else:
                raise Exception(f"OCR API error: {resp.status_code}")
    except httpx.ConnectError:
        raise Exception("Cannot reach OpenRouter API. Image saved but not OCR'd.")
    except Exception as e:
        if "401" in str(e) or "invalid" in str(e).lower():
            raise
        raise Exception(f"OCR failed: {str(e)[:100]}")


# ─── PDF Text Extraction ────────────────────────────────────────────────────

async def extract_pdf_text(pdf_path: Path) -> str:
    """Extract text from PDF using PyPDF2 or pdftotext."""
    try:
        from PyPDF2 import PdfReader
        reader = PdfReader(str(pdf_path))
        text = []
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text.append(page_text)
        return "\n\n".join(text)
    except ImportError:
        # Fallback to pdftotext command
        result = subprocess.run(
            ["pdftotext", str(pdf_path), "-"],
            capture_output=True, text=True, timeout=30
        )
        return result.stdout


# ─── Audio Transcription ────────────────────────────────────────────────────

async def transcribe_audio(audio_path: Path) -> str:
    """Transcribe audio using faster-whisper."""
    from faster_whisper import WhisperModel
    
    # Convert to WAV if needed
    wav_path = audio_path
    if audio_path.suffix not in (".wav",):
        wav_path = audio_path.with_suffix(".wav")
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(audio_path), "-ar", "16000", "-ac", "1", "-f", "wav", str(wav_path)],
            capture_output=True, timeout=60
        )
    
    model = WhisperModel("small", device="cpu", compute_type="int8")
    segments, info = model.transcribe(str(wav_path), beam_size=2)
    
    text = " ".join(seg.text.strip() for seg in segments)
    
    # Cleanup temp WAV
    if wav_path != audio_path and wav_path.exists():
        wav_path.unlink()
    
    return text


# ─── API Endpoints ──────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "kb": KNOWLEDGE_BASE}


@app.post("/ingest")
async def ingest_file(
    file: UploadFile = File(...),
    file_type: str = Form("auto"),  # auto, pdf, image, audio
    title: str = Form(""),
    tags: str = Form(""),
):
    """
    Universal ingestion endpoint.
    - Images → OCR via GPT-5 vision → save as .txt → upload to DeepTutor
    - PDFs → extract text → save original + .txt → upload to DeepTutor
    - Audio → transcribe via Whisper → save as .txt → upload to DeepTutor
    """
    
    # Save uploaded file
    original_name = file.filename or "upload"
    suffix = Path(original_name).suffix.lower()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = f"{timestamp}_{original_name}"
    file_path = INGEST_DIR / "processed" / safe_name
    file_path.parent.mkdir(exist_ok=True)
    
    content = await file.read()
    file_path.write_bytes(content)
    
    # Determine type
    image_exts = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp", ".tiff", ".tif", ".heic"}
    audio_exts = {".mp3", ".wav", ".m4a", ".ogg", ".webm", ".opus", ".flac"}
    pdf_exts = {".pdf"}
    
    if file_type == "auto":
        if suffix in image_exts:
            file_type = "image"
        elif suffix in audio_exts:
            file_type = "audio"
        elif suffix in pdf_exts:
            file_type = "pdf"
        elif suffix in {".txt", ".md", ".markdown", ".text"}:
            file_type = "text"
        else:
            raise HTTPException(400, f"Unsupported file type: {suffix}. Supported: PDF, images (JPG/PNG/WebP), audio (MP3/WAV/M4A), text files (TXT/MD)")
    
    result = {
        "original_name": original_name,
        "type": file_type,
        "text_extracted": "",
        "word_count": 0,
        "deep_tutor_uploaded": False
    }
    
    try:
        # Process based on type
        if file_type == "image":
            result["text_extracted"] = await ocr_image(file_path)
        elif file_type == "audio":
            result["text_extracted"] = await transcribe_audio(file_path)
        elif file_type == "pdf":
            result["text_extracted"] = await extract_pdf_text(file_path)
        elif file_type == "text":
            # Plain text — just read it
            result["text_extracted"] = file_path.read_text()
        else:
            raise HTTPException(400, f"Unsupported file type: {suffix}")
        
        result["word_count"] = len(result["text_extracted"].split())
        
        # Save extracted text
        txt_path = file_path.with_suffix(".txt")
        header = f"# {title or original_name}\n"
        if tags:
            header += f"Tags: {tags}\n"
        header += f"Source: {original_name}\n"
        header += f"Processed: {datetime.now().isoformat()}\n"
        header += f"Type: {file_type}\n\n"
        txt_path.write_text(header + result["text_extracted"])
        
        # Upload to DeepTutor
        dt_result = await upload_to_deeptutor(txt_path, f"{Path(original_name).stem}.txt")
        result["deep_tutor_uploaded"] = dt_result.get("status") in (200, 201)
        result["deep_tutor_response"] = dt_result
        
    except Exception as e:
        result["error"] = str(e)
    
    return result


@app.post("/ingest/batch")
async def ingest_batch(
    files: list[UploadFile] = File(...),
    tags: str = Form(""),
):
    """Ingest multiple files at once with shared tags."""
    results = []
    for file in files:
        result = await ingest_file(file=file, tags=tags)
        results.append(result)
    return {"files_processed": len(results), "results": results}


@app.get("/sessions")
async def list_sessions():
    """List processed files."""
    files = []
    for f in sorted((INGEST_DIR / "processed").glob("*.txt"), reverse=True):
        files.append({
            "name": f.name,
            "size": f.stat().st_size,
            "modified": datetime.fromtimestamp(f.stat().st_mtime).isoformat()
        })
    return {"files": files[:50]}


# ─── Serve the UI ───────────────────────────────────────────────────────────

HTML_PATH = Path("/root/capture-index.html")
if HTML_PATH.exists():
    @app.get("/")
    async def serve_ui():
        return FileResponse(str(HTML_PATH))


if __name__ == "__main__":
    import asyncio
    
    # Try to ensure DeepTutor KB exists (non-fatal if DeepTutor is unreachable)
    try:
        print("Ensuring DeepTutor knowledge base...")
        asyncio.run(ensure_knowledge_base())
        print("Knowledge base ready.")
    except Exception as e:
        print(f"Warning: Could not connect to DeepTutor: {e}")
        print("Second Brain will still work — just upload to DeepTutor later.")
    
    print("Loading faster-whisper for audio processing...")
    from faster_whisper import WhisperModel
    WhisperModel("small", device="cpu", compute_type="int8")
    print("Ready. Starting Second Brain on port 8899...")
    
    uvicorn.run(app, host="0.0.0.0", port=8899)