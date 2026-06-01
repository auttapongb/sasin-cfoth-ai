"""
Sasin Live Assistant — Pipecat Bot
Real-time lecture companion: listens → transcribes → RAG → answers
WebSocket server: clients stream audio, receive answers

Latency targets:
- STT (Deepgram): ~300ms
- LLM (Groq): ~200-500ms  
- Total: ~1-2 seconds from question to answer
"""
import asyncio
import json
import os
import sys
from pathlib import Path

# ─── Config (set via env vars or change defaults) ──────────────────────────

# LLM: primary = Groq (fastest), fallback = OpenRouter
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openrouter")  # groq | openrouter
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
OPENROUTER_KEY = os.getenv("OPENROUTER_KEY", "sk-or-v1-093cb90db13c72b4ebad936adec7e07923cef1f92be366b30397ea2cc7017e8")

# Model selection
GROQ_MODEL = "llama-4-maverick-17b-128e"  # Groq's fastest thinking model
OPENROUTER_MODEL = "deepseek/deepseek-v4-pro"

# STT
STT_PROVIDER = os.getenv("STT_PROVIDER", "deepgram")  # deepgram | groq
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY", "")

# DeepTutor KB for RAG
DEEPTUTOR_KB = "emba-2026"
DEEPTUTOR_URL = "http://localhost:8001"

# System prompt
SYSTEM_PROMPT = """You are a Sasin EMBA teaching assistant, live in the classroom.
You hear the professor's lecture and student discussions in real-time.

Your job:
1. When the professor asks a question to the class, provide a concise answer with relevant frameworks
2. When students discuss, supplement with data points and counterarguments
3. When frameworks are mentioned, explain them briefly with examples
4. Always cite sources if using content from course materials

Keep answers SHORT (2-4 sentences). You're in a live classroom - be fast, not verbose.
Speak in a direct, professional tone. No greetings, no fluff."""

# ─── Imports ────────────────────────────────────────────────────────────────

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
import uvicorn
import httpx

app = FastAPI(title="Sasin Live Assistant")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ─── RAG Context Fetcher ───────────────────────────────────────────────────

async def get_rag_context(query: str, top_k: int = 3) -> str:
    """Fetch relevant context from DeepTutor knowledge base."""
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            # Query DeepTutor's RAG
            resp = await client.post(
                f"{DEEPTUTOR_URL}/api/v1/chat/rag",
                json={"kb_name": DEEPTUTOR_KB, "query": query, "top_k": top_k}
            )
            if resp.status_code == 200:
                chunks = resp.json().get("chunks", [])
                context = "\n\n".join(
                    f"[{c.get('source', 'lecture')}]: {c.get('text', '')}"
                    for c in chunks[:top_k]
                )
                return context
    except:
        pass
    return ""


# ─── LLM Call ──────────────────────────────────────────────────────────────

async def call_llm(messages: list, stream: bool = False) -> str:
    """Call LLM with fallback."""
    context = await get_rag_context(messages[-1]["content"])
    if context:
        messages.insert(-1, {"role": "system", "content": f"Relevant course context:\n{context}"})
    
    if LLM_PROVIDER == "groq" and GROQ_API_KEY:
        return await _call_groq(messages, stream)
    else:
        return await _call_openrouter(messages, stream)


async def _call_groq(messages: list, stream: bool = False) -> str:
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"},
            json={"model": GROQ_MODEL, "messages": messages, "max_tokens": 300, "temperature": 0.3}
        )
        if resp.status_code == 200:
            return resp.json()["choices"][0]["message"]["content"]
        raise Exception(f"Groq error: {resp.status_code}")


async def _call_openrouter(messages: list, stream: bool = False) -> str:
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={"Authorization": f"Bearer {OPENROUTER_KEY}", "Content-Type": "application/json"},
            json={"model": OPENROUTER_MODEL, "messages": messages, "max_tokens": 300, "temperature": 0.3}
        )
        if resp.status_code == 200:
            return resp.json()["choices"][0]["message"]["content"]
        raise Exception(f"OpenRouter error: {resp.status_code} {resp.text[:200]}")


# ─── STT ────────────────────────────────────────────────────────────────────

async def transcribe_audio(audio_data: bytes, format: str = "webm") -> str:
    """Transcribe audio chunk. Primary: Deepgram streaming, Fallback: Groq Whisper."""
    if STT_PROVIDER == "deepgram" and DEEPGRAM_API_KEY:
        return await _deepgram_stt(audio_data, format)
    elif GROQ_API_KEY:
        return await _groq_stt(audio_data, format)
    else:
        return ""


async def _deepgram_stt(audio_data: bytes, format: str = "webm") -> str:
    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.post(
            "https://api.deepgram.com/v1/listen?model=nova-3&smart_format=true&language=en",
            headers={"Authorization": f"Token {DEEPGRAM_API_KEY}", "Content-Type": f"audio/{format}"},
            content=audio_data
        )
        if resp.status_code == 200:
            result = resp.json()
            return result["results"]["channels"][0]["alternatives"][0].get("transcript", "")
        return ""


async def _groq_stt(audio_data: bytes, format: str = "webm") -> str:
    """Use Groq's Whisper endpoint."""
    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.post(
            "https://api.groq.com/openai/v1/audio/transcriptions",
            headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
            files={"file": ("audio.webm", audio_data, f"audio/{format}")},
            data={"model": "whisper-large-v3"}
        )
        if resp.status_code == 200:
            return resp.json().get("text", "")
        return ""


# ─── Conversation Manager ──────────────────────────────────────────────────

class ConversationManager:
    """Manages streaming conversation state per session."""
    
    def __init__(self):
        self.transcript_buffer = []
        self.message_history = [{"role": "system", "content": SYSTEM_PROMPT}]
        self.last_question_time = 0
        self.min_interval = 3.0  # minimum seconds between LLM calls
    
    async def process_transcript(self, text: str, ws: WebSocket):
        """Called when new transcript text arrives."""
        if not text.strip():
            return
        
        self.transcript_buffer.append(text)
        
        # Detect if this looks like a question or complete thought
        text_lower = text.lower().strip()
        is_question = (
            "?" in text or 
            any(text_lower.startswith(w) for w in ["what", "how", "why", "can", "could", "would", "explain", "describe", "define", "discuss"])
        )
        
        # Also trigger if we've accumulated enough text
        full_text = " ".join(self.transcript_buffer[-3:])
        should_respond = is_question and len(full_text) > 30
        
        if should_respond and (asyncio.get_event_loop().time() - self.last_question_time > self.min_interval):
            self.last_question_time = asyncio.get_event_loop().time()
            query = full_text
            
            # Send "thinking" indicator
            await ws.send_text(json.dumps({"type": "status", "text": "thinking..."}))
            
            try:
                self.message_history.append({"role": "user", "content": query})
                answer = await call_llm(self.message_history[-5:])  # Last 5 messages for context
                self.message_history.append({"role": "assistant", "content": answer})
                
                # Trim history
                if len(self.message_history) > 20:
                    self.message_history = [self.message_history[0]] + self.message_history[-10:]
                
                await ws.send_text(json.dumps({
                    "type": "answer",
                    "text": answer,
                    "timestamp": asyncio.get_event_loop().time()
                }))
            except Exception as e:
                await ws.send_text(json.dumps({"type": "error", "text": str(e)[:100]}))
    
    def reset(self):
        self.transcript_buffer = []
        self.message_history = [{"role": "system", "content": SYSTEM_PROMPT}]


# ─── WebSocket Endpoint ────────────────────────────────────────────────────

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    conv = ConversationManager()
    
    # Send config info
    await ws.send_text(json.dumps({
        "type": "config",
        "llm_provider": LLM_PROVIDER,
        "stt_provider": STT_PROVIDER,
        "llm_ready": bool(GROQ_API_KEY or OPENROUTER_KEY),
        "stt_ready": bool(DEEPGRAM_API_KEY or GROQ_API_KEY)
    }))
    
    try:
        while True:
            data = await ws.receive()
            
            if "text" in data:
                msg = json.loads(data["text"])
                msg_type = msg.get("type", "")
                
                if msg_type == "transcript":
                    # Browser did STT locally, sent text
                    await conv.process_transcript(msg["text"], ws)
                    
                elif msg_type == "audio":
                    # Raw audio chunk - needs STT
                    import base64
                    audio_bytes = base64.b64decode(msg["data"])
                    text = await transcribe_audio(audio_bytes, msg.get("format", "webm"))
                    if text:
                        # Send transcript back
                        await ws.send_text(json.dumps({"type": "transcript", "text": text}))
                        await conv.process_transcript(text, ws)
                
                elif msg_type == "reset":
                    conv.reset()
                    await ws.send_text(json.dumps({"type": "status", "text": "reset"}))
                    
            elif "bytes" in data:
                # Binary audio data
                text = await transcribe_audio(data["bytes"], "webm")
                if text:
                    await ws.send_text(json.dumps({"type": "transcript", "text": text}))
                    await conv.process_transcript(text, ws)
                    
    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"WS error: {e}")


# ─── Health + Client Page ──────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "llm_provider": LLM_PROVIDER,
        "stt_provider": STT_PROVIDER,
        "llm_ready": bool(GROQ_API_KEY or OPENROUTER_KEY),
        "stt_ready": bool(DEEPGRAM_API_KEY or GROQ_API_KEY),
    }

CLIENT_HTML = Path("/root/live-assistant.html")
if CLIENT_HTML.exists():
    @app.get("/")
    async def serve_client():
        return HTMLResponse(CLIENT_HTML.read_text())


# ─── Main ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"🎙️ Sasin Live Assistant")
    print(f"   LLM: {LLM_PROVIDER} ({'✅' if GROQ_API_KEY or OPENROUTER_KEY else '❌ no key'})")
    print(f"   STT: {STT_PROVIDER} ({'✅' if DEEPGRAM_API_KEY or GROQ_API_KEY else '❌ no key'})")
    print(f"   KB:  {DEEPTUTOR_KB}")
    print()
    print("Starting on port 9000...")
    uvicorn.run(app, host="0.0.0.0", port=9000)