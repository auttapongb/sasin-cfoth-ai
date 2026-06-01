# Sasin EMBA Course Assistant — Architecture Plan

## Overview

A multi-layered AI assistant system for Sasin EMBA students, deployed at `learn.cfoth.ai` (or `sasin.cfoth.ai/learn`). Built on **DeepTutor** as the core platform, extended with custom EMBA-specific modules.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                  learn.cfoth.ai                          │
│  (Next.js frontend — similar to MERN example UI)         │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌─────────┐ │
│  │  MODULE  │  │  MODULE  │  │  MODULE  │  │  EXAM   │ │
│  │    1     │  │    2     │  │    3     │  │  PREP   │ │
│  │ Recorder │  │  Realtime│  │  Study   │  │  Agent  │ │
│  │  + Sum.  │  │ Listener │  │  Tools   │  │         │ │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬────┘ │
│       │              │             │              │     │
│  ┌────┴──────────────┴─────────────┴──────────────┴───┐ │
│  │           COMMON BACKEND (FastAPI/Python)           │ │
│  │  • RAG Engine (LlamaIndex)                          │ │
│  │  • STT Pipeline (Whisper Large-v3 / Deepgram)       │ │
│  │  • LLM Orchestrator (OpenRouter + local Ollama)     │ │
│  │  • Document Processor (PDF, DOCX, Images, Audio)     │ │
│  │  • Vector DB (ChromaDB / Qdrant)                    │ │
│  │  • Auth (Supabase / LINE Login)                     │ │
│  └─────────────────────────────────────────────────────┘ │
│                                                          │
│  ┌─────────────────────────────────────────────────────┐ │
│  │              DEEPTUTOR (as base platform)           │ │
│  │  • RAG with citations                               │ │
│  │  • Quiz generator                                   │ │
│  │  • Book compiler ("living book" per subject)         │ │
│  │  • 3-layer memory (session → course → cross-course) │ │
│  │  • Agentic Deep Research / Solve                    │ │
│  └─────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────┘
```

---

## Module 1: Lecture Recorder & Summarizer

**What it does:** Record lectures (audio/video), auto-transcribe, summarize, extract key points.

### Tech Stack

| Component | Choice | Why |
|-----------|--------|-----|
| **Transcription** | Whisper Large-v3 (local) + Deepgram Nova-3 (cloud fallback) | Whisper: free, self-hosted, 57+ languages. Deepgram: 5.26% WER batch, sub-300ms streaming |
| **Summarization** | DeepSeek-V4 / Claude (via OpenRouter) + local Mistral Nemo 12B (Ollama) | Cloud for quality, local for privacy/offline |
| **Key Point Extraction** | LLM structured output (JSON) | Extract: key concepts, definitions, frameworks, professor's emphasis |
| **Speaker Diarization** | PyAnnote (local) or Deepgram built-in | Identify professor vs student questions |

### User Flow
1. Student hits "Record" on phone/laptop during class
2. Audio streams to backend → real-time transcription chunks appear
3. After class → full transcript + AI summary generated automatically
4. Key points extracted into structured notes:
   - **Core concepts** defined
   - **Frameworks** mentioned (SWOT, PESTLE, Porter's Five Forces, etc.)
   - **Professor's emphasis** (things repeated or stressed)
   - **Questions asked** by students (potential exam topics)
5. Notes sync to subject-specific knowledge base

### Image & PDF Upload
- Phone photos of whiteboard/slides → OCR (Tesseract / GPT-4V)
- PDF handouts → parsed, chunked, added to RAG index
- All materials organized by subject → `AgriTech/`, `Finance/`, `Strategy/`, etc.

---

## Module 2: Realtime Lecture Assistant

**What it does:** Live transcription + real-time context analysis + Q&A prediction during class.

### Tech Stack

| Component | Choice | Why |
|-----------|--------|-----|
| **Real-time STT** | Deepgram Streaming (WebSocket) | Sub-300ms latency, best for live transcription |
| **Context Engine** | RAG over pre-loaded course materials | Compares live lecture against syllabus, readings, past lectures |
| **Q&A Predictor** | LLM with course context | Predicts likely exam questions based on emphasis patterns |

### User Flow
1. Before class: upload syllabus, readings, slides (if available)
2. During class:
   - Live transcript scrolls on screen
   - Sidebar shows **"Related concepts from readings"** in real-time
   - Professor mentions a term → related definition pops up
   - System detects emphasis patterns → flags "⚠️ Likely exam topic"
3. After class: generates **"Predicted Exam Questions"** based on:
   - Topics professor spent most time on
   - Questions students asked
   - Connections to syllabus learning objectives
   - Past exam patterns (if available)

### The "Prediction Engine"
- Tracks time spent per topic
- Detects verbal cues: "this is important", "remember this", "on the exam"
- Cross-references with syllabus LOs
- Generates 5-10 predicted questions per lecture
- Builds cumulative exam prep bank across the term

---

## Module 3: AI Study Tools

### 3a. RAG Chat ("Ask Your Course")
- Upload all course materials (PDFs, slides, notes, transcripts)
- Chat interface: "Explain Porter's Five Forces as Professor Cavite taught it"
- Citations back to specific lectures/slides
- Powered by DeepTutor's RAG engine or custom LlamaIndex pipeline

### 3b. Auto-Flashcards
- Generated from lecture transcripts + materials
- Spaced repetition (Anki-compatible export or built-in)
- AI identifies key term → definition → example from lecture
- Format: `{"front": "What is...", "back": "...", "source": "Lecture 3, May 31"}`

### 3c. Quiz Generator
- Multiple choice, short answer, case-based
- Difficulty levels: Recall → Application → Analysis
- Timed quiz mode simulating exam conditions
- GPT-4/DeepSeek generated, verified against course materials

### 3d. "Explain Like I'm 5" Mode
- Take complex frameworks → simplify
- Generate analogies relevant to student's industry
- "Explain CAPM using a restaurant analogy" (based on student's industry)

### 3e. Study Progress Dashboard
- Track: hours studied, topics covered, quiz scores
- Identify weak areas (topics with low quiz scores)
- Recommend focus areas for exam prep
- Streak tracking + study reminders

### 3f. "Book Compiler" (DeepTutor Feature)
- Compiles all materials per subject into a "living book"
- Searchable, chapter-organized
- Auto-updates as new lectures are added

---

## Tech Stack Decision

### Frontend: Next.js 14 (App Router) + Tailwind CSS
- Same pattern as MERN example shown
- Deploy to Vercel or static export to GitHub Pages
- LINE Login integration for Sasin cohort
- Sasin design system (Playfair Display + Inter, gold accents)

### Backend: FastAPI (Python)
- RAG engine: **LlamaIndex** (better for academic RAG than LangChain)
- Vector DB: **ChromaDB** (lightweight, local) or **Qdrant** (cloud, if needed)
- STT: **Whisper** (local) + **Deepgram** (cloud for real-time)
- LLM: **OpenRouter** (primary, same as current setup) + **Ollama** (local fallback with Mistral Nemo 12B or Llama 3.1 8B)

### Database: Supabase (PostgreSQL)
- Free tier sufficient for single-cohort use
- Built-in auth (LINE, Google)
- Row-level security for privacy
- Real-time subscriptions for live transcription sync

### Deployment

| Environment | Where | Cost |
|-------------|-------|------|
| Frontend | GitHub Pages (static) or Vercel | $0 |
| Backend API | Contabo VPS (existing) or Railway | $0-5/mo |
| STT (Whisper) | Local on VPS GPU, or Modal serverless GPU | $0-0.50/hr |
| STT (Deepgram) | Cloud API | $0.0043/min batch |
| LLM | OpenRouter (existing credits) | Variable |
| Vector DB | ChromaDB (local) | $0 |
| Auth | Supabase | $0 (free tier) |

---

## Implementation Phases

### Phase 1: Foundation (Week 1-2)
- [x] Create repo `sasin-course-assistant`
- [x] Deploy landing page at `sasin.cfoth.ai/learn`
- [ ] Set up Next.js project with Sasin design system
- [ ] Supabase auth (LINE + Google)
- [ ] File upload (PDF, images, audio)

### Phase 2: Core RAG (Week 3-4)
- [ ] Document processing pipeline (PDF → text → chunks → embeddings)
- [ ] ChromaDB setup + indexing
- [ ] RAG chat interface ("Ask Your Course")
- [ ] Source citations in responses

### Phase 3: Lecture Pipeline (Week 5-6)
- [ ] Audio upload → Whisper transcription
- [ ] AI summarization
- [ ] Key point extraction
- [ ] Auto-organization by subject

### Phase 4: Study Tools (Week 7-8)
- [ ] Flashcard generator
- [ ] Quiz generator
- [ ] Progress dashboard
- [ ] "Book compiler" per subject

### Phase 5: Realtime Mode (Week 9-10)
- [ ] Deepgram streaming integration
- [ ] Live transcription display
- [ ] Real-time context sidebar
- [ ] Exam question predictor

### Phase 6: Polish (Week 11-12)
- [ ] Mobile PWA
- [ ] Offline mode (local STT + LLM)
- [ ] Export to Anki/Notion
- [ ] Study groups feature

---

## Alternative: DeepTutor as Full Platform

**DeepTutor already provides 80% of what we need:**
- ✅ RAG with citations
- ✅ Quiz generator
- ✅ Book compiler ("living book")
- ✅ Document attachments (PDF/DOCX/XLSX/PPTX)
- ✅ 3-layer memory
- ✅ Multi-user with isolated workspaces
- ✅ Local LLM support (Ollama, LM Studio, llama.cpp)
- ✅ Multi-provider (OpenAI, Anthropic, Gemini, DeepSeek, etc.)
- ✅ Self-hostable via Docker

**What we'd add on top of DeepTutor:**
1. Audio recording + transcription pipeline
2. Real-time lecture streaming mode
3. Flashcard generation + spaced repetition
4. Exam prediction engine
5. LINE Login integration
6. Sasin UI theming

**Recommendation:** Fork DeepTutor → add EMBA-specific modules → brand as Sasin Course Assistant.

---

## References

1. **STT APIs 2026**: [FutureAGI comparison](https://futureagi.com/blog/speech-to-text-apis-in-2026-benchmarks-pricing-developer-s-decision-guide/) — Deepgram 5.26% WER, ElevenLabs 150ms real-time, Whisper free self-host
2. **DeepTutor**: [HKUDS/DeepTutor](https://github.com/HKUDS/DeepTutor) — Agent-native tutoring platform with RAG, quizzes, book compiler
3. **Lecture Summarizer**: [deep-div/Lecture-Summarize](https://github.com/deep-div/Lecture-Summarize) — Whisper + Gemini pipeline
4. **Local STT + RAG**: [Alibaba guide](https://www.alibaba.com/product-insights/how-to-run-private-offline-ai-video-summarization-for-lecture-recordings-using-whisper-llama-index.html) — Whisper + LlamaIndex offline pipeline
5. **AI Notetaker Architecture**: [Gladia guide](https://www.gladia.io/blog/how-to-build-an-ai-note-taker-complete-architecture-guide-with-async-transcription-and-llm-integration)
6. **Local AI Study Assistant**: [Ollama + Open WebUI](https://openwebui.com) — Self-hosted ChatGPT-like interface with RAG
7. **Offline Transcription**: [Whisper + Phi + FastAPI](https://askaresh.com/2025/01/09/offline-transcribing-and-summarizing-audio-with-whisper-phi-fastapi-docker-on-nvidia-gpu/)
