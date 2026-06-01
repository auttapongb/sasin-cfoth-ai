# Permanent Infrastructure Plan — Sasin Course Assistant

## Decision: Supabase (Free Tier) + Contabo VPS (Existing)

### Why this stack:

| Need | Solution | Cost |
|------|----------|------|
| **Database** (users, transcripts, notes, quiz results) | Supabase PostgreSQL | $0 (500MB) |
| **Auth** (LINE + Google login, user profiles) | Supabase Auth | $0 (50K MAU) |
| **File Storage** (audio recordings, PDFs, images) | Supabase Storage | $0 (1GB) |
| **Backend API** (STT, LLM, RAG, quiz gen) | Contabo VPS (existing) | $0 (already paid) |
| **Vector DB** (embeddings for RAG) | ChromaDB on VPS | $0 (local) |
| **Frontend** | GitHub Pages or Vercel | $0 |
| **Real-time** (live transcription sync) | Supabase Realtime | $0 (included) |

### Why Supabase is the right call:

1. **All-in-one**: Auth + DB + Storage + Realtime in one service. No stitching together separate services.
2. **Free tier is generous**: 500MB DB, 1GB storage, 50K monthly users, 2GB bandwidth — more than enough for 43 students.
3. **LINE Login support**: Supabase Auth supports LINE Login natively (OAuth provider).
4. **Row-Level Security**: Each student's data is private by default. Simple SQL policies.
5. **PostgreSQL**: Full relational DB. Not a limited NoSQL store.
6. **Migration path**: If we outgrow free tier, $25/mo for Pro. Or export to any PostgreSQL host.

### Database Schema (draft):

```sql
-- Users (extends Supabase auth.users)
CREATE TABLE profiles (
  id UUID PRIMARY KEY REFERENCES auth.users(id),
  line_display_name TEXT,
  avatar_url TEXT,
  industry TEXT,
  created_at TIMESTAMPTZ DEFAULT now()
);

-- Subjects/Courses
CREATE TABLE subjects (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  name TEXT NOT NULL,           -- 'AgriTech', 'Finance', 'Strategy'
  code TEXT,                    -- 'EMBA-601'
  instructor TEXT,              -- 'Prof. Harry Jay Cavite'
  created_by UUID REFERENCES profiles(id)
);

-- Lecture recordings + transcripts
CREATE TABLE lectures (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  subject_id UUID REFERENCES subjects(id),
  title TEXT NOT NULL,
  lecture_date DATE,
  audio_url TEXT,               -- Supabase Storage URL
  transcript_text TEXT,         -- Full Whisper transcript
  summary_text TEXT,            -- AI-generated summary
  key_points JSONB,             -- [{"concept": "...", "definition": "..."}]
  duration_seconds INTEGER,
  created_at TIMESTAMPTZ DEFAULT now()
);

-- Student notes on lectures
CREATE TABLE notes (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID REFERENCES profiles(id),
  lecture_id UUID REFERENCES lectures(id),
  content TEXT,
  created_at TIMESTAMPTZ DEFAULT now()
);

-- Quizzes
CREATE TABLE quizzes (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  subject_id UUID REFERENCES subjects(id),
  lecture_id UUID REFERENCES lectures(id),
  questions JSONB,              -- Array of {question, options, correct, explanation}
  created_at TIMESTAMPTZ DEFAULT now()
);

-- Quiz attempts
CREATE TABLE quiz_attempts (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID REFERENCES profiles(id),
  quiz_id UUID REFERENCES quizzes(id),
  answers JSONB,                -- User's answers
  score DECIMAL(5,2),
  completed_at TIMESTAMPTZ DEFAULT now()
);

-- Flashcards
CREATE TABLE flashcards (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  subject_id UUID REFERENCES subjects(id),
  lecture_id UUID REFERENCES lectures(id),
  front TEXT NOT NULL,
  back TEXT NOT NULL,
  source TEXT,                  -- 'Lecture 3, AgriTech'
  created_at TIMESTAMPTZ DEFAULT now()
);

-- Spaced repetition progress per user
CREATE TABLE flashcard_progress (
  user_id UUID REFERENCES profiles(id),
  card_id UUID REFERENCES flashcards(id),
  ease_factor DECIMAL(4,2) DEFAULT 2.5,
  interval_days INTEGER DEFAULT 1,
  next_review DATE,
  last_reviewed TIMESTAMPTZ,
  PRIMARY KEY (user_id, card_id)
);

-- Course materials (PDFs, slides, etc.)
CREATE TABLE materials (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  subject_id UUID REFERENCES subjects(id),
  title TEXT NOT NULL,
  file_url TEXT,                -- Supabase Storage URL
  file_type TEXT,               -- 'pdf', 'pptx', 'docx', 'image'
  uploaded_by UUID REFERENCES profiles(id),
  uploaded_at TIMESTAMPTZ DEFAULT now()
);
```

### Architecture Diagram:

```
┌─────────────────────────────────────────────┐
│                 BROWSER                      │
│  sasin.cfoth.ai/learn  (Next.js / static)    │
├─────────────────────────────────────────────┤
│           Supabase Client (JS SDK)           │
│     Auth │ Database │ Storage │ Realtime     │
└──────┬───────┬──────────┬──────────┬────────┘
       │       │          │          │
       ▼       ▼          ▼          ▼
┌──────────────────────────────────────────────┐
│              SUPABASE (Cloud)                │
│  • Auth (LINE + Google OAuth)                │
│  • PostgreSQL (users, lectures, quizzes...)  │
│  • Storage (audio, PDFs, images)             │
│  • Realtime (live transcript streaming)      │
│  Cost: $0/mo (free tier)                     │
└──────────────────────┬───────────────────────┘
                       │ API calls from VPS
                       ▼
┌──────────────────────────────────────────────┐
│          CONTABO VPS (Existing)              │
│  ┌─────────────────────────────────────────┐ │
│  │  FastAPI Backend (port 8000)             │ │
│  │  • /transcribe  (Whisper STT)            │ │
│  │  • /summarize   (DeepSeek-V4)            │ │
│  │  • /generate-quiz                         │ │
│  │  • /generate-flashcards                   │ │
│  │  • /predict-exam-questions                │ │
│  │  • /rag-query   (LlamaIndex + ChromaDB)  │ │
│  └─────────────────────────────────────────┘ │
│  ┌─────────────────────────────────────────┐ │
│  │  ChromaDB (port 8001)                    │ │
│  │  • Document embeddings                    │ │
│  │  • Vector search for RAG                  │ │
│  └─────────────────────────────────────────┘ │
│  ┌─────────────────────────────────────────┐ │
│  │  DeepTutor (Docker, port 3000)           │ │
│  │  • RAG engine                             │ │
│  │  • Quiz generator                         │ │
│  │  • Book compiler                          │ │
│  │  • Agentic tutoring                       │ │
│  └─────────────────────────────────────────┘ │
└──────────────────────────────────────────────┘
```

### Migration Path from Current Setup:

| Current | → | New |
|---------|---|-----|
| localStorage for profiles | → | Supabase `profiles` table |
| localStorage for roster data | → | Supabase `profiles` + `subjects` tables |
| GitHub upload server for images | → | Supabase Storage buckets |
| Static HTML/CSS/JS | → | Next.js 14 + Supabase JS client |
| LINE LIFF (stateless) | → | Supabase Auth + LINE OAuth |

### Cost Summary:

| Service | Monthly Cost |
|---------|-------------|
| Supabase (Free Tier) | $0 |
| Contabo VPS (existing) | $0 (already paid) |
| OpenRouter LLM API | Variable (~$5-20/mo for cohort) |
| Deepgram (optional, real-time STT) | ~$0.26/hour of lecture |
| **Total** | **$0-20/mo** |

### What stays the same:
- GitHub Pages for static pages (landing page, roster, agritech brief)
- Contabo VPS for backend compute
- OpenRouter for LLM access
