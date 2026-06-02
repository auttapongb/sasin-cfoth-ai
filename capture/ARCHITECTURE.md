# Sasin EMBA Learning Toolkit — v5 Architecture
## 14-Feature Expansion Plan
### 2026-06-03

## Current State
- **Backend:** server.py (1177 lines), FastAPI, SQLite
- **Frontend:** index.html (~2600 lines), 3 tabs: Capture, Chat, Second Brain
- **Port:** 8898 (inside Docker), served at capture.sasin.cfoth.ai
- **APIs:** Groq STT, Deepgram fallback, Gemini Vision, DeepSeek chat/co-learner
- **KB:** DeepTutor at localhost:8001 → emba-2026

## Target Architecture

### New DB Tables (7)
1. `flashcards` — id, session_id, front, back, source, fsrs_state(JSON), next_review, created_at
2. `quizzes` — id, session_id, title, questions(JSON), created_at  
3. `quiz_attempts` — id, quiz_id, answers(JSON), score, created_at
4. `briefs` — id, session_id, topic, content(md), frameworks(JSON), created_at
5. `case_studies` — id, session_id, company, industry, step(1-5), analysis(md), frameworks(JSON), created_at
6. `audio_overviews` — id, session_id, title, script, audio_path, voices, created_at
7. `action_items` — id, session_id, text, speaker, assignee, deadline, source_entry_id, status, created_at

### New API Endpoints (35+)

**Briefs (4):**
- POST /briefs/generate {session_id, topic?} → brief
- GET /briefs → [{id, topic, created_at}]
- GET /briefs/{id} → {full brief}
- GET /briefs/{id}/download → .md file

**Flashcards / Spaced Repetition (5):**
- POST /flashcards/generate {session_id} → [{front, back, source}]
- GET /flashcards/due → [{cards due today}]
- POST /flashcards/{id}/review {rating: 1-4} → {next_review, state}
- GET /flashcards/stats → {total, due, mastered, streak}
- DELETE /flashcards/{id}

**Quizzes (4):**
- POST /quizzes/generate {session_id, type: mcq|mixed, count: 10} → quiz
- GET /quizzes → [{id, title, created_at}]
- GET /quizzes/{id} → {title, questions: [{q, choices?, answer, explanation}]}
- POST /quizzes/{id}/submit {answers: [{question_idx, answer}]} → {score, corrections}

**Audio Overviews (4):**
- POST /audio/generate {session_id, style: podcast|summary|lecture} → overview
- GET /audio → [{id, title, created_at, duration}]
- GET /audio/{id} → {script, audio_url}
- GET /audio/{id}/download.mp3 → audio file

**Case Studies (5):**
- POST /cases/analyze {session_id, company?, industry?} → analysis
- GET /cases → [{id, company, created_at}]
- GET /cases/{id} → {full analysis with steps}
- POST /cases/{id}/step {step_num, question?} → {step_result}
- GET /cases/{id}/export → .md download

**Templates (3):**
- GET /templates → [{id, name, structure}]
- POST /templates/apply {session_id, template_id} → {formatted}
- POST /templates {name, structure} → template

**Action Items (3):**
- GET /sessions/{id}/action-items → [{text, speaker, status}]
- POST /sessions/{id}/action-items/extract → [{text, speaker, deadline?}]
- PATCH /action-items/{id} {status} → updated

**Collaboration (3):**
- POST /sessions/{id}/share → {share_token, url}
- GET /shared/{token} → {session_data}
- DELETE /sessions/{id}/share → unshare

**Mind Maps (1):**
- GET /mindmap/{session_id} → {nodes, edges} JSON

**Web Research (1):**
- POST /research/search {query, session_id?} → {results with citations}

**Schedule Planner (2):**
- GET /schedule/due → [{cards, quizzes, reviews due}]
- GET /schedule/calendar?days=7 → [{date, items}]

**Presentation Generator (1):**
- POST /present/generate {session_id, style?} → {slides: [{title, bullets, notes}]}

### Frontend Tabs (8)
1. **📝 Capture** (existing) — live transcription, slide capture, co-learner
2. **💬 Chat** (enhanced) — with source citations, KB-aware
3. **🧠 2nd Brain** (existing) — knowledge graph
4. **📋 Briefs** (new) — executive briefs list + generator
5. **🃏 Flashcards** (new) — spaced repetition review + generator
6. **📝 Quizzes** (new) — quiz list + take quiz + results
7. **🎧 Audio** (new) — podcast overviews list + player
8. **📊 Cases** (new) — case study analyzer

### Implementation Order
Phase 1: DB extension + Briefs + Citations + Templates (foundation)
Phase 2: Flashcards + Quizzes + Audio Overviews (core learning)
Phase 3: Cases + Action Items + Collaboration (advanced)
Phase 4: Mind Maps + Presentation + Schedule + Web Research + Mobile CSS (polish)
