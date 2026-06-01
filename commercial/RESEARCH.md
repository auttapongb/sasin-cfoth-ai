# Sasin AI Learning Toolkit — Feature Research

## Phase 1: Immediate (Weeks 1-4) — CRITICAL

### 1. Content Moderation (1 week, FREE)
- **Tool:** OpenAI Moderation API — FREE for API users
- **Setup:** Wrap Live Assistant + DeepTutor endpoints with pre-flight check
- **Complexity:** Low. <1 day integration.
- **Saas:** Use existing OpenAI key

### 2. Speaker Diarization (1-2 weeks, ~$40/mo)
- **Tool:** Deepgram Nova-3 + diarize=true — $0.0068/min total
- **Alternative:** AssemblyAI Universal-3 — $0.21/hr
- **Note:** Google STT diarization does NOT support Thai
- **Self-hosted:** WhisperX + PyAnnote (free, needs GPU ~$150/mo)
- **Complexity:** Low. Add `diarize=true&diarize_version=2` to existing Deepgram calls

### 3. Multi-Language Thai+English (3-4 weeks, ~$100-200/mo)
- **STT:** Deepgram Nova-3 supports Thai with 69.43% WER reduction vs Nova-2
- **Translation:** DeepL (~$25/mo per 1M chars) or Google Cloud Translation
- **LLM:** GPT-4o + Gemini 2.5 handle Thai well
- **⚠️ Whisper has Brahmic script bug — not recommended for production Thai**
- **DeepTutor:** i18n framework exists (Chinese done), needs Thai UI strings
- **Complexity:** Medium

## Phase 2: High Value (Weeks 5-8)

### 4. Quiz Generation (2-3 weeks, ~$10/mo LLM costs)
- **DeepTutor has quiz generation mode built in since v0.5.0**
- **Needs:** lecture-specific quiz trigger, QTI 2.1 LMS export, per-student analytics
- **Complexity:** Low-Medium (most infrastructure exists)

### 5. Accessibility (3-4 weeks, $0-100/mo)
- **Captions/Transcripts:** Already generating — need VTT/SRT export
- **Screen Reader:** WCAG 2.1 AA — axe-core (free), WAVE (free), Pa11y (free)
- **Complexity:** Medium

## Phase 3: Nice-to-Have (Weeks 9-16)

### 6. Mobile PWA (1-2 weeks, $0)
- **Tool:** next-pwa (free, open source)
- **DeepTutor:** Next.js 16 + React 19, no PWA config yet

### 7. Offline Mode (2-3 weeks, $0)
- **What works:** Cached lectures/transcripts/quizzes via Service Workers
- **What doesn't:** LLM queries, STT, RAG
- **Offline AI:** NOT practical in 2026

### 8. Study Groups (8-12 weeks, $100-200/mo)
- **Tool:** Liveblocks ($99/mo) for shared notebooks + presence
- **Complexity:** HIGH — CRDT/OT, permissions, notifications

---

## Strategic Recommendations

1. **Deepgram is the strategic vendor** — Thai + English + diarization in ONE API
2. **Content moderation is fastest win** — FREE, <1 day
3. **Thai + English is commercial moat** — few AI platforms support Thai well
4. **Quiz gen already in DeepTutor** — just needs lecture-specific trigger
5. **Offline AI not practical** — focus on cached content, not offline LLM
