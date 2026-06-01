-- ============================================================
-- Sasin EMBA Course Assistant — Supabase Migration
-- Run this in Supabase SQL Editor after creating project
-- ============================================================

-- 1. STORAGE BUCKETS
-- ============================================================
-- Note: Buckets must be created via Supabase Dashboard → Storage
-- or via the Supabase Management API. This SQL sets up RLS policies.

-- ── Lectures bucket (audio recordings, transcripts) ──
-- Create via dashboard: bucket name = 'lectures', public = false
-- Then apply these policies:

-- Allow authenticated users to upload their own lecture recordings
CREATE POLICY "Users can upload lecture recordings"
ON storage.objects FOR INSERT
TO authenticated
WITH CHECK (
  bucket_id = 'lectures' 
  AND (storage.foldername(name))[1] = auth.uid()::text
);

-- Allow users to read all lectures (shared course materials)
CREATE POLICY "Users can read all lectures"
ON storage.objects FOR SELECT
TO authenticated
USING (bucket_id = 'lectures');

-- ── Materials bucket (PDFs, slides, handouts) ──
-- Create via dashboard: bucket name = 'materials', public = false

CREATE POLICY "Users can read course materials"
ON storage.objects FOR SELECT
TO authenticated
USING (bucket_id = 'materials');

CREATE POLICY "Admins can upload materials"
ON storage.objects FOR INSERT
TO authenticated
WITH CHECK (
  bucket_id = 'materials'
  AND EXISTS (
    SELECT 1 FROM profiles 
    WHERE id = auth.uid() AND is_admin = true
  )
);

-- ── Avatars bucket (profile pictures) ──
-- Create via dashboard: bucket name = 'avatars', public = true

CREATE POLICY "Anyone can view avatars"
ON storage.objects FOR SELECT
TO authenticated
USING (bucket_id = 'avatars');

CREATE POLICY "Users can upload own avatar"
ON storage.objects FOR INSERT
TO authenticated
WITH CHECK (
  bucket_id = 'avatars'
  AND (storage.foldername(name))[1] = auth.uid()::text
);

-- ============================================================
-- 2. DATABASE TABLES
-- ============================================================

-- ── Profiles (extends auth.users) ──
CREATE TABLE profiles (
  id UUID PRIMARY KEY REFERENCES auth.users(id) ON DELETE CASCADE,
  display_name TEXT NOT NULL,
  line_user_id TEXT UNIQUE,
  avatar_url TEXT,
  industry TEXT,
  role TEXT DEFAULT 'student' CHECK (role IN ('student', 'admin', 'professor')),
  is_admin BOOLEAN DEFAULT false,
  created_at TIMESTAMPTZ DEFAULT now(),
  updated_at TIMESTAMPTZ DEFAULT now()
);

-- Auto-create profile on signup
CREATE OR REPLACE FUNCTION handle_new_user()
RETURNS TRIGGER AS $$
BEGIN
  INSERT INTO profiles (id, display_name, role, is_admin)
  VALUES (
    NEW.id,
    COALESCE(NEW.raw_user_meta_data->>'full_name', NEW.raw_user_meta_data->>'name', 'EMBA Student'),
    'student',
    false
  );
  RETURN NEW;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

CREATE TRIGGER on_auth_user_created
  AFTER INSERT ON auth.users
  FOR EACH ROW EXECUTE FUNCTION handle_new_user();

-- ── Subjects / Courses ──
CREATE TABLE subjects (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  name TEXT NOT NULL,
  code TEXT,
  instructor TEXT,
  term TEXT DEFAULT '2026',
  created_at TIMESTAMPTZ DEFAULT now(),
  created_by UUID REFERENCES profiles(id)
);

-- ── Lectures (recordings + transcripts) ──
CREATE TABLE lectures (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  subject_id UUID REFERENCES subjects(id) ON DELETE CASCADE,
  title TEXT NOT NULL,
  lecture_date DATE,
  audio_url TEXT,          -- Supabase Storage URL
  transcript_text TEXT,    -- Full Whisper transcript
  summary_text TEXT,       -- AI-generated summary
  key_points JSONB,        -- [{"concept": "...", "definition": "..."}]
  duration_seconds INTEGER,
  status TEXT DEFAULT 'pending' CHECK (status IN ('pending', 'transcribing', 'summarizing', 'complete')),
  created_at TIMESTAMPTZ DEFAULT now()
);

-- ── Course Materials (PDFs, slides, handouts) ──
CREATE TABLE materials (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  subject_id UUID REFERENCES subjects(id) ON DELETE CASCADE,
  lecture_id UUID REFERENCES lectures(id) ON DELETE SET NULL,
  title TEXT NOT NULL,
  file_url TEXT NOT NULL,
  file_type TEXT CHECK (file_type IN ('pdf', 'pptx', 'docx', 'image', 'other')),
  file_size_bytes INTEGER,
  uploaded_by UUID REFERENCES profiles(id),
  uploaded_at TIMESTAMPTZ DEFAULT now()
);

-- ── Student Notes ──
CREATE TABLE notes (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID REFERENCES profiles(id) ON DELETE CASCADE,
  lecture_id UUID REFERENCES lectures(id) ON DELETE CASCADE,
  content TEXT,
  created_at TIMESTAMPTZ DEFAULT now(),
  updated_at TIMESTAMPTZ DEFAULT now()
);

-- ── Quizzes ──
CREATE TABLE quizzes (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  subject_id UUID REFERENCES subjects(id) ON DELETE CASCADE,
  lecture_id UUID REFERENCES lectures(id) ON DELETE SET NULL,
  title TEXT NOT NULL,
  questions JSONB NOT NULL,  -- [{"q": "...", "options": [...], "correct": 0, "explanation": "..."}]
  difficulty TEXT DEFAULT 'medium' CHECK (difficulty IN ('easy', 'medium', 'hard')),
  created_at TIMESTAMPTZ DEFAULT now()
);

-- ── Quiz Attempts ──
CREATE TABLE quiz_attempts (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID REFERENCES profiles(id) ON DELETE CASCADE,
  quiz_id UUID REFERENCES quizzes(id) ON DELETE CASCADE,
  answers JSONB,
  score DECIMAL(5,2),
  completed_at TIMESTAMPTZ DEFAULT now(),
  UNIQUE(user_id, quiz_id)  -- One attempt per quiz per student (or remove for retakes)
);

-- ── Flashcards ──
CREATE TABLE flashcards (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  subject_id UUID REFERENCES subjects(id) ON DELETE CASCADE,
  lecture_id UUID REFERENCES lectures(id) ON DELETE SET NULL,
  front TEXT NOT NULL,
  back TEXT NOT NULL,
  source TEXT,
  created_at TIMESTAMPTZ DEFAULT now()
);

-- ── Spaced Repetition Progress ──
CREATE TABLE flashcard_progress (
  user_id UUID REFERENCES profiles(id) ON DELETE CASCADE,
  card_id UUID REFERENCES flashcards(id) ON DELETE CASCADE,
  ease_factor DECIMAL(4,2) DEFAULT 2.5,
  interval_days INTEGER DEFAULT 1,
  next_review DATE DEFAULT CURRENT_DATE,
  last_reviewed TIMESTAMPTZ,
  PRIMARY KEY (user_id, card_id)
);

-- ============================================================
-- 3. ROW-LEVEL SECURITY
-- ============================================================

-- Profiles: users can read all profiles, edit only their own
ALTER TABLE profiles ENABLE ROW LEVEL SECURITY;
CREATE POLICY "Profiles are viewable by authenticated users"
  ON profiles FOR SELECT TO authenticated USING (true);
CREATE POLICY "Users can update own profile"
  ON profiles FOR UPDATE TO authenticated USING (id = auth.uid());

-- Lectures: all authenticated users can read
ALTER TABLE lectures ENABLE ROW LEVEL SECURITY;
CREATE POLICY "Lectures are viewable by cohort"
  ON lectures FOR SELECT TO authenticated USING (true);

-- Notes: users can only see their own
ALTER TABLE notes ENABLE ROW LEVEL SECURITY;
CREATE POLICY "Users can CRUD own notes"
  ON notes FOR ALL TO authenticated USING (user_id = auth.uid());

-- Quiz attempts: users see only their own
ALTER TABLE quiz_attempts ENABLE ROW LEVEL SECURITY;
CREATE POLICY "Users see own attempts"
  ON quiz_attempts FOR SELECT TO authenticated USING (user_id = auth.uid());
CREATE POLICY "Users can insert own attempts"
  ON quiz_attempts FOR INSERT TO authenticated WITH CHECK (user_id = auth.uid());

-- Flashcard progress: per-user
ALTER TABLE flashcard_progress ENABLE ROW LEVEL SECURITY;
CREATE POLICY "Users manage own flashcard progress"
  ON flashcard_progress FOR ALL TO authenticated USING (user_id = auth.uid());

-- ============================================================
-- 4. INDEXES (for performance)
-- ============================================================
CREATE INDEX idx_lectures_subject ON lectures(subject_id);
CREATE INDEX idx_lectures_date ON lectures(lecture_date);
CREATE INDEX idx_materials_subject ON materials(subject_id);
CREATE INDEX idx_notes_user_lecture ON notes(user_id, lecture_id);
CREATE INDEX idx_quiz_attempts_user ON quiz_attempts(user_id);
CREATE INDEX idx_flashcard_progress_user_next ON flashcard_progress(user_id, next_review);

-- ============================================================
-- 5. SEED DATA — initial subjects
-- ============================================================
INSERT INTO subjects (name, code, instructor) VALUES
  ('AgriTech: Challenges and Opportunities in the Region', 'EMBA-601', 'Prof. Harry Jay Cavite'),
  ('AI, ESG & Future Skills', 'EMBA-602', 'TBD'),
  ('Leadership & Strategy', 'EMBA-603', 'TBD'),
  ('Finance for Executives', 'EMBA-604', 'TBD'),
  ('Digital Transformation', 'EMBA-605', 'TBD');
