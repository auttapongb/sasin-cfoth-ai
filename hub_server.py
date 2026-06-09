#!/usr/bin/env python3
"""
Sasin EMBA Hub Server — Dynamic landing page for sasin.cfoth.ai
Serves the hub UI + proxies to capture (8898) and brain (8400) servers.
"""
import json, os, httpx
from pathlib import Path
from datetime import datetime, timezone
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

BASE = Path("/data/sasin-cfoth-ai")
HUB_HTML = BASE / "index.html"
CALENDAR_FILE = Path("/data/emba-second-brain/calendar_events.json")
BRAIN_DOCS = Path("/data/emba-second-brain/corpus.json")
BRAIN_STATE = Path("/data/emba-second-brain/drive_sync_state.json")

BRAIN = "http://localhost:8400"

app = FastAPI(title="Sasin Hub")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

_client = None

async def get_client():
    global _client
    if _client is None:
        _client = httpx.AsyncClient(timeout=httpx.Timeout(30.0))
    return _client

# ─── HUB PAGE ────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def hub():
    if not HUB_HTML.exists():
        return "<h1>Sasin EMBA 2026</h1><p>Hub not built yet.</p>"
    html = HUB_HTML.read_text()
    # Inject calendar data if not already embedded
    if 'id="calendarData"' not in html and CALENDAR_FILE.exists():
        cal = json.loads(CALENDAR_FILE.read_text())
        cal_json = json.dumps(cal)
        html = html.replace("<body>", f'<body>\n<script id="calendarData" type="application/json">{cal_json}</script>')
    # Inject course deadlines
    if 'id="coursesData"' not in html and COURSES_FILE.exists():
        courses = json.loads(COURSES_FILE.read_text())
        # Extract deadlines per course code
        deadlines = {}
        for code, c in courses.items():
            if c.get("deadlines"):
                deadlines[code] = c["deadlines"]
        courses_json = json.dumps(deadlines)
        html = html.replace("<body>", f'<body>\n<script id="coursesData" type="application/json">{courses_json}</script>')
    return html

# ─── CALENDAR (direct from JSON) ────────────────────────
@app.get("/calendar")
async def calendar():
    if CALENDAR_FILE.exists():
        return JSONResponse(json.loads(CALENDAR_FILE.read_text()))
    return {"events": [], "total": 0}

# ─── DOCUMENTS (direct from corpus) ─────────────────────
@app.get("/documents")
async def documents():
    drive = []
    if BRAIN_STATE.exists():
        state = json.loads(BRAIN_STATE.read_text())
        corpus = json.loads(BRAIN_DOCS.read_text()) if BRAIN_DOCS.exists() else []
        for folder_id, fs in state.get("folders", {}).items():
            for fid, info in fs.get("processed", {}).items():
                name = info.get("name", "")
                matched = [e for e in corpus if e.get("drive_id") == fid]
                folder = matched[0].get("source_folder", "") if matched else ""
                if not folder:
                    folder = "EMBA2026"
                drive.append({
                    "drive_id": fid, "name": name, "folder": folder,
                    "modified": info.get("modifiedTime", ""),
                    "synced_at": info.get("synced_at", ""),
                })
    brain_files = []
    if BRAIN_DOCS.exists():
        corpus = json.loads(BRAIN_DOCS.read_text())
        for entry in corpus:
            brain_files.append({
                "name": entry.get("name", ""), "title": entry.get("title", ""),
                "drive_id": entry.get("drive_id", ""),
                "source_folder": entry.get("source_folder", ""),
                "frameworks": entry.get("frameworks", []),
                "topics": entry.get("topics", []),
                "difficulty": entry.get("difficulty", ""),
                "summary": entry.get("summary", "")[:200],
                "size": entry.get("size", 0),
                "synced_at": entry.get("synced_at", ""),
                "modified": entry.get("modified", ""),
            })
    return {
        "drive_files": drive, "brain_files": brain_files,
        "drive_count": len(drive), "brain_count": len(brain_files),
        "last_sync": state.get("last_sync") if BRAIN_STATE.exists() else None,
    }

# ─── PROXY: Capture Server (8898) ───────────────────────

# ─── PROXY: Brain Server (8400) ─────────────────────────
@app.api_route("/brain/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
async def proxy_brain(path: str, request: Request):
    client = await get_client()
    url = f"{BRAIN}/{path}"
    body = await request.body() if request.method in ("POST", "PUT", "PATCH") else None
    headers = {k: v for k, v in request.headers.items() if k.lower() not in ("host", "content-length")}
    try:
        resp = await client.request(request.method, url, content=body, headers=headers, timeout=30.0)
        return Response(content=resp.content, status_code=resp.status_code,
                       headers=dict(resp.headers))
    except Exception as e:
        raise HTTPException(502, f"Brain server unreachable: {e}")

# ─── FLASHCARDS ──────────────────────────────────────────
@app.get("/api/flashcards")
async def flashcards():
    """Return flashcard-worthy content from the brain corpus."""
    if not BRAIN_DOCS.exists():
        return {"decks": [], "total": 0, "message": "No documents processed yet"}
    corpus = json.loads(BRAIN_DOCS.read_text())
    decks = []
    for entry in corpus:
        name = entry.get("name", "")
        content = entry.get("content", "")
        # Extract key points as potential flashcards
        sentences = [s.strip() for s in content.replace("\n", ". ").split(". ") if len(s.strip()) > 40]
        decks.append({
            "name": name,
            "size": entry.get("size", 0),
            "card_count": min(len(sentences), 20),
            "preview": content[:200] + "..." if len(content) > 200 else content,
        })
    return {"decks": decks, "total": len(decks)}

# ─── QUIZ ────────────────────────────────────────────────
@app.get("/api/quiz")
async def quiz():
    """Return quiz-worthy content from the brain corpus."""
    if not BRAIN_DOCS.exists():
        return {"quizzes": [], "total": 0, "message": "No documents processed yet"}
    corpus = json.loads(BRAIN_DOCS.read_text())
    quizzes = []
    for entry in corpus:
        name = entry.get("name", "")
        content = entry.get("content", "")
        # Count paragraphs as potential question sources
        paragraphs = [p.strip() for p in content.split("\n\n") if len(p.strip()) > 80]
        quizzes.append({
            "topic": name,
            "question_count": min(len(paragraphs), 10),
            "size_kb": round(entry.get("size", 0) / 1024, 1),
            "preview": content[:150] + "..." if len(content) > 150 else content,
        })
    return {"quizzes": quizzes, "total": len(quizzes)}

# ─── ROSTER ──────────────────────────────────────────────
@app.get("/api/roster")
async def roster():
    """Return class roster (synced from Google Sheets or static file)."""
    roster_file = BASE / "roster.json"
    if roster_file.exists():
        return json.loads(roster_file.read_text())
    return {
        "students": [],
        "total": 0,
        "message": "Roster not yet loaded. Place roster.json in sasin-cfoth-ai directory."
    }

# ─── COURSES ────────────────────────────────────────────
COURSES_FILE = BASE / "courses.json"

@app.get("/api/courses")
async def courses_api():
    """Return all course data with deadlines."""
    if not COURSES_FILE.exists():
        return {"courses": {}, "total": 0}
    courses = json.loads(COURSES_FILE.read_text())
    # Calculate days remaining for each deadline
    from datetime import date
    today = date.today()
    for code, course in courses.items():
        for dl in course.get("deadlines", []):
            dl_date = datetime.strptime(dl["date"], "%Y-%m-%d").date()
            dl["days_left"] = (dl_date - today).days
    return {"courses": courses, "total": len(courses)}


@app.api_route("/emba2026/{path:path}", methods=["GET"])
async def serve_emba2026(path: str, request: Request):
    """Serve EMBA 2026 roster static files."""
    base_dir = BASE / "emba2026"
    clean_path = path.rstrip("/")
    file_path = base_dir / clean_path
    if file_path.is_file():
        content_type = "text/html"
        if path.endswith(".png"): content_type = "image/png"
        elif path.endswith(".jpg") or path.endswith(".jpeg"): content_type = "image/jpeg"
        elif path.endswith(".svg"): content_type = "image/svg+xml"
        return Response(content=file_path.read_bytes(), media_type=content_type)
    index_path = file_path / "index.html"
    if index_path.exists():
        return HTMLResponse(index_path.read_text())
    raise HTTPException(404, f"EMBA2026 resource not found: {path}")

@app.get("/emba2026", response_class=HTMLResponse)
@app.get("/emba2026/", response_class=HTMLResponse)
async def serve_emba2026_index():
    index = BASE / "emba2026" / "index.html"
    if index.exists():
        return HTMLResponse(index.read_text())
    raise HTTPException(404, "EMBA2026 page not found")

@app.api_route("/courses/{path:path}", methods=["GET"])
async def serve_course(path: str, request: Request):
    """Serve static course content (knowledge pages, materials)."""
    # Resolve path under courses directory (case-insensitive)
    courses_dir = BASE / "courses"
    # Normalize: strip trailing slash for directory comparison
    clean_path = path.rstrip("/")
    file_path = courses_dir / clean_path
    if not file_path.exists():
        # Case-insensitive fallback
        path_lower = clean_path.lower()
        for entry in courses_dir.iterdir():
            if entry.name.lower() == path_lower:
                file_path = entry
                break
        else:
            # Try splitting path for nested lookups
            parts = path.split("/")
            current = courses_dir
            for i, part in enumerate(parts):
                found = False
                part_lower = part.lower()
                for entry in current.iterdir():
                    if entry.name.lower() == part_lower:
                        current = entry
                        found = True
                        break
                if not found:
                    raise HTTPException(404, f"Course resource not found: {path}")
            file_path = current
    if file_path.is_file():
        content_type = "text/html"
        if path.endswith(".pdf"):
            content_type = "application/pdf"
        elif path.endswith(".md"):
            content_type = "text/markdown"
        elif path.endswith(".png"):
            content_type = "image/png"
        elif path.endswith((".jpg", ".jpeg")):
            content_type = "image/jpeg"
        elif path.endswith(".svg"):
            content_type = "image/svg+xml"
        elif path.endswith(".gif"):
            content_type = "image/gif"
        elif path.endswith(".webp"):
            content_type = "image/webp"
        elif path.endswith(".css"):
            content_type = "text/css"
        elif path.endswith(".js"):
            content_type = "application/javascript"
        elif path.endswith(".json"):
            content_type = "application/json"
        elif path.endswith(".mp4"):
            content_type = "video/mp4"
        elif path.endswith(".webm"):
            content_type = "video/webm"
        return Response(content=file_path.read_bytes(), media_type=content_type)
    # Try index.html in knowledge dir
    index_path = file_path / "index.html"
    if index_path.exists():
        return HTMLResponse(index_path.read_text())
    # Try knowledge/index.html if path is just course code
    knowledge_path = file_path / "knowledge" / "index.html"
    if knowledge_path.exists():
        return HTMLResponse(knowledge_path.read_text())
    raise HTTPException(404, f"Course resource not found: {path}")

# ─── STATIC HTML PAGES ─────────────────────────────────
@app.get("/{filename}.html", response_class=HTMLResponse)
async def serve_static(filename: str):
    file_path = BASE / f"{filename}.html"
    if file_path.exists() and file_path.is_file():
        return file_path.read_text()
    raise HTTPException(404, f"Page not found: {filename}.html")

# ─── HEALTH ─────────────────────────────────────────────

@app.post("/emba2026/save-profile")
async def save_profile_image(request: Request):
    """Download and permanently save LINE profile image."""
    try:
        data = await request.json()
        image_url = data.get("image_url")
        user_id = data.get("user_id", "unknown")
        if not image_url:
            return {"error": "No image_url provided"}
        
        import aiohttp
        async with aiohttp.ClientSession() as session:
            async with session.get(image_url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                if resp.status == 200:
                    img_data = await resp.read()
                    # Save to emba2026/images/
                    img_dir = BASE / "emba2026" / "images"
                    img_dir.mkdir(exist_ok=True)
                    ext = ".jpg"
                    if ".png" in image_url.lower(): ext = ".png"
                    filename = f"profile_{user_id}{ext}"
                    filepath = img_dir / filename
                    filepath.write_bytes(img_data)
                    return {"saved": True, "url": f"/emba2026/images/{filename}"}
        return {"saved": False, "error": f"Failed to fetch image: {resp.status}"}
    except Exception as e:
        return {"saved": False, "error": str(e)}

@app.get("/health")
async def health():
    return {"status": "ok", "hub": True}

# ─── STARTUP ────────────────────────────────────────────
@app.on_event("startup")
async def startup():
    global _client
    _client = httpx.AsyncClient(timeout=httpx.Timeout(30.0))
    print("Hub server ready — sasin.cfoth.ai on port 8900")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8081))
    uvicorn.run(app, host="0.0.0.0", port=port)
