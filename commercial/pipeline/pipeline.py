"""Capture → DeepTutor Knowledge Base Pipeline.

Watches for new transcript files and auto-uploads them to DeepTutor KB.
Usage: python pipeline.py [--kb-name emba-2026] [--watch-dir /path/to/transcripts]
"""

import os
import sys
import time
import json
import hashlib
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional

import httpx

# ── Config ──
DEEPTUTOR_API = os.environ.get("DEEPTUTOR_API", "http://127.0.0.1:8001")
KB_NAME = os.environ.get("PIPELINE_KB_NAME", "emba-2026")
WATCH_DIR = os.environ.get(
    "PIPELINE_WATCH_DIR",
    "/docker/hermes-bot/data/sasin-cfoth-ai/capture/transcripts",
)
BILLING_API = os.environ.get("BILLING_API", "http://127.0.0.1:8500")
ORG_ID = os.environ.get("PIPELINE_ORG_ID", "default")
POLL_INTERVAL = int(os.environ.get("PIPELINE_POLL_INTERVAL", "15"))
STATE_FILE = os.environ.get(
    "PIPELINE_STATE_FILE",
    "/root/sasin-commercial/data/pipeline_state.json",
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [Pipeline] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("pipeline")


class Pipeline:
    def __init__(self):
        self.state = self._load_state()
        self.client = httpx.Client(timeout=120.0)
        log.info(f"Pipeline started — KB: {KB_NAME}, Watch: {WATCH_DIR}")
        log.info(f"DeepTutor API: {DEEPTUTOR_API}, Billing: {BILLING_API}")

    def _load_state(self) -> set:
        """Load already-processed file hashes."""
        try:
            if os.path.exists(STATE_FILE):
                with open(STATE_FILE) as f:
                    return set(json.load(f))
        except Exception:
            pass
        return set()

    def _save_state(self):
        """Save processed file hashes."""
        os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
        with open(STATE_FILE, "w") as f:
            json.dump(list(self.state), f)

    def _file_hash(self, path: Path) -> str:
        """SHA256 of file content."""
        sha = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha.update(chunk)
        return sha.hexdigest()

    def _find_new_files(self) -> list[Path]:
        """Find new transcript files not yet processed."""
        watch = Path(WATCH_DIR)
        if not watch.exists():
            log.warning(f"Watch directory does not exist: {WATCH_DIR}")
            return []

        new_files = []
        for ext in [".txt", ".md", ".json"]:
            for path in sorted(watch.glob(f"**/*{ext}")):
                if path.is_file():
                    fhash = self._file_hash(path)
                    key = f"{path.name}:{fhash}"
                    if key not in self.state:
                        new_files.append(path)
        return new_files

    def _extract_metadata(self, path: Path) -> dict:
        """Extract lecture metadata from filename or content."""
        name = path.stem
        # Try to parse date from filename (e.g., "2026-06-01-lecture.txt")
        date_str = None
        title = name
        for part in name.split("-"):
            if len(part) == 10 and part[4] == "-" and part[7] == "-":
                date_str = part
                title = name.replace(f"{date_str}-", "").replace("-", " ")
                break

        return {
            "filename": path.name,
            "title": title,
            "date": date_str or datetime.now().strftime("%Y-%m-%d"),
            "size_bytes": path.stat().st_size,
            "type": "lecture_transcript",
        }

    def upload_to_deeptutor(self, path: Path) -> Optional[str]:
        """Upload transcript file to DeepTutor Knowledge Base."""
        metadata = self._extract_metadata(path)

        # Prepend metadata header to content
        with open(path, "r") as f:
            content = f.read()

        header = f"""---
title: {metadata['title']}
date: {metadata['date']}
type: {metadata['type']}
source: Sasin Capture Server
---

"""
        full_content = header + content

        # Write enriched file
        enriched_path = path.with_suffix(".enriched.txt")
        with open(enriched_path, "w") as f:
            f.write(full_content)

        try:
            # Upload to DeepTutor
            url = f"{DEEPTUTOR_API}/api/knowledge/{KB_NAME}/upload"
            log.info(f"Uploading to DeepTutor: {path.name} ({metadata['size_bytes']} bytes)")

            with open(enriched_path, "rb") as f:
                files = {"files": (enriched_path.name, f, "text/plain")}
                resp = self.client.post(url, files=files)

            if resp.status_code == 200:
                result = resp.json()
                task_id = result.get("task_id", "unknown")
                log.info(f"✅ Uploaded: {path.name} → KB '{KB_NAME}' (task: {task_id})")
                return task_id
            else:
                log.error(f"❌ Upload failed ({resp.status_code}): {resp.text[:200]}")
                return None

        except Exception as e:
            log.error(f"❌ Upload error: {e}")
            return None
        finally:
            # Clean up enriched file
            enriched_path.unlink(missing_ok=True)

    def record_usage(self, path: Path, task_id: Optional[str]):
        """Record usage metrics in billing service."""
        metadata = self._extract_metadata(path)
        word_count = 0
        try:
            with open(path, "r") as f:
                word_count = len(f.read().split())
        except Exception:
            pass

        try:
            # Record STT minutes (rough estimate: 150 words/minute)
            stt_minutes = max(1, word_count / 150)
            self.client.post(
                f"{BILLING_API}/usage",
                json={
                    "org_id": ORG_ID,
                    "metric": "stt_minutes",
                    "value": round(stt_minutes, 1),
                },
                timeout=5,
            )

            # Record KB uploads
            self.client.post(
                f"{BILLING_API}/usage",
                json={
                    "org_id": ORG_ID,
                    "metric": "kb_uploads",
                    "value": 1,
                },
                timeout=5,
            )
        except Exception as e:
            log.debug(f"Usage recording failed (non-critical): {e}")

    def mark_processed(self, path: Path):
        """Mark file as processed."""
        fhash = self._file_hash(path)
        key = f"{path.name}:{fhash}"
        self.state.add(key)
        self._save_state()

    def process_file(self, path: Path) -> bool:
        """Process a single transcript file end-to-end."""
        log.info(f"📄 Processing: {path.name}")
        task_id = self.upload_to_deeptutor(path)
        self.record_usage(path, task_id)
        self.mark_processed(path)
        return task_id is not None

    def run_once(self):
        """Scan and process all new files (one-shot mode)."""
        new_files = self._find_new_files()
        if not new_files:
            log.debug("No new files")
            return

        log.info(f"Found {len(new_files)} new transcript(s)")
        success = 0
        for path in new_files:
            if self.process_file(path):
                success += 1
        log.info(f"Batch done: {success}/{len(new_files)} uploaded")

    def run_forever(self):
        """Watch mode — poll for new files continuously."""
        log.info("Starting watch mode...")
        while True:
            try:
                self.run_once()
            except Exception as e:
                log.error(f"Poll error: {e}")
            time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Capture → DeepTutor Pipeline")
    parser.add_argument("--once", action="store_true", help="Run once and exit")
    parser.add_argument("--watch", action="store_true", default=True, help="Watch mode")
    args = parser.parse_args()

    pipeline = Pipeline()

    if args.once:
        pipeline.run_once()
    else:
        pipeline.run_forever()
