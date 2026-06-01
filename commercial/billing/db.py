"""Simple file-based DB for billing — migrate to PostgreSQL for production."""

import json
import os
import threading
from pathlib import Path
from typing import Optional

DB_DIR = Path(os.environ.get("BILLING_DB_DIR", "/root/sasin-commercial/data"))
DB_DIR.mkdir(parents=True, exist_ok=True)
_lock = threading.Lock()


def _read(table: str) -> dict:
    path = DB_DIR / f"{table}.json"
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def _write(table: str, data: dict):
    path = DB_DIR / f"{table}.json"
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)


# ── Organizations ──

def get_org(org_id: str) -> Optional[dict]:
    with _lock:
        return _read("orgs").get(org_id)


def list_orgs() -> list[dict]:
    with _lock:
        return list(_read("orgs").values())


def upsert_org(org: dict):
    with _lock:
        data = _read("orgs")
        org["updated_at"] = str(org.get("updated_at", ""))
        data[org["org_id"]] = org
        _write("orgs", data)


def delete_org(org_id: str):
    with _lock:
        data = _read("orgs")
        data.pop(org_id, None)
        _write("orgs", data)


# ── Users ──

def get_user(user_id: str) -> Optional[dict]:
    with _lock:
        return _read("users").get(user_id)


def list_users(org_id: Optional[str] = None) -> list[dict]:
    with _lock:
        users = list(_read("users").values())
    if org_id:
        users = [u for u in users if u.get("org_id") == org_id]
    return users


def upsert_user(user: dict):
    with _lock:
        data = _read("users")
        data[user["user_id"]] = user
        _write("users", data)


# ── Usage ──

def record_usage(org_id: str, metric: str, value: float):
    with _lock:
        data = _read("usage")
        entries = data.get(org_id, [])
        from datetime import datetime
        entries.append({
            "metric": metric,
            "value": value,
            "timestamp": datetime.utcnow().isoformat(),
        })
        data[org_id] = entries[-10000:]  # keep last 10k records
        _write("usage", data)


def get_usage(org_id: str, metric: Optional[str] = None, limit: int = 100) -> list[dict]:
    with _lock:
        entries = _read("usage").get(org_id, [])
    if metric:
        entries = [e for e in entries if e["metric"] == metric]
    return entries[-limit:]


def get_usage_summary(org_id: str, since_days: int = 30) -> dict:
    from datetime import datetime, timedelta
    cutoff = datetime.utcnow() - timedelta(days=since_days)
    entries = get_usage(org_id)
    summary = {}
    for e in entries:
        try:
            ts = datetime.fromisoformat(e["timestamp"])
        except (ValueError, TypeError):
            continue
        if ts < cutoff:
            continue
        metric = e["metric"]
        summary[metric] = summary.get(metric, 0) + e["value"]
    return summary
