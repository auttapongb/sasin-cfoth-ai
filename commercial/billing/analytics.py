"""Analytics endpoints — add to billing server.

Usage: import and add routes to server.py
"""

from datetime import datetime, timedelta
from typing import Optional

from fastapi import Query
from db import list_orgs, get_usage_summary, get_usage


# ── Pricing calculator ──

PRICING = {
    "slide_analysis": 0.00003,  # $0.00003 per slide (Gemini Flash)
    "stt_minutes": 0.004,       # $0.004/min (Groq Whisper)
    "ai_tokens": 0.000001,      # $0.001/1K tokens (average)
    "kb_queries": 0.0001,       # $0.0001 per query (embedding + LLM)
    "kb_uploads": 0.0,          # Free
}


def calculate_costs(org_id: str, days: int = 30) -> dict:
    """Calculate per-metric costs for an org."""
    usage = get_usage_summary(org_id, since_days=days)
    costs = {}
    total = 0.0

    for metric, value in usage.items():
        rate = PRICING.get(metric, 0)
        cost = value * rate
        costs[metric] = {
            "usage": round(value, 2),
            "rate": rate,
            "cost": round(cost, 4),
        }
        total += cost

    return {
        "org_id": org_id,
        "period_days": days,
        "metrics": costs,
        "total_cost": round(total, 4),
        "projected_monthly": round(total * (30 / max(days, 1)), 2),
    }


def org_analytics(org_id: str) -> dict:
    """Comprehensive org analytics."""
    org = __import__("db").get_org(org_id)
    if not org:
        return {"error": "Org not found"}

    costs_30d = calculate_costs(org_id, 30)
    costs_7d = calculate_costs(org_id, 7)

    return {
        "org_id": org_id,
        "plan": org.get("plan", "free"),
        "seats": f"{org.get('seats_used', 0)}/{org.get('seats_max', 5)}",
        "costs": {
            "last_7_days": costs_7d["total_cost"],
            "last_30_days": costs_30d["total_cost"],
            "projected_monthly": costs_30d["projected_monthly"],
            "breakdown": costs_30d["metrics"],
        },
        "storage_used_gb": round(org.get("storage_used_bytes", 0) / (1024**3), 2),
    }


def global_analytics() -> dict:
    """Platform-wide analytics."""
    orgs = list_orgs()
    total_orgs = len(orgs)
    total_users = sum(1 for o in orgs for u in __import__("db").list_users(o["org_id"]))
    active_orgs = sum(1 for o in orgs if o.get("subscription_status") == "active")

    total_cost_30d = sum(
        calculate_costs(o["org_id"], 30)["total_cost"] for o in orgs
    )

    plans = {}
    for o in orgs:
        plan = o.get("plan", "free")
        plans[plan] = plans.get(plan, 0) + 1

    return {
        "total_orgs": total_orgs,
        "active_orgs": active_orgs,
        "total_users": total_users,
        "total_cost_30d": round(total_cost_30d, 2),
        "plans": plans,
        "revenue_mrr": round(
            sum(
                (o.get("seats_used", 0) or 0) * 29
                for o in orgs if o.get("plan") == "pro"
            ),
            2,
        ),
    }
