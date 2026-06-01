"""Billing models for Sasin AI Learning Toolkit."""

from pydantic import BaseModel, Field
from enum import Enum
from typing import Optional
from datetime import datetime


class PlanID(str, Enum):
    FREE = "free"
    PRO = "pro"
    ENTERPRISE = "enterprise"


class SubscriptionStatus(str, Enum):
    ACTIVE = "active"
    PAST_DUE = "past_due"
    CANCELED = "canceled"
    TRIALING = "trialing"
    INCOMPLETE = "incomplete"


# ── Stripe price IDs (create these in Stripe dashboard) ──
# These are placeholder IDs — replace with real Stripe price IDs
STRIPE_PRICES = {
    PlanID.PRO: "price_pro_monthly",       # $29/seat/month
    PlanID.ENTERPRISE: "price_enterprise",  # custom quote
}

PLAN_DETAILS = {
    PlanID.FREE: {
        "name": "Free",
        "price_per_seat": 0,
        "max_seats": 5,
        "storage_gb": 10,
        "features": [
            "AI Lecture Capture",
            "Slide Analysis (100/mo)",
            "Basic RAG Search",
            "5 Knowledge Bases",
            "Community Support",
        ],
    },
    PlanID.PRO: {
        "name": "Pro",
        "price_per_seat": 29,
        "max_seats": 50,
        "storage_gb": 100,
        "features": [
            "Everything in Free",
            "Unlimited Slide Analysis",
            "Unlimited Knowledge Bases",
            "Live AI Assistant",
            "SSO (Google/Microsoft)",
            "LMS Integration",
            "Priority Support",
        ],
    },
    PlanID.ENTERPRISE: {
        "name": "Enterprise",
        "price_per_seat": 0,  # custom quote
        "max_seats": None,     # unlimited
        "storage_gb": None,    # unlimited
        "features": [
            "Everything in Pro",
            "White-Label / Custom Domain",
            "Dedicated Deployment",
            "Custom AI Models",
            "API Access",
            "SAML/SSO",
            "Dedicated Support",
            "SLA Guarantee",
        ],
    },
}


class Organization(BaseModel):
    org_id: str
    name: str
    slug: str
    stripe_customer_id: Optional[str] = None
    plan: PlanID = PlanID.FREE
    subscription_status: SubscriptionStatus = SubscriptionStatus.ACTIVE
    seats_used: int = 1
    seats_max: int = 5
    storage_used_bytes: int = 0
    created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())


class OrgUser(BaseModel):
    user_id: str
    org_id: str
    email: str
    name: str
    role: str = "member"  # admin | member | viewer
    created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())


class CheckoutSessionRequest(BaseModel):
    org_id: str
    plan: PlanID
    seats: int = 5
    success_url: str
    cancel_url: str


class BillingPortalRequest(BaseModel):
    org_id: str
    return_url: str


class UsageRecord(BaseModel):
    org_id: str
    metric: str  # "slide_analysis", "stt_minutes", "ai_tokens", "kb_queries"
    value: float
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())


class Invoice(BaseModel):
    invoice_id: str
    org_id: str
    stripe_invoice_id: Optional[str] = None
    amount_cents: int
    currency: str = "usd"
    status: str = "draft"
    period_start: str
    period_end: str
    created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
