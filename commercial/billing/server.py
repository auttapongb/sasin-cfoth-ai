"""Billing & Management API Server for Sasin AI Learning Toolkit.

Endpoints:
  POST /stripe/webhook          — Stripe event webhook
  POST /checkout                — Create Stripe checkout session
  POST /portal                  — Create Stripe billing portal session
  GET  /orgs                    — List all organizations
  GET  /orgs/{org_id}           — Get org details + usage
  POST /orgs                    — Create organization
  PUT  /orgs/{org_id}           — Update organization
  POST /usage                   — Record usage metric
  GET  /health                  — Health check
  GET  /dashboard               — Admin dashboard HTML
"""

import os
import sys
from datetime import datetime
from contextlib import asynccontextmanager
from typing import Optional

import stripe
import uvicorn
from fastapi import FastAPI, HTTPException, Request, Query
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware

from models import (
    Organization, OrgUser, PlanID, SubscriptionStatus,
    CheckoutSessionRequest, BillingPortalRequest, UsageRecord,
    PLAN_DETAILS,
)
from db import (
    get_org, list_orgs, upsert_org, delete_org,
    get_user, list_users, upsert_user,
    record_usage, get_usage, get_usage_summary,
)
from sso import (
    generate_state, get_google_auth_url, handle_google_callback,
    get_microsoft_auth_url, handle_microsoft_callback,
    GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET,
    MICROSOFT_CLIENT_ID, MICROSOFT_CLIENT_SECRET,
    ALLOWED_DOMAINS, SSO_BASE_URL,
)

# ── Config ──
STRIPE_SECRET_KEY = os.environ.get("STRIPE_SECRET_KEY", "")
STRIPE_WEBHOOK_SECRET = os.environ.get("STRIPE_WEBHOOK_SECRET", "")
STRIPE_PRICE_PRO = os.environ.get("STRIPE_PRICE_PRO", "price_pro_monthly")
STRIPE_PRICE_ENTERPRISE = os.environ.get("STRIPE_PRICE_ENTERPRISE", "price_enterprise")

if STRIPE_SECRET_KEY:
    stripe.api_key = STRIPE_SECRET_KEY

BILLING_PORT = int(os.environ.get("BILLING_PORT", "8500"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Seed default org if none exists
    if not list_orgs():
        upsert_org(Organization(
            org_id="default",
            name="Sasin School of Management",
            slug="sasin",
            plan=PlanID.ENTERPRISE,
            subscription_status=SubscriptionStatus.ACTIVE,
            seats_max=999,
        ).model_dump())
        # Create admin user
        upsert_user(OrgUser(
            user_id="admin",
            org_id="default",
            email="admin@sasin.cfoth.ai",
            name="Admin",
            role="admin",
        ).model_dump())
    yield


app = FastAPI(
    title="Sasin Billing API",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Stripe Webhook ──

@app.post("/stripe/webhook")
async def stripe_webhook(request: Request):
    """Handle Stripe events."""
    payload = await request.body()
    sig_header = request.headers.get("stripe-signature")

    try:
        event = stripe.Webhook.construct_event(
            payload, sig_header, STRIPE_WEBHOOK_SECRET
        )
    except ValueError:
        raise HTTPException(400, "Invalid payload")
    except stripe.error.SignatureVerificationError:
        raise HTTPException(400, "Invalid signature")

    event_type = event["type"]
    data = event["data"]["object"]

    if event_type == "checkout.session.completed":
        _handle_checkout_completed(data)
    elif event_type == "customer.subscription.updated":
        _handle_subscription_updated(data)
    elif event_type == "customer.subscription.deleted":
        _handle_subscription_deleted(data)
    elif event_type == "invoice.paid":
        _handle_invoice_paid(data)

    return JSONResponse({"status": "ok"})


def _handle_checkout_completed(session):
    """Provision subscription after checkout."""
    client_ref = session.get("client_reference_id")
    customer_id = session.get("customer")
    subscription_id = session.get("subscription")

    if not client_ref:
        return

    org = get_org(client_ref)
    if not org:
        return

    org["stripe_customer_id"] = customer_id
    org["subscription_status"] = "active"

    # Determine plan from metadata
    plan = session.get("metadata", {}).get("plan", "pro")
    org["plan"] = plan
    org["seats_max"] = int(session.get("metadata", {}).get("seats", 5))

    upsert_org(org)
    print(f"[Billing] Upgraded org {client_ref} to {plan}")


def _handle_subscription_updated(sub):
    """Sync subscription status."""
    customer_id = sub.get("customer")
    status = sub.get("status")
    # Find org by stripe customer ID
    for org in list_orgs():
        if org.get("stripe_customer_id") == customer_id:
            org["subscription_status"] = status
            upsert_org(org)
            break


def _handle_subscription_deleted(sub):
    """Downgrade to free when subscription ends."""
    customer_id = sub.get("customer")
    for org in list_orgs():
        if org.get("stripe_customer_id") == customer_id:
            org["plan"] = "free"
            org["subscription_status"] = "canceled"
            org["seats_max"] = 5
            upsert_org(org)
            break


def _handle_invoice_paid(invoice):
    """Log successful payment."""
    customer_id = invoice.get("customer")
    amount = invoice.get("amount_paid")
    print(f"[Billing] Invoice paid: ${amount/100:.2f} (customer: {customer_id})")


# ── Checkout / Billing Portal ──

@app.post("/checkout")
async def create_checkout(req: CheckoutSessionRequest):
    """Create a Stripe Checkout session for subscription."""
    if not STRIPE_SECRET_KEY:
        raise HTTPException(500, "Stripe not configured")

    org = get_org(req.org_id)
    if not org:
        raise HTTPException(404, "Organization not found")

    price_id = {
        PlanID.PRO: STRIPE_PRICE_PRO,
        PlanID.ENTERPRISE: STRIPE_PRICE_ENTERPRISE,
    }.get(req.plan)

    if not price_id:
        raise HTTPException(400, f"Invalid plan: {req.plan}")

    try:
        # Create or get Stripe customer
        customer_id = org.get("stripe_customer_id")
        if customer_id:
            stripe.Customer.modify(customer_id, name=org["name"])
        else:
            customer = stripe.Customer.create(
                name=org["name"],
                metadata={"org_id": req.org_id},
            )
            customer_id = customer.id
            org["stripe_customer_id"] = customer_id
            upsert_org(org)

        session = stripe.checkout.Session.create(
            customer=customer_id,
            client_reference_id=req.org_id,
            mode="subscription",
            line_items=[{
                "price": price_id,
                "quantity": req.seats,
            }],
            metadata={
                "org_id": req.org_id,
                "plan": req.plan.value,
                "seats": str(req.seats),
            },
            success_url=req.success_url,
            cancel_url=req.cancel_url,
        )
        return {"url": session.url, "session_id": session.id}

    except stripe.error.StripeError as e:
        raise HTTPException(400, str(e))


@app.post("/portal")
async def create_portal(req: BillingPortalRequest):
    """Create Stripe Customer Portal session."""
    if not STRIPE_SECRET_KEY:
        raise HTTPException(500, "Stripe not configured")

    org = get_org(req.org_id)
    if not org or not org.get("stripe_customer_id"):
        raise HTTPException(404, "No billing account")

    try:
        session = stripe.billing_portal.Session.create(
            customer=org["stripe_customer_id"],
            return_url=req.return_url,
        )
        return {"url": session.url}
    except stripe.error.StripeError as e:
        raise HTTPException(400, str(e))


# ── Organization CRUD ──

@app.get("/orgs")
async def api_list_orgs():
    return {"orgs": list_orgs()}


@app.get("/orgs/{org_id}")
async def api_get_org(org_id: str):
    org = get_org(org_id)
    if not org:
        raise HTTPException(404, "Organization not found")

    # Attach usage
    org["usage_30d"] = get_usage_summary(org_id, since_days=30)
    org["users"] = list_users(org_id)
    org["plan_details"] = PLAN_DETAILS.get(PlanID(org.get("plan", "free")), {})
    return org


@app.post("/orgs")
async def api_create_org(org: Organization):
    existing = get_org(org.org_id)
    if existing:
        raise HTTPException(409, "Organization already exists")
    upsert_org(org.model_dump())
    return org


@app.put("/orgs/{org_id}")
async def api_update_org(org_id: str, updates: dict):
    org = get_org(org_id)
    if not org:
        raise HTTPException(404, "Organization not found")
    org.update(updates)
    org["updated_at"] = datetime.utcnow().isoformat()
    upsert_org(org)
    return org


@app.delete("/orgs/{org_id}")
async def api_delete_org(org_id: str):
    org = get_org(org_id)
    if not org:
        raise HTTPException(404, "Organization not found")
    delete_org(org_id)
    return {"status": "deleted"}


# ── Usage Tracking ──

@app.post("/usage")
async def api_record_usage(record: UsageRecord):
    record_usage(record.org_id, record.metric, record.value)
    return {"status": "recorded"}


@app.get("/usage/{org_id}")
async def api_get_usage(
    org_id: str,
    metric: Optional[str] = None,
    days: int = Query(default=30, ge=1, le=365),
):
    if metric and days > 0:
        # Get filtered, time-bounded usage
        summary = get_usage_summary(org_id, since_days=days)
        return {"org_id": org_id, "days": days, "usage": summary}
    raw = get_usage(org_id, metric=metric)
    return {"org_id": org_id, "count": len(raw), "records": raw}


# ── Health ──

@app.get("/health")
async def health():
    stripe_ok = bool(STRIPE_SECRET_KEY)
    return {
        "status": "ok",
        "stripe_configured": stripe_ok,
        "orgs": len(list_orgs()),
        "users": len(list_users()),
    }


# ── Admin Dashboard ──

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard():
    """Simple admin dashboard HTML."""
    return HTMLResponse("""
    <!DOCTYPE html>
    <html><head><title>Sasin Admin Dashboard</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        * { margin:0; padding:0; box-sizing:border-box; }
        body { font-family: system-ui, -apple-system, sans-serif; background:#0f172a; color:#e2e8f0; padding:2rem; }
        h1 { font-size:1.5rem; margin-bottom:1.5rem; }
        .cards { display:grid; grid-template-columns:repeat(auto-fill,minmax(280px,1fr)); gap:1rem; }
        .card { background:#1e293b; border-radius:12px; padding:1.5rem; border:1px solid #334155; }
        .card h2 { font-size:1rem; color:#94a3b8; margin-bottom:0.5rem; }
        .card .value { font-size:2rem; font-weight:700; }
        .card .sub { font-size:0.85rem; color:#64748b; margin-top:0.25rem; }
        .green { color:#22c55e; }
        .blue { color:#3b82f6; }
        .yellow { color:#eab308; }
        table { width:100%; border-collapse:collapse; margin-top:1.5rem; background:#1e293b; border-radius:12px; overflow:hidden; }
        th { text-align:left; padding:0.75rem 1rem; background:#334155; font-size:0.85rem; color:#94a3b8; }
        td { padding:0.75rem 1rem; border-bottom:1px solid #334155; font-size:0.9rem; }
        .badge { display:inline-block; padding:0.2rem 0.6rem; border-radius:999px; font-size:0.75rem; font-weight:600; }
        .badge-pro { background:#3b82f620; color:#3b82f6; }
        .badge-enterprise { background:#22c55e20; color:#22c55e; }
        .badge-free { background:#64748b20; color:#64748b; }
    </style></head><body>
    <h1>📊 Sasin AI Learning Toolkit — Admin Dashboard</h1>
    <div class="cards" id="cards">Loading...</div>
    <table id="orgs-table"><thead><tr><th>Organization</th><th>Plan</th><th>Seats</th><th>Status</th><th>Usage (30d)</th></tr></thead><tbody></tbody></table>
    <script>
    async function load() {
        try {
            const orgs = await (await fetch('/orgs')).json();
            const cards = document.getElementById('cards');
            cards.innerHTML = '';
            const counts = {active:0, total:0, pro:0, enterprise:0};
            orgs.orgs.forEach(o => {
                counts.total++;
                if (o.subscription_status === 'active') counts.active++;
                if (o.plan === 'pro') counts.pro++;
                if (o.plan === 'enterprise') counts.enterprise++;
            });
            cards.innerHTML = `
                <div class="card"><h2>Total Organizations</h2><div class="value blue">${counts.total}</div></div>
                <div class="card"><h2>Active Subscriptions</h2><div class="value green">${counts.active}</div></div>
                <div class="card"><h2>Pro Plans</h2><div class="value blue">${counts.pro}</div></div>
                <div class="card"><h2>Enterprise</h2><div class="value green">${counts.enterprise}</div></div>
            `;
            const tbody = document.querySelector('#orgs-table tbody');
            tbody.innerHTML = '';
            for (const o of orgs.orgs) {
                const resp = await fetch(`/orgs/${o.org_id}`);
                const detail = await resp.json();
                const planClass = 'badge-' + o.plan;
                const usageStr = Object.entries(detail.usage_30d||{}).map(([k,v]) => `${k}: ${parseInt(v)}`).join(', ') || '—';
                tbody.innerHTML += `<tr>
                    <td><strong>${o.name}</strong><br><small>${o.org_id}</small></td>
                    <td><span class="badge ${planClass}">${o.plan}</span></td>
                    <td>${o.seats_used}/${o.seats_max}</td>
                    <td>${o.subscription_status}</td>
                    <td>${usageStr}</td>
                </tr>`;
            }
        } catch(e) { document.getElementById('cards').innerHTML = '⚠️ API Error: ' + e.message; }
    }
    load();
    </script></body></html>
    """)


# ── SSO / OAuth ──

@app.get("/sso/login")
async def sso_login(
    provider: str = Query(...),
    redirect_after: str = Query("/"),
    org_id: str = Query("default"),
):
    """Initiate SSO login flow."""
    state = generate_state(provider, redirect_after, org_id)

    if provider == "google":
        url = get_google_auth_url(state)
    elif provider == "microsoft":
        url = get_microsoft_auth_url(state)
    else:
        raise HTTPException(400, f"Unknown provider: {provider}")

    return {"url": url}


@app.get("/sso/google/callback")
async def sso_google_callback(code: str, state: str):
    """Google OAuth callback."""
    result = handle_google_callback(code, state)
    if result and "error" in result:
        raise HTTPException(400, result["error"])

    # Redirect to frontend with token
    redirect_url = result.get("redirect_after", "/")
    token = result.get("token", "")
    return HTMLResponse(f"""
    <html><body>
    <script>
        document.cookie = "dt_token={token};path=/;max-age=86400;SameSite=Lax";
        window.location.href = "{redirect_url}";
    </script>
    <p>Signing in... <a href="{redirect_url}">Click here if not redirected</a></p>
    </body></html>
    """)


@app.get("/sso/microsoft/callback")
async def sso_microsoft_callback(code: str, state: str):
    """Microsoft OAuth callback."""
    result = handle_microsoft_callback(code, state)
    if result and "error" in result:
        raise HTTPException(400, result["error"])

    redirect_url = result.get("redirect_after", "/")
    token = result.get("token", "")
    return HTMLResponse(f"""
    <html><body>
    <script>
        document.cookie = "dt_token={token};path=/;max-age=86400;SameSite=Lax";
        window.location.href = "{redirect_url}";
    </script>
    <p>Signing in... <a href="{redirect_url}">Click here if not redirected</a></p>
    </body></html>
    """)


@app.get("/sso/config")
async def sso_config():
    """Return SSO configuration for frontend."""
    return {
        "providers": {
            "google": bool(GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET),
            "microsoft": bool(MICROSOFT_CLIENT_ID and MICROSOFT_CLIENT_SECRET),
        },
        "allowed_domains": ALLOWED_DOMAINS or None,
        "login_url": f"{SSO_BASE_URL}/sso/login",
    }


# ── Main ──

if __name__ == "__main__":
    print(f"[Billing] Starting on port {BILLING_PORT}")
    print(f"[Billing] Stripe configured: {bool(STRIPE_SECRET_KEY)}")
    uvicorn.run(app, host="0.0.0.0", port=BILLING_PORT, log_level="info")
