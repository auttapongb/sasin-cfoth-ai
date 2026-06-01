"""SSO / OAuth module for Sasin AI Learning Toolkit.

Supports:
- Google Workspace OAuth (OpenID Connect)
- Microsoft 365 OAuth (Azure AD / Entra ID)
- Generic OIDC (SAML via middleware later)

Flow:
1. User clicks "Sign in with Google/Microsoft"
2. Redirected to provider
3. Callback → verify token, find/create DeepTutor user
4. Issue DeepTutor JWT → redirect to app
"""

import os
import json
import hashlib
import secrets
import time
from urllib.parse import urlencode
from typing import Optional

import httpx
from jose import jwt as jose_jwt

# ── Config ──
DEEPTUTOR_API = os.environ.get("DEEPTUTOR_API", "http://127.0.0.1:8001")
SSO_BASE_URL = os.environ.get("SSO_BASE_URL", "https://billing.sasin.cfoth.ai")

GOOGLE_CLIENT_ID = os.environ.get("GOOGLE_CLIENT_ID", "")
GOOGLE_CLIENT_SECRET = os.environ.get("GOOGLE_CLIENT_SECRET", "")
GOOGLE_REDIRECT_URI = f"{SSO_BASE_URL}/sso/google/callback"

MICROSOFT_CLIENT_ID = os.environ.get("MICROSOFT_CLIENT_ID", "")
MICROSOFT_CLIENT_SECRET = os.environ.get("MICROSOFT_CLIENT_SECRET", "")
MICROSOFT_TENANT = os.environ.get("MICROSOFT_TENANT", "common")  # or specific tenant ID
MICROSOFT_REDIRECT_URI = f"{SSO_BASE_URL}/sso/microsoft/callback"

# DeepTutor auth settings
DT_JWT_SECRET = os.environ.get("DT_JWT_SECRET", "change-me-in-production")
DT_TOKEN_EXPIRE_HOURS = int(os.environ.get("DT_TOKEN_EXPIRE_HOURS", "24"))

# Allowed email domains (empty = allow all)
ALLOWED_DOMAINS = [
    d.strip() for d in os.environ.get("SSO_ALLOWED_DOMAINS", "").split(",") if d.strip()
]

# OAuth state storage (in production, use Redis/DB)
_oauth_states: dict[str, dict] = {}  # state → {provider, redirect_after, org_id}


def generate_state(provider: str, redirect_after: str, org_id: str = "default") -> str:
    """Generate OAuth state token with redirect info."""
    state = secrets.token_urlsafe(32)
    _oauth_states[state] = {
        "provider": provider,
        "redirect_after": redirect_after,
        "org_id": org_id,
        "created": time.time(),
    }
    # Cleanup old states (>10 min)
    now = time.time()
    for s in list(_oauth_states):
        if now - _oauth_states[s].get("created", 0) > 600:
            del _oauth_states[s]
    return state


def get_state_data(state: str) -> Optional[dict]:
    """Retrieve and consume OAuth state."""
    return _oauth_states.pop(state, None)


def check_domain(email: str) -> bool:
    """Check if email domain is allowed."""
    if not ALLOWED_DOMAINS:
        return True
    domain = email.split("@")[-1].lower()
    return domain in [d.lower() for d in ALLOWED_DOMAINS]


# ── Google OAuth ──

def get_google_auth_url(state: str) -> str:
    """Build Google OAuth authorization URL."""
    params = {
        "client_id": GOOGLE_CLIENT_ID,
        "redirect_uri": GOOGLE_REDIRECT_URI,
        "response_type": "code",
        "scope": "openid email profile",
        "state": state,
        "access_type": "offline",
        "prompt": "consent",
    }
    return f"https://accounts.google.com/o/oauth2/v2/auth?{urlencode(params)}"


def handle_google_callback(code: str, state: str) -> Optional[dict]:
    """Handle Google OAuth callback — exchange code for tokens, verify, create user."""
    state_data = get_state_data(state)
    if not state_data:
        return {"error": "Invalid or expired state"}

    # Exchange code for tokens
    try:
        resp = httpx.post(
            "https://oauth2.googleapis.com/token",
            data={
                "client_id": GOOGLE_CLIENT_ID,
                "client_secret": GOOGLE_CLIENT_SECRET,
                "code": code,
                "grant_type": "authorization_code",
                "redirect_uri": GOOGLE_REDIRECT_URI,
            },
            timeout=15,
        )
        tokens = resp.json()
        if "error" in tokens:
            return {"error": tokens.get("error_description", tokens["error"])}
    except Exception as e:
        return {"error": f"Token exchange failed: {e}"}

    id_token = tokens.get("id_token")
    if not id_token:
        return {"error": "No ID token received"}

    # Verify and decode ID token (in production, verify signature with Google certs)
    # For now, decode without verification then verify with Google's tokeninfo endpoint
    try:
        resp = httpx.get(
            f"https://oauth2.googleapis.com/tokeninfo?id_token={id_token}",
            timeout=10,
        )
        user_info = resp.json()
        if "error" in user_info:
            return {"error": f"Invalid ID token: {user_info['error']}"}
    except Exception as e:
        return {"error": f"Token verification failed: {e}"}

    email = user_info.get("email", "")
    if not email:
        return {"error": "No email in user info"}

    if not check_domain(email):
        return {"error": f"Email domain not allowed: {email.split('@')[-1]}"}

    return _create_or_get_dt_user(
        email=email,
        name=user_info.get("name", email.split("@")[0]),
        provider="google",
        provider_id=user_info.get("sub", email),
        org_id=state_data.get("org_id", "default"),
        redirect_after=state_data.get("redirect_after", "/"),
    )


# ── Microsoft OAuth ──

def get_microsoft_auth_url(state: str) -> str:
    """Build Microsoft OAuth authorization URL."""
    params = {
        "client_id": MICROSOFT_CLIENT_ID,
        "redirect_uri": MICROSOFT_REDIRECT_URI,
        "response_type": "code",
        "scope": "openid email profile User.Read",
        "state": state,
        "response_mode": "query",
    }
    base = f"https://login.microsoftonline.com/{MICROSOFT_TENANT}/oauth2/v2.0/authorize"
    return f"{base}?{urlencode(params)}"


def handle_microsoft_callback(code: str, state: str) -> Optional[dict]:
    """Handle Microsoft OAuth callback."""
    state_data = get_state_data(state)
    if not state_data:
        return {"error": "Invalid or expired state"}

    # Exchange code for tokens
    try:
        resp = httpx.post(
            f"https://login.microsoftonline.com/{MICROSOFT_TENANT}/oauth2/v2.0/token",
            data={
                "client_id": MICROSOFT_CLIENT_ID,
                "client_secret": MICROSOFT_CLIENT_SECRET,
                "code": code,
                "grant_type": "authorization_code",
                "redirect_uri": MICROSOFT_REDIRECT_URI,
                "scope": "openid email profile User.Read",
            },
            timeout=15,
        )
        tokens = resp.json()
        if "error" in tokens:
            return {"error": tokens.get("error_description", tokens["error"])}
    except Exception as e:
        return {"error": f"Token exchange failed: {e}"}

    id_token = tokens.get("id_token", "")
    access_token = tokens.get("access_token", "")

    # Decode JWT claims
    try:
        claims = jose_jwt.get_unverified_claims(id_token)
    except Exception:
        claims = {}

    email = claims.get("email") or claims.get("preferred_username") or claims.get("upn", "")

    if not email:
        # Try Microsoft Graph
        try:
            resp = httpx.get(
                "https://graph.microsoft.com/v1.0/me",
                headers={"Authorization": f"Bearer {access_token}"},
                timeout=10,
            )
            profile = resp.json()
            email = profile.get("mail") or profile.get("userPrincipalName", "")
        except Exception:
            pass

    if not email:
        return {"error": "Could not determine email"}

    if not check_domain(email):
        return {"error": f"Email domain not allowed: {email.split('@')[-1]}"}

    name = claims.get("name", email.split("@")[0])

    return _create_or_get_dt_user(
        email=email,
        name=name,
        provider="microsoft",
        provider_id=claims.get("oid", email),
        org_id=state_data.get("org_id", "default"),
        redirect_after=state_data.get("redirect_after", "/"),
    )


# ── DeepTutor User Management ──

def _create_or_get_dt_user(
    email: str,
    name: str,
    provider: str,
    provider_id: str,
    org_id: str,
    redirect_after: str,
) -> dict:
    """Find or create a DeepTutor user, issue JWT."""
    user_id = hashlib.sha256(f"{provider}:{provider_id}".encode()).hexdigest()[:16]

    # Try to find existing user via DeepTutor API
    try:
        resp = httpx.post(
            f"{DEEPTUTOR_API}/api/v1/auth/sso-login",
            json={
                "user_id": user_id,
                "email": email,
                "name": name,
                "provider": provider,
                "org_id": org_id,
            },
            timeout=10,
        )
        if resp.status_code == 200:
            data = resp.json()
            dt_token = data.get("token", "")
        else:
            # First-time user — register first
            resp = httpx.post(
                f"{DEEPTUTOR_API}/api/v1/auth/sso-register",
                json={
                    "user_id": user_id,
                    "email": email,
                    "name": name,
                    "provider": provider,
                    "org_id": org_id,
                },
                timeout=10,
            )
            data = resp.json()
            dt_token = data.get("token", "")
    except Exception:
        # If DeepTutor API is unavailable, issue our own token
        dt_token = _issue_dt_token(user_id, email, name)

    return {
        "token": dt_token,
        "user_id": user_id,
        "email": email,
        "name": name,
        "redirect_after": redirect_after,
    }


def _issue_dt_token(user_id: str, email: str, name: str) -> str:
    """Issue a DeepTutor-compatible JWT."""
    payload = {
        "sub": user_id,
        "email": email,
        "name": name,
        "iat": int(time.time()),
        "exp": int(time.time()) + DT_TOKEN_EXPIRE_HOURS * 3600,
    }
    return jose_jwt.encode(payload, DT_JWT_SECRET, algorithm="HS256")
