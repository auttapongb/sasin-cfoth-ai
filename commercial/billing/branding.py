"""White-label / branding service for Sasin AI Learning Toolkit.

Stores per-org branding config and serves custom CSS, domains, and assets.
Integrated into the billing server as additional routes.
"""

import os
from pathlib import Path
from typing import Optional

from pydantic import BaseModel

from db import get_org, upsert_org


# ── Branding Config Model ──

class BrandingConfig(BaseModel):
    """Per-organization branding settings."""
    org_name: str = "Sasin AI Learning Toolkit"
    logo_url: str = ""
    favicon_url: str = ""
    primary_color: str = "#3b82f6"       # Blue
    secondary_color: str = "#22c55e"     # Green
    accent_color: str = "#eab308"        # Yellow
    background_color: str = "#0f172a"    # Dark navy
    surface_color: str = "#1e293b"       # Card bg
    text_color: str = "#e2e8f0"          # Light text
    font_family: str = "system-ui, -apple-system, sans-serif"
    custom_css: str = ""
    custom_domain: str = ""              # e.g., "learn.mycompany.com"
    custom_domain_verified: bool = False
    platform_title: str = "AI Learning Toolkit"
    footer_text: str = "Powered by Sasin AI"
    hide_powered_by: bool = False


DEFAULT_BRANDING = BrandingConfig()


def get_branding(org_id: str) -> BrandingConfig:
    """Get branding for an organization."""
    org = get_org(org_id)
    if not org or "branding" not in org:
        return DEFAULT_BRANDING
    return BrandingConfig(**org["branding"])


def save_branding(org_id: str, branding: BrandingConfig):
    """Save branding for an organization."""
    org = get_org(org_id)
    if not org:
        return
    org["branding"] = branding.model_dump()
    upsert_org(org)


def get_org_by_domain(domain: str) -> Optional[str]:
    """Find org_id by custom domain."""
    from db import list_orgs
    for org in list_orgs():
        branding = org.get("branding", {})
        if branding.get("custom_domain") == domain and branding.get("custom_domain_verified"):
            return org["org_id"]
    return None


def generate_css(branding: BrandingConfig) -> str:
    """Generate CSS variables from branding config."""
    custom = branding.custom_css or ""

    css = f"""
/* White-label theme for {branding.org_name} */
:root {{
  --primary: {branding.primary_color};
  --secondary: {branding.secondary_color};
  --accent: {branding.accent_color};
  --bg: {branding.background_color};
  --surface: {branding.surface_color};
  --text: {branding.text_color};
  --font: {branding.font_family};
}}

/* Override platform title */
.platform-title::after {{
  content: "{branding.platform_title}";
}}

/* Hide powered-by if requested */
{"footer .powered-by { display: none; }" if branding.hide_powered_by else ""}

/* Custom header logo */
.header-logo {{
  content: url({branding.logo_url or '/default-logo.svg'});
}}

/* Org custom CSS */
{custom}
"""
    return css.strip()


def verify_domain(org_id: str, domain: str) -> dict:
    """Verify domain ownership (DNS TXT record check).
    
    In production, this would actually check DNS.
    For now, returns the verification token.
    """
    import hashlib
    token = hashlib.sha256(f"{org_id}:{domain}".encode()).hexdigest()[:32]
    return {
        "domain": domain,
        "token": token,
        "txt_record": f"sasin-verify={token}",
        "verified": False,  # Set to True after DNS check
    }
