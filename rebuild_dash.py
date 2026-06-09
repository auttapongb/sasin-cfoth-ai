#!/usr/bin/env python3
"""Rebuild dashboard: inject Godji's AI-analyzed emails from email_data.json.
IDEMPOTENT — cleans up before injecting. No duplicate JS/CSS injection.
"""
import re, base64, os, json, subprocess

HTML = '/data/sasin-cfoth-ai/index.html'
DATA = '/data/sasin-cfoth-ai/email_data.json'
GITHUB_RAW = 'https://raw.githubusercontent.com/auttapongb/sasin-cfoth-ai/main/index.html'

# ── 1. Always start from clean GitHub baseline ──
try:
    import urllib.request
    with urllib.request.urlopen(GITHUB_RAW) as r:
        fresh = r.read().decode()
    with open(HTML, 'w') as f:
        f.write(fresh)
    print(f"Pulled fresh HTML from GitHub ({len(fresh)} bytes)")
except Exception as e:
    print(f"GitHub pull failed ({e}), using local copy")

# ── 2. Read Godji's email data ──
if not os.path.exists(DATA):
    print("No email_data.json yet — skipping")
    exit(0)

with open(DATA) as f:
    email_data = json.load(f)

# ── 3. Transform data for inboxBox (pass through AI fields directly) ──
transformed = {"unread_total": email_data.get("unread_total", 0), "emails": []}
for e in email_data.get("emails", []):
    cat = e.get("cat", e.get("category", "other"))
    lbl = e.get("label", cat.upper())
    transformed["emails"].append({
        "cat": cat,
        "label": lbl,
        "subject": e.get("subject", ""),
        "summary": e.get("summary", ""),
        "action": e.get("action", ""),
        "from": e.get("from", ""),
    })

# ── 4. Inject emailData script (replacing any existing) ──
with open(HTML) as f:
    html = f.read()

data_b64 = base64.b64encode(json.dumps(transformed).encode()).decode()

# Remove ALL old emailData scripts
html = re.sub(r'<script id="emailData"[^>]*>[^<]*</script>\n?', '', html)

# Remove ALL old EMAIL_DATA parsers (var and const variants)
html = re.sub(r'(?:var|const)\s+EMAIL_DATA\s*=\s*\(function\(\)\{[^}]+\}\)\(\);\n?', '', html)

# Inject fresh emailData before calendarData
html = html.replace(
    '<script id="calendarData"',
    '<script id="emailData" type="application/octet-stream">' + data_b64 + '</script>\n<script id="calendarData"',
    1
)

# ── 5. Write back ──
with open(HTML, 'w') as f:
    f.write(html)

# ── 6. Verify ──
ec = html.count('id="emailData"')
pc = html.count('var EMAIL_DATA')
ic = html.count('var inboxBox=')
print(f"Done: emailData={ec}, EMAIL_DATA={pc}, inboxBox={ic}")
