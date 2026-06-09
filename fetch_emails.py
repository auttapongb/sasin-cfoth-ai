#!/usr/bin/env python3
"""Fetch unread Gmail + AI-analyze each email via DeepSeek."""
import json, requests, os, re
from datetime import datetime, timezone

TF = '/data/lgiap/oauth_gmail_token.json'
OF = '/data/sasin-cfoth-ai/email_data.json'
SF = '/data/sasin-cfoth-ai/seen_emails.json'
DEEPSEEK_KEY = 'sk-fc0f26ac5a6c441e83950fb87fe93660'
BLOCKED = ['capitaliq.spglobal.com', 'spglobal.com']

def rt():
    with open(TF) as f:
        t = json.load(f)
    r = requests.post('https://oauth2.googleapis.com/token', data={
        'client_id': t['client_id'], 'client_secret': t['client_secret'],
        'refresh_token': t['refresh_token'], 'grant_type': 'refresh_token'
    })
    return r.json()['access_token']

def gh(payload):
    hs = {}
    for h in payload.get('headers', []):
        hs[h['name'].lower()] = h['value']
    return hs

# ── Fetch from Gmail ──
seen = set()
if os.path.exists(SF):
    with open(SF) as f:
        seen = set(json.load(f).get('ids', []))

at = rt()
base = 'https://gmail.googleapis.com/gmail/v1/users/me'
hdrs = {'Authorization': 'Bearer ' + at}
resp = requests.get(base + '/messages', headers=hdrs,
    params={'maxResults': 20, 'q': 'is:unread newer_than:7d'})
messages = resp.json().get('messages', [])

raw_emails = []
for msg in messages[:20]:
    mid = msg['id']
    short_id = mid[:16]
    detail = requests.get(base + '/messages/' + mid, headers=hdrs,
        params={'format': 'full'}).json()
    payload = detail.get('payload', {})
    hs = gh(payload)
    
    subj = hs.get('subject', '(no subject)')
    frm = hs.get('from', '?')
    snip = detail.get('snippet', '')[:300]
    body = detail.get('snippet', '')
    # Try to get full body
    if 'parts' in payload:
        for p in payload['parts']:
            if p.get('mimeType') == 'text/plain':
                data = p.get('body', {}).get('data', '')
                if data:
                    import base64
                    body = base64.urlsafe_b64decode(data + '==').decode('utf-8', errors='ignore')[:500]
                    break
    
    if any(b in frm.lower() for b in BLOCKED):
        continue
    
    raw_emails.append({
        'id': short_id,
        'from': frm,
        'subject': subj,
        'body': body[:500],
    })

if not raw_emails:
    print('NO_MAIL')
    exit(0)

# ── AI Analysis via DeepSeek ──
emails_json = json.dumps([{'id': e['id'], 'from': e['from'], 'subject': e['subject'], 'body': e['body'][:400]} for e in raw_emails], ensure_ascii=False)

prompt = f"""Analyze these unread emails for an EMBA student named Auttapong at Sasin School of Management. For each email, provide:
- cat: category (one of: sasin, finance, security, event, resource, personal, newsletter, other)
- label: a short display label (e.g. "URGENT", "Networking", "Invoice", "Class Material", "Account")
- summary: one sentence summarizing what the email is about and why it matters to the student
- action: one sentence with a specific recommended action

Return ONLY valid JSON array, no other text. Format:
[{{"id":"...","cat":"...","label":"...","summary":"...","action":"..."}},...]

Emails:
{emails_json}"""

resp = requests.post('https://api.deepseek.com/v1/chat/completions',
    headers={'Authorization': f'Bearer {DEEPSEEK_KEY}', 'Content-Type': 'application/json'},
    json={
        'model': 'deepseek-chat',
        'messages': [{'role': 'user', 'content': prompt}],
        'temperature': 0.3,
        'max_tokens': 2000
    },
    timeout=60
)

ai_result = resp.json()
content = ai_result.get('choices', [{}])[0].get('message', {}).get('content', '[]')

# Parse JSON from response (handle markdown wrapping)
content = re.sub(r'```json\s*|```', '', content).strip()
try:
    analyses = json.loads(content)
except:
    print(f'AI_PARSE_ERROR: {content[:200]}')
    # Fallback: use basic rules
    analyses = []
    for e in raw_emails:
        analyses.append({
            'id': e['id'],
            'cat': 'other',
            'label': 'INFO',
            'summary': e['body'][:120] if e.get('body') else e['subject'],
            'action': 'Review and decide'
        })

# ── Merge with raw data ──
analysis_map = {a['id']: a for a in analyses}
all_emails = []
new_emails = []

for e in raw_emails:
    a = analysis_map.get(e['id'], {})
    email = {
        'cat': a.get('cat', 'other'),
        'label': a.get('label', 'INFO'),
        'subject': e['subject'],
        'summary': a.get('summary', e.get('body', '')[:120]),
        'action': a.get('action', 'Review and decide'),
        'from': e['from'],
    }
    all_emails.append(email)
    if e['id'] not in seen:
        new_emails.append(email)
        seen.add(e['id'])

# ── Save ──
seen_list = list(seen)[-500:]
os.makedirs(os.path.dirname(SF), exist_ok=True)
with open(SF, 'w') as f:
    json.dump({'ids': seen_list, 'updated': datetime.now(timezone.utc).isoformat()}, f)

counts = {}
for e in all_emails:
    counts[e['cat']] = counts.get(e['cat'], 0) + 1

with open(OF, 'w') as f:
    json.dump({
        'updated': datetime.now(timezone.utc).isoformat(),
        'unread_total': len(all_emails),
        'counts': counts,
        'emails': all_emails
    }, f, ensure_ascii=False)

print(f'AI_ANALYZED: {len(all_emails)} emails, new: {len(new_emails)}')
