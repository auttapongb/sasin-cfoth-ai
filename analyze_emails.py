#!/usr/bin/env python3
"""Generate AI-style email analysis from raw email data."""
import json, os

INFILE = '/data/sasin-cfoth-ai/email_data.json'
OUTFILE = '/data/sasin-cfoth-ai/email_analysis.json'

if not os.path.exists(INFILE):
    print("No email data yet")
    exit(0)

with open(INFILE) as f:
    data = json.load(f)

def analyze(e):
    subj = (e.get('subject') or '').lower()
    summ = (e.get('summary') or '').lower()
    cat = e.get('category', 'other')
    
    if cat == 'security':
        if any(w in (subj+summ) for w in ['recovery', 'changed', 'phone', 'email was']):
            return 'CRITICAL: Account recovery info changed - verify it was you immediately'
        device = ''
        if 'windows' in summ: device = 'Windows'
        elif 'mac' in summ: device = 'Mac'
        elif 'ipad' in summ or 'iphone' in summ: device = 'iOS'
        elif 'android' in summ: device = 'Android'
        if device:
            return f'Sign-in from {device} - verify this was you'
        return 'Security alert - verify recent activity'
    
    if cat == 'sasin':
        if 'sign in' in subj or 'signed in' in summ:
            return 'Sasinware login from new device - verify, no action if recognized'
        if 'form' in subj or 'filling out' in summ:
            return 'Form submission receipt - no action needed'
        if 'support' in subj or 'ticket' in summ:
            return 'Support ticket created - team is working on it'
        return 'Sasin communication - check Sasinware for updates'
    
    if cat == 'finance':
        if 'invoice' in subj or 'license' in summ:
            return 'Invoice received - review and pay if legitimate purchase'
        return 'Payment-related email - review'
    
    if cat == 'newsletter':
        return 'Newsletter - skim or archive'
    
    if cat == 'personal':
        return 'Personal email - read and reply if needed'
    
    return 'Low priority - review when free'

output = {'unread_total': len(data['emails']), 'emails': []}
for e in data['emails']:
    output['emails'].append({
        'subject': e['subject'],
        'category': e['category'],
        'analysis': analyze(e)
    })

os.makedirs(os.path.dirname(OUTFILE), exist_ok=True)
with open(OUTFILE, 'w') as f:
    json.dump(output, f, ensure_ascii=False)
print(f'OK: {len(output["emails"])} analyses')
