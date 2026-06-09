# Sasin EMBA 2026 Dashboard — Requirements & Specifications

> **Read this before making ANY change to `index.html`.**
> Last updated: 10 Jun 2026

## 📐 Overall Layout Rules

1. **Two-column layout**: Sidebar (260px, resizable 180–500px) + Main content area
2. **Content area**: Flex column, `overflow-y: auto` on full page
3. **Box priority**: Inbox > Deadlines > Calendar > Action buttons (top to bottom)
4. **Calendar MUST be visible without excessive scrolling** — this is why boxes have fixed max-heights with scrollbars
5. **Never delete existing sections** — only modify/add
6. **Always preserve backups** before modifying

## 📬 Inbox Section

### Position
- First element in `#content` innerHTML
- Injected by cron job via `inboxBox` variable
- Prepended to the template literal in `showHome()`: `inboxBox + \`...\``

### HTML Classes
| Class | Element | Purpose |
|-------|---------|---------|
| `.dash-card.ib` | Container | Blue left border (#4285f4) |
| `.ib-body` | Scroll wrapper | **max-height: 140px**, overflow-y: auto |
| `.ir` | Row / email item | Flex row with gap, border-bottom separator |
| `.il` | Category label | Colored badge (SASIN/PAY/SEC/NEWS/PERS/INFO) |
| `.ia` | Analysis text | Dimmed small text below subject |

### Data Fields (from EMAIL_DATA)
```json
{
  "unread_total": 5,
  "emails": [
    {
      "cat": "PAY|EVENT|sasin|security|newsletter|personal|Resource|2nd Brain|...",
      "label": "Display label text",
      "subject": "Email subject line",
      "summary": "AI-generated summary of the email",
      "action": "Recommended action for the user",
      "from": "sender@domain.com",
      "date": "Jun 9"
    }
  ]
}
```
**CRITICAL**: Field names are `cat` (not `category`), `label`, `subject`, `summary` (not `analysis`), `action`.
Categories include: `PAY`, `EVENT`, `sasin`, `security`, `newsletter`, `personal`, `Resource`, `2nd Brain`, and others.

### Categories & Colors
| Category | Label | CSS Class | Color |
|----------|-------|-----------|-------|
| `PAY` / `finance` | PAY | `.il-PAY` / `.il-finance` | Yellow (#facc15) |
| `sasin` | SASIN | `.il-sasin` | Green (#4ade80) |
| `security` | SEC | `.il-security` | Red (#f87171) |
| `EVENT` | EVENT | `.il-EVENT` | Purple (#a78bfa) |
| `newsletter` | NEWS | `.il-newsletter` | Gray (#94a3b8) |
| `personal` | PERS | `.il-personal` | Blue (#60a5fa) |
| other | (from data label) | `.il-other` | Dim gray (#94a3b8) |

### Layout Rules
- Max 8 emails shown
- Each row: `[LABEL] Subject (truncated with ellipsis) / Analysis text`
- Subject: `font-weight: 600; font-size: 12px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis`
- Analysis: `font-size: 10px; color: var(--dim); margin-top: 1px`
- Timestamp after the box: `🕐 Last updated: <live JS date>`

## ⚠️ Deadlines Section

### Position
- Second element after Inbox
- Generated in `showHome()` from calendar data + courses.json

### HTML Classes
| Class | Element | Purpose |
|-------|---------|---------|
| `.dash-card.deadline-alert` | Container | Orange left border (#d97706), clickable |
| `.dl-body` | Scroll wrapper | **max-height: 160px**, overflow-y: auto |

### Data Sources
1. **Calendar events** with tags `DEADLINE` or `EXAM`
2. **courses.json** entries with deadline dates

### Display
- Urgency icons: 🔴 ≤1 day, 🟠 ≤3 days, 🟢 >3 days
- Format: `[icon] Label [EXAM badge?] / Date · DaysUntil`
- Sorted by days_until ascending
- Max 8 deadlines
- Timestamp after the box

## 📅 Calendar Section

### Position
- Third element after Deadlines
- Google Calendar iframe embed

### Rules
- Height: 500px (was 550px)
- Source: `auttapong.budhsombatwarakul@sasin.edu`
- Timezone: Asia/Bangkok
- Mode: MONTH view
- Do NOT replace with anything else — user wants Google Calendar specifically

## 🎨 CSS Conventions

### Design Tokens
```css
--bg: #0a0a0f      /* Page background */
--card: #131320     /* Card/dash-card background */
--border: #1e1e35   /* Card borders */
--text: #e2e8f0     /* Primary text */
--dim: #64748b      /* Dimmed/secondary text */
--gold: #fbbf24     /* Accent/highlight */
```

### Card Style
```css
.dash-card {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 20px;
}
```

## 🐛 Known Pitfalls

### Cron Duplication Problem
**The #1 recurring bug:** The cron inject script **appends** instead of **replaces**, causing:
- Duplicate `<script id="emailData">` blocks (data bloat)
- Duplicate CSS blocks (`.ib-body`, `.dl-body`, etc.)
- Duplicate `var EMAIL_DATA` parsers
- Duplicate `var inboxBox` code
- File size grows linearly with each cron run

**Fix:** The inject script MUST use `re.sub()` to remove old blocks before inserting new ones. Always grep for duplicates after every change.

### Field Name Stability
- EMAIL_DATA uses: `category`, `subject`, `analysis`
- The inboxBox JS MUST use these exact field names
- Do NOT change to `cat`, `label`, `summary`, `action` — these broke formatting before

### Box Heights
- Inbox `.ib-body`: max-height **140px** (smaller = more room for calendar)
- Deadlines `.dl-body`: max-height **160px**
- These are intentionally small — the user asked for small boxes with scrollbars

### JS Syntax
- Timestamps use string concatenation: `' + new Date().toLocaleString(...) + '`
- Do NOT use template literals (`` `${...}` ``) inside the showHome() template literal
- The `var inboxBox = "..."` uses escaped quotes inside the JS

### Never Remove
- Sidebar (with resizer)
- Calendar iframe
- Action buttons at bottom
- Course modals
- Mobile hamburger menu

## 📦 Deployment

- Source: `/root/sasin-cfoth-ai/index.html` (this repo)
- Live: Served via hub_server.py on new Contabo (13.140.145.203)
- Path on server: `/data/sasin-cfoth-ai/index.html`
- Push → GitHub → server pulls
- Restart NOT needed (hub_server.py reads file on each request)

## 🔗 Related Repos

- Sasin dashboard: `github.com/auttapongb/sasin-cfoth-ai`
- Requirements doc: `REQUIREMENTS.md` (this file)
- Capture server: `capture.sasin.cfoth.ai`
- Calendar: Google Calendar `auttapong.budhsombatwarakul@sasin.edu`
