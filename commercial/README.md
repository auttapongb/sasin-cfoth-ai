# Sasin AI Learning Toolkit — Commercial Layer

Production-grade billing, pipeline, and management services.

## Services

| Service | Port | Systemd | Description |
|---------|------|---------|-------------|
| Billing API | 8500 | sasin-billing | Stripe, orgs, usage tracking, admin dashboard |
| Pipeline | — | sasin-pipeline | Auto-ingest capture transcripts → DeepTutor KB |

## Deployment

```bash
# Billing
systemctl start sasin-billing

# Pipeline
systemctl start sasin-pipeline
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | /health | Health check |
| GET | /orgs | List organizations |
| GET | /orgs/{id} | Org detail + usage |
| POST | /orgs | Create organization |
| POST | /checkout | Stripe checkout session |
| POST | /portal | Stripe billing portal |
| POST | /stripe/webhook | Stripe event webhook |
| POST | /usage | Record usage metric |
| GET | /dashboard | Admin dashboard HTML |
