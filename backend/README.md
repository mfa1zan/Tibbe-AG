# FastAPI Backend (Phase 2)

## Setup

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Fill `.env` with Neo4j and Groq credentials.

## Run

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## API

- `POST /api/chat`
  - Request: `{ "message": "...", "session_id": "optional" }`
  - Response: `{ "reply": "...", "provenance": ["Honey", "DrugX"] }`
