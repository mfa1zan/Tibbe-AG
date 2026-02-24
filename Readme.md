# Biomedical Knowledge Graph Chatbot UI (Phase 1)

Phase 1 delivers an MVP React chat interface that is ready to connect to a backend API.

## Phase 2 (FastAPI + Neo4j + Groq)

Phase 2 adds a production-oriented backend in `backend/`:

- FastAPI endpoint: `POST /api/chat`
- User question preprocessing (normalization + synonym expansion)
- Neo4j KG retrieval with per-keyword TTL caching
- Async LLM reasoning via Groq-compatible OpenAI endpoint
- Graceful fallback reply on KG/LLM failure
- Request/response logging for debugging
- Optional session memory via `session_id`

### Run full stack locally

Backend (Terminal 1):

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Frontend (Terminal 2):

```bash
npm install
npm run dev
```

Vite proxies `/api/*` to `http://localhost:8000` in development.

## Completed in Phase 1

- React 18 app using functional components and hooks
- TailwindCSS styling for responsive chat layout
- Chat components split into:
	- `src/App.jsx`
	- `src/components/ChatHistory.jsx`
	- `src/components/ChatBubble.jsx`
	- `src/components/ChatInput.jsx`
	- `src/api.js`
- Backend integration via `POST /api/chat` with payload `{ "message": "..." }`
- Expects response shape `{ "reply": "..." }`
- Typing indicator while bot reply is pending
- Graceful API error handling

## Run locally

```bash
npm install
npm run dev
```

Open the local Vite URL shown in terminal.

## Build

```bash
npm run build
```

## Optional placeholder mode

Set this in a local `.env` file to bypass backend during UI testing:

```bash
VITE_USE_PLACEHOLDER_BOT=true
```

The app then returns:

`Hello, I am your biomedical assistant.`
