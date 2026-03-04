# Tibbe-AG

Biomedical chatbot project using a Knowledge Graph (Neo4j) + LLM reasoning (Groq/Llama), with a React + Tailwind frontend and FastAPI backend.

Repository: https://github.com/mfa1zan/Tibbe-AG.git

## Features

- React chat interface with typing indicator, provenance display, and responsive layout
- Dark/light mode, font selection, and primary color customization with local persistence
- FastAPI backend endpoint: `POST /api/chat`
- Frontend API adapter with runtime response validation and normalized error categories
- ESLint setup for React + Hooks code-quality enforcement
- Query preprocessing (normalization + synonym expansion)
- Neo4j KG retrieval with TTL caching
- Async LLM reasoning via Groq-compatible API
- Optional session-based multi-turn memory

## Tech Stack

- Frontend: React 18, Vite, TailwindCSS
- Frontend quality: ESLint 9 (flat config), React Hooks lint rules
- Backend: FastAPI, Neo4j Python Driver, HTTPX
- LLM: Groq Chat Completions API (Llama models)

## Project Structure

```text
Tibbe-AG/
├── backend/
│   ├── app/
│   │   ├── main.py
│   │   ├── config.py
│   │   ├── schemas.py
│   │   └── services/
│   │       ├── preprocess.py
│   │       ├── kg_service.py
│   │       ├── llm_service.py
│   │       └── orchestrator.py
│   ├── .env
│   ├── .env.example
│   └── requirements.txt
├── src/
│   ├── App.jsx
│   ├── api.js
│   ├── components/
│   ├── context/ThemeContext.jsx
│   └── *.css
├── eslint.config.js
├── .env
├── package.json
└── vite.config.js
```

## Prerequisites

- Node.js 18+
- Python 3.10+
- Neo4j database (local or Aura)
- Groq API key

## Setup

### 1) Clone

```bash
git clone https://github.com/mfa1zan/Tibbe-AG.git
cd Tibbe-AG
```

### 2) Frontend

```bash
npm install
```

Create/update root `.env`:

```env
VITE_USE_PLACEHOLDER_BOT=false
```

### 3) Backend

#### macOS / Linux

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt
```

#### Windows (Command Prompt)

```cmd
python -m venv .venv
.venv\Scripts\activate
pip install -r backend\requirements.txt
```

#### Windows (PowerShell)

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r backend\requirements.txt
```

> **Note (Windows PowerShell):** If you get an execution policy error, run `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser` first.

Create/update `backend/.env`:

```env
NEO4J_URI=neo4j+s://<your-db>.databases.neo4j.io
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=<your-neo4j-password>

GROQ_API_KEY=<your-groq-api-key>
GROQ_MODEL=llama-3.3-70b-versatile
GROQ_BASE_URL=https://api.groq.com/openai/v1

KG_CACHE_TTL_SECONDS=600
KG_CACHE_MAXSIZE=512
```

## Run (Recommended)

### Start full stack from root

```bash
npm run dev:full
```

This runs:

- Frontend on `http://localhost:5173`
- Backend on `http://localhost:8010`

### Or run separately

Terminal 1:

```bash
npm run dev:backend
```

Terminal 2:

```bash
npm run dev:frontend
```

## API

### POST `/api/chat`

Request:

```json
{
  "query": "What helps with fever?",
  "history": [
    { "role": "user", "content": "What helps with fever?" },
    { "role": "bot", "content": "..." }
  ]
}
```

Response:

```json
{
  "final_answer": "...",
  "evidence_strength": "weak|moderate|strong",
  "graph_paths_used": 0,
  "confidence_score": 0.0,
  "reasoning_trace": {}
}
```

Notes:

- Frontend maps API fields to UI message fields (`reply`, `confidenceScore`, etc.) in `src/api.js`.
- `structured_fields` may be emitted by backend pipelines but are not yet rendered in the frontend UI.

## Code Quality

Run lint checks:

```bash
npm run lint
```

Auto-fix lint issues where possible:

```bash
npm run lint:fix
```

Current lint config is defined in `eslint.config.js` and enforces:

- core JavaScript correctness rules
- React Hooks best practices
- no unused variables and consistent module hygiene

## Frontend Status (March 2026)

Implemented baseline:

- Runtime API payload validation + normalized frontend error handling
- Safer conversation history composition in chat send flow
- Project-wide ESLint command + configuration

Planned next modernization steps:

- Streaming responses in chat UI
- Persistent message/session storage
- Structured field rendering in chat bubbles
- Markdown rendering with sanitization
- Mobile input layout improvements and long-history performance optimization

## Build

```bash
npm run build
```

## Troubleshooting

### `POST http://localhost:5173/api/chat 500`

Usually means frontend proxy cannot reach backend or backend failed at runtime.

1. Ensure backend is running on port `8010`:

```bash
curl http://127.0.0.1:8010/health
```

2. If health fails, start backend with:

```bash
npm run dev:backend
```

3. Check backend terminal logs for Neo4j/Groq errors.
4. Verify `backend/.env` values (URI, keys, passwords).
5. Restart both frontend and backend after any `.env` changes.

### Theme not changing

1. Hard refresh browser (`Cmd+Shift+R` on macOS, `Ctrl+Shift+R` on Windows/Linux).
2. Check that `<html>` class toggles between `light` and `dark` in devtools.
3. Clear site data/localStorage if stale values exist.

## Security

- Never commit real API keys/passwords.
- Rotate any credentials if they were accidentally exposed.

## License

Add your preferred license here.
