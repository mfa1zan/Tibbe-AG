# Tibbe-AG

Biomedical chatbot project using a Knowledge Graph (Neo4j) + LLM reasoning (Groq/Llama), with a React + Tailwind frontend and FastAPI backend.

Repository: https://github.com/mfa1zan/Tibbe-AG.git

## Features

- GraphRAG pipeline: Entity Resolution → Intent Classification → Neo4j Query → LLM Answer → Judge Evaluation
- Hybrid entity matching: LLM extraction + fuzzy matching against 154 diseases, 59 ingredients, 972 drugs from DB
- Intent-specific answer generation (substitution, treatment, compounds, hadith, full chain)
- Answer quality judge (3C3H LLM-as-judge + NLP metrics)
- React chat interface with typing indicator, provenance display, and responsive layout
- Dark/light mode, font selection, and primary color customization with local persistence
- Markdown rendering (sanitized) for assistant responses
- Structured evidence field cards rendered in chat bubbles
- Local message persistence for chat session recovery

## Tech Stack

- **Frontend**: React 18, Vite, TailwindCSS, React Router, React Virtuoso
- **Backend**: FastAPI, Neo4j Python Driver, HTTPX, Pydantic Settings
- **LLM**: Groq Chat Completions API (Llama models)
- **Database**: Neo4j (Aura or local)

## Project Structure

```text
Tibbe-AG/
├── backend/                    ← New clean backend (v2)
│   ├── main.py                 ← FastAPI app + lifespan
│   ├── api/chat.py             ← /api/chat, /api/chat/debug
│   ├── core/
│   │   ├── config.py           ← Settings from .env
│   │   └── models.py           ← Request/Response schemas
│   ├── services/
│   │   ├── entity_resolver.py  ← Hybrid LLM + fuzzy entity matching
│   │   ├── query_router.py     ← Intent classification + query routing
│   │   ├── graph_service.py    ← Neo4j query execution
│   │   ├── llm_service.py      ← LLM calls (entity extraction + answer)
│   │   ├── judge_service.py    ← 3C3H / NLP answer evaluation
│   │   └── response_builder.py ← Final response assembly
│   ├── queries/
│   │   └── query_library.py    ← 9 predefined Cypher queries
│   ├── utils/helpers.py
│   ├── .env.example
│   └── requirements.txt
├── backend_legacy/             ← Old backend (preserved, not used)
├── src/                        ← React frontend
│   ├── App.jsx
│   ├── api.js
│   ├── components/
│   ├── pages/
│   └── *.css
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
git clone https://github.com/mfa1zan/Tibbe-AG
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

If your `package.json` still uses the Unix-style backend command, replace it with the Windows-friendly version below:

```json
"dev:backend": "python -m uvicorn backend.main:app --reload --host 0.0.0.0 --port 8010"
```

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

### 4) Configure Environment

Copy the example and fill in your credentials:

```bash
cp backend/.env.example backend/.env
```

Edit `backend/.env`:

```env
NEO4J_URI=neo4j+s://<your-db>.databases.neo4j.io
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=<your-neo4j-password>

GROQ_API_KEY=<your-groq-api-key>
GROQ_MODEL=llama-3.3-70b-versatile
GROQ_BASE_URL=https://api.groq.com/openai/v1

MODEL_INTENT=llama-3.1-8b-instant
MODEL_CHAT=llama-3.3-70b-versatile

ENABLE_JUDGE=true
LOG_LEVEL=INFO
```

## Run

### Start full stack from root

```bash
npm run dev:full
```

This runs:

- Frontend on `http://localhost:5173`
- Backend on `http://localhost:8010`

### Or run separately

Terminal 1 (backend):

```bash
npm run dev:backend
```

Terminal 2 (frontend):

```bash
npm run dev:frontend
```

### Or run backend directly

```bash
source .venv/bin/activate
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8010 --reload
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
  "evidence_strength": "weak|moderate|strong|none",
  "graph_paths_used": 8,
  "confidence_score": 0.88,
  "safety": null,
  "reasoning_trace": {
    "entity_detected": { "disease": "Fever", "ingredient": null, "drug": null },
    "intent": "disease_treatment",
    "query_used": "Disease → Ingredient → Hadith → Reference",
    "evidence_count": 8
  },
  "structured_fields": { "ingredients": ["Honey", "Water"] },
  "pipeline_debug_trace": null
}
```

### POST `/api/chat/debug`

Same as `/api/chat` but includes `pipeline_debug_trace` with full pipeline details (entities, Cypher query, DB timing, LLM timing, judge report).

### GET `/health`

Returns `{"status": "ok"}`.

## Pipeline

```
User Query
  → Entity Resolution (LLM + fuzzy matching against DB names)
  → Intent Classification (keyword + entity-aware)
  → Query Selection (deterministic routing to 1 of 9 Cypher queries)
  → Neo4j Execution (predefined query, case-insensitive)
  → Answer Generation (LLM with intent-specific prompt)
  → Judge Evaluation (3C3H or NLP metrics)
  → Response Assembly → Frontend
```

## Code Quality

```bash
npm run lint
npm run lint:fix
```

## Troubleshooting

### `POST http://localhost:5173/api/chat 500`

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

## Security

- Never commit real API keys/passwords.
- Rotate any credentials if they were accidentally exposed.

## License

MIT License

Copyright (c) 2026 Tibbe-AG
