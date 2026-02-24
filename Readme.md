# Biomedical Knowledge Graph Chatbot UI (Phase 1)

Phase 1 delivers an MVP React chat interface that is ready to connect to a backend API.

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
