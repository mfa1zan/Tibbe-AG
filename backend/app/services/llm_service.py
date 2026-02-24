import logging

import httpx

logger = logging.getLogger(__name__)


class LLMService:
    def __init__(self, api_key: str, model: str, base_url: str) -> None:
        self._api_key = api_key
        self._model = model
        self._base_url = base_url.rstrip("/")

    async def generate_reply(
        self,
        user_message: str,
        kg_context: str,
        session_history: list[dict[str, str]] | None = None,
    ) -> str:
        history_text = ""
        if session_history:
            turns = [f"{item['role']}: {item['content']}" for item in session_history[-6:]]
            history_text = "\nRecent conversation:\n" + "\n".join(turns)

        prompt = (
            "You are a biomedical reasoning assistant. "
            "Use the provided knowledge graph context first, and if context is sparse, state uncertainty clearly.\n\n"
            f"Knowledge graph context:\n{kg_context}\n"
            f"{history_text}\n"
            f"Question: {user_message}\n"
            "Answer in a concise, clinically careful tone."
        )

        payload = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": "You are a biomedical reasoning assistant."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.2,
        }

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

        endpoint = f"{self._base_url}/chat/completions"

        async with httpx.AsyncClient(timeout=45.0) as client:
            response = await client.post(endpoint, json=payload, headers=headers)
            response.raise_for_status()
            body = response.json()

        reply = body["choices"][0]["message"]["content"].strip()
        logger.debug("LLM reply generated (%s chars)", len(reply))
        return reply
