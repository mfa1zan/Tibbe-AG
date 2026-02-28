import logging

import httpx

logger = logging.getLogger(__name__)


class LLMService:
    def __init__(self, api_key: str, model: str, base_url: str) -> None:
        self._api_key = api_key
        self._default_model = model
        self._base_url = base_url.rstrip("/")

    async def generate_completion(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.2,
        model: str | None = None,
    ) -> str:
        payload = {
            "model": model or self._default_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
        }

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

        endpoint = f"{self._base_url}/chat/completions"

        async with httpx.AsyncClient(timeout=90.0) as client:
            response = await client.post(endpoint, json=payload, headers=headers)
            response.raise_for_status()
            body = response.json()

        reply = body["choices"][0]["message"]["content"].strip()
        logger.debug("LLM completion generated (%s chars)", len(reply))
        return reply

    async def generate_fallback_answer(self, user_query: str, model: str | None = None) -> str:
        system_prompt = (
            "You are PRO-MedGraph, a biomedical assistant. Provide a careful answer and clearly state uncertainty when needed."
        )
        user_prompt = f"User question: {user_query}"
        return await self.generate_completion(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.3,
            model=model,
        )
