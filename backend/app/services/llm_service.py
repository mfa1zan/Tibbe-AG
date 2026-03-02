import asyncio
import logging

import httpx

logger = logging.getLogger(__name__)

_MAX_RETRIES = 4
_INITIAL_BACKOFF = 1.0  # seconds
_MAX_ACCEPTABLE_RETRY_AFTER = 120.0  # seconds


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

        last_exc: Exception | None = None
        for attempt in range(_MAX_RETRIES):
            try:
                async with httpx.AsyncClient(timeout=90.0) as client:
                    response = await client.post(endpoint, json=payload, headers=headers)

                    if response.status_code == 429:
                        # Respect Retry-After header if present, else exponential backoff
                        retry_after = response.headers.get("retry-after")
                        retry_after_seconds: float | None = None
                        if retry_after:
                            try:
                                retry_after_seconds = float(retry_after)
                            except ValueError:
                                retry_after_seconds = None
                        if (
                            retry_after_seconds is not None
                            and retry_after_seconds > _MAX_ACCEPTABLE_RETRY_AFTER
                        ):
                            logger.error(
                                "LLM hard rate-limited by provider (retry-after=%ss). Failing fast.",
                                int(retry_after_seconds),
                            )
                            raise RuntimeError(
                                f"LLM provider rate-limited (retry-after={int(retry_after_seconds)}s)"
                            )

                        if retry_after_seconds is not None:
                            wait = min(retry_after_seconds, 15.0)
                        else:
                            wait = _INITIAL_BACKOFF * (2 ** attempt)
                        wait = min(wait, 15.0)  # cap at 15s
                        logger.warning(
                            "LLM 429 rate-limited (attempt %d/%d), retrying in %.1fs  [retry-after=%s]",
                            attempt + 1, _MAX_RETRIES, wait,
                            retry_after or "none",
                        )
                        await asyncio.sleep(wait)
                        continue

                    response.raise_for_status()
                    body = response.json()

                reply = body["choices"][0]["message"]["content"].strip()
                logger.debug("LLM completion generated (%s chars)", len(reply))
                return reply

            except httpx.HTTPStatusError as exc:
                if exc.response.status_code == 429:
                    wait = min(_INITIAL_BACKOFF * (2 ** attempt), 15.0)
                    logger.warning(
                        "LLM 429 (attempt %d/%d), retrying in %.1fs",
                        attempt + 1, _MAX_RETRIES, wait,
                    )
                    last_exc = exc
                    await asyncio.sleep(wait)
                    continue
                raise
            except Exception:
                raise

        # All retries exhausted
        logger.error("LLM rate-limit retries exhausted after %d attempts", _MAX_RETRIES)
        if last_exc:
            raise last_exc
        raise RuntimeError("LLM request failed after retries")

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
