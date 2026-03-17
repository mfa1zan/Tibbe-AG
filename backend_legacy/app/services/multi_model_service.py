"""
PRO-MedGraph Multi-Model Service
=================================

Provides the low-level ``_call_groq`` helper and a lazy ``config`` object
that reads per-stage model IDs from ``.env``.  Each pipeline module
(intent_extractor, cypher_query_generator, clinical_reasoner, etc.)
imports ``_call_groq`` and ``config`` directly and supplies its own
system prompt.

Environment variables consumed
------------------------------
GROQ_API_KEY      – Groq API secret key (shared by all stages)
GROQ_BASE_URL     – Base URL for the Groq chat-completions endpoint
MODEL_INTENT      – Fast model for intent / query parsing
MODEL_KG          – Model for Neo4j Cypher query generation
MODEL_REASONER    – Large model for draft answer / reasoning (A0)
MODEL_VALIDATOR   – Model for validation / safety / mechanistic check (Af)
MODEL_CHAT        – Conversational chat model for general queries
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

import httpx
from dotenv import load_dotenv

# ── Load .env ──────────────────────────────────────────────────────────────────
# find_dotenv() walks up from cwd; explicit path keeps things deterministic when
# the module is imported from different working directories.

_ENV_LOADED = False


def _ensure_env() -> None:
    """Load the .env file exactly once (idempotent)."""
    global _ENV_LOADED
    if _ENV_LOADED:
        return
    # Try the backend/.env first (standard layout), fall back to cwd.
    from pathlib import Path

    backend_env = Path(__file__).resolve().parents[2] / ".env"
    load_dotenv(dotenv_path=backend_env if backend_env.exists() else ".env")
    _ENV_LOADED = True


logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════════
#  Configuration helpers
# ═══════════════════════════════════════════════════════════════════════════════


class _ModelConfig:
    """
    Lazy accessor for model IDs and the shared API key / base URL.

    Values are read **once** from the environment on first access and cached
    for the process lifetime.  A ``MissingModelError`` is raised if a
    required variable is absent so the caller gets a clear, actionable
    message rather than a cryptic ``None`` somewhere downstream.
    """

    class MissingModelError(EnvironmentError):
        """Raised when a required model env-var is not set."""

    def __init__(self) -> None:
        self._cache: dict[str, str] = {}

    # -- private helpers ----------------------------------------------------

    def _get(self, key: str) -> str:
        """Return env var *key*, raising ``MissingModelError`` if unset."""
        if key not in self._cache:
            _ensure_env()
            value = os.getenv(key)
            if not value:
                raise self.MissingModelError(
                    f"Environment variable '{key}' is not set.  "
                    f"Add it to your .env file or export it before starting the app."
                )
            self._cache[key] = value
        return self._cache[key]

    # -- public properties --------------------------------------------------

    @property
    def api_key(self) -> str:
        """Groq API key (shared across all models)."""
        return self._get("GROQ_API_KEY")

    @property
    def base_url(self) -> str:
        """Groq base URL for the chat-completions endpoint."""
        return self._get("GROQ_BASE_URL").rstrip("/")

    @property
    def intent(self) -> str:
        """Model ID for intent / query parsing."""
        return self._get("MODEL_INTENT")

    @property
    def kg(self) -> str:
        """Model ID for Neo4j Cypher generation."""
        return self._get("MODEL_KG")

    @property
    def reasoner(self) -> str:
        """Model ID for draft answer / causal reasoning (A0)."""
        return self._get("MODEL_REASONER")

    @property
    def validator(self) -> str:
        """Model ID for answer validation / safety check (Af)."""
        return self._get("MODEL_VALIDATOR")

    @property
    def chat(self) -> str:
        """Model ID for general conversational chat."""
        return self._get("MODEL_CHAT")


# Singleton – created once, reused everywhere.
config = _ModelConfig()

# ═══════════════════════════════════════════════════════════════════════════════
#  Low-level Groq API caller
# ═══════════════════════════════════════════════════════════════════════════════

# Default HTTP timeout (seconds).  Reasoning models may need more headroom.
_DEFAULT_TIMEOUT = 45.0
_REASONING_TIMEOUT = 90.0


async def _call_groq(
    *,
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.2,
    timeout: float = _DEFAULT_TIMEOUT,
) -> str:
    """
    Send a chat-completion request to the Groq API.

    Parameters
    ----------
    model : str
        The model ID to use (e.g. ``llama-3.3-70b-versatile``).
    system_prompt : str
        System-level instruction for the model.
    user_prompt : str
        The user-facing content / context.
    temperature : float
        Sampling temperature (0 → deterministic, 1 → creative).
    timeout : float
        HTTP timeout in seconds.

    Returns
    -------
    str
        The assistant's reply text, stripped of leading/trailing whitespace.

    Raises
    ------
    httpx.HTTPStatusError
        If the API returns a non-2xx status code.
    """
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
    }

    headers = {
        "Authorization": f"Bearer {config.api_key}",
        "Content-Type": "application/json",
    }

    endpoint = f"{config.base_url}/chat/completions"

    logger.debug("Groq request → model=%s temp=%.2f endpoint=%s", model, temperature, endpoint)

    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.post(endpoint, json=payload, headers=headers)
        response.raise_for_status()
        body = response.json()

    reply: str = body["choices"][0]["message"]["content"].strip()
    logger.debug("Groq response ← model=%s chars=%d", model, len(reply))
    return reply


# ═══════════════════════════════════════════════════════════════════════════════
#  Utility: list configured models (useful for health-checks / logging)
# ═══════════════════════════════════════════════════════════════════════════════


def get_model_map() -> dict[str, str]:
    """
    Return a dict of ``{stage_name: model_id}`` for every configured model.

    Safe to call at startup — catches ``MissingModelError`` per variable and
    marks missing ones as ``"<NOT SET>"``.
    """
    stages = {
        "intent": "MODEL_INTENT",
        "kg": "MODEL_KG",
        "reasoner": "MODEL_REASONER",
        "validator": "MODEL_VALIDATOR",
        "chat": "MODEL_CHAT",
    }
    result: dict[str, str] = {}
    _ensure_env()
    for stage, env_key in stages.items():
        result[stage] = os.getenv(env_key, "<NOT SET>")
    return result


# ═══════════════════════════════════════════════════════════════════════════════
#  Public API
# ═══════════════════════════════════════════════════════════════════════════════

__all__ = [
    "_call_groq",
    "get_model_map",
    "config",
]
