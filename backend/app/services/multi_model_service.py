"""
PRO-MedGraph Multi-Model Service
=================================

Provides dedicated Groq API wrapper functions for each stage of the
GraphRAG pipeline.  Every wrapper reads its model ID from the `.env`
file so models can be swapped without touching code.

Environment variables consumed
------------------------------
GROQ_API_KEY      – Groq API secret key (shared by all wrappers)
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
#  Stage-specific wrapper functions
# ═══════════════════════════════════════════════════════════════════════════════


async def call_intent_model(query: str) -> str:
    """
    **Stage: Intent / Entity Extraction**

    Accepts the raw user query and returns a structured JSON string
    identifying the user's intent and extracted biomedical entities.

    Uses MODEL_INTENT (typically a fast, small model like ``llama-3.1-8b-instant``).
    """
    system_prompt = (
        "You are an intent classifier for a biomedical GraphRAG system called PRO-MedGraph.\n"
        "Given a user query, return ONLY a JSON object with these keys:\n"
        '  "intent": one of ["disease_query", "ingredient_query", "drug_query", "general"]\n'
        '  "disease": extracted disease name or null\n'
        '  "ingredient": extracted ingredient name or null\n'
        '  "drug": extracted drug name or null\n'
        '  "confidence": float 0-1 for extraction confidence\n'
        "Do NOT include any extra text outside the JSON."
    )

    try:
        return await _call_groq(
            model=config.intent,
            system_prompt=system_prompt,
            user_prompt=query,
            temperature=0.0,  # deterministic for classification
        )
    except Exception:
        logger.exception("call_intent_model failed for query: %s", query[:120])
        raise


async def call_kg_model(intent_json: str | dict) -> str:
    """
    **Stage: Knowledge-Graph Cypher Generation**

    Accepts the parsed intent JSON (from ``call_intent_model``) and returns
    a Neo4j Cypher query that traverses the PRO-MedGraph knowledge graph.

    Uses MODEL_KG (tuned for structured query generation).
    """
    # Normalise input – accept both raw string and dict.
    if isinstance(intent_json, dict):
        intent_json = json.dumps(intent_json, ensure_ascii=False)

    system_prompt = (
        "You are a Neo4j Cypher expert for the PRO-MedGraph biomedical knowledge graph.\n"
        "The graph has these node labels: Disease, Ingredient, ChemicalCompound, "
        "DrugChemicalCompound, Drug, HadithReference, Prophet, Book, Narrator.\n"
        "Relationship types: CURES, CONTAINS, IS_IDENTICAL_TO, IS_LIKELY_EQUIVALENT_TO, "
        "IS_WEAK_MATCH_TO, REFERENCES, NARRATED_BY, COMPILED_IN, MENTIONED_BY.\n\n"
        "Given the following intent JSON, generate a Cypher query that retrieves all "
        "relevant subgraph data.  Return ONLY the Cypher query, no explanation."
    )

    try:
        return await _call_groq(
            model=config.kg,
            system_prompt=system_prompt,
            user_prompt=f"Intent JSON:\n{intent_json}",
            temperature=0.0,  # deterministic for query generation
        )
    except Exception:
        logger.exception("call_kg_model failed for intent: %s", intent_json[:200])
        raise


async def call_reasoner_model(evidence_json: str | dict) -> str:
    """
    **Stage: Draft Answer / Causal Reasoning (A0)**

    Accepts structured evidence (graph paths, causal scores, dosage data)
    and produces a comprehensive, graph-grounded draft answer following
    the chain-of-thought protocol.

    Uses MODEL_REASONER (large model like ``llama-3.3-70b-versatile``).
    """
    if isinstance(evidence_json, dict):
        evidence_json = json.dumps(evidence_json, ensure_ascii=False, indent=2)

    system_prompt = (
        "You are PRO-MedGraph, a biomedical and faith-aligned assistant.\n"
        "Use ONLY the provided structured graph reasoning evidence.\n"
        "Do NOT hallucinate entities, studies, or claims.\n\n"
        "Follow this chain-of-thought structure:\n"
        "1. Identify the disease / ingredient / drug from the evidence\n"
        "2. List relevant entities linked via KG relationships\n"
        "3. Trace the biochemical pathway (Ingredient → Compound → Drug)\n"
        "4. Explain mapping strengths (IDENTICAL / LIKELY / WEAK)\n"
        "5. Include Hadith references with respectful, non-exclusivist framing\n"
        "6. Include dosage / preparation notes if available\n"
        "7. State uncertainty level and recommend consulting a healthcare professional"
    )

    try:
        return await _call_groq(
            model=config.reasoner,
            system_prompt=system_prompt,
            user_prompt=f"Structured Evidence:\n{evidence_json}",
            temperature=0.1,  # low temperature for factual reasoning
            timeout=_REASONING_TIMEOUT,  # extra time for large model
        )
    except Exception:
        logger.exception("call_reasoner_model failed")
        raise


async def call_validator_model(draft_answer: str, evidence_json: str | dict) -> str:
    """
    **Stage: Answer Validation / Safety Check (Af)**

    Accepts the draft A0 answer and the original evidence, then validates
    for hallucinations, adds safety disclaimers, and ensures faith alignment.

    Uses MODEL_VALIDATOR (safety / instruction-tuned model).
    """
    if isinstance(evidence_json, dict):
        evidence_json = json.dumps(evidence_json, ensure_ascii=False, indent=2)

    system_prompt = (
        "You are PRO-MedGraph Validator (Af).\n"
        "Your job is to validate the draft answer against the graph evidence ONLY.\n\n"
        "Rules:\n"
        "- Remove any claim not grounded in the provided evidence\n"
        "- Add uncertainty language for WEAK mapping links\n"
        "- Add concise medical safety disclaimers where needed\n"
        "- Ensure respectful Hadith framing (no miracle / guarantee language)\n"
        "- Ensure the answer recommends consulting a healthcare professional\n"
        "- Do NOT add unsupported claims\n\n"
        "Return the final validated answer only."
    )

    user_prompt = (
        f"Draft Answer (A0):\n{draft_answer}\n\n"
        f"Graph Evidence:\n{evidence_json}"
    )

    try:
        return await _call_groq(
            model=config.validator,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.1,  # low temperature for deterministic validation
        )
    except Exception:
        logger.exception("call_validator_model failed")
        raise


async def call_chat_model(user_input: str) -> str:
    """
    **Stage: General Conversational Chat**

    Handles non-biomedical user messages (greetings, thanks, clarification
    requests) with a friendly, helpful tone.

    Uses MODEL_CHAT (general-purpose conversational model).
    """
    system_prompt = (
        "You are PRO-MedGraph, a helpful biomedical and Islamic-medicine assistant.\n"
        "Respond naturally and briefly to general user messages.\n"
        "If the user asks a medical question without specific details,\n"
        "ask a clarifying follow-up.  Avoid definitive medical claims\n"
        "unless backed by evidence from the knowledge graph."
    )

    try:
        return await _call_groq(
            model=config.chat,
            system_prompt=system_prompt,
            user_prompt=user_input,
            temperature=0.3,  # slightly creative for natural chat
        )
    except Exception:
        logger.exception("call_chat_model failed for input: %s", user_input[:120])
        raise


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
    "call_intent_model",
    "call_kg_model",
    "call_reasoner_model",
    "call_validator_model",
    "call_chat_model",
    "get_model_map",
    "config",
]
