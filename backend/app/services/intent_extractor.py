"""
PRO-MedGraph  —  Step 2: Intent Extraction & Entity Recognition
================================================================

This module wraps ``call_intent_model()`` from Step 1 (multi_model_service)
to produce **validated, structured JSON** that downstream stages (Cypher
generation, reasoning, etc.) can consume directly.

Output schema
-------------
.. code-block:: json

    {
      "intent_type": "ask_remedy | drug_interaction | food_remedy | symptom_check | dosage_info | general",
      "entities": [
        { "category": "drug | food | condition | symptom | dosage", "value": "<text>" }
      ],
      "confidence_score": 0.0 – 1.0
    }

Design notes
------------
* The system prompt explicitly forbids free-text output and instructs the
  model to return **only** the JSON object — no markdown fences, no
  explanation.
* After receiving the raw LLM text we:
  1. Strip accidental Markdown code fences (```json … ```)
  2. Locate the first ``{ … }`` block (regex fallback)
  3. Deserialize with ``json.loads``
  4. Validate against the expected schema
  5. Retry up to ``MAX_RETRIES`` on failure before raising
* A battery of test queries is included at the bottom so the module can be
  executed standalone (``python -m app.services.intent_extractor``).
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any

# Step 1 wrapper — sends the query to MODEL_INTENT via Groq.
from app.services.multi_model_service import call_intent_model

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────

# How many times we re-prompt the model when it returns invalid JSON.
MAX_RETRIES: int = 2

# Accepted values for "intent_type".
VALID_INTENT_TYPES: set[str] = {
    "ask_remedy",
    "drug_interaction",
    "food_remedy",
    "symptom_check",
    "dosage_info",
    "general",
}

# Accepted values for each entity's "category".
VALID_ENTITY_CATEGORIES: set[str] = {
    "drug",
    "food",
    "condition",
    "symptom",
    "dosage",
}

# ── System prompt ──────────────────────────────────────────────────────────────
# This prompt is injected as the *system* message every time we call the intent
# model.  It overrides the generic prompt in multi_model_service because we need
# tighter control over output format for downstream Cypher generation.
#
# Key design choices:
#   • "Return ONLY …" — suppresses explanatory text.
#   • Explicit enum lists — constrains hallucination surface.
#   • "If uncertain, use 'general'" — safe fallback.

SYSTEM_PROMPT: str = (
    "You are the intent-extraction engine of PRO-MedGraph, a biomedical "
    "knowledge-graph system rooted in Prophetic (Tibb-e-Nabawi) medicine.\n\n"
    #
    "Given a user query, return ONLY a single JSON object — no markdown "
    "fences, no explanation, no extra text.\n\n"
    #
    "JSON schema:\n"
    "{\n"
    '  "intent_type": "<one of: ask_remedy, drug_interaction, food_remedy, '
    'symptom_check, dosage_info, general>",\n'
    '  "entities": [\n'
    '    { "category": "<drug | food | condition | symptom | dosage>", '
    '"value": "<extracted text>" }\n'
    "  ],\n"
    '  "confidence_score": <float between 0.0 and 1.0>\n'
    "}\n\n"
    #
    "Rules:\n"
    "1. Extract EVERY biomedical entity present in the query.\n"
    "2. Map each entity to exactly one category.\n"
    '3. If the query is a greeting or off-topic, use intent_type "general" '
    "with an empty entities list.\n"
    "4. confidence_score reflects how certain you are about the extraction.\n"
    "5. Do NOT hallucinate entities that are not explicitly in the query.\n"
    "6. Return valid JSON only — no trailing commas, no comments."
)

# ── JSON extraction helpers ────────────────────────────────────────────────────


def _strip_markdown_fences(text: str) -> str:
    """
    Remove optional Markdown code fences that LLMs sometimes wrap around JSON.

    Handles: ```json { … } ``` and bare ``` { … } ```.
    """
    text = text.strip()
    # Remove opening fence (with optional language tag)
    text = re.sub(r"^```(?:json)?\s*\n?", "", text)
    # Remove closing fence
    text = re.sub(r"\n?```\s*$", "", text)
    return text.strip()


def _extract_first_json_object(text: str) -> str | None:
    """
    Locate the first top-level ``{ … }`` block using brace-depth tracking.

    This is more robust than a simple regex when the model accidentally
    appends trailing text after the JSON.
    """
    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def _parse_json(raw: str) -> dict[str, Any]:
    """
    Try progressively looser strategies to obtain a Python dict from raw LLM
    output.

    Raises ``ValueError`` if nothing works.
    """
    # 1. Direct parse.
    cleaned = _strip_markdown_fences(raw)
    try:
        obj = json.loads(cleaned)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass

    # 2. Extract the first JSON object (handles trailing garbage).
    block = _extract_first_json_object(cleaned)
    if block:
        try:
            obj = json.loads(block)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass

    raise ValueError(f"Could not parse JSON from LLM output: {raw[:300]}")


# ── Schema validation ─────────────────────────────────────────────────────────


class IntentValidationError(ValueError):
    """Raised when the parsed JSON does not match the expected schema."""


def _validate_intent_json(data: dict[str, Any]) -> dict[str, Any]:
    """
    Validate and normalise the parsed intent JSON.

    Checks
    ------
    * ``intent_type`` is one of the allowed enum values.
    * ``entities`` is a list of dicts, each with a valid ``category`` and
      a non-empty ``value``.
    * ``confidence_score`` is a float in [0, 1].

    Returns
    -------
    dict
        The (possibly normalised) intent JSON.

    Raises
    ------
    IntentValidationError
        If any check fails.
    """
    errors: list[str] = []

    # ── intent_type ────────────────────────────────────────────────────────
    intent_type = data.get("intent_type")
    if not isinstance(intent_type, str) or intent_type not in VALID_INTENT_TYPES:
        errors.append(
            f"intent_type '{intent_type}' is invalid; expected one of {sorted(VALID_INTENT_TYPES)}"
        )

    # ── entities ───────────────────────────────────────────────────────────
    entities = data.get("entities")
    if not isinstance(entities, list):
        errors.append("'entities' must be a JSON array")
    else:
        for idx, ent in enumerate(entities):
            if not isinstance(ent, dict):
                errors.append(f"entities[{idx}] is not an object")
                continue
            cat = ent.get("category")
            val = ent.get("value")
            if not isinstance(cat, str) or cat not in VALID_ENTITY_CATEGORIES:
                errors.append(
                    f"entities[{idx}].category '{cat}' invalid; "
                    f"expected one of {sorted(VALID_ENTITY_CATEGORIES)}"
                )
            if not isinstance(val, str) or not val.strip():
                errors.append(f"entities[{idx}].value must be a non-empty string")

    # ── confidence_score ───────────────────────────────────────────────────
    conf = data.get("confidence_score")
    try:
        conf_float = float(conf)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        errors.append(f"confidence_score '{conf}' is not a valid number")
        conf_float = 0.0

    if conf_float < 0 or conf_float > 1:
        errors.append(f"confidence_score {conf_float} out of range [0, 1]")

    if errors:
        raise IntentValidationError("; ".join(errors))

    # Normalise confidence to a float (model sometimes returns a string).
    data["confidence_score"] = round(conf_float, 4)
    return data


# ═══════════════════════════════════════════════════════════════════════════════
#  Public API
# ═══════════════════════════════════════════════════════════════════════════════


async def extract_intent(query: str) -> dict[str, Any]:
    """
    Extract structured intent + entities from a natural-language query.

    Pipeline
    --------
    1. Call ``call_intent_model`` (Step 1) with the specialised
       ``SYSTEM_PROMPT`` that forces JSON-only output.
    2. Parse the raw LLM string into a Python dict.
    3. Validate the dict against the expected schema.
    4. If parsing or validation fails, retry up to ``MAX_RETRIES`` times.
    5. Return the validated dict, or raise on persistent failure.

    Parameters
    ----------
    query : str
        The raw user query (e.g. *"Does honey interact with metformin?"*).

    Returns
    -------
    dict
        Validated intent JSON conforming to the schema above.

    Raises
    ------
    IntentValidationError
        If all retry attempts produce invalid JSON.
    RuntimeError
        If the underlying LLM call fails for non-schema reasons.
    """
    last_error: Exception | None = None

    for attempt in range(1, MAX_RETRIES + 2):  # +2 → first try + MAX_RETRIES retries
        try:
            # ── 1. Call the intent model via Step 1 wrapper ────────────────
            # NOTE: call_intent_model has its own generic system prompt.  We
            # override it here by importing the low-level _call_groq and using
            # our stricter SYSTEM_PROMPT.  This keeps multi_model_service
            # untouched while giving us tighter control.
            from app.services.multi_model_service import _call_groq, config

            raw_response: str = await _call_groq(
                model=config.intent,
                system_prompt=SYSTEM_PROMPT,
                user_prompt=query,
                temperature=0.0,
            )

            logger.debug(
                "Intent model attempt %d/%d raw response: %s",
                attempt, MAX_RETRIES + 1, raw_response[:300],
            )

            # ── 2. Parse the raw string into a dict ────────────────────────
            parsed = _parse_json(raw_response)

            # ── 3. Validate against the schema ─────────────────────────────
            validated = _validate_intent_json(parsed)

            logger.info(
                "Intent extracted: type=%s entities=%d confidence=%.2f",
                validated.get("intent_type"),
                len(validated.get("entities", [])),
                validated.get("confidence_score", 0),
            )
            return validated

        except (ValueError, IntentValidationError) as exc:
            # Schema / parse failure — worth retrying.
            last_error = exc
            logger.warning(
                "Intent extraction attempt %d/%d failed: %s",
                attempt, MAX_RETRIES + 1, exc,
            )

        except Exception as exc:
            # Network / auth / unexpected — propagate immediately.
            logger.exception("Intent extraction hit a non-retryable error")
            raise RuntimeError(f"Intent extraction failed: {exc}") from exc

    # All attempts exhausted.
    raise IntentValidationError(
        f"Intent extraction failed after {MAX_RETRIES + 1} attempts. "
        f"Last error: {last_error}"
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  Synchronous convenience wrapper
# ═══════════════════════════════════════════════════════════════════════════════


def extract_intent_sync(query: str) -> dict[str, Any]:
    """
    Blocking wrapper around :func:`extract_intent` for use in non-async
    code paths (e.g. scripts, tests, orchestrator sync entry points).
    """
    return asyncio.run(extract_intent(query))


# ═══════════════════════════════════════════════════════════════════════════════
#  Standalone test harness
# ═══════════════════════════════════════════════════════════════════════════════
#  Run with:   python -m app.services.intent_extractor
#  (from the backend/ directory with .env in place)

# Diverse sample queries covering the request's six scenarios.
_TEST_QUERIES: list[dict[str, str]] = [
    # ── Drug interactions ──────────────────────────────────────────────────
    {
        "label": "Drug interaction (simple)",
        "query": "Does honey interact with metformin?",
    },
    {
        "label": "Drug interaction (multi-drug)",
        "query": "Can I take aspirin and warfarin together with ginger tea?",
    },
    {
        "label": "Drug interaction (side-effects)",
        "query": "What are the side effects of combining ibuprofen with turmeric?",
    },
    # ── Food / ingredient remedies ─────────────────────────────────────────
    {
        "label": "Food remedy (Prophetic)",
        "query": "Is black seed oil effective for asthma?",
    },
    {
        "label": "Food remedy (general)",
        "query": "What foods help lower cholesterol naturally?",
    },
    {
        "label": "Food remedy (preparation)",
        "query": "How should I prepare Talbina for stomach problems?",
    },
    # ── Disease / condition specific ───────────────────────────────────────
    {
        "label": "Disease query (diabetes)",
        "query": "What are the Prophetic medicine treatments for diabetes?",
    },
    {
        "label": "Disease query (kidney stones)",
        "query": "Which natural ingredients treat kidney stones?",
    },
    {
        "label": "Symptom check",
        "query": "I have a persistent cough and mild fever — what should I look into?",
    },
    # ── Dosage questions ───────────────────────────────────────────────────
    {
        "label": "Dosage query",
        "query": "What is the recommended dosage of Nigella sativa for allergies?",
    },
    # ── General / off-topic ────────────────────────────────────────────────
    {
        "label": "General greeting",
        "query": "Hello, how are you?",
    },
    {
        "label": "General thanks",
        "query": "Thanks for the help!",
    },
]


async def _run_tests() -> None:
    """Execute the test battery and print results."""
    import sys

    # Minimal logging so we see warnings / retries.
    logging.basicConfig(level=logging.INFO, stream=sys.stdout, format="%(levelname)s %(message)s")

    passed = 0
    failed = 0
    total = len(_TEST_QUERIES)

    print("=" * 72)
    print("  PRO-MedGraph · Intent Extraction Test Harness")
    print("=" * 72)

    for tc in _TEST_QUERIES:
        label = tc["label"]
        query = tc["query"]
        print(f"\n── {label} ──")
        print(f"   Query:  {query}")

        try:
            result = await extract_intent(query)
            print(f"   Intent: {result['intent_type']}")
            print(f"   Entities ({len(result['entities'])}):")
            for ent in result["entities"]:
                print(f"      • [{ent['category']}] {ent['value']}")
            print(f"   Confidence: {result['confidence_score']}")
            passed += 1
        except Exception as exc:
            print(f"   ✗ FAILED: {exc}")
            failed += 1

    print("\n" + "=" * 72)
    print(f"  Results:  {passed}/{total} passed, {failed}/{total} failed")
    print("=" * 72)


if __name__ == "__main__":
    asyncio.run(_run_tests())


# ── Module exports ─────────────────────────────────────────────────────────────

__all__ = [
    "extract_intent",
    "extract_intent_sync",
    "IntentValidationError",
    "VALID_INTENT_TYPES",
    "VALID_ENTITY_CATEGORIES",
]
