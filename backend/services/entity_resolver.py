"""Entity Resolver — hybrid LLM + fuzzy matching for entity extraction.

Pre-fetches all disease, ingredient, and drug names from Neo4j on startup,
then uses this list to:
1. Provide the LLM with known entity names for accurate extraction
2. Fuzzy-match extracted entities to correct DB-stored names
3. Handle misspellings like "henna" → "Heena", "fever" → "Fever"
"""

from __future__ import annotations

import logging
import re
import time
from difflib import SequenceMatcher
from functools import lru_cache
from typing import Any

from neo4j.exceptions import SessionExpired

from backend.core.config import get_settings
from backend.services import llm_service
from backend.services.graph_service import _get_driver

logger = logging.getLogger(__name__)

FUZZY_THRESHOLD = 0.65
MIN_CHUNK_CHARS = 3

# ── Entity Name Cache ────────────────────────────────────────────────────────

_entity_cache: dict[str, list[str]] = {}
_cache_timestamp: float = 0.0
_CACHE_TTL = 600  # 10 minutes


def _normalize(text: str) -> str:
    lowered = text.lower().strip()
    lowered = re.sub(r"[^\w\s]", " ", lowered)
    lowered = re.sub(r"\s+", " ", lowered)
    return lowered.strip()


def _fetch_all_entity_names() -> dict[str, list[str]]:
    """Fetch all disease, ingredient, drug, and chemical compound names from Neo4j."""
    global _entity_cache, _cache_timestamp

    now = time.time()
    if _entity_cache and (now - _cache_timestamp) < _CACHE_TTL:
        return _entity_cache

    driver = _get_driver()
    result: dict[str, list[str]] = {"Disease": [], "Ingredient": [], "Drug": [], "ChemicalCompound": []}

    def _run_name_query(label: str) -> list:
        with driver.session() as session:
            return list(session.run(
                f"MATCH (n:{label}) WHERE n.name IS NOT NULL "
                f"RETURN DISTINCT n.name AS name"
            ))

    for label in result:
        try:
            try:
                rows = _run_name_query(label)
            except SessionExpired:
                logger.warning("Neo4j session expired while loading %s names; retrying once", label)
                rows = _run_name_query(label)

            result[label] = [
                row["name"] for row in rows
                if isinstance(row.get("name"), str) and row["name"].strip()
            ]
            logger.info("Loaded %d %s names from Neo4j", len(result[label]), label)
        except Exception:
            logger.exception("Failed to fetch %s names", label)

    # Only cache if at least the core labels loaded successfully
    if not result["Disease"] and not result["Ingredient"]:
        logger.warning("Entity cache not updated — core labels failed to load")
        return _entity_cache if _entity_cache else result

    _entity_cache = result
    _cache_timestamp = now
    return result


def get_entity_lists() -> dict[str, list[str]]:
    """Public accessor for cached entity name lists."""
    return _fetch_all_entity_names()


def _fuzzy_match(query: str, candidates: list[str]) -> tuple[str | None, float]:
    """Find the best fuzzy match for query against candidate names.

    Returns (matched_name, confidence_score).
    """
    if not candidates:
        return None, 0.0

    normalized_query = _normalize(query)
    if len(normalized_query) < MIN_CHUNK_CHARS:
        return None, 0.0

    best_name = None
    best_score = 0.0

    for candidate in candidates:
        if not isinstance(candidate, str) or not candidate.strip():
            continue
        score = SequenceMatcher(None, normalized_query, _normalize(candidate)).ratio()
        if score > best_score:
            best_score = score
            best_name = candidate

    if best_score >= FUZZY_THRESHOLD:
        return best_name, best_score
    return None, 0.0


def resolve_entities(query: str, history: list[dict[str, str]] | None = None) -> dict[str, Any]:
    """Hybrid entity resolution: LLM extraction → fuzzy matching against DB names.

    Steps:
    1. Fetch all entity names from Neo4j (cached)
    2. LLM extracts entities WITH the known name list in the prompt
    3. For each extracted entity, fuzzy-match against DB names to correct spelling
    4. For null entities, try fuzzy matching the full query

    Returns dict with disease, ingredient, drug (corrected to DB names).
    """
    t0 = time.perf_counter()
    entity_names = _fetch_all_entity_names()

    # ── Step 1: LLM extraction with entity list context ──────────────────
    llm_entities = _extract_with_entity_context(query, entity_names, history=history)
    logger.info("----- Input to Entity Resolver -----")
    logger.info("query: %s", query.replace("\n", " "))
    logger.info("----- Output from Entity LLM Extraction -----")
    logger.info(
        "disease=%s ingredient=%s drug=%s compound=%s",
        llm_entities.get("disease"), llm_entities.get("ingredient"), llm_entities.get("drug"), llm_entities.get("compound"),
    )

    # ── Step 2: Fuzzy-match each entity against DB names ─────────────────
    final: dict[str, str | None] = {}
    resolution_log: list[str] = []

    for entity_type, label in [("disease", "Disease"), ("ingredient", "Ingredient"), ("drug", "Drug")]:
        llm_value = llm_entities.get(entity_type)
        candidates = entity_names.get(label, [])

        if llm_value:
            # Try to fuzzy-match the LLM-extracted value
            matched, score = _fuzzy_match(llm_value, candidates)
            if matched and score >= FUZZY_THRESHOLD:
                final[entity_type] = matched
                resolution_log.append(f"{entity_type}='{matched}' (LLM+fuzzy, score={score:.2f})")
                continue
            else:
                # LLM value doesn't match DB — still use it but flag
                final[entity_type] = llm_value
                resolution_log.append(f"{entity_type}='{llm_value}' (LLM only, no DB match)")
                continue

        final[entity_type] = None
        resolution_log.append(f"{entity_type}=None")

    # ── Compound: no fuzzy matching — use LLM value directly ──────────────
    compound_value = llm_entities.get("compound")
    if compound_value:
        final["compound"] = compound_value
        resolution_log.append(f"compound='{compound_value}' (LLM only, direct extraction)")
    else:
        final["compound"] = None
        resolution_log.append("compound=None")

    duration_ms = round((time.perf_counter() - t0) * 1000, 1)
    # Decide the dominant extraction method for concise logging
    method_used = "unknown"
    if any("(LLM+fuzzy" in entry for entry in resolution_log):
        method_used = "LLM+Fuzzy"
    elif any("(LLM only" in entry for entry in resolution_log):
        method_used = "LLM only"
    else:
        method_used = "fallback/no-entities"

    logger.info("----- Entity Resolution Result (%s) -----", method_used)
    logger.info("duration_ms=%.0f, resolved=%s", duration_ms, " | ".join(resolution_log))

    return {
        "disease": final.get("disease"),
        "ingredient": final.get("ingredient"),
        "drug": final.get("drug"),
        "compound": final.get("compound"),
        "_resolution_log": resolution_log,
        "_duration_ms": duration_ms,
    }


# ── LLM Extraction with Entity Context ──────────────────────────────────────


def _extract_with_entity_context(
    query: str,
    entity_names: dict[str, list[str]],
    history: list[dict[str, str]] | None = None,
) -> dict[str, str | None]:
    """LLM entity extraction with known entity names in the prompt."""
    settings = get_settings()

    # Build compact entity lists for the prompt
    diseases_str = ", ".join(entity_names.get("Disease", [])[:80])
    ingredients_str = ", ".join(entity_names.get("Ingredient", [])[:100])
    drugs_str = ", ".join(entity_names.get("Drug", [])[:80])

    normalized_history: list[dict[str, str]] = []
    for msg in history or []:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        content = msg.get("content")
        if not isinstance(content, str) or not content.strip():
            continue
        if role == "bot":
            role = "assistant"
        if role not in {"user", "assistant"}:
            continue
        normalized_history.append({"role": role, "content": content})

    messages = [
        {
            "role": "system",
            "content": (
                "You are an entity extraction engine for a Prophetic medicine (Tibb-e-Nabawi) knowledge graph.\n"
                "Extract biomedical entities from the user's query.\n\n"
                "KNOWN ENTITIES IN THE DATABASE:\n"
                f"Diseases: {diseases_str}\n"
                f"Ingredients: {ingredients_str}\n"
                f"Drugs: {drugs_str}\n\n"
                "RULES:\n"
                "- Match the user's mention to the CLOSEST known entity name above\n"
                "- If the user says 'henna' or 'hena', match to 'Heena' (the DB name)\n"
                "- If no entity of a type is mentioned, use null\n"
                "- compound: if the user mentions any chemical compound, nutrient, mineral, or vitamin by name (e.g. potassium, calcium, vitamin C, glucose), extract it exactly as written — do not try to match against a list\n"
                "- CRITICAL: Your entire response must be only the JSON object. No notes, no explanations, nothing else.\n"
                "- Reply ONLY with valid JSON, no extra text\n\n"
                'Example: {"disease": null, "ingredient": "Heena", "drug": null, "compound": null}'
            ),
        },
        *normalized_history,
        {"role": "user", "content": query},
    ]

    # Tag this LLM call so logs clearly show which pipeline step produced it
    llm_service.set_current_step("Step 1.1 - Entity Extraction")
    try:
        result = llm_service._call_llm(
            messages,
            model=settings.model_intent or settings.groq_model,
            temperature=0.0,
            max_tokens=200,
        )
    finally:
        llm_service.clear_current_step()

    import json
    content = result.get("content", "")
    try:
        match = re.search(r'\{[^{}]+\}', content)
        if match is None:
            raise ValueError("No JSON object found in entity extraction response")
        parsed = json.loads(match.group(0))
        return {
            "disease": _safe_value(parsed.get("disease")),
            "ingredient": _safe_value(parsed.get("ingredient")),
            "drug": _safe_value(parsed.get("drug")),
            "compound": _safe_value(parsed.get("compound")),
        }
    except Exception:
        logger.warning("Entity extraction parse failed: %s", content[:200])
        return {"disease": None, "ingredient": None, "drug": None, "compound": None}


def _safe_value(val: Any) -> str | None:
    """Clean an entity value — reject null/none/empty strings."""
    if val is None:
        return None
    if not isinstance(val, str):
        return None
    cleaned = val.strip()
    if not cleaned or cleaned.lower() in {"null", "none", "unknown", "n/a"}:
        return None
    return cleaned
