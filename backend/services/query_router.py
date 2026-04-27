"""Query Router — maps user intent to the correct predefined Cypher query.

Uses simple keyword matching (fast, deterministic) to classify intent, then
returns the query ID + parameters for graph_service to execute.

NO dynamic Cypher generation.  Every path leads to a predefined query.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re

from backend.services import llm_service

logger = logging.getLogger(__name__)

# ── Intent definitions ───────────────────────────────────────────────────────

INTENT_DISEASE_TREATMENT = "disease_treatment"
INTENT_DISEASE_FULL_CHAIN = "disease_full_chain"
INTENT_DISEASE_DRUG = "disease_drug"
INTENT_INGREDIENT_TREATMENT = "ingredient_treatment"
INTENT_INGREDIENT_COMPOUNDS = "ingredient_compounds"
INTENT_INGREDIENT_DRUG_MAP = "ingredient_drug_mapping"
INTENT_DRUG_BOOK = "drug_book"
INTENT_HADITH_INFO = "hadith_info"
INTENT_DRUG_COUNT = "drug_count"
INTENT_DRUG_SUBSTITUTE = "drug_substitute"
INTENT_INGREDIENT_SUBSTITUTE = "ingredient_substitute"
INTENT_GENERAL = "general"

# ── Keyword patterns (compiled once) ─────────────────────────────────────────

# Substitution patterns (checked first — takes priority)
_SUBSTITUTION_RE = re.compile(
    r"\b(instead of|replace|substitut|alternative|equivalent|swap|use in place of)\b",
    re.I,
)

_PATTERNS: list[tuple[re.Pattern, str]] = [
    # Hadith / Islamic references
    (re.compile(r"\b(hadith|hadis|sunnah|prophet|nabawi|narrat)", re.I), INTENT_HADITH_INFO),
    # Compound / chemical details of ingredients (before drug — "compound" should not hit drug)
    (re.compile(r"\b(compound|chemical|composition|molecule)", re.I), INTENT_INGREDIENT_COMPOUNDS),
    # Drug mapping / alternative / equivalent
    (re.compile(r"\b(mapping|map|equivalent|alternativ|similar.drug|replac)", re.I), INTENT_INGREDIENT_DRUG_MAP),
    (
        re.compile(
            r"(drug[s]?\s+(for|to|that|which)\s+(cur[ei]|treat|help)|"
            r"(cur[ei]|treat|help)\s+\w+\s+with\s+drug|"
            r"medicine[s]?\s+(for|to|that)\s+(cur[ei]|treat)|"
            r"what\s+drug[s]?\s+(cur[ei]|treat|help|for)|"
            r"which\s+drug[s]?\s+(can|to|for)|"
            r"tell\s+me\s+(the\s+)?drug[s]?\s+(to|for|which|that))",
            re.I
        ),
        INTENT_DISEASE_DRUG,
    ),
    # Drug / book / source
    (re.compile(r"\b(drug|pharma|tablet|capsule|book|download|source)", re.I), INTENT_DRUG_BOOK),
    # Count drugs
    (re.compile(r"\b(how many drug|count drug|number of drug)", re.I), INTENT_DRUG_COUNT),
    # Full chain (explicit)
    (re.compile(r"\b(full chain|full trace|complete|explain.*chain|end.to.end)", re.I), INTENT_DISEASE_FULL_CHAIN),
    (
        re.compile(
            r"(what\s+does\s+\w+\s+(cur[ei]|treat|heal|help)|"
            r"which\s+disease.*\w+\s+(cur[ei]|treat)|"
            r"what\s+(can|does)\s+\w+\s+(cur[ei]|treat|help)|"
            r"what\s+is\s+\w+\s+good\s+for|"
            r"diseases?\s+(treated|cured|healed)\s+by|"
            r"what\s+does\s+.+\s+treat)",
            re.I
        ),
        INTENT_INGREDIENT_TREATMENT,
    ),
    # Treatment / cure / remedy (most common — catch-all for disease questions)
    (re.compile(r"\b(help|helps|relief|treat|cur[ei]|remed|heal|therap|disease|illness|condition|symptom|pain|headach|fever|infect)", re.I), INTENT_DISEASE_TREATMENT),
]


def classify_intent_regex(user_query: str, entities: dict[str, str | None] | None = None) -> str:
    """Classify user intent using keyword matching. Returns an intent string.

    Substitution queries ("drugs instead of X") take priority over other intents.
    """
    # 1) Check for substitution first
    if _SUBSTITUTION_RE.search(user_query):
        # "drugs instead of ingredient" → ingredient_substitute
        # "alternative to drug" → drug_substitute  
        if entities:
            ingredient = entities.get("ingredient")
            drug = entities.get("drug")
            if ingredient and not drug:
                return INTENT_INGREDIENT_SUBSTITUTE
            if drug and not ingredient:
                return INTENT_DRUG_SUBSTITUTE
        # If entities unclear, check for drug keyword
        if re.search(r"\bdrug", user_query, re.I):
            return INTENT_INGREDIENT_SUBSTITUTE  # "drugs instead of X" = find drugs for ingredient
        return INTENT_INGREDIENT_SUBSTITUTE  # default: ingredient → drug mapping

    # 2) Standard keyword patterns
    for pattern, intent in _PATTERNS:
        if pattern.search(user_query):
            return intent
    return INTENT_GENERAL


async def classify_intent_llm(user_query: str, entities: dict[str, str | None] | None = None) -> str:
    """Classify user intent with an LLM, then fall back to regex if needed."""
    messages = [
        {
            "role": "system",
            "content": (
                "You are an intent classifier for a Prophetic medicine knowledge graph chatbot.\n"
                "Given a user query and the detected entities, return the single best intent as JSON.\n\n"
                "INTENT OPTIONS:\n"
                "- disease_treatment: user asks what treats/cures a disease\n"
                "- disease_full_chain: user wants the full chain for a disease\n"
                "- disease_drug: user asks which drugs treat a disease\n"
                "- ingredient_treatment: user asks which diseases an ingredient treats/cures\n"
                "- ingredient_compounds: user asks about chemical composition of an ingredient\n"
                "- ingredient_drug_mapping: user asks for drug equivalents of an ingredient\n"
                "- drug_book: user wants drug reference/source info\n"
                "- hadith_info: user asks for hadith or prophetic references\n"
                "- drug_count: user asks how many drugs are linked to an ingredient\n"
                "- drug_substitute: user wants natural alternatives to a drug\n"
                "- ingredient_substitute: user wants drugs instead of a natural ingredient\n"
                "- general: none of the above\n\n"
                "KEY RULE: If the query mentions an ingredient (like honey, black seed, ginger) and asks what it cures/treats/helps, ALWAYS use ingredient_treatment — never disease_treatment.\n\n"
                "If the user explicitly says drug(s) or medicine(s) and asks which one treats/cures a disease, ALWAYS return disease_drug — never disease_treatment.\n\n"
                'Reply ONLY with valid JSON: {"intent": "<intent_name>", "reason": "<one line why>"}'
            ),
        },
        {
            "role": "user",
            "content": (
                f"Query: {user_query}\n"
                f"Entities: {json.dumps(entities or {}, ensure_ascii=True, sort_keys=True)}"
            ),
        },
    ]

    try:
        result = await asyncio.to_thread(
            llm_service._call_llm,
            messages,
            model="llama-3.1-8b-instant",
            temperature=0.0,
            max_tokens=120,
        )
        content = result.get("content", "")
        cleaned = content.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        parsed = json.loads(cleaned)
        intent = parsed.get("intent")
        reason = parsed.get("reason", "")

        if isinstance(intent, str):
            normalized_intent = intent.strip()
            valid_intents = {
                INTENT_DISEASE_TREATMENT,
                INTENT_DISEASE_FULL_CHAIN,
                INTENT_DISEASE_DRUG,
                INTENT_INGREDIENT_TREATMENT,
                INTENT_INGREDIENT_COMPOUNDS,
                INTENT_INGREDIENT_DRUG_MAP,
                INTENT_DRUG_BOOK,
                INTENT_HADITH_INFO,
                INTENT_DRUG_COUNT,
                INTENT_DRUG_SUBSTITUTE,
                INTENT_INGREDIENT_SUBSTITUTE,
                INTENT_GENERAL,
            }
            if normalized_intent in valid_intents:
                logger.info("Intent LLM classified '%s' (%s)", normalized_intent, reason)
                return normalized_intent

        logger.warning("Intent LLM returned invalid intent: %s", content)
    except Exception:
        logger.exception("Intent LLM classification failed; falling back to regex")

    return classify_intent_regex(user_query, entities)


def route_query(
    intent: str,
    entities: dict[str, str | None],
) -> tuple[str | None, dict[str, str], str]:
    """Given an intent and extracted entities, pick a query ID + build params.

    Returns ``(query_id, params, resolved_intent)``.

    If no suitable query can be determined, returns ``(None, {}, intent)``.
    """
    disease = entities.get("disease")
    ingredient = entities.get("ingredient")
    drug = entities.get("drug")

    # ── Disease-centric intents ──────────────────────────────────────────
    if intent == INTENT_DISEASE_FULL_CHAIN and disease:
        return "E", {"disease_name": disease}, intent

    if intent == INTENT_DISEASE_TREATMENT and disease:
        return "A", {"disease_name": disease}, intent

    if intent == INTENT_HADITH_INFO and disease:
        return "F", {"disease_name": disease}, intent

    # ── Ingredient-centric intents ───────────────────────────────────────
    if intent == INTENT_INGREDIENT_COMPOUNDS and ingredient:
        return "B", {"ingredient_name": ingredient}, intent

    if intent == INTENT_INGREDIENT_DRUG_MAP and ingredient:
        return "C", {"ingredient_name": ingredient}, intent

    if intent == INTENT_DRUG_COUNT and ingredient:
        return "I", {"ingredient_name": ingredient}, intent

    # ── Drug-centric intents ─────────────────────────────────────────────
    if intent == INTENT_DISEASE_DRUG and disease:
        return "E", {"disease_name": disease}, intent

    if intent == INTENT_DRUG_BOOK and drug:
        return "D1", {"drug_name": drug}, intent

    # ── Substitution intents ─────────────────────────────────────────────
    if intent == INTENT_INGREDIENT_SUBSTITUTE and ingredient:
        # "drugs instead of Heena" → query C (ingredient → drug mapping)
        return "C", {"ingredient_name": ingredient}, intent

    if intent == INTENT_DRUG_SUBSTITUTE and drug:
        # "alternative to paracetamol" → query D1 (drug → book)
        return "D1", {"drug_name": drug}, intent

    if intent == INTENT_INGREDIENT_TREATMENT and ingredient:
        return "J", {"ingredient_name": ingredient}, intent

    # ── Fallback: try to use whatever entity we have ─────────────────────
    if disease:
        return "A", {"disease_name": disease}, INTENT_DISEASE_TREATMENT

    if ingredient:
        return "C", {"ingredient_name": ingredient}, INTENT_INGREDIENT_DRUG_MAP

    if drug:
        return "D1", {"drug_name": drug}, INTENT_DRUG_BOOK

    logger.warning("No entity found for intent '%s' — cannot route to a query", intent)
    return None, {}, intent
