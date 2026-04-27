"""Query Router — maps user intent to the correct predefined Cypher query.

Uses simple keyword matching (fast, deterministic) to classify intent, then
returns the query ID + parameters for graph_service to execute.

NO dynamic Cypher generation.  Every path leads to a predefined query.
"""

from __future__ import annotations

import logging
import json
import re

from backend.core.config import get_settings
from backend.utils.trace_logger import sanitize_payload

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


def classify_intent(user_query: str, entities: dict[str, str | None] | None = None) -> str:
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


def classify_intent_with_reason(
    user_query: str,
    entities: dict[str, str | None] | None = None,
) -> tuple[str, str]:
    """Classify intent and return a brief reason for trace logging."""
    settings = get_settings()

    def _log(intent_val: str, reason_val: str) -> None:
        if settings.debug_trace:
            logger.info(
                "INTENT_CLASSIFICATION_RUNTIME\n%s",
                json.dumps(
                    sanitize_payload(
                        {
                            "query_text": user_query,
                            "matched_rule": reason_val,
                            "final_intent": intent_val,
                        }
                    ),
                    ensure_ascii=True,
                    indent=2,
                    default=str,
                ),
            )

    if _SUBSTITUTION_RE.search(user_query):
        if entities:
            ingredient = entities.get("ingredient")
            drug = entities.get("drug")
            if ingredient and not drug:
                intent = INTENT_INGREDIENT_SUBSTITUTE
                reason = "matched substitution regex + ingredient entity"
                _log(intent, reason)
                return intent, reason
            if drug and not ingredient:
                intent = INTENT_DRUG_SUBSTITUTE
                reason = "matched substitution regex + drug entity"
                _log(intent, reason)
                return intent, reason
        if re.search(r"\bdrug", user_query, re.I):
            intent = INTENT_INGREDIENT_SUBSTITUTE
            reason = "matched substitution regex + drug keyword fallback"
            _log(intent, reason)
            return intent, reason
        intent = INTENT_INGREDIENT_SUBSTITUTE
        reason = "matched substitution regex fallback"
        _log(intent, reason)
        return intent, reason

    for pattern, intent in _PATTERNS:
        if pattern.search(user_query):
            reason = f"matched regex: {pattern.pattern}"
            _log(intent, reason)
            return intent, reason

    _log(INTENT_GENERAL, "fallback: no regex matched")
    return INTENT_GENERAL, "fallback: no regex matched"


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


def route_query_with_reason(
    intent: str,
    entities: dict[str, str | None],
    user_query: str | None = None,
) -> tuple[str | None, dict[str, str], str, str]:
    """Resolve query route and include rationale for trace logging."""
    settings = get_settings()
    disease = entities.get("disease")
    ingredient = entities.get("ingredient")
    drug = entities.get("drug")

    def _ret(
        query_id: str | None,
        params: dict[str, str],
        resolved_intent: str,
        reason: str,
    ) -> tuple[str | None, dict[str, str], str, str]:
        if settings.debug_trace:
            logger.info(
                "QUERY_ROUTING_RUNTIME\n%s",
                json.dumps(
                    sanitize_payload(
                        {
                            "query_text": user_query,
                            "chosen_route": query_id,
                            "params": params,
                            "resolved_intent": resolved_intent,
                            "why_selected": reason,
                        }
                    ),
                    ensure_ascii=True,
                    indent=2,
                    default=str,
                ),
            )
        return query_id, params, resolved_intent, reason

    if intent == INTENT_DISEASE_FULL_CHAIN and disease:
        return _ret("E", {"disease_name": disease}, intent, "disease_full_chain intent + disease entity")

    if intent == INTENT_DISEASE_TREATMENT and disease:
        return _ret("A", {"disease_name": disease}, intent, "disease_treatment intent + disease entity")

    if intent == INTENT_HADITH_INFO and disease:
        return _ret("F", {"disease_name": disease}, intent, "hadith_info intent + disease entity")

    if intent == INTENT_INGREDIENT_COMPOUNDS and ingredient:
        return _ret("B", {"ingredient_name": ingredient}, intent, "ingredient_compounds intent + ingredient entity")

    if intent == INTENT_INGREDIENT_DRUG_MAP and ingredient:
        return _ret("C", {"ingredient_name": ingredient}, intent, "ingredient_drug_mapping intent + ingredient entity")

    if intent == INTENT_DRUG_COUNT and ingredient:
        return _ret("I", {"ingredient_name": ingredient}, intent, "drug_count intent + ingredient entity")

    if intent == INTENT_DISEASE_DRUG and disease:
        return _ret("E", {"disease_name": disease}, intent, "disease_drug intent + disease entity")

    if intent == INTENT_DRUG_BOOK and drug:
        return _ret("D1", {"drug_name": drug}, intent, "drug_book intent + drug entity")

    if intent == INTENT_INGREDIENT_SUBSTITUTE and ingredient:
        return _ret("C", {"ingredient_name": ingredient}, intent, "ingredient_substitute intent + ingredient entity")

    if intent == INTENT_DRUG_SUBSTITUTE and drug:
        return _ret("D1", {"drug_name": drug}, intent, "drug_substitute intent + drug entity")

    if intent == INTENT_INGREDIENT_TREATMENT and ingredient:
        return _ret("J", {"ingredient_name": ingredient}, intent, "ingredient_treatment intent + ingredient entity")

    if disease:
        return _ret("A", {"disease_name": disease}, INTENT_DISEASE_TREATMENT, "fallback disease entity → disease_treatment")

    if ingredient:
        return _ret("C", {"ingredient_name": ingredient}, INTENT_INGREDIENT_DRUG_MAP, "fallback ingredient entity → ingredient_drug_mapping")

    if drug:
        return _ret("D1", {"drug_name": drug}, INTENT_DRUG_BOOK, "fallback drug entity → drug_book")

    return _ret(None, {}, intent, "no resolvable entity for routing")
