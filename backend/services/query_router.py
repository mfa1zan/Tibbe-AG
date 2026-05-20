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
from backend.services.entity_resolver import resolve_entities

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
INTENT_COMPOUND_SEARCH = "compound_search"
INTENT_GENERAL = "general"

VALID_INTENTS = {
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
    INTENT_COMPOUND_SEARCH,
    INTENT_GENERAL,
}

# ── Intent Classification Flow ───────────────────────────────────────────────
# Primary:  analyze_query_llm()      — LLM-based task decomposition (multi-intent)
#   └── on failure: resolve_entities() + classify_intent_llm()  — single intent LLM
#         └── on failure: classify_intent_regex()               — regex last resort

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
                "CLASSIFICATION STEPS - follow in order:\n"
                "1. What is the SUBJECT of the question? (the thing the user is asking ABOUT)\n"
                "2. If subject is a DISEASE -> consider: disease_treatment, disease_drug, disease_full_chain\n"
                "3. If subject is an INGREDIENT -> consider: ingredient_treatment, ingredient_compounds, ingredient_substitute\n"
                "4. If subject is a DRUG -> consider: drug_substitute, drug_book\n"
                "5. If subject is a CHEMICAL, NUTRIENT, MINERAL, or VITAMIN -> compound_search\n"
                "6. If user says 'drug', 'medicine', 'tablet', 'capsule' explicitly -> disease_drug (NOT disease_treatment)\n"
                "7. If user says 'instead of', 'replace', 'alternative' -> check direction:\n"
                "   ingredient -> drug = ingredient_substitute | drug -> natural = drug_substitute\n\n"
                "INTENT OPTIONS (read carefully - these are easy to confuse):\n"
                "- disease_treatment: User asks what NATURAL INGREDIENT or FOOD treats/cures a DISEASE.\n"
                "  Entity needed: disease.\n"
                "  Examples: 'What treats diabetes?' / 'What is good for fever?' / 'Remedies for headache'\n"
                "  NEVER use this if the subject is an ingredient. NEVER use this if user says drug/medicine.\n\n"
                "- ingredient_treatment: User asks what DISEASES or CONDITIONS a specific INGREDIENT cures/treats/helps.\n"
                "  Entity needed: ingredient.\n"
                "  Examples: 'What does honey cure?' / 'What is black seed good for?' / 'What diseases does ginger treat?'\n"
                "  KEY: If the SUBJECT of the sentence is an ingredient name, ALWAYS use this - not disease_treatment.\n\n"
                "- disease_drug: User asks which PHARMACEUTICAL DRUG or MEDICINE treats a DISEASE.\n"
                "  Entity needed: disease.\n"
                "  Examples: 'Which drugs treat malaria?' / 'What medicines cure fever?' / 'Tablets for diabetes'\n"
                "  KEY: Only use when user explicitly says 'drug', 'medicine', 'pharmaceutical', 'tablet', or 'capsule'.\n\n"
                "- disease_full_chain: User wants the COMPLETE CHAIN from disease to ingredients to compounds to drugs.\n"
                "  Entity needed: disease.\n"
                "  Examples: 'Show full chain for diabetes' / 'Complete treatment path for fever' / 'Explain everything about malaria treatment'\n\n"
                "- ingredient_compounds: User asks about the CHEMICAL COMPOSITION or COMPOUNDS inside an ingredient.\n"
                "  Entity needed: ingredient.\n"
                "  Examples: 'What chemicals are in honey?' / 'What compounds does black seed contain?' / 'Chemical makeup of ginger'\n\n"
                "- ingredient_drug_mapping: User asks which DRUGS share compounds with an ingredient - no substitution framing.\n"
                "  Entity needed: ingredient.\n"
                "  Examples: 'What drugs are equivalent to honey?' / 'Map honey to pharmaceutical drugs'\n\n"
                "- ingredient_substitute: User wants DRUGS TO USE INSTEAD OF a natural ingredient. Direction: ingredient -> drug.\n"
                "  Entity needed: ingredient.\n"
                "  Examples: 'What drug can replace honey?' / 'Drugs instead of black seed' / 'Pharmaceutical alternative to henna'\n\n"
                "- drug_substitute: User wants NATURAL INGREDIENTS as alternatives to a DRUG. Direction: drug -> ingredient.\n"
                "  Entity needed: drug.\n"
                "  Examples: 'Natural alternative to paracetamol' / 'What can replace ibuprofen naturally?' / 'Herbal substitute for aspirin'\n\n"
                "- compound_search: User asks which INGREDIENT or FOOD CONTAINS a specific CHEMICAL COMPOUND, NUTRIENT, MINERAL, or VITAMIN.\n"
                "  Entity needed: compound.\n"
                "  Examples: 'Which foods contain potassium?' / 'What ingredient has vitamin C?' / 'Which herb contains quercetin?'\n\n"
                "- hadith_info: User explicitly asks for HADITH, SUNNAH, or PROPHETIC REFERENCES.\n"
                "  Examples: 'What does the hadith say about honey?' / 'Prophetic references for black seed' / 'Sunnah remedy for fever'\n\n"
                "- drug_book: User asks for a drug SOURCE, BOOK, or DOWNLOAD LINK.\n"
                "  Entity needed: drug.\n"
                "  Examples: 'Where is paracetamol mentioned?' / 'Source book for ibuprofen'\n\n"
                "- drug_count: User asks HOW MANY drugs are linked to an ingredient.\n"
                "  Entity needed: ingredient.\n"
                "  Examples: 'How many drugs does honey map to?' / 'Count drugs for black seed'\n\n"
                "- general: Greeting, vague question, or completely out-of-scope query. Use as last resort only.\n\n"
                "ENTITY HINT - use detected entities to break ties:\n"
                "- ingredient detected, no disease -> lean toward ingredient_* intents\n"
                "- disease detected, no ingredient -> lean toward disease_* intents\n"
                "- both detected -> check sentence direction: what is being ASKED ABOUT vs what is the ANSWER\n"
                "- drug detected -> drug_substitute or drug_book\n"
                "- compound/nutrient/vitamin detected -> compound_search\n\n"
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
        def _thread_call():
            llm_service.set_current_step("Step 1 - Intent Classification")
            try:
                return llm_service._call_llm(
                    messages,
                    model="llama-3.1-8b-instant",
                    temperature=0.0,
                    max_tokens=120,
                )
            finally:
                llm_service.clear_current_step()

        result = await asyncio.to_thread(_thread_call)
        content = result.get("content", "")
        if not content.strip():
            logger.warning(
                "Intent LLM returned empty response; falling back to regex (error=%s)",
                result.get("error"),
            )
            return classify_intent_regex(user_query, entities)
        cleaned = content.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        parsed = json.loads(cleaned)
        intent = parsed.get("intent")
        reason = parsed.get("reason", "")

        if isinstance(intent, str):
            normalized_intent = intent.strip()
            if normalized_intent in VALID_INTENTS:
                logger.info("Intent LLM classified '%s' (%s)", normalized_intent, reason)
                return normalized_intent

        logger.warning("Intent LLM returned invalid intent: %s", content)
    except Exception:
        logger.exception("Intent LLM classification failed; falling back to regex")

    return classify_intent_regex(user_query, entities)


async def analyze_query_llm(
    query: str,
    history: list[dict[str, str]] | None = None,
) -> dict:
    """Analyze a full query and return entities plus task list for routing."""
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
                "You analyze Prophetic medicine chatbot queries and decompose them into tasks.\n"
                "CRITICAL OUTPUT RULES:\n"
                "- Output ONLY valid JSON\n"
                "- Do NOT include explanations, headings, or any text outside JSON\n"
                "- Do NOT wrap the response in markdown (no ```json)\n"
                "- Start the response directly with '{' and end with '}'\n"
                "- Ensure the JSON is syntactically valid and parsable by json.loads()\n\n"
                "Return a JSON object with one key: 'tasks'.\n\n"
                "CLASSIFICATION STEPS - follow in order:\n"
                "1. What is the SUBJECT of the question? (the thing the user is asking ABOUT)\n"
                "2. If subject is a DISEASE -> consider: disease_treatment, disease_drug, disease_full_chain\n"
                "3. If subject is an INGREDIENT -> consider: ingredient_treatment, ingredient_compounds, ingredient_substitute\n"
                "4. If subject is a DRUG -> consider: drug_substitute, drug_book\n"
                "5. If subject is a CHEMICAL, NUTRIENT, MINERAL, or VITAMIN -> compound_search\n"
                "6. If user says 'drug', 'medicine', 'tablet', 'capsule' explicitly -> disease_drug (NOT disease_treatment)\n"
                "7. If user says 'instead of', 'replace', 'alternative' -> check direction:\n"
                "   ingredient -> drug = ingredient_substitute | drug -> natural = drug_substitute\n\n"
                "INTENT OPTIONS (read carefully - these are easy to confuse):\n"
                "- disease_treatment: User asks what NATURAL INGREDIENT or FOOD treats/cures a DISEASE.\n"
                "  Entity needed: disease.\n"
                "  Examples: 'What treats diabetes?' / 'What is good for fever?' / 'Remedies for headache'\n"
                "  NEVER use this if the subject is an ingredient. NEVER use this if user says drug/medicine.\n\n"
                "- ingredient_treatment: User asks what DISEASES or CONDITIONS a specific INGREDIENT cures/treats/helps.\n"
                "  Entity needed: ingredient.\n"
                "  Examples: 'What does honey cure?' / 'What is black seed good for?' / 'What diseases does ginger treat?'\n"
                "  KEY: If the SUBJECT of the sentence is an ingredient name, ALWAYS use this - not disease_treatment.\n\n"
                "- disease_drug: User asks which PHARMACEUTICAL DRUG or MEDICINE treats a DISEASE.\n"
                "  Entity needed: disease.\n"
                "  Examples: 'Which drugs treat malaria?' / 'What medicines cure fever?' / 'Tablets for diabetes'\n"
                "  KEY: Only use when user explicitly says 'drug', 'medicine', 'pharmaceutical', 'tablet', or 'capsule'.\n\n"
                "- disease_full_chain: User wants the COMPLETE CHAIN from disease to ingredients to compounds to drugs.\n"
                "  Entity needed: disease.\n"
                "  Examples: 'Show full chain for diabetes' / 'Complete treatment path for fever' / 'Explain everything about malaria treatment'\n\n"
                "- ingredient_compounds: User asks about the CHEMICAL COMPOSITION or COMPOUNDS inside an ingredient.\n"
                "  Entity needed: ingredient.\n"
                "  Examples: 'What chemicals are in honey?' / 'What compounds does black seed contain?' / 'Chemical makeup of ginger'\n\n"
                "- ingredient_drug_mapping: User asks which DRUGS share compounds with an ingredient - no substitution framing.\n"
                "  Entity needed: ingredient.\n"
                "  Examples: 'What drugs are equivalent to honey?' / 'Map honey to pharmaceutical drugs'\n\n"
                "- ingredient_substitute: User wants DRUGS TO USE INSTEAD OF a natural ingredient. Direction: ingredient -> drug.\n"
                "  Entity needed: ingredient.\n"
                "  Examples: 'What drug can replace honey?' / 'Drugs instead of black seed' / 'Pharmaceutical alternative to henna'\n\n"
                "- drug_substitute: User wants NATURAL INGREDIENTS as alternatives to a DRUG. Direction: drug -> ingredient.\n"
                "  Entity needed: drug.\n"
                "  Examples: 'Natural alternative to paracetamol' / 'What can replace ibuprofen naturally?' / 'Herbal substitute for aspirin'\n\n"
                "- compound_search: User asks which INGREDIENT or FOOD CONTAINS a specific CHEMICAL COMPOUND, NUTRIENT, MINERAL, or VITAMIN.\n"
                "  Entity needed: compound.\n"
                "  Examples: 'Which foods contain potassium?' / 'What ingredient has vitamin C?' / 'Which herb contains quercetin?'\n\n"
                "- hadith_info: User explicitly asks for HADITH, SUNNAH, or PROPHETIC REFERENCES.\n"
                "  Examples: 'What does the hadith say about honey?' / 'Prophetic references for black seed' / 'Sunnah remedy for fever'\n\n"
                "- drug_book: User asks for a drug SOURCE, BOOK, or DOWNLOAD LINK.\n"
                "  Entity needed: drug.\n"
                "  Examples: 'Where is paracetamol mentioned?' / 'Source book for ibuprofen'\n\n"
                "- drug_count: User asks HOW MANY drugs are linked to an ingredient.\n"
                "  Entity needed: ingredient.\n"
                "  Examples: 'How many drugs does honey map to?' / 'Count drugs for black seed'\n\n"
                "- general: Greeting, vague question, or completely out-of-scope query. Use as last resort only.\n\n"
                "RULES:\n"
                "- Tasks must only use intents from the list above\n"
                "- If the query is simple or single-intent, return exactly one task\n"
                "- Cap tasks at 4 maximum\n"
                "- Each task's \"sub_question\" MUST be a full, self-contained natural-language question (not a single word or bare entity).\n"
                "  If the user only provides an entity or fragment, reformulate it into a clear sub-question (e.g. 'What does <entity> treat?' or 'How is <entity> used for <condition>?').\n\n"
                "ENTITY HINT - use detected entities to break ties:\n"
                "- ingredient detected, no disease -> lean toward ingredient_* intents\n"
                "- disease detected, no ingredient -> lean toward disease_* intents\n"
                "- both detected -> check sentence direction: what is being ASKED ABOUT vs what is the ANSWER\n"
                "- drug detected -> drug_substitute or drug_book\n"
                "- compound/nutrient/vitamin detected -> compound_search\n\n"
                "OUTPUT FORMAT (JSON only):\n"
                "{\"tasks\": [{\"intent\": \"...\", \"sub_question\": \"...\"}]}"
            ),
        },
        *normalized_history,
        {"role": "user", "content": query},
    ]

    try:
        def _thread_call_analysis():
            llm_service.set_current_step("Step 1 - Analyzer")
            try:
                return llm_service._call_llm(
                    messages,
                    model="llama-3.1-8b-instant",
                    temperature=0.0,
                    max_tokens=400,
                )
            finally:
                llm_service.clear_current_step()

        result = await asyncio.to_thread(_thread_call_analysis)
        content = result.get("content", "")
        if not content.strip():
            logger.warning(
                "Query analysis LLM returned empty response; using fallback (error=%s)",
                result.get("error"),
            )
            entities = resolve_entities(query, history=history)
            intent = await classify_intent_llm(query, entities)
            return {
                "entities": entities,
                "tasks": [{"intent": intent, "sub_question": query}],
            }
        cleaned = content.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        parsed = json.loads(cleaned)

        tasks: list[dict[str, str]] = []
        raw_tasks = parsed.get("tasks") if isinstance(parsed, dict) else None
        if isinstance(raw_tasks, list):
            for item in raw_tasks:
                if not isinstance(item, dict):
                    continue
                intent = item.get("intent")
                sub_question = item.get("sub_question")
                if not isinstance(intent, str) or not isinstance(sub_question, str):
                    continue
                normalized_intent = intent.strip()
                normalized_subq = sub_question.strip()
                if normalized_intent in VALID_INTENTS and normalized_subq:
                    tasks.append({"intent": normalized_intent, "sub_question": normalized_subq})

        if tasks:
            entities = resolve_entities(query, history=history)
            return {"entities": entities, "tasks": tasks[:4]}

        logger.warning("Query analysis LLM returned invalid tasks: %s", content)
    except Exception:
        logger.exception("Query analysis LLM failed; falling back to single-intent pipeline")

    entities = resolve_entities(query, history=history)
    intent = await classify_intent_llm(query, entities)
    return {
        "entities": entities,
        "tasks": [{"intent": intent, "sub_question": query}],
    }


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
    compound = entities.get("compound")

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

    # ── Compound-centric intents ────────────────────────────────────────
    if intent == INTENT_COMPOUND_SEARCH and compound:
        return "K", {"compound_name": compound}, intent

    # ── Drug-centric intents ─────────────────────────────────────────────
    if intent == INTENT_DISEASE_DRUG and disease:
        return "E2", {"disease_name": disease}, intent

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

    logger.warning("No entity found for intent '%s' — returning general fallback", intent)
    return None, {}, INTENT_GENERAL
