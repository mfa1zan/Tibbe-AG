import json
import logging
import re
from dataclasses import dataclass
from difflib import SequenceMatcher, get_close_matches
from functools import lru_cache

from cachetools import TTLCache
from neo4j import GraphDatabase

from app.config import get_settings
from app.services.llm_service import LLMService

logger = logging.getLogger(__name__)

ENTITY_KEYS = ("disease", "ingredient", "drug")
FUZZY_CONFIDENCE_THRESHOLD = 0.75
MIN_FUZZY_CHUNK_CHARS = 4


@dataclass
class _EntityResult:
    value: str | None
    confidence: float


@lru_cache(maxsize=1)
def _get_driver():
    settings = get_settings()
    return GraphDatabase.driver(
        settings.neo4j_uri,
        auth=(settings.neo4j_username, settings.neo4j_password),
    )


@lru_cache(maxsize=1)
def _get_llm_service() -> LLMService:
    settings = get_settings()
    return LLMService(
        api_key=settings.groq_api_key,
        model=settings.groq_model,
        base_url=settings.groq_base_url,
    )


# Cache entity names pulled from Neo4j so fuzzy matching remains fast in production.
_ENTITY_NAME_CACHE: TTLCache = TTLCache(maxsize=3, ttl=600)


def _normalize(text: str) -> str:
    lowered = text.lower().strip()
    lowered = re.sub(r"[^\w\s]", " ", lowered)
    lowered = re.sub(r"\s+", " ", lowered)
    return lowered.strip()


def _extract_json_object(raw_text: str) -> dict | None:
    try:
        parsed = json.loads(raw_text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    # Handle accidental wrappers by finding the first JSON object in the text.
    match = re.search(r"\{[\s\S]*\}", raw_text)
    if not match:
        return None


def _is_low_signal_query(query: str) -> bool:
    """
    Detect short conversational inputs where fuzzy biomedical matching is unsafe.
    Example: hi, hello, ok, thanks.
    """
    normalized = _normalize(query)
    if not normalized:
        return True

    tokens = normalized.split()
    if len(tokens) == 1 and len(tokens[0]) < MIN_FUZZY_CHUNK_CHARS:
        return True

    common_general = {
        "hi",
        "hello",
        "hey",
        "thanks",
        "thank you",
        "ok",
        "okay",
        "yo",
    }
    return normalized in common_general

    try:
        parsed = json.loads(match.group(0))
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        return None


def _safe_entity_value(value) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        return None

    cleaned = value.strip()
    if not cleaned or cleaned.lower() in {"null", "none", "unknown"}:
        return None
    return cleaned


def _safe_confidence(value) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, score))


def _extract_entities_llm_with_confidence(query: str) -> dict[str, _EntityResult]:
    from app.services.pipeline_tracer import get_tracer

    llm_service = _get_llm_service()
    settings = get_settings()

    # Prompt constrained to extraction only (no reasoning, no hallucination).
    system_prompt = (
        "You are an entity extraction engine for PRO-MedGraph.\n"
        "Extract only explicit biomedical entities from the user query.\n"
        "Return strict JSON only.\n"
        "No markdown, no explanation, no extra keys.\n"
        "If an entity is missing, use null and confidence 0.\n"
        "Do not infer entities not present in the query."
    )

    user_prompt = (
        f"Query: {query}\n\n"
        "Return exactly this JSON schema:\n"
        "{\n"
        '  "disease": null | string,\n'
        '  "ingredient": null | string,\n'
        '  "drug": null | string,\n'
        '  "confidence": {\n'
        '    "disease": number,\n'
        '    "ingredient": number,\n'
        '    "drug": number\n'
        "  }\n"
        "}"
    )

    used_model = settings.llm_cypher_model or settings.groq_model
    logger.info(
        "┌─ ENTITY EXTRACTION LLM call model=%s query='%.100s'",
        used_model, query,
    )

    raw = ""
    try:
        # Low temperature for deterministic extraction behavior.
        raw = _run_async_completion(
            llm_service=llm_service,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model=used_model,
            temperature=0.0,
        )
        logger.info("└─ ENTITY EXTRACTION LLM raw response: %.300s", raw)
    except Exception:
        logger.exception("LLM entity extraction failed")

    parsed = _extract_json_object(raw) if raw else None
    if not parsed:
        return {key: _EntityResult(value=None, confidence=0.0) for key in ENTITY_KEYS}

    confidence_obj = parsed.get("confidence", {}) if isinstance(parsed.get("confidence"), dict) else {}

    return {
        key: _EntityResult(
            value=_safe_entity_value(parsed.get(key)),
            confidence=_safe_confidence(confidence_obj.get(key)),
        )
        for key in ENTITY_KEYS
    }


def extract_entities_llm(query: str) -> dict:
    """
    LLM-first entity extraction.
    Returns only {disease, ingredient, drug} with missing values as null.
    """
    results = _extract_entities_llm_with_confidence(query)
    return {key: results[key].value for key in ENTITY_KEYS}


def _fetch_entity_names(entity_label: str) -> list[str]:
    cache_key = entity_label.upper()
    if cache_key in _ENTITY_NAME_CACHE:
        return _ENTITY_NAME_CACHE[cache_key]

    driver = _get_driver()

    # Hard-coded label queries (not user-controlled) keep the query safe.
    if entity_label == "Disease":
        query = """
        // Load all disease names for fuzzy matching fallback.
        MATCH (n:Disease)
        WHERE n.name IS NOT NULL
        RETURN DISTINCT n.name AS name
        """
    elif entity_label == "Ingredient":
        query = """
        // Load all ingredient names for fuzzy matching fallback.
        MATCH (n:Ingredient)
        WHERE n.name IS NOT NULL
        RETURN DISTINCT n.name AS name
        """
    elif entity_label == "Drug":
        query = """
        // Load all drug names for fuzzy matching fallback.
        MATCH (n:Drug)
        WHERE n.name IS NOT NULL
        RETURN DISTINCT n.name AS name
        """
    else:
        return []

    with driver.session() as session:
        rows = list(session.run(query))

    names = [row.get("name") for row in rows if isinstance(row.get("name"), str) and row.get("name").strip()]
    _ENTITY_NAME_CACHE[cache_key] = names
    return names


def _generate_query_ngrams(query: str, max_ngram: int = 5) -> list[str]:
    normalized = _normalize(query)
    tokens = normalized.split()
    if not tokens:
        return []

    chunks: list[str] = []
    for size in range(1, min(max_ngram, len(tokens)) + 1):
        for start in range(0, len(tokens) - size + 1):
            chunk = " ".join(tokens[start : start + size])
            if len(chunk) < MIN_FUZZY_CHUNK_CHARS:
                continue
            chunks.append(chunk)

    # Longer chunks first often produce better biomedical name matching.
    chunks.sort(key=len, reverse=True)
    return list(dict.fromkeys(chunks))


def _best_fuzzy_match(query: str, candidates: list[str]) -> _EntityResult:
    if not candidates:
        return _EntityResult(value=None, confidence=0.0)

    normalized_query = _normalize(query)
    if len(normalized_query) < MIN_FUZZY_CHUNK_CHARS:
        return _EntityResult(value=None, confidence=0.0)

    candidate_map = {_normalize(name): name for name in candidates}
    normalized_candidates = list(candidate_map.keys())

    # 1) Fast exact containment over normalized strings.
    for normalized_name, original_name in candidate_map.items():
        if normalized_name and normalized_name in normalized_query:
            return _EntityResult(value=original_name, confidence=1.0)

    # 2) Fuzzy match against meaningful query chunks (handles minor misspellings).
    best_value = None
    best_score = 0.0
    chunks = _generate_query_ngrams(query)

    for chunk in chunks:
        close = get_close_matches(chunk, normalized_candidates, n=1, cutoff=FUZZY_CONFIDENCE_THRESHOLD)
        if close:
            matched = close[0]
            score = SequenceMatcher(None, chunk, matched).ratio()
            if score > best_score:
                best_score = score
                best_value = candidate_map.get(matched)

    return _EntityResult(value=best_value, confidence=best_score)


def _extract_entities_fuzzy_with_confidence(query: str) -> dict[str, _EntityResult]:
    disease_names = _fetch_entity_names("Disease")
    ingredient_names = _fetch_entity_names("Ingredient")
    drug_names = _fetch_entity_names("Drug")

    return {
        "disease": _best_fuzzy_match(query, disease_names),
        "ingredient": _best_fuzzy_match(query, ingredient_names),
        "drug": _best_fuzzy_match(query, drug_names),
    }


def extract_entities_fuzzy(query: str) -> dict:
    """
    Fuzzy-only extraction fallback based on KG node names.
    Returns only {disease, ingredient, drug}.
    """
    results = _extract_entities_fuzzy_with_confidence(query)
    return {key: results[key].value for key in ENTITY_KEYS}


def extract_entities(query: str) -> dict:
    """
    Hybrid extraction orchestration:
    1) Try LLM extraction.
    2) For null/low-confidence entities, fallback to fuzzy matching from Neo4j names.
    3) Return final structured entity JSON.
    """
    from app.services.pipeline_tracer import get_tracer

    logger.info("═══ ENTITY EXTRACTION START query='%.200s' ═══", query)

    llm_results = _extract_entities_llm_with_confidence(query)
    use_fuzzy_fallback = not _is_low_signal_query(query)

    logger.info(
        "LLM extraction results: disease=%s(%.2f) ingredient=%s(%.2f) drug=%s(%.2f) | fuzzy_fallback=%s",
        llm_results['disease'].value, llm_results['disease'].confidence,
        llm_results['ingredient'].value, llm_results['ingredient'].confidence,
        llm_results['drug'].value, llm_results['drug'].confidence,
        use_fuzzy_fallback,
    )

    fuzzy_results = _extract_entities_fuzzy_with_confidence(query) if use_fuzzy_fallback else {
        key: _EntityResult(value=None, confidence=0.0) for key in ENTITY_KEYS
    }

    if use_fuzzy_fallback:
        logger.info(
            "Fuzzy extraction results: disease=%s(%.2f) ingredient=%s(%.2f) drug=%s(%.2f)",
            fuzzy_results['disease'].value, fuzzy_results['disease'].confidence,
            fuzzy_results['ingredient'].value, fuzzy_results['ingredient'].confidence,
            fuzzy_results['drug'].value, fuzzy_results['drug'].confidence,
        )

    final_entities: dict[str, str | None] = {}
    merge_decisions: list[str] = []
    for key in ENTITY_KEYS:
        llm_entity = llm_results[key]
        if llm_entity.value is not None and llm_entity.confidence >= FUZZY_CONFIDENCE_THRESHOLD:
            final_entities[key] = llm_entity.value
            decision = f"{key}='{llm_entity.value}' via LLM (conf={llm_entity.confidence:.2f})"
            merge_decisions.append(decision)
            logger.info("entity=%s method=LLM value=%s confidence=%.2f", key, llm_entity.value, llm_entity.confidence)
            continue

        fuzzy_entity = fuzzy_results[key]
        if fuzzy_entity.value is not None and use_fuzzy_fallback:
            final_entities[key] = fuzzy_entity.value
            decision = f"{key}='{fuzzy_entity.value}' via FUZZY (conf={fuzzy_entity.confidence:.2f})"
            merge_decisions.append(decision)
            logger.info(
                "entity=%s method=FUZZY value=%s confidence=%.2f",
                key,
                fuzzy_entity.value,
                fuzzy_entity.confidence,
            )
        else:
            final_entities[key] = llm_entity.value
            method = "LLM_LOW_CONFIDENCE_NO_FUZZY" if not use_fuzzy_fallback else "NONE"
            decision = f"{key}='{llm_entity.value}' via {method}"
            merge_decisions.append(decision)
            logger.info(
                "entity=%s method=%s value=%s",
                key, method, llm_entity.value,
            )

    logger.info("═══ ENTITY EXTRACTION RESULT: %s ═══", " | ".join(merge_decisions))

    # Log to tracer
    tracer = get_tracer()
    if tracer:
        tracer.log_data("entity_extraction_merge_decisions", merge_decisions)
        tracer.log_data("entity_extraction_llm_results", {
            k: {"value": v.value, "confidence": v.confidence} for k, v in llm_results.items()
        })
        if use_fuzzy_fallback:
            tracer.log_data("entity_extraction_fuzzy_results", {
                k: {"value": v.value, "confidence": v.confidence} for k, v in fuzzy_results.items()
            })

    return {
        "disease": final_entities.get("disease"),
        "ingredient": final_entities.get("ingredient"),
        "drug": final_entities.get("drug"),
    }


def _run_async_completion(
    llm_service: LLMService,
    system_prompt: str,
    user_prompt: str,
    model: str,
    temperature: float,
) -> str:
    """
    Small sync wrapper to call the existing async LLM service from sync utility functions.
    This keeps this module callable from existing synchronous service flows.
    """
    import asyncio
    from concurrent.futures import ThreadPoolExecutor

    async def _call_llm() -> str:
        return await llm_service.generate_completion(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=temperature,
            model=model,
        )

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(_call_llm())

    # If already inside an event loop, run in a dedicated thread to avoid RuntimeError.
    with ThreadPoolExecutor(max_workers=1) as executor:
        return executor.submit(lambda: asyncio.run(_call_llm())).result()