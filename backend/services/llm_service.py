"""LLM service — lightweight entity extraction + single-call answer generation.

Uses the Groq API (OpenAI-compatible) via httpx.
"""

from __future__ import annotations

import contextvars
import json
import logging
import re
import time
from typing import Any

import httpx

from backend.core.config import get_settings

logger = logging.getLogger(__name__)

_TIMEOUT = httpx.Timeout(60.0, connect=10.0)

# Per-request API call counter (context-local)
_api_call_count: contextvars.ContextVar[int] = contextvars.ContextVar("api_call_count", default=0)

# Optional context var to tag the current pipeline step for logs
_current_step: contextvars.ContextVar[str] = contextvars.ContextVar("current_step", default="")


def reset_api_call_count() -> None:
    """Reset the per-request API call counter to zero."""
    _api_call_count.set(0)


def get_api_call_count() -> int:
    """Return current per-request API call count."""
    return _api_call_count.get()


def set_current_step(step: str) -> None:
    """Set a short tag describing the current pipeline step for logging (e.g. 'Step 1')."""
    try:
        _current_step.set(step or "")
    except Exception:
        pass


def clear_current_step() -> None:
    """Clear the current pipeline step tag."""
    try:
        _current_step.set("")
    except Exception:
        pass


def _should_log_llm_trace() -> bool:
    """Return True when detailed LLM request/response logging is enabled."""
    settings = get_settings()
    # Keep for compatibility but prefer always-on structured LLM logging.
    return True


def _log_llm_messages(model: str, messages: list[dict[str, str]], *, temperature: float, max_tokens: int) -> None:
    """Log the exact request payload sent to the LLM provider."""
    step = _current_step.get() or None
    if step:
        logger.info("----- Input to LLM (%s) [%s] -----", model, step)
    else:
        logger.info("----- Input to LLM (%s) -----", model)
    logger.info("temperature=%.2f max_tokens=%d messages=%d", temperature, max_tokens, len(messages))
    for index, message in enumerate(messages, start=1):
        role = message.get("role", "unknown")
        content = message.get("content", "") or ""
        single_line = content.replace("\n", " ")
        logger.info("[%d] role=%s: %s", index, role, single_line)
    logger.info("----- End Input to LLM (%s) -----", model)


def _call_llm(
    messages: list[dict[str, str]],
    *,
    model: str | None = None,
    temperature: float = 0.1,
    max_tokens: int = 1024,
) -> dict[str, Any]:
    """Make a single chat-completion call to Groq and return the parsed result.

    Returns dict with ``content``, ``model``, ``duration_ms``.
    """
    settings = get_settings()
    use_model = model or settings.groq_model

    if _should_log_llm_trace():
        _log_llm_messages(use_model, messages, temperature=temperature, max_tokens=max_tokens)

    t0 = time.perf_counter()
    try:
        # increment per-request API call counter
        count = _api_call_count.get() + 1
        _api_call_count.set(count)

        with httpx.Client(timeout=_TIMEOUT) as client:
            resp = client.post(
                f"{settings.groq_base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {settings.groq_api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": use_model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                },
            )
            resp.raise_for_status()
            data = resp.json()

        duration_ms = round((time.perf_counter() - t0) * 1000, 1)
        content = data["choices"][0]["message"]["content"]

        step = _current_step.get() or None
        if step:
            logger.info("***** Output from LLM (%s) [%s] *****", use_model, step)
        else:
            logger.info("***** Output from LLM (%s) *****", use_model)
        logger.info("chars=%d duration_ms=%.0f api_call_index=%d", len(content), duration_ms, _api_call_count.get())
        logger.info("%s", content)
        logger.info("***** End Output from LLM (%s) *****", use_model)
        return {
            "content": content,
            "model": use_model,
            "duration_ms": duration_ms,
            "api_call_count": _api_call_count.get(),
        }
    except Exception as exc:
        duration_ms = round((time.perf_counter() - t0) * 1000, 1)
        step = _current_step.get() or None
        if step:
            logger.exception("LLM call failed [%s] [%s] (%.0fms)", use_model, step, duration_ms)
        else:
            logger.exception("LLM call failed [%s] (%.0fms)", use_model, duration_ms)
        return {
            "content": "",
            "model": use_model,
            "duration_ms": duration_ms,
            "error": str(exc),
            "api_call_count": _api_call_count.get(),
        }


# ── Entity Extraction ────────────────────────────────────────────────────────


# DEPRECATED: unused legacy function. Use resolve_entities() instead.
def extract_entities(user_query: str) -> dict[str, str | None]:
    """Use a lightweight LLM call to extract disease, ingredient, and drug names.

    Returns a dict like:
        {"disease": "hypertension", "ingredient": None, "drug": None}
    """
    settings = get_settings()

    messages = [
        {
            "role": "system",
            "content": (
                "You are an entity extractor for a Prophetic medicine knowledge graph. "
                "Given the user's question, extract the following entities if present:\n"
                "- disease: the disease or medical condition mentioned\n"
                "- ingredient: the natural ingredient (herb, food, spice) mentioned\n"
                "- drug: the pharmaceutical drug mentioned\n\n"
                "Reply ONLY with valid JSON, no extra text. Example:\n"
                '{"disease": "hypertension", "ingredient": null, "drug": null}'
            ),
        },
        {"role": "user", "content": user_query},
    ]

    result = _call_llm(
        messages,
        model=settings.model_intent or settings.groq_model,
        temperature=0.0,
        max_tokens=200,
    )

    content = result.get("content", "")
    try:
        # Handle case where LLM wraps JSON in markdown code fences
        cleaned = content.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        parsed = json.loads(cleaned)
        return {
            "disease": parsed.get("disease"),
            "ingredient": parsed.get("ingredient"),
            "drug": parsed.get("drug"),
            "_llm_debug": result,
        }
    except (json.JSONDecodeError, KeyError):
        logger.warning("Entity extraction failed to parse: %s", content)
        return {
            "disease": None,
            "ingredient": None,
            "drug": None,
            "_llm_debug": result,
        }


# ── Answer Generation ────────────────────────────────────────────────────────

HADITH_INTENTS = {
    "hadith_where_mentioned", "hadith_full_citation",
    "hadith_collection_filter", "hadith_context_type",
    "hadith_arabic_name", "hadith_frequency", "hadith_info",
}
_SYSTEM_PROMPTS = {
    "ingredient_substitute": (
        "You are PRO-MedGraph, a biomedical assistant specializing in Prophetic medicine (Tibb-e-Nabawi).\n"
        "The user is asking what pharmaceutical drugs share chemical compounds with a natural ingredient.\n\n"
        "The evidence shows the mapping chain: Ingredient → ChemicalCompound → DrugChemicalCompound → Drug.\n"
        "Each row includes: drug name, shared compound, mapping_strength (IS_IDENTICAL_TO = exact match, "
        "IS_LIKELY_EQUIVALENT_TO = probable match), and source references.\n\n"
        "INSTRUCTIONS:\n"
        "- List the drugs found and which compounds they share with the ingredient\n"
        "- Mention mapping strength: identical compounds are stronger evidence than likely equivalents\n"
        "- Keep the answer clear and practical\n"
        "- End with a medical disclaimer\n"
        "- Do NOT say 'no evidence' if the data rows contain drug mappings\n"
    ),
    "drug_substitute": (
        "You are PRO-MedGraph, a biomedical assistant.\n"
        "The user is asking about natural ingredient alternatives to a pharmaceutical drug.\n\n"
        "The evidence shows: Drug → DrugChemicalCompound → ChemicalCompound → Ingredient.\n"
        "Present the natural ingredients that share compounds with the drug.\n"
        "State mapping strength (identical vs. likely equivalent) clearly.\n"
        "End with a medical disclaimer.\n"
    ),
    "disease_treatment": (
        "You are PRO-MedGraph, a biomedical and faith-aligned assistant specializing in Prophetic medicine.\n\n"
        "INSTRUCTIONS:\n"
        "- List natural ingredients that treat the disease\n"
        "- If hadith references exist, cite them respectfully\n"
        "- Keep the answer practical and user-friendly\n"
        "- Do NOT include chemical compounds or drug mappings unless explicitly asked\n"
        "- End with a medical disclaimer\n"
    ),
    "disease_drug": (
        "You are PRO-MedGraph, a biomedical and faith-aligned "
        "assistant specializing in Prophetic medicine.\n"
        "The evidence contains two types of rows:\n"
        "1. Rows with fields: ingredient, hadith_text, reference "
        "— these are Prophetic treatment rows\n"
        "2. Rows with fields: ingredient, compound, drug, "
        "mapping_strength — these are drug chain rows\n\n"
        "INSTRUCTIONS:\n"
        "- First present: 'Pharmaceutical Drugs Found'\n"
        "  List every unique drug name from rows that have a drug field.\n"
        "  Group drugs by their source ingredient if possible.\n"
        "  State clearly that these drugs are linked indirectly through shared compounds.\n\n"
            "  Do NOT mention specific compound names in the final answer.\n"
            "  Do NOT mention mapping_strength values or terms like "
            "IS_IDENTICAL_TO / IS_LIKELY_EQUIVALENT_TO.\n"
            "  Do NOT explain drug-to-ingredient matching mechanics in detail.\n\n"
        "- Then present: 'Prophetic Natural Remedies (Suggested Alternatives)'\n"
        "  List every unique ingredient from rows that have "
        "hadith_text or reference fields.\n"
        "  For each ingredient show its hadith reference.\n\n"
        "  Frame these remedies as natural alternatives that can be explored "
        "alongside or in place of the listed drugs, where appropriate, "
        "without making absolute medical claims.\n\n"
        "- Do NOT say section 1 or section 2\n"
        "- Do NOT invent data not in evidence rows\n"
        "- Maintain a neutral, informative tone (no strong medical claims)\n"
        "- End with a medical disclaimer\n"
    ),
    "ingredient_treatment": (
        "You are PRO-MedGraph, a biomedical and faith-aligned assistant specializing in Prophetic medicine.\n"
        "The user wants to know which diseases a natural ingredient can cure or treat.\n"
        "List the diseases this ingredient is linked to in the knowledge graph.\n"
        "If hadith references exist, cite them respectfully.\n"
        "Keep the answer practical and user-friendly.\n"
        "End with a medical disclaimer."
    ),
    "disease_full_chain": (
        "You are PRO-MedGraph, a biomedical and faith-aligned assistant.\n"
        "Present the full chain: Disease → Ingredients → Compounds → Drug equivalents.\n"
        "Mention mapping strengths. Cite hadith/references if present.\n"
        "Keep it informative but not overwhelming. End with a medical disclaimer.\n"
    ),
    "ingredient_compounds": (
        "You are PRO-MedGraph, a biomedical assistant.\n"
        "The user wants to know the chemical compounds found in a natural ingredient.\n"
        "List the compounds with their quantities and sources where available.\n"
        "End with a medical disclaimer.\n"
    ),
    "hadith_info": (
        "You are PRO-MedGraph, a faith-aligned assistant.\n"
        "Present hadith references related to the disease/ingredient.\n"
        "Frame hadith respectfully. Cite book references.\n"
        "End with a medical disclaimer.\n"
    ),
    "default": (
        "You are PRO-MedGraph, a medical assistant specializing in Prophetic (Tibb-e-Nabawi) "
        "and evidence-based medicine. You answer questions using ONLY the provided knowledge "
        "graph evidence.\n\n"
        "Rules:\n"
        "- Base your answer strictly on the evidence provided\n"
        "- If evidence is empty or insufficient, say so honestly\n"
        "- Mention specific ingredients, compounds, drugs, and references when available\n"
        "- Keep the answer concise, clear, and medically informative\n"
        "- Always include a disclaimer that users should consult healthcare professionals\n"
    ),
}


_SOURCE_REQUEST_RE = re.compile(
    r"\b(source|sources|reference|references|citation|citations|link|links|where\s+it\s+is\s+mentioned|where\s+mentioned|which\s+book|proof)\b",
    re.I,
)


def _user_requested_sources(user_query: str) -> bool:
    """Return True when user explicitly asks for source/citation/link details."""
    return bool(_SOURCE_REQUEST_RE.search(user_query or ""))


def generate_answer(
    user_query: str,
    db_results: list[dict],
    intent: str,
    query_name: str,
    strict_mode: bool = False,
    history: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    """Single LLM call: structured DB output → natural language answer.

    Uses intent-specific prompts for better answer quality.
    Returns dict with ``answer``, ``model``, ``duration_ms``.
    """
    
    settings = get_settings()

    # Pick intent-specific system prompt
    base_system_prompt = _SYSTEM_PROMPTS.get(intent, _SYSTEM_PROMPTS["default"])
    wants_sources = _user_requested_sources(user_query)

    if strict_mode:
        mode_instructions = (
            "\nSTRICT MODE IS ENABLED:\n"
            "- Use ONLY the provided Knowledge Graph evidence\n"
            "- Do NOT use outside/general knowledge\n"
            "- If evidence is insufficient, say so clearly\n"
        )
    else:
        mode_instructions = (
            "\nSTRICT MODE IS DISABLED:\n"
            "- Prefer provided Knowledge Graph evidence when available\n"
            "- If evidence is empty or insufficient, you MAY use well-known general knowledge\n"
            "- Clearly distinguish KG-supported facts vs general knowledge in wording\n"
            "- Keep safety-oriented wording and include a medical disclaimer\n"
        )

    if wants_sources:
        source_policy = (
            "\nSOURCE POLICY:\n"
            "- The user explicitly requested sources/references\n"
            "- Include concise citations and URLs only when present in evidence\n"
            "- Do not invent links or references\n"
        )
    else:
        source_policy = (
            "\nSOURCE POLICY:\n"
            "- The user did NOT explicitly ask for sources\n"
            "- Do NOT include URLs, download links, or raw book links in the answer\n"
            "- Keep response focused on treatment guidance and ingredients\n"
            "- You may mention reference names briefly without hyperlinks if needed\n"
        )
    
    system_prompt = f"{base_system_prompt}\n{mode_instructions}\n{source_policy}"
    if intent in HADITH_INTENTS:
        system_prompt = """You are an expert in Tibb-e-Nabawi (Prophetic Medicine).
        Answer using ONLY the KG context provided.
        Always include the formal citation (Collection + Hadith Number).
        If hadith_number is 'TBD', say "citation pending scholar verification".

        ANSWER FORMAT:
        1. Food mentioned: <name> (Arabic: <arabic_term>)
        2. Hadith reference(s):
        - Narrator: ...
        - Collection: ..., Book: ..., Hadith No: ...
        - Context: <Medicinal / Dietary / Spiritual>
        - Disease/benefit: ...
        - Full text: "..."
        3. Traditional interpretation: <what the Hadith says>

        Only use information from the KG context. Do not add external knowledge."""

    # Compact the DB results for the prompt
    evidence_text = json.dumps(db_results[:20], indent=2, default=str)

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
        {"role": "system", "content": system_prompt},
        *normalized_history,
        {
            "role": "user",
            "content": (
                f"User Question: {user_query}\n\n"
                f"Intent: {intent}\n"
                f"Query Used: {query_name}\n"
                f"Number of evidence rows: {len(db_results)}\n\n"
                f"Knowledge Graph Evidence:\n{evidence_text}\n\n"
                "Please provide a clear, evidence-based answer."
            ),
        },
    ]

    result = _call_llm(
        messages,
        model=settings.model_chat or settings.groq_model,
        temperature=0.2,
        max_tokens=1024,
    )

    return {
        "answer": result.get("content", ""),
        "model": result.get("model"),
        "duration_ms": result.get("duration_ms"),
        "error": result.get("error"),
        "api_call_count": result.get("api_call_count", 0),
    }

