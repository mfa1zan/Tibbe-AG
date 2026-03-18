"""LLM service — lightweight entity extraction + single-call answer generation.

Uses the Groq API (OpenAI-compatible) via httpx.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

import httpx

from backend.core.config import get_settings

logger = logging.getLogger(__name__)

_TIMEOUT = httpx.Timeout(60.0, connect=10.0)


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

    t0 = time.perf_counter()
    try:
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

        logger.info("LLM [%s] → %d chars (%.0fms)", use_model, len(content), duration_ms)
        return {
            "content": content,
            "model": use_model,
            "duration_ms": duration_ms,
        }
    except Exception as exc:
        duration_ms = round((time.perf_counter() - t0) * 1000, 1)
        logger.exception("LLM call failed [%s] (%.0fms): %s", use_model, duration_ms, exc)
        return {
            "content": "",
            "model": use_model,
            "duration_ms": duration_ms,
            "error": str(exc),
        }


# ── Entity Extraction ────────────────────────────────────────────────────────


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
        "Frame hadith respectfully. Cite book references and links if available.\n"
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


def generate_answer(
    user_query: str,
    db_results: list[dict],
    intent: str,
    query_name: str,
    strict_mode: bool = False,
) -> dict[str, Any]:
    """Single LLM call: structured DB output → natural language answer.

    Uses intent-specific prompts for better answer quality.
    Returns dict with ``answer``, ``model``, ``duration_ms``.
    """
    settings = get_settings()

    # Pick intent-specific system prompt
    base_system_prompt = _SYSTEM_PROMPTS.get(intent, _SYSTEM_PROMPTS["default"])

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

    system_prompt = f"{base_system_prompt}\n{mode_instructions}"

    # Compact the DB results for the prompt
    evidence_text = json.dumps(db_results[:20], indent=2, default=str)

    messages = [
        {"role": "system", "content": system_prompt},
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
    }

