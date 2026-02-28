"""
Rationale-Guided Retrieval Planner (RAG² style) for PRO-MedGraph.

Before KG traversal, asks the LLM to produce a short reasoning plan that
identifies the biomedical steps needed to answer the user query.  The plan
is then used to filter and prioritise which parts of the KG subgraph are
actually relevant, avoiding irrelevant node retrieval and improving LLM
grounding quality.

Pipeline position:
    Entity Extraction → **Rationale Planner** → Intent Routing → Graph Retrieval

The planner output is a structured JSON list of reasoning steps.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

# ── Default reasoning plan when LLM fails ──────────────────────────────────────

_DEFAULT_PLAN: list[dict[str, str]] = [
    {"step": "1", "action": "Identify disease from query", "target": "Disease"},
    {"step": "2", "action": "Find ingredients that cure the disease", "target": "Ingredient"},
    {"step": "3", "action": "Extract chemical compounds from ingredients", "target": "ChemicalCompound"},
    {"step": "4", "action": "Match compounds to drug equivalents", "target": "DrugChemicalCompound"},
    {"step": "5", "action": "Identify linked pharmaceutical drugs", "target": "Drug"},
]


def _extract_json_array(raw: str) -> list[dict] | None:
    """Attempt to parse a JSON array from raw LLM output."""
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return parsed
    except json.JSONDecodeError:
        pass

    match = re.search(r"\[[\s\S]*\]", raw)
    if match:
        try:
            parsed = json.loads(match.group(0))
            return parsed if isinstance(parsed, list) else None
        except json.JSONDecodeError:
            return None
    return None


async def generate_rationale_plan(
    query: str,
    entities: dict[str, str | None],
    llm_service: Any,
    model: str,
) -> dict[str, Any]:
    """
    Ask the LLM to produce a reasoning plan for the given query.

    Returns:
        {
            "rationale_plan": [{"step": "1", "action": "...", "target": "NodeType"}, ...],
            "plan_source": "llm" | "default",
            "relevant_node_types": ["Disease", "Ingredient", ...],
        }
    """
    system_prompt = (
        "You are a biomedical reasoning planner for PRO-MedGraph.\n"
        "Given a user query and detected entities, produce a JSON array of reasoning steps.\n"
        "Each step must have: step (number), action (what to do), target (KG node type).\n"
        "Valid targets: Disease, Ingredient, ChemicalCompound, DrugChemicalCompound, Drug, Hadith.\n"
        "Return ONLY a JSON array. No markdown, no explanation.\n"
        "Steps should follow the biomedical causal chain logically.\n"
        "If the query asks about drugs, include steps to trace back to ingredients.\n"
        "If the query asks about ingredients, include steps to trace forward to drugs."
    )

    entity_summary = ", ".join(
        f"{k}={v}" for k, v in (entities or {}).items() if v
    ) or "none detected"

    user_prompt = (
        f"Query: {query}\n"
        f"Detected entities: {entity_summary}\n\n"
        "Produce the reasoning plan as a JSON array."
    )

    try:
        raw = await llm_service.generate_completion(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.1,
            model=model,
        )

        plan = _extract_json_array(raw)
        if plan and len(plan) >= 2:
            # Extract relevant node types from the plan
            relevant_types = list(dict.fromkeys(
                step.get("target")
                for step in plan
                if isinstance(step, dict) and isinstance(step.get("target"), str)
            ))

            logger.info("Rationale plan generated: %d steps, node_types=%s", len(plan), relevant_types)
            return {
                "rationale_plan": plan,
                "plan_source": "llm",
                "relevant_node_types": relevant_types,
            }
    except Exception:
        logger.exception("Rationale plan generation failed, using default plan")

    # Fallback to default plan
    relevant_types = [step["target"] for step in _DEFAULT_PLAN]
    return {
        "rationale_plan": _DEFAULT_PLAN,
        "plan_source": "default",
        "relevant_node_types": relevant_types,
    }


def filter_subgraph_by_plan(
    subgraph: dict,
    relevant_node_types: list[str],
) -> dict:
    """
    Optionally prune subgraph nodes not mentioned in the rationale plan.
    For now, this is conservative — it only logs pruning opportunities but
    keeps all data intact to avoid information loss.  Future implementations
    can perform actual filtering when the plan is high-confidence.
    """
    if not relevant_node_types:
        return subgraph

    # Map plan targets to subgraph keys
    type_to_key = {
        "Disease": ["Disease", "Diseases"],
        "Ingredient": ["Ingredient", "Ingredients"],
        "ChemicalCompound": ["ChemicalCompounds"],
        "DrugChemicalCompound": ["DrugChemicalCompounds"],
        "Drug": ["Drug", "Drugs"],
        "Hadith": ["HadithReferences"],
    }

    present_keys = set()
    for node_type in relevant_node_types:
        for key in type_to_key.get(node_type, []):
            present_keys.add(key)

    # Log which parts of subgraph are not in the plan
    for key in subgraph:
        if key in ("Relations", "error", "HadithReferences"):
            continue
        if key not in present_keys and isinstance(subgraph.get(key), (list, dict)):
            items = subgraph[key]
            count = len(items) if isinstance(items, list) else 1
            if count > 0:
                logger.debug(
                    "Subgraph key '%s' (%d items) not in rationale plan targets",
                    key, count,
                )

    # Conservative: return subgraph as-is for now to preserve all evidence
    return subgraph


__all__ = [
    "generate_rationale_plan",
    "filter_subgraph_by_plan",
]
