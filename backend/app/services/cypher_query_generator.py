"""
PRO-MedGraph  —  Step 3: Dynamic Cypher Query Generator
========================================================

Accepts the validated intent JSON produced by Step 2
(:mod:`app.services.intent_extractor`) and generates a **production-ready
Neo4j Cypher query** that can be executed directly against the PRO-MedGraph
knowledge graph.

Two generation strategies
-------------------------
1. **Template-based (deterministic)** — for well-known intent types we emit
   hand-crafted, battle-tested Cypher templates that already power the
   existing ``graph_service.py``.  This is the default path and guarantees
   syntactic correctness.

2. **LLM-based (dynamic)** — when the intent does not match a template *or*
   the caller explicitly requests it, we fall back to ``MODEL_KG`` via
   :func:`app.services.multi_model_service._call_groq` with a strict
   system prompt that embeds the **full KG schema**.  The raw LLM output is
   then validated before being returned.

Why two strategies?
~~~~~~~~~~~~~~~~~~~
Template queries are deterministic, fast, and free (no API call).  But the
knowledge graph may evolve — new node labels, new relationships — and users
may ask questions that fall outside the six known intent types.  The LLM
path handles that long-tail gracefully while the template path keeps the
happy path rock-solid.

N-hop expansion
---------------
Single-entity intents produce 1–2 hop queries (e.g. ``Disease → Ingredient``).
Multi-hop intents (drug interactions, compound mappings) expand to 3–4 hops
(e.g. ``Disease → Ingredient → ChemicalCompound → DrugChemicalCompound → Drug``).
The caller can override with ``max_hops``.

Usage
-----
.. code-block:: python

    from app.services.cypher_query_generator import generate_cypher

    intent = {
        "intent_type": "ask_remedy",
        "entities": [{"category": "condition", "value": "hypertension"}],
        "confidence_score": 0.92
    }
    cypher = await generate_cypher(intent)
    # → "MATCH (d:Disease) WHERE toLower(d.name) = toLower($entity_name) ..."

Run tests: ``python -m app.services.cypher_query_generator``
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════════
#  Actual KG schema (sourced from knowledge_graph_schema.json)
# ═══════════════════════════════════════════════════════════════════════════════
#
#  Embedded here so the LLM system prompt always reflects reality and we
#  don't need file I/O at import time.

KG_NODE_LABELS: list[str] = [
    "Disease",
    "DiseaseCategory",
    "Ingredient",
    "Cure",
    "ChemicalCompound",
    "Hadith",
    "Reference",
    "Drug",
    "DrugChemicalCompound",
    "Book",
]

KG_RELATIONSHIP_TYPES: list[str] = [
    "HAS_DISEASE",       # DiseaseCategory → Disease
    "HAS_HADITH",        # Reference → Hadith
    "CONTAINS",          # Ingredient → ChemicalCompound ; Drug → DrugChemicalCompound
    "CURES",             # Ingredient → Disease
    "MENTIONED_IN",      # Disease → Hadith
    "IS_IN_BOOK",        # Drug → Book
    "IS_IDENTICAL_TO",   # ChemicalCompound → DrugChemicalCompound (strength: IDENTICAL)
    "IS_LIKELY_EQUIVALENT_TO",  # ChemicalCompound → DrugChemicalCompound (strength: LIKELY)
    "IS_WEAK_MATCH_TO",         # ChemicalCompound → DrugChemicalCompound (strength: WEAK)
]

KG_RELATIONSHIP_PROPERTIES: dict[str, list[str]] = {
    "CONTAINS": ["source", "quantity", "unit", "food_part"],
}

# Compact text representation injected into the LLM system prompt.
_SCHEMA_TEXT: str = (
    "Node labels: " + ", ".join(KG_NODE_LABELS) + "\n"
    "Relationship types:\n"
    "  DiseaseCategory -[:HAS_DISEASE]-> Disease\n"
    "  Ingredient -[:CURES]-> Disease\n"
    "  Ingredient -[:CONTAINS]-> ChemicalCompound\n"
    "  ChemicalCompound -[:IS_IDENTICAL_TO]-> DrugChemicalCompound\n"
    "  ChemicalCompound -[:IS_LIKELY_EQUIVALENT_TO]-> DrugChemicalCompound\n"
    "  ChemicalCompound -[:IS_WEAK_MATCH_TO]-> DrugChemicalCompound\n"
    "  Drug -[:CONTAINS]-> DrugChemicalCompound\n"
    "  Drug -[:IS_IN_BOOK]-> Book\n"
    "  Disease -[:MENTIONED_IN]-> Hadith\n"
    "  Reference -[:HAS_HADITH]-> Hadith\n"
    "CONTAINS relationship properties: source, quantity, unit, food_part\n"
    "All nodes have a 'name' property (String). Reference has 'reference'. Book has 'name' and 'link'."
)

# ═══════════════════════════════════════════════════════════════════════════════
#  Cypher validation
# ═══════════════════════════════════════════════════════════════════════════════


class CypherSyntaxError(ValueError):
    """Raised when a generated Cypher query fails basic syntax checks."""


# Disallowed keywords that indicate destructive operations.
_DESTRUCTIVE_KEYWORDS: re.Pattern = re.compile(
    r"\b(CREATE|MERGE|DELETE|DETACH|SET|REMOVE|DROP|CALL\s+db\.)\b",
    re.IGNORECASE,
)

# Minimum tokens a valid read query must contain.
_MIN_QUERY_TOKENS: int = 3


def validate_cypher(query: str) -> str:
    """
    Perform basic safety and syntax validation on a Cypher query string.

    Checks
    ------
    1. Non-empty and contains a ``MATCH`` or ``RETURN`` keyword.
    2. All opened parentheses / brackets are closed.
    3. No destructive write keywords (``CREATE``, ``DELETE``, etc.).
    4. Contains at least one node pattern ``(…)`` or relationship pattern ``[…]``.

    Returns
    -------
    str
        The validated (and stripped) query.

    Raises
    ------
    CypherSyntaxError
        If any check fails.
    """
    stripped = query.strip()
    if not stripped:
        raise CypherSyntaxError("Cypher query is empty")

    # -- Must contain a MATCH or RETURN (reads only) -----------------------
    if not re.search(r"\b(MATCH|RETURN)\b", stripped, re.IGNORECASE):
        raise CypherSyntaxError(
            "Query does not contain MATCH or RETURN — not a valid read query"
        )

    # -- No destructive write operations -----------------------------------
    destructive = _DESTRUCTIVE_KEYWORDS.search(stripped)
    if destructive:
        raise CypherSyntaxError(
            f"Query contains disallowed write keyword: '{destructive.group()}'"
        )

    # -- Balanced parentheses / brackets -----------------------------------
    for open_ch, close_ch, label in [("(", ")", "parentheses"), ("[", "]", "brackets")]:
        depth = 0
        for ch in stripped:
            if ch == open_ch:
                depth += 1
            elif ch == close_ch:
                depth -= 1
            if depth < 0:
                raise CypherSyntaxError(f"Unbalanced {label}: unexpected '{close_ch}'")
        if depth != 0:
            raise CypherSyntaxError(f"Unbalanced {label}: {depth} unclosed")

    # -- Must contain at least one node pattern ----------------------------
    if "(" not in stripped:
        raise CypherSyntaxError("Query has no node pattern — missing '(…)'")

    logger.debug("Cypher validation passed (%d chars)", len(stripped))
    return stripped


# ═══════════════════════════════════════════════════════════════════════════════
#  Template-based Cypher generation (deterministic, no API call)
# ═══════════════════════════════════════════════════════════════════════════════
#
#  Each template mirrors the production queries in graph_service.py and uses
#  a $entity_name parameter so the caller binds the actual value at run time
#  (prevents injection).

# ── Helper: pick the primary entity from the intent ───────────────────────────

def _primary_entity(intent: dict[str, Any]) -> tuple[str, str]:
    """
    Return ``(category, value)`` of the most relevant entity in the intent.

    Priority: condition > drug > food > symptom > dosage.
    Raises ``ValueError`` if there are no entities.
    """
    entities = intent.get("entities") or []
    priority = ["condition", "drug", "food", "symptom", "dosage"]
    by_cat: dict[str, str] = {}
    for ent in entities:
        cat = ent.get("category", "")
        val = ent.get("value", "")
        if cat and val:
            by_cat.setdefault(cat, val)
    for cat in priority:
        if cat in by_cat:
            return cat, by_cat[cat]
    if by_cat:
        first_cat = next(iter(by_cat))
        return first_cat, by_cat[first_cat]
    raise ValueError("Intent JSON contains no entities to build a query for")


def _all_entities_by_category(intent: dict[str, Any]) -> dict[str, list[str]]:
    """Group entity values by category."""
    out: dict[str, list[str]] = {}
    for ent in intent.get("entities") or []:
        cat = ent.get("category", "")
        val = ent.get("value", "")
        if cat and val:
            out.setdefault(cat, []).append(val)
    return out


# ── Templates ─────────────────────────────────────────────────────────────────

def _template_ask_remedy(intent: dict[str, Any], max_hops: int) -> str | None:
    """
    ask_remedy + condition entity → Disease → Ingredient path (1–2 hops).

    If max_hops >= 3, extends the query to include the full biochemical chain
    Disease → Ingredient → ChemicalCompound → DrugChemicalCompound → Drug.
    """
    cat, val = _primary_entity(intent)
    if cat != "condition":
        return None  # fall through to LLM

    if max_hops >= 3:
        # Full multi-hop: 4 hops (Disease → Ingredient → Compound → DrugCompound → Drug)
        return (
            "// ask_remedy — full multi-hop chain\n"
            "MATCH (d:Disease)\n"
            "WHERE toLower(d.name) = toLower($entity_name)\n"
            "OPTIONAL MATCH (d)<-[:CURES]-(i:Ingredient)\n"
            "OPTIONAL MATCH (i)-[:CONTAINS]->(cc:ChemicalCompound)\n"
            "OPTIONAL MATCH (cc)-[mapping:IS_IDENTICAL_TO|IS_LIKELY_EQUIVALENT_TO|IS_WEAK_MATCH_TO]->(dcc:DrugChemicalCompound)\n"
            "OPTIONAL MATCH (drug:Drug)-[:CONTAINS]->(dcc)\n"
            "OPTIONAL MATCH (d)-[:MENTIONED_IN]->(h:Hadith)\n"
            "RETURN d.name AS disease,\n"
            "       collect(DISTINCT i.name) AS ingredients,\n"
            "       collect(DISTINCT cc.name) AS compounds,\n"
            "       collect(DISTINCT {compound: cc.name, drugCompound: dcc.name, strength: type(mapping)}) AS mappings,\n"
            "       collect(DISTINCT drug.name) AS drugs,\n"
            "       collect(DISTINCT h.name) AS hadith_references\n"
            "LIMIT 300"
        )

    # Default 1–2 hops: Disease → Ingredient + Hadith
    return (
        "// ask_remedy — 1-2 hop retrieval\n"
        "MATCH (d:Disease)\n"
        "WHERE toLower(d.name) = toLower($entity_name)\n"
        "OPTIONAL MATCH (d)<-[:CURES]-(i:Ingredient)\n"
        "OPTIONAL MATCH (d)-[:MENTIONED_IN]->(h:Hadith)\n"
        "RETURN d.name AS disease,\n"
        "       collect(DISTINCT i.name) AS ingredients,\n"
        "       collect(DISTINCT h.name) AS hadith_references\n"
        "LIMIT 300"
    )


def _template_drug_interaction(intent: dict[str, Any], max_hops: int) -> str | None:
    """
    drug_interaction → find shared ChemicalCompound / DrugChemicalCompound
    nodes between a drug and an ingredient / food.

    Multi-hop: Drug → DrugChemicalCompound ← ChemicalCompound ← Ingredient → Disease
    """
    cats = _all_entities_by_category(intent)
    drug_val = (cats.get("drug") or [None])[0]
    food_val = (cats.get("food") or cats.get("condition") or [None])[0]

    if drug_val and food_val:
        # Two entities: find shared compound links
        return (
            "// drug_interaction — shared compound path\n"
            "MATCH (drug:Drug)-[:CONTAINS]->(dcc:DrugChemicalCompound)\n"
            "  <-[:IS_IDENTICAL_TO|IS_LIKELY_EQUIVALENT_TO|IS_WEAK_MATCH_TO]-(cc:ChemicalCompound)\n"
            "  <-[:CONTAINS]-(i:Ingredient)\n"
            "WHERE toLower(drug.name) = toLower($drug_name)\n"
            "  AND toLower(i.name) = toLower($food_name)\n"
            "RETURN drug.name AS drug,\n"
            "       i.name AS ingredient,\n"
            "       collect(DISTINCT {compound: cc.name, drugCompound: dcc.name}) AS shared_compounds\n"
            "LIMIT 200"
        )

    if drug_val:
        # Single drug: retrieve its compound tree
        return (
            "// drug_interaction — single drug compound tree\n"
            "MATCH (drug:Drug)-[:CONTAINS]->(dcc:DrugChemicalCompound)\n"
            "OPTIONAL MATCH (cc:ChemicalCompound)-[mapping:IS_IDENTICAL_TO|IS_LIKELY_EQUIVALENT_TO|IS_WEAK_MATCH_TO]->(dcc)\n"
            "OPTIONAL MATCH (i:Ingredient)-[:CONTAINS]->(cc)\n"
            "WHERE toLower(drug.name) = toLower($entity_name)\n"
            "RETURN drug.name AS drug,\n"
            "       collect(DISTINCT {compound: cc.name, drugCompound: dcc.name, strength: type(mapping), ingredient: i.name}) AS compound_map\n"
            "LIMIT 200"
        )

    return None  # fall through to LLM


def _template_food_remedy(intent: dict[str, Any], max_hops: int) -> str | None:
    """
    food_remedy + food entity → Ingredient subgraph.

    Ingredient → Disease, Ingredient → ChemicalCompound → DrugChemicalCompound → Drug
    """
    cat, val = _primary_entity(intent)
    if cat != "food":
        return None

    if max_hops >= 3:
        return (
            "// food_remedy — full ingredient subgraph\n"
            "MATCH (i:Ingredient)\n"
            "WHERE toLower(i.name) = toLower($entity_name)\n"
            "OPTIONAL MATCH (i)-[:CURES]->(d:Disease)\n"
            "OPTIONAL MATCH (i)-[:CONTAINS]->(cc:ChemicalCompound)\n"
            "OPTIONAL MATCH (cc)-[mapping:IS_IDENTICAL_TO|IS_LIKELY_EQUIVALENT_TO|IS_WEAK_MATCH_TO]->(dcc:DrugChemicalCompound)\n"
            "OPTIONAL MATCH (drug:Drug)-[:CONTAINS]->(dcc)\n"
            "RETURN i.name AS ingredient,\n"
            "       collect(DISTINCT d.name) AS diseases_cured,\n"
            "       collect(DISTINCT cc.name) AS compounds,\n"
            "       collect(DISTINCT {compound: cc.name, drugCompound: dcc.name, strength: type(mapping), drug: drug.name}) AS drug_mappings\n"
            "LIMIT 300"
        )

    return (
        "// food_remedy — ingredient to disease\n"
        "MATCH (i:Ingredient)\n"
        "WHERE toLower(i.name) = toLower($entity_name)\n"
        "OPTIONAL MATCH (i)-[:CURES]->(d:Disease)\n"
        "RETURN i.name AS ingredient,\n"
        "       collect(DISTINCT d.name) AS diseases_cured\n"
        "LIMIT 300"
    )


def _template_symptom_check(intent: dict[str, Any], max_hops: int) -> str | None:
    """
    symptom_check → treat the symptom value as a disease name and look up
    ingredients that cure it (the KG stores symptoms as Disease nodes).
    """
    cat, val = _primary_entity(intent)
    if cat not in ("symptom", "condition"):
        return None

    return (
        "// symptom_check — symptom-as-disease lookup\n"
        "MATCH (d:Disease)\n"
        "WHERE toLower(d.name) CONTAINS toLower($entity_name)\n"
        "OPTIONAL MATCH (d)<-[:CURES]-(i:Ingredient)\n"
        "OPTIONAL MATCH (dc:DiseaseCategory)-[:HAS_DISEASE]->(d)\n"
        "RETURN d.name AS disease,\n"
        "       dc.name AS category,\n"
        "       collect(DISTINCT i.name) AS ingredients\n"
        "LIMIT 200"
    )


def _template_dosage_info(intent: dict[str, Any], max_hops: int) -> str | None:
    """
    dosage_info → retrieve an ingredient or drug with dosage-related
    properties if they exist.
    """
    cat, val = _primary_entity(intent)

    if cat in ("food", "dosage"):
        # Look up ingredient dosage
        return (
            "// dosage_info — ingredient dosage\n"
            "MATCH (i:Ingredient)\n"
            "WHERE toLower(i.name) = toLower($entity_name)\n"
            "OPTIONAL MATCH (i)-[:CURES]->(d:Disease)\n"
            "RETURN i.name AS ingredient,\n"
            "       i.traditional_dosage AS dosage,\n"
            "       i.preparation_method AS preparation,\n"
            "       collect(DISTINCT d.name) AS diseases_cured\n"
            "LIMIT 100"
        )

    if cat == "drug":
        return (
            "// dosage_info — drug dosage\n"
            "MATCH (drug:Drug)\n"
            "WHERE toLower(drug.name) = toLower($entity_name)\n"
            "RETURN drug.name AS drug,\n"
            "       drug.standard_dosage AS dosage,\n"
            "       drug.contraindications AS contraindications,\n"
            "       drug.side_effects AS side_effects\n"
            "LIMIT 100"
        )

    return None


# Map intent_type → template function.
_TEMPLATE_DISPATCH: dict[str, Any] = {
    "ask_remedy": _template_ask_remedy,
    "drug_interaction": _template_drug_interaction,
    "food_remedy": _template_food_remedy,
    "symptom_check": _template_symptom_check,
    "dosage_info": _template_dosage_info,
}

# ═══════════════════════════════════════════════════════════════════════════════
#  LLM-based Cypher generation (fallback for unknown intents)
# ═══════════════════════════════════════════════════════════════════════════════

# System prompt for MODEL_KG.
#
# Key design choices:
#   • The **full** KG schema is embedded so the model never invents labels.
#   • "Return ONLY valid Cypher" — suppresses explanatory prose.
#   • "Use OPTIONAL MATCH" — ensures partial results are returned even when
#     some paths do not exist.
#   • "Use $entity_name as a parameter" — prevents Cypher injection.

_LLM_SYSTEM_PROMPT: str = (
    "You are a Neo4j Cypher query generator for the PRO-MedGraph biomedical "
    "knowledge graph.\n\n"
    f"SCHEMA:\n{_SCHEMA_TEXT}\n\n"
    "RULES:\n"
    "1. Return ONLY a valid Cypher READ query — no markdown fences, no "
    "   explanations, no extra text.\n"
    "2. Use $entity_name as the parameter placeholder for the primary entity "
    "   value.  For queries with two entities use $entity_name and $entity_name_2.\n"
    "3. Always use toLower() for case-insensitive matching.\n"
    "4. Use OPTIONAL MATCH for traversals beyond the anchor node so partial "
    "   results are still returned.\n"
    "5. Never use CREATE, MERGE, DELETE, SET, REMOVE, or DROP.\n"
    "6. Add a LIMIT clause (max 300) to prevent unbounded results.\n"
    "7. Return meaningful aliases in the RETURN clause.\n"
    "8. For multi-hop queries expand up to the number of hops specified.\n"
)

MAX_LLM_RETRIES: int = 2


async def _generate_cypher_via_llm(
    intent: dict[str, Any],
    max_hops: int,
) -> str:
    """
    Call MODEL_KG with the full schema prompt and validated intent JSON.

    The response is validated with :func:`validate_cypher`.  Up to
    ``MAX_LLM_RETRIES`` retries are attempted on validation failure.
    """
    from app.services.multi_model_service import _call_groq, config

    user_prompt = (
        f"Intent JSON:\n{json.dumps(intent, ensure_ascii=False, indent=2)}\n\n"
        f"Generate a Cypher query with up to {max_hops} hops of graph traversal."
    )

    last_error: Exception | None = None

    for attempt in range(1, MAX_LLM_RETRIES + 2):
        try:
            raw = await _call_groq(
                model=config.kg,
                system_prompt=_LLM_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                temperature=0.0,
            )

            # Strip markdown fences the model may wrap around the query.
            cleaned = raw.strip()
            cleaned = re.sub(r"^```(?:cypher)?\s*\n?", "", cleaned)
            cleaned = re.sub(r"\n?```\s*$", "", cleaned)
            cleaned = cleaned.strip()

            validated = validate_cypher(cleaned)
            logger.info(
                "LLM Cypher generated (attempt %d): %d chars",
                attempt, len(validated),
            )
            return validated

        except CypherSyntaxError as exc:
            last_error = exc
            logger.warning(
                "LLM Cypher attempt %d/%d failed validation: %s",
                attempt, MAX_LLM_RETRIES + 1, exc,
            )
        except Exception as exc:
            logger.exception("LLM Cypher generation hit a non-retryable error")
            raise RuntimeError(f"Cypher generation failed: {exc}") from exc

    raise CypherSyntaxError(
        f"LLM Cypher generation failed after {MAX_LLM_RETRIES + 1} attempts. "
        f"Last error: {last_error}"
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  Public API
# ═══════════════════════════════════════════════════════════════════════════════


async def generate_cypher(
    intent_json: dict[str, Any],
    *,
    max_hops: int = 4,
    force_llm: bool = False,
) -> dict[str, Any]:
    """
    Generate a Neo4j Cypher query from a Step 2 intent JSON.

    Parameters
    ----------
    intent_json : dict
        Validated intent from :func:`app.services.intent_extractor.extract_intent`.
        Must contain ``intent_type``, ``entities``, and ``confidence_score``.
    max_hops : int, default 4
        Maximum graph traversal depth.  Templates ignore this for known
        patterns (they pick the right depth automatically) but the LLM path
        uses it as guidance.
    force_llm : bool, default False
        Bypass template matching and always use the LLM to generate the
        query.  Useful for benchmarking or queries that need creative
        traversals.

    Returns
    -------
    dict
        ``{ "cypher": str, "params": dict, "source": "template"|"llm",
            "intent_type": str }``

        * ``cypher`` — the ready-to-execute Cypher string.
        * ``params`` — parameter bindings (e.g. ``{"entity_name": "hypertension"}``).
        * ``source`` — whether the query came from a template or the LLM.
        * ``intent_type`` — echoed from the input for traceability.

    Raises
    ------
    CypherSyntaxError
        If the generated query fails validation.
    ValueError
        If the intent contains no usable entities.
    """
    intent_type = intent_json.get("intent_type", "general")
    entities = intent_json.get("entities") or []

    # ── Build parameter bindings ───────────────────────────────────────────
    params: dict[str, str] = {}
    try:
        cat, val = _primary_entity(intent_json)
        params["entity_name"] = val
    except ValueError:
        # No entities — only "general" intents can survive this.
        if intent_type == "general":
            return {
                "cypher": "// No query needed for general/chat intent",
                "params": {},
                "source": "template",
                "intent_type": intent_type,
            }
        raise

    # Add secondary entity if present (for drug_interaction with two entities)
    if len(entities) >= 2:
        cats = _all_entities_by_category(intent_json)
        drug_vals = cats.get("drug", [])
        food_vals = cats.get("food", [])
        if drug_vals:
            params["drug_name"] = drug_vals[0]
        if food_vals:
            params["food_name"] = food_vals[0]

    # ── Strategy 1: Template ──────────────────────────────────────────────
    cypher: str | None = None
    source = "template"

    if not force_llm:
        template_fn = _TEMPLATE_DISPATCH.get(intent_type)
        if template_fn:
            try:
                cypher = template_fn(intent_json, max_hops)
            except ValueError:
                cypher = None  # missing entity — fall through to LLM

    # ── Strategy 2: LLM fallback ──────────────────────────────────────────
    if cypher is None:
        logger.info(
            "No template matched intent_type='%s' — falling back to LLM",
            intent_type,
        )
        cypher = await _generate_cypher_via_llm(intent_json, max_hops)
        source = "llm"

    # ── Final validation (templates are trusted but we double-check) ──────
    validated = validate_cypher(cypher)

    logger.info(
        "Cypher generated: source=%s intent=%s params=%s hops=%d",
        source, intent_type, list(params.keys()), max_hops,
    )

    return {
        "cypher": validated,
        "params": params,
        "source": source,
        "intent_type": intent_type,
    }


def generate_cypher_sync(
    intent_json: dict[str, Any],
    *,
    max_hops: int = 4,
    force_llm: bool = False,
) -> dict[str, Any]:
    """Blocking wrapper for non-async callers."""
    return asyncio.run(generate_cypher(intent_json, max_hops=max_hops, force_llm=force_llm))


# ═══════════════════════════════════════════════════════════════════════════════
#  Standalone test harness
# ═══════════════════════════════════════════════════════════════════════════════
#
#  Run with:  python -m app.services.cypher_query_generator
#  (from backend/ with .env in place)

_TEST_INTENTS: list[dict[str, Any]] = [
    # ── 1. Remedies for hypertension (condition → Disease lookup) ──────────
    {
        "label": "Remedies for hypertension",
        "intent": {
            "intent_type": "ask_remedy",
            "entities": [{"category": "condition", "value": "hypertension"}],
            "confidence_score": 0.92,
        },
        "max_hops": 2,
    },
    # ── 2. Full multi-hop: diabetes remedies through to drugs ──────────────
    {
        "label": "Full multi-hop — diabetes to drugs",
        "intent": {
            "intent_type": "ask_remedy",
            "entities": [{"category": "condition", "value": "diabetes"}],
            "confidence_score": 0.95,
        },
        "max_hops": 4,
    },
    # ── 3. Chemical mapping: ingredient to drug ────────────────────────────
    {
        "label": "Ingredient-to-drug compound mapping (black seed)",
        "intent": {
            "intent_type": "food_remedy",
            "entities": [{"category": "food", "value": "black seed"}],
            "confidence_score": 0.88,
        },
        "max_hops": 4,
    },
    # ── 4. Symptom + remedy relationship ───────────────────────────────────
    {
        "label": "Symptom check — persistent cough",
        "intent": {
            "intent_type": "symptom_check",
            "entities": [{"category": "symptom", "value": "cough"}],
            "confidence_score": 0.80,
        },
        "max_hops": 2,
    },
    # ── 5. Drug interaction (two entities) ─────────────────────────────────
    {
        "label": "Drug interaction — metformin + honey",
        "intent": {
            "intent_type": "drug_interaction",
            "entities": [
                {"category": "drug", "value": "metformin"},
                {"category": "food", "value": "honey"},
            ],
            "confidence_score": 0.85,
        },
        "max_hops": 4,
    },
    # ── 6. Dosage info for an ingredient ───────────────────────────────────
    {
        "label": "Dosage info — Nigella sativa",
        "intent": {
            "intent_type": "dosage_info",
            "entities": [{"category": "food", "value": "Nigella sativa"}],
            "confidence_score": 0.90,
        },
        "max_hops": 1,
    },
    # ── 7. General / off-topic (no query needed) ──────────────────────────
    {
        "label": "General greeting — no Cypher expected",
        "intent": {
            "intent_type": "general",
            "entities": [],
            "confidence_score": 0.99,
        },
        "max_hops": 1,
    },
    # ── 8. Single drug reverse lookup ──────────────────────────────────────
    {
        "label": "Drug interaction — aspirin (single drug)",
        "intent": {
            "intent_type": "drug_interaction",
            "entities": [{"category": "drug", "value": "aspirin"}],
            "confidence_score": 0.82,
        },
        "max_hops": 3,
    },
]


async def _run_tests() -> None:
    import sys

    logging.basicConfig(level=logging.INFO, stream=sys.stdout, format="%(levelname)s %(message)s")

    passed = 0
    failed = 0
    total = len(_TEST_INTENTS)

    print("=" * 72)
    print("  PRO-MedGraph · Cypher Query Generator Test Harness")
    print("=" * 72)

    for tc in _TEST_INTENTS:
        label = tc["label"]
        intent = tc["intent"]
        hops = tc.get("max_hops", 4)

        print(f"\n── {label} (max_hops={hops}) ──")
        print(f"   Intent: {json.dumps(intent, ensure_ascii=False)[:120]}…")

        try:
            result = await generate_cypher(intent, max_hops=hops)
            print(f"   Source:  {result['source']}")
            print(f"   Params:  {result['params']}")
            print(f"   Cypher:\n")
            for line in result["cypher"].splitlines():
                print(f"      {line}")
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
    "generate_cypher",
    "generate_cypher_sync",
    "validate_cypher",
    "CypherSyntaxError",
    "KG_NODE_LABELS",
    "KG_RELATIONSHIP_TYPES",
]
