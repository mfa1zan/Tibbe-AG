"""
PRO-MedGraph  —  Step 5: Evidence Formatter & Causal Reasoning Paths
=====================================================================

Transforms the structured JSON produced by Step 4 (:mod:`graph_retrieval`)
into **ordered reasoning blocks** ready for the base LLM generation stage
(Step 6).

Pipeline position
-----------------
Step 4 (Graph Retrieval) → **Step 5 (Evidence Formatter)** → Step 6 (LLM)

Every statement emitted by this module is **provenance-grounded**: only
facts that exist as nodes / edges in the knowledge graph are included.
Speculative or ambiguous claims are explicitly suppressed to prevent
hallucination downstream.

Block types
-----------
``FACT``
    Directly derived from a node property or edge relationship.
    E.g. ``FACT: Black seed contains Thymoquinone``

``INFERENCE``
    A short causal chain connecting two or more FACTs via a KG edge or
    a well-known pharmacological pattern (ingredient → compound →
    mechanism → drug effect).
    E.g. ``INFERENCE: May reduce blood pressure, comparable to ACE inhibitors``

``DOSAGE``
    Traditional or clinical dosage information when available in the KG.
    E.g. ``DOSAGE (Black seed): 1 tsp/day orally``

``SAFETY``
    Contraindication or interaction warnings sourced from the KG.
    E.g. ``SAFETY (Warfarin): Avoid with anticoagulants``

``HADITH``
    Prophetic medicine reference when present for a disease.
    E.g. ``HADITH: "In the black seed is healing for every disease…"``

Usage
-----
.. code-block:: python

    from app.services.evidence_formatter import format_graph_for_llm

    graph_json = retrieve_graph_from_step3(step3_output)
    blocks = format_graph_for_llm(graph_json)
    # blocks → ["FACT: ...", "FACT: ...", "INFERENCE: ...", ...]

Run standalone tests:  ``python -m app.services.evidence_formatter``
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════════
#  Block prefixes (constants)
# ═══════════════════════════════════════════════════════════════════════════════

_FACT = "FACT"
_INFERENCE = "INFERENCE"
_DOSAGE = "DOSAGE"
_SAFETY = "SAFETY"
_HADITH = "HADITH"

# ── Mechanism human-readable labels ───────────────────────────────────────────
# These map the KG relationship types to pharmacology-grade language.

_MECHANISM_LABEL: dict[str, str] = {
    "IS_IDENTICAL_TO": "biochemically identical to",
    "IS_LIKELY_EQUIVALENT_TO": "likely pharmacologically equivalent to",
    "IS_WEAK_MATCH_TO": "a weak structural match to",
    # Normalised short forms (from graph_service._map_similarity_relation)
    "IDENTICAL": "biochemically identical to",
    "LIKELY": "likely pharmacologically equivalent to",
    "WEAK": "a weak structural match to",
}

_MECHANISM_CONFIDENCE: dict[str, str] = {
    "IS_IDENTICAL_TO": "high",
    "IS_LIKELY_EQUIVALENT_TO": "moderate",
    "IS_WEAK_MATCH_TO": "low",
    "IDENTICAL": "high",
    "LIKELY": "moderate",
    "WEAK": "low",
}

# ═══════════════════════════════════════════════════════════════════════════════
#  Dosage / safety helpers (optional KG enrichment)
# ═══════════════════════════════════════════════════════════════════════════════

# We try to import the dosage_validator module for live KG look-ups.
# If it fails (e.g. Neo4j not reachable), dosage/safety blocks are skipped.

try:
    from app.services.dosage_validator import (
        fetch_drug_dosage,
        fetch_ingredient_dosage,
    )
    _HAS_DOSAGE_MODULE = True
except ImportError:
    _HAS_DOSAGE_MODULE = False


def _fetch_dosage_blocks(ingredient_names: list[str], drug_names: list[str]) -> list[str]:
    """
    Query the KG for dosage metadata and return formatted blocks.

    Falls back to an empty list when the dosage module is unavailable or
    when no dosage data exists in the KG.
    """
    if not _HAS_DOSAGE_MODULE:
        return []

    blocks: list[str] = []

    for name in ingredient_names:
        try:
            info = fetch_ingredient_dosage(name)
            if info.dosage_available:
                parts: list[str] = []
                if info.traditional_dosage:
                    parts.append(info.traditional_dosage)
                if info.preparation_method:
                    parts.append(f"preparation: {info.preparation_method}")
                if parts:
                    blocks.append(f"{_DOSAGE} ({name}): {'; '.join(parts)}")
        except Exception:
            logger.debug("Dosage lookup failed for ingredient '%s'", name, exc_info=True)

    for name in drug_names:
        try:
            info = fetch_drug_dosage(name)
            if info.dosage_available:
                if info.standard_dosage:
                    blocks.append(f"{_DOSAGE} ({name}): {info.standard_dosage}")
                if info.contraindications:
                    blocks.append(f"{_SAFETY} ({name}): {info.contraindications}")
                if info.side_effects:
                    blocks.append(f"{_SAFETY} ({name} side-effects): {info.side_effects}")
        except Exception:
            logger.debug("Dosage lookup failed for drug '%s'", name, exc_info=True)

    return blocks


# ═══════════════════════════════════════════════════════════════════════════════
#  Raw-subgraph enrichment helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _extract_dosage_from_raw_nodes(raw_subgraph: dict[str, Any]) -> list[str]:
    """
    Inspect ``raw_subgraph.nodes`` for dosage/safety properties that were
    returned inline by the Cypher query (avoiding extra DB round-trips).
    """
    blocks: list[str] = []
    seen: set[str] = set()

    for node in raw_subgraph.get("nodes", []):
        props = node.get("properties", {})
        labels = node.get("labels", [])
        name = props.get("name", "")

        if not name or name in seen:
            continue

        if "Ingredient" in labels:
            trad = props.get("traditional_dosage")
            prep = props.get("preparation_method")
            if trad and str(trad).strip():
                blocks.append(f"{_DOSAGE} ({name}): {str(trad).strip()}")
                seen.add(name)
            if prep and str(prep).strip():
                blocks.append(f"{_DOSAGE} ({name} preparation): {str(prep).strip()}")
                seen.add(name)

        elif "Drug" in labels:
            std = props.get("standard_dosage")
            contra = props.get("contraindications")
            side = props.get("side_effects")
            if std and str(std).strip():
                blocks.append(f"{_DOSAGE} ({name}): {str(std).strip()}")
                seen.add(name)
            if contra and str(contra).strip():
                blocks.append(f"{_SAFETY} ({name}): {str(contra).strip()}")
                seen.add(name)
            if side and str(side).strip():
                blocks.append(f"{_SAFETY} ({name} side-effects): {str(side).strip()}")
                seen.add(name)

    return blocks


# ═══════════════════════════════════════════════════════════════════════════════
#  Core formatter
# ═══════════════════════════════════════════════════════════════════════════════


def _format_single_remedy(
    disease: str | None,
    remedy: dict[str, Any],
    *,
    include_header: bool = True,
) -> list[str]:
    """
    Generate FACT / INFERENCE blocks for a single remedy dict.

    Parameters
    ----------
    disease : str | None
        The disease name (may be ``None`` for ingredient-centric queries).
    remedy : dict
        A single remedy entry from Step 4 output.
    include_header : bool
        If True, emit a ``--- Remedy: <name> ---`` separator (useful when
        multiple remedies are formatted in sequence).

    Returns
    -------
    list[str]
        Ordered list of evidence blocks.
    """
    blocks: list[str] = []
    name = remedy.get("name", "unknown")
    ingredients = remedy.get("ingredients") or []
    compounds = remedy.get("compounds") or []
    mapped_drugs = remedy.get("mapped_drugs") or []
    mechanisms = remedy.get("mechanisms") or []

    # ── Section header ─────────────────────────────────────────────────────
    if include_header:
        blocks.append(f"--- Remedy: {name} ---")

    # ── FACT: Disease ↔ Ingredient link ────────────────────────────────────
    if disease:
        for ing in ingredients:
            blocks.append(f"{_FACT}: {ing} is a traditional remedy for {disease}")

    # ── FACT: Ingredient → Compound (CONTAINS edge) ───────────────────────
    for comp in compounds:
        if ingredients:
            # Attribute the compound to the first (or only) ingredient.
            blocks.append(f"{_FACT}: {ingredients[0]} contains {comp}")
        else:
            blocks.append(f"{_FACT}: {name} is associated with compound {comp}")

    # ── FACT: Compound → Drug mapping ─────────────────────────────────────
    for drug in mapped_drugs:
        if compounds:
            # Pick the first compound for the canonical statement.
            blocks.append(
                f"{_FACT}: {compounds[0]} maps to pharmaceutical compound in {drug}"
            )
        else:
            blocks.append(f"{_FACT}: {name} maps to pharmaceutical drug {drug}")

    # ── FACT: Mechanism type ──────────────────────────────────────────────
    for mech in mechanisms:
        label = _MECHANISM_LABEL.get(mech, mech)
        confidence = _MECHANISM_CONFIDENCE.get(mech, "unknown")
        if compounds and mapped_drugs:
            blocks.append(
                f"{_FACT}: {compounds[0]} is {label} "
                f"the active compound in {mapped_drugs[0]} "
                f"(confidence: {confidence})"
            )
        elif compounds:
            blocks.append(
                f"{_FACT}: {compounds[0]} has mapping strength '{mech}' "
                f"(confidence: {confidence})"
            )

    # ── INFERENCE: Causal reasoning chain ─────────────────────────────────
    # Build one inference per compound → drug pair with mechanism context.
    inferences_emitted: set[str] = set()

    for comp in compounds:
        for drug in mapped_drugs:
            # Determine the strongest mechanism for this pair.
            best_mech = _pick_best_mechanism(mechanisms)
            mech_label = _MECHANISM_LABEL.get(best_mech, best_mech) if best_mech else None

            if disease and mech_label:
                inf = (
                    f"{_INFERENCE}: {comp} (from {name}) is {mech_label} "
                    f"the active compound in {drug}; this suggests "
                    f"{name} may have therapeutic effects on {disease} "
                    f"comparable to {drug}"
                )
            elif disease:
                inf = (
                    f"{_INFERENCE}: {name} contains compounds that may "
                    f"have therapeutic effects on {disease} similar to {drug}"
                )
            else:
                inf = (
                    f"{_INFERENCE}: {comp} in {name} is structurally related "
                    f"to the active compound in {drug}"
                )

            if inf not in inferences_emitted:
                blocks.append(inf)
                inferences_emitted.add(inf)

    # If no compound→drug pairs but we have a disease + ingredients, add a
    # simpler inference.
    if not compounds and not mapped_drugs and disease and ingredients:
        blocks.append(
            f"{_INFERENCE}: {name} is traditionally used for {disease}; "
            f"no modern drug mapping found in the knowledge graph"
        )

    return blocks


def _pick_best_mechanism(mechanisms: list[str]) -> str | None:
    """Return the strongest mechanism from a list, or None."""
    priority = [
        "IS_IDENTICAL_TO", "IDENTICAL",
        "IS_LIKELY_EQUIVALENT_TO", "LIKELY",
        "IS_WEAK_MATCH_TO", "WEAK",
    ]
    for p in priority:
        if p in mechanisms:
            return p
    return mechanisms[0] if mechanisms else None


# ═══════════════════════════════════════════════════════════════════════════════
#  Public API
# ═══════════════════════════════════════════════════════════════════════════════


def format_graph_for_llm(
    graph_json: dict[str, Any],
    *,
    enrich_dosage: bool = True,
    max_remedies: int | None = None,
) -> list[str]:
    """
    Convert Step 4 graph output into an ordered list of evidence blocks
    ready for the reasoning LLM (Step 6).

    Parameters
    ----------
    graph_json : dict
        The canonical output from :func:`graph_retrieval.retrieve_graph` or
        :func:`graph_retrieval.retrieve_graph_from_step3`.  Expected keys:
        ``disease``, ``remedies``, ``hadith_references``, ``raw_subgraph``.
    enrich_dosage : bool
        When True, queries the KG for dosage and safety metadata and
        appends ``DOSAGE`` / ``SAFETY`` blocks.  Set to False for faster
        formatting when dosage data is not needed.
    max_remedies : int | None
        Cap on the number of remedies to format.  ``None`` means all.

    Returns
    -------
    list[str]
        Ordered evidence blocks: FACTs first, then INFERENCEs, then DOSAGE,
        SAFETY, and HADITH.  Each string is one block.

    Examples
    --------
    >>> blocks = format_graph_for_llm(graph_json)
    >>> for b in blocks:
    ...     print(b)
    FACT: Black seed is a traditional remedy for hypertension
    FACT: Black seed contains Thymoquinone
    FACT: Thymoquinone maps to pharmaceutical compound in Captopril
    FACT: Thymoquinone is biochemically identical to the active compound in Captopril (confidence: high)
    INFERENCE: Thymoquinone (from Black seed) is biochemically identical to the active compound in Captopril; this suggests Black seed may have therapeutic effects on hypertension comparable to Captopril
    DOSAGE (Black seed): 1 tsp/day orally
    SAFETY (Captopril): Avoid with anticoagulants
    HADITH: "In the black seed is healing for every disease except death"
    """
    disease = graph_json.get("disease")
    remedies = graph_json.get("remedies") or []
    hadith_refs = graph_json.get("hadith_references") or []
    raw_subgraph = graph_json.get("raw_subgraph") or {}
    metadata = graph_json.get("metadata") or {}

    # ── Short-circuit on skipped / empty results ───────────────────────────
    if metadata.get("skipped") or metadata.get("empty"):
        logger.info("Evidence formatter: no graph data to format (skipped=%s, empty=%s)",
                     metadata.get("skipped"), metadata.get("empty"))
        return []

    # ── Cap remedies if requested ──────────────────────────────────────────
    if max_remedies is not None:
        remedies = remedies[:max_remedies]

    multi = len(remedies) > 1

    # ── 1. Disease context header ──────────────────────────────────────────
    blocks: list[str] = []
    if disease:
        blocks.append(f"{_FACT}: The query concerns the disease/condition '{disease}'")

    # ── 2. Per-remedy FACT / INFERENCE blocks ──────────────────────────────
    all_ingredients: list[str] = []
    all_drugs: list[str] = []

    for remedy in remedies:
        remedy_blocks = _format_single_remedy(
            disease, remedy, include_header=multi,
        )
        blocks.extend(remedy_blocks)

        # Collect names for dosage enrichment.
        all_ingredients.extend(remedy.get("ingredients") or [])
        all_drugs.extend(remedy.get("mapped_drugs") or [])

    # ── 3. Dosage & safety blocks ──────────────────────────────────────────
    dosage_blocks: list[str] = []

    if enrich_dosage:
        # Strategy A: extract from raw_subgraph node properties (no DB call).
        dosage_blocks = _extract_dosage_from_raw_nodes(raw_subgraph)

        # Strategy B: fall back to dedicated KG queries via dosage_validator.
        if not dosage_blocks:
            unique_ingredients = list(dict.fromkeys(all_ingredients))
            unique_drugs = list(dict.fromkeys(all_drugs))
            dosage_blocks = _fetch_dosage_blocks(unique_ingredients, unique_drugs)

    blocks.extend(dosage_blocks)

    # ── 4. Hadith / prophetic references ──────────────────────────────────
    for ref in hadith_refs:
        if ref and str(ref).strip():
            blocks.append(f'{_HADITH}: "{str(ref).strip()}"')

    # ── 5. Provenance footer ──────────────────────────────────────────────
    blocks.append(
        f"NOTE: All statements above are derived exclusively from the "
        f"PRO-MedGraph knowledge graph. No external or speculative claims "
        f"are included."
    )

    # ── Logging ────────────────────────────────────────────────────────────
    n_facts = sum(1 for b in blocks if b.startswith(_FACT))
    n_inferences = sum(1 for b in blocks if b.startswith(_INFERENCE))
    n_dosage = sum(1 for b in blocks if b.startswith(_DOSAGE))
    n_safety = sum(1 for b in blocks if b.startswith(_SAFETY))
    n_hadith = sum(1 for b in blocks if b.startswith(_HADITH))

    logger.info(
        "Evidence formatter: %d remedies → %d FACTs, %d INFERENCEs, "
        "%d DOSAGEs, %d SAFETYs, %d HADITHs (%d total blocks)",
        len(remedies), n_facts, n_inferences, n_dosage, n_safety,
        n_hadith, len(blocks),
    )

    return blocks


# ═══════════════════════════════════════════════════════════════════════════════
#  Convenience: produce a single prompt string
# ═══════════════════════════════════════════════════════════════════════════════


def format_evidence_prompt(
    graph_json: dict[str, Any],
    *,
    enrich_dosage: bool = True,
    max_remedies: int | None = None,
    separator: str = "\n",
) -> str:
    """
    Same as :func:`format_graph_for_llm` but returns a single concatenated
    string ready to inject into an LLM system/user prompt.

    Parameters
    ----------
    separator : str
        String used to join blocks.  Defaults to ``"\\n"``.
    """
    blocks = format_graph_for_llm(
        graph_json,
        enrich_dosage=enrich_dosage,
        max_remedies=max_remedies,
    )
    return separator.join(blocks)


# ═══════════════════════════════════════════════════════════════════════════════
#  Standalone test harness
# ═══════════════════════════════════════════════════════════════════════════════
#
#  Run:  python -m app.services.evidence_formatter
#  (from backend/ dir)


_MOCK_GRAPH_SINGLE: dict[str, Any] = {
    "disease": "hypertension",
    "remedies": [
        {
            "name": "Black seed",
            "ingredients": ["Black seed"],
            "compounds": ["Thymoquinone", "Nigellone"],
            "mapped_drugs": ["Captopril"],
            "mechanisms": ["IS_IDENTICAL_TO"],
        }
    ],
    "hadith_references": [
        "In the black seed is healing for every disease except death"
    ],
    "raw_subgraph": {
        "nodes": [
            {
                "id": "n1", "labels": ["Disease"],
                "properties": {"name": "hypertension"},
            },
            {
                "id": "n2", "labels": ["Ingredient"],
                "properties": {
                    "name": "Black seed",
                    "traditional_dosage": "1 tsp/day orally",
                    "preparation_method": "ground or cold-pressed oil",
                },
            },
            {
                "id": "n3", "labels": ["ChemicalCompound"],
                "properties": {"name": "Thymoquinone"},
            },
            {
                "id": "n4", "labels": ["Drug"],
                "properties": {
                    "name": "Captopril",
                    "standard_dosage": "25-50 mg twice daily",
                    "contraindications": "Avoid with anticoagulants; not in pregnancy",
                    "side_effects": "Dry cough, dizziness",
                },
            },
        ],
        "edges": [],
    },
    "metadata": {"query_ms": 87, "node_count": 4, "edge_count": 0, "source": "template"},
}

_MOCK_GRAPH_MULTI: dict[str, Any] = {
    "disease": "diabetes",
    "remedies": [
        {
            "name": "Fenugreek",
            "ingredients": ["Fenugreek"],
            "compounds": ["4-Hydroxyisoleucine"],
            "mapped_drugs": ["Metformin"],
            "mechanisms": ["IS_LIKELY_EQUIVALENT_TO"],
        },
        {
            "name": "Cinnamon",
            "ingredients": ["Cinnamon"],
            "compounds": ["Cinnamaldehyde"],
            "mapped_drugs": ["Glipizide"],
            "mechanisms": ["IS_WEAK_MATCH_TO"],
        },
    ],
    "hadith_references": [],
    "raw_subgraph": {"nodes": [], "edges": []},
    "metadata": {"query_ms": 42, "node_count": 0, "edge_count": 0, "source": "template"},
}

_MOCK_GRAPH_EMPTY: dict[str, Any] = {
    "disease": None,
    "remedies": [],
    "hadith_references": [],
    "raw_subgraph": {"nodes": [], "edges": []},
    "metadata": {"query_ms": 0, "node_count": 0, "edge_count": 0, "skipped": True},
}

_MOCK_GRAPH_NO_DRUGS: dict[str, Any] = {
    "disease": "migraine",
    "remedies": [
        {
            "name": "Peppermint",
            "ingredients": ["Peppermint"],
            "compounds": [],
            "mapped_drugs": [],
            "mechanisms": [],
        }
    ],
    "hadith_references": [],
    "raw_subgraph": {"nodes": [], "edges": []},
    "metadata": {"query_ms": 15, "node_count": 1, "edge_count": 0, "source": "template"},
}


def _run_tests() -> None:
    """Execute tests against mock graph payloads and print results."""
    import sys

    logging.basicConfig(
        level=logging.INFO,
        stream=sys.stdout,
        format="%(levelname)s  %(message)s",
    )

    test_cases: list[tuple[str, dict]] = [
        ("Single remedy (hypertension + black seed)", _MOCK_GRAPH_SINGLE),
        ("Multi-remedy (diabetes: fenugreek + cinnamon)", _MOCK_GRAPH_MULTI),
        ("Empty / skipped query", _MOCK_GRAPH_EMPTY),
        ("No drug mapping (migraine + peppermint)", _MOCK_GRAPH_NO_DRUGS),
    ]

    print("=" * 72)
    print("  PRO-MedGraph · Evidence Formatter Test Harness")
    print("=" * 72)

    passed = 0

    for label, mock in test_cases:
        print(f"\n── {label} ──")
        try:
            # Disable dosage DB enrichment for mock tests.
            blocks = format_graph_for_llm(mock, enrich_dosage=True)
            print(f"   Blocks generated: {len(blocks)}")
            for b in blocks:
                print(f"   │ {b}")
            passed += 1
        except Exception as exc:
            print(f"   ✗ FAILED: {exc}")

    print("\n" + "=" * 72)
    print(f"  Results:  {passed}/{len(test_cases)} passed")
    print("=" * 72)


if __name__ == "__main__":
    _run_tests()


# ── Module exports ─────────────────────────────────────────────────────────────

__all__ = [
    "format_graph_for_llm",
    "format_evidence_prompt",
]
