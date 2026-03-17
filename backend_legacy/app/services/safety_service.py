from __future__ import annotations

import copy
import logging
from typing import Any

logger = logging.getLogger(__name__)

_STRENGTH_WEIGHT = {
    "IDENTICAL": 1.0,
    "LIKELY": 0.7,
    "WEAK": 0.35,
}


def _ensure_dict(value: Any) -> dict:
    """Return a dictionary payload or an empty dict for malformed inputs."""
    return value if isinstance(value, dict) else {}


def _ensure_list(value: Any) -> list:
    """Return list payloads safely while avoiding type errors downstream."""
    return value if isinstance(value, list) else []


def _get_mapping_strengths(reasoning: dict) -> list[str]:
    """
    Read mapping strengths from BiochemicalMappings first, then fallback to
    DrugChemicalCompounds if needed.
    """
    strengths: list[str] = []

    mappings = _ensure_list(reasoning.get("BiochemicalMappings"))
    for mapping in mappings:
        payload = _ensure_dict(mapping)
        strength = payload.get("mapping_strength")
        if isinstance(strength, str) and strength in _STRENGTH_WEIGHT:
            strengths.append(strength)

    if strengths:
        return strengths

    # Fallback for partial reasoning payloads with no explicit path mappings.
    drug_compounds = _ensure_list(reasoning.get("DrugChemicalCompounds"))
    for compound in drug_compounds:
        payload = _ensure_dict(compound)
        strength = payload.get("mapping_strength") or payload.get("relation_type")
        if isinstance(strength, str) and strength in _STRENGTH_WEIGHT:
            strengths.append(strength)

    return strengths


def _graph_paths_used(reasoning: dict) -> int:
    """Count structured biochemical paths available for evidence grounding."""
    return len(_ensure_list(reasoning.get("BiochemicalMappings")))


def _has_drug_equivalent(reasoning: dict) -> bool:
    """
    Return True if at least one mapping includes a concrete linked Drug,
    otherwise False to signal limited biochemical alignment.
    """
    for mapping in _ensure_list(reasoning.get("BiochemicalMappings")):
        payload = _ensure_dict(mapping)
        drug = _ensure_dict(payload.get("drug"))
        drug_id = drug.get("id")
        drug_name = drug.get("name")
        if (isinstance(drug_id, str) and drug_id.strip()) or (isinstance(drug_name, str) and drug_name.strip()):
            return True
    return False


def _has_hadith_references(reasoning: dict) -> bool:
    """Check if the reasoning payload includes at least one hadith reference."""
    hadith = _ensure_list(reasoning.get("HadithReferences"))
    return len(hadith) > 0


def _compute_evidence_strength_label(strengths: list[str]) -> str:
    """
    Coarse evidence label used by orchestrator/UI:
    - strong: any IDENTICAL
    - moderate: any LIKELY and no IDENTICAL
    - weak: only WEAK or no mappings
    """
    if any(value == "IDENTICAL" for value in strengths):
        return "strong"
    if any(value == "LIKELY" for value in strengths):
        return "moderate"
    return "weak"


def calculate_confidence(reasoning: dict) -> float:
    """
    Calculate confidence score in [0, 1] from:
    1) Mapping strength quality
    2) Number of graph paths
    3) Presence of Hadith references

    Weighting:
    - strength quality: 70%
    - path coverage: 20% (saturates at 10 paths)
    - hadith presence: 10%
    """
    safe_reasoning = _ensure_dict(reasoning)
    strengths = _get_mapping_strengths(safe_reasoning)

    if strengths:
        strength_score = sum(_STRENGTH_WEIGHT[value] for value in strengths) / len(strengths)
    else:
        # No mappings means weak grounding.
        strength_score = 0.25

    path_count = _graph_paths_used(safe_reasoning)
    path_score = min(1.0, path_count / 10)

    hadith_score = 1.0 if _has_hadith_references(safe_reasoning) else 0.0

    confidence = (0.7 * strength_score) + (0.2 * path_score) + (0.1 * hadith_score)
    return round(max(0.0, min(1.0, confidence)), 2)


def generate_caution_notes(reasoning: dict) -> list[str]:
    """
    Generate user-safe caution notes based on evidence quality and coverage.
    Notes are short, non-alarming, and suitable for direct response appending.
    """
    safe_reasoning = _ensure_dict(reasoning)
    strengths = _get_mapping_strengths(safe_reasoning)

    notes: list[str] = []

    # Rule 1: Any WEAK mapping should trigger a caution note.
    if any(value == "WEAK" for value in strengths):
        notes.append(
            "Some biochemical mappings are weak; interpret these links as tentative rather than definitive."
        )

    # Rule 2: No mapped drug equivalent should be communicated explicitly.
    if not _has_drug_equivalent(safe_reasoning):
        notes.append(
            "No clear drug equivalent was identified in the current graph context, indicating limited biochemical mapping."
        )

    # Optional useful warning when no biochemical paths exist at all.
    if _graph_paths_used(safe_reasoning) == 0:
        notes.append(
            "Graph path coverage is limited for this query, so confidence in mechanistic conclusions is reduced."
        )

    return notes


def _append_notes_to_answer(answer: str, notes: list[str]) -> str:
    """Append caution notes to final answer text in a readable, deterministic format."""
    if not notes:
        return answer

    safe_answer = answer.strip() if isinstance(answer, str) else ""
    note_lines = "\n".join(f"- {note}" for note in notes)
    suffix = f"\n\nSafety Notes:\n{note_lines}"

    if not safe_answer:
        return f"Safety Notes:\n{note_lines}"
    return f"{safe_answer}{suffix}"


def apply_safety_checks(reasoning: dict, llm_output: dict) -> dict:
    """
    Main safety post-processor for orchestrator outputs.

    Inputs:
    - reasoning: structured graph reasoning from reasoning_builder
    - llm_output: A0/Af output envelope (dictionary)

    Returns:
    - llm_output enriched with:
      - safety flags/notes
      - evidence strength
      - graph path count
      - confidence score
    """
    safe_reasoning = _ensure_dict(reasoning)
    safe_output = copy.deepcopy(_ensure_dict(llm_output))

    meta = _ensure_dict(safe_reasoning.get("meta"))
    kg_applicable = meta.get("kg_applicable")
    if kg_applicable is False:
        # For non-graph conversational queries, do not force KG caution notes.
        # Keep the orchestrator output user-friendly and minimal.
        safe_output.setdefault("evidence_strength", "weak")
        safe_output.setdefault("graph_paths_used", 0)
        safe_output.setdefault("confidence_score", None)
        safe_output["safety"] = {
            "caution_flag": False,
            "caution_notes": [],
        }
        logger.info("Safety checks skipped graph cautions for non-KG query")
        return safe_output

    caution_notes = generate_caution_notes(safe_reasoning)
    confidence_score = calculate_confidence(safe_reasoning)
    strengths = _get_mapping_strengths(safe_reasoning)
    evidence_strength = _compute_evidence_strength_label(strengths)
    graph_paths = _graph_paths_used(safe_reasoning)

    has_weak_mapping = any(value == "WEAK" for value in strengths)

    # Keep answer text clean; expose cautions only in structured metadata.
    # Frontend can render these notes separately when needed.

    # Attach structured metadata for API consumers and UI rendering.
    safe_output["safety"] = {
        "caution_flag": has_weak_mapping,
        "caution_notes": caution_notes,
    }
    safe_output["evidence_strength"] = evidence_strength
    safe_output["graph_paths_used"] = graph_paths
    safe_output["confidence_score"] = confidence_score

    logger.info(
        "Safety checks applied caution_flag=%s notes=%s evidence_strength=%s graph_paths=%s confidence=%.2f",
        has_weak_mapping,
        len(caution_notes),
        evidence_strength,
        graph_paths,
        confidence_score,
    )

    return safe_output


__all__ = [
    "apply_safety_checks",
    "calculate_confidence",
    "generate_caution_notes",
]