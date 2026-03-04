"""
Causal Reasoning Layer for PRO-MedGraph.

Transforms flat biochemical mappings from the reasoning builder into explicit
causal path objects with quantitative scoring.  Each CausalPath captures one
traversal from Disease ← Ingredient → ChemicalCompound → DrugChemicalCompound → Drug
and attaches three sub-scores:

    mapping_strength  – derived from IS_IDENTICAL_TO / IS_LIKELY_EQUIVALENT_TO / IS_WEAK_MATCH_TO
    mechanism_score   – compound-description overlap between natural & pharmaceutical sides
    path_depth_score  – rewards multi-hop reasoning (≥ 3 hops)

The composite causal_score is:

    causal_score = 0.5 * mapping_strength + 0.3 * mechanism_score + 0.2 * path_depth_score

Paths are ranked by causal_score descending and returned alongside summary
statistics consumed by A0 generation, the evaluation framework, and the
experiment logger.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from difflib import SequenceMatcher
from typing import Any

logger = logging.getLogger(__name__)

# ── Strength constants ─────────────────────────────────────────────────────────
STRENGTH_MAP: dict[str, float] = {
    "IDENTICAL": 1.0,
    "LIKELY": 0.7,
    "WEAK": 0.35,
}
NEUTRAL_STRENGTH_SCORE = 0.55

# Path-depth bucketing (hop count → score)
_DEPTH_SCORE: dict[int, float] = {
    1: 0.3,
    2: 0.5,
    3: 0.8,
    4: 1.0,
}


# ── Data classes ───────────────────────────────────────────────────────────────


@dataclass
class CausalPath:
    """One biochemical causal chain from disease to drug (or reverse)."""

    disease: str | None = None
    ingredient: str | None = None
    compound: str | None = None
    drug_compound: str | None = None
    drug: str | None = None
    mapping_strength_label: str = "WEAK"
    mapping_strength: float = 0.35
    mechanism_score: float = 0.0
    path_depth_score: float = 0.0
    causal_score: float = 0.0
    hop_count: int = 0
    is_hypothesis: bool = False
    causal_chain: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CausalRankingSummary:
    """Aggregated statistics over all scored causal paths for one query."""

    total_paths: int = 0
    avg_causal_score: float = 0.0
    max_causal_score: float = 0.0
    min_causal_score: float = 0.0
    strong_paths: int = 0
    moderate_paths: int = 0
    weak_paths: int = 0
    avg_mechanism_score: float = 0.0
    avg_path_depth: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ── Helper functions ───────────────────────────────────────────────────────────


def _safe_str(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    cleaned = value.strip()
    return cleaned or None


def _compute_mechanism_overlap(
    compound_name: str | None,
    drug_compound_name: str | None,
) -> float:
    """
    Approximate mechanism overlap via name similarity of compound vs drug-compound.
    In a production system this would use embedding cosine-similarity or
    a lookup table of pharmacological mechanism annotations.  For the thesis
    prototype we use SequenceMatcher which captures lexical overlap in
    chemical nomenclature (e.g. "Thymoquinone" vs "Thymoquinone").
    """
    if not compound_name or not drug_compound_name:
        return 0.0
    return round(
        SequenceMatcher(None, compound_name.lower(), drug_compound_name.lower()).ratio(),
        4,
    )


def _compute_path_depth_score(hop_count: int) -> float:
    """
    Reward deeper multi-hop paths.  Saturates at 4 hops.
    """
    if hop_count <= 0:
        return 0.0
    return _DEPTH_SCORE.get(min(hop_count, 4), 1.0)


def _normalize_similarity_label(value: Any) -> str:
    text = _safe_str(value)
    if not text:
        return "UNKNOWN"
    upper = text.upper()
    if "IDENTICAL" in upper:
        return "IDENTICAL"
    if "LIKELY" in upper or "EQUIVALENT" in upper:
        return "LIKELY"
    if "WEAK" in upper or "MATCH" in upper:
        return "WEAK"
    return "UNKNOWN"


def _compute_numeric_evidence_score(value: Any) -> float:
    if isinstance(value, (int, float)):
        numeric = float(value)
        if numeric <= 0:
            return 0.5
        if numeric <= 1:
            return numeric
        return round(numeric / (numeric + 1.0), 4)

    if isinstance(value, dict):
        scores: list[float] = []
        for item in value.values():
            if isinstance(item, (int, float)):
                scores.append(_compute_numeric_evidence_score(item))
        if scores:
            return round(sum(scores) / len(scores), 4)

    return 0.5


def _is_full_ontology_path(path: CausalPath) -> bool:
    return bool(path.ingredient and path.compound and path.drug_compound and path.drug)


def _count_hops(path: CausalPath) -> int:
    """Count non-null nodes in the traversal chain."""
    count = 0
    for node in (path.disease, path.ingredient, path.compound, path.drug_compound, path.drug):
        if node:
            count += 1
    return max(0, count - 1)  # hops = edges = nodes - 1


# ── Core scoring ───────────────────────────────────────────────────────────────


def score_causal_path(path: CausalPath, numeric_evidence: Any = None) -> CausalPath:
    """
    Fill in all sub-scores and the composite causal_score for a single path.
    Mutates the path in-place and returns it for chaining.
    """
    similarity_label = _normalize_similarity_label(path.mapping_strength_label)
    mapping_strength_score = STRENGTH_MAP.get(similarity_label, NEUTRAL_STRENGTH_SCORE)

    similarity_presence_score = 1.0 if similarity_label in STRENGTH_MAP else 0.5
    numeric_evidence_score = _compute_numeric_evidence_score(numeric_evidence)

    # Reasoning layer is schema-aligned and ontology-aware, but no longer brittle.
    path.mapping_strength_label = similarity_label
    path.mapping_strength = mapping_strength_score
    path.hop_count = _count_hops(path)
    path.path_depth_score = _compute_path_depth_score(path.hop_count)
    path.mechanism_score = _compute_mechanism_overlap(path.compound, path.drug_compound)

    if _is_full_ontology_path(path):
        ontology_path_factor = 1.0
    else:
        ontology_path_factor = 0.6

    path.causal_score = round(
        0.35 * path.mapping_strength
        + 0.20 * path.mechanism_score
        + 0.15 * path.path_depth_score
        + 0.20 * similarity_presence_score
        + 0.10 * numeric_evidence_score,
        4,
    )
    path.causal_score = round(path.causal_score * ontology_path_factor, 4)

    # Build human-readable traversal chain
    chain: list[str] = []
    if path.disease:
        chain.append(f"Disease({path.disease})")
    if path.ingredient:
        chain.append(f"Ingredient({path.ingredient})")
    if path.compound:
        chain.append(f"Compound({path.compound})")
    if path.drug_compound:
        chain.append(f"DrugCompound({path.drug_compound})")
    if path.drug:
        chain.append(f"Drug({path.drug})")
    path.causal_chain = chain

    return path


def build_causal_paths_from_reasoning(reasoning: dict) -> list[CausalPath]:
    """
    Extract causal paths from structured reasoning produced by reasoning_builder.

    Each BiochemicalMapping becomes one CausalPath.  The Disease is attached
    from reasoning["Disease"] (disease-centric) or from reasoning["Diseases"]
    (ingredient/drug-centric).
    """
    paths: list[CausalPath] = []
    mappings = reasoning.get("BiochemicalMappings", []) if isinstance(reasoning, dict) else []
    if not isinstance(mappings, list):
        return paths

    # Determine disease name(s) from reasoning context
    disease_obj = reasoning.get("Disease") if isinstance(reasoning, dict) else {}
    if not isinstance(disease_obj, dict):
        disease_obj = {}
    primary_disease = _safe_str(disease_obj.get("name")) if isinstance(disease_obj, dict) else None

    # For ingredient/drug-centric queries, diseases may be a list
    diseases_list = reasoning.get("Diseases", []) if isinstance(reasoning, dict) else []
    fallback_disease = None
    if not primary_disease and isinstance(diseases_list, list) and diseases_list:
        first = diseases_list[0] if diseases_list else {}
        fallback_disease = _safe_str(first.get("name")) if isinstance(first, dict) else None

    disease_name = primary_disease or fallback_disease

    for mapping in mappings:
        if not isinstance(mapping, dict):
            continue

        ingredient_obj = mapping.get("ingredient", {}) if isinstance(mapping.get("ingredient"), dict) else {}
        compound_obj = mapping.get("chemical_compound", {}) if isinstance(mapping.get("chemical_compound"), dict) else {}
        drug_compound_obj = (
            mapping.get("drug_chemical_compound", {})
            if isinstance(mapping.get("drug_chemical_compound"), dict)
            else {}
        )
        drug_obj = mapping.get("drug", {}) if isinstance(mapping.get("drug"), dict) else {}

        strength_label = _normalize_similarity_label(
            mapping.get("similarity_relation_type")
            or mapping.get("mapping_strength")
            or drug_compound_obj.get("mapping_strength")
        )

        path = CausalPath(
            disease=disease_name,
            ingredient=_safe_str(ingredient_obj.get("name")),
            compound=_safe_str(compound_obj.get("name")),
            drug_compound=_safe_str(drug_compound_obj.get("name")),
            drug=_safe_str(drug_obj.get("name")),
            mapping_strength_label=strength_label,
        )
        score_causal_path(path, numeric_evidence=mapping.get("numeric_evidence"))
        paths.append(path)

    return paths


def rank_causal_paths(paths: list[CausalPath]) -> list[CausalPath]:
    """Sort paths by causal_score descending, then by mapping_strength descending."""
    return sorted(paths, key=lambda p: (p.causal_score, p.mapping_strength), reverse=True)


def summarize_causal_ranking(paths: list[CausalPath]) -> CausalRankingSummary:
    """Aggregate statistics across all causal paths."""
    if not paths:
        return CausalRankingSummary()

    scores = [p.causal_score for p in paths]
    mechanism_scores = [p.mechanism_score for p in paths]
    depths = [float(p.hop_count) for p in paths]

    strong_count = 0
    moderate_count = 0
    weak_count = 0
    for path in paths:
        if path.mapping_strength_label == "IDENTICAL":
            strong_count += 1
        elif path.mapping_strength_label == "LIKELY":
            moderate_count += 1
        elif path.mapping_strength_label == "WEAK":
            weak_count += 1
        elif path.mapping_strength >= 0.85:
            strong_count += 1
        elif path.mapping_strength >= 0.55:
            moderate_count += 1
        else:
            weak_count += 1

    return CausalRankingSummary(
        total_paths=len(paths),
        avg_causal_score=round(sum(scores) / len(scores), 4),
        max_causal_score=round(max(scores), 4),
        min_causal_score=round(min(scores), 4),
        strong_paths=strong_count,
        moderate_paths=moderate_count,
        weak_paths=weak_count,
        avg_mechanism_score=round(sum(mechanism_scores) / len(mechanism_scores), 4),
        avg_path_depth=round(sum(depths) / len(depths), 2),
    )


def run_causal_analysis(reasoning: dict) -> dict[str, Any]:
    """
    Top-level entry point called by the orchestrator after reasoning_builder.

    Returns:
        {
            "causal_paths": [CausalPath.to_dict(), ...],
            "causal_ranking": CausalRankingSummary.to_dict(),
        }
    """
    try:
        paths = build_causal_paths_from_reasoning(reasoning)
        ranked = rank_causal_paths(paths)
        summary = summarize_causal_ranking(ranked)

        logger.info(
            "Causal analysis complete: total_paths=%d avg_score=%.3f max_score=%.3f",
            summary.total_paths,
            summary.avg_causal_score,
            summary.max_causal_score,
        )

        return {
            "causal_paths": [p.to_dict() for p in ranked],
            "causal_ranking": summary.to_dict(),
        }
    except Exception:
        logger.exception("Causal analysis failed")
        return {
            "causal_paths": [],
            "causal_ranking": CausalRankingSummary().to_dict(),
        }


__all__ = [
    "CausalPath",
    "CausalRankingSummary",
    "run_causal_analysis",
    "build_causal_paths_from_reasoning",
    "rank_causal_paths",
    "summarize_causal_ranking",
    "score_causal_path",
]
