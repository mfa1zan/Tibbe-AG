"""
Dosage & Clinical Metadata Validator for PRO-MedGraph.

Extends the pipeline with dosage-aware validation:
- Retrieves traditional_dosage / preparation_method for Ingredients
- Retrieves standard_dosage / contraindications / side_effects for Drugs
- Computes a dosage_alignment_score that penalises missing or conflicting data
- Produces a structured dosage comparison block for LLM context

Pipeline position:
    ... → Causal Reasoner → **Dosage Validator** → A0 Generation
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from typing import Any

from neo4j import GraphDatabase

from app.config import get_settings

logger = logging.getLogger(__name__)


# ── Data classes ───────────────────────────────────────────────────────────────


@dataclass
class IngredientDosage:
    name: str = ""
    traditional_dosage: str | None = None
    preparation_method: str | None = None
    dosage_available: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DrugDosage:
    name: str = ""
    standard_dosage: str | None = None
    contraindications: str | None = None
    side_effects: str | None = None
    dosage_available: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DosageComparison:
    ingredient: IngredientDosage = field(default_factory=IngredientDosage)
    drug: DrugDosage = field(default_factory=DrugDosage)
    alignment_score: float = 0.0
    alignment_notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ingredient": self.ingredient.to_dict(),
            "drug": self.drug.to_dict(),
            "alignment_score": self.alignment_score,
            "alignment_notes": self.alignment_notes,
        }


@dataclass
class DosageValidationResult:
    comparisons: list[DosageComparison] = field(default_factory=list)
    overall_alignment_score: float = 0.0
    has_dosage_data: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "comparisons": [c.to_dict() for c in self.comparisons],
            "overall_alignment_score": self.overall_alignment_score,
            "has_dosage_data": self.has_dosage_data,
        }


# ── Neo4j dosage queries ───────────────────────────────────────────────────────


def _get_driver():
    settings = get_settings()
    return GraphDatabase.driver(
        settings.neo4j_uri,
        auth=(settings.neo4j_username, settings.neo4j_password),
    )


def fetch_ingredient_dosage(ingredient_name: str) -> IngredientDosage:
    """
    Fetch dosage metadata for an ingredient from Neo4j.
    Properties: traditional_dosage, preparation_method (may not exist yet in KG).
    """
    driver = _get_driver()
    query = """
    MATCH (i:Ingredient)
    WHERE toLower(i.name) = toLower($name)
    RETURN i.name AS name,
           i.traditional_dosage AS traditional_dosage,
           i.preparation_method AS preparation_method
    LIMIT 1
    """
    try:
        with driver.session() as session:
            row = session.run(query, name=ingredient_name).single()

        if not row:
            return IngredientDosage(name=ingredient_name)

        trad = row.get("traditional_dosage")
        prep = row.get("preparation_method")
        has_data = bool(trad or prep)

        return IngredientDosage(
            name=row.get("name") or ingredient_name,
            traditional_dosage=trad if isinstance(trad, str) and trad.strip() else None,
            preparation_method=prep if isinstance(prep, str) and prep.strip() else None,
            dosage_available=has_data,
        )
    except Exception:
        logger.exception("Failed to fetch ingredient dosage for '%s'", ingredient_name)
        return IngredientDosage(name=ingredient_name)


def fetch_drug_dosage(drug_name: str) -> DrugDosage:
    """
    Fetch clinical metadata for a drug from Neo4j.
    Properties: standard_dosage, contraindications, side_effects (may not exist yet in KG).
    """
    driver = _get_driver()
    query = """
    MATCH (d:Drug)
    WHERE toLower(d.name) = toLower($name)
    RETURN d.name AS name,
           d.standard_dosage AS standard_dosage,
           d.contraindications AS contraindications,
           d.side_effects AS side_effects
    LIMIT 1
    """
    try:
        with driver.session() as session:
            row = session.run(query, name=drug_name).single()

        if not row:
            return DrugDosage(name=drug_name)

        dosage = row.get("standard_dosage")
        contra = row.get("contraindications")
        effects = row.get("side_effects")
        has_data = bool(dosage or contra or effects)

        return DrugDosage(
            name=row.get("name") or drug_name,
            standard_dosage=dosage if isinstance(dosage, str) and dosage.strip() else None,
            contraindications=contra if isinstance(contra, str) and contra.strip() else None,
            side_effects=effects if isinstance(effects, str) and effects.strip() else None,
            dosage_available=has_data,
        )
    except Exception:
        logger.exception("Failed to fetch drug dosage for '%s'", drug_name)
        return DrugDosage(name=drug_name)


# ── Scoring ────────────────────────────────────────────────────────────────────


def _compute_alignment_score(
    ingredient: IngredientDosage,
    drug: DrugDosage,
) -> tuple[float, list[str]]:
    """
    Compute dosage alignment score in [0, 1] with explanatory notes.

    Scoring rules:
    - Base score: 0.5 (neutral)
    - +0.15 if ingredient dosage available
    - +0.15 if drug dosage available
    - +0.10 if preparation method available
    - +0.10 if no contraindications
    - -0.20 if ingredient dosage unclear (missing)
    - -0.15 if drug has contraindications
    """
    score = 0.5
    notes: list[str] = []

    if ingredient.dosage_available:
        if ingredient.traditional_dosage:
            score += 0.15
            notes.append(f"Traditional dosage available: {ingredient.traditional_dosage}")
        if ingredient.preparation_method:
            score += 0.10
            notes.append(f"Preparation method: {ingredient.preparation_method}")
    else:
        score -= 0.20
        notes.append("Traditional dosage information not available in KG — interpret with caution")

    if drug.dosage_available:
        if drug.standard_dosage:
            score += 0.15
            notes.append(f"Standard drug dosage: {drug.standard_dosage}")
        if drug.contraindications:
            score -= 0.15
            notes.append(f"Drug contraindications noted: {drug.contraindications}")
        else:
            score += 0.10
            notes.append("No known contraindications recorded")
        if drug.side_effects:
            notes.append(f"Known side effects: {drug.side_effects}")
    else:
        notes.append("Drug dosage metadata not available in KG")

    return round(max(0.0, min(1.0, score)), 2), notes


# ── Main entry point ───────────────────────────────────────────────────────────


def validate_dosage(
    reasoning: dict,
    causal_paths: list[dict] | None = None,
) -> DosageValidationResult:
    """
    Main entry called by orchestrator after causal reasoning.

    Extracts unique ingredient/drug pairs from causal paths or reasoning
    and produces dosage comparisons with alignment scores.
    """
    comparisons: list[DosageComparison] = []
    seen_pairs: set[tuple[str, str]] = set()

    # Prefer causal paths if available
    paths_source = causal_paths or []

    for path in paths_source:
        if not isinstance(path, dict):
            continue
        ing_name = path.get("ingredient")
        drug_name = path.get("drug")
        if not ing_name or not drug_name:
            continue

        pair_key = (ing_name.lower(), drug_name.lower())
        if pair_key in seen_pairs:
            continue
        seen_pairs.add(pair_key)

        ing_dosage = fetch_ingredient_dosage(ing_name)
        drug_dosage = fetch_drug_dosage(drug_name)
        alignment, notes = _compute_alignment_score(ing_dosage, drug_dosage)

        comparisons.append(DosageComparison(
            ingredient=ing_dosage,
            drug=drug_dosage,
            alignment_score=alignment,
            alignment_notes=notes,
        ))

    # Fallback: if no causal paths, try extracting from reasoning directly
    if not comparisons:
        ingredients = reasoning.get("Ingredients", []) or []
        drugs = reasoning.get("Drugs", []) or []

        ingredient_names = [
            i.get("name") for i in ingredients
            if isinstance(i, dict) and isinstance(i.get("name"), str)
        ][:5]  # Limit to top 5

        drug_names = [
            d.get("name") for d in drugs
            if isinstance(d, dict) and isinstance(d.get("name"), str)
        ][:5]

        for ing_name in ingredient_names:
            ing_dosage = fetch_ingredient_dosage(ing_name)
            for drug_name in drug_names:
                pair_key = (ing_name.lower(), drug_name.lower())
                if pair_key in seen_pairs:
                    continue
                seen_pairs.add(pair_key)

                drug_dosage = fetch_drug_dosage(drug_name)
                alignment, notes = _compute_alignment_score(ing_dosage, drug_dosage)
                comparisons.append(DosageComparison(
                    ingredient=ing_dosage,
                    drug=drug_dosage,
                    alignment_score=alignment,
                    alignment_notes=notes,
                ))

    has_data = any(c.ingredient.dosage_available or c.drug.dosage_available for c in comparisons)
    overall = (
        round(sum(c.alignment_score for c in comparisons) / len(comparisons), 2)
        if comparisons else 0.0
    )

    logger.info(
        "Dosage validation: %d comparisons, overall_alignment=%.2f, has_data=%s",
        len(comparisons), overall, has_data,
    )

    return DosageValidationResult(
        comparisons=comparisons,
        overall_alignment_score=overall,
        has_dosage_data=has_data,
    )


__all__ = [
    "validate_dosage",
    "DosageValidationResult",
    "IngredientDosage",
    "DrugDosage",
    "DosageComparison",
]
