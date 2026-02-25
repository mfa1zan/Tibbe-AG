from __future__ import annotations

from typing import Any

VALID_STRENGTHS = {"IDENTICAL", "LIKELY", "WEAK"}


def _safe_str(value: Any) -> str | None:
    """Normalize incoming values to clean strings or None."""
    if not isinstance(value, str):
        return None
    cleaned = value.strip()
    return cleaned or None


def _ensure_list(value: Any) -> list:
    """Return a list for iterable-like payload fields, otherwise an empty list."""
    return value if isinstance(value, list) else []


def _normalize_strength(value: Any, default: str = "WEAK") -> str:
    """Clamp mapping strength values to IDENTICAL/LIKELY/WEAK only."""
    strength = _safe_str(value)
    if strength in VALID_STRENGTHS:
        return strength
    return default


def _format_named_node(node_dict: dict[str, Any] | None) -> dict[str, str | None]:
    """Normalize node-like payloads to an id/name dictionary for LLM-safe output."""
    payload = node_dict or {}
    return {
        "id": _safe_str(payload.get("id")),
        "name": _safe_str(payload.get("name")),
    }


def _format_disease(disease_dict: dict[str, Any] | None) -> dict[str, str | None]:
    """Normalize disease payload and keep category metadata if available."""
    payload = disease_dict or {}
    return {
        "id": _safe_str(payload.get("id")),
        "name": _safe_str(payload.get("name")),
        "category": _safe_str(payload.get("category")),
    }


def _format_drug_compound(compound_dict: dict[str, Any] | None) -> dict[str, str | None]:
    """Normalize drug-chemical compound and expose a sanitized mapping strength."""
    payload = compound_dict or {}
    return {
        "id": _safe_str(payload.get("id")),
        "name": _safe_str(payload.get("name")),
        "mapping_strength": _normalize_strength(payload.get("relation_type")),
    }


def _format_hadith(hadith_dict: dict[str, Any] | None) -> dict[str, str | None]:
    """Normalize hadith references for direct LLM context use."""
    payload = hadith_dict or {}
    return {
        "name": _safe_str(payload.get("name")),
        "book": _safe_str(payload.get("book")),
        "reference": _safe_str(payload.get("reference")),
    }


def _index_by_id(nodes: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Create fast lookup map by id while skipping malformed rows."""
    indexed: dict[str, dict[str, Any]] = {}
    for node in nodes:
        if not isinstance(node, dict):
            continue
        node_id = _safe_str(node.get("id"))
        if node_id:
            indexed[node_id] = node
    return indexed


def _extract_hadith_references(subgraph: dict[str, Any]) -> list[dict[str, str | None]]:
    """
    Support multiple possible keys so callers can pass either raw subgraph
    or a pre-merged object containing hadith data.
    """
    raw_hadith = (
        subgraph.get("HadithReferences")
        or subgraph.get("Hadith")
        or subgraph.get("hadith_references")
        or []
    )

    hadith_items: list[dict[str, str | None]] = []
    for item in _ensure_list(raw_hadith):
        if not isinstance(item, dict):
            continue
        hadith_items.append(_format_hadith(item))
    return hadith_items


def _build_biochemical_mappings(subgraph: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Build path-level biochemical mappings:
    Ingredient -> ChemicalCompound -> DrugChemicalCompound -> Drug.

    Mapping strength is attached to each returned mapping and derived from
    the ChemicalCompound -> DrugChemicalCompound similarity edge.
    """
    relations = _ensure_list(subgraph.get("Relations"))

    ingredients = _index_by_id(_ensure_list(subgraph.get("Ingredients")))
    compounds = _index_by_id(_ensure_list(subgraph.get("ChemicalCompounds")))
    drug_compounds = _index_by_id(_ensure_list(subgraph.get("DrugChemicalCompounds")))
    drugs = _index_by_id(_ensure_list(subgraph.get("Drugs")))

    ingredient_to_compounds: dict[str, set[str]] = {}
    compound_to_drug_compounds: dict[str, list[tuple[str, str]]] = {}
    drug_compound_to_drugs: dict[str, set[str]] = {}

    # Parse relation payload into typed adjacency maps.
    for relation in relations:
        if not isinstance(relation, dict):
            continue

        source = _safe_str(relation.get("from"))
        target = _safe_str(relation.get("to"))
        rel_type = _safe_str(relation.get("type"))
        if not source or not target or not rel_type:
            continue

        if rel_type == "CONTAINS" and source in ingredients and target in compounds:
            ingredient_to_compounds.setdefault(source, set()).add(target)
        elif rel_type in VALID_STRENGTHS and source in compounds and target in drug_compounds:
            compound_to_drug_compounds.setdefault(source, []).append((target, rel_type))
        elif rel_type == "CONTAINS" and source in drugs and target in drug_compounds:
            drug_compound_to_drugs.setdefault(target, set()).add(source)

    mappings: list[dict[str, Any]] = []

    # Combine all reachable hops into LLM-ready reasoning path objects.
    for ingredient_id, compound_ids in ingredient_to_compounds.items():
        for compound_id in compound_ids:
            edges = compound_to_drug_compounds.get(compound_id, [])
            if not edges:
                continue

            for dcc_id, edge_strength in edges:
                linked_drugs = drug_compound_to_drugs.get(dcc_id) or {None}
                for drug_id in linked_drugs:
                    ingredient_obj = _format_named_node(ingredients.get(ingredient_id))
                    compound_obj = _format_named_node(compounds.get(compound_id))
                    drug_compound_obj = _format_drug_compound(drug_compounds.get(dcc_id))

                    # Preserve relation-derived strength as the authoritative mapping signal.
                    mapping_strength = _normalize_strength(edge_strength)
                    drug_compound_obj["mapping_strength"] = mapping_strength

                    drug_obj = _format_named_node(drugs.get(drug_id)) if drug_id else {"id": None, "name": None}

                    mappings.append(
                        {
                            "ingredient": ingredient_obj,
                            "chemical_compound": compound_obj,
                            "drug_chemical_compound": drug_compound_obj,
                            "drug": drug_obj,
                            "mapping_strength": mapping_strength,
                        }
                    )

    # Graceful fallback: if relations are missing, still expose standalone similarity hints.
    if not mappings:
        for dcc in _ensure_list(subgraph.get("DrugChemicalCompounds")):
            if not isinstance(dcc, dict):
                continue
            formatted_dcc = _format_drug_compound(dcc)
            mappings.append(
                {
                    "ingredient": {"id": None, "name": None},
                    "chemical_compound": {"id": None, "name": None},
                    "drug_chemical_compound": formatted_dcc,
                    "drug": {"id": None, "name": None},
                    "mapping_strength": formatted_dcc["mapping_strength"],
                }
            )

    # Deduplicate mappings to keep context compact for downstream LLM calls.
    seen: set[tuple[Any, ...]] = set()
    deduped: list[dict[str, Any]] = []
    for item in mappings:
        fingerprint = (
            item["ingredient"]["id"],
            item["chemical_compound"]["id"],
            item["drug_chemical_compound"]["id"],
            item["drug"]["id"],
            item["mapping_strength"],
        )
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        deduped.append(item)

    return deduped


def build_graph_reasoning(subgraph: dict) -> dict:
    """
    Convert raw `graph_service.get_disease_subgraph()` output into a clean,
    orchestrator-ready reasoning dictionary for LLM input.

    This function intentionally exposes only structured dictionaries/lists and
    never returns raw Neo4j records or Cypher fragments.
    """
    safe_subgraph = subgraph if isinstance(subgraph, dict) else {}

    disease = _format_disease(safe_subgraph.get("Disease"))
    ingredients = [_format_named_node(item) for item in _ensure_list(safe_subgraph.get("Ingredients")) if isinstance(item, dict)]
    compounds = [
        _format_named_node(item)
        for item in _ensure_list(safe_subgraph.get("ChemicalCompounds"))
        if isinstance(item, dict)
    ]
    drug_compounds = [
        _format_drug_compound(item)
        for item in _ensure_list(safe_subgraph.get("DrugChemicalCompounds"))
        if isinstance(item, dict)
    ]
    drugs = [_format_named_node(item) for item in _ensure_list(safe_subgraph.get("Drugs")) if isinstance(item, dict)]
    hadith_refs = _extract_hadith_references(safe_subgraph)

    reasoning = {
        "Disease": disease,
        "Ingredients": ingredients,
        "ChemicalCompounds": compounds,
        "DrugChemicalCompounds": drug_compounds,
        "Drugs": drugs,
        "HadithReferences": hadith_refs,
        # Path-level biochemical links used by the orchestrator for grounded generation.
        "BiochemicalMappings": _build_biochemical_mappings(safe_subgraph),
        # Optional metadata for robust downstream handling in A0 prompts.
        "meta": {
            "has_error": bool(safe_subgraph.get("error")),
            "source_error": _safe_str(safe_subgraph.get("error")),
        },
    }

    return reasoning


__all__ = [
    "build_graph_reasoning",
]