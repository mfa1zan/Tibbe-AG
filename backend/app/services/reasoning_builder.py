from __future__ import annotations

import json
from pathlib import Path
from typing import Any

VALID_STRENGTHS = {"IDENTICAL", "LIKELY", "WEAK"}


def _load_kg_schema() -> dict[str, Any]:
    try:
        project_root = Path(__file__).resolve().parents[3]
        schema_path = project_root / "knowledge_graph_schema.json"
        with open(schema_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        if isinstance(raw, list) and raw and isinstance(raw[0], dict):
            schema = raw[0].get("FULL_KG_SCHEMA_JSON")
            if isinstance(schema, dict):
                return schema
        if isinstance(raw, dict):
            return raw
    except Exception:
        return {}
    return {}


_KG_SCHEMA: dict[str, Any] = _load_kg_schema()
_SCHEMA_LABELS: set[str] = {
    str(label).strip() for label in (_KG_SCHEMA.get("nodeLabels") or []) if isinstance(label, str) and str(label).strip()
}


def _relationship_types_between(from_label: str, to_label: str) -> set[str]:
    rels: set[str] = set()
    for edge in (_KG_SCHEMA.get("relationshipStructure") or []):
        if not isinstance(edge, dict):
            continue
        rel = edge.get("relationship")
        from_labels = edge.get("from") or []
        to_labels = edge.get("to") or []
        if isinstance(rel, str) and from_label in from_labels and to_label in to_labels:
            rels.add(rel)
    return rels


def _schema_has_relationship(rel_type: str) -> bool:
    rels = _KG_SCHEMA.get("relationshipTypes") or []
    if not rels:
        return True
    return rel_type in rels


def _label_from_key(key: str) -> str | None:
    if not isinstance(key, str):
        return None
    norm = key.strip().lower()
    alias = {
        "disease": "Disease",
        "diseases": "Disease",
        "ingredient": "Ingredient",
        "ingredients": "Ingredient",
        "chemicalcompound": "ChemicalCompound",
        "chemicalcompounds": "ChemicalCompound",
        "drugchemicalcompound": "DrugChemicalCompound",
        "drugchemicalcompounds": "DrugChemicalCompound",
        "drug": "Drug",
        "drugs": "Drug",
        "hadith": "Hadith",
        "hadithreferences": "Hadith",
        "reference": "Reference",
        "references": "Reference",
        "book": "Book",
        "books": "Book",
        "diseasecategory": "DiseaseCategory",
        "diseasecategories": "DiseaseCategory",
        "cure": "Cure",
        "cures": "Cure",
    }
    label = alias.get(norm)
    if label and (_SCHEMA_LABELS and label not in _SCHEMA_LABELS):
        return None
    return label


def _relation_to_strength_label(rel_type: str | None) -> str:
    rel_upper = (rel_type or "").upper()
    if "IDENTICAL" in rel_upper:
        return "IDENTICAL"
    if "LIKELY" in rel_upper or "EQUIVALENT" in rel_upper:
        return "LIKELY"
    if "WEAK" in rel_upper or "MATCH" in rel_upper:
        return "WEAK"
    return "WEAK"


def _safe_str(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    cleaned = value.strip()
    return cleaned or None


def _safe_id(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _ensure_list(value: Any) -> list:
    return value if isinstance(value, list) else []


def _normalize_strength(value: Any, default: str = "WEAK") -> str:
    strength = _safe_str(value)
    if strength in VALID_STRENGTHS:
        return strength
    return default


def _format_named_node(node_dict: dict[str, Any] | None) -> dict[str, str | None]:
    payload = node_dict if isinstance(node_dict, dict) else {}
    return {
        "id": _safe_id(payload.get("id")),
        "name": _safe_str(payload.get("name")),
    }


def _format_ingredient_node(node_dict: dict[str, Any] | None) -> dict[str, str | None]:
    payload = node_dict if isinstance(node_dict, dict) else {}
    result = {
        "id": _safe_id(payload.get("id")),
        "name": _safe_str(payload.get("name")),
    }
    trad = _safe_str(payload.get("traditional_dosage"))
    prep = _safe_str(payload.get("preparation_method"))
    if trad:
        result["traditional_dosage"] = trad
    if prep:
        result["preparation_method"] = prep
    return result


def _format_drug_node(node_dict: dict[str, Any] | None) -> dict[str, str | None]:
    payload = node_dict if isinstance(node_dict, dict) else {}
    result = {
        "id": _safe_id(payload.get("id")),
        "name": _safe_str(payload.get("name")),
    }
    dosage = _safe_str(payload.get("standard_dosage"))
    contra = _safe_str(payload.get("contraindications"))
    effects = _safe_str(payload.get("side_effects"))
    if dosage:
        result["standard_dosage"] = dosage
    if contra:
        result["contraindications"] = contra
    if effects:
        result["side_effects"] = effects
    return result


def _format_disease(disease_dict: dict[str, Any] | None) -> dict[str, str | None]:
    payload = disease_dict if isinstance(disease_dict, dict) else {}
    return {
        "id": _safe_id(payload.get("id")),
        "name": _safe_str(payload.get("name")),
        "category": _safe_str(payload.get("category")),
    }


def _format_drug_compound(compound_dict: dict[str, Any] | None) -> dict[str, str | None]:
    payload = compound_dict if isinstance(compound_dict, dict) else {}
    relation_type = _safe_str(payload.get("relation_type"))
    inferred_strength = _relation_to_strength_label(relation_type)
    return {
        "id": _safe_id(payload.get("id")),
        "name": _safe_str(payload.get("name")),
        "mapping_strength": _normalize_strength(payload.get("mapping_strength"), default=inferred_strength),
    }


def _format_hadith(hadith_dict: dict[str, Any] | None) -> dict[str, str | None]:
    payload = hadith_dict if isinstance(hadith_dict, dict) else {}
    return {
        "id": _safe_id(payload.get("id")),
        "name": _safe_str(payload.get("name")),
        "book": _safe_str(payload.get("book")),
        "reference": _safe_str(payload.get("reference")),
    }


def _index_by_id(nodes: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for node in nodes:
        if not isinstance(node, dict):
            continue
        node_id = _safe_id(node.get("id"))
        if node_id:
            indexed[node_id] = node
    return indexed


def _collect_nodes_by_label(subgraph: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}

    for key, value in subgraph.items():
        label = _label_from_key(key)
        if not label:
            continue

        if isinstance(value, dict):
            grouped.setdefault(label, []).append(value)
            continue

        if isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    grouped.setdefault(label, []).append(item)

    raw_subgraph = subgraph.get("raw_subgraph")
    if isinstance(raw_subgraph, dict):
        raw_nodes = raw_subgraph.get("nodes")
        for raw_node in _ensure_list(raw_nodes):
            if not isinstance(raw_node, dict):
                continue
            labels = raw_node.get("labels")
            properties = raw_node.get("properties")
            if not isinstance(labels, list) or not isinstance(properties, dict):
                continue
            node_id = _safe_id(raw_node.get("id"))
            for label in labels:
                if not isinstance(label, str):
                    continue
                if _SCHEMA_LABELS and label not in _SCHEMA_LABELS:
                    continue
                grouped.setdefault(label, []).append({
                    "id": node_id,
                    **properties,
                })

    return grouped


def _extract_hadith_references(
    subgraph: dict[str, Any],
    nodes_by_label: dict[str, list[dict[str, Any]]],
) -> list[dict[str, str | None]]:
    hadith_nodes = [
        _format_hadith(item)
        for item in _ensure_list(nodes_by_label.get("Hadith"))
        if isinstance(item, dict)
    ]
    reference_nodes = [
        _format_named_node(item)
        for item in _ensure_list(nodes_by_label.get("Reference"))
        if isinstance(item, dict)
    ]

    if not hadith_nodes:
        raw_hadith = subgraph.get("HadithReferences") or subgraph.get("hadith_references") or []
        hadith_nodes = [_format_hadith(item) for item in _ensure_list(raw_hadith) if isinstance(item, dict)]

    if not hadith_nodes:
        return []

    relations = _ensure_list(subgraph.get("Relations"))
    hadith_by_id = {
        item.get("id"): item for item in hadith_nodes if isinstance(item, dict) and item.get("id")
    }
    reference_by_id = {
        item.get("id"): item for item in reference_nodes if isinstance(item, dict) and item.get("id")
    }

    has_hadith_rel_types = _relationship_types_between("Reference", "Hadith")
    linked_refs: dict[str, str] = {}
    if has_hadith_rel_types:
        for rel in relations:
            if not isinstance(rel, dict):
                continue
            rel_type = _safe_str(rel.get("type"))
            source_id = _safe_id(rel.get("from"))
            target_id = _safe_id(rel.get("to"))
            if not rel_type or not source_id or not target_id:
                continue
            if rel_type not in has_hadith_rel_types:
                continue
            ref = reference_by_id.get(source_id)
            hadith = hadith_by_id.get(target_id)
            if not ref or not hadith:
                continue
            linked_refs[target_id] = _safe_str(ref.get("name")) or "Unknown"

    results: list[dict[str, str | None]] = []
    for hadith in hadith_nodes:
        hadith_id = hadith.get("id") if isinstance(hadith, dict) else None
        explicit_reference = _safe_str(hadith.get("reference")) if isinstance(hadith, dict) else None
        linked_reference = linked_refs.get(hadith_id) if hadith_id else None
        reference_value = explicit_reference or linked_reference or "Unknown"

        results.append(
            {
                "name": _safe_str(hadith.get("name")) if isinstance(hadith, dict) else None,
                "book": _safe_str(hadith.get("book")) if isinstance(hadith, dict) else None,
                "reference": reference_value,
            }
        )

    deduped: list[dict[str, str | None]] = []
    seen: set[tuple[str | None, str | None]] = set()
    for item in results:
        fingerprint = (item.get("name"), item.get("reference"))
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        deduped.append(item)

    return deduped


def _build_biochemical_mappings(
    subgraph: dict[str, Any],
    nodes_by_label: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    relations = _ensure_list(subgraph.get("Relations"))

    ingredients = _index_by_id(_ensure_list(nodes_by_label.get("Ingredient")))
    compounds = _index_by_id(_ensure_list(nodes_by_label.get("ChemicalCompound")))
    drug_compounds = _index_by_id(_ensure_list(nodes_by_label.get("DrugChemicalCompound")))
    drugs = _index_by_id(_ensure_list(nodes_by_label.get("Drug")))

    ingredient_to_compounds: dict[str, set[str]] = {}
    compound_to_drug_compounds: dict[str, list[tuple[str, str, dict[str, float]]]] = {}
    drug_compound_to_drugs: dict[str, set[str]] = {}

    # Reasoning layer is schema-aligned and ontology-aware, but no longer brittle.
    ingredient_compound_rel_types = _relationship_types_between("Ingredient", "ChemicalCompound")
    drug_dcc_rel_types = _relationship_types_between("Drug", "DrugChemicalCompound")
    compound_dcc_rel_types = _relationship_types_between("ChemicalCompound", "DrugChemicalCompound")

    for relation in relations:
        if not isinstance(relation, dict):
            continue

        source = _safe_id(relation.get("from"))
        target = _safe_id(relation.get("to"))
        rel_type = _safe_str(relation.get("type"))
        if not source or not target or not rel_type:
            continue
        if not _schema_has_relationship(rel_type):
            continue

        if (
            source in ingredients
            and target in compounds
            and (not ingredient_compound_rel_types or rel_type in ingredient_compound_rel_types)
        ):
            ingredient_to_compounds.setdefault(source, set()).add(target)
            continue

        if (
            source in compounds
            and target in drug_compounds
            and (not compound_dcc_rel_types or rel_type in compound_dcc_rel_types)
        ):
            numeric_evidence: dict[str, float] = {}
            for key, value in relation.items():
                if key in {"from", "to", "type"}:
                    continue
                if isinstance(value, (int, float)):
                    numeric_evidence[key] = float(value)
            compound_to_drug_compounds.setdefault(source, []).append(
                (target, rel_type, numeric_evidence)
            )
            continue

        if (
            source in drugs
            and target in drug_compounds
            and (not drug_dcc_rel_types or rel_type in drug_dcc_rel_types)
        ):
            drug_compound_to_drugs.setdefault(target, set()).add(source)

    mappings: list[dict[str, Any]] = []

    for ingredient_id, compound_ids in ingredient_to_compounds.items():
        for compound_id in compound_ids:
            similarity_edges = compound_to_drug_compounds.get(compound_id, [])
            if not similarity_edges:
                continue

            for dcc_id, similarity_rel_type, numeric_evidence in similarity_edges:
                linked_drugs = drug_compound_to_drugs.get(dcc_id) or {None}
                for drug_id in linked_drugs:
                    ingredient_obj = _format_named_node(ingredients.get(ingredient_id))
                    compound_obj = _format_named_node(compounds.get(compound_id))
                    dcc_obj = _format_drug_compound(drug_compounds.get(dcc_id))
                    drug_obj = _format_named_node(drugs.get(drug_id)) if drug_id else {"id": None, "name": None}

                    mapping_strength = _relation_to_strength_label(similarity_rel_type)
                    dcc_obj["mapping_strength"] = mapping_strength

                    mappings.append(
                        {
                            "ingredient": ingredient_obj,
                            "chemical_compound": compound_obj,
                            "drug_chemical_compound": dcc_obj,
                            "drug": drug_obj,
                            "mapping_strength": mapping_strength,
                            "similarity_relation_type": similarity_rel_type,
                            "similarity_relation_present": True,
                            "numeric_evidence": numeric_evidence,
                        }
                    )

    if not mappings:
        for dcc in _ensure_list(nodes_by_label.get("DrugChemicalCompound")):
            if not isinstance(dcc, dict):
                continue
            dcc_obj = _format_drug_compound(dcc)
            mappings.append(
                {
                    "ingredient": {"id": None, "name": None},
                    "chemical_compound": {"id": None, "name": None},
                    "drug_chemical_compound": dcc_obj,
                    "drug": {"id": None, "name": None},
                    "mapping_strength": _normalize_strength(dcc_obj.get("mapping_strength"), default="WEAK"),
                    "similarity_relation_type": None,
                    "similarity_relation_present": False,
                    "numeric_evidence": {},
                }
            )

    seen: set[tuple[Any, ...]] = set()
    deduped: list[dict[str, Any]] = []
    for item in mappings:
        ingredient = item.get("ingredient") if isinstance(item.get("ingredient"), dict) else {}
        compound = item.get("chemical_compound") if isinstance(item.get("chemical_compound"), dict) else {}
        dcc = item.get("drug_chemical_compound") if isinstance(item.get("drug_chemical_compound"), dict) else {}
        drug = item.get("drug") if isinstance(item.get("drug"), dict) else {}

        fingerprint = (
            ingredient.get("id"),
            compound.get("id"),
            dcc.get("id"),
            drug.get("id"),
            item.get("mapping_strength"),
            item.get("similarity_relation_type"),
        )
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        deduped.append(item)

    return deduped


def build_graph_reasoning(subgraph: dict) -> dict:
    import logging
    from app.services.pipeline_tracer import get_tracer

    _logger = logging.getLogger(__name__)

    safe_subgraph = subgraph if isinstance(subgraph, dict) else {}
    nodes_by_label = _collect_nodes_by_label(safe_subgraph)

    _logger.info(
        "REASONING BUILDER: Input subgraph keys=%s | Detected labels=%s",
        list(safe_subgraph.keys()),
        sorted(nodes_by_label.keys()),
    )

    disease_nodes = _ensure_list(nodes_by_label.get("Disease"))
    ingredient_nodes = _ensure_list(nodes_by_label.get("Ingredient"))
    drug_nodes = _ensure_list(nodes_by_label.get("Drug"))
    compound_nodes = _ensure_list(nodes_by_label.get("ChemicalCompound"))
    drug_compound_nodes = _ensure_list(nodes_by_label.get("DrugChemicalCompound"))

    disease = _format_disease(disease_nodes[0] if disease_nodes else {})
    diseases = [_format_disease(item) for item in disease_nodes if isinstance(item, dict)]

    ingredient = _format_ingredient_node(ingredient_nodes[0] if ingredient_nodes else {})
    ingredients = [_format_ingredient_node(item) for item in ingredient_nodes if isinstance(item, dict)]

    drug = _format_drug_node(drug_nodes[0] if drug_nodes else {})
    drugs = [_format_drug_node(item) for item in drug_nodes if isinstance(item, dict)]

    compounds = [_format_named_node(item) for item in compound_nodes if isinstance(item, dict)]
    drug_compounds = [_format_drug_compound(item) for item in drug_compound_nodes if isinstance(item, dict)]

    hadith_refs = _extract_hadith_references(safe_subgraph, nodes_by_label)

    reasoning = {
        "Disease": disease,
        "Diseases": diseases,
        "Ingredient": ingredient,
        "Ingredients": ingredients,
        "Drug": drug,
        "Drugs": drugs,
        "ChemicalCompounds": compounds,
        "DrugChemicalCompounds": drug_compounds,
        "HadithReferences": hadith_refs,
        "BiochemicalMappings": _build_biochemical_mappings(safe_subgraph, nodes_by_label),
        "meta": {
            "has_error": bool(safe_subgraph.get("error")),
            "source_error": _safe_str(safe_subgraph.get("error")),
            "detected_labels": sorted(nodes_by_label.keys()),
            "schema_node_labels": sorted(_SCHEMA_LABELS) if _SCHEMA_LABELS else [],
            "hadith_linking_mode": "dynamic",
        },
    }

    _logger.info(
        "REASONING BUILDER: Output → ingredients=%d compounds=%d drug_compounds=%d drugs=%d hadith=%d mappings=%d",
        len(ingredients), len(compounds), len(drug_compounds), len(drugs), len(hadith_refs),
        len(reasoning["BiochemicalMappings"]),
    )

    tracer = get_tracer()
    if tracer:
        tracer.log_data("reasoning_builder_output_counts", {
            "ingredients": len(ingredients),
            "compounds": len(compounds),
            "drug_compounds": len(drug_compounds),
            "drugs": len(drugs),
            "hadith_refs": len(hadith_refs),
            "biochemical_mappings": len(reasoning["BiochemicalMappings"]),
        })

    return reasoning


__all__ = [
    "build_graph_reasoning",
]
