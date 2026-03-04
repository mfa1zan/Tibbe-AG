"""
Multi-Hop Discovery Mode (Level-2 Reasoning) for PRO-MedGraph.

When no direct Disease→Ingredient→Compound→Drug mapping is found, this
module performs a 2-hop exploratory search to discover indirect compound
similarities.  Results are flagged as "Hypothesis-Level" evidence.

Pipeline position:
    Intent Routing → Graph Retrieval → (if empty) **Multi-Hop Discovery** → Reasoning

This is an optional mode activated only when primary retrieval yields
zero biochemical mappings.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from functools import lru_cache

from neo4j import GraphDatabase

from app.config import get_settings

logger = logging.getLogger(__name__)


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
        logger.exception("Failed to load KG schema for multi-hop discovery")
    return {}


_KG_SCHEMA: dict[str, Any] = _load_kg_schema()


def _schema_label(name: str) -> str:
    labels = _KG_SCHEMA.get("nodeLabels") or []
    if isinstance(labels, list) and name in labels:
        return name
    return name


def _relationships_between(from_label: str, to_label: str) -> list[str]:
    rels: list[str] = []
    for edge in (_KG_SCHEMA.get("relationshipStructure") or []):
        if not isinstance(edge, dict):
            continue
        rel = edge.get("relationship")
        from_nodes = edge.get("from") or []
        to_nodes = edge.get("to") or []
        if isinstance(rel, str) and from_label in from_nodes and to_label in to_nodes:
            rels.append(rel)
    return rels


def _name_property(label: str) -> str:
    for node_prop in (_KG_SCHEMA.get("nodeProperties") or []):
        if not isinstance(node_prop, dict):
            continue
        labels = node_prop.get("labels") or []
        prop = node_prop.get("property")
        if label in labels and prop == "name":
            return "name"
    return "name"


def _first_relationship_between(from_label: str, to_label: str) -> str | None:
    rels = _relationships_between(from_label, to_label)
    return rels[0] if rels else None


def _typed_rel(rel_type: str | None, *, variable: str | None = None) -> str:
    if rel_type and variable:
        return f"[{variable}:{rel_type}]"
    if rel_type:
        return f"[:{rel_type}]"
    if variable:
        return f"[{variable}]"
    return "[]"


def _typed_rel_union(rel_types: list[str], *, variable: str) -> str:
    if rel_types:
        return f"[{variable}:{'|'.join(rel_types)}]"
    return f"[{variable}]"


@lru_cache(maxsize=1)
def _get_driver():
    settings = get_settings()
    return GraphDatabase.driver(
        settings.neo4j_uri,
        auth=(settings.neo4j_username, settings.neo4j_password),
    )


def discover_indirect_compound_links(disease_name: str) -> dict[str, Any]:
    """
    2-hop exploratory search:
    Disease ← CURES ← Ingredient → CONTAINS → Compound
    Then find OTHER Compounds with similar names or shared parent Ingredients
    that link to DrugChemicalCompounds via weaker paths.

    All results flagged as hypothesis_level=True.
    """
    driver = _get_driver()

    # Schema-driven KG retrieval enforced.
    disease_label = _schema_label("Disease")
    ingredient_label = _schema_label("Ingredient")
    chemical_label = _schema_label("ChemicalCompound")
    drug_chemical_label = _schema_label("DrugChemicalCompound")
    drug_label = _schema_label("Drug")

    disease_name_prop = _name_property(disease_label)

    cures_rel = _first_relationship_between(ingredient_label, disease_label)
    ingredient_contains_rel = _first_relationship_between(ingredient_label, chemical_label)
    drug_contains_rel = _first_relationship_between(drug_label, drug_chemical_label)
    similarity_rels = _relationships_between(chemical_label, drug_chemical_label)

    cures_pattern = _typed_rel(cures_rel)
    ingredient_contains_pattern = _typed_rel(ingredient_contains_rel)
    drug_contains_pattern = _typed_rel(drug_contains_rel)
    similarity_pattern = _typed_rel_union(similarity_rels, variable="rel")

    # Step 1: Find compounds from ingredients that cure this disease
    step1_query = f"""
        MATCH (d:{disease_label})<-{cures_pattern}-(i:{ingredient_label})-{ingredient_contains_pattern}->(cc:{chemical_label})
    WHERE toLower(d.{disease_name_prop}) = toLower($disease_name)
    RETURN DISTINCT elementId(cc) AS cc_id, cc.name AS cc_name,
           elementId(i) AS ing_id, i.name AS ing_name
    LIMIT 50
    """

    # Step 2: From those compounds, find any DrugChemicalCompound links
    # (including very weak ones) and the drugs they belong to
    step2_query = f"""
    MATCH (cc:{chemical_label})
    WHERE elementId(cc) IN $compound_ids
    OPTIONAL MATCH (cc)-{similarity_pattern}->(dcc:{drug_chemical_label})
    OPTIONAL MATCH (drug:{drug_label})-{drug_contains_pattern}->(dcc)
    RETURN elementId(cc) AS cc_id, cc.name AS cc_name,
           elementId(dcc) AS dcc_id, dcc.name AS dcc_name,
           type(rel) AS rel_type,
           elementId(drug) AS drug_id, drug.name AS drug_name
    LIMIT 100
    """

    # Step 3: Exploratory — find OTHER ingredients sharing compounds
    # with the disease's ingredients (indirect 2-hop connection)
    step3_query = f"""
    MATCH (d:{disease_label})<-{cures_pattern}-(i1:{ingredient_label})-{ingredient_contains_pattern}->(cc:{chemical_label})
    WHERE toLower(d.{disease_name_prop}) = toLower($disease_name)
    WITH COLLECT(DISTINCT cc) AS disease_compounds
    UNWIND disease_compounds AS cc
    MATCH (cc)<-{ingredient_contains_pattern}-(i2:{ingredient_label})
    WHERE NOT (i2)-{cures_pattern}->(d:{disease_label} {{{disease_name_prop}: $disease_name}})
    WITH i2, COLLECT(DISTINCT cc.name) AS shared_compounds
    WHERE size(shared_compounds) >= 1
    MATCH (i2)-{ingredient_contains_pattern}->(cc2:{chemical_label})
    OPTIONAL MATCH (cc2)-{similarity_pattern}->(dcc:{drug_chemical_label})
    OPTIONAL MATCH (drug:{drug_label})-{drug_contains_pattern}->(dcc)
    RETURN DISTINCT i2.name AS indirect_ingredient,
           shared_compounds AS shared_compounds,
           cc2.name AS exploratory_compound,
           dcc.name AS exploratory_dcc,
           type(rel) AS rel_type,
           drug.name AS exploratory_drug
    LIMIT 30
    """

    try:
        with driver.session() as session:
            # Step 1
            step1_rows = list(session.run(step1_query, disease_name=disease_name))
            compound_ids = [r.get("cc_id") for r in step1_rows if r.get("cc_id")]
            compound_map = {
                r.get("cc_id"): {"name": r.get("cc_name"), "ingredient": r.get("ing_name")}
                for r in step1_rows if r.get("cc_id")
            }

            # Step 2: Direct (but possibly weak) drug mappings
            direct_discoveries: list[dict[str, Any]] = []
            if compound_ids:
                step2_rows = list(session.run(step2_query, compound_ids=compound_ids))
                for row in step2_rows:
                    if row.get("dcc_id"):
                        cc_info = compound_map.get(row.get("cc_id"), {})
                        direct_discoveries.append({
                            "ingredient": cc_info.get("ingredient"),
                            "compound": row.get("cc_name"),
                            "drug_compound": row.get("dcc_name"),
                            "drug": row.get("drug_name"),
                            "relation_type": row.get("rel_type"),
                            "hop_type": "direct_weak",
                            "hypothesis_level": True,
                        })

            # Step 3: Indirect / exploratory (2-hop)
            indirect_discoveries: list[dict[str, Any]] = []
            try:
                step3_rows = list(session.run(step3_query, disease_name=disease_name))
                for row in step3_rows:
                    if row.get("exploratory_dcc"):
                        indirect_discoveries.append({
                            "indirect_ingredient": row.get("indirect_ingredient"),
                            "shared_compounds": row.get("shared_compounds"),
                            "exploratory_compound": row.get("exploratory_compound"),
                            "exploratory_drug_compound": row.get("exploratory_dcc"),
                            "exploratory_drug": row.get("exploratory_drug"),
                            "relation_type": row.get("rel_type"),
                            "hop_type": "indirect_2hop",
                            "hypothesis_level": True,
                        })
            except Exception:
                logger.warning("Step 3 exploratory query failed for '%s'", disease_name)

        total = len(direct_discoveries) + len(indirect_discoveries)
        logger.info(
            "Multi-hop discovery for '%s': %d direct, %d indirect",
            disease_name, len(direct_discoveries), len(indirect_discoveries),
        )

        return {
            "disease": disease_name,
            "direct_discoveries": direct_discoveries,
            "indirect_discoveries": indirect_discoveries,
            "total_discoveries": total,
            "hypothesis_level": True,
            "discovery_note": (
                "These are exploratory multi-hop discoveries. "
                "They should be treated as hypothesis-level evidence, "
                "not confirmed causal relationships."
            ),
        }

    except Exception:
        logger.exception("Multi-hop discovery failed for '%s'", disease_name)
        return {
            "disease": disease_name,
            "direct_discoveries": [],
            "indirect_discoveries": [],
            "total_discoveries": 0,
            "hypothesis_level": True,
            "error": "Multi-hop discovery query failed",
        }


def should_activate_discovery(reasoning: dict) -> bool:
    """
    Check if multi-hop discovery should be activated.
    Criteria: zero BiochemicalMappings in primary retrieval.
    """
    mappings = reasoning.get("BiochemicalMappings", [])
    return not mappings or len(mappings) == 0


def merge_discoveries_into_reasoning(
    reasoning: dict,
    discoveries: dict,
) -> dict:
    """
    Merge multi-hop discovery results into reasoning structure
    so they appear in the LLM context with appropriate hypothesis flags.
    """
    if not discoveries or discoveries.get("total_discoveries", 0) == 0:
        return reasoning

    reasoning["MultiHopDiscoveries"] = {
        "direct": discoveries.get("direct_discoveries", []),
        "indirect": discoveries.get("indirect_discoveries", []),
        "total": discoveries.get("total_discoveries", 0),
        "note": discoveries.get("discovery_note", ""),
    }

    reasoning.setdefault("meta", {})
    reasoning["meta"]["multi_hop_activated"] = True
    reasoning["meta"]["hypothesis_level_evidence"] = True

    return reasoning


__all__ = [
    "discover_indirect_compound_links",
    "should_activate_discovery",
    "merge_discoveries_into_reasoning",
]
