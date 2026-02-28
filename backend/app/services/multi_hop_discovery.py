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

import logging
from typing import Any

from neo4j import GraphDatabase

from app.config import get_settings

logger = logging.getLogger(__name__)


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

    # Step 1: Find compounds from ingredients that cure this disease
    step1_query = """
    MATCH (d:Disease)<-[:CURES]-(i:Ingredient)-[:CONTAINS]->(cc:ChemicalCompound)
    WHERE toLower(d.name) = toLower($disease_name)
    RETURN DISTINCT elementId(cc) AS cc_id, cc.name AS cc_name,
           elementId(i) AS ing_id, i.name AS ing_name
    LIMIT 50
    """

    # Step 2: From those compounds, find any DrugChemicalCompound links
    # (including very weak ones) and the drugs they belong to
    step2_query = """
    MATCH (cc:ChemicalCompound)
    WHERE elementId(cc) IN $compound_ids
    OPTIONAL MATCH (cc)-[rel:IS_IDENTICAL_TO|IS_LIKELY_EQUIVALENT_TO|IS_WEAK_MATCH_TO]->(dcc:DrugChemicalCompound)
    OPTIONAL MATCH (drug:Drug)-[:CONTAINS]->(dcc)
    RETURN elementId(cc) AS cc_id, cc.name AS cc_name,
           elementId(dcc) AS dcc_id, dcc.name AS dcc_name,
           type(rel) AS rel_type,
           elementId(drug) AS drug_id, drug.name AS drug_name
    LIMIT 100
    """

    # Step 3: Exploratory — find OTHER ingredients sharing compounds
    # with the disease's ingredients (indirect 2-hop connection)
    step3_query = """
    MATCH (d:Disease)<-[:CURES]-(i1:Ingredient)-[:CONTAINS]->(cc:ChemicalCompound)
    WHERE toLower(d.name) = toLower($disease_name)
    WITH COLLECT(DISTINCT cc) AS disease_compounds
    UNWIND disease_compounds AS cc
    MATCH (cc)<-[:CONTAINS]-(i2:Ingredient)
    WHERE NOT (i2)-[:CURES]->(d:Disease {name: $disease_name})
    WITH i2, COLLECT(DISTINCT cc.name) AS shared_compounds
    WHERE size(shared_compounds) >= 1
    MATCH (i2)-[:CONTAINS]->(cc2:ChemicalCompound)
    OPTIONAL MATCH (cc2)-[rel:IS_IDENTICAL_TO|IS_LIKELY_EQUIVALENT_TO|IS_WEAK_MATCH_TO]->(dcc:DrugChemicalCompound)
    OPTIONAL MATCH (drug:Drug)-[:CONTAINS]->(dcc)
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
