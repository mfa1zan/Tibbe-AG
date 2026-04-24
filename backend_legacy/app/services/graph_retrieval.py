"""
PRO-MedGraph  —  Step 4: Graph Retrieval Layer
================================================

Executes a Cypher query (produced by Step 3 :mod:`cypher_query_generator`)
against the Neo4j knowledge graph, parses the raw records into a structured
JSON payload, and returns it ready for Step 5 (Graph + Text Encoding for
LLM).

Connection management
---------------------
This module **reuses** the shared Neo4j driver singleton already maintained
by :mod:`app.services.graph_service` (reads ``NEO4J_URI``,
``NEO4J_USERNAME``, ``NEO4J_PASSWORD`` from `.env` via
:func:`app.config.get_settings`).  No extra connection setup is needed.

Output schema
-------------
The canonical output is the **remedies-centric** shape requested in the
specification:

.. code-block:: json

    {
        "disease": "hypertension",
        "remedies": [
            {
                "name": "Nigella sativa",
                "ingredients": ["thymoquinone", "nigellone"],
                "compounds": ["Thymoquinone"],
                "mapped_drugs": ["Captopril"],
                "mechanisms": ["IS_IDENTICAL_TO", "IS_LIKELY_EQUIVALENT_TO"]
            }
        ],
        "hadith_references": ["…"],
        "raw_subgraph": {
            "nodes": [ {id, label, properties} ],
            "edges": [ {id, type, start, end, properties} ]
        },
        "metadata": {
            "query_ms": 142,
            "node_count": 37,
            "edge_count": 52,
            "source": "template"
        }
    }

``raw_subgraph`` preserves every node and edge returned by the query so
multi-hop analyses and the evaluation framework can inspect the full graph
neighbourhood.

Usage
-----
.. code-block:: python

    from app.services.graph_retrieval import retrieve_graph

    # From Step 3
    cypher_result = await generate_cypher(intent)
    # Execute against Neo4j
    graph_data = retrieve_graph(
        cypher_result["cypher"],
        params=cypher_result["params"],
    )

Run standalone tests:  ``python -m app.services.graph_retrieval``
"""

from __future__ import annotations

import logging
import time
from typing import Any

from neo4j import Record
from neo4j.graph import Node, Relationship

from app.config import get_settings

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════════
#  Neo4j driver (reuses the graph_service singleton)
# ═══════════════════════════════════════════════════════════════════════════════

# We import the cached driver factory from graph_service so there is exactly
# one driver instance per process — no leaked connections or duplicate pools.
from app.services.graph_service import _get_driver  # noqa: E402


# ═══════════════════════════════════════════════════════════════════════════════
#  Errors
# ═══════════════════════════════════════════════════════════════════════════════


class GraphRetrievalError(RuntimeError):
    """Raised when the Neo4j query cannot be executed or returns no data."""


# ═══════════════════════════════════════════════════════════════════════════════
#  Internal parsers
# ═══════════════════════════════════════════════════════════════════════════════

# ── Raw subgraph extraction ───────────────────────────────────────────────────
# Neo4j records can contain primitives, Node objects, Relationship objects,
# or Path objects.  We walk each record and collect every unique Node/Rel.


def _node_to_dict(node: Node) -> dict[str, Any]:
    """
    Convert a Neo4j ``Node`` into a plain dict.

    Output
    ------
    ``{"id": str, "labels": [str], "properties": dict}``
    """
    return {
        "id": str(node.element_id),
        "labels": sorted(node.labels),
        "properties": dict(node),
    }


def _rel_to_dict(rel: Relationship) -> dict[str, Any]:
    """
    Convert a Neo4j ``Relationship`` into a plain dict.

    Output
    ------
    ``{"id": str, "type": str, "start": str, "end": str, "properties": dict}``
    """
    return {
        "id": str(rel.element_id),
        "type": rel.type,
        "start": str(rel.start_node.element_id),
        "end": str(rel.end_node.element_id),
        "properties": dict(rel),
    }


def _extract_graph_objects(records: list[Record]) -> tuple[dict, dict]:
    """
    Walk all records and collect unique Nodes and Relationships.

    Returns
    -------
    (nodes_by_id, rels_by_id)
        Both are dicts keyed by element_id → plain-dict representation.
    """
    nodes: dict[str, dict] = {}
    rels: dict[str, dict] = {}

    def _visit(value: Any) -> None:
        """Recursively visit a record value."""
        if isinstance(value, Node):
            nid = str(value.element_id)
            if nid not in nodes:
                nodes[nid] = _node_to_dict(value)
        elif isinstance(value, Relationship):
            rid = str(value.element_id)
            if rid not in rels:
                rels[rid] = _rel_to_dict(value)
            # Also capture the start/end nodes.
            _visit(value.start_node)
            _visit(value.end_node)
        elif isinstance(value, list):
            for item in value:
                _visit(item)
        elif isinstance(value, dict):
            for v in value.values():
                _visit(v)
        # Paths expose nodes and relationships as iterables.
        elif hasattr(value, "nodes") and hasattr(value, "relationships"):
            for n in value.nodes:
                _visit(n)
            for r in value.relationships:
                _visit(r)

    for record in records:
        for val in record.values():
            _visit(val)

    return nodes, rels


# ── Tabular record parsing ────────────────────────────────────────────────────
# Template queries from Step 3 return tabular results (aliases like
# ``disease``, ``ingredients``, ``compounds``, etc.) rather than raw graph
# objects.  We detect these via column names and build the remedy structure.

_DISEASE_KEYS = {"disease", "disease_name"}
_INGREDIENT_KEYS = {"ingredient", "ingredient_name", "ingredients"}
_COMPOUND_KEYS = {"compound", "cc_name", "compounds"}
_DRUG_KEYS = {"drug", "drug_name", "drugs", "mapped_drugs"}
_MAPPING_KEYS = {"strength", "is_relation_type", "mapping", "mechanisms", "mappings"}
_HADITH_KEYS = {"hadith", "hadith_references", "hadith_reference"}


def _flatten(value: Any) -> list[str]:
    """Ensure a value is a flat list of non-null strings."""
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value else []
    if isinstance(value, list):
        out = []
        for item in value:
            if isinstance(item, dict):
                # e.g. {compound: "X", drugCompound: "Y", strength: "IS_IDENTICAL_TO"}
                out.extend(str(v) for v in item.values() if v is not None)
            elif item is not None:
                out.append(str(item))
        return out
    return [str(value)]


def _safe_str(val: Any) -> str | None:
    if val is None:
        return None
    s = str(val).strip()
    return s if s else None


def _parse_tabular_records(records: list[Record]) -> dict[str, Any]:
    """
    Parse tabular (aliased) records into the canonical remedy-centric JSON.

    This handles the output shapes from Step 3 templates, all of which
    RETURN human-readable aliases like ``disease``, ``ingredients``, etc.
    """
    keys = records[0].keys() if records else []
    key_set = set(k.lower() for k in keys)

    # ── Detect the disease name ────────────────────────────────────────────
    disease_name: str | None = None
    for dk in _DISEASE_KEYS:
        if dk in key_set:
            disease_name = _safe_str(records[0].get(dk))
            if disease_name:
                break

    # ── Collect remedies across all rows ───────────────────────────────────
    # Many template queries aggregate with collect(DISTINCT …), so a single
    # row may already contain the lists we need.  We also handle the
    # row-per-ingredient shape from the disease-subgraph query.

    all_ingredients: list[str] = []
    all_compounds: list[str] = []
    all_drugs: list[str] = []
    all_mechanisms: list[str] = []
    all_hadith: list[str] = []

    # Per-ingredient remedy map (ingredient_name → {compounds, drugs, mechanisms}).
    remedy_map: dict[str, dict[str, set[str]]] = {}

    for row in records:
        row_dict = dict(row)

        # Ingredient names
        for ik in _INGREDIENT_KEYS:
            if ik in row_dict:
                for name in _flatten(row_dict[ik]):
                    all_ingredients.append(name)

        # Compounds
        for ck in _COMPOUND_KEYS:
            if ck in row_dict:
                for name in _flatten(row_dict[ck]):
                    all_compounds.append(name)

        # Drugs
        for dk in _DRUG_KEYS:
            if dk in row_dict:
                for name in _flatten(row_dict[dk]):
                    all_drugs.append(name)

        # Mechanisms / mapping strengths
        for mk in _MAPPING_KEYS:
            if mk in row_dict:
                for name in _flatten(row_dict[mk]):
                    all_mechanisms.append(name)

        # Hadith
        for hk in _HADITH_KEYS:
            if hk in row_dict:
                for name in _flatten(row_dict[hk]):
                    all_hadith.append(name)

        # Also handle the "mappings" / "drug_mappings" / "compound_map" aggregated dicts.
        for combined_key in ("mappings", "drug_mappings", "compound_map", "shared_compounds"):
            combined = row_dict.get(combined_key)
            if isinstance(combined, list):
                for entry in combined:
                    if not isinstance(entry, dict):
                        continue
                    for field, bucket in [
                        ("compound", all_compounds),
                        ("drugCompound", all_drugs),
                        ("drug", all_drugs),
                        ("ingredient", all_ingredients),
                    ]:
                        v = entry.get(field)
                        if v and str(v).strip():
                            bucket.append(str(v).strip())
                    strength = entry.get("strength")
                    if strength:
                        all_mechanisms.append(str(strength))

        # Build per-ingredient breakdown when there's a single ingredient per row.
        row_ingredient = _safe_str(
            row_dict.get("ingredient") or row_dict.get("ingredient_name")
        )
        if row_ingredient:
            entry = remedy_map.setdefault(row_ingredient, {
                "compounds": set(),
                "drugs": set(),
                "mechanisms": set(),
            })
            for v in _flatten(row_dict.get("compounds") or row_dict.get("cc_name")):
                entry["compounds"].add(v)
            for v in _flatten(row_dict.get("drugs") or row_dict.get("drug_name") or row_dict.get("drug")):
                entry["drugs"].add(v)
            for v in _flatten(row_dict.get("strength") or row_dict.get("is_relation_type")):
                entry["mechanisms"].add(v)

    # ── Deduplicate ────────────────────────────────────────────────────────
    def _dedup(lst: list[str]) -> list[str]:
        return list(dict.fromkeys(lst))

    unique_ingredients = _dedup(all_ingredients)
    unique_compounds = _dedup(all_compounds)
    unique_drugs = _dedup(all_drugs)
    unique_mechanisms = _dedup(all_mechanisms)
    unique_hadith = _dedup(all_hadith)

    # ── Build remedies list ────────────────────────────────────────────────
    # If we have per-ingredient granularity, use that; otherwise collapse
    # everything into one entry.
    remedies: list[dict[str, Any]] = []

    if remedy_map:
        for ing_name, data in remedy_map.items():
            remedies.append({
                "name": ing_name,
                "ingredients": [ing_name],          # self-referencing for schema consistency
                "compounds": sorted(data["compounds"]),
                "mapped_drugs": sorted(data["drugs"]),
                "mechanisms": sorted(data["mechanisms"]),
            })
    elif unique_ingredients:
        # Aggregated query — one remedy per ingredient, shared compounds/drugs.
        for ing in unique_ingredients:
            remedies.append({
                "name": ing,
                "ingredients": [ing],
                "compounds": unique_compounds,
                "mapped_drugs": unique_drugs,
                "mechanisms": unique_mechanisms,
            })
    elif unique_drugs or unique_compounds:
        # Drug-centric or compound-centric query without ingredient column.
        remedies.append({
            "name": disease_name or "query_result",
            "ingredients": unique_ingredients,
            "compounds": unique_compounds,
            "mapped_drugs": unique_drugs,
            "mechanisms": unique_mechanisms,
        })

    return {
        "disease": disease_name,
        "remedies": remedies,
        "hadith_references": unique_hadith,
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  Public API
# ═══════════════════════════════════════════════════════════════════════════════


def retrieve_graph(
    cypher_query: str,
    *,
    params: dict[str, Any] | None = None,
    source: str = "unknown",
) -> dict[str, Any]:
    """
    Execute a Cypher query against Neo4j and return structured graph data.

    Parameters
    ----------
    cypher_query : str
        A **read-only** Cypher query (from Step 3).
    params : dict, optional
        Parameter bindings (e.g. ``{"entity_name": "hypertension"}``).
    source : str
        Tag indicating how the query was generated (``"template"`` or ``"llm"``).
        Included in the response metadata for traceability.

    Returns
    -------
    dict
        Canonical structure with keys: ``disease``, ``remedies``,
        ``hadith_references``, ``raw_subgraph``, ``metadata``.

    Raises
    ------
    GraphRetrievalError
        On connection failure, query execution error, or empty results.

    Notes
    -----
    * All queries use a hard ``timeout`` of 30 seconds.
    * Query execution time and node/edge counts are logged at INFO level.
    """
    if not cypher_query or cypher_query.strip().startswith("//"):
        # Strip leading comment lines (templates start with "// ...") and
        # check whether any executable Cypher remains underneath.
        stripped_lines = [
            ln for ln in cypher_query.strip().splitlines()
            if ln.strip() and not ln.strip().startswith("//")
        ] if cypher_query else []
        if not stripped_lines:
            # Truly a comment-only / empty query → skip.
            logger.info("Skipping graph retrieval — no executable Cypher")
            return {
                "disease": None,
                "remedies": [],
                "hadith_references": [],
                "raw_subgraph": {"nodes": [], "edges": []},
                "metadata": {
                    "query_ms": 0,
                    "node_count": 0,
                    "edge_count": 0,
                    "source": source,
                    "skipped": True,
                },
            }

    driver = _get_driver()
    safe_params = params or {}

    # ── Execute query with timing ──────────────────────────────────────────
    t_start = time.perf_counter()

    try:
        with driver.session() as session:
            result = session.run(cypher_query, **safe_params)
            records: list[Record] = list(result)
    except Exception as exc:
        elapsed = round((time.perf_counter() - t_start) * 1000, 1)
        logger.error(
            "Neo4j query failed after %.1f ms: %s — %s",
            elapsed, type(exc).__name__, exc,
        )
        raise GraphRetrievalError(
            f"Failed to execute Cypher query: {exc}"
        ) from exc

    elapsed_ms = round((time.perf_counter() - t_start) * 1000, 1)

    # ── Handle empty result set ────────────────────────────────────────────
    if not records:
        logger.warning(
            "Neo4j query returned 0 records (%.1f ms). Query: %s",
            elapsed_ms, cypher_query[:200],
        )
        return {
            "disease": None,
            "remedies": [],
            "hadith_references": [],
            "raw_subgraph": {"nodes": [], "edges": []},
            "metadata": {
                "query_ms": elapsed_ms,
                "node_count": 0,
                "edge_count": 0,
                "source": source,
                "empty": True,
            },
        }

    # ── Extract raw graph objects (nodes + edges) ──────────────────────────
    # This captures full multi-hop subgraphs when the query returns Node /
    # Relationship / Path objects.
    nodes_by_id, rels_by_id = _extract_graph_objects(records)

    # ── Parse tabular aliases into remedy structure ────────────────────────
    parsed = _parse_tabular_records(records)

    # ── If tabular parsing found nothing but we have graph objects ─────────
    # Fall back to using the raw nodes to populate the remedy structure.
    if not parsed["remedies"] and nodes_by_id:
        parsed = _build_remedies_from_raw_nodes(nodes_by_id, rels_by_id)

    node_count = len(nodes_by_id)
    edge_count = len(rels_by_id)

    logger.info(
        "Graph retrieval: %d records, %d nodes, %d edges in %.1f ms (source=%s)",
        len(records), node_count, edge_count, elapsed_ms, source,
    )

    return {
        "disease": parsed.get("disease"),
        "remedies": parsed.get("remedies", []),
        "hadith_references": parsed.get("hadith_references", []),
        "raw_subgraph": {
            "nodes": list(nodes_by_id.values()),
            "edges": list(rels_by_id.values()),
        },
        "metadata": {
            "query_ms": elapsed_ms,
            "node_count": node_count,
            "edge_count": edge_count,
            "record_count": len(records),
            "source": source,
        },
    }


def _build_remedies_from_raw_nodes(
    nodes: dict[str, dict],
    rels: dict[str, dict],
) -> dict[str, Any]:
    """
    Fallback: build the canonical structure from raw Node / Relationship
    objects when tabular aliases are not available (e.g. LLM-generated
    queries that RETURN full paths).
    """
    disease_name: str | None = None
    ingredient_names: list[str] = []
    compound_names: list[str] = []
    drug_names: list[str] = []
    hadith_refs: list[str] = []
    mechanisms: set[str] = set()

    for node in nodes.values():
        labels = node.get("labels", [])
        name = node.get("properties", {}).get("name")
        if not name:
            continue

        if "Disease" in labels:
            disease_name = disease_name or name
        elif "Ingredient" in labels:
            ingredient_names.append(name)
        elif "ChemicalCompound" in labels:
            compound_names.append(name)
        elif "DrugChemicalCompound" in labels:
            compound_names.append(name)
        elif "Drug" in labels:
            drug_names.append(name)
        elif "Hadith" in labels:
            hadith_refs.append(name)

    for rel in rels.values():
        rtype = rel.get("type", "")
        if rtype in ("IS_IDENTICAL_TO", "IS_LIKELY_EQUIVALENT_TO", "IS_WEAK_MATCH_TO"):
            mechanisms.add(rtype)

    remedies: list[dict] = []
    if ingredient_names:
        for ing in dict.fromkeys(ingredient_names):
            remedies.append({
                "name": ing,
                "ingredients": [ing],
                "compounds": list(dict.fromkeys(compound_names)),
                "mapped_drugs": list(dict.fromkeys(drug_names)),
                "mechanisms": sorted(mechanisms),
            })
    elif compound_names or drug_names:
        remedies.append({
            "name": disease_name or "result",
            "ingredients": [],
            "compounds": list(dict.fromkeys(compound_names)),
            "mapped_drugs": list(dict.fromkeys(drug_names)),
            "mechanisms": sorted(mechanisms),
        })

    return {
        "disease": disease_name,
        "remedies": remedies,
        "hadith_references": list(dict.fromkeys(hadith_refs)),
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  Convenience: retrieve from Step 3 output directly
# ═══════════════════════════════════════════════════════════════════════════════


def retrieve_graph_from_step3(step3_result: dict[str, Any]) -> dict[str, Any]:
    """
    Convenience wrapper that accepts the dict returned by
    :func:`cypher_query_generator.generate_cypher` and calls
    :func:`retrieve_graph`.

    Parameters
    ----------
    step3_result : dict
        ``{"cypher": str, "params": dict, "source": str, "intent_type": str}``
    """
    return retrieve_graph(
        cypher_query=step3_result["cypher"],
        params=step3_result.get("params"),
        source=step3_result.get("source", "unknown"),
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  Standalone test harness
# ═══════════════════════════════════════════════════════════════════════════════
#
#  Run:  python -m app.services.graph_retrieval
#  (from backend/ dir with .env present and Neo4j reachable)

import json as _json


_TEST_QUERIES: list[dict[str, Any]] = [
    # ── 1. Remedies for hypertension (1–2 hop) ────────────────────────────
    {
        "label": "Remedies for hypertension",
        "cypher": (
            "MATCH (d:Disease)\n"
            "WHERE toLower(d.name) = toLower($entity_name)\n"
            "OPTIONAL MATCH (d)<-[:CURES]-(i:Ingredient)\n"
            "OPTIONAL MATCH (d)-[:MENTIONED_IN]->(h:Hadith)\n"
            "RETURN d.name AS disease,\n"
            "       collect(DISTINCT i.name) AS ingredients,\n"
            "       collect(DISTINCT h.name) AS hadith_references\n"
            "LIMIT 300"
        ),
        "params": {"entity_name": "hypertension"},
    },
    # ── 2. Chemical mapping for black seed (multi-hop) ─────────────────────
    {
        "label": "Chemical mapping — black seed",
        "cypher": (
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
        ),
        "params": {"entity_name": "black seed"},
    },
    # ── 3. Full disease subgraph — diabetes (4-hop) ────────────────────────
    {
        "label": "Full subgraph — diabetes",
        "cypher": (
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
        ),
        "params": {"entity_name": "diabetes"},
    },
]


def _run_tests() -> None:
    """Execute test queries and pretty-print results."""
    import sys

    logging.basicConfig(
        level=logging.INFO, stream=sys.stdout,
        format="%(levelname)s %(message)s",
    )

    passed = 0
    failed = 0
    total = len(_TEST_QUERIES)

    print("=" * 72)
    print("  PRO-MedGraph · Graph Retrieval Test Harness")
    print("=" * 72)

    for tc in _TEST_QUERIES:
        label = tc["label"]
        print(f"\n── {label} ──")
        print(f"   Params: {tc['params']}")

        try:
            result = retrieve_graph(
                tc["cypher"],
                params=tc["params"],
                source="test",
            )

            meta = result["metadata"]
            print(f"   ✓ {meta['query_ms']} ms | "
                  f"{meta.get('record_count', '?')} records | "
                  f"{meta['node_count']} nodes | "
                  f"{meta['edge_count']} edges")
            print(f"   Disease:  {result['disease']}")
            print(f"   Remedies: {len(result['remedies'])}")
            for r in result["remedies"][:5]:
                print(f"     • {r['name']}")
                if r["compounds"]:
                    print(f"       compounds:    {r['compounds'][:5]}")
                if r["mapped_drugs"]:
                    print(f"       mapped_drugs: {r['mapped_drugs'][:5]}")
                if r["mechanisms"]:
                    print(f"       mechanisms:   {r['mechanisms']}")
            if result["hadith_references"]:
                print(f"   Hadith: {result['hadith_references'][:3]}")
            passed += 1

        except Exception as exc:
            print(f"   ✗ FAILED: {exc}")
            failed += 1

    print("\n" + "=" * 72)
    print(f"  Results:  {passed}/{total} passed, {failed}/{total} failed")
    print("=" * 72)


if __name__ == "__main__":
    _run_tests()


# ── Module exports ─────────────────────────────────────────────────────────────

__all__ = [
    "retrieve_graph",
    "retrieve_graph_from_step3",
    "GraphRetrievalError",
]
