from functools import lru_cache
from typing import Any

from neo4j import GraphDatabase

from app.config import get_settings


def _clean_required_param(value: str | None) -> str | None:
    if not isinstance(value, str):
        return None
    cleaned = value.strip()
    return cleaned or None


def _log_missing_param(
    *,
    purpose: str,
    query: str,
    param_name: str,
    raw_value: Any,
) -> None:
    from app.services.pipeline_tracer import get_tracer

    logger.error("KG QUERY SKIPPED: %s missing required parameter '%s' (value=%r)", purpose, param_name, raw_value)
    tracer = get_tracer()
    if tracer:
        tracer.log_kg_query(
            purpose=purpose,
            cypher=query.strip(),
            parameters={param_name: raw_value},
            row_count=0,
            duration_ms=0.0,
            error=f"Missing required parameter: {param_name}",
        )


def _map_similarity_relation(relation_type: str | None) -> str | None:
    if relation_type == "IS_IDENTICAL_TO":
        return "IDENTICAL"
    if relation_type == "IS_LIKELY_EQUIVALENT_TO":
        return "LIKELY"
    if relation_type == "IS_WEAK_MATCH_TO":
        return "WEAK"
    return None


@lru_cache(maxsize=1)
def _get_driver():
    settings = get_settings()
    return GraphDatabase.driver(
        settings.neo4j_uri,
        auth=(settings.neo4j_username, settings.neo4j_password),
    )


def close_graph_driver() -> None:
    """Close Neo4j driver explicitly on shutdown hooks/tests."""
    driver = _get_driver()
    driver.close()
    _get_driver.cache_clear()


def find_disease_by_name(name: str) -> dict:
    """Fetch a Disease node by name (case-insensitive) with category metadata."""
    driver = _get_driver()

    query = """
    // Find disease by case-insensitive name and attach optional disease category.
    MATCH (d:Disease)
    WHERE toLower(d.name) = toLower($name)
    OPTIONAL MATCH (dc:DiseaseCategory)-[:HAS_DISEASE]->(d)
    RETURN elementId(d) AS id,
           d.name AS name,
           dc.name AS category
    LIMIT 1
    """

    logger.info("┌─ KG QUERY: find_disease_by_name('%s')", name)
    t0 = time.perf_counter()

    clean_name = _clean_required_param(name)
    if not clean_name:
        _log_missing_param(
            purpose="find_disease_by_name",
            query=query,
            param_name="name",
            raw_value=name,
        )
        return {"error": "Missing required parameter: name", "name": name}

    try:
        with driver.session() as session:
            params = {"name": clean_name}
            result = session.run(query, params)
            row = result.single()

        if not row:
            logger.info("└─ KG RESULT: disease NOT FOUND for '%s' (%.0fms)", name, duration_ms)
            tracer = get_tracer()
            if tracer:
                tracer.log_kg_query(
                    purpose=f"find_disease_by_name('{name}')",
                    cypher=query.strip(),
                    parameters={"name": clean_name},
                    row_count=0,
                    result_sample=None,
                    duration_ms=duration_ms,
                )
            return {"error": "Disease not found", "name": name}

        return {
            "id": row.get("id"),
            "name": row.get("name"),
            "category": row.get("category"),
        }
        logger.info("└─ KG RESULT: disease='%s' category='%s' (%.0fms)", result_dict["name"], result_dict["category"], duration_ms)

        tracer = get_tracer()
        if tracer:
            tracer.log_kg_query(
                purpose=f"find_disease_by_name('{name}')",
                cypher=query.strip(),
                parameters={"name": clean_name},
                row_count=1,
                result_sample=result_dict,
                duration_ms=duration_ms,
            )

        return result_dict
    except Exception as exc:
        duration_ms = (time.perf_counter() - t0) * 1000
        logger.exception("KG QUERY FAILED: find_disease_by_name('%s') (%.0fms)", name, duration_ms)
        tracer = get_tracer()
        if tracer:
            tracer.log_kg_query(
                purpose=f"find_disease_by_name('{name}')",
                cypher=query.strip(),
                parameters={"name": clean_name},
                row_count=0,
                duration_ms=duration_ms,
                error=str(exc),
            )
        return {
            "error": "Failed to query disease",
            "details": str(exc),
        }


def get_disease_subgraph(disease_name: str) -> dict:
    """
    Traverse:
    Disease <-[CURES]- Ingredient -[CONTAINS]-> ChemicalCompound
      -[IS_*]-> DrugChemicalCompound <-[CONTAINS]- Drug
    Return a structured subgraph dictionary with relation type markers.
    """
    disease = find_disease_by_name(disease_name)
    if disease.get("error"):
        return {
            "error": disease["error"],
            "Disease": None,
            "Ingredients": [],
            "ChemicalCompounds": [],
            "DrugChemicalCompounds": [],
            "Drugs": [],
            "Relations": [],
        }

    driver = _get_driver()
    query = """
    // Traverse the requested biomedical chain from disease to ingredients, compounds, and drugs.
    MATCH (d:Disease)
    WHERE toLower(d.name) = toLower($disease_name)
    OPTIONAL MATCH (d)-[:CURES]-(ingredient:Ingredient)
    OPTIONAL MATCH (ingredient)-[:CONTAINS]->(cc:ChemicalCompound)
    OPTIONAL MATCH (cc)-[isrel:IS_IDENTICAL_TO|IS_LIKELY_EQUIVALENT_TO|IS_WEAK_MATCH_TO]->(dcc:DrugChemicalCompound)
    OPTIONAL MATCH (drug:Drug)-[:CONTAINS]->(dcc)
    RETURN DISTINCT elementId(d) AS disease_id,
           d.name AS disease_name,
           elementId(ingredient) AS ingredient_id,
           ingredient.name AS ingredient_name,
           elementId(cc) AS cc_id,
           cc.name AS cc_name,
           elementId(dcc) AS dcc_id,
           dcc.name AS dcc_name,
           elementId(drug) AS drug_id,
           drug.name AS drug_name,
           type(isrel) AS is_relation_type
    """

    try:
        clean_disease = _clean_required_param(disease_name)
        if not clean_disease:
            _log_missing_param(
                purpose="get_disease_subgraph",
                query=query,
                param_name="disease_name",
                raw_value=disease_name,
            )
            return {
                "error": "Missing required parameter: disease_name",
                "Disease": disease,
                "Ingredients": [],
                "ChemicalCompounds": [],
                "DrugChemicalCompounds": [],
                "Drugs": [],
                "Relations": [],
            }

        with driver.session() as session:
            params = {"disease_name": clean_disease}
            rows = list(session.run(query, params))

        raw_rows = [dict(row) for row in rows]

        logger.info(
            "┌─ KG QUERY: get_disease_subgraph('%s') → %d rows",
            disease_name, len(rows),
        )
        logger.debug("│  KG RAW ROWS: %s", raw_rows)

        ingredients: dict[str, dict[str, Any]] = {}
        compounds: dict[str, dict[str, Any]] = {}
        drug_compounds: dict[str, dict[str, Any]] = {}
        drugs: dict[str, dict[str, Any]] = {}
        relations: list[dict[str, Any]] = []
        relation_keys: set[tuple[str, str, str]] = set()

        def add_relation(from_id: str | None, to_id: str | None, relation_type: str | None) -> None:
            if not from_id or not to_id or not relation_type:
                return
            key = (from_id, to_id, relation_type)
            if key in relation_keys:
                return
            relation_keys.add(key)
            relations.append({"from": from_id, "to": to_id, "type": relation_type})

        for row in rows:
            ingredient_id = row.get("ingredient_id")
            if ingredient_id:
                ingredients[ingredient_id] = {
                    "id": ingredient_id,
                    "name": row.get("ingredient_name"),
                }

            cc_id = row.get("cc_id")
            if cc_id:
                compounds[cc_id] = {
                    "id": cc_id,
                    "name": row.get("cc_name"),
                }

            dcc_id = row.get("dcc_id")
            raw_rel_type = row.get("is_relation_type")
            match_type = _map_similarity_relation(raw_rel_type)
            if dcc_id:
                existing = drug_compounds.get(dcc_id, {"id": dcc_id, "name": row.get("dcc_name")})
                if isinstance(raw_rel_type, str) and raw_rel_type.strip():
                    existing["relation_type"] = raw_rel_type.strip()
                if match_type:
                    existing["mapping_strength"] = match_type
                drug_compounds[dcc_id] = existing

            drug_id = row.get("drug_id")
            if drug_id:
                drugs[drug_id] = {
                    "id": drug_id,
                    "name": row.get("drug_name"),
                }

            add_relation(ingredient_id, cc_id, "CONTAINS")
            add_relation(cc_id, dcc_id, raw_rel_type)
            add_relation(drug_id, dcc_id, "CONTAINS")
            add_relation(ingredient_id, disease.get("id"), "CURES")

        return {
            "Disease": disease,
            "Ingredients": list(ingredients.values()),
            "ChemicalCompounds": list(compounds.values()),
            "DrugChemicalCompounds": list(drug_compounds.values()),
            "Drugs": list(drugs.values()),
            "Relations": relations,
        }

        logger.info(
            "└─ KG SUBGRAPH: disease='%s' ingredients=%d compounds=%d drug_compounds=%d drugs=%d relations=%d",
            disease_name,
            len(result["Ingredients"]),
            len(result["ChemicalCompounds"]),
            len(result["DrugChemicalCompounds"]),
            len(result["Drugs"]),
            len(result["Relations"]),
        )

        from app.services.pipeline_tracer import get_tracer
        tracer = get_tracer()
        if tracer:
            tracer.log_kg_query(
                purpose=f"get_disease_subgraph('{disease_name}')",
                cypher=query.strip(),
                parameters={"disease_name": clean_disease},
                row_count=len(rows),
                result_sample={
                    "ingredients": [i.get("name") for i in list(ingredients.values())[:5]],
                    "compounds": [c.get("name") for c in list(compounds.values())[:5]],
                    "drug_compounds": [dc.get("name") for dc in list(drug_compounds.values())[:5]],
                    "drugs": [d.get("name") for d in list(drugs.values())[:5]],
                },
            )

        return result
    except Exception as exc:
        return {
            "error": "Failed to query disease subgraph",
            "details": str(exc),
            "Disease": disease,
            "Ingredients": [],
            "ChemicalCompounds": [],
            "DrugChemicalCompounds": [],
            "Drugs": [],
            "Relations": [],
        }


def get_hadith_for_disease(disease_name: str) -> list:
    """Return hadith references connected to a disease.

    Schema-aligned fields:
    - Hadith.name
    - Reference.reference via (Reference)-[:HAS_HADITH]->(Hadith)
    """
    driver = _get_driver()

    query = """
    // Fetch hadith linked to disease and enrich with optional reference details.
    MATCH (d:Disease)
    WHERE toLower(d.name) = toLower($disease_name)
    OPTIONAL MATCH (d)-[:MENTIONED_IN]->(h:Hadith)
    OPTIONAL MATCH (r:Reference)-[:HAS_HADITH]->(h)
    RETURN DISTINCT h.name AS name,
           coalesce(r.reference, "Unknown") AS reference
    LIMIT $limit
    """

    logger.info("┌─ KG QUERY: get_hadith_for_disease('%s')", disease_name)
    t0 = time.perf_counter()

    clean_disease = _clean_required_param(disease_name)
    if not clean_disease:
        _log_missing_param(
            purpose="get_hadith_for_disease",
            query=query,
            param_name="disease_name",
            raw_value=disease_name,
        )
        return []

    try:
        with driver.session() as session:
            params = {"disease_name": clean_disease, "limit": 100}
            rows = list(session.run(query, params))

        hadith_items: list[dict[str, str]] = []
        for row in rows:
            name = row.get("name")
            if not name:
                continue

            hadith_items.append(
                {
                    "name": name,
                    "book": "Unknown",
                    "reference": row.get("reference") or "Unknown",
                }
            )

        logger.info("└─ KG RESULT: %d hadith references for '%s' (%.0fms)", len(hadith_items), disease_name, duration_ms)

        tracer = get_tracer()
        if tracer:
            tracer.log_kg_query(
                purpose=f"get_hadith_for_disease('{disease_name}')",
                cypher=query.strip(),
                parameters={"disease_name": clean_disease, "limit": 100},
                row_count=len(hadith_items),
                result_sample=hadith_items[:3],
                duration_ms=duration_ms,
            )

        return hadith_items
    except Exception:
        return []


def get_ingredients_for_disease(disease_name: str) -> list[dict[str, str]]:
    """Return all ingredients linked to a disease via CURES as a clean list of id/name items."""
    driver = _get_driver()

    query = """
    // Fetch all disease-linked ingredients directly for deterministic ingredient answers.
    MATCH (d:Disease)
    WHERE toLower(d.name) = toLower($disease_name)
    OPTIONAL MATCH (d)<-[:CURES]-(i:Ingredient)
    WHERE i.name IS NOT NULL
    RETURN DISTINCT elementId(i) AS id,
           i.name AS name
    ORDER BY i.name ASC
    LIMIT $limit
    """

    logger.info("┌─ KG QUERY: get_ingredients_for_disease('%s')", disease_name)
    t0 = time.perf_counter()

    clean_disease = _clean_required_param(disease_name)
    if not clean_disease:
        _log_missing_param(
            purpose="get_ingredients_for_disease",
            query=query,
            param_name="disease_name",
            raw_value=disease_name,
        )
        return []

    try:
        with driver.session() as session:
            params = {"disease_name": clean_disease, "limit": 500}
            rows = list(session.run(query, params))

        ingredients: list[dict[str, str]] = []
        for row in rows:
            name = row.get("name")
            node_id = row.get("id")
            if not isinstance(name, str) or not name.strip():
                continue
            if not isinstance(node_id, str) or not node_id.strip():
                continue
            ingredients.append({"id": node_id.strip(), "name": name.strip()})

        logger.info("└─ KG RESULT: %d ingredients for '%s' (%.0fms)", len(ingredients), disease_name, duration_ms)

        tracer = get_tracer()
        if tracer:
            tracer.log_kg_query(
                purpose=f"get_ingredients_for_disease('{disease_name}')",
                cypher=query.strip(),
                parameters={"disease_name": clean_disease, "limit": 500},
                row_count=len(ingredients),
                result_sample=[i["name"] for i in ingredients[:10]],
                duration_ms=duration_ms,
            )

        return ingredients
    except Exception:
        return []


def get_ingredient_subgraph(ingredient_name: str) -> dict:
    """
    Traverse from Ingredient as the starting point:
    Ingredient -[CONTAINS]-> ChemicalCompound -[IS_*]-> DrugChemicalCompound <-[CONTAINS]- Drug
    Ingredient -[CURES]-> Disease
    Return structured subgraph with ingredient as the primary entity.
    """
    from app.services.pipeline_tracer import get_tracer

    driver = _get_driver()
    
    # First, find the ingredient node
    find_query = """
    MATCH (i:Ingredient)
    WHERE toLower(i.name) = toLower($ingredient_name)
    RETURN elementId(i) AS id, i.name AS name
    LIMIT 1
    """
    
    t0 = time.perf_counter()
    clean_ingredient = _clean_required_param(ingredient_name)
    if not clean_ingredient:
        _log_missing_param(
            purpose="get_ingredient_subgraph",
            query=find_query,
            param_name="ingredient_name",
            raw_value=ingredient_name,
        )
        return {
            "error": "Missing required parameter: ingredient_name",
            "Ingredient": {"id": None, "name": ingredient_name},
            "Diseases": [],
            "ChemicalCompounds": [],
            "DrugChemicalCompounds": [],
            "Drugs": [],
            "Relations": [],
        }

    try:
        with driver.session() as session:
            find_params = {"ingredient_name": clean_ingredient}
            ingredient_row = session.run(find_query, find_params).single()
            
        if not ingredient_row:
            duration_ms = (time.perf_counter() - t0) * 1000
            tracer = get_tracer()
            if tracer:
                tracer.log_kg_query(
                    purpose=f"get_ingredient_subgraph('{ingredient_name}')",
                    cypher=(find_query.strip() + "\n" + "(ingredient not found; traversal query skipped)"),
                    parameters={"ingredient_name": clean_ingredient},
                    row_count=0,
                    result_sample=None,
                    duration_ms=duration_ms,
                )
            return {
                "error": "Ingredient not found",
                "Ingredient": {"id": None, "name": ingredient_name},
                "Diseases": [],
                "ChemicalCompounds": [],
                "DrugChemicalCompounds": [],
                "Drugs": [],
                "Relations": [],
            }
        
        ingredient_info = {
            "id": ingredient_row.get("id"),
            "name": ingredient_row.get("name"),
        }
        
        # Traverse the ingredient-centered subgraph
        query = """
        MATCH (i:Ingredient)
        WHERE toLower(i.name) = toLower($ingredient_name)
        OPTIONAL MATCH (i)-[:CURES]->(d:Disease)
        OPTIONAL MATCH (i)-[:CONTAINS]->(cc:ChemicalCompound)
        OPTIONAL MATCH (cc)-[isrel:IS_IDENTICAL_TO|IS_LIKELY_EQUIVALENT_TO|IS_WEAK_MATCH_TO]->(dcc:DrugChemicalCompound)
        OPTIONAL MATCH (drug:Drug)-[:CONTAINS]->(dcc)
        RETURN elementId(i) AS ingredient_id,
               i.name AS ingredient_name,
               elementId(d) AS disease_id,
               d.name AS disease_name,
               elementId(cc) AS cc_id,
               cc.name AS cc_name,
               elementId(dcc) AS dcc_id,
               dcc.name AS dcc_name,
               elementId(drug) AS drug_id,
               drug.name AS drug_name,
               type(isrel) AS is_relation_type
        LIMIT $limit
        """
        
        with driver.session() as session:
            params = {"ingredient_name": clean_ingredient, "limit": 300}
            rows = list(session.run(query, params))
        
        diseases: dict[str, dict[str, Any]] = {}
        compounds: dict[str, dict[str, Any]] = {}
        drug_compounds: dict[str, dict[str, Any]] = {}
        drugs: dict[str, dict[str, Any]] = {}
        relations: list[dict[str, Any]] = []
        
        for row in rows:
            disease_id = row.get("disease_id")
            if disease_id:
                diseases[disease_id] = {
                    "id": disease_id,
                    "name": row.get("disease_name"),
                }
            
            cc_id = row.get("cc_id")
            if cc_id:
                compounds[cc_id] = {
                    "id": cc_id,
                    "name": row.get("cc_name"),
                }
            
            dcc_id = row.get("dcc_id")
            raw_rel_type = row.get("is_relation_type")
            match_type = _map_similarity_relation(raw_rel_type)
            if dcc_id:
                existing = drug_compounds.get(dcc_id, {"id": dcc_id, "name": row.get("dcc_name")})
                if isinstance(raw_rel_type, str) and raw_rel_type.strip():
                    existing["relation_type"] = raw_rel_type.strip()
                if match_type:
                    existing["mapping_strength"] = match_type
                drug_compounds[dcc_id] = existing
            
            drug_id = row.get("drug_id")
            if drug_id:
                drugs[drug_id] = {
                    "id": drug_id,
                    "name": row.get("drug_name"),
                }
            
            # Build relations
            if ingredient_info["id"] and disease_id:
                relations.append({
                    "from": ingredient_info["id"],
                    "to": disease_id,
                    "type": "CURES",
                })
            
            if ingredient_info["id"] and cc_id:
                relations.append({
                    "from": ingredient_info["id"],
                    "to": cc_id,
                    "type": "CONTAINS",
                })
            
            if cc_id and dcc_id and isinstance(raw_rel_type, str) and raw_rel_type.strip():
                relations.append({
                    "from": cc_id,
                    "to": dcc_id,
                    "type": raw_rel_type.strip(),
                })
            
            if drug_id and dcc_id:
                relations.append({
                    "from": drug_id,
                    "to": dcc_id,
                    "type": "CONTAINS",
                })
        
        result_payload = {
            "Ingredient": ingredient_info,
            "Diseases": list(diseases.values()),
            "ChemicalCompounds": list(compounds.values()),
            "DrugChemicalCompounds": list(drug_compounds.values()),
            "Drugs": list(drugs.values()),
            "Relations": relations,
        }

        duration_ms = (time.perf_counter() - t0) * 1000
        tracer = get_tracer()
        if tracer:
            tracer.log_kg_query(
                purpose=f"get_ingredient_subgraph('{ingredient_name}')",
                cypher=query.strip(),
                parameters={"ingredient_name": clean_ingredient, "limit": 300},
                row_count=len(rows),
                result_sample={
                    "ingredient": ingredient_info.get("name"),
                    "diseases": [d.get("name") for d in list(diseases.values())[:5]],
                    "compounds": [c.get("name") for c in list(compounds.values())[:5]],
                    "drug_compounds": [dc.get("name") for dc in list(drug_compounds.values())[:5]],
                    "drugs": [d.get("name") for d in list(drugs.values())[:5]],
                },
                duration_ms=duration_ms,
            )

        return result_payload
    except Exception as exc:
        duration_ms = (time.perf_counter() - t0) * 1000
        tracer = get_tracer()
        if tracer:
            tracer.log_kg_query(
                purpose=f"get_ingredient_subgraph('{ingredient_name}')",
                cypher=query.strip(),
                parameters={"ingredient_name": clean_ingredient, "limit": 300},
                row_count=0,
                duration_ms=duration_ms,
                error=str(exc),
            )
        return {
            "error": "Failed to query ingredient subgraph",
            "details": str(exc),
            "Ingredient": {"id": None, "name": ingredient_name},
            "Diseases": [],
            "ChemicalCompounds": [],
            "DrugChemicalCompounds": [],
            "Drugs": [],
            "Relations": [],
        }


def get_ingredient_drug_substitute_subgraph(ingredient_name: str) -> dict:
    """
    Specialized traversal for ingredient-led drug-substitute queries:
    Ingredient -> ChemicalCompound -> DrugChemicalCompound -> Drug

    Cypher chain is intentionally strict to preserve substitute discovery behavior.
    """
    from app.services.pipeline_tracer import get_tracer

    driver = _get_driver()

    query = """
    MATCH (i:Ingredient)
    WHERE toLower(i.name) = toLower($ingredient)
    OPTIONAL MATCH (i)-[:CONTAINS]->(cc:ChemicalCompound)
    OPTIONAL MATCH (cc)-[isrel:IS_IDENTICAL_TO|IS_LIKELY_EQUIVALENT_TO|IS_WEAK_MATCH_TO]->(dcc:DrugChemicalCompound)
    OPTIONAL MATCH (dcc)<-[:CONTAINS]-(d:Drug)
    RETURN i, cc, dcc, d, type(isrel) AS is_relation_type
    """

    t0 = time.perf_counter()
    clean_ingredient = _clean_required_param(ingredient_name)
    if not clean_ingredient:
        _log_missing_param(
            purpose="get_ingredient_drug_substitute_subgraph",
            query=query,
            param_name="ingredient",
            raw_value=ingredient_name,
        )
        return {
            "error": "Missing required parameter: ingredient",
            "Ingredient": {"id": None, "name": ingredient_name},
            "Diseases": [],
            "ChemicalCompounds": [],
            "DrugChemicalCompounds": [],
            "Drugs": [],
            "Relations": [],
        }

    try:
        with driver.session() as session:
            params = {"ingredient": clean_ingredient}
            rows = list(session.run(query, params))

        ingredient_info: dict[str, Any] = {"id": None, "name": ingredient_name}
        compounds: dict[str, dict[str, Any]] = {}
        drug_compounds: dict[str, dict[str, Any]] = {}
        drugs: dict[str, dict[str, Any]] = {}
        relations: list[dict[str, Any]] = []
        rel_seen: set[tuple[str, str, str]] = set()

        def _add_rel(from_id: str | None, to_id: str | None, rel_type: str | None) -> None:
            if not from_id or not to_id or not rel_type:
                return
            key = (from_id, to_id, rel_type)
            if key in rel_seen:
                return
            rel_seen.add(key)
            relations.append({"from": from_id, "to": to_id, "type": rel_type})

        for row in rows:
            i_node = row.get("i")
            if i_node is not None:
                ingredient_info = {
                    "id": i_node.element_id,
                    "name": i_node.get("name") or ingredient_name,
                }

            cc_node = row.get("cc")
            if cc_node is not None:
                cc_id = cc_node.element_id
                compounds[cc_id] = {"id": cc_id, "name": cc_node.get("name")}

            dcc_node = row.get("dcc")
            if dcc_node is not None:
                dcc_id = dcc_node.element_id
                raw_rel_type = row.get("is_relation_type")
                match_type = _map_similarity_relation(raw_rel_type)
                existing = drug_compounds.get(dcc_id, {"id": dcc_id, "name": dcc_node.get("name")})
                if isinstance(raw_rel_type, str) and raw_rel_type.strip():
                    existing["relation_type"] = raw_rel_type.strip()
                if match_type:
                    existing["mapping_strength"] = match_type
                drug_compounds[dcc_id] = existing

            d_node = row.get("d")
            if d_node is not None:
                drug_id = d_node.element_id
                drugs[drug_id] = {"id": drug_id, "name": d_node.get("name")}

            ing_id = ingredient_info.get("id")
            if cc_node is not None:
                _add_rel(ing_id, cc_node.element_id, "CONTAINS")

            if cc_node is not None and dcc_node is not None:
                raw_rel_type = row.get("is_relation_type")
                if isinstance(raw_rel_type, str) and raw_rel_type.strip():
                    _add_rel(cc_node.element_id, dcc_node.element_id, raw_rel_type.strip())

            if d_node is not None and dcc_node is not None:
                _add_rel(d_node.element_id, dcc_node.element_id, "CONTAINS")

        result_payload = {
            "Ingredient": ingredient_info,
            "Diseases": [],
            "ChemicalCompounds": list(compounds.values()),
            "DrugChemicalCompounds": list(drug_compounds.values()),
            "Drugs": list(drugs.values()),
            "Relations": relations,
        }

        duration_ms = (time.perf_counter() - t0) * 1000
        tracer = get_tracer()
        if tracer:
            tracer.log_kg_query(
                purpose=f"get_ingredient_drug_substitute_subgraph('{ingredient_name}')",
                cypher=query.strip(),
                parameters={"ingredient": clean_ingredient},
                row_count=len(rows),
                result_sample={
                    "ingredient": result_payload["Ingredient"].get("name"),
                    "compounds": [c.get("name") for c in result_payload["ChemicalCompounds"][:5]],
                    "drug_compounds": [dc.get("name") for dc in result_payload["DrugChemicalCompounds"][:5]],
                    "drugs": [d.get("name") for d in result_payload["Drugs"][:5]],
                },
                duration_ms=duration_ms,
            )

        return result_payload
    except Exception as exc:
        duration_ms = (time.perf_counter() - t0) * 1000
        tracer = get_tracer()
        if tracer:
            tracer.log_kg_query(
                purpose=f"get_ingredient_drug_substitute_subgraph('{ingredient_name}')",
                cypher=query.strip(),
                parameters={"ingredient": clean_ingredient},
                row_count=0,
                duration_ms=duration_ms,
                error=str(exc),
            )
        return {
            "error": "Failed to query ingredient substitute subgraph",
            "details": str(exc),
            "Ingredient": {"id": None, "name": ingredient_name},
            "Diseases": [],
            "ChemicalCompounds": [],
            "DrugChemicalCompounds": [],
            "Drugs": [],
            "Relations": [],
        }


def get_drug_subgraph(drug_name: str) -> dict:
    """
    Traverse from Drug as the starting point:
    Drug -[CONTAINS]-> DrugChemicalCompound <-[IS_*]- ChemicalCompound <-[CONTAINS]- Ingredient -[CURES]-> Disease
    Return structured subgraph with drug as the primary entity.
    """
    from app.services.pipeline_tracer import get_tracer

    driver = _get_driver()
    
    # First, find the drug node
    find_query = """
    MATCH (drug:Drug)
    WHERE toLower(drug.name) = toLower($drug_name)
    RETURN elementId(drug) AS id, drug.name AS name
    LIMIT 1
    """
    
    t0 = time.perf_counter()
    clean_drug = _clean_required_param(drug_name)
    if not clean_drug:
        _log_missing_param(
            purpose="get_drug_subgraph",
            query=find_query,
            param_name="drug_name",
            raw_value=drug_name,
        )
        return {
            "error": "Missing required parameter: drug_name",
            "Drug": {"id": None, "name": drug_name},
            "DrugChemicalCompounds": [],
            "ChemicalCompounds": [],
            "Ingredients": [],
            "Diseases": [],
            "Relations": [],
        }

    try:
        with driver.session() as session:
            find_params = {"drug_name": clean_drug}
            drug_row = session.run(find_query, find_params).single()
        
        if not drug_row:
            duration_ms = (time.perf_counter() - t0) * 1000
            tracer = get_tracer()
            if tracer:
                tracer.log_kg_query(
                    purpose=f"get_drug_subgraph('{drug_name}')",
                    cypher=(find_query.strip() + "\n" + "(drug not found; traversal query skipped)"),
                    parameters={"drug_name": clean_drug},
                    row_count=0,
                    result_sample=None,
                    duration_ms=duration_ms,
                )
            return {
                "error": "Drug not found",
                "Drug": {"id": None, "name": drug_name},
                "DrugChemicalCompounds": [],
                "ChemicalCompounds": [],
                "Ingredients": [],
                "Diseases": [],
                "Relations": [],
            }
        
        drug_info = {
            "id": drug_row.get("id"),
            "name": drug_row.get("name"),
        }
        
        # Traverse the drug-centered subgraph (reverse of disease→drug path)
        query = """
        MATCH (drug:Drug)
        WHERE toLower(drug.name) = toLower($drug_name)
        OPTIONAL MATCH (drug)-[:CONTAINS]->(dcc:DrugChemicalCompound)
        OPTIONAL MATCH (dcc)<-[isrel:IS_IDENTICAL_TO|IS_LIKELY_EQUIVALENT_TO|IS_WEAK_MATCH_TO]-(cc:ChemicalCompound)
        OPTIONAL MATCH (cc)<-[:CONTAINS]-(i:Ingredient)
        OPTIONAL MATCH (i)-[:CURES]->(d:Disease)
        RETURN elementId(drug) AS drug_id,
               drug.name AS drug_name,
               elementId(dcc) AS dcc_id,
               dcc.name AS dcc_name,
               elementId(cc) AS cc_id,
               cc.name AS cc_name,
               elementId(i) AS ingredient_id,
               i.name AS ingredient_name,
               elementId(d) AS disease_id,
               d.name AS disease_name,
               type(isrel) AS is_relation_type
        LIMIT $limit
        """
        
        with driver.session() as session:
            params = {"drug_name": clean_drug, "limit": 300}
            rows = list(session.run(query, params))
        
        drug_compounds: dict[str, dict[str, Any]] = {}
        compounds: dict[str, dict[str, Any]] = {}
        ingredients: dict[str, dict[str, Any]] = {}
        diseases: dict[str, dict[str, Any]] = {}
        relations: list[dict[str, Any]] = []
        
        for row in rows:
            dcc_id = row.get("dcc_id")
            raw_rel_type = row.get("is_relation_type")
            match_type = _map_similarity_relation(raw_rel_type)
            if dcc_id:
                existing = drug_compounds.get(dcc_id, {"id": dcc_id, "name": row.get("dcc_name")})
                if isinstance(raw_rel_type, str) and raw_rel_type.strip():
                    existing["relation_type"] = raw_rel_type.strip()
                if match_type:
                    existing["mapping_strength"] = match_type
                drug_compounds[dcc_id] = existing
            
            cc_id = row.get("cc_id")
            if cc_id:
                compounds[cc_id] = {
                    "id": cc_id,
                    "name": row.get("cc_name"),
                }
            
            ingredient_id = row.get("ingredient_id")
            if ingredient_id:
                ingredients[ingredient_id] = {
                    "id": ingredient_id,
                    "name": row.get("ingredient_name"),
                }
            
            disease_id = row.get("disease_id")
            if disease_id:
                diseases[disease_id] = {
                    "id": disease_id,
                    "name": row.get("disease_name"),
                }
            
            # Build relations
            if drug_info["id"] and dcc_id:
                relations.append({
                    "from": drug_info["id"],
                    "to": dcc_id,
                    "type": "CONTAINS",
                })
            
            if cc_id and dcc_id and isinstance(raw_rel_type, str) and raw_rel_type.strip():
                relations.append({
                    "from": cc_id,
                    "to": dcc_id,
                    "type": raw_rel_type.strip(),
                })
            
            if ingredient_id and cc_id:
                relations.append({
                    "from": ingredient_id,
                    "to": cc_id,
                    "type": "CONTAINS",
                })
            
            if ingredient_id and disease_id:
                relations.append({
                    "from": ingredient_id,
                    "to": disease_id,
                    "type": "CURES",
                })
        
        result_payload = {
            "Drug": drug_info,
            "DrugChemicalCompounds": list(drug_compounds.values()),
            "ChemicalCompounds": list(compounds.values()),
            "Ingredients": list(ingredients.values()),
            "Diseases": list(diseases.values()),
            "Relations": relations,
        }

        duration_ms = (time.perf_counter() - t0) * 1000
        tracer = get_tracer()
        if tracer:
            tracer.log_kg_query(
                purpose=f"get_drug_subgraph('{drug_name}')",
                cypher=query.strip(),
                parameters={"drug_name": clean_drug, "limit": 300},
                row_count=len(rows),
                result_sample={
                    "drug": drug_info.get("name"),
                    "drug_compounds": [dc.get("name") for dc in list(drug_compounds.values())[:5]],
                    "compounds": [c.get("name") for c in list(compounds.values())[:5]],
                    "ingredients": [i.get("name") for i in list(ingredients.values())[:5]],
                    "diseases": [d.get("name") for d in list(diseases.values())[:5]],
                },
                duration_ms=duration_ms,
            )

        return result_payload
    except Exception as exc:
        duration_ms = (time.perf_counter() - t0) * 1000
        tracer = get_tracer()
        if tracer:
            tracer.log_kg_query(
                purpose=f"get_drug_subgraph('{drug_name}')",
                cypher=query.strip(),
                parameters={"drug_name": clean_drug, "limit": 300},
                row_count=0,
                duration_ms=duration_ms,
                error=str(exc),
            )
        return {
            "error": "Failed to query drug subgraph",
            "details": str(exc),
            "Drug": {"id": None, "name": drug_name},
            "DrugChemicalCompounds": [],
            "ChemicalCompounds": [],
            "Ingredients": [],
            "Diseases": [],
            "Relations": [],
        }