from functools import lru_cache
from typing import Any

from neo4j import GraphDatabase

from app.config import get_settings


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

    try:
        with driver.session() as session:
            result = session.run(query, name=name)
            row = result.single()

        if not row:
            return {"error": "Disease not found", "name": name}

        return {
            "id": row.get("id"),
            "name": row.get("name"),
            "category": row.get("category"),
        }
    except Exception as exc:
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
    OPTIONAL MATCH (d)<-[:CURES]-(ingredient:Ingredient)
    OPTIONAL MATCH (ingredient)-[:CONTAINS]->(cc:ChemicalCompound)
    OPTIONAL MATCH (cc)-[isrel:IS_IDENTICAL_TO|IS_LIKELY_EQUIVALENT_TO|IS_WEAK_MATCH_TO]->(dcc:DrugChemicalCompound)
    OPTIONAL MATCH (drug:Drug)-[:CONTAINS]->(dcc)
    RETURN elementId(d) AS disease_id,
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
    LIMIT $limit
    """

    try:
        with driver.session() as session:
            rows = list(session.run(query, disease_name=disease_name, limit=1500))

        ingredients: dict[str, dict[str, Any]] = {}
        compounds: dict[str, dict[str, Any]] = {}
        drug_compounds: dict[str, dict[str, Any]] = {}
        drugs: dict[str, dict[str, Any]] = {}
        relations: list[dict[str, Any]] = []

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
            match_type = _map_similarity_relation(row.get("is_relation_type"))
            if dcc_id:
                existing = drug_compounds.get(dcc_id, {"id": dcc_id, "name": row.get("dcc_name")})
                if match_type:
                    existing["relation_type"] = match_type
                drug_compounds[dcc_id] = existing

            drug_id = row.get("drug_id")
            if drug_id:
                drugs[drug_id] = {
                    "id": drug_id,
                    "name": row.get("drug_name"),
                }

            if ingredient_id and cc_id:
                relations.append(
                    {
                        "from": ingredient_id,
                        "to": cc_id,
                        "type": "CONTAINS",
                    }
                )

            if cc_id and dcc_id and match_type:
                relations.append(
                    {
                        "from": cc_id,
                        "to": dcc_id,
                        "type": match_type,
                    }
                )

            if drug_id and dcc_id:
                relations.append(
                    {
                        "from": drug_id,
                        "to": dcc_id,
                        "type": "CONTAINS",
                    }
                )

            if ingredient_id:
                relations.append(
                    {
                        "from": ingredient_id,
                        "to": disease.get("id"),
                        "type": "CURES",
                    }
                )

        return {
            "Disease": disease,
            "Ingredients": list(ingredients.values()),
            "ChemicalCompounds": list(compounds.values()),
            "DrugChemicalCompounds": list(drug_compounds.values()),
            "Drugs": list(drugs.values()),
            "Relations": relations,
        }
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

    try:
        with driver.session() as session:
            rows = list(session.run(query, disease_name=disease_name, limit=100))

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

    try:
        with driver.session() as session:
            rows = list(session.run(query, disease_name=disease_name, limit=500))

        ingredients: list[dict[str, str]] = []
        for row in rows:
            name = row.get("name")
            node_id = row.get("id")
            if not isinstance(name, str) or not name.strip():
                continue
            if not isinstance(node_id, str) or not node_id.strip():
                continue
            ingredients.append({"id": node_id.strip(), "name": name.strip()})

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
    driver = _get_driver()
    
    # First, find the ingredient node
    find_query = """
    MATCH (i:Ingredient)
    WHERE toLower(i.name) = toLower($ingredient_name)
    RETURN elementId(i) AS id, i.name AS name
    LIMIT 1
    """
    
    try:
        with driver.session() as session:
            ingredient_row = session.run(find_query, ingredient_name=ingredient_name).single()
            
        if not ingredient_row:
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
            rows = list(session.run(query, ingredient_name=ingredient_name, limit=300))
        
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
            match_type = _map_similarity_relation(row.get("is_relation_type"))
            if dcc_id:
                existing = drug_compounds.get(dcc_id, {"id": dcc_id, "name": row.get("dcc_name")})
                if match_type:
                    existing["relation_type"] = match_type
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
            
            if cc_id and dcc_id and match_type:
                relations.append({
                    "from": cc_id,
                    "to": dcc_id,
                    "type": match_type,
                })
            
            if drug_id and dcc_id:
                relations.append({
                    "from": drug_id,
                    "to": dcc_id,
                    "type": "CONTAINS",
                })
        
        return {
            "Ingredient": ingredient_info,
            "Diseases": list(diseases.values()),
            "ChemicalCompounds": list(compounds.values()),
            "DrugChemicalCompounds": list(drug_compounds.values()),
            "Drugs": list(drugs.values()),
            "Relations": relations,
        }
    except Exception as exc:
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


def get_drug_subgraph(drug_name: str) -> dict:
    """
    Traverse from Drug as the starting point:
    Drug -[CONTAINS]-> DrugChemicalCompound <-[IS_*]- ChemicalCompound <-[CONTAINS]- Ingredient -[CURES]-> Disease
    Return structured subgraph with drug as the primary entity.
    """
    driver = _get_driver()
    
    # First, find the drug node
    find_query = """
    MATCH (drug:Drug)
    WHERE toLower(drug.name) = toLower($drug_name)
    RETURN elementId(drug) AS id, drug.name AS name
    LIMIT 1
    """
    
    try:
        with driver.session() as session:
            drug_row = session.run(find_query, drug_name=drug_name).single()
        
        if not drug_row:
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
            rows = list(session.run(query, drug_name=drug_name, limit=300))
        
        drug_compounds: dict[str, dict[str, Any]] = {}
        compounds: dict[str, dict[str, Any]] = {}
        ingredients: dict[str, dict[str, Any]] = {}
        diseases: dict[str, dict[str, Any]] = {}
        relations: list[dict[str, Any]] = []
        
        for row in rows:
            dcc_id = row.get("dcc_id")
            match_type = _map_similarity_relation(row.get("is_relation_type"))
            if dcc_id:
                existing = drug_compounds.get(dcc_id, {"id": dcc_id, "name": row.get("dcc_name")})
                if match_type:
                    existing["relation_type"] = match_type
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
            
            if cc_id and dcc_id and match_type:
                relations.append({
                    "from": cc_id,
                    "to": dcc_id,
                    "type": match_type,
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
        
        return {
            "Drug": drug_info,
            "DrugChemicalCompounds": list(drug_compounds.values()),
            "ChemicalCompounds": list(compounds.values()),
            "Ingredients": list(ingredients.values()),
            "Diseases": list(diseases.values()),
            "Relations": relations,
        }
    except Exception as exc:
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