from __future__ import annotations

import asyncio
import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable

from app.config import get_settings
from app.services import entity_service, graph_service, reasoning_builder, safety_service
from app.services.llm_service import LLMService

logger = logging.getLogger(__name__)

FALLBACK_MESSAGE = (
    "PRO-MedGraph could not fully process your request right now. "
    "Please try again with a more specific disease or treatment query."
)

_GENERAL_QUERY_PATTERN = re.compile(
    r"^(hi|hello|hey|salam|assalam\s*o\s*alaikum|thanks|thank\s*you|ok|okay|yo|sup|how are you)"
    r"(?:[\s!,.?]*)$",
    re.IGNORECASE,
)
_INGREDIENT_QUERY_PATTERN = re.compile(r"\b(ingredient|ingredients|herb|herbal|remedy|remedies)\b", re.IGNORECASE)
_DRUG_QUERY_PATTERN = re.compile(
    r"\b(drug|drugs|medicine|medicines|medication|medications|pharmaceutical|pharmaceuticals|tablet|capsule|pill)\b",
    re.IGNORECASE,
)

# Knowledge Graph schema - loaded once at module initialization
_KG_SCHEMA: dict[str, Any] | None = None


def _load_kg_schema() -> dict[str, Any]:
    """
    Load knowledge graph schema from JSON file at project root.
    Schema defines node types, relationships, and valid traversal paths.
    Used for intent-based routing and query validation.
    """
    global _KG_SCHEMA
    if _KG_SCHEMA is not None:
        return _KG_SCHEMA
    
    try:
        # Navigate from backend/app/services to project root
        project_root = Path(__file__).resolve().parents[3]
        schema_path = project_root / "knowledge_graph_schema.json"
        
        if not schema_path.exists():
            logger.warning("KG schema file not found at %s, using empty schema", schema_path)
            _KG_SCHEMA = {}
            return _KG_SCHEMA
        
        with open(schema_path, "r", encoding="utf-8") as f:
            schema_data = json.load(f)
        
        # Extract the nested FULL_KG_SCHEMA_JSON structure
        if isinstance(schema_data, list) and len(schema_data) > 0:
            _KG_SCHEMA = schema_data[0].get("FULL_KG_SCHEMA_JSON", {})
        else:
            _KG_SCHEMA = schema_data if isinstance(schema_data, dict) else {}
        
        logger.info(
            "KG schema loaded: %d node types, %d relationship types",
            len(_KG_SCHEMA.get("nodeLabels", [])),
            len(_KG_SCHEMA.get("relationshipTypes", [])),
        )
        return _KG_SCHEMA
    except Exception:
        logger.exception("Failed to load KG schema, using empty schema")
        _KG_SCHEMA = {}
        return _KG_SCHEMA


def _run_sync_from_async(coro):
    """Run coroutine safely from both sync and async execution contexts."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    # When already inside an event loop, execute coroutine in a dedicated thread.
    with ThreadPoolExecutor(max_workers=1) as executor:
        return executor.submit(lambda: asyncio.run(coro)).result()


class GraphRAGOrchestrator:
    """
    Production-ready GraphRAG orchestration service for PRO-MedGraph.
    
    Implements intent-based retrieval routing using KG schema structure from
    knowledge_graph_schema.json. Supports three primary query types:
    
    1. Disease-centric queries:
       - Traversal: Disease ← CURES ← Ingredient → CONTAINS → ChemicalCompound
                   → IS_* → DrugChemicalCompound ← CONTAINS ← Drug
       - Retriever: graph_service.get_disease_subgraph()
       
    2. Ingredient-centric queries (when no disease specified):
       - Traversal: Ingredient → CURES → Disease
                    Ingredient → CONTAINS → ChemicalCompound → IS_* → DrugChemicalCompound
       - Retriever: graph_service.get_ingredient_subgraph()
       
    3. Drug-centric queries (when no disease/ingredient specified):
       - Traversal: Drug → CONTAINS → DrugChemicalCompound ← IS_* ← ChemicalCompound
                   ← CONTAINS ← Ingredient → CURES → Disease
       - Retriever: graph_service.get_drug_subgraph()
    
    KG Schema Usage:
    - Loaded once at module initialization from knowledge_graph_schema.json
    - Defines valid node types, relationship types, and traversal paths
    - Ensures graph retrievals align with actual Neo4j structure
    - No raw Cypher or schema details exposed to LLM (structured JSON only)
    
    Pipeline Stages (preserved from original):
    1. Entity extraction → detect disease/ingredient/drug entities
    2. Intent-based graph retrieval → route to appropriate fetcher
    3. Reasoning builder → normalize subgraph to LLM-ready structure
    4. A0 generation → base answer from graph evidence
    5. Af validation → refine with safety/uncertainty controls
    6. Safety service → add medical disclaimers as needed
    
    Stages are dependency-injected to keep future replacement of entity extraction,
    graph retrieval, reasoning builder, or LLM provider straightforward.
    """

    def __init__(
        self,
        llm_service: LLMService | None = None,
        entity_extractor: Callable[[str], dict] | None = None,
        disease_subgraph_fetcher: Callable[[str], dict] | None = None,
        ingredient_subgraph_fetcher: Callable[[str], dict] | None = None,
        drug_subgraph_fetcher: Callable[[str], dict] | None = None,
        hadith_fetcher: Callable[[str], list] | None = None,
        ingredient_fetcher: Callable[[str], list] | None = None,
        reasoning_formatter: Callable[[dict], dict] | None = None,
    ) -> None:
        settings = get_settings()

        self._llm_service = llm_service or LLMService(
            api_key=settings.groq_api_key,
            model=settings.groq_model,
            base_url=settings.groq_base_url,
        )

        self._a0_model = settings.llm_answer_model or settings.groq_model
        self._af_model = settings.llm_validator_model or settings.groq_model

        # Load KG schema for intent-based routing validation
        self._kg_schema = _load_kg_schema()

        # Entity extraction and specialized subgraph fetchers for each primary entity type
        self._entity_extractor = entity_extractor or entity_service.extract_entities
        self._disease_subgraph_fetcher = disease_subgraph_fetcher or graph_service.get_disease_subgraph
        self._ingredient_subgraph_fetcher = ingredient_subgraph_fetcher or graph_service.get_ingredient_subgraph
        self._drug_subgraph_fetcher = drug_subgraph_fetcher or graph_service.get_drug_subgraph
        self._hadith_fetcher = hadith_fetcher or graph_service.get_hadith_for_disease
        self._ingredient_fetcher = ingredient_fetcher or graph_service.get_ingredients_for_disease
        self._reasoning_formatter = reasoning_formatter or reasoning_builder.build_graph_reasoning

    def process_user_query(self, query: str) -> dict:
        """
        Synchronous entrypoint required by the specification.
        Internally delegates to async pipeline for network-bound LLM stages.
        """
        return _run_sync_from_async(self.process_user_query_async(query))

    async def process_user_query_async(self, query: str) -> dict:
        """
        Full pipeline:
        1) Entity extraction
        2) Graph retrieval
        3) Reasoning formatting
        4) A0 generation
        5) Af validation/refinement
        6) Final response envelope
        """
        clean_query = (query or "").strip()
        if not clean_query:
            return self._fallback_response("Empty query")

        try:
            context = await self.process_user_query_with_context_async(clean_query)
            return context.get("output", self._fallback_response("Missing orchestrator output"))
        except Exception:
            logger.exception("Orchestrator pipeline failed")
            return self._fallback_response("Pipeline failure")

    async def process_user_query_with_context_async(self, query: str) -> dict:
        """
        Extended async pipeline result for integrations that need structured
        intermediate data (e.g., safety post-processing using reasoning).
        """
        clean_query = (query or "").strip()
        if not clean_query:
            fallback = self._fallback_response("Empty query")
            return {
                "query": clean_query,
                "entities": {"disease": None, "ingredient": None, "drug": None},
                "reasoning": {},
                "a0_answer": "",
                "af_answer": fallback["final_answer"],
                "output": fallback,
            }

        if self._is_general_query(clean_query):
            logger.info("General conversational query detected; bypassing KG pipeline")
            general_answer = await self._generate_general_answer(clean_query)
            output = {
                "final_answer": general_answer,
                "evidence_strength": "weak",
                "graph_paths_used": 0,
                "confidence_score": None,
            }
            return {
                "query": clean_query,
                "entities": {"disease": None, "ingredient": None, "drug": None},
                "reasoning": {
                    "meta": {
                        "kg_applicable": False,
                    }
                },
                "a0_answer": "",
                "af_answer": general_answer,
                "output": output,
            }

        entities = self._extract_entities(clean_query)
        
        # Intent-based routing: determine primary entity type and retrieval strategy
        # Priority: Disease > Ingredient > Drug (based on biomedical specificity)
        disease = entities.get("disease")
        ingredient = entities.get("ingredient")
        drug = entities.get("drug")
        
        # Edge case: no biomedical entities detected → fallback to general conversation
        if not disease and not ingredient and not drug:
            logger.info("No biomedical entities detected; routing to general conversational response")
            general_answer = await self._generate_general_answer(clean_query)
            output = {
                "final_answer": general_answer,
                "evidence_strength": "weak",
                "graph_paths_used": 0,
                "confidence_score": None,
            }
            return {
                "query": clean_query,
                "entities": entities,
                "reasoning": {
                    "meta": {
                        # Flag used by safety_service to avoid KG caution notes
                        # for non-graph conversational queries (e.g., "hello").
                        "kg_applicable": False,
                    }
                },
                "a0_answer": "",
                "af_answer": general_answer,
                "output": output,
            }
        
        # Retrieve subgraph using intent-based routing with primary entity
        subgraph, primary_entity_type, primary_entity_name = self._retrieve_graph_by_intent(
            disease=disease,
            ingredient=ingredient,
            drug=drug,
        )
        
        # Build reasoning from subgraph (reasoning_builder handles different entity types)
        reasoning = self._build_reasoning(
            subgraph=subgraph,
            primary_entity_type=primary_entity_type,
            primary_entity_name=primary_entity_name,
        )

        # Step-level structured logging for operational observability.
        graph_paths_used = len(reasoning.get("BiochemicalMappings", []))
        logger.info("Reasoning built: graph_paths_used=%s", graph_paths_used)

        # Edge case: disease was found in KG but subgraph returned an error
        # (e.g., disease not in graph, or graph query failed)
        subgraph_error = subgraph.get("error") if isinstance(subgraph, dict) else None
        if subgraph_error and primary_entity_type == "Disease":
            logger.warning(
                "Subgraph error for disease='%s': %s",
                primary_entity_name,
                subgraph_error,
            )
            # Still attempt LLM generation -- it may produce a useful partial answer
            # with appropriate uncertainty language, rather than a hard fallback.

        # Deterministic KG answer path: if user explicitly asks for ingredients,
        # return ingredients directly from graph reasoning instead of relying on
        # LLM generation that can fail for large contexts.
        # Note: This only applies to disease-centric queries with ingredient intent
        if self._is_ingredient_query(clean_query) and disease:
            direct_ingredient_answer = self._build_ingredient_answer(reasoning, disease)
            if direct_ingredient_answer:
                evidence_strength = self._compute_evidence_strength(reasoning)
                confidence_score = self._compute_confidence_score(evidence_strength, graph_paths_used)
                output = {
                    "final_answer": direct_ingredient_answer,
                    "evidence_strength": evidence_strength,
                    "graph_paths_used": graph_paths_used,
                    "confidence_score": confidence_score,
                }
                logger.info("Ingredient intent served by direct KG response")
                return {
                    "query": clean_query,
                    "entities": entities,
                    "reasoning": {
                        **reasoning,
                        "meta": {
                            **(reasoning.get("meta", {}) if isinstance(reasoning, dict) else {}),
                            # Deterministic KG answer already grounded in direct graph data.
                            # Keep final text clean (no inline safety note injection).
                            "deterministic_kg_answer": True,
                        },
                    },
                    "a0_answer": direct_ingredient_answer,
                    "af_answer": direct_ingredient_answer,
                    "output": output,
                }

        # Core LLM pipeline: A0 generation -> Af validation
        # For disease queries, the full path Disease->Ingredient->Compound->Drug
        # is ALWAYS included in reasoning, so A0 receives complete evidence.
        a0_answer = await self._generate_a0_answer(clean_query, reasoning)
        af_answer = await self._validate_answer(clean_query, a0_answer, reasoning)

        evidence_strength = self._compute_evidence_strength(reasoning)
        confidence_score = self._compute_confidence_score(evidence_strength, graph_paths_used)

        logger.info("Pipeline completed: evidence_strength=%s graph_paths_used=%s", evidence_strength, graph_paths_used)

        output = {
            "final_answer": af_answer,
            "evidence_strength": evidence_strength,
            "graph_paths_used": graph_paths_used,
            "confidence_score": confidence_score,
        }

        # Apply safety post-processing: caution flags, confidence recalc, notes
        safe_output = safety_service.apply_safety_checks(
            reasoning=reasoning,
            llm_output=output,
        )

        return {
            "query": clean_query,
            "entities": entities,
            "reasoning": reasoning,
            "a0_answer": a0_answer,
            "af_answer": af_answer,
            "output": safe_output,
        }


    def _extract_entities(self, query: str) -> dict:
        """
        Stage 1: Extract entities from user query using hybrid extractor.
        
        Returns dict with keys: disease, ingredient, drug (all may be None).
        Edge cases handled:
        - Extractor returns None -> convert to empty dict
        - Missing keys in result -> default to None
        - Extraction exceptions -> log and return all-None dict
        """
        try:
            entities = self._entity_extractor(query) or {}
            result = {
                "disease": entities.get("disease"),
                "ingredient": entities.get("ingredient"),
                "drug": entities.get("drug"),
            }
            logger.info("Entities extracted disease=%s ingredient=%s drug=%s", result["disease"], result["ingredient"], result["drug"])
            return result
        except Exception:
            logger.exception("Entity extraction stage failed")
            return {"disease": None, "ingredient": None, "drug": None}


    def _retrieve_graph_by_intent(
        self,
        disease: str | None,
        ingredient: str | None,
        drug: str | None,
    ) -> tuple[dict, str, str]:
        """
        Stage 2: Intent-based graph retrieval using KG schema structure.
        
        Routing logic based on detected entities (priority order):
        1. Disease → Disease-centric subgraph (Disease ← CURES ← Ingredient → ... → Drug)
        2. Ingredient (no disease) → Ingredient-centric subgraph (Ingredient → CURES → Disease, CONTAINS → ...)
        3. Drug (no disease/ingredient) → Drug-centric subgraph (Drug → CONTAINS → ... → Ingredient → Disease)
        
        Returns:
            tuple: (subgraph_dict, primary_entity_type, primary_entity_name)
        
        KG traversal paths validated against schema:
        - Disease: Disease ← CURES ← Ingredient → CONTAINS → ChemicalCompound → IS_* → DrugChemicalCompound ← CONTAINS ← Drug
        - Ingredient: Ingredient → CURES → Disease, Ingredient → CONTAINS → ChemicalCompound → IS_* → DrugChemicalCompound ← CONTAINS ← Drug
        - Drug: Drug → CONTAINS → DrugChemicalCompound ← IS_* ← ChemicalCompound ← CONTAINS ← Ingredient → CURES → Disease
        """
        try:
            # Priority 1: Disease-centric retrieval (most specific biomedical context)
            if disease:
                logger.info("Intent routing: DISEASE query for '%s'", disease)
                subgraph = self._disease_subgraph_fetcher(disease) or {}
                
                # Augment with hadith references (disease-specific religious evidence)
                hadith_refs = self._hadith_fetcher(disease) or []
                subgraph["HadithReferences"] = hadith_refs
                
                logger.info(
                    "Disease subgraph retrieved: ingredients=%d compounds=%d drug_compounds=%d drugs=%d hadith=%d",
                    len(subgraph.get("Ingredients", []) or []),
                    len(subgraph.get("ChemicalCompounds", []) or []),
                    len(subgraph.get("DrugChemicalCompounds", []) or []),
                    len(subgraph.get("Drugs", []) or []),
                    len(hadith_refs),
                )
                return subgraph, "Disease", disease
            
            # Priority 2: Ingredient-centric retrieval (when disease not specified)
            if ingredient:
                logger.info("Intent routing: INGREDIENT query for '%s' (no disease specified)", ingredient)
                subgraph = self._ingredient_subgraph_fetcher(ingredient) or {}
                
                # Note: Hadith typically linked to diseases, not ingredients directly
                # If diseases found via CURES, could aggregate hadith from those diseases
                subgraph.setdefault("HadithReferences", [])
                
                logger.info(
                    "Ingredient subgraph retrieved: diseases=%d compounds=%d drug_compounds=%d drugs=%d",
                    len(subgraph.get("Diseases", []) or []),
                    len(subgraph.get("ChemicalCompounds", []) or []),
                    len(subgraph.get("DrugChemicalCompounds", []) or []),
                    len(subgraph.get("Drugs", []) or []),
                )
                return subgraph, "Ingredient", ingredient
            
            # Priority 3: Drug-centric retrieval (least common, but valid for pharma queries)
            if drug:
                logger.info("Intent routing: DRUG query for '%s' (no disease/ingredient specified)", drug)
                subgraph = self._drug_subgraph_fetcher(drug) or {}
                
                # Hadith potentially retrievable from discovered diseases via reverse traversal
                subgraph.setdefault("HadithReferences", [])
                
                logger.info(
                    "Drug subgraph retrieved: drug_compounds=%d compounds=%d ingredients=%d diseases=%d",
                    len(subgraph.get("DrugChemicalCompounds", []) or []),
                    len(subgraph.get("ChemicalCompounds", []) or []),
                    len(subgraph.get("Ingredients", []) or []),
                    len(subgraph.get("Diseases", []) or []),
                )
                return subgraph, "Drug", drug
            
            # Edge case: should not reach here (caller validates at least one entity exists)
            logger.warning("Intent routing fallback: no valid entity for graph retrieval")
            return {
                "error": "No valid entity for graph retrieval",
                "Relations": [],
                "HadithReferences": [],
            }, "Unknown", "N/A"
        
        except Exception:
            logger.exception("Graph retrieval stage failed for entities: disease=%s ingredient=%s drug=%s", disease, ingredient, drug)
            # Return safe fallback structure based on primary entity
            primary_entity = disease or ingredient or drug or "Unknown"
            primary_type = "Disease" if disease else ("Ingredient" if ingredient else "Drug")
            
            return {
                "error": "Graph retrieval failed",
                primary_type: {"id": None, "name": primary_entity},
                "Relations": [],
                "HadithReferences": [],
            }, primary_type, primary_entity

    def _build_reasoning(
        self,
        subgraph: dict,
        primary_entity_type: str,
        primary_entity_name: str,
    ) -> dict:
        """
        Stage 3: Convert graph payload into strict reasoning structure for LLM.
        
        Handles different primary entity types (Disease, Ingredient, Drug) and ensures
        consistent reasoning structure regardless of entry point.
        """
        try:
            reasoning = self._reasoning_formatter(subgraph) or {}
            
            # Ensure minimum schema exists for downstream prompt formatting
            # Structure varies based on primary entity type from intent routing
            if primary_entity_type == "Disease":
                reasoning.setdefault("Disease", {"id": None, "name": primary_entity_name, "category": None})
                reasoning.setdefault("Ingredients", [])
            elif primary_entity_type == "Ingredient":
                reasoning.setdefault("Ingredient", {"id": None, "name": primary_entity_name})
                reasoning.setdefault("Diseases", [])
            elif primary_entity_type == "Drug":
                reasoning.setdefault("Drug", {"id": None, "name": primary_entity_name})
                reasoning.setdefault("Ingredients", [])
                reasoning.setdefault("Diseases", [])
            
            # Common fields across all entity types
            reasoning.setdefault("ChemicalCompounds", [])
            reasoning.setdefault("DrugChemicalCompounds", [])
            reasoning.setdefault("Drugs", [])
            reasoning.setdefault("HadithReferences", [])
            reasoning.setdefault("BiochemicalMappings", [])
            
            # Add metadata for downstream processing
            reasoning.setdefault("meta", {})
            reasoning["meta"]["primary_entity_type"] = primary_entity_type
            reasoning["meta"]["primary_entity_name"] = primary_entity_name
            
            logger.info(
                "Reasoning formatter completed: primary_entity_type=%s mappings=%d",
                primary_entity_type,
                len(reasoning.get("BiochemicalMappings", [])),
            )
            return reasoning
        except Exception:
            logger.exception("Reasoning formatter stage failed for %s: %s", primary_entity_type, primary_entity_name)
            
            # Return safe fallback reasoning structure
            fallback = {
                "ChemicalCompounds": [],
                "DrugChemicalCompounds": [],
                "Drugs": [],
                "HadithReferences": [],
                "BiochemicalMappings": [],
                "meta": {
                    "has_error": True,
                    "source_error": "Reasoning formatting failed",
                    "primary_entity_type": primary_entity_type,
                    "primary_entity_name": primary_entity_name,
                },
            }
            
            # Add primary entity to fallback
            if primary_entity_type == "Disease":
                fallback["Disease"] = {"id": None, "name": primary_entity_name, "category": None}
                fallback["Ingredients"] = []
            elif primary_entity_type == "Ingredient":
                fallback["Ingredient"] = {"id": None, "name": primary_entity_name}
                fallback["Diseases"] = []
            elif primary_entity_type == "Drug":
                fallback["Drug"] = {"id": None, "name": primary_entity_name}
                fallback["Ingredients"] = []
                fallback["Diseases"] = []
            
            return fallback

    async def _generate_a0_answer(self, query: str, reasoning: dict) -> str:
        """
        Stage 4 (A0): Generate a base grounded answer from structured graph reasoning.
        
        Handles different primary entity types (Disease, Ingredient, Drug) and adapts
        the generation prompt to the specific query intent and available evidence.
        """
        # Extract primary entity metadata from reasoning for context-aware generation
        meta = reasoning.get("meta", {}) if isinstance(reasoning, dict) else {}
        primary_entity_type = meta.get("primary_entity_type", "Disease")
        
        # Build entity-type-aware system prompt
        # For Disease queries: always instruct LLM to cover the FULL traversal path
        # Disease <- CURES <- Ingredient -> CONTAINS -> ChemicalCompound -> IS_* -> DrugChemicalCompound <- CONTAINS <- Drug
        if primary_entity_type == "Disease":
            system_prompt = (
                "You are PRO-MedGraph, a biomedical and faith-aligned assistant.\n"
                "Use ONLY the provided structured graph reasoning evidence.\n"
                "Do NOT hallucinate entities, studies, or claims.\n\n"
                "For this disease query, your answer MUST cover ALL of the following when evidence exists:\n"
                "1. Which traditional/natural Ingredients are linked to treating this disease (via CURES relationship)\n"
                "2. What ChemicalCompounds these ingredients contain (via CONTAINS relationship)\n"
                "3. Which modern Drug equivalents share these compounds (via IS_IDENTICAL_TO / IS_LIKELY_EQUIVALENT_TO / IS_WEAK_MATCH_TO)\n"
                "4. The biochemical mechanism connecting ingredients to drugs step-by-step\n"
                "5. Any relevant Hadith references from the evidence\n\n"
                "Label the mapping strength for each compound-drug link:\n"
                "- IDENTICAL = confirmed equivalent\n"
                "- LIKELY = probable equivalent\n"
                "- WEAK = tentative, interpret with caution\n\n"
                "If no drug equivalents exist, explicitly state that.\n"
                "If evidence is missing or weak, explicitly state uncertainty."
            )
        elif primary_entity_type == "Ingredient":
            system_prompt = (
                "You are PRO-MedGraph, a biomedical and faith-aligned assistant.\n"
                "Use ONLY the provided structured graph reasoning evidence.\n"
                "Do NOT hallucinate entities, studies, or claims.\n\n"
                "For this ingredient query, explain:\n"
                "1. Which diseases this ingredient is linked to (via CURES)\n"
                "2. What chemical compounds this ingredient contains\n"
                "3. Any modern drug equivalents with mapping strength labels\n"
                "4. Relevant Hadith references if present\n\n"
                "If evidence is missing or weak, explicitly state uncertainty."
            )
        elif primary_entity_type == "Drug":
            system_prompt = (
                "You are PRO-MedGraph, a biomedical and faith-aligned assistant.\n"
                "Use ONLY the provided structured graph reasoning evidence.\n"
                "Do NOT hallucinate entities, studies, or claims.\n\n"
                "For this drug query, explain:\n"
                "1. What chemical compounds this drug contains\n"
                "2. Which natural ingredients share these compounds (with mapping strength)\n"
                "3. What diseases those ingredients can treat\n"
                "4. Relevant Hadith references if present\n\n"
                "If evidence is missing or weak, explicitly state uncertainty."
            )
        else:
            system_prompt = (
                "You are PRO-MedGraph, a biomedical and faith-aligned assistant.\n"
                "Use ONLY the provided structured graph reasoning evidence.\n"
                "Do NOT hallucinate entities, studies, or claims.\n"
                "If evidence is missing or weak, explicitly state uncertainty."
            )

        user_prompt = (
            f"User Query:\n{query}\n\n"
            f"Structured Graph Reasoning (JSON):\n{json.dumps(reasoning, ensure_ascii=False, indent=2)}"
        )

        try:
            logger.info("Calling A0 LLM model=%s", self._a0_model)
            return await self._llm_service.generate_completion(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=0.2,
                model=self._a0_model,
            )
        except Exception:
            logger.exception("A0 generation stage failed")
            return (
                "I could not generate a complete graph-grounded draft answer. "
                "Available evidence appears limited, so please refine the query."
            )

    async def _validate_answer(self, query: str, a0_answer: str, reasoning: dict) -> str:
        """
        Stage 5 (Af): Validate and refine A0 with safety and uncertainty controls.
        """
        system_prompt = (
            "You are PRO-MedGraph Validator (Af).\n"
            "Validate the draft answer against the provided graph reasoning only.\n"
            "Add uncertainty language for WEAK links.\n"
            "Add concise medical safety disclaimers where needed.\n"
            "Ensure faith alignment and respectful Hadith framing.\n"
            "Do NOT add unsupported claims.\n"
            "Return the final user-facing answer only."
        )

        user_prompt = (
            f"User Query:\n{query}\n\n"
            f"Draft Answer (A0):\n{a0_answer}\n\n"
            f"Structured Graph Reasoning (JSON):\n{json.dumps(reasoning, ensure_ascii=False, indent=2)}"
        )

        try:
            logger.info("Calling Af LLM model=%s", self._af_model)
            return await self._llm_service.generate_completion(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=0.2,
                model=self._af_model,
            )
        except Exception:
            logger.exception("Af validation stage failed")
            return a0_answer

    async def _generate_general_answer(self, query: str) -> str:
        """
        Generate a normal conversational response when KG-specific extraction
        does not find a disease entity. This keeps UX natural for greetings and
        generic questions while reserving graph reasoning for biomedical cases.
        """
        system_prompt = (
            "You are PRO-MedGraph, a helpful assistant.\n"
            "Respond naturally and briefly to general user messages.\n"
            "If the user asks a medical question without specific disease details,\n"
            "ask a clarifying follow-up and avoid definitive medical claims."
        )
        user_prompt = f"User message:\n{query}"

        try:
            logger.info("Calling general-response LLM model=%s", self._a0_model)
            return await self._llm_service.generate_completion(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=0.3,
                model=self._a0_model,
            )
        except Exception:
            logger.exception("General response generation failed")
            return (
                "Hello! I’m PRO-MedGraph. I can help with biomedical questions. "
                "If you share a disease or treatment topic, I can provide a graph-grounded answer."
            )

    @staticmethod
    def _is_general_query(query: str) -> bool:
        """
        Detect lightweight conversational messages that should not trigger
        biomedical KG retrieval.
        """
        normalized = query.strip().lower()
        if not normalized:
            return True

        # Single short token (e.g., "hi", "hello", "thanks") is general chat.
        tokens = normalized.split()
        if len(tokens) == 1 and len(tokens[0]) <= 8:
            if _GENERAL_QUERY_PATTERN.match(normalized):
                return True

        return bool(_GENERAL_QUERY_PATTERN.match(normalized))

    @staticmethod
    def _is_ingredient_query(query: str) -> bool:
        """Detect ingredient-focused user intents for deterministic KG responses."""
        return bool(_INGREDIENT_QUERY_PATTERN.search(query or ""))

    @staticmethod
    def _is_drug_query(query: str) -> bool:
        """Detect drug/medicine-focused user intents (e.g., 'what drugs cure fever')."""
        return bool(_DRUG_QUERY_PATTERN.search(query or ""))

    def _build_ingredient_answer(self, reasoning: dict, disease_name: str) -> str | None:
        """
        Build a direct ingredient list answer from graph_service + reasoning data.
        
        Edge cases handled:
        - Missing or malformed reasoning structure
        - Empty ingredient lists
        - Duplicate ingredients from different sources
        - Non-string ingredient names
        """
        if not isinstance(reasoning, dict):
            logger.warning("Cannot build ingredient answer: reasoning is not a dict")
            return None

        ingredients = reasoning.get("Ingredients", [])
        if not isinstance(ingredients, list):
            logger.warning("Ingredients field in reasoning is not a list")
            ingredients = []

        # Prefer complete direct disease->ingredient retrieval from graph_service,
        # then merge with reasoning ingredients as a defensive fallback.
        try:
            direct_rows = self._ingredient_fetcher(disease_name)
        except Exception:
            logger.exception("Failed to fetch ingredients directly for disease=%s", disease_name)
            direct_rows = []
        
        direct_names: list[str] = []
        for item in direct_rows:
            if not isinstance(item, dict):
                continue
            name = item.get("name")
            if isinstance(name, str) and name.strip():
                direct_names.append(name.strip())

        reasoning_names: list[str] = []
        for item in ingredients:
            if not isinstance(item, dict):
                continue
            name = item.get("name")
            if isinstance(name, str) and name.strip():
                reasoning_names.append(name.strip())

        # Preserve order while removing duplicates (direct query first for completeness).
        deduped_names = list(dict.fromkeys([*direct_names, *reasoning_names]))
        if not deduped_names:
            logger.info("No ingredients found for disease=%s", disease_name)
            return None

        # Limit to top 25 ingredients to avoid overwhelming response
        bullet_lines = "\n".join(f"{index + 1}. {name}" for index, name in enumerate(deduped_names[:25]))
        
        suffix = "" if len(deduped_names) <= 25 else f"\n\n(Showing top 25 of {len(deduped_names)} ingredients)"
        
        return (
            f"Based on the knowledge graph, the ingredients associated with {disease_name} are:\n\n"
            f"{bullet_lines}{suffix}"
        )

    @staticmethod
    def _compute_evidence_strength(reasoning: dict) -> str:
        """
        Convert biochemical mapping strengths into a coarse overall confidence band.
        """
        mappings = reasoning.get("BiochemicalMappings", []) if isinstance(reasoning, dict) else []
        if not isinstance(mappings, list) or not mappings:
            return "weak"

        strengths = [
            (item.get("mapping_strength") if isinstance(item, dict) else None)
            for item in mappings
        ]

        strong_count = sum(1 for value in strengths if value == "IDENTICAL")
        moderate_count = sum(1 for value in strengths if value == "LIKELY")

        if strong_count >= 1:
            return "strong"
        if moderate_count >= 1:
            return "moderate"
        return "weak"

    @staticmethod
    def _compute_confidence_score(evidence_strength: str, graph_paths_used: int) -> float | None:
        """Optional normalized confidence score for downstream ranking/UI display."""
        if graph_paths_used <= 0:
            return None

        strength_base = {
            "strong": 0.85,
            "moderate": 0.65,
            "weak": 0.45,
        }.get(evidence_strength, 0.4)

        # Modest path-based boost, capped for stability.
        boost = min(0.1, graph_paths_used * 0.01)
        return round(min(0.99, strength_base + boost), 2)

    @staticmethod
    def _fallback_response(reason: str) -> dict:
        """Safe fallback response that does not expose internals or raw graph/Cypher outputs."""
        logger.warning("Returning fallback response reason=%s", reason)
        return {
            "final_answer": FALLBACK_MESSAGE,
            "evidence_strength": "weak",
            "graph_paths_used": 0,
            "confidence_score": None,
        }


_DEFAULT_ORCHESTRATOR: GraphRAGOrchestrator | None = None


def _get_default_orchestrator() -> GraphRAGOrchestrator:
    """Lazy singleton accessor for endpoint-level direct function usage."""
    global _DEFAULT_ORCHESTRATOR
    if _DEFAULT_ORCHESTRATOR is None:
        _DEFAULT_ORCHESTRATOR = GraphRAGOrchestrator()
    return _DEFAULT_ORCHESTRATOR


def process_user_query(query: str) -> dict:
    """
    Main required function for direct orchestrator pipeline execution.
    Suitable for direct integration with /api/chat handlers.
    """
    return _get_default_orchestrator().process_user_query(query)


async def process_user_query_async(query: str) -> dict:
    """Async variant for native async endpoint flows."""
    return await _get_default_orchestrator().process_user_query_async(query)


async def process_user_query_with_context_async(query: str) -> dict:
    """Async variant returning intermediate reasoning context for post-processing layers."""
    return await _get_default_orchestrator().process_user_query_with_context_async(query)


__all__ = [
    "GraphRAGOrchestrator",
    "process_user_query",
    "process_user_query_async",
    "process_user_query_with_context_async",
]