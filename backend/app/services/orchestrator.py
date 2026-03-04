"""
PRO-MedGraph GraphRAG Orchestrator — Research-Grade Pipeline (v2).

Implements the full 13-stage pipeline:

    1.  Intent Classification   – model-driven query intent labeling
    2.  Entity Extraction       – LLM + fuzzy hybrid
    3.  Rationale Plan          – RAG2-style LLM reasoning plan
    4.  Intent Routing          – Disease > Ingredient > Drug priority
    5.  Graph Retrieval         – KG traversal via Cypher
    6.  Multi-Hop Discovery     – optional 2-hop exploratory search
    7.  Reasoning Builder       – normalise subgraph for LLM
    8.  Causal Reasoner         – score & rank biochemical paths
    9.  Dosage Validator        – ingredient / drug dosage comparison
   10.  A0 Generation           – structured chain-of-thought answer draft
   11.  Af Validation           – safety / uncertainty refinement
   12.  Faith Alignment Scorer  – Hadith + science alignment
   13.  Safety Scorer           – caution flags, confidence recalc
   14.  Evaluation Module       – white-box & optional black-box metrics
   15.  Experiment Logger       – persist structured log to disk

All stages are dependency-injected.  The pipeline is backward-compatible:
the existing API contract (final_answer, evidence_strength, graph_paths_used,
confidence_score) is preserved, with the new reasoning_trace added as an
optional extension.
"""

from __future__ import annotations

import asyncio
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable

from app.config import get_settings
from app.services import (
    causal_reasoner,
    entity_service,
    evaluation_framework,
    experiment_logger,
    faith_alignment_service,
    graph_service,
    multi_hop_discovery,
    reasoning_builder,
    safety_service,
)
from app.services.dosage_validator import validate_dosage
from app.services.llm_service import LLMService
from app.services.rationale_planner import generate_rationale_plan, filter_subgraph_by_plan

logger = logging.getLogger(__name__)

FALLBACK_MESSAGE = (
    "PRO-MedGraph could not fully process your request right now. "
    "Please try again with a more specific disease or treatment query."
)

# ── KG Schema ──────────────────────────────────────────────────────────────────

_KG_SCHEMA: dict[str, Any] | None = None


def _load_kg_schema() -> dict[str, Any]:
    global _KG_SCHEMA
    if _KG_SCHEMA is not None:
        return _KG_SCHEMA

    try:
        project_root = Path(__file__).resolve().parents[3]
        schema_path = project_root / "knowledge_graph_schema.json"

        if not schema_path.exists():
            logger.warning("KG schema file not found at %s, using empty schema", schema_path)
            _KG_SCHEMA = {}
            return _KG_SCHEMA

        with open(schema_path, "r", encoding="utf-8") as f:
            schema_data = json.load(f)

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
        logger.exception("Failed to load KG schema")
        _KG_SCHEMA = {}
        return _KG_SCHEMA


def _run_sync_from_async(coro):
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    with ThreadPoolExecutor(max_workers=1) as executor:
        return executor.submit(lambda: asyncio.run(coro)).result()


# ═══════════════════════════════════════════════════════════════════════════════
#  Orchestrator
# ═══════════════════════════════════════════════════════════════════════════════


class GraphRAGOrchestrator:
    """
    Research-grade GraphRAG orchestration for PRO-MedGraph.

    Full pipeline (13+ stages):
        General Check -> Entity Extraction -> Rationale Plan -> Intent Routing
        -> Graph Retrieval -> Multi-Hop Discovery -> Reasoning Builder
        -> Causal Reasoner -> Dosage Validator -> A0 -> Af
        -> Faith Alignment -> Safety -> Evaluation -> Logging -> Response
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
        enable_evaluation: bool = False,
        enable_black_box_eval: bool = False,
    ) -> None:
        settings = get_settings()

        self._llm_service = llm_service or LLMService(
            api_key=settings.groq_api_key,
            model=settings.groq_model,
            base_url=settings.groq_base_url,
        )

        self._a0_model = settings.llm_answer_model or settings.groq_model
        self._af_model = settings.llm_validator_model or settings.groq_model
        self._eval_model = settings.llm_judge_model or settings.groq_model
        self._intent_model = settings.model_intent or settings.groq_model
        self._coref_model = settings.model_intent or settings.groq_model  # lightweight rewrite

        self._kg_schema = _load_kg_schema()

        self._entity_extractor = entity_extractor or entity_service.extract_entities
        self._disease_subgraph_fetcher = disease_subgraph_fetcher or graph_service.get_disease_subgraph
        self._ingredient_subgraph_fetcher = ingredient_subgraph_fetcher or graph_service.get_ingredient_subgraph
        self._drug_subgraph_fetcher = drug_subgraph_fetcher or graph_service.get_drug_subgraph
        self._hadith_fetcher = hadith_fetcher or graph_service.get_hadith_for_disease
        self._ingredient_fetcher = ingredient_fetcher or graph_service.get_ingredients_for_disease
        self._reasoning_formatter = reasoning_formatter or reasoning_builder.build_graph_reasoning

        self._enable_evaluation = enable_evaluation or settings.enable_judge_scoring
        self._enable_black_box_eval = enable_black_box_eval

    # ── Public entry points ────────────────────────────────────────────────

    def process_user_query(self, query: str) -> dict:
        return _run_sync_from_async(self.process_user_query_async(query))

    async def process_user_query_async(self, query: str) -> dict:
        clean_query = (query or "").strip()
        if not clean_query:
            return self._fallback_response("Empty query")
        try:
            context = await self.process_user_query_with_context_async(clean_query)
            return context.get("output", self._fallback_response("Missing orchestrator output"))
        except Exception:
            logger.exception("Orchestrator pipeline failed")
            return self._fallback_response("Pipeline failure")

    async def process_user_query_with_context_async(
        self, query: str, *, history: list[dict] | None = None,
    ) -> dict:
        """Extended 13-stage pipeline returning full context + reasoning trace."""
        clean_query = (query or "").strip()
        original_query = clean_query
        pipeline_stages: list[str] = []

        if not clean_query:
            fallback = self._fallback_response("Empty query")
            return self._wrap_context(clean_query, {}, {}, "", fallback["final_answer"], fallback, pipeline_stages)

        # Static routing removed: intent and response shaping are model-driven.
        # Execution order:
        #   co-reference rewrite -> intent model -> entity extraction -> KG retrieval
        #   -> reasoning/causal/dosage -> A0 -> Af -> faith/safety -> return

        # ── Stage 0: Co-reference resolution via conversation history ──────
        if history:
            resolved = await self._resolve_coreferences(clean_query, history)
            if resolved and resolved != clean_query:
                logger.info("Query rewritten: '%s' -> '%s'", clean_query, resolved)
                clean_query = resolved
                pipeline_stages.append("coreference_resolution")

        # ── Stage 1: Intent Classification (model-driven) ─────────────────
        pipeline_stages.append("intent_model_classification")
        followup_intent = await self._classify_followup_intent(original_query, history or [])

        if followup_intent == "general":
            logger.info("General intent detected by model; using conversational response")
            general_answer = await self._generate_general_answer(clean_query, history=history)
            output = {
                "final_answer": general_answer,
                "evidence_strength": "weak",
                "graph_paths_used": 0,
                "confidence_score": None,
            }
            pipeline_stages.append("general_response_model")
            return self._wrap_context(
                clean_query,
                {"disease": None, "ingredient": None, "drug": None},
                {"meta": {"kg_applicable": False}},
                "", general_answer, output, pipeline_stages,
                kg_applicable=False,
            )

        # ── Stage 2: Entity Extraction ─────────────────────────────────────
        pipeline_stages.append("entity_extraction")
        entities = self._extract_entities(clean_query)

        disease = entities.get("disease")
        ingredient = entities.get("ingredient")
        drug = entities.get("drug")
        traversal_intent = self._resolve_traversal_intent(followup_intent, entities)

        # ── Stage 3 + 4: Rationale Plan & Graph Retrieval (PARALLEL) ─────
        pipeline_stages.append("rationale_plan")
        pipeline_stages.append("intent_routing")

        async def _graph_retrieval_async():
            """Run synchronous graph retrieval in a thread to enable parallelism."""
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(
                None,
                self._retrieve_graph_by_intent,
                disease,
                ingredient,
                drug,
                traversal_intent,
                traversal_intent == "cure" or followup_intent == "hadith",
            )

        rationale_task = generate_rationale_plan(
            query=clean_query,
            entities=entities,
            llm_service=self._llm_service,
            model=self._a0_model,
        )

        rationale_result, (subgraph, primary_entity_type, primary_entity_name) = (
            await asyncio.gather(rationale_task, _graph_retrieval_async())
        )

        # ── Stage 5: Graph Retrieval (already done by intent routing) ──────
        pipeline_stages.append("graph_retrieval")

        # Apply rationale-based filtering
        subgraph = filter_subgraph_by_plan(
            subgraph,
            rationale_result.get("relevant_node_types", []),
        )

        # ── Stage 6: Multi-Hop Discovery (if needed) ──────────────────────
        multi_hop_data = None
        reasoning = self._build_reasoning(subgraph, primary_entity_type, primary_entity_name)

        if multi_hop_discovery.should_activate_discovery(reasoning) and disease:
            pipeline_stages.append("multi_hop_discovery")
            multi_hop_data = multi_hop_discovery.discover_indirect_compound_links(disease)
            reasoning = multi_hop_discovery.merge_discoveries_into_reasoning(reasoning, multi_hop_data)

        # ── Stage 7: Reasoning Builder (already called above) ──────────────
        pipeline_stages.append("reasoning_builder")
        graph_paths_used = len(reasoning.get("BiochemicalMappings", []))
        logger.info("Reasoning built: graph_paths_used=%s", graph_paths_used)

        # Handle subgraph errors
        subgraph_error = subgraph.get("error") if isinstance(subgraph, dict) else None
        if subgraph_error and primary_entity_type == "Disease":
            logger.warning("Subgraph error for disease='%s': %s", primary_entity_name, subgraph_error)

        # ── Stage 8: Causal Reasoner ───────────────────────────────────────
        pipeline_stages.append("causal_reasoner")
        causal_analysis = causal_reasoner.run_causal_analysis(reasoning)

        # ── Stage 9: Dosage Validator ──────────────────────────────────────
        pipeline_stages.append("dosage_validator")
        dosage_result = validate_dosage(
            reasoning=reasoning,
            causal_paths=causal_analysis.get("causal_paths"),
        )

        # ── Stage 10: A0 Generation (structured chain-of-thought) ─────────
        # Traversal-intent aware subgraph passed to A0/Af
        pipeline_stages.append("a0_generation")
        a0_answer = await self._generate_a0_answer(
            clean_query, reasoning,
            causal_analysis=causal_analysis,
            dosage_validation=dosage_result.to_dict(),
            followup_intent=followup_intent,
            traversal_intent=traversal_intent,
        )

        # ── Stage 11: Af Validation ───────────────────────────────────────
        pipeline_stages.append("af_validation")
        af_answer = await self._validate_answer(
            clean_query,
            a0_answer,
            reasoning,
            followup_intent=followup_intent,
            traversal_intent=traversal_intent,
        )

        # ── Stage 12: Faith Alignment ──────────────────────────────────────
        pipeline_stages.append("faith_alignment")
        faith_result = faith_alignment_service.score_faith_alignment(
            reasoning=reasoning,
            answer_text=af_answer,
        )

        # ── Stage 13: Safety Scorer ────────────────────────────────────────
        pipeline_stages.append("safety_scorer")
        evidence_strength = self._compute_evidence_strength(reasoning)

        # Enhanced confidence with faith alignment
        base_confidence = self._compute_confidence_score(evidence_strength, graph_paths_used)
        faith_boost = faith_result.faith_alignment_score * 0.05 if faith_result.faith_alignment_score else 0
        causal_summary = causal_analysis.get("causal_ranking", {})
        causal_boost = causal_summary.get("avg_causal_score", 0) * 0.05 if causal_summary else 0
        enhanced_confidence = (
            round(min(0.99, base_confidence + faith_boost + causal_boost), 2)
            if base_confidence is not None else None
        )

        output = {
            "final_answer": af_answer,
            "evidence_strength": evidence_strength,
            "graph_paths_used": graph_paths_used,
            "confidence_score": enhanced_confidence,
        }

        safe_output = safety_service.apply_safety_checks(reasoning=reasoning, llm_output=output)

        # ── Stage 14: Evaluation (optional) ────────────────────────────────
        eval_result = None
        if self._enable_evaluation:
            pipeline_stages.append("evaluation")
            eval_result = await evaluation_framework.run_evaluation(
                query=clean_query,
                answer=af_answer,
                reasoning=reasoning,
                causal_paths=causal_analysis.get("causal_paths", []),
                llm_service=self._llm_service if self._enable_black_box_eval else None,
                model=self._eval_model if self._enable_black_box_eval else None,
                enable_black_box=self._enable_black_box_eval,
            )
            evaluation_framework.save_evaluation(eval_result)

        # ── Stage 15: Experiment Logger ────────────────────────────────────
        pipeline_stages.append("experiment_logger")

        # Build reasoning trace
        confidence_breakdown = {
            "mapping_strength": self._get_avg_mapping_strength(reasoning),
            "path_coverage": causal_summary.get("path_coverage_score") if isinstance(causal_summary, dict) else None,
            "hadith_presence": faith_result.hadith_score,
            "faith_alignment": faith_result.faith_alignment_score,
            "causal_avg": causal_summary.get("avg_causal_score") if isinstance(causal_summary, dict) else None,
            "dosage_alignment": dosage_result.overall_alignment_score,
        }

        # Slim dosage_validation for the reasoning trace (clients don't need
        # dozens of identical "no data" comparison objects).
        dosage_dict = dosage_result.to_dict()
        slim_dosage = {
            "overall_alignment_score": dosage_dict.get("overall_alignment_score"),
            "has_dosage_data": dosage_dict.get("has_dosage_data"),
            "comparison_count": len(dosage_dict.get("comparisons", [])),
        }

        reasoning_trace = {
            "entity_detected": entities,
            "rationale_plan": rationale_result.get("rationale_plan"),
            "retrieved_paths": causal_analysis.get("causal_paths", [])[:10],
            "causal_ranking": causal_analysis.get("causal_paths", [])[:5],
            "causal_summary": causal_summary,
            "dosage_validation": slim_dosage,
            "faith_alignment_notes": faith_result.faith_alignment_notes,
            "faith_alignment_score": faith_result.faith_alignment_score,
            "confidence_breakdown": confidence_breakdown,
            "multi_hop_activated": reasoning.get("meta", {}).get("multi_hop_activated", False),
            "evaluation_metrics": eval_result.to_dict() if eval_result else None,
            "pipeline_stages": pipeline_stages,
        }

        experiment_logger.log_experiment(
            query=clean_query,
            entities=entities,
            reasoning=reasoning,
            rationale_plan=rationale_result,
            causal_analysis=causal_analysis,
            dosage_validation=dosage_result.to_dict(),
            faith_alignment=faith_result.to_dict(),
            a0_answer=a0_answer,
            af_answer=af_answer,
            final_answer=safe_output.get("final_answer", af_answer),
            confidence_score=safe_output.get("confidence_score"),
            evidence_strength=safe_output.get("evidence_strength", evidence_strength),
            graph_paths_used=graph_paths_used,
            evaluation_metrics=eval_result.to_dict() if eval_result else None,
            safety=safe_output.get("safety"),
            reasoning_trace=reasoning_trace,
            pipeline_metadata={"stages": pipeline_stages},
        )

        # Attach reasoning trace to output
        safe_output["reasoning_trace"] = reasoning_trace

        logger.info(
            "Pipeline completed: stages=%d evidence=%s paths=%d confidence=%s faith=%.3f",
            len(pipeline_stages), evidence_strength, graph_paths_used,
            safe_output.get("confidence_score"), faith_result.faith_alignment_score,
        )

        return {
            "query": clean_query,
            "entities": entities,
            "reasoning": reasoning,
            "rationale_plan": rationale_result,
            "causal_analysis": causal_analysis,
            "dosage_validation": dosage_result.to_dict(),
            "faith_alignment": faith_result.to_dict(),
            "a0_answer": a0_answer,
            "af_answer": af_answer,
            "output": safe_output,
            "reasoning_trace": reasoning_trace,
        }

    # ── Internal helpers ───────────────────────────────────────────────────

    @staticmethod
    def _trim_reasoning_for_llm(reasoning: dict, *, max_ingredients: int = 8,
                                 max_compounds: int = 15, max_drugs: int = 10,
                                 max_mappings: int = 10) -> dict:
        """Return a slimmed copy of reasoning suitable for LLM prompts.

        The full reasoning dict can contain hundreds of compounds and drugs
        (especially for diseases with many ingredients).  Sending 50 KB+ of
        JSON to the LLM inflates prompt tokens and slows completion.  This
        helper caps the largest lists while keeping all scalar fields and
        metadata intact so the LLM still has good coverage.
        """
        trimmed = dict(reasoning)  # shallow copy

        for key, limit in [
            ("Ingredients", max_ingredients),
            ("ChemicalCompounds", max_compounds),
            ("DrugChemicalCompounds", max_compounds),
            ("Drugs", max_drugs),
            ("BiochemicalMappings", max_mappings),
            ("HadithReferences", 5),
            ("Relations", max_mappings * 3),
        ]:
            items = trimmed.get(key)
            if isinstance(items, list) and len(items) > limit:
                trimmed[key] = items[:limit]

        return trimmed

    def _wrap_context(
        self,
        query: str,
        entities: dict,
        reasoning: dict,
        a0_answer: str,
        af_answer: str,
        output: dict,
        pipeline_stages: list[str],
        kg_applicable: bool = True,
        rationale_plan: dict | None = None,
    ) -> dict:
        """Wrap pipeline output in a standardised context envelope."""
        if not kg_applicable:
            reasoning.setdefault("meta", {})
            reasoning["meta"]["kg_applicable"] = False

        return {
            "query": query,
            "entities": entities,
            "reasoning": reasoning,
            "rationale_plan": rationale_plan,
            "a0_answer": a0_answer,
            "af_answer": af_answer,
            "output": output,
            "reasoning_trace": {
                "entity_detected": entities,
                "pipeline_stages": pipeline_stages,
            },
        }

    # ── Stage implementations ──────────────────────────────────────────────

    def _schema_has_label(self, label: str) -> bool:
        labels = self._kg_schema.get("nodeLabels", []) if isinstance(self._kg_schema, dict) else []
        if not labels:
            return True
        return label in labels

    def _schema_has_relationship(self, relationship: str) -> bool:
        rels = self._kg_schema.get("relationshipTypes", []) if isinstance(self._kg_schema, dict) else []
        if not rels:
            return True
        return relationship in rels

    def _schema_has_edge(self, from_label: str, to_label: str) -> bool:
        structure = self._kg_schema.get("relationshipStructure", []) if isinstance(self._kg_schema, dict) else []
        if not structure:
            return True
        for edge in structure:
            if not isinstance(edge, dict):
                continue
            from_nodes = edge.get("from") or []
            to_nodes = edge.get("to") or []
            if from_label in from_nodes and to_label in to_nodes:
                return True
        return False

    def _schema_relationships_between(self, from_label: str, to_label: str) -> set[str]:
        structure = self._kg_schema.get("relationshipStructure", []) if isinstance(self._kg_schema, dict) else []
        if not structure:
            return set()
        result: set[str] = set()
        for edge in structure:
            if not isinstance(edge, dict):
                continue
            rel = edge.get("relationship")
            from_nodes = edge.get("from") or []
            to_nodes = edge.get("to") or []
            if (
                isinstance(rel, str)
                and rel.strip()
                and from_label in from_nodes
                and to_label in to_nodes
                and self._schema_has_relationship(rel)
            ):
                result.add(rel.strip())
        return result

    @staticmethod
    def _collect_node_ids(payload: Any) -> set[str]:
        ids: set[str] = set()
        if isinstance(payload, dict):
            node_id = payload.get("id")
            if isinstance(node_id, str) and node_id.strip():
                ids.add(node_id.strip())
            return ids
        if isinstance(payload, list):
            for item in payload:
                if not isinstance(item, dict):
                    continue
                node_id = item.get("id")
                if isinstance(node_id, str) and node_id.strip():
                    ids.add(node_id.strip())
        return ids

    def _resolve_traversal_intent(self, followup_intent: str, entities: dict[str, Any]) -> str:
        label = (followup_intent or "").strip().lower()
        drug_entity = entities.get("drug") if isinstance(entities, dict) else None

        if label in {"cure", "hadith", "ingredient"}:
            return "cure"
        if label in {"reasoning", "technical"}:
            return "reasoning"
        if label == "drug" or (isinstance(drug_entity, str) and drug_entity.strip()):
            return "drug"
        if label in {"full", "auto", ""}:
            return "full"
        return "full"

    def _prune_subgraph_by_traversal_intent(
        self,
        subgraph: dict[str, Any],
        traversal_intent: str,
        include_hadith: bool,
    ) -> dict[str, Any]:
        # Intent-driven traversal depth control enabled.
        if not isinstance(subgraph, dict):
            return {"error": "Invalid subgraph payload", "Relations": [], "HadithReferences": []}

        pruned = dict(subgraph)
        relations = pruned.get("Relations")
        relation_rows = relations if isinstance(relations, list) else []

        disease_ids = self._collect_node_ids(pruned.get("Disease")) | self._collect_node_ids(pruned.get("Diseases"))
        ingredient_ids = self._collect_node_ids(pruned.get("Ingredient")) | self._collect_node_ids(pruned.get("Ingredients"))
        compound_ids = self._collect_node_ids(pruned.get("ChemicalCompounds"))
        drug_compound_ids = self._collect_node_ids(pruned.get("DrugChemicalCompounds"))
        drug_ids = self._collect_node_ids(pruned.get("Drug")) | self._collect_node_ids(pruned.get("Drugs"))
        book_ids = self._collect_node_ids(pruned.get("Book")) | self._collect_node_ids(pruned.get("Books"))

        ingredient_disease_rels = self._schema_relationships_between("Ingredient", "Disease") | self._schema_relationships_between("Disease", "Ingredient")
        ingredient_compound_rels = self._schema_relationships_between("Ingredient", "ChemicalCompound") | self._schema_relationships_between("ChemicalCompound", "Ingredient")
        compound_dcc_rels = self._schema_relationships_between("ChemicalCompound", "DrugChemicalCompound") | self._schema_relationships_between("DrugChemicalCompound", "ChemicalCompound")
        drug_dcc_rels = self._schema_relationships_between("Drug", "DrugChemicalCompound") | self._schema_relationships_between("DrugChemicalCompound", "Drug")
        disease_hadith_rels = self._schema_relationships_between("Disease", "Hadith") | self._schema_relationships_between("Hadith", "Disease")
        hadith_reference_rels = self._schema_relationships_between("Reference", "Hadith") | self._schema_relationships_between("Hadith", "Reference")
        drug_book_rels = self._schema_relationships_between("Drug", "Book") | self._schema_relationships_between("Book", "Drug")

        allowed_rel_types: set[str] = set()
        allowed_endpoints: set[str] = set()

        if traversal_intent == "cure":
            allowed_rel_types |= ingredient_disease_rels
            allowed_rel_types |= disease_hadith_rels
            allowed_rel_types |= hadith_reference_rels
            allowed_endpoints |= disease_ids | ingredient_ids

            pruned["ChemicalCompounds"] = []
            pruned["DrugChemicalCompounds"] = []
            pruned["Drugs"] = []
            pruned.setdefault("Books", [])
            if not include_hadith:
                pruned["HadithReferences"] = []

        elif traversal_intent == "reasoning":
            allowed_rel_types |= ingredient_compound_rels
            allowed_endpoints |= ingredient_ids | compound_ids

            pruned["DrugChemicalCompounds"] = []
            pruned["Drugs"] = []
            pruned.setdefault("Books", [])
            pruned["HadithReferences"] = []

        elif traversal_intent == "drug":
            allowed_rel_types |= ingredient_compound_rels
            allowed_rel_types |= compound_dcc_rels
            allowed_rel_types |= drug_dcc_rels
            allowed_rel_types |= drug_book_rels
            allowed_endpoints |= ingredient_ids | compound_ids | drug_compound_ids | drug_ids | book_ids

            pruned.setdefault("Books", pruned.get("Book") if isinstance(pruned.get("Book"), list) else [])
            if not include_hadith:
                pruned["HadithReferences"] = []

        else:
            return pruned

        if not allowed_rel_types:
            logger.warning("Traversal intent '%s' has no schema-valid relationships; applying defensive empty relations", traversal_intent)
            pruned["Relations"] = []
            return pruned

        filtered_relations: list[dict[str, Any]] = []
        for relation in relation_rows:
            if not isinstance(relation, dict):
                continue
            rel_type = relation.get("type")
            source = relation.get("from")
            target = relation.get("to")
            if not isinstance(rel_type, str) or not rel_type.strip():
                continue
            if rel_type not in allowed_rel_types:
                continue
            if allowed_endpoints:
                if not isinstance(source, str) or not isinstance(target, str):
                    continue
                if source not in allowed_endpoints or target not in allowed_endpoints:
                    continue
            filtered_relations.append(dict(relation))

        pruned["Relations"] = filtered_relations
        return pruned

    def _extract_entities(self, query: str) -> dict:
        """Stage 2: Hybrid LLM + fuzzy entity extraction."""
        try:
            entities = self._entity_extractor(query) or {}
            result = {
                "disease": entities.get("disease"),
                "ingredient": entities.get("ingredient"),
                "drug": entities.get("drug"),
            }
            logger.info(
                "Entities extracted disease=%s ingredient=%s drug=%s",
                result["disease"], result["ingredient"], result["drug"],
            )
            return result
        except Exception:
            logger.exception("Entity extraction stage failed")
            return {"disease": None, "ingredient": None, "drug": None}

    def _retrieve_graph_by_intent(
        self,
        disease: str | None,
        ingredient: str | None,
        drug: str | None,
        traversal_intent: str = "full",
        include_hadith: bool = False,
    ) -> tuple[dict, str, str]:
        """Stage 4+5: Intent-based routing + KG retrieval."""
        try:
            # Schema-driven KG retrieval enforced.
            if disease:
                if not self._schema_has_label("Disease"):
                    return (
                        {"error": "Disease label not available in schema", "Relations": [], "HadithReferences": []},
                        "Disease",
                        disease,
                    )
                logger.info("Intent routing: DISEASE query for '%s'", disease)
                subgraph = self._disease_subgraph_fetcher(disease) or {}
                should_fetch_hadith = (
                    include_hadith
                    and traversal_intent in {"cure", "full"}
                    and self._schema_has_label("Reference")
                    and self._schema_has_label("Hadith")
                    and self._schema_has_edge("Disease", "Hadith")
                )
                hadith_refs = self._hadith_fetcher(disease) if should_fetch_hadith else []
                subgraph["HadithReferences"] = hadith_refs
                subgraph = self._prune_subgraph_by_traversal_intent(subgraph, traversal_intent, include_hadith)
                logger.info(
                    "Disease subgraph: ingredients=%d compounds=%d drugs=%d hadith=%d",
                    len(subgraph.get("Ingredients", []) or []),
                    len(subgraph.get("ChemicalCompounds", []) or []),
                    len(subgraph.get("Drugs", []) or []),
                    len(hadith_refs),
                )
                return subgraph, "Disease", disease

            if ingredient:
                if not self._schema_has_label("Ingredient"):
                    return (
                        {"error": "Ingredient label not available in schema", "Relations": [], "HadithReferences": []},
                        "Ingredient",
                        ingredient,
                    )
                logger.info("Intent routing: INGREDIENT query for '%s'", ingredient)
                subgraph = self._ingredient_subgraph_fetcher(ingredient) or {}
                subgraph.setdefault("HadithReferences", [])
                subgraph = self._prune_subgraph_by_traversal_intent(subgraph, traversal_intent, include_hadith)
                return subgraph, "Ingredient", ingredient

            if drug:
                if not self._schema_has_label("Drug"):
                    return (
                        {"error": "Drug label not available in schema", "Relations": [], "HadithReferences": []},
                        "Drug",
                        drug,
                    )
                logger.info("Intent routing: DRUG query for '%s'", drug)
                subgraph = self._drug_subgraph_fetcher(drug) or {}
                subgraph.setdefault("HadithReferences", [])
                subgraph = self._prune_subgraph_by_traversal_intent(subgraph, traversal_intent, include_hadith)
                return subgraph, "Drug", drug

            return {"error": "No entity", "Relations": [], "HadithReferences": []}, "Unknown", "N/A"

        except Exception:
            logger.exception("Graph retrieval failed")
            primary = disease or ingredient or drug or "Unknown"
            ptype = "Disease" if disease else ("Ingredient" if ingredient else "Drug")
            return {"error": "Graph retrieval failed", "Relations": [], "HadithReferences": []}, ptype, primary

    def _build_reasoning(self, subgraph: dict, primary_entity_type: str, primary_entity_name: str) -> dict:
        """Stage 7: Convert subgraph to structured reasoning."""
        try:
            # Schema-driven KG retrieval enforced.
            reasoning = self._reasoning_formatter(subgraph) or {}

            if primary_entity_type == "Disease" and self._schema_has_label("Disease"):
                reasoning.setdefault("Disease", {"id": None, "name": primary_entity_name, "category": None})
                reasoning.setdefault("Ingredients", [])
            elif primary_entity_type == "Ingredient" and self._schema_has_label("Ingredient"):
                reasoning.setdefault("Ingredient", {"id": None, "name": primary_entity_name})
                reasoning.setdefault("Diseases", [])
            elif primary_entity_type == "Drug" and self._schema_has_label("Drug"):
                reasoning.setdefault("Drug", {"id": None, "name": primary_entity_name})
                reasoning.setdefault("Ingredients", [])
                reasoning.setdefault("Diseases", [])

            reasoning.setdefault("ChemicalCompounds", [])
            reasoning.setdefault("DrugChemicalCompounds", [])
            reasoning.setdefault("Drugs", [])
            reasoning.setdefault("HadithReferences", [])
            reasoning.setdefault("BiochemicalMappings", [])
            reasoning.setdefault("meta", {})
            reasoning["meta"]["primary_entity_type"] = primary_entity_type
            reasoning["meta"]["primary_entity_name"] = primary_entity_name

            return reasoning
        except Exception:
            logger.exception("Reasoning builder failed")
            fallback = {
                "ChemicalCompounds": [], "DrugChemicalCompounds": [],
                "Drugs": [], "HadithReferences": [], "BiochemicalMappings": [],
                "meta": {"has_error": True, "primary_entity_type": primary_entity_type,
                         "primary_entity_name": primary_entity_name},
            }
            if primary_entity_type == "Disease":
                fallback["Disease"] = {"id": None, "name": primary_entity_name, "category": None}
                fallback["Ingredients"] = []
            return fallback

    async def _generate_a0_answer(
        self,
        query: str,
        reasoning: dict,
        causal_analysis: dict | None = None,
        dosage_validation: dict | None = None,
        followup_intent: str = "auto",
        traversal_intent: str = "full",
    ) -> str:
        """
        Stage 10: A0 generation with structured chain-of-thought prompting.

        System prompt enforces:
        1. Identify disease
        2. List ingredients via CURES
        3. Extract compounds
        4. Match compounds with drugs
        5. Explain biochemical mechanism
        6. Frame Hadith respectfully
        7. State uncertainty level
        """
        meta = reasoning.get("meta", {}) if isinstance(reasoning, dict) else {}
        primary_entity_type = meta.get("primary_entity_type", "Disease")

        traversal_instruction = self._get_traversal_intent_prompt_instruction(traversal_intent)

        if primary_entity_type == "Disease":
            include_technical_details = followup_intent == "technical"
            system_prompt = (
                "You are PRO-MedGraph, a biomedical and faith-aligned assistant.\n"
                "Use ONLY the provided structured graph reasoning evidence.\n"
                "Do NOT hallucinate entities, studies, or claims.\n\n"
                "Internally reason through these steps (do NOT expose them in the output):\n"
                "- Identify the disease\n"
                "- List ALL traditional/natural ingredients linked via CURES\n"
                "- For each ingredient, note its chemical compounds\n"
                "- Map compounds to drug equivalents (IDENTICAL/LIKELY/WEAK)\n"
                "- Note Hadith references if present\n\n"
                "OUTPUT FORMAT — write a clean, natural-language answer:\n"
                "1. Start with a concise summary sentence.\n"
                "2. List every ingredient with its key compounds and drug mappings.\n"
                "3. Label mapping strengths: IDENTICAL = confirmed, LIKELY = probable, WEAK = tentative.\n"
                "4. If Hadith evidence exists, include it with respectful framing.\n"
                "5. End with a brief safety disclaimer recommending a healthcare professional.\n\n"
                "NEVER output step numbers, headings like 'Step 1:', or chain-of-thought markers.\n"
                "NEVER say phrases like 'Based on the knowledge graph', 'According to the graph', "
                "or mention internal retrieval pipelines unless the user explicitly asks about sources or method.\n"
                f"Technical detail mode: {'ON' if include_technical_details else 'OFF'}.\n"
                "When technical detail mode is OFF, do NOT include chemical compounds, biochemical mappings, "
                "or drug-equivalent names. Keep it practical and user-friendly.\n"
                "Write as a helpful, friendly answer tailored to the user's wording.\n\n"
                f"Traversal-intent constraints:\n{traversal_instruction}"
            )
        elif primary_entity_type == "Ingredient":
            system_prompt = (
                "You are PRO-MedGraph, a biomedical and faith-aligned assistant.\n"
                "Use ONLY the provided structured graph reasoning evidence.\n"
                "Do NOT hallucinate entities, studies, or claims.\n\n"
                "Internally reason through: ingredient → diseases it cures → its chemical compounds → "
                "drug equivalents with mapping strength → Hadith references.\n\n"
                "OUTPUT a clean, natural-language answer (no step numbers, no headings).\n"
                "Include mapping strengths (IDENTICAL/LIKELY/WEAK) and uncertainty.\n"
                "End with a brief safety disclaimer recommending a healthcare professional.\n\n"
                f"Traversal-intent constraints:\n{traversal_instruction}"
            )
        elif primary_entity_type == "Drug":
            system_prompt = (
                "You are PRO-MedGraph, a biomedical and faith-aligned assistant.\n"
                "Use ONLY the provided structured graph reasoning evidence.\n"
                "Do NOT hallucinate entities, studies, or claims.\n\n"
                "Internally reason through: drug → its compounds → natural ingredient equivalents "
                "with mapping strength → diseases those ingredients treat → Hadith if present.\n\n"
                "OUTPUT a clean, natural-language answer (no step numbers, no headings).\n"
                "Include mapping strengths (IDENTICAL/LIKELY/WEAK) and uncertainty.\n"
                "End with a brief safety disclaimer recommending a healthcare professional.\n\n"
                f"Traversal-intent constraints:\n{traversal_instruction}"
            )
        else:
            system_prompt = (
                "You are PRO-MedGraph, a biomedical and faith-aligned assistant.\n"
                "Use ONLY the provided structured graph reasoning evidence.\n"
                "Do NOT hallucinate. State uncertainty when evidence is weak.\n"
                "Always recommend consulting a healthcare professional.\n\n"
                f"Traversal-intent constraints:\n{traversal_instruction}"
            )

        # Build enriched user prompt with causal + dosage context
        # Trim reasoning to avoid inflating prompt tokens
        trimmed_reasoning = self._trim_reasoning_for_llm(reasoning)
        context_parts = [
            f"User Query:\n{query}\n",
            f"Traversal Intent:\n{traversal_intent}\n",
            "The structured reasoning JSON below is already pruned for traversal intent. "
            "Use only the nodes/edges present there and do not infer omitted graph layers.\n",
            f"Structured Graph Reasoning (JSON):\n{json.dumps(trimmed_reasoning, ensure_ascii=False)}",
        ]

        if causal_analysis and causal_analysis.get("causal_paths"):
            top_paths = causal_analysis["causal_paths"][:5]
            context_parts.append(
                f"\nTop Causal Paths (ranked by causal_score):\n"
                f"{json.dumps(top_paths, ensure_ascii=False)}"
            )

        if dosage_validation and dosage_validation.get("comparisons"):
            context_parts.append(
                f"\nDosage Comparison Data:\n"
                f"{json.dumps(dosage_validation, ensure_ascii=False)}"
            )

        user_prompt = "\n".join(context_parts)

        try:
            logger.info("Calling A0 LLM model=%s", self._a0_model)
            return await self._llm_service.generate_completion(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=0.1,
                model=self._a0_model,
            )
        except Exception as exc:
            logger.exception("A0 generation failed")
            if "rate-limit" in str(exc).lower() or "429" in str(exc):
                return (
                    "I’m temporarily rate-limited by the language model provider, so I can’t "
                    "complete this answer right now. Please try again shortly."
                )
            return (
                "I could not generate a complete graph-grounded draft answer. "
                "Available evidence appears limited, so please refine the query."
            )

    async def _validate_answer(
        self,
        query: str,
        a0_answer: str,
        reasoning: dict,
        followup_intent: str = "auto",
        traversal_intent: str = "full",
    ) -> str:
        """Stage 11: Af validation with safety and uncertainty controls."""
        traversal_instruction = self._get_traversal_intent_prompt_instruction(traversal_intent)
        system_prompt = (
            "You are PRO-MedGraph Validator (Af).\n"
            "Your ONLY job is to produce a polished, user-facing answer.\n\n"
            "RULES:\n"
            "1. Validate facts against the provided graph reasoning — remove anything unsupported.\n"
            "2. Add uncertainty language for WEAK links (e.g., 'tentatively', 'may').\n"
            "3. Add a concise medical safety disclaimer recommending a healthcare professional.\n"
            "4. Ensure respectful, non-exclusivist Hadith framing.\n"
            "5. Do NOT add unsupported claims or miracle/guarantee/divine-cure language.\n\n"
            "FORMAT:\n"
            "- Strip ALL step numbers, headings ('Step 1:', '## Step 2:', etc.), and chain-of-thought scaffolding.\n"
            "- Output a clean, conversational, well-structured answer suitable for a patient or end user.\n"
            "- Use bullet points or numbered lists only for ingredient/drug listings, not for reasoning steps.\n"
            "- Do NOT mention internal systems such as 'knowledge graph', 'graph retrieval', or similar wording "
            "unless the user explicitly asks for method/source details.\n"
            "- Unless the user explicitly asks for technical details, remove chemical compounds, mapping labels, "
            "and drug-equivalent lists from the final answer.\n"
            "- Respect traversal-intent boundaries strictly; do not add entities/edges outside allowed path scope.\n"
            f"- Traversal-intent constraints: {traversal_instruction}\n"
            "- Return ONLY the final answer text — no meta-commentary."
        )

        user_prompt = (
            f"User Query:\n{query}\n\n"
            f"Detected Intent Label:\n{followup_intent}\n\n"
            f"Traversal Intent:\n{traversal_intent}\n\n"
            "The structured reasoning JSON below is already pruned for traversal intent.\n"
            f"Draft Answer (A0):\n{a0_answer}\n\n"
            f"Structured Graph Reasoning (JSON):\n"
            f"{json.dumps(self._trim_reasoning_for_llm(reasoning), ensure_ascii=False)}"
        )

        try:
            logger.info("Calling Af LLM model=%s", self._af_model)
            return await self._llm_service.generate_completion(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=0.1,
                model=self._af_model,
            )
        except Exception:
            logger.exception("Af validation failed")
            return a0_answer

    async def _generate_general_answer(
        self, query: str, history: list[dict] | None = None,
    ) -> str:
        """Generate conversational response for non-biomedical queries."""
        system_prompt = (
            "You are PRO-MedGraph, a helpful assistant.\n"
            "Respond naturally and briefly to general user messages.\n"
            "If the user asks a medical question without specific disease details,\n"
            "ask a clarifying follow-up and avoid definitive medical claims.\n"
            "If conversation history is provided, use it to give a contextual answer."
        )
        # Build user prompt with optional history for context
        parts: list[str] = []
        if history:
            recent = history[-6:]
            convo = "\n".join(
                f"{'User' if m.get('role') == 'user' else 'Assistant'}: {m.get('content', '')[:300]}"
                for m in recent
            )
            parts.append(f"Conversation history:\n{convo}\n")
        parts.append(f"User message:\n{query}")
        user_prompt = "\n".join(parts)

        try:
            return await self._llm_service.generate_completion(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=0.3,
                model=self._a0_model,
            )
        except Exception:
            logger.exception("General response failed")
            return (
                "Hello! I am PRO-MedGraph. I can help with biomedical questions. "
                "If you share a disease or treatment topic, I can provide a graph-grounded answer."
            )

    # ── Co-reference resolution ──────────────────────────────────────────

    async def _resolve_coreferences(self, query: str, history: list[dict]) -> str:
        """Rewrite *query* into a self-contained sentence using conversation history.

        If the query already looks self-contained the LLM returns it unchanged.
        Only the last 6 exchanges are considered to limit token use.
        """
        recent = history[-6:]
        convo_lines = "\n".join(
            f"{'User' if m.get('role') == 'user' else 'Assistant'}: {m.get('content', '')[:300]}"
            for m in recent
        )

        system_prompt = (
            "You are a query rewriter. Given a conversation history and the "
            "latest user message, rewrite the user message so it is fully "
            "self-contained (resolve pronouns, 'it', 'that', 'the hadith you "
            "mentioned', etc.). Keep the rewrite short — one sentence.\n"
            "If the message is already self-contained, return it unchanged.\n"
            "Output ONLY the rewritten query, nothing else."
        )
        user_prompt = (
            f"Conversation history:\n{convo_lines}\n\n"
            f"Latest user message: {query}\n\n"
            f"Rewritten query:"
        )

        try:
            rewritten = await self._llm_service.generate_completion(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=0.0,
                model=self._coref_model,
            )
            # Sanity: if the LLM returns garbage or something way too long, keep original
            rewritten = (rewritten or "").strip().strip('"').strip("'").strip()
            if not rewritten or len(rewritten) > len(query) * 4:
                return query
            return rewritten
        except Exception:
            logger.warning("Coreference resolution failed; using original query")
            return query

    async def _classify_followup_intent(self, query: str, history: list[dict]) -> str:
        """Classify follow-up intent using conversation context.

        Returns one of: hadith | ingredient | technical | general | auto
        """
        recent = history[-6:]
        convo_lines = "\n".join(
            f"{'User' if m.get('role') == 'user' else 'Assistant'}: {m.get('content', '')[:240]}"
            for m in recent
        )
        if not convo_lines:
            convo_lines = "(none)"

        system_prompt = (
            "You are an intent classifier for follow-up chat messages.\n"
            "Choose exactly one label: hadith, ingredient, technical, general, auto.\n"
            "Definitions:\n"
            "- hadith: asks for hadith/sunnah/reference/citation support\n"
            "- ingredient: asks what food/herb/remedy to use\n"
            "- technical: asks for compounds/mechanism/mappings/reasoning details\n"
            "- general: casual chat/greeting/non-medical\n"
            "- auto: unclear/none of the above\n"
            "Output ONLY the label."
        )
        user_prompt = (
            f"Conversation history:\n{convo_lines}\n\n"
            f"Latest user message: {query}\n\n"
            "Label:"
        )

        try:
            label = await self._llm_service.generate_completion(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=0.0,
                model=self._intent_model,
            )
            label = (label or "").strip().lower()
            if label in {"hadith", "ingredient", "technical", "general", "auto"}:
                return label
            return "auto"
        except Exception:
            logger.warning("Follow-up intent classification failed; defaulting to auto")
            return "auto"

    # ── Utility methods ────────────────────────────────────────────────────

    @staticmethod
    def _get_traversal_intent_prompt_instruction(traversal_intent: str) -> str:
        intent = (traversal_intent or "full").strip().lower()
        if intent == "cure":
            return (
                "Allowed scope only: Disease → Ingredient → Hadith → Reference. "
                "Do not use or infer ChemicalCompound, DrugChemicalCompound, or Drug layers. "
                "Emphasize prophetic remedies and hadith-grounded guidance respectfully."
            )
        if intent == "reasoning":
            return (
                "Allowed scope only: Ingredient → ChemicalCompound. "
                "Use CONTAINS relationship properties (source, quantity) when present in evidence. "
                "Emphasize chemical reasoning; do not use Drug nodes."
            )
        if intent == "drug":
            return (
                "Allowed scope only: Ingredient → ChemicalCompound → DrugChemicalCompound → Drug (+Book if present). "
                "Emphasize modern drug references and book context when available. "
                "Do not prioritize Hadith unless explicitly requested by user intent."
            )
        return (
            "Full traversal scope is allowed. Use only provided evidence and do not hallucinate missing graph edges."
        )

    @staticmethod
    def _compute_evidence_strength(reasoning: dict) -> str:
        mappings = reasoning.get("BiochemicalMappings", []) if isinstance(reasoning, dict) else []
        if not isinstance(mappings, list) or not mappings:
            return "weak"
        strengths = [
            (item.get("mapping_strength") if isinstance(item, dict) else None) for item in mappings
        ]
        if sum(1 for v in strengths if v == "IDENTICAL") >= 1:
            return "strong"
        if sum(1 for v in strengths if v == "LIKELY") >= 1:
            return "moderate"
        return "weak"

    @staticmethod
    def _compute_confidence_score(evidence_strength: str, graph_paths_used: int) -> float | None:
        if graph_paths_used <= 0:
            return None
        base = {"strong": 0.85, "moderate": 0.65, "weak": 0.45}.get(evidence_strength, 0.4)
        boost = min(0.1, graph_paths_used * 0.01)
        return round(min(0.99, base + boost), 2)

    @staticmethod
    def _get_avg_mapping_strength(reasoning: dict) -> float | None:
        """Get average numeric mapping strength for confidence breakdown."""
        strength_map = {"IDENTICAL": 1.0, "LIKELY": 0.7, "WEAK": 0.35}
        mappings = reasoning.get("BiochemicalMappings", []) if isinstance(reasoning, dict) else []
        scores = []
        for m in mappings:
            if isinstance(m, dict):
                label = m.get("mapping_strength")
                if label in strength_map:
                    scores.append(strength_map[label])
        return round(sum(scores) / len(scores), 3) if scores else None

    @staticmethod
    def _fallback_response(reason: str) -> dict:
        logger.warning("Returning fallback response reason=%s", reason)
        return {
            "final_answer": FALLBACK_MESSAGE,
            "evidence_strength": "weak",
            "graph_paths_used": 0,
            "confidence_score": None,
        }


# ═══════════════════════════════════════════════════════════════════════════════
#  Module-level convenience functions (backward compatible)
# ═══════════════════════════════════════════════════════════════════════════════

_DEFAULT_ORCHESTRATOR: GraphRAGOrchestrator | None = None


def _get_default_orchestrator() -> GraphRAGOrchestrator:
    global _DEFAULT_ORCHESTRATOR
    if _DEFAULT_ORCHESTRATOR is None:
        _DEFAULT_ORCHESTRATOR = GraphRAGOrchestrator()
    return _DEFAULT_ORCHESTRATOR


def process_user_query(query: str) -> dict:
    return _get_default_orchestrator().process_user_query(query)


async def process_user_query_async(query: str) -> dict:
    return await _get_default_orchestrator().process_user_query_async(query)


async def process_user_query_with_context_async(query: str) -> dict:
    return await _get_default_orchestrator().process_user_query_with_context_async(query)


__all__ = [
    "GraphRAGOrchestrator",
    "process_user_query",
    "process_user_query_async",
    "process_user_query_with_context_async",
]
