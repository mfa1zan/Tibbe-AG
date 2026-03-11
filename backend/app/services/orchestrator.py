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
import re
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

KG_MISSING_INFO_MESSAGE = "I could not find this information in the knowledge graph."

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
            """Run synchronous graph retrieval in a thread while preserving ContextVar trace state."""
            return await asyncio.to_thread(
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

        rationale_outcome, retrieval_outcome = await asyncio.gather(
            rationale_task,
            _graph_retrieval_async(),
            return_exceptions=True,
        )

        if isinstance(rationale_outcome, Exception):
            logger.exception("STAGE 3: Rationale plan generation failed; using default plan")
            rationale_result = self._build_default_rationale_plan(traversal_intent)
        else:
            rationale_result = rationale_outcome if isinstance(rationale_outcome, dict) else self._build_default_rationale_plan(traversal_intent)

        if isinstance(retrieval_outcome, Exception):
            logger.exception("STAGE 4: Primary graph retrieval raised an exception")
            retrieval_outcome = (
                {"error": "Primary retrieval failed", "Relations": [], "HadithReferences": []},
                "Unknown",
                "N/A",
            )

        subgraph, primary_entity_type, primary_entity_name = retrieval_outcome

        if self._has_detected_entities(entities) and not self._has_subgraph_evidence(subgraph):
            logger.warning("STAGE 4: No subgraph evidence from primary retrieval; forcing fallback traversal")
            subgraph, primary_entity_type, primary_entity_name = await asyncio.to_thread(
                self._run_fallback_traversal,
                entities,
                traversal_intent,
            )

        logger.info(
            "STAGE 3: Rationale plan source='%s' steps=%d node_types=%s",
            rationale_result.get("plan_source"),
            len(rationale_result.get("rationale_plan", [])),
            rationale_result.get("relevant_node_types"),
        )
        logger.info(
            "STAGE 4: Graph retrieval → entity_type='%s' entity_name='%s'",
            primary_entity_type, primary_entity_name,
        )
        tracer.end_step({
            "rationale_plan_source": rationale_result.get("plan_source"),
            "rationale_plan_steps": len(rationale_result.get("rationale_plan", [])),
            "relevant_node_types": rationale_result.get("relevant_node_types"),
            "primary_entity_type": primary_entity_type,
            "primary_entity_name": primary_entity_name,
            "subgraph_keys": list(subgraph.keys()) if isinstance(subgraph, dict) else [],
        })
        tracer.log_data("rationale_plan_full", rationale_result)

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

        # ── Stage 9b: Retrieval failsafe check before A0 ─────────────────
        if self._has_detected_entities(entities):
            kg_query_count = len((tracer.to_dict().get("kg_queries") or []))
            if kg_query_count == 0:
                tracer.start_step("stage_9b_retrieval_failsafe", {
                    "reason": "entities_detected_but_no_neo4j_queries_logged",
                    "entities": entities,
                    "traversal_intent": traversal_intent,
                })
                logger.warning(
                    "STAGE 9b: Neo4j query count is 0 despite detected entities; triggering fallback traversal"
                )

                subgraph, primary_entity_type, primary_entity_name = await asyncio.to_thread(
                    self._run_fallback_traversal,
                    entities,
                    traversal_intent,
                )

                reasoning = self._build_reasoning(subgraph, primary_entity_type, primary_entity_name)
                graph_paths_used = len(reasoning.get("BiochemicalMappings", []))
                causal_analysis = causal_reasoner.run_causal_analysis(reasoning)
                dosage_result = validate_dosage(
                    reasoning=reasoning,
                    causal_paths=causal_analysis.get("causal_paths"),
                )

                tracer.end_step({
                    "fallback_primary_entity_type": primary_entity_type,
                    "fallback_primary_entity_name": primary_entity_name,
                    "fallback_graph_paths_used": graph_paths_used,
                    "fallback_kg_query_count_after": len((tracer.to_dict().get("kg_queries") or [])),
                })

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

        mappings = trimmed.get("BiochemicalMappings")
        if isinstance(mappings, list) and mappings:
            strength_rank = {"IDENTICAL": 3, "LIKELY": 2, "WEAK": 1}

            def _mapping_sort_key(item: Any) -> tuple[int, int, int, int, int]:
                if not isinstance(item, dict):
                    return (0, 0, 0, 0, 0)

                ingredient = item.get("ingredient") if isinstance(item.get("ingredient"), dict) else {}
                compound = item.get("chemical_compound") if isinstance(item.get("chemical_compound"), dict) else {}
                dcc = item.get("drug_chemical_compound") if isinstance(item.get("drug_chemical_compound"), dict) else {}
                drug = item.get("drug") if isinstance(item.get("drug"), dict) else {}

                has_ingredient = 1 if ingredient.get("name") else 0
                has_compound = 1 if compound.get("name") else 0
                has_dcc = 1 if dcc.get("name") else 0
                has_drug = 1 if drug.get("name") else 0
                relation = item.get("mapping_strength")
                rel_score = strength_rank.get(relation, 0)
                completeness = has_ingredient + has_compound + has_dcc + has_drug
                return (rel_score, completeness, has_drug, has_compound, has_ingredient)

            ranked_mappings = sorted(mappings, key=_mapping_sort_key, reverse=True)
            trimmed["BiochemicalMappings"] = ranked_mappings

            mapping_drug_ids = {
                m.get("drug", {}).get("id")
                for m in ranked_mappings
                if isinstance(m, dict)
                and isinstance(m.get("drug"), dict)
                and m.get("drug", {}).get("id")
            }
            drugs = trimmed.get("Drugs")
            if isinstance(drugs, list) and drugs:
                prioritized = []
                others = []
                for d in drugs:
                    if not isinstance(d, dict):
                        continue
                    if d.get("id") in mapping_drug_ids:
                        prioritized.append(d)
                    else:
                        others.append(d)
                trimmed["Drugs"] = prioritized + others

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

    @staticmethod
    def _has_drug_substitute_evidence(reasoning: dict | None) -> bool:
        if not isinstance(reasoning, dict):
            return False
        mappings = reasoning.get("BiochemicalMappings")
        if not isinstance(mappings, list) or not mappings:
            return False
        for mapping in mappings:
            if not isinstance(mapping, dict):
                continue
            drug = mapping.get("drug") if isinstance(mapping.get("drug"), dict) else {}
            compound = mapping.get("chemical_compound") if isinstance(mapping.get("chemical_compound"), dict) else {}
            dcc = mapping.get("drug_chemical_compound") if isinstance(mapping.get("drug_chemical_compound"), dict) else {}
            if drug.get("name") and compound.get("name") and dcc.get("name"):
                return True
        return False

    @staticmethod
    def _build_drug_substitute_answer_from_reasoning(query: str, reasoning: dict) -> str:
        mappings = reasoning.get("BiochemicalMappings") if isinstance(reasoning, dict) else []
        if not isinstance(mappings, list):
            return KG_MISSING_INFO_MESSAGE

        strength_order = {"IDENTICAL": 3, "LIKELY": 2, "WEAK": 1}

        grouped: dict[str, dict[str, Any]] = {}
        for mapping in mappings:
            if not isinstance(mapping, dict):
                continue
            ingredient = mapping.get("ingredient") if isinstance(mapping.get("ingredient"), dict) else {}
            compound = mapping.get("chemical_compound") if isinstance(mapping.get("chemical_compound"), dict) else {}
            dcc = mapping.get("drug_chemical_compound") if isinstance(mapping.get("drug_chemical_compound"), dict) else {}
            drug = mapping.get("drug") if isinstance(mapping.get("drug"), dict) else {}
            strength = mapping.get("mapping_strength")

            ingredient_name = ingredient.get("name")
            compound_name = compound.get("name")
            dcc_name = dcc.get("name")
            drug_name = drug.get("name")

            if not (ingredient_name and compound_name and dcc_name and drug_name):
                continue

            bucket = grouped.setdefault(drug_name, {
                "best_strength": "WEAK",
                "compound_links": [],
                "ingredient": ingredient_name,
                "identical_count": 0,
                "likely_count": 0,
                "weak_count": 0,
                "compound_names": set(),
            })

            if strength_order.get(strength, 0) > strength_order.get(bucket["best_strength"], 0):
                bucket["best_strength"] = strength

            if strength == "IDENTICAL":
                bucket["identical_count"] += 1
            elif strength == "LIKELY":
                bucket["likely_count"] += 1
            else:
                bucket["weak_count"] += 1

            bucket["compound_names"].add(compound_name)

            link_text = f"{compound_name} → {dcc_name}"
            if link_text not in bucket["compound_links"]:
                bucket["compound_links"].append(link_text)

        if not grouped:
            return KG_MISSING_INFO_MESSAGE

        def _overlap_score(payload: dict[str, Any]) -> int:
            # Weighted score: IDENTICAL links are trusted most, then LIKELY, then WEAK.
            return (
                int(payload.get("identical_count", 0)) * 100
                + int(payload.get("likely_count", 0)) * 10
                + int(payload.get("weak_count", 0))
            )

        ranked_drugs = sorted(
            grouped.items(),
            key=lambda kv: (
                _overlap_score(kv[1]),
                len(kv[1].get("compound_names", set())),
                strength_order.get(kv[1].get("best_strength"), 0),
                len(kv[1].get("compound_links", [])),
            ),
            reverse=True,
        )

        def _strength_word(value: str) -> str:
            if value == "IDENTICAL":
                return "identical"
            if value == "LIKELY":
                return "likely corresponding"
            return "weakly matching"

        top = ranked_drugs[:6]
        ingredient_name = top[0][1].get("ingredient")

        lines = [
            f"For {ingredient_name}, I found drug candidates in the knowledge graph ranked by shared chemical overlap:",
            "",
        ]
        for drug_name, payload in top:
            links = payload.get("compound_links", [])[:2]
            relation_word = _strength_word(payload.get("best_strength", "WEAK"))
            overlap = len(payload.get("compound_names", set()))
            identical = int(payload.get("identical_count", 0))
            likely = int(payload.get("likely_count", 0))
            weak = int(payload.get("weak_count", 0))

            confidence_bits: list[str] = []
            if identical:
                confidence_bits.append(f"{identical} identical")
            if likely:
                confidence_bits.append(f"{likely} likely")
            if weak:
                confidence_bits.append(f"{weak} weak")
            confidence_text = ", ".join(confidence_bits) if confidence_bits else relation_word

            if links:
                lines.append(
                    f"• {drug_name} — shared compounds: {overlap} ({confidence_text}); "
                    f"examples: {', '.join(links)}"
                )
            else:
                lines.append(f"• {drug_name} — shared compounds: {overlap} ({confidence_text})")

        lines.extend([
            "",
            "These drugs are chemical-overlap candidates and not guaranteed complete alternatives to the ingredient.",
            "Please verify suitability, dosage, and interactions with a qualified healthcare professional before use.",
        ])
        return "\n".join(lines)

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
        if label in {"reasoning", "technical", "mechanism", "chemical"}:
            return "reasoning"
        if label == "drug" or label == "drug_substitute" or (isinstance(drug_entity, str) and drug_entity.strip()):
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

    @staticmethod
    def _has_detected_entities(entities: dict[str, Any] | None) -> bool:
        if not isinstance(entities, dict):
            return False
        for key in ("disease", "ingredient", "drug"):
            value = entities.get(key)
            if isinstance(value, str) and value.strip():
                return True
        return False

    @staticmethod
    def _has_subgraph_evidence(subgraph: dict[str, Any] | None) -> bool:
        if not isinstance(subgraph, dict):
            return False
        for key in (
            "Disease", "Diseases", "Ingredient", "Ingredients", "Drug", "Drugs",
            "ChemicalCompounds", "DrugChemicalCompounds", "Relations", "HadithReferences",
        ):
            value = subgraph.get(key)
            if isinstance(value, list) and len(value) > 0:
                return True
            if isinstance(value, dict) and any(v for v in value.values()):
                return True
        return False

    @staticmethod
    def _build_default_rationale_plan(traversal_intent: str) -> dict[str, Any]:
        intent = (traversal_intent or "full").strip().lower()
        if intent == "drug":
            plan = [
                {"step": "1", "action": "Identify primary drug or ingredient anchor", "target": "Drug"},
                {"step": "2", "action": "Traverse to DrugChemicalCompounds", "target": "DrugChemicalCompound"},
                {"step": "3", "action": "Map to ChemicalCompounds", "target": "ChemicalCompound"},
                {"step": "4", "action": "Map to Ingredients", "target": "Ingredient"},
            ]
        elif intent == "reasoning":
            plan = [
                {"step": "1", "action": "Identify ingredient", "target": "Ingredient"},
                {"step": "2", "action": "Retrieve contained compounds", "target": "ChemicalCompound"},
                {"step": "3", "action": "Retrieve mapped drug compounds", "target": "DrugChemicalCompound"},
            ]
        elif intent == "cure":
            plan = [
                {"step": "1", "action": "Identify disease", "target": "Disease"},
                {"step": "2", "action": "Retrieve linked ingredients", "target": "Ingredient"},
            ]
        else:
            plan = [
                {"step": "1", "action": "Identify primary biomedical entity", "target": "Disease"},
                {"step": "2", "action": "Retrieve linked graph neighborhood", "target": "Ingredient"},
            ]

        return {
            "rationale_plan": plan,
            "plan_source": "fallback_default",
            "relevant_node_types": [step["target"] for step in plan],
        }

    def _run_fallback_traversal(
        self,
        entities: dict[str, Any],
        traversal_intent: str,
    ) -> tuple[dict, str, str]:
        disease = entities.get("disease") if isinstance(entities, dict) else None
        ingredient = entities.get("ingredient") if isinstance(entities, dict) else None
        drug = entities.get("drug") if isinstance(entities, dict) else None

        if isinstance(ingredient, str) and ingredient.strip():
            if (traversal_intent or "").strip().lower() == "drug":
                logger.info("Fallback traversal: ingredient->compound->drug substitute chain for '%s'", ingredient)
                subgraph = graph_service.get_ingredient_drug_substitute_subgraph(ingredient)
                return subgraph, "Ingredient", ingredient
            logger.info("Fallback traversal: ingredient-centered traversal for '%s'", ingredient)
            subgraph = self._ingredient_subgraph_fetcher(ingredient) or {}
            return subgraph, "Ingredient", ingredient

        if isinstance(disease, str) and disease.strip():
            logger.info("Fallback traversal: disease-centered traversal for '%s'", disease)
            subgraph = self._disease_subgraph_fetcher(disease) or {}
            return subgraph, "Disease", disease

        if isinstance(drug, str) and drug.strip():
            logger.info("Fallback traversal: drug-centered traversal for '%s'", drug)
            subgraph = self._drug_subgraph_fetcher(drug) or {}
            return subgraph, "Drug", drug

        return {"error": "No entities for fallback traversal", "Relations": [], "HadithReferences": []}, "Unknown", "N/A"

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
                disease = disease.strip() if isinstance(disease, str) else disease
                if not disease:
                    logger.error("Intent routing: disease entity is empty; skipping query execution")
                    return (
                        {"error": "Missing required parameter: disease", "Relations": [], "HadithReferences": []},
                        "Disease",
                        "N/A",
                    )
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
                ingredient = ingredient.strip() if isinstance(ingredient, str) else ingredient
                if not ingredient:
                    logger.error("Intent routing: ingredient entity is empty; skipping query execution")
                    return (
                        {"error": "Missing required parameter: ingredient", "Relations": [], "HadithReferences": []},
                        "Ingredient",
                        "N/A",
                    )
                if not self._schema_has_label("Ingredient"):
                    return (
                        {"error": "Ingredient label not available in schema", "Relations": [], "HadithReferences": []},
                        "Ingredient",
                        ingredient,
                    )
                logger.info("Intent routing: INGREDIENT query for '%s'", ingredient)
                if traversal_intent == "drug":
                    logger.info("Intent routing: applying ingredient drug-substitute traversal for '%s'", ingredient)
                    subgraph = graph_service.get_ingredient_drug_substitute_subgraph(ingredient) or {}
                else:
                    subgraph = self._ingredient_subgraph_fetcher(ingredient) or {}
                subgraph.setdefault("HadithReferences", [])
                subgraph = self._prune_subgraph_by_traversal_intent(subgraph, traversal_intent, include_hadith)
                return subgraph, "Ingredient", ingredient

            if drug:
                drug = drug.strip() if isinstance(drug, str) else drug
                if not drug:
                    logger.error("Intent routing: drug entity is empty; skipping query execution")
                    return (
                        {"error": "Missing required parameter: drug", "Relations": [], "HadithReferences": []},
                        "Drug",
                        "N/A",
                    )
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

        if not self._has_graph_grounding_evidence(reasoning):
            logger.info("A0: No graph grounding evidence found; returning KG missing info message")
            return KG_MISSING_INFO_MESSAGE

        traversal_instruction = self._get_traversal_intent_prompt_instruction(traversal_intent)

        # ── Global Graph Grounding Preamble (shared by all entity prompts) ──
        _GRAPH_GROUNDING_PREAMBLE = (
            "You are PRO-MedGraph, a biomedical and faith-aligned assistant.\n"
            "════ STRICT GRAPH-GROUNDING MODE ════\n"
            "You must ONLY use information provided in the 'Structured Graph Reasoning (JSON)' input.\n"
            "You must NEVER:\n"
            "  • invent ingredients\n"
            "  • invent diseases\n"
            "  • invent chemical compounds\n"
            "  • invent drug mappings\n"
            "  • cite external studies, URLs, or statistics\n"
            "  • use pretrained medical knowledge\n"
            "If the requested information is not present in the graph evidence, respond with:\n"
            "\"I could not find this information in the knowledge graph.\"\n"
            "The model must never answer from pretrained knowledge.\n\n"
        )

        # ── Chemical Relation Interpretation Rules ──
        _CHEMICAL_RELATION_RULES = (
            "════ CHEMICAL MAPPING RELATION INTERPRETATION ════\n"
            "The graph contains three chemical mapping relations. Treat them differently:\n\n"
            "IS_IDENTICAL_TO\n"
            "  Meaning: Confirmed same chemical compound.\n"
            "  Use confidently without further verification.\n"
            "  Wording: 'identical chemical compound'\n\n"
            "IS_LIKELY_EQUIVALENT_TO\n"
            "  Meaning: High probability match.\n"
            "  Verify that compound names represent the same chemical concept before presenting.\n"
            "  Wording: 'likely corresponds to'\n\n"
            "IS_WEAK_MATCH_TO\n"
            "  Meaning: Possible but uncertain mapping.\n"
            "  Mention only if necessary and clearly mark as speculative.\n"
            "  Wording: 'may correspond to'\n\n"
        )

        # ── Multihop Reasoning Control ──
        _MULTIHOP_CONTROL = (
            "════ MULTIHOP REASONING CONTROL ════\n"
            "Use multi-hop reasoning chains ONLY when the query requires it:\n"
            "  Multihop REQUIRED for: mechanism questions, drug alternatives, compound mapping, "
            "ingredient→drug reasoning, 'how does X cure Y' questions.\n"
            "  Multihop NOT required for: simple ingredient lists, compound lists, "
            "disease-ingredient mapping, 'what cures X' questions.\n"
            "When multihop is not required, return a direct answer without unnecessary reasoning chains.\n\n"
        )

        # ── A0 Output Rules ──
        _A0_OUTPUT_RULES = (
            "════ A0 GENERATION RULES ════\n"
            "1. Only use graph evidence.\n"
            "2. Never invent nodes.\n"
            "3. Include chemical reasoning only when the query explicitly asks for it "
            "(mechanism, 'how does it work', compound details, drug mapping).\n"
            "4. Keep answers concise and structured.\n"
            "5. Use bullet lists for ingredients or compounds.\n"
            "6. Mention relation properties (source, quantity) when available in the evidence.\n"
            "7. End with a brief safety disclaimer recommending a healthcare professional.\n\n"
        )

        # ── Query-Type Behavior Rules ──
        _QUERY_TYPE_BEHAVIOR = (
            "════ QUERY-TYPE BEHAVIOR ════\n"
            "Determine what the user is asking and respond accordingly:\n\n"
            "TYPE 1: INGREDIENT → DISEASE QUERY (e.g. 'Which ingredients cure fever?')\n"
            "  • Return ingredients connected to the disease node.\n"
            "  • List them clearly using bullet points.\n"
            "  • Do NOT add chemical explanations unless the user explicitly requests them.\n\n"
            "TYPE 2: INGREDIENT → CHEMICAL QUERY (e.g. 'What chemicals are in heena?')\n"
            "  • Retrieve Ingredient → CONTAINS → ChemicalCompound from the graph.\n"
            "  • List chemical compound names.\n"
            "  • Include relation properties (source, quantity) when available.\n"
            "  • Only include compounds present in the graph evidence.\n\n"
            "TYPE 3: MECHANISM / 'HOW DOES IT CURE' QUESTIONS (e.g. 'How does honey cure fever?')\n"
            "  • Build the full reasoning chain: Disease → Ingredient → ChemicalCompound → DrugChemicalCompound (if mapped).\n"
            "  • The answer MUST include:\n"
            "    1. Ingredient name\n"
            "    2. Active chemical compounds (from graph)\n"
            "    3. Drug compound mapping with relation type (IS_IDENTICAL_TO / IS_LIKELY_EQUIVALENT_TO / IS_WEAK_MATCH_TO)\n"
            "    4. Explanation of mechanism using only graph evidence\n"
            "  • Include relation properties when they exist.\n\n"
            "TYPE 4: DRUG SUBSTITUTE QUERY (e.g. 'I don't have paracetamol. Which food ingredient can I use?')\n"
            "  • Trace: Drug → DrugChemicalCompound → ChemicalCompound → Ingredient\n"
            "  • Recommend ingredients containing chemically similar compounds.\n"
            "  • Mention chemical equivalence relations from the graph.\n\n"
        )

        if primary_entity_type == "Disease":
            include_technical_details = followup_intent == "technical"
            system_prompt = (
                _GRAPH_GROUNDING_PREAMBLE
                + _CHEMICAL_RELATION_RULES
                + _MULTIHOP_CONTROL
                + _QUERY_TYPE_BEHAVIOR
                + _A0_OUTPUT_RULES
                + "════ DISEASE-CENTRIC REASONING ════\n"
                "Internally reason through these steps (do NOT expose them in the output):\n"
                "- Identify the disease\n"
                "- List ALL traditional/natural ingredients linked via CURES\n"
                "- For each ingredient, note its chemical compounds\n"
                "- Map compounds to drug equivalents using the 3-tier relation system (IDENTICAL/LIKELY/WEAK)\n"
                "- Note Hadith references if present\n\n"
                "OUTPUT FORMAT — write a clean, natural-language answer:\n"
                "1. Start with a concise summary sentence.\n"
                "2. For cure questions, keep the answer practical. List ingredients using bullet points.\n"
                "3. Do NOT list every ingredient by default; mention only the most relevant remedy guidance unless explicitly asked for full detail.\n"
                "4. If Hadith evidence exists, include it with respectful framing.\n\n"
                "NEVER output step numbers, headings like 'Step 1:', or chain-of-thought markers.\n"
                "NEVER say phrases like 'Based on the knowledge graph', 'According to the graph', "
                "or mention internal retrieval pipelines unless the user explicitly asks about sources or method.\n"
                f"Technical detail mode: {'ON' if include_technical_details else 'OFF'}.\n"
                "Only provide chemical compounds, ingredient→compound→drug mapping chains, drug-equivalent names, "
                "or mapping strengths when the user explicitly asks for these technical details OR asks a mechanism question.\n"
                "When technical detail mode is OFF and it is NOT a mechanism question, do NOT include chemical compounds, "
                "biochemical mappings, drug-equivalent names, or mapping strengths. Keep it practical and user-friendly.\n"
                "When answering ingredient, disease, chemical compound, or drug questions, list ONLY nodes/relations that exist in the JSON evidence.\n"
                "If a requested item is absent in the JSON evidence, explicitly state it is not present in the knowledge graph.\n"
                "Write as a helpful, friendly answer tailored to the user's wording. Keep it concise.\n\n"
                f"Traversal-intent constraints:\n{traversal_instruction}"
            )
        elif primary_entity_type == "Ingredient":
            system_prompt = (
                _GRAPH_GROUNDING_PREAMBLE
                + _CHEMICAL_RELATION_RULES
                + _MULTIHOP_CONTROL
                + _QUERY_TYPE_BEHAVIOR
                + _A0_OUTPUT_RULES
                + "════ INGREDIENT-CENTRIC REASONING ════\n"
                "Internally reason through: ingredient → diseases it cures → its chemical compounds → "
                "drug equivalents with mapping strength → Hadith references.\n\n"
                "OUTPUT a clean, natural-language answer (no step numbers, no headings).\n"
                "Use the 3-tier chemical relation system: IDENTICAL ('identical chemical compound'), "
                "LIKELY ('likely corresponds to'), WEAK ('may correspond to').\n"
                "For simple ingredient/compound listing queries, return a direct list — no reasoning chains.\n"
                "For mechanism queries, build the full Ingredient → ChemicalCompound → DrugChemicalCompound chain.\n"
                "Only list nodes/relations present in the JSON evidence.\n"
                "If a requested item is absent in the JSON evidence, explicitly state it is not present in the knowledge graph.\n"
                "End with a brief safety disclaimer recommending a healthcare professional.\n\n"
                f"Traversal-intent constraints:\n{traversal_instruction}"
            )
        elif primary_entity_type == "Drug":
            system_prompt = (
                _GRAPH_GROUNDING_PREAMBLE
                + _CHEMICAL_RELATION_RULES
                + _MULTIHOP_CONTROL
                + _QUERY_TYPE_BEHAVIOR
                + _A0_OUTPUT_RULES
                + "════ DRUG-CENTRIC REASONING ════\n"
                "Internally reason through: drug → its DrugChemicalCompounds → matching ChemicalCompounds → "
                "natural Ingredients containing those compounds → diseases those ingredients treat → Hadith if present.\n\n"
                "For DRUG SUBSTITUTE QUERIES (user wants natural alternatives to a drug):\n"
                "  • Trace: Drug → DrugChemicalCompound → ChemicalCompound → Ingredient\n"
                "  • Recommend ingredients containing chemically similar compounds.\n"
                "  • Use the 3-tier relation system: IDENTICAL ('identical compound'), "
                "LIKELY ('likely corresponds to'), WEAK ('may correspond to').\n\n"
                "OUTPUT a clean, natural-language answer (no step numbers, no headings).\n"
                "Only list nodes/relations present in the JSON evidence.\n"
                "If a requested item is absent in the JSON evidence, explicitly state it is not present in the knowledge graph.\n"
                "End with a brief safety disclaimer recommending a healthcare professional.\n\n"
                f"Traversal-intent constraints:\n{traversal_instruction}"
            )
        else:
            system_prompt = (
                _GRAPH_GROUNDING_PREAMBLE
                + _CHEMICAL_RELATION_RULES
                + _A0_OUTPUT_RULES
                + "If graph evidence is missing for the requested entity/relation, reply EXACTLY: "
                "\"I could not find this information in the knowledge graph.\"\n"
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
            a0_text = await self._llm_service.generate_completion(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=0.1,
                model=self._a0_model,
            )

            if (
                (a0_text or "").strip() == KG_MISSING_INFO_MESSAGE
                and (traversal_intent or "").strip().lower() == "drug"
                and self._has_drug_substitute_evidence(reasoning)
            ):
                logger.warning(
                    "A0 returned KG missing message despite available drug-substitute evidence; "
                    "using deterministic graph-grounded substitute answer"
                )
                return self._build_drug_substitute_answer_from_reasoning(query, reasoning)

            return a0_text
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
        if (a0_answer or "").strip() == KG_MISSING_INFO_MESSAGE:
            return KG_MISSING_INFO_MESSAGE

        traversal_instruction = self._get_traversal_intent_prompt_instruction(traversal_intent)
        system_prompt = (
            "You are PRO-MedGraph Validator (Af).\n"
            "Your ONLY job is to validate the draft answer A0 against the graph evidence "
            "and produce a polished, user-facing answer.\n\n"
            "════ STRICT GRAPH-GROUNDING MODE ════\n"
            "You must ONLY use information provided in the 'Structured Graph Reasoning (JSON)' input.\n"
            "You must NEVER invent ingredients, diseases, chemical compounds, or drug mappings.\n"
            "If the graph evidence lacks the requested information, respond with:\n"
            "\"I could not find this information in the knowledge graph.\"\n\n"
            "════ 3C3H VALIDATION RULE ════\n"
            "Evaluate the draft answer (A0) using exactly these six criteria:\n\n"
            "── 3C (Content Quality) ──\n"
            "1. CORRECTNESS: Check if ALL facts in A0 exist in the graph evidence. "
            "Remove any fact not traceable to the JSON evidence.\n"
            "2. COMPLETENESS: Ensure all relevant graph facts that answer the user's question are included. "
            "Add any important graph evidence that A0 missed.\n"
            "3. CONSISTENCY: Ensure the reasoning matches the relations in the graph. "
            "Verify chemical mapping relations are used correctly:\n"
            "   • IS_IDENTICAL_TO → 'identical chemical compound' (use confidently)\n"
            "   • IS_LIKELY_EQUIVALENT_TO → 'likely corresponds to' (verify before presenting)\n"
            "   • IS_WEAK_MATCH_TO → 'may correspond to' (mark as speculative)\n\n"
            "── 3H (Communication Quality) ──\n"
            "4. HONESTY: No invented knowledge. Every claim must trace to graph evidence. "
            "Remove any hallucinated entities, studies, or mechanisms.\n"
            "5. HUMILITY: Admit when the graph lacks information. "
            "Do not fill gaps with pretrained knowledge. Use hedging language for uncertain mappings.\n"
            "6. HELPFULNESS: Ensure the answer is clear, well-structured, and directly useful to the user. "
            "Use bullet lists for ingredients or compounds. Keep it concise.\n\n"
            "════ VIOLATION HANDLING ════\n"
            "If any 3C3H violation is detected, REWRITE the answer so it fully aligns with graph evidence. "
            "Do NOT just flag issues — produce a corrected answer.\n\n"
            "════ ADDITIONAL RULES ════\n"
            "• If the draft answer says \"I could not find this information in the knowledge graph.\", "
            "return exactly that same sentence unchanged.\n"
            "• Add uncertainty language for WEAK links (e.g., 'tentatively', 'may').\n"
            "• Add a concise medical safety disclaimer recommending a healthcare professional.\n"
            "• Ensure respectful, non-exclusivist Hadith framing.\n"
            "• Do NOT add unsupported claims or miracle/guarantee/divine-cure language.\n\n"
            "FORMAT:\n"
            "- Strip ALL step numbers, headings ('Step 1:', '## Step 2:', etc.), and chain-of-thought scaffolding.\n"
            "- Output a clean, conversational, well-structured answer suitable for a patient or end user.\n"
            "- Use bullet points or numbered lists only for ingredient/drug listings, not for reasoning steps.\n"
            "- Do NOT mention internal systems such as 'knowledge graph', 'graph retrieval', or similar wording "
            "unless the user explicitly asks for method/source details.\n"
            "- Unless the user explicitly asks for technical details or mechanism, remove mapping labels "
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

    @staticmethod
    def _has_graph_grounding_evidence(reasoning: dict | None) -> bool:
        if not isinstance(reasoning, dict):
            return False

        if reasoning.get("error"):
            return False

        meta = reasoning.get("meta")
        if isinstance(meta, dict) and meta.get("has_error"):
            return False

        evidence_keys = [
            "Ingredients",
            "Diseases",
            "ChemicalCompounds",
            "DrugChemicalCompounds",
            "Drugs",
            "HadithReferences",
            "BiochemicalMappings",
            "Relations",
        ]

        for key in evidence_keys:
            value = reasoning.get(key)
            if isinstance(value, list) and len(value) > 0:
                return True

        return False

    # ── Co-reference resolution ──────────────────────────────────────────

    async def _resolve_coreferences(self, query: str, history: list[dict]) -> str:
        """Rewrite *query* into a self-contained sentence using conversation history.

        If the query already looks self-contained the LLM returns it unchanged.
        Only the last 6 exchanges are considered to limit token use.
        """
        pronoun_pattern = re.compile(r"\b(it|this|that|they|them)\b", re.IGNORECASE)
        if not pronoun_pattern.search(query or ""):
            return query

        original_entities = self._extract_entities(query)
        protected_tokens = [
            value.strip()
            for value in (
                original_entities.get("disease"),
                original_entities.get("ingredient"),
                original_entities.get("drug"),
            )
            if isinstance(value, str) and value.strip()
        ]

        recent = history[-6:]
        convo_lines = "\n".join(
            f"{'User' if m.get('role') == 'user' else 'Assistant'}: {m.get('content', '')[:300]}"
            for m in recent
        )

        system_prompt = (
            "You are a query rewriter. Given a conversation history and the "
            "latest user message, rewrite the user message so it is fully "
            "self-contained by resolving only pronouns: it, this, that, they, them.\n"
            "STRICT ENTITY PRESERVATION RULE: named entities must NEVER be rewritten.\n"
            "Ingredient names, disease names, and drug names must remain unchanged.\n"
            "If the latest user message already contains a specific entity name, keep the exact token unchanged.\n"
            "Forbidden rewrites include examples like: heena→honey, zamzam→water, black seed→cumin.\n"
            "Do NOT paraphrase, normalize, translate, or substitute named entities.\n"
            "If no pronoun resolution is needed, return the latest user message unchanged.\n"
            "Keep the rewrite short — one sentence.\n"
            "Output ONLY the rewritten query, nothing else."
        )
        user_prompt = (
            f"Conversation history:\n{convo_lines}\n\n"
            f"Protected entity tokens (must stay exact): {protected_tokens}\n\n"
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

            # Hard guard: if an originally-detected entity token disappears, keep original query.
            lowered_rewritten = rewritten.lower()
            for token in protected_tokens:
                if token.lower() not in lowered_rewritten:
                    logger.warning(
                        "Coreference rewrite rejected: protected token '%s' missing in rewritten query",
                        token,
                    )
                    return query

            # Hard guard: if detected entities change category values, keep original query.
            rewritten_entities = self._extract_entities(rewritten)
            for key in ("disease", "ingredient", "drug"):
                original_value = original_entities.get(key)
                rewritten_value = rewritten_entities.get(key)

                if isinstance(original_value, str) and original_value.strip():
                    if not (isinstance(rewritten_value, str) and rewritten_value.strip()):
                        logger.warning(
                            "Coreference rewrite rejected: detected '%s' entity was dropped (%s)",
                            key,
                            original_value,
                        )
                        return query
                    if original_value.strip().lower() != rewritten_value.strip().lower():
                        logger.warning(
                            "Coreference rewrite rejected: detected '%s' entity changed '%s' -> '%s'",
                            key,
                            original_value,
                            rewritten_value,
                        )
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
            "Choose exactly one label: hadith, ingredient, technical, mechanism, chemical, drug_substitute, general, auto.\n"
            "Definitions:\n"
            "- hadith: asks for hadith/sunnah/reference/citation support\n"
            "- ingredient: asks what food/herb/remedy to use, or 'what cures X'\n"
            "- technical: asks for compounds/mappings/reasoning details\n"
            "- mechanism: asks 'how does X cure Y', 'how does it work', 'what is the mechanism'\n"
            "- chemical: asks about chemical compounds in an ingredient (e.g. 'what chemicals are in X')\n"
            "- drug_substitute: asks for natural alternatives to a drug (e.g. 'I don't have paracetamol, what food can I use')\n"
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
            if label in {"hadith", "ingredient", "technical", "mechanism", "chemical", "drug_substitute", "general", "auto"}:
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
                "Emphasize prophetic remedies and hadith-grounded guidance respectfully. "
                "Multi-hop reasoning is NOT required for this intent — return a direct answer with ingredient lists."
            )
        if intent == "reasoning":
            return (
                "Allowed scope only: Ingredient → ChemicalCompound → DrugChemicalCompound (if mapped). "
                "Use CONTAINS relationship properties (source, quantity) when present in evidence. "
                "Emphasize chemical reasoning using the 3-tier relation system (IDENTICAL / LIKELY / WEAK). "
                "Multi-hop reasoning IS required: build the full Ingredient → Compound → DrugCompound chain. "
                "Do not use Drug nodes unless the compound maps to a DrugChemicalCompound."
            )
        if intent == "drug":
            return (
                "Allowed scope only: Ingredient → ChemicalCompound → DrugChemicalCompound → Drug (+Book if present). "
                "Emphasize modern drug references and book context when available. "
                "Multi-hop reasoning IS required: trace the full Drug ↔ DrugChemicalCompound ↔ ChemicalCompound ↔ Ingredient chain. "
                "Use the 3-tier relation system (IDENTICAL / LIKELY / WEAK) to qualify mappings. "
                "Do not prioritize Hadith unless explicitly requested by user intent."
            )
        return (
            "Full traversal scope is allowed. Use only provided evidence and do not hallucinate missing graph edges. "
            "Apply multi-hop reasoning only when the query requires mechanism explanation, drug alternatives, "
            "or compound mapping. For simple listing queries, return a direct answer."
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
