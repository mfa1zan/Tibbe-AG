"""
PRO-MedGraph GraphRAG Orchestrator — Research-Grade Pipeline (v2).

Implements the full 13-stage pipeline:

    1.  General Check           – bypass KG for greetings
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

    async def process_user_query_with_context_async(self, query: str) -> dict:
        """Extended 13-stage pipeline returning full context + reasoning trace."""
        clean_query = (query or "").strip()
        pipeline_stages: list[str] = []

        if not clean_query:
            fallback = self._fallback_response("Empty query")
            return self._wrap_context(clean_query, {}, {}, "", fallback["final_answer"], fallback, pipeline_stages)

        # ── Stage 1: General Query Check ───────────────────────────────────
        pipeline_stages.append("general_check")

        if self._is_general_query(clean_query):
            logger.info("General conversational query detected; bypassing KG pipeline")
            general_answer = await self._generate_general_answer(clean_query)
            output = {
                "final_answer": general_answer,
                "evidence_strength": "weak",
                "graph_paths_used": 0,
                "confidence_score": None,
            }
            pipeline_stages.append("general_response")
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

        # No biomedical entities -> general response
        if not disease and not ingredient and not drug:
            logger.info("No biomedical entities detected; routing to general response")
            general_answer = await self._generate_general_answer(clean_query)
            output = {
                "final_answer": general_answer,
                "evidence_strength": "weak",
                "graph_paths_used": 0,
                "confidence_score": None,
            }
            pipeline_stages.append("general_response_no_entities")
            return self._wrap_context(
                clean_query, entities,
                {"meta": {"kg_applicable": False}},
                "", general_answer, output, pipeline_stages,
                kg_applicable=False,
            )

        # ── Stage 3: Rationale Plan (RAG2 style) ──────────────────────────
        pipeline_stages.append("rationale_plan")
        rationale_result = await generate_rationale_plan(
            query=clean_query,
            entities=entities,
            llm_service=self._llm_service,
            model=self._a0_model,
        )

        # ── Stage 4: Intent Routing ────────────────────────────────────────
        pipeline_stages.append("intent_routing")
        subgraph, primary_entity_type, primary_entity_name = self._retrieve_graph_by_intent(
            disease=disease, ingredient=ingredient, drug=drug,
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

        # ── Deterministic ingredient shortcut ──────────────────────────────
        if self._is_ingredient_query(clean_query) and disease:
            direct_answer = self._build_ingredient_answer(reasoning, disease)
            if direct_answer:
                pipeline_stages.append("deterministic_ingredient_answer")
                evidence_strength = self._compute_evidence_strength(reasoning)
                confidence_score = self._compute_confidence_score(evidence_strength, graph_paths_used)
                output = {
                    "final_answer": direct_answer,
                    "evidence_strength": evidence_strength,
                    "graph_paths_used": graph_paths_used,
                    "confidence_score": confidence_score,
                }
                return self._wrap_context(
                    clean_query, entities, reasoning,
                    direct_answer, direct_answer, output, pipeline_stages,
                    rationale_plan=rationale_result,
                )

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
        pipeline_stages.append("a0_generation")
        a0_answer = await self._generate_a0_answer(
            clean_query, reasoning,
            causal_analysis=causal_analysis,
            dosage_validation=dosage_result.to_dict(),
        )

        # ── Stage 11: Af Validation ───────────────────────────────────────
        pipeline_stages.append("af_validation")
        af_answer = await self._validate_answer(clean_query, a0_answer, reasoning)

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

        reasoning_trace = {
            "entity_detected": entities,
            "rationale_plan": rationale_result.get("rationale_plan"),
            "retrieved_paths": causal_analysis.get("causal_paths", [])[:10],
            "causal_ranking": causal_analysis.get("causal_paths", [])[:5],
            "causal_summary": causal_summary,
            "dosage_validation": dosage_result.to_dict(),
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
    ) -> tuple[dict, str, str]:
        """Stage 4+5: Intent-based routing + KG retrieval."""
        try:
            if disease:
                logger.info("Intent routing: DISEASE query for '%s'", disease)
                subgraph = self._disease_subgraph_fetcher(disease) or {}
                hadith_refs = self._hadith_fetcher(disease) or []
                subgraph["HadithReferences"] = hadith_refs
                logger.info(
                    "Disease subgraph: ingredients=%d compounds=%d drugs=%d hadith=%d",
                    len(subgraph.get("Ingredients", []) or []),
                    len(subgraph.get("ChemicalCompounds", []) or []),
                    len(subgraph.get("Drugs", []) or []),
                    len(hadith_refs),
                )
                return subgraph, "Disease", disease

            if ingredient:
                logger.info("Intent routing: INGREDIENT query for '%s'", ingredient)
                subgraph = self._ingredient_subgraph_fetcher(ingredient) or {}
                subgraph.setdefault("HadithReferences", [])
                return subgraph, "Ingredient", ingredient

            if drug:
                logger.info("Intent routing: DRUG query for '%s'", drug)
                subgraph = self._drug_subgraph_fetcher(drug) or {}
                subgraph.setdefault("HadithReferences", [])
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
            reasoning = self._reasoning_formatter(subgraph) or {}

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

        if primary_entity_type == "Disease":
            system_prompt = (
                "You are PRO-MedGraph, a biomedical and faith-aligned assistant.\n"
                "Use ONLY the provided structured graph reasoning evidence.\n"
                "Do NOT hallucinate entities, studies, or claims.\n\n"
                "Follow this EXACT chain-of-thought structure:\n"
                "Step 1: Identify the disease from the query\n"
                "Step 2: List all traditional/natural Ingredients linked via CURES relationship\n"
                "Step 3: For each ingredient, list the ChemicalCompounds it CONTAINS\n"
                "Step 4: Match each compound to Drug equivalents via IS_IDENTICAL_TO / IS_LIKELY_EQUIVALENT_TO / IS_WEAK_MATCH_TO\n"
                "Step 5: Explain the biochemical mechanism connecting ingredients to drugs step-by-step\n"
                "Step 6: Include any Hadith references with respectful, non-exclusivist framing\n"
                "Step 7: Clearly state the uncertainty level and mapping strength for each link\n\n"
                "Label mapping strengths: IDENTICAL = confirmed, LIKELY = probable, WEAK = tentative.\n"
                "If no drug equivalents exist, explicitly state that.\n"
                "If causal scores are provided, mention the strongest causal paths.\n"
                "If dosage data is available, include dosage comparison notes.\n"
                "Always recommend consulting a healthcare professional."
            )
        elif primary_entity_type == "Ingredient":
            system_prompt = (
                "You are PRO-MedGraph, a biomedical and faith-aligned assistant.\n"
                "Use ONLY the provided structured graph reasoning evidence.\n"
                "Do NOT hallucinate entities, studies, or claims.\n\n"
                "For this ingredient query, follow this reasoning chain:\n"
                "1. Identify the ingredient\n"
                "2. List diseases it cures (via CURES)\n"
                "3. List chemical compounds it contains\n"
                "4. Map compounds to drug equivalents with strengths\n"
                "5. Include Hadith references if present\n"
                "6. State uncertainty level\n"
                "Always recommend consulting a healthcare professional."
            )
        elif primary_entity_type == "Drug":
            system_prompt = (
                "You are PRO-MedGraph, a biomedical and faith-aligned assistant.\n"
                "Use ONLY the provided structured graph reasoning evidence.\n"
                "Do NOT hallucinate entities, studies, or claims.\n\n"
                "For this drug query, follow this reasoning chain:\n"
                "1. Identify the drug\n"
                "2. List chemical compounds it contains\n"
                "3. Map compounds back to natural ingredients with strengths\n"
                "4. List diseases those ingredients can treat\n"
                "5. Include Hadith references if present\n"
                "6. State uncertainty level\n"
                "Always recommend consulting a healthcare professional."
            )
        else:
            system_prompt = (
                "You are PRO-MedGraph, a biomedical and faith-aligned assistant.\n"
                "Use ONLY the provided structured graph reasoning evidence.\n"
                "Do NOT hallucinate. State uncertainty when evidence is weak.\n"
                "Always recommend consulting a healthcare professional."
            )

        # Build enriched user prompt with causal + dosage context
        context_parts = [
            f"User Query:\n{query}\n",
            f"Structured Graph Reasoning (JSON):\n{json.dumps(reasoning, ensure_ascii=False, indent=2)}",
        ]

        if causal_analysis and causal_analysis.get("causal_paths"):
            top_paths = causal_analysis["causal_paths"][:5]
            context_parts.append(
                f"\nTop Causal Paths (ranked by causal_score):\n"
                f"{json.dumps(top_paths, ensure_ascii=False, indent=2)}"
            )

        if dosage_validation and dosage_validation.get("comparisons"):
            context_parts.append(
                f"\nDosage Comparison Data:\n"
                f"{json.dumps(dosage_validation, ensure_ascii=False, indent=2)}"
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
        except Exception:
            logger.exception("A0 generation failed")
            return (
                "I could not generate a complete graph-grounded draft answer. "
                "Available evidence appears limited, so please refine the query."
            )

    async def _validate_answer(self, query: str, a0_answer: str, reasoning: dict) -> str:
        """Stage 11: Af validation with safety and uncertainty controls."""
        system_prompt = (
            "You are PRO-MedGraph Validator (Af).\n"
            "Validate the draft answer against the provided graph reasoning only.\n"
            "Add uncertainty language for WEAK links.\n"
            "Add concise medical safety disclaimers where needed.\n"
            "Ensure faith alignment and respectful Hadith framing.\n"
            "Ensure the answer recommends consulting a healthcare professional.\n"
            "Do NOT add unsupported claims.\n"
            "Do NOT use miracle/guarantee/divine-cure language.\n"
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
                temperature=0.1,
                model=self._af_model,
            )
        except Exception:
            logger.exception("Af validation failed")
            return a0_answer

    async def _generate_general_answer(self, query: str) -> str:
        """Generate conversational response for non-biomedical queries."""
        system_prompt = (
            "You are PRO-MedGraph, a helpful assistant.\n"
            "Respond naturally and briefly to general user messages.\n"
            "If the user asks a medical question without specific disease details,\n"
            "ask a clarifying follow-up and avoid definitive medical claims."
        )
        try:
            return await self._llm_service.generate_completion(
                system_prompt=system_prompt,
                user_prompt=f"User message:\n{query}",
                temperature=0.3,
                model=self._a0_model,
            )
        except Exception:
            logger.exception("General response failed")
            return (
                "Hello! I am PRO-MedGraph. I can help with biomedical questions. "
                "If you share a disease or treatment topic, I can provide a graph-grounded answer."
            )

    # ── Utility methods ────────────────────────────────────────────────────

    @staticmethod
    def _is_general_query(query: str) -> bool:
        normalized = query.strip().lower()
        if not normalized:
            return True
        tokens = normalized.split()
        if len(tokens) == 1 and len(tokens[0]) <= 8:
            if _GENERAL_QUERY_PATTERN.match(normalized):
                return True
        return bool(_GENERAL_QUERY_PATTERN.match(normalized))

    @staticmethod
    def _is_ingredient_query(query: str) -> bool:
        return bool(_INGREDIENT_QUERY_PATTERN.search(query or ""))

    @staticmethod
    def _is_drug_query(query: str) -> bool:
        return bool(_DRUG_QUERY_PATTERN.search(query or ""))

    def _build_ingredient_answer(self, reasoning: dict, disease_name: str) -> str | None:
        """Build direct ingredient list from KG data."""
        if not isinstance(reasoning, dict):
            return None

        ingredients = reasoning.get("Ingredients", [])
        if not isinstance(ingredients, list):
            ingredients = []

        try:
            direct_rows = self._ingredient_fetcher(disease_name)
        except Exception:
            direct_rows = []

        direct_names = [
            item.get("name").strip()
            for item in direct_rows
            if isinstance(item, dict) and isinstance(item.get("name"), str) and item.get("name").strip()
        ]
        reasoning_names = [
            item.get("name").strip()
            for item in ingredients
            if isinstance(item, dict) and isinstance(item.get("name"), str) and item.get("name").strip()
        ]

        deduped = list(dict.fromkeys([*direct_names, *reasoning_names]))
        if not deduped:
            return None

        bullet_lines = "\n".join(f"{i + 1}. {name}" for i, name in enumerate(deduped[:25]))
        suffix = "" if len(deduped) <= 25 else f"\n\n(Showing top 25 of {len(deduped)} ingredients)"
        return (
            f"Based on the knowledge graph, the ingredients associated with {disease_name} are:\n\n"
            f"{bullet_lines}{suffix}"
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
