from __future__ import annotations

import asyncio
import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable

from app.config import get_settings
from app.services import entity_service, graph_service, reasoning_builder
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

    Stages are dependency-injected to keep future replacement of entity extraction,
    graph retrieval, reasoning builder, or LLM provider straightforward.
    """

    def __init__(
        self,
        llm_service: LLMService | None = None,
        entity_extractor: Callable[[str], dict] | None = None,
        subgraph_fetcher: Callable[[str], dict] | None = None,
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

        self._entity_extractor = entity_extractor or entity_service.extract_entities
        self._subgraph_fetcher = subgraph_fetcher or graph_service.get_disease_subgraph
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
        disease = entities.get("disease")
        if not disease:
            logger.info("No disease entity detected; routing query to general conversational response")
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

        subgraph = self._retrieve_graph(disease)
        reasoning = self._build_reasoning(subgraph=subgraph, disease=disease)

        # Step-level structured logging for operational observability.
        graph_paths_used = len(reasoning.get("BiochemicalMappings", []))
        logger.info("Reasoning built: graph_paths_used=%s", graph_paths_used)

        # Deterministic KG answer path: if user explicitly asks for ingredients,
        # return ingredients directly from graph reasoning instead of relying on
        # LLM generation that can fail for large contexts.
        if self._is_ingredient_query(clean_query):
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
        return {
            "query": clean_query,
            "entities": entities,
            "reasoning": reasoning,
            "a0_answer": a0_answer,
            "af_answer": af_answer,
            "output": output,
        }

    def _extract_entities(self, query: str) -> dict:
        """Stage 1: Extract entities from user query using hybrid extractor."""
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

    def _retrieve_graph(self, disease: str) -> dict:
        """
        Stage 2: Retrieve disease subgraph and optionally augment with hadith refs.
        Only structured dictionaries/lists are returned.
        """
        try:
            subgraph = self._subgraph_fetcher(disease) or {}
            hadith_refs = self._hadith_fetcher(disease) or []

            # Merge hadith in normalized key expected by reasoning builder.
            subgraph["HadithReferences"] = hadith_refs

            logger.info(
                "Subgraph retrieved disease=%s ingredients=%s compounds=%s drug_compounds=%s drugs=%s hadith=%s",
                disease,
                len(subgraph.get("Ingredients", []) or []),
                len(subgraph.get("ChemicalCompounds", []) or []),
                len(subgraph.get("DrugChemicalCompounds", []) or []),
                len(subgraph.get("Drugs", []) or []),
                len(hadith_refs),
            )
            return subgraph
        except Exception:
            logger.exception("Graph retrieval stage failed")
            return {
                "error": "Graph retrieval failed",
                "Disease": {"id": None, "name": disease, "category": None},
                "Ingredients": [],
                "ChemicalCompounds": [],
                "DrugChemicalCompounds": [],
                "Drugs": [],
                "Relations": [],
                "HadithReferences": [],
            }

    def _build_reasoning(self, subgraph: dict, disease: str) -> dict:
        """Stage 3: Convert graph payload into strict reasoning structure for LLM."""
        try:
            reasoning = self._reasoning_formatter(subgraph) or {}
            # Ensure minimum schema exists for downstream prompt formatting.
            reasoning.setdefault("Disease", {"id": None, "name": disease, "category": None})
            reasoning.setdefault("Ingredients", [])
            reasoning.setdefault("ChemicalCompounds", [])
            reasoning.setdefault("DrugChemicalCompounds", [])
            reasoning.setdefault("Drugs", [])
            reasoning.setdefault("HadithReferences", [])
            reasoning.setdefault("BiochemicalMappings", [])
            logger.info("Reasoning formatter stage completed")
            return reasoning
        except Exception:
            logger.exception("Reasoning formatter stage failed")
            return {
                "Disease": {"id": None, "name": disease, "category": None},
                "Ingredients": [],
                "ChemicalCompounds": [],
                "DrugChemicalCompounds": [],
                "Drugs": [],
                "HadithReferences": [],
                "BiochemicalMappings": [],
                "meta": {"has_error": True, "source_error": "Reasoning formatting failed"},
            }

    async def _generate_a0_answer(self, query: str, reasoning: dict) -> str:
        """
        Stage 4 (A0): Generate a base grounded answer from structured graph reasoning.
        """
        system_prompt = (
            "You are PRO-MedGraph, a biomedical and faith-aligned assistant.\n"
            "Use ONLY the provided structured graph reasoning evidence.\n"
            "Do NOT hallucinate entities, studies, or claims.\n"
            "Explain the likely biochemical mechanism step-by-step when evidence exists.\n"
            "Cite relevant Hadith references present in the evidence.\n"
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

    def _build_ingredient_answer(self, reasoning: dict, disease_name: str) -> str | None:
        """Build a direct ingredient list answer from graph_service + reasoning data."""
        if not isinstance(reasoning, dict):
            return None

        ingredients = reasoning.get("Ingredients", [])
        if not isinstance(ingredients, list):
            ingredients = []

        # Prefer complete direct disease->ingredient retrieval from graph_service,
        # then merge with reasoning ingredients as a defensive fallback.
        direct_rows = self._ingredient_fetcher(disease_name)
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
            return None

        bullet_lines = "\n".join(f"{index + 1}. {name}" for index, name in enumerate(deduped_names[:25]))
        return (
            f"Based on the knowledge graph, the ingredients associated with {disease_name} are:\n\n"
            f"{bullet_lines}"
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