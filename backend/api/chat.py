"""Chat API endpoints.

Pipeline flow:
  User Query → Entity Resolution (LLM+Fuzzy) → Intent Classification
  → Query Selection → Neo4j Execution → LLM Answer Generation
  → Judge Evaluation → Response
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter

from backend.core.config import get_settings
from backend.core.models import ChatRequest
from backend.services import (
    graph_service,
    judge_service,
    llm_service,
    query_router,
    response_builder,
)
from backend.services.entity_resolver import resolve_entities

logger = logging.getLogger(__name__)

router = APIRouter()


async def _run_pipeline(query: str, history: list[dict], debug: bool = False, strict_mode: bool = False) -> dict[str, Any]:
    """Execute the full GraphRAG pipeline and return the response dict."""
    logger.info("Pipeline strict_mode=%s", strict_mode)
    settings = get_settings()

    # ── Step 1: Hybrid entity resolution (LLM + fuzzy DB matching) ───────
    entities = resolve_entities(query)
    logger.info("Resolved entities: disease=%s ingredient=%s drug=%s",
                entities.get("disease"), entities.get("ingredient"), entities.get("drug"))

    # ── Step 2: Classify intent (keyword-based, entity-aware) ────────────
    intent = query_router.classify_intent(query, entities)
    logger.info("Intent: %s", intent)

    # ── Step 3: Route to predefined query ────────────────────────────────
    query_id, params, resolved_intent = query_router.route_query(intent, entities)

    if query_id is None:
        if strict_mode:
            no_data_answer = (
                "I could not find relevant information for this query in the knowledge graph database. "
                "Strict mode is enabled, so I cannot supplement with general knowledge. "
                "Please try rephrasing your question or ask about a specific disease, ingredient, or drug."
            )
            return response_builder.build_response(
                user_query=query,
                final_answer=no_data_answer,
                intent=resolved_intent,
                db_result={"rows": [], "row_count": 0, "query_name": "none", "duration_ms": 0},
                llm_result={"answer": no_data_answer},
                entities=entities,
                debug=debug,
            )
        # No suitable query — answer without graph evidence
        llm_result = llm_service.generate_answer(
            user_query=query,
            db_results=[],
            intent=resolved_intent,
            query_name="none",
            strict_mode=strict_mode,
        )
        return response_builder.build_response(
            user_query=query,
            final_answer=llm_result.get("answer", "I could not find relevant information."),
            intent=resolved_intent,
            db_result={"rows": [], "row_count": 0, "query_name": "none", "duration_ms": 0},
            llm_result=llm_result,
            entities=entities,
            debug=debug,
        )

    # ── Step 4: Execute query against Neo4j ──────────────────────────────
    db_result = graph_service.execute_query(query_id, params)

    if db_result.get("error"):
        logger.error("Graph query error: %s", db_result["error"])

    rows = db_result.get("rows", [])

    # ── Step 5: Generate answer (single LLM call) ────────────────────────
    if strict_mode and not rows:
        final_answer = (
            "I could not find relevant information for this query in the knowledge graph database. "
            "Strict mode is enabled, so I cannot supplement with general knowledge. "
            "Please try rephrasing your question or ask about a specific disease, ingredient, or drug."
        )
        llm_result = {"answer": final_answer}
    else:
        llm_result = llm_service.generate_answer(
            user_query=query,
            db_results=rows,
            intent=resolved_intent,
            query_name=db_result.get("query_name", ""),
            strict_mode=strict_mode,
        )
        final_answer = llm_result.get("answer", "")
        if not final_answer.strip():
            final_answer = "PRO-MedGraph could not generate an answer. Please try rephrasing your question."

    # ── Step 6: Judge evaluation ─────────────────────────────────────────
    judge_report: dict[str, Any] | None = None
    if settings.enable_judge and rows:
        try:
            # Try 3C3H first (LLM-based)
            judge_report = judge_service.evaluate_3c3h(
                user_query=query,
                answer=final_answer,
                evidence=rows,
                llm_call_fn=lambda msgs, **kw: llm_service._call_llm(msgs, **kw),
            )
        except Exception:
            logger.warning("3C3H judge failed, falling back to NLP metrics")
            judge_report = judge_service.evaluate_nlp_metrics(
                answer=final_answer,
                evidence=rows,
            )
    elif rows:
        # Judge disabled — use NLP metrics as lightweight alternative
        judge_report = judge_service.evaluate_nlp_metrics(
            answer=final_answer,
            evidence=rows,
        )

    # ── Step 7: Build response ───────────────────────────────────────────
    return response_builder.build_response(
        user_query=query,
        final_answer=final_answer,
        intent=resolved_intent,
        db_result=db_result,
        llm_result=llm_result,
        entities=entities,
        judge_report=judge_report,
        debug=debug,
    )


@router.post("/api/chat")
async def chat(request: ChatRequest) -> dict:
    """Standard chat endpoint."""
    try:
        return await _run_pipeline(
            request.query,
            [m.model_dump() for m in request.history],
            strict_mode=request.strict_mode,
        )
    except Exception:
        logger.exception("/api/chat failed")
        return response_builder.build_error_response()


@router.post("/api/chat/debug")
async def chat_debug(request: ChatRequest) -> dict:
    """Chat endpoint with full pipeline debug trace."""
    try:
        return await _run_pipeline(
            request.query,
            [m.model_dump() for m in request.history],
            debug=True,
            strict_mode=request.strict_mode,
        )
    except Exception:
        logger.exception("/api/chat/debug failed")
        return response_builder.build_error_response(
            "Pipeline debug failed. Check server logs."
        )
