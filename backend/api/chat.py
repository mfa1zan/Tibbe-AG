"""Chat API endpoints.

Pipeline flow:
  User Query → Entity Resolution (LLM+Fuzzy) → Intent Classification
  → Query Selection → Neo4j Execution → LLM Answer Generation
  → Judge Evaluation → Response
"""

from __future__ import annotations

import logging
import re
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
from backend.services.entity_resolver import get_entity_lists, resolve_entities

logger = logging.getLogger(__name__)

router = APIRouter()

_PRIMARY_SPLIT_RE = re.compile(r"(?:\?+|;+|\n+|\b(?:also|plus|as well as)\b)", re.I)
_AND_SPLIT_RE = re.compile(r"\band\b", re.I)


def _split_query_segments(query: str) -> list[str]:
    """Split a potentially multi-part user query into meaningful segments.

    The splitter is conservative to avoid over-fragmenting normal single questions.
    """
    normalized = " ".join(query.split()).strip()
    if not normalized:
        return []

    primary_parts = [part.strip(" .,!?") for part in _PRIMARY_SPLIT_RE.split(normalized)]
    primary_parts = [part for part in primary_parts if part]

    if len(primary_parts) > 1:
        return primary_parts[:4]

    and_parts = [part.strip(" .,!?") for part in _AND_SPLIT_RE.split(normalized)]
    and_parts = [part for part in and_parts if part]

    # Guardrail: only split on "and" when we get 2-3 substantial parts.
    if 2 <= len(and_parts) <= 3 and all(len(part.split()) >= 3 for part in and_parts):
        return and_parts

    return [normalized]


def _normalize_text(text: str) -> str:
    """Normalize free text for robust containment matching."""
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", text.lower())).strip()


def _detect_diseases_in_query(query: str) -> list[str]:
    """Detect multiple known disease names directly mentioned in a user query."""
    normalized_query = f" {_normalize_text(query)} "
    if not normalized_query.strip():
        return []

    try:
        entity_lists = get_entity_lists()
    except Exception:
        logger.exception("Failed to load entity lists while detecting diseases")
        return []

    diseases = entity_lists.get("Disease", []) if isinstance(entity_lists, dict) else []
    matches: list[str] = []
    for disease in diseases:
        if not isinstance(disease, str) or not disease.strip():
            continue
        normalized_disease = _normalize_text(disease)
        if not normalized_disease:
            continue
        if f" {normalized_disease} " in normalized_query:
            matches.append(disease)

    # Prefer longer names first, then keep original order of first appearance in query text.
    matches = sorted(set(matches), key=lambda d: (-len(d), normalized_query.find(f" {_normalize_text(d)} ")))
    return matches[:4]


def _strip_repeated_disclaimer(answer: str) -> str:
    """Remove trailing medical disclaimer blocks to avoid duplication in merged responses."""
    marker = re.search(r"\bmedical disclaimer\b", answer, re.I)
    if marker:
        return answer[:marker.start()].strip()
    return answer.strip()


def _compose_multi_segment_answer(
    user_query: str,
    routed_segments: list[dict[str, Any]],
    strict_mode: bool,
    history: list[dict[str, str]] | None = None,
) -> tuple[str, dict[str, Any]]:
    """Build a separated answer with one section per segment + a combined guidance section."""
    section_outputs: list[tuple[str, str]] = []

    for result in routed_segments:
        db_result = result["db_result"]
        rows = db_result.get("rows", []) or []
        entities = result.get("entities", {})

        disease = entities.get("disease") if isinstance(entities, dict) else None
        ingredient = entities.get("ingredient") if isinstance(entities, dict) else None
        drug = entities.get("drug") if isinstance(entities, dict) else None

        if isinstance(disease, str) and disease.strip():
            label = disease.strip()
        elif isinstance(ingredient, str) and ingredient.strip():
            label = ingredient.strip()
        elif isinstance(drug, str) and drug.strip():
            label = drug.strip()
        else:
            label = result.get("segment", "this condition")

        single_answer = llm_service.generate_answer(
            user_query=result.get("segment", user_query),
            db_results=rows,
            intent=result.get("intent", "general"),
            query_name=db_result.get("query_name", ""),
            strict_mode=strict_mode,
            history=history,
        ).get("answer", "")

        cleaned = _strip_repeated_disclaimer(single_answer)
        if not cleaned:
            cleaned = "I could not find sufficient condition-specific evidence in the knowledge graph."

        section_outputs.append((label, cleaned))

    section_text = "\n\n".join(
        [f"For {label}:\n{content}" for label, content in section_outputs]
    )

    labels = [label for label, _ in section_outputs]
    both_line = (
        "\n\nIf you are experiencing multiple conditions at the same time, follow the relevant guidance for each condition above, "
        "and prioritize professional clinical evaluation to confirm diagnosis and safe treatment."
    )
    if len(labels) >= 2:
        both_line = (
            f"\n\nIf you are experiencing both {labels[0]} and {labels[1]}, use the condition-specific guidance above for each, "
            "and seek medical advice for an integrated treatment plan."
        )

    final_answer = (
        "Based on the Knowledge Graph evidence, here is a condition-wise response:\n\n"
        f"{section_text}"
        f"{both_line}\n\n"
        "Medical Disclaimer: This information is for educational purposes and is not a substitute for medical diagnosis or treatment."
    )

    return final_answer, {"answer": final_answer, "model": "composed-multi-segment", "duration_ms": 0}


def _run_graph_retrieval_for_segment(segment: str, history: list[dict[str, str]] | None = None) -> dict[str, Any]:
    """Resolve entities, route intent, and execute one graph query for a segment."""
    entities = resolve_entities(segment, history=history)
    intent = query_router.classify_intent(segment, entities)
    query_id, params, resolved_intent = query_router.route_query(intent, entities)

    if query_id is None:
        return {
            "segment": segment,
            "entities": entities,
            "intent": resolved_intent,
            "query_id": None,
            "db_result": {
                "rows": [],
                "row_count": 0,
                "query_name": "none",
                "duration_ms": 0,
                "cypher": "",
            },
        }

    db_result = graph_service.execute_query(query_id, params)
    return {
        "segment": segment,
        "entities": entities,
        "intent": resolved_intent,
        "query_id": query_id,
        "db_result": db_result,
    }


async def _run_pipeline(query: str, history: list[dict], debug: bool = False, strict_mode: bool = False) -> dict[str, Any]:
    """Execute the full GraphRAG pipeline and return the response dict."""
    logger.info("Pipeline strict_mode=%s", strict_mode)
    settings = get_settings()

    normalized_history: list[dict[str, str]] = []
    for msg in history or []:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        content = msg.get("content")
        if not isinstance(content, str) or not content.strip():
            continue
        if role == "bot":
            role = "assistant"
        if role not in {"user", "assistant"}:
            continue
        normalized_history.append({"role": role, "content": content})

    trimmed_history = normalized_history[-6:]

    # ── Multi-segment handling for combined prompts ─────────────────────
    segments = _split_query_segments(query)
    if len(segments) == 1:
        detected_diseases = _detect_diseases_in_query(query)
        if len(detected_diseases) >= 2:
            segments = [f"what helps with {disease}" for disease in detected_diseases]

    if len(segments) > 1:
        logger.info("Detected %d query segments: %s", len(segments), segments)

        segment_results = [_run_graph_retrieval_for_segment(segment, history=trimmed_history) for segment in segments]
        routed_segments = [res for res in segment_results if res.get("query_id")]

        # Only switch to multi-query mode when at least 2 segments were routed.
        if len(routed_segments) >= 2:
            combined_rows: list[dict[str, Any]] = []
            query_names: list[str] = []
            cyphers: list[str] = []
            total_duration_ms = 0.0

            for result in routed_segments:
                db_result = result["db_result"]
                rows = db_result.get("rows", []) or []
                for row in rows:
                    merged_row = dict(row)
                    merged_row["_segment"] = result["segment"]
                    combined_rows.append(merged_row)

                query_name = db_result.get("query_name")
                if isinstance(query_name, str) and query_name:
                    query_names.append(query_name)

                cypher = db_result.get("cypher")
                if isinstance(cypher, str) and cypher:
                    cyphers.append(cypher)

                total_duration_ms += float(db_result.get("duration_ms", 0) or 0)

            merged_db_result = {
                "rows": combined_rows,
                "row_count": len(combined_rows),
                "query_name": f"multi[{', '.join(dict.fromkeys(query_names))}]" if query_names else "multi",
                "duration_ms": round(total_duration_ms, 1),
                "cypher": "\n\n".join(dict.fromkeys(cyphers)),
            }

            intents = [res.get("intent", "general") for res in routed_segments]
            merged_intent = intents[0] if intents and all(i == intents[0] for i in intents) else "multi_query"

            entities = resolve_entities(query, history=trimmed_history)

            if strict_mode and not combined_rows:
                final_answer = (
                    "I could not find relevant information for this query in the knowledge graph database. "
                    "Strict mode is enabled, so I cannot supplement with general knowledge. "
                    "Please try rephrasing your question or ask about a specific disease, ingredient, or drug."
                )
                llm_result = {"answer": final_answer}
            else:
                final_answer, llm_result = _compose_multi_segment_answer(
                    user_query=query,
                    routed_segments=routed_segments,
                    strict_mode=strict_mode,
                    history=trimmed_history,
                )
                if not final_answer:
                    final_answer = "PRO-MedGraph could not generate an answer. Please try rephrasing your question."

            judge_report: dict[str, Any] | None = None
            if settings.enable_judge and combined_rows:
                try:
                    judge_report = judge_service.evaluate_3c3h(
                        user_query=query,
                        answer=final_answer,
                        evidence=combined_rows,
                        llm_call_fn=lambda msgs, **kw: llm_service._call_llm(
                            msgs, model=settings.model_intent, **kw
                        ),
                    )
                except Exception:
                    logger.warning("3C3H judge failed for multi-segment query, falling back to NLP metrics")
                    judge_report = judge_service.evaluate_nlp_metrics(
                        answer=final_answer,
                        evidence=combined_rows,
                    )
            elif combined_rows:
                judge_report = judge_service.evaluate_nlp_metrics(
                    answer=final_answer,
                    evidence=combined_rows,
                )

            return response_builder.build_response(
                user_query=query,
                final_answer=final_answer,
                intent=merged_intent,
                db_result=merged_db_result,
                llm_result=llm_result,
                entities=entities,
                judge_report=judge_report,
                debug=debug,
            )

    # ── Step 1: Hybrid entity resolution (LLM + fuzzy DB matching) ───────
    entities = resolve_entities(query, history=trimmed_history)
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
            history=trimmed_history,
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
            history=trimmed_history,
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
                llm_call_fn=lambda msgs, **kw: llm_service._call_llm(
                    msgs, model=settings.model_intent, **kw
                ),
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
