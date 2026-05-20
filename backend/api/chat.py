"""Chat API endpoints.

Pipeline flow:
  User Query → Entity Resolution (LLM+Fuzzy) → Intent Classification
  → Query Selection → Neo4j Execution → LLM Answer Generation
  → Judge Evaluation → Response
"""

from __future__ import annotations

import asyncio
import logging
import re
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from fastapi import APIRouter, Body, Depends, HTTPException, status
from sqlalchemy.orm import Session

from backend.core.config import get_settings
from backend.core.dependencies import get_current_user
from backend.core.models import ChatRequest
from backend.db.database import get_db
from backend.db.models import ChatSession, ChatMessage, User
from backend.services import (
    graph_service,
    judge_service,
    llm_service,
    query_router,
    response_builder,
)
from backend.services.llm_service import (
    reset_api_call_count,
    get_api_call_count,
    set_current_step,
    clear_current_step,
)
from backend.services.entity_resolver import get_entity_lists, resolve_entities

logger = logging.getLogger(__name__)

router = APIRouter()

_PRIMARY_SPLIT_RE = re.compile(r"(?:\?+|;+|\n+|\b(?:also|plus|as well as)\b)", re.I)
_AND_SPLIT_RE = re.compile(r"\band\b", re.I)

_STRICT_NO_DATA_MESSAGE = (
    "I could not find relevant information for this query in the knowledge graph database. "
    "Strict mode is enabled, so I cannot supplement with general knowledge. "
    "Please try rephrasing your question or ask about a specific disease, ingredient, or drug."
)

_PRIMARY_SPLIT_RE = re.compile(r"(?:\?+|;+|\n+|\b(?:also|plus|as well as)\b)", re.I)
_AND_SPLIT_RE = re.compile(r"\band\b", re.I)

_STRICT_NO_DATA_MESSAGE = (
    "I could not find relevant information for this query in the knowledge graph database. "
    "Strict mode is enabled, so I cannot supplement with general knowledge. "
    "Please try rephrasing your question or ask about a specific disease, ingredient, or drug."
)


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


def _find_known_entity_in_text(text: str, candidates: list[str]) -> str | None:
    """Return the first matching entity name found in text, preferring longer matches."""
    normalized_text = f" {_normalize_text(text)} "
    if not normalized_text.strip():
        return None

    matches: list[str] = []
    for candidate in candidates or []:
        if not isinstance(candidate, str) or not candidate.strip():
            continue
        normalized_candidate = _normalize_text(candidate)
        if not normalized_candidate:
            continue
        if f" {normalized_candidate} " in normalized_text:
            matches.append(candidate)

    matches = sorted(set(matches), key=lambda c: (-len(c), normalized_text.find(f" {_normalize_text(c)} ")))
    return matches[0] if matches else None


def _resolve_task_entities(
    sub_question: str,
    base_entities: dict[str, Any],
    entity_names: dict[str, list[str]],
) -> dict[str, str | None]:
    """Resolve per-task entities using known name lists and the task sub-question."""
    task_entities = {
        "disease": base_entities.get("disease") if isinstance(base_entities, dict) else None,
        "ingredient": base_entities.get("ingredient") if isinstance(base_entities, dict) else None,
        "drug": base_entities.get("drug") if isinstance(base_entities, dict) else None,
        "compound": base_entities.get("compound") if isinstance(base_entities, dict) else None,
    }

    if not isinstance(sub_question, str) or not isinstance(entity_names, dict):
        return task_entities

    disease_match = _find_known_entity_in_text(sub_question, entity_names.get("Disease", []))
    if disease_match:
        task_entities["disease"] = disease_match

    ingredient_match = _find_known_entity_in_text(sub_question, entity_names.get("Ingredient", []))
    if ingredient_match:
        task_entities["ingredient"] = ingredient_match

    drug_match = _find_known_entity_in_text(sub_question, entity_names.get("Drug", []))
    if drug_match:
        task_entities["drug"] = drug_match

    compound_match = _find_known_entity_in_text(sub_question, entity_names.get("ChemicalCompound", []))
    if compound_match:
        task_entities["compound"] = compound_match

    return task_entities


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
    """Build a unified answer from all segment evidence using one LLM call."""
    all_rows: list[dict[str, Any]] = []
    query_names: list[str] = []

    for result in routed_segments:
        if not isinstance(result, dict):
            continue

        db_result = result.get("db_result", {})
        if not isinstance(db_result, dict):
            db_result = {}

        rows = db_result.get("rows", []) or []
        segment_label = result.get("segment", "")
        for row in rows:
            merged_row = dict(row)
            merged_row["_segment"] = segment_label
            all_rows.append(merged_row)

        query_name = db_result.get("query_name")
        if isinstance(query_name, str) and query_name.strip():
            query_names.append(query_name.strip())

    unique_query_name = ", ".join(dict.fromkeys(query_names)) if query_names else "multi"

    # Group rows by their evidence type based on fields present
    disease_rows = [r for r in all_rows if "disease" in r and "hadith_text" in r]
    compound_rows = [r for r in all_rows if "compound" in r and "source" in r and "drug" not in r]
    drug_rows = [r for r in all_rows if "drug" in r and "mapping_strength" in r]
    other_rows = [r for r in all_rows if r not in disease_rows and r not in compound_rows and r not in drug_rows]

    # Take a balanced sample from each type — 7 rows each, capped at 20 total
    representative_rows = (
        disease_rows[:7] +
        compound_rows[:7] +
        drug_rows[:7] +
        other_rows[:2]
    )[:20]

    llm_result = llm_service.generate_answer(
        user_query=user_query,
        db_results=representative_rows,
        intent="multi_query",
        query_name=unique_query_name,
        strict_mode=strict_mode,
        history=history,
    )

    final_answer = llm_result.get("answer", "")
    if not isinstance(final_answer, str) or not final_answer.strip():
        final_answer = "PRO-MedGraph could not generate an answer. Please try rephrasing your question."

    return final_answer, llm_result


async def _run_graph_retrieval_for_segment(segment: str, history: list[dict[str, str]] | None = None) -> dict[str, Any]:
    """Resolve entities, route intent, and execute one graph query for a segment."""
    entities = resolve_entities(segment, history=history)
    intent = await query_router.classify_intent_llm(segment, entities)
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

    # ── Step 1: Analyze the whole query upfront ──────────────────────────
    # reset per-request LLM api call counter
    try:
        reset_api_call_count()
    except Exception:
        logger.debug("Could not reset api_call_count contextvar")

    logger.info("Step 1 — Analyzer: starting")
    set_current_step("Step 1 - Analyzer")
    analysis = await query_router.analyze_query_llm(
        query,
        history=trimmed_history,
    )
    clear_current_step()

    entities = analysis.get("entities") if isinstance(analysis, dict) else {}
    if not isinstance(entities, dict):
        entities = {}

    tasks = analysis.get("tasks") if isinstance(analysis, dict) else []
    if not isinstance(tasks, list) or not tasks:
        tasks = [{"intent": query_router.INTENT_GENERAL, "sub_question": query}]
    logger.info(
        "Step 1 — Analyzer: %d tasks detected: %s api_calls=%d",
        len(tasks),
        [t.get("intent") for t in tasks if isinstance(t, dict)],
        get_api_call_count(),
    )

    # ── Step 2: Route all tasks to query IDs ─────────────────────────────
    entity_names = get_entity_lists()
    logger.info("Step 2 — Routing: starting for %d tasks", len(tasks))
    routed_tasks: list[dict[str, Any]] = []
    for task in tasks:
        if not isinstance(task, dict):
            continue
        sub_question = task.get("sub_question") if isinstance(task.get("sub_question"), str) else query
        intent = task.get("intent") if isinstance(task.get("intent"), str) else query_router.INTENT_GENERAL
        sub_question = sub_question.strip() if isinstance(sub_question, str) else query
        intent = intent.strip() if isinstance(intent, str) else query_router.INTENT_GENERAL

        task_entities = _resolve_task_entities(sub_question, entities, entity_names)
        query_id, params, resolved_intent = query_router.route_query(intent, task_entities)
        routed_tasks.append(
            {
                "segment": sub_question,
                "sub_question": sub_question,
                "intent": resolved_intent,
                "query_id": query_id,
                "params": params,
                "entities": task_entities,
            }
        )

    if not routed_tasks:
        routed_tasks.append(
            {
                "segment": query,
                "sub_question": query,
                "intent": query_router.INTENT_GENERAL,
                "query_id": None,
                "params": {},
                "entities": entities,
            }
        )

    # ── Step 3: Execute all DB queries in parallel ───────────────────────
    logger.info("Step 3 — Graph retrieval: executing %d query jobs", len(routed_tasks))
    query_jobs: list[dict[str, Any]] = []
    for idx, task in enumerate(routed_tasks):
        query_id = task.get("query_id")
        if query_id:
            query_jobs.append(
                {
                    "idx": idx,
                    "kind": "primary",
                    "future": asyncio.to_thread(
                        graph_service.execute_query,
                        query_id,
                        task["params"],
                    ),
                }
            )

            if task.get("intent") == "disease_drug":
                task_ents = task.get("entities", {}) if isinstance(task.get("entities"), dict) else {}
                disease = task_ents.get("disease")
                if isinstance(disease, str) and disease.strip():
                    query_jobs.append(
                        {
                            "idx": idx,
                            "kind": "disease_drug_extra",
                            "future": asyncio.to_thread(
                                graph_service.execute_query,
                                "A",
                                {"disease_name": disease},
                            ),
                        }
                    )

    if query_jobs:
        results = await asyncio.gather(*[job["future"] for job in query_jobs])
        extra_results: dict[int, dict[str, Any]] = {}
        for job, db_result in zip(query_jobs, results):
            idx = job["idx"]
            if job["kind"] == "primary":
                routed_tasks[idx]["db_result"] = db_result
            else:
                extra_results[idx] = db_result

        for idx, extra_result in extra_results.items():
            primary_result = routed_tasks[idx].get("db_result")
            if not isinstance(primary_result, dict):
                primary_result = {
                    "rows": [],
                    "row_count": 0,
                    "query_name": "none",
                    "duration_ms": 0,
                    "cypher": "",
                }
                routed_tasks[idx]["db_result"] = primary_result

            rows_primary = primary_result.get("rows", []) or []
            rows_extra = extra_result.get("rows", []) or []
            combined_rows = rows_primary + rows_extra
            primary_result.update({
                "rows": combined_rows,
                "row_count": len(combined_rows),
            })

    for task in routed_tasks:
        if "db_result" not in task:
            task["db_result"] = {
                "rows": [],
                "row_count": 0,
                "query_name": "none",
                "duration_ms": 0,
                "cypher": "",
            }

    # Log DB queries summary
    try:
        for idx, task in enumerate(routed_tasks):
            db = task.get("db_result", {})
            logger.info(
                "Step 3 — Query[%d]: id=%s rows=%d query_name=%s duration_ms=%.1f",
                idx,
                task.get("query_id"),
                int(db.get("row_count", 0) or 0),
                db.get("query_name"),
                float(db.get("duration_ms", 0) or 0),
            )
    except Exception:
        logger.debug("Failed to log DB query summaries")

    # ── Step 4: Single-task flow ─────────────────────────────────────────
    logger.info("Step 4 — Single-task flow check: %d tasks", len(routed_tasks))
    if len(routed_tasks) == 1:
        task = routed_tasks[0]
        resolved_intent = task.get("intent", "general")
        task_entities = task.get("entities", {}) if isinstance(task.get("entities"), dict) else {}
        db_result = task.get("db_result", {})

        if task.get("query_id") is None:
            if strict_mode:
                final_answer = _STRICT_NO_DATA_MESSAGE
                llm_result = {"answer": final_answer, "api_call_count": get_api_call_count()}
            else:
                set_current_step("Step 4 - Answer Generation")
                llm_result = llm_service.generate_answer(
                    user_query=query,
                    db_results=[],
                    intent=resolved_intent,
                    query_name="none",
                    strict_mode=strict_mode,
                    history=trimmed_history,
                )
                clear_current_step()
                final_answer = llm_result.get("answer", "I could not find relevant information.")

            logger.info("Step 4 — Single-task flow completed api_calls=%d", get_api_call_count())

            return response_builder.build_response(
                user_query=query,
                final_answer=final_answer,
                intent=resolved_intent,
                db_result={"rows": [], "row_count": 0, "query_name": "none", "duration_ms": 0},
                llm_result=llm_result,
                entities=task_entities or entities,
                debug=debug,
            )

        # ── Special handling: disease_drug runs two queries ──
        if resolved_intent == "disease_drug" and task_entities.get("disease"):
            db_result_a = await asyncio.to_thread(
                graph_service.execute_query,
                "A",
                {"disease_name": task_entities["disease"]},
            )
            rows_a = db_result_a.get("rows", []) or []
            rows_e = db_result.get("rows", []) or []
            combined = rows_a + rows_e
            db_result = {
                **db_result,
                "rows": combined,
                "row_count": len(combined),
            }

        if db_result.get("error"):
            logger.error("Graph query error: %s", db_result["error"])

        rows = db_result.get("rows", [])

        if strict_mode and not rows:
            final_answer = _STRICT_NO_DATA_MESSAGE
            llm_result = {"answer": final_answer, "api_call_count": get_api_call_count()}
        else:
            set_current_step("Step 4 - Answer Generation")
            llm_result = llm_service.generate_answer(
                user_query=query,
                db_results=rows,
                intent=resolved_intent,
                query_name=db_result.get("query_name", ""),
                strict_mode=strict_mode,
                history=trimmed_history,
            )
            clear_current_step()
            final_answer = llm_result.get("answer", "")
            if not final_answer.strip():
                final_answer = "PRO-MedGraph could not generate an answer. Please try rephrasing your question."
        logger.info("Step 4 — LLM answer generation api_calls=%d", get_api_call_count())

        judge_report: dict[str, Any] | None = None
        if settings.enable_judge and rows:
            try:
                set_current_step("Step 4 - Judge")
                judge_report = judge_service.evaluate_3c3h(
                    user_query=query,
                    answer=final_answer,
                    evidence=rows,
                    llm_call_fn=lambda msgs, **kw: llm_service._call_llm(
                        msgs, model=settings.model_intent, **kw
                    ),
                )
                clear_current_step()
            except Exception:
                logger.warning("3C3H judge failed, falling back to NLP metrics")
                judge_report = judge_service.evaluate_nlp_metrics(
                    answer=final_answer,
                    evidence=rows,
                )
        elif rows:
            judge_report = judge_service.evaluate_nlp_metrics(
                answer=final_answer,
                evidence=rows,
            )

        if (
            judge_report is not None
            and judge_report.get("composite_score") is not None
            and judge_report["composite_score"] < 0.5
            and rows
        ):
            logger.warning(
                "Low confidence score %.2f — overriding answer",
                judge_report["composite_score"],
            )
            final_answer = (
                "The available evidence was insufficient to generate a reliable answer for your query. "
                "Please try rephrasing your question or ask about a specific disease, ingredient, or drug."
            )

        return response_builder.build_response(
            user_query=query,
            final_answer=final_answer,
            intent=resolved_intent,
            db_result=db_result,
            llm_result=llm_result,
            entities=task_entities or entities,
            judge_report=judge_report,
            debug=debug,
        )

    # ── Step 5: Multi-task flow ──────────────────────────────────────────
    logger.info("Step 5 — Multi-task flow: composing %d segments", len(routed_tasks))
    combined_rows: list[dict[str, Any]] = []
    query_names: list[str] = []
    cyphers: list[str] = []
    total_duration_ms = 0.0

    for result in routed_tasks:
        db_result = result["db_result"]
        rows = db_result.get("rows", []) or []
        for row in rows:
            merged_row = dict(row)
            merged_row["_segment"] = result.get("segment", "")
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

    intents = [res.get("intent", "general") for res in routed_tasks]
    merged_intent = intents[0] if intents and all(i == intents[0] for i in intents) else "multi_query"

    if strict_mode and not combined_rows:
        final_answer = _STRICT_NO_DATA_MESSAGE
        llm_result = {"answer": final_answer, "api_call_count": get_api_call_count()}
    else:
        set_current_step("Step 5 - Composition")
        final_answer, llm_result = _compose_multi_segment_answer(
            user_query=query,
            routed_segments=routed_tasks,
            strict_mode=strict_mode,
            history=trimmed_history,
        )
        clear_current_step()
        if not final_answer:
            final_answer = "PRO-MedGraph could not generate an answer. Please try rephrasing your question."

    logger.info("Step 5 — Composition complete api_calls=%d", get_api_call_count())

    judge_report: dict[str, Any] | None = None
    if settings.enable_judge and combined_rows:
        try:
            set_current_step("Step 5 - Judge")
            judge_report = judge_service.evaluate_3c3h(
                user_query=query,
                answer=final_answer,
                evidence=combined_rows,
                llm_call_fn=lambda msgs, **kw: llm_service._call_llm(
                    msgs, model=settings.model_intent, **kw
                ),
            )
            clear_current_step()
        except Exception:
            logger.warning("3C3H judge failed for multi-task query, falling back to NLP metrics")
            judge_report = judge_service.evaluate_nlp_metrics(
                answer=final_answer,
                evidence=combined_rows,
            )
    elif combined_rows:
        judge_report = judge_service.evaluate_nlp_metrics(
            answer=final_answer,
            evidence=combined_rows,
        )

    if (
        judge_report is not None
        and judge_report.get("composite_score") is not None
        and judge_report["composite_score"] < 0.5
        and combined_rows
    ):
        logger.warning(
            "Low confidence score %.2f — overriding answer",
            judge_report["composite_score"],
        )
        final_answer = (
            "The available evidence was insufficient to generate a reliable answer for your query. "
            "Please try rephrasing your question or ask about a specific disease, ingredient, or drug."
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


# ── Session Management Endpoints ─────────────────────────────────────────────

@router.get("/api/sessions")
async def list_sessions(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
) -> list[dict]:
    """List all chat sessions for the current user.
    
    Note: Empty sessions (sessions with no messages) are excluded from the list.
    This ensures that new chats which are created but have no messages sent are not 
    shown in the chat history.
    """
    # Get sessions that have at least one message
    sessions = db.query(ChatSession).filter(
        ChatSession.user_id == current_user.id,
        db.query(ChatMessage).filter(
            ChatMessage.session_id == ChatSession.id
        ).exists()
    ).order_by(
        ChatSession.updated_at.desc()
    ).all()
    
    return [
        {
            "id": session.id,
            "title": session.title,
            "created_at": session.created_at.isoformat(),
            "updated_at": session.updated_at.isoformat(),
        }
        for session in sessions
    ]


@router.post("/api/sessions")
async def create_session(
    payload: dict[str, Any] = Body(default_factory=dict),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
) -> dict:
    """Create a new chat session for the current user."""
    try:
        title = payload.get("title")
        if not title:
            title = "New Chat"
        session = ChatSession(
            user_id=current_user.id,
            title=title,
        )
        db.add(session)
        db.commit()
        db.refresh(session)
        return {
            "id": session.id,
            "title": session.title,
            "created_at": session.created_at.isoformat(),
            "updated_at": session.updated_at.isoformat(),
        }
    except Exception as e:
        logger.exception("Failed to create session")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create session: {str(e)}"
        )


@router.delete("/api/sessions/{session_id}")
async def delete_session(
    session_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Delete a chat session and all its messages."""
    session = db.query(ChatSession).filter(
        ChatSession.id == session_id,
        ChatSession.user_id == current_user.id
    ).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    db.delete(session)
    db.commit()
    return {"message": "Session deleted"}


@router.delete("/api/sessions")
async def delete_empty_sessions(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
) -> dict:
    """Delete all empty sessions (sessions with no messages) for the current user.
    
    This is a cleanup endpoint to remove abandoned sessions that were created
    but never had any messages sent. This can be called periodically or when
    the user logs out.
    """
    from sqlalchemy import func, and_
    
    try:
        # Get all sessions for the user that have no messages
        empty_sessions = db.query(ChatSession).filter(
            ChatSession.user_id == current_user.id
        ).filter(
            ~db.query(ChatMessage).filter(
                ChatMessage.session_id == ChatSession.id
            ).exists()
        ).all()
        
        count = len(empty_sessions)
        for session in empty_sessions:
            db.delete(session)
        
        db.commit()
        return {
            "message": f"Deleted {count} empty session(s)",
            "deleted_count": count
        }
    except Exception as e:
        logger.exception("Failed to delete empty sessions")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete empty sessions: {str(e)}"
        )



@router.patch("/api/sessions/{session_id}")
async def update_session(
    session_id: str,
    payload: dict[str, Any] = Body(default_factory=dict),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
) -> dict:
    """Update a chat session."""
    session = db.query(ChatSession).filter(
        ChatSession.id == session_id,
        ChatSession.user_id == current_user.id
    ).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    title = payload.get("title")
    if title is not None:
        session.title = title
        session.updated_at = datetime.now(timezone.utc)
    
    db.commit()
    db.refresh(session)
    return {
        "id": session.id,
        "title": session.title,
        "created_at": session.created_at.isoformat(),
        "updated_at": session.updated_at.isoformat(),
    }


@router.get("/api/sessions/{session_id}/messages")
async def get_session_messages(
    session_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
) -> list[dict]:
    """Get all messages for a specific session."""
    session = db.query(ChatSession).filter(
        ChatSession.id == session_id,
        ChatSession.user_id == current_user.id
    ).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    messages = db.query(ChatMessage).filter(ChatMessage.session_id == session_id).all()
    return [
        {
            "id": msg.id,
            "role": msg.role,
            "content": msg.content,
            "confidence_score": msg.confidence_score,
            "evidence_strength": msg.evidence_strength,
            "graph_paths_used": msg.graph_paths_used,
            "safety": msg.safety,
            "reasoning_trace": msg.reasoning_trace,
            "structured_fields": msg.structured_fields,
            "variant": msg.variant,
            "created_at": msg.created_at.isoformat(),
        }
        for msg in messages
    ]


@router.post("/api/sessions/{session_id}/messages")
async def add_message_to_session(
    session_id: str,
    role: str,
    content: str,
    metadata: dict | None = None,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Add a message to a specific session."""
    session = db.query(ChatSession).filter(
        ChatSession.id == session_id,
        ChatSession.user_id == current_user.id
    ).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    if role not in ["user", "bot"]:
        raise HTTPException(status_code=400, detail="Invalid role")
    
    message = ChatMessage(
        session_id=session_id,
        role=role,
        content=content,
        confidence_score=metadata.get("confidence_score") if metadata else None,
        evidence_strength=metadata.get("evidence_strength") if metadata else None,
        graph_paths_used=metadata.get("graph_paths_used") if metadata else None,
        safety=metadata.get("safety") if metadata else None,
        reasoning_trace=metadata.get("reasoning_trace") if metadata else None,
        structured_fields=metadata.get("structured_fields") if metadata else None,
        variant=metadata.get("variant") if metadata else None,
    )
    db.add(message)
    db.commit()
    
    # Update session updated_at
    session.updated_at = message.created_at
    db.commit()
    
    return {
        "id": message.id,
        "role": message.role,
        "content": message.content,
        "created_at": message.created_at.isoformat(),
    }


@router.post("/api/chat")
async def chat(
    request: ChatRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
) -> dict:
    """Standard chat endpoint with message persistence."""
    try:
        # Save user message if session_id is provided
        if request.session_id:
            # Verify the session exists and belongs to the user
            session = db.query(ChatSession).filter(
                ChatSession.id == request.session_id,
                ChatSession.user_id == current_user.id
            ).first()
            if not session:
                raise HTTPException(status_code=404, detail="Session not found")
            
            user_message = ChatMessage(
                session_id=request.session_id,
                role="user",
                content=request.query,
            )
            db.add(user_message)
            db.commit()
            
            # Update session updated_at
            session.updated_at = user_message.created_at
            db.commit()
        
        response = await _run_pipeline(
            request.query,
            [m.model_dump() for m in request.history],
            strict_mode=request.strict_mode,
        )
        
        # Save bot response if session_id is provided
        if request.session_id:
            # Verify the session exists and belongs to the user
            session = db.query(ChatSession).filter(
                ChatSession.id == request.session_id,
                ChatSession.user_id == current_user.id
            ).first()
            if not session:
                raise HTTPException(status_code=404, detail="Session not found")
            
            bot_message = ChatMessage(
                session_id=request.session_id,
                role="bot",
                content=response.get("final_answer", ""),
                confidence_score=response.get("confidence_score"),
                evidence_strength=response.get("evidence_strength"),
                graph_paths_used=response.get("graph_paths_used"),
                safety=response.get("safety"),
                reasoning_trace=response.get("reasoning_trace"),
                structured_fields=response.get("structured_fields"),
            )
            db.add(bot_message)
            db.commit()
            
            # Update session updated_at
            session.updated_at = bot_message.created_at
            db.commit()
        
        return response
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
