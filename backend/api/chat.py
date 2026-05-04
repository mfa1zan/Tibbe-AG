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
    """Build a separated answer with one section per segment + a combined guidance section."""
    section_outputs: list[tuple[str, str]] = []
    intent_label_map = {
        "ingredient_treatment": "Diseases Treated",
        "ingredient_compounds": "Chemical Composition",
        "ingredient_substitute": "Pharmaceutical Drug Equivalents",
        "ingredient_drug_mapping": "Drug Mapping",
        "disease_treatment": "Natural Remedies",
        "disease_drug": "Pharmaceutical Drugs",
        "disease_full_chain": "Full Treatment Chain",
        "hadith_info": "Prophetic References",
        "drug_substitute": "Natural Alternatives",
        "compound_search": "Ingredients Containing Compound",
        "drug_book": "Drug Reference",
        "drug_count": "Drug Count",
        "general": "General Information",
    }

    ingredient_intents = {
        "ingredient_treatment",
        "ingredient_compounds",
        "ingredient_substitute",
        "ingredient_drug_mapping",
        "drug_count",
    }
    disease_intents = {
        "disease_treatment",
        "disease_drug",
        "disease_full_chain",
        "hadith_info",
    }

    def _shared_entity_value(entity_key: str) -> str | None:
        values: list[str] = []
        for segment in routed_segments:
            entities = segment.get("entities", {}) if isinstance(segment, dict) else {}
            if not isinstance(entities, dict):
                return None
            value = entities.get(entity_key)
            if not isinstance(value, str) or not value.strip():
                return None
            values.append(value.strip())
        if values and all(val == values[0] for val in values):
            return values[0]
        return None

    same_entity_multi_intent = False
    if len(routed_segments) > 1:
        same_entity_multi_intent = any(
            _shared_entity_value(key) for key in ("disease", "ingredient", "drug")
        )

    intent_list = [
        segment.get("intent", "general")
        for segment in routed_segments
        if isinstance(segment, dict)
    ]
    if intent_list and all(intent in ingredient_intents for intent in intent_list):
        opening = "Based on the Knowledge Graph evidence, here is an ingredient-wise response:"
    elif intent_list and all(intent in disease_intents for intent in intent_list):
        opening = "Based on the Knowledge Graph evidence, here is a condition-wise response:"
    else:
        opening = "Based on the Knowledge Graph evidence, here is a detailed response:"

    for result in routed_segments:
        db_result = result["db_result"]
        rows = db_result.get("rows", []) or []
        entities = result.get("entities", {})

        disease = entities.get("disease") if isinstance(entities, dict) else None
        ingredient = entities.get("ingredient") if isinstance(entities, dict) else None
        drug = entities.get("drug") if isinstance(entities, dict) else None

        intent = result.get("intent", "general")
        if not isinstance(intent, str) or not intent.strip():
            intent = "general"

        if same_entity_multi_intent:
            label = intent_label_map.get(intent, intent)
        elif isinstance(disease, str) and disease.strip():
            label = disease.strip()
        elif isinstance(drug, str) and drug.strip():
            label = drug.strip()
        elif isinstance(ingredient, str) and ingredient.strip():
            label = ingredient.strip()
        else:
            label = result.get("segment", "this condition")

        if strict_mode and not rows:
            cleaned = _STRICT_NO_DATA_MESSAGE
        else:
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
    if same_entity_multi_intent:
        both_line = (
            "\n\nFor a complete picture, review each section above. "
            "Always consult a qualified healthcare professional before making "
            "any treatment decisions."
        )
    elif len(labels) >= 2:
        both_line = (
            f"\n\nIf you are experiencing both {labels[0]} and {labels[1]}, use the condition-specific guidance above for each, "
            "and seek medical advice for an integrated treatment plan."
        )
    else:
        both_line = ""

    final_answer = (
        f"{opening}\n\n"
        f"{section_text}"
        f"{both_line}\n\n"
        "Medical Disclaimer: This information is for educational purposes and is not a substitute for medical diagnosis or treatment."
    )

    return final_answer, {"answer": final_answer, "model": "composed-multi-segment", "duration_ms": 0, "api_call_count": get_api_call_count()}


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
    entity_names = get_entity_lists()
    analysis = await query_router.analyze_query_llm(
        query,
        entity_names,
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
