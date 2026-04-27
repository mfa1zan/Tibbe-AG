"""Response builder — assembles the final ChatResponse dict.

Takes the outputs of graph_service + llm_service + judge_service and builds
a response payload that satisfies the frontend Zod schema.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from backend.core.config import get_settings
from backend.utils.trace_logger import log_step

logger = logging.getLogger(__name__)


def build_response(
    *,
    user_query: str,
    final_answer: str,
    intent: str,
    db_result: dict[str, Any],
    llm_result: dict[str, Any],
    entities: dict[str, str | None],
    judge_report: dict[str, Any] | None = None,
    debug: bool = False,
    trace_id: str | None = None,
    trace_context: Any | None = None,
) -> dict[str, Any]:
    """Build the response dict matching the frontend ``chatResponseSchema``.

    Fields the frontend expects (all optional except ``final_answer``):
        final_answer, evidence_strength, graph_paths_used, confidence_score,
        safety, reasoning_trace, structured_fields, pipeline_debug_trace
    """
    rows = db_result.get("rows", [])
    row_count = db_result.get("row_count", 0)

    # ── Evidence strength ────────────────────────────────────────────────
    if row_count >= 10:
        evidence_strength = "strong"
    elif row_count >= 3:
        evidence_strength = "moderate"
    elif row_count >= 1:
        evidence_strength = "weak"
    else:
        evidence_strength = "none"

    # ── Confidence score ─────────────────────────────────────────────────
    confidence_score = None
    if judge_report and judge_report.get("composite_score") is not None:
        confidence_score = judge_report["composite_score"]

    # ── Reasoning trace (lightweight) ────────────────────────────────────
    reasoning_trace = {
        "entity_detected": {
            k: v for k, v in entities.items() if k != "_llm_debug" and v is not None
        },
        "intent": intent,
        "query_used": db_result.get("query_name", ""),
        "evidence_count": row_count,
    }

    # ── Structured fields ────────────────────────────────────────────────
    structured_fields = {
        "ingredients": _extract_unique(rows, "ingredient"),
        "compounds": _extract_unique(rows, "compound"),
        "drugs": _extract_unique(rows, "drug"),
        "references": _extract_unique(rows, "reference"),
    }
    # Drop empty keys
    structured_fields = {k: v for k, v in structured_fields.items() if v}

    # ── Base response ────────────────────────────────────────────────────
    response: dict[str, Any] = {
        "final_answer": final_answer,
        "evidence_strength": evidence_strength,
        "graph_paths_used": row_count,
        "confidence_score": confidence_score,
        "safety": None,
        "reasoning_trace": reasoning_trace,
        "structured_fields": structured_fields or None,
        "pipeline_debug_trace": None,
        "trace_id": trace_id,
    }

    # ── Debug trace (only when requested) ────────────────────────────────
    if debug:
        response["pipeline_debug_trace"] = {
            "user_query": user_query,
            "intent": intent,
            "entities": {k: v for k, v in entities.items() if k != "_llm_debug"},
            "query_id": db_result.get("query_name"),
            "cypher": db_result.get("cypher"),
            "db_row_count": row_count,
            "db_sample": rows[:5],
            "db_duration_ms": db_result.get("duration_ms"),
            "llm_model": llm_result.get("model"),
            "llm_duration_ms": llm_result.get("duration_ms"),
            "entity_extraction_debug": entities.get("_llm_debug"),
            "judge_report": judge_report,
        }

    if trace_context is not None:
        log_step(
            trace_context,
            "PIPELINE",
            "Response Builder",
            details={
                "final_answer_length": len(final_answer or ""),
                "evidence_strength": evidence_strength,
                "graph_paths_used": row_count,
                "confidence_score": confidence_score,
                "structured_fields_keys": list((structured_fields or {}).keys()),
                "trace_id": trace_id,
                "final_response_json": response,
            },
        )

    settings = get_settings()
    if settings.debug_trace:
        logger.info(
            "FINAL_RESPONSE_OBJECT\n%s",
            json.dumps(response, ensure_ascii=True, indent=2, default=str),
        )

    return response


def build_error_response(
    error_msg: str = "Could not process your request. Please try again.",
    trace_id: str | None = None,
) -> dict[str, Any]:
    """Build a safe fallback response when the pipeline fails."""
    return {
        "final_answer": error_msg,
        "evidence_strength": "none",
        "graph_paths_used": 0,
        "confidence_score": None,
        "safety": None,
        "reasoning_trace": None,
        "structured_fields": None,
        "pipeline_debug_trace": None,
        "trace_id": trace_id,
    }


# ── Helpers ──────────────────────────────────────────────────────────────────


def _extract_unique(rows: list[dict], key: str) -> list[str]:
    """Extract unique non-null string values for a given key across all rows."""
    seen: set[str] = set()
    result: list[str] = []
    for row in rows:
        val = row.get(key)
        if isinstance(val, str) and val.strip() and val not in seen:
            seen.add(val)
            result.append(val)
    return result
