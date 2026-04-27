"""Request-level trace logging utilities for end-to-end debugging."""

from __future__ import annotations

import json
import logging
import time
import traceback
import uuid
from datetime import datetime
from typing import Any

from backend.core.config import get_settings

logger = logging.getLogger("backend.trace")

_SECRET_KEYS = {
    "authorization",
    "api_key",
    "apikey",
    "token",
    "password",
    "secret",
    "neo4j_password",
    "groq_api_key",
}


class TraceContext:
    """In-memory trace context for one request lifecycle."""

    def __init__(self, trace_id: str, debug_enabled: bool):
        self.trace_id = trace_id
        self.debug_enabled = debug_enabled
        self.start_time = time.perf_counter()
        self.step_count = 0


def is_trace_enabled() -> bool:
    settings = get_settings()
    return bool(settings.debug_trace)


def _mask_text(value: str) -> str:
    lowered = value.lower()
    if "bearer " in lowered:
        return "***"
    return value


def sanitize_payload(payload: Any) -> Any:
    """Recursively sanitize payloads to avoid leaking secrets in logs."""
    if isinstance(payload, dict):
        sanitized: dict[str, Any] = {}
        for key, value in payload.items():
            if isinstance(key, str) and key.lower() in _SECRET_KEYS:
                sanitized[key] = "***"
            else:
                sanitized[key] = sanitize_payload(value)
        return sanitized
    if isinstance(payload, list):
        return [sanitize_payload(item) for item in payload]
    if isinstance(payload, str):
        return _mask_text(payload)
    return payload


def _dump(payload: Any) -> str:
    try:
        return json.dumps(sanitize_payload(payload), ensure_ascii=True, indent=2, default=str)
    except Exception:
        return repr(payload)


def _banner(title: str) -> str:
    return f"\n{'=' * 50}\n{title}\n{'=' * 50}"


def start_trace(trace_id: str | None = None) -> TraceContext:
    """Start and log a new request trace."""
    context = TraceContext(trace_id=trace_id or uuid.uuid4().hex[:8], debug_enabled=is_trace_enabled())
    logger.info(
        "%s\nTrace ID: %s\nTime: %s",
        _banner("REQUEST TRACE START"),
        context.trace_id,
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    )
    return context


def log_step(context: TraceContext, scope: str, title: str, details: dict[str, Any] | None = None) -> None:
    """Log a trace step with optional detailed payload."""
    context.step_count += 1
    # In DEBUG_TRACE mode, runtime logs emitted at execution points are the
    # source of truth; suppress wrapper logs to avoid duplicate payload noise.
    if context.debug_enabled:
        return

    if details:
        logger.info(
            "[%s][STEP %d] %s | trace_id=%s\n%s",
            scope,
            context.step_count,
            title,
            context.trace_id,
            _dump(details),
        )
        return

    logger.info("[%s][STEP %d] %s | trace_id=%s", scope, context.step_count, title, context.trace_id)


def log_model_call(
    context: TraceContext,
    stage: str,
    *,
    model: str,
    system_prompt: str,
    messages: list[dict[str, Any]],
    output: Any,
    duration_ms: float | int,
    temperature: float | int | None = None,
    max_tokens: int | None = None,
    extra_details: dict[str, Any] | None = None,
) -> None:
    """Log model call inputs/outputs in one structured entry."""
    details = {
        "model": model,
        "system_prompt": system_prompt,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "raw_model_response": output,
        "duration_ms": duration_ms,
    }
    if extra_details:
        details.update(extra_details)
    log_step(context, "PIPELINE", stage, details=details)


def log_db_call(
    context: TraceContext,
    *,
    query_id: str,
    cypher: str,
    params: dict[str, Any],
    rows: list[dict[str, Any]],
    duration_ms: float | int,
) -> None:
    """Log database query details for trace inspection."""
    sample = rows[:3] if isinstance(rows, list) else []
    details = {
        "database_call_started": True,
        "query_id": query_id,
        "cypher": cypher,
        "params": params,
        "rows_returned": len(rows) if isinstance(rows, list) else 0,
        "first_3_rows": sample,
        "duration_ms": duration_ms,
    }
    log_step(context, "PIPELINE", "Database Query", details=details)


def log_error(
    context: TraceContext,
    *,
    stage: str,
    exception: Exception,
    input_snapshot: dict[str, Any] | None = None,
) -> None:
    """Log an error in a trace-friendly format with stack trace."""
    stack = traceback.format_exc()
    details = {
        "stage": stage,
        "exception": repr(exception),
        "stack_trace": stack,
        "input_snapshot": input_snapshot or {},
    }
    logger.error("[ERROR] trace_id=%s\n%s", context.trace_id, _dump(details))


def end_trace(context: TraceContext, *, success: bool) -> None:
    """Emit final summary for request trace."""
    duration_ms = round((time.perf_counter() - context.start_time) * 1000, 1)
    logger.info(
        "%s\nTRACE COMPLETE\nTrace ID: %s\nTotal Duration: %sms\nSteps Completed: %d\nSuccess: %s\n%s",
        "\n" + "=" * 50,
        context.trace_id,
        duration_ms,
        context.step_count,
        str(success).lower(),
        "=" * 50,
    )
