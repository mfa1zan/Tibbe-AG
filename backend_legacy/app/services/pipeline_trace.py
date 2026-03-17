"""
PRO-MedGraph Pipeline Tracer
==============================

Lightweight observability layer that captures per-step timing, inputs/outputs,
and KG query logs during a single pipeline execution.  The tracer is scoped to
the current asyncio task via a ``contextvars.ContextVar`` so concurrent requests
never share state.

Usage inside the orchestrator::

    tracer = PipelineTracer()
    set_tracer(tracer)

    tracer.set_user_input(query)
    tracer.start_step("stage_1_intent", {"query": query})
    # … do work …
    tracer.end_step({"intent": "auto"})

    clear_tracer()

Graph-service modules can retrieve the active tracer anywhere::

    from app.services.pipeline_tracer import get_tracer
    tracer = get_tracer()
    if tracer:
        tracer.log_kg_query(…)
"""

from __future__ import annotations

import logging
import time
from contextvars import ContextVar
from typing import Any

logger = logging.getLogger(__name__)

# ── Context-scoped tracer variable ─────────────────────────────────────────────

_tracer_var: ContextVar[PipelineTracer | None] = ContextVar("_pipeline_tracer", default=None)


def get_tracer() -> PipelineTracer | None:
    """Return the active tracer for the current async context, or ``None``."""
    return _tracer_var.get()


def set_tracer(tracer: PipelineTracer) -> None:
    """Attach *tracer* to the current async context."""
    _tracer_var.set(tracer)


def clear_tracer() -> None:
    """Remove the tracer from the current async context."""
    _tracer_var.set(None)


# ── PipelineTracer ─────────────────────────────────────────────────────────────


class PipelineTracer:
    """Captures pipeline execution metadata for debugging and experiment logs."""

    def __init__(self) -> None:
        self._user_input: str = ""
        self._final_output: str = ""
        self._steps: list[dict[str, Any]] = []
        self._extra_data: dict[str, Any] = {}
        self._kg_queries: list[dict[str, Any]] = []
        self._current_step: dict[str, Any] | None = None
        self._t0: float | None = None
        self._pipeline_start: float = time.perf_counter()

    # ── User input / final output ──────────────────────────────────────────

    def set_user_input(self, text: str) -> None:
        self._user_input = text

    def set_final_output(self, text: str) -> None:
        self._final_output = text

    # ── Step lifecycle ─────────────────────────────────────────────────────

    def start_step(self, name: str, metadata: dict[str, Any] | None = None) -> None:
        """Mark the start of a named pipeline step."""
        # Auto-close a previous step if the caller forgot
        if self._current_step is not None:
            self.end_step({})

        self._current_step = {
            "name": name,
            "start_metadata": metadata or {},
            "end_metadata": {},
        }
        self._t0 = time.perf_counter()

    def end_step(self, metadata: dict[str, Any] | None = None) -> None:
        """Finalise the current step with timing and end-metadata."""
        if self._current_step is None:
            return

        elapsed = (time.perf_counter() - self._t0) * 1000 if self._t0 else 0
        self._current_step["end_metadata"] = metadata or {}
        self._current_step["elapsed_ms"] = round(elapsed, 1)
        self._steps.append(self._current_step)
        self._current_step = None
        self._t0 = None

    # ── Auxiliary data logging ─────────────────────────────────────────────

    def log_data(self, key: str, value: Any) -> None:
        """Attach arbitrary data to the trace under *key*."""
        self._extra_data[key] = value

    def log_kg_query(
        self,
        *,
        purpose: str,
        cypher: str,
        parameters: dict[str, Any] | None = None,
        row_count: int = 0,
        result_sample: Any = None,
        duration_ms: float | None = None,
        error: str | None = None,
    ) -> None:
        """Record a Neo4j / KG query execution for the trace."""
        entry: dict[str, Any] = {
            "purpose": purpose,
            "cypher": cypher,
            "parameters": parameters,
            "row_count": row_count,
            "duration_ms": round(duration_ms, 1) if duration_ms is not None else None,
        }
        if result_sample is not None:
            entry["result_sample"] = result_sample
        if error:
            entry["error"] = error
        self._kg_queries.append(entry)

    # ── Serialisation ──────────────────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        total_ms = round((time.perf_counter() - self._pipeline_start) * 1000, 1)
        return {
            "user_input": self._user_input,
            "final_output": self._final_output[:500] if self._final_output else "",
            "total_elapsed_ms": total_ms,
            "steps": self._steps,
            "kg_queries": self._kg_queries,
            "extra_data": self._extra_data,
        }


__all__ = [
    "PipelineTracer",
    "get_tracer",
    "set_tracer",
    "clear_tracer",
]