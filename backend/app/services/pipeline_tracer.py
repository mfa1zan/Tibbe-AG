"""
Pipeline Tracer for PRO-MedGraph — Full Debug Visibility.

A per-request context object that captures EVERY intermediate step in the
pipeline:  LLM calls (system prompt, user prompt, model, response), KG
Cypher queries (query text, parameters, row count, raw results), entity
extraction, intent classification, reasoning, scoring — everything.

Usage:
    tracer = PipelineTracer()         # create at request start
    tracer.start_step("stage_name")
    tracer.log_llm_call(...)          # inside any LLM call
    tracer.log_kg_query(...)          # inside any Neo4j query
    tracer.log_data("key", value)     # arbitrary intermediate data
    tracer.end_step()

At the end the orchestrator calls tracer.to_dict() and attaches the full
trace to the API response for inspection.
"""

from __future__ import annotations

import time
import json
import logging
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# ── Context-var for per-request tracing ──────────────────────────────────────

_current_tracer: ContextVar["PipelineTracer | None"] = ContextVar(
    "pipeline_tracer", default=None
)


def get_tracer() -> "PipelineTracer | None":
    """Return the current request's tracer (or None if tracing is off)."""
    return _current_tracer.get()


def set_tracer(tracer: "PipelineTracer") -> None:
    _current_tracer.set(tracer)


def clear_tracer() -> None:
    _current_tracer.set(None)


# ── Data structures ──────────────────────────────────────────────────────────


@dataclass
class LLMCallRecord:
    """One LLM API call."""
    step: str = ""
    purpose: str = ""
    model: str = ""
    temperature: float = 0.0
    system_prompt: str = ""
    user_prompt: str = ""
    response: str = ""
    response_length: int = 0
    duration_ms: float = 0.0
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "purpose": self.purpose,
            "model": self.model,
            "temperature": self.temperature,
            "system_prompt": self.system_prompt,
            "user_prompt": self.user_prompt,
            "response": self.response,
            "response_length": self.response_length,
            "duration_ms": round(self.duration_ms, 2),
            "error": self.error,
        }


@dataclass
class KGQueryRecord:
    """One Neo4j Cypher query."""
    step: str = ""
    purpose: str = ""
    cypher: str = ""
    parameters: dict[str, Any] = field(default_factory=dict)
    row_count: int = 0
    result_sample: Any = None  # first few rows
    duration_ms: float = 0.0
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "purpose": self.purpose,
            "cypher": self.cypher,
            "parameters": self.parameters,
            "row_count": self.row_count,
            "result_sample": self.result_sample,
            "duration_ms": round(self.duration_ms, 2),
            "error": self.error,
        }


@dataclass
class StepRecord:
    """One named pipeline stage."""
    name: str = ""
    started_at: float = 0.0
    ended_at: float = 0.0
    duration_ms: float = 0.0
    input_data: dict[str, Any] = field(default_factory=dict)
    output_data: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "duration_ms": round(self.duration_ms, 2),
            "input_data": self.input_data,
            "output_data": self.output_data,
            "notes": self.notes,
        }


def _safe_serialize(obj: Any, max_str_len: int = 5000) -> Any:
    """Make an object JSON-safe, truncating huge strings."""
    if obj is None or isinstance(obj, (bool, int, float)):
        return obj
    if isinstance(obj, str):
        return obj[:max_str_len] + "..." if len(obj) > max_str_len else obj
    if isinstance(obj, (list, tuple)):
        return [_safe_serialize(item, max_str_len) for item in obj[:50]]
    if isinstance(obj, dict):
        return {str(k): _safe_serialize(v, max_str_len) for k, v in list(obj.items())[:100]}
    return str(obj)[:max_str_len]


class PipelineTracer:
    """Collects all debug data for one request."""

    def __init__(self) -> None:
        self._start_time = time.perf_counter()
        self._steps: list[StepRecord] = []
        self._llm_calls: list[LLMCallRecord] = []
        self._kg_queries: list[KGQueryRecord] = []
        self._current_step: StepRecord | None = None
        self._extra: dict[str, Any] = {}  # arbitrary key-value
        self._user_input: str = ""
        self._final_output: str = ""

    # ── Step management ─────────────────────────────────────────────────

    def set_user_input(self, query: str) -> None:
        self._user_input = query

    def set_final_output(self, answer: str) -> None:
        self._final_output = answer

    def start_step(self, name: str, input_data: dict[str, Any] | None = None) -> None:
        # Auto-close previous step
        if self._current_step is not None:
            self.end_step()
        self._current_step = StepRecord(
            name=name,
            started_at=time.perf_counter(),
            input_data=_safe_serialize(input_data) if input_data else {},
        )

    def add_step_note(self, note: str) -> None:
        if self._current_step:
            self._current_step.notes.append(note)

    def end_step(self, output_data: dict[str, Any] | None = None) -> None:
        if self._current_step is None:
            return
        now = time.perf_counter()
        self._current_step.ended_at = now
        self._current_step.duration_ms = (now - self._current_step.started_at) * 1000
        if output_data:
            self._current_step.output_data = _safe_serialize(output_data)
        self._steps.append(self._current_step)
        self._current_step = None

    # ── LLM call logging ────────────────────────────────────────────────

    def log_llm_call(
        self,
        *,
        purpose: str,
        model: str,
        temperature: float,
        system_prompt: str,
        user_prompt: str,
        response: str = "",
        duration_ms: float = 0.0,
        error: str | None = None,
    ) -> None:
        step_name = self._current_step.name if self._current_step else "unknown"
        record = LLMCallRecord(
            step=step_name,
            purpose=purpose,
            model=model,
            temperature=temperature,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response=response,
            response_length=len(response),
            duration_ms=duration_ms,
            error=error,
        )
        self._llm_calls.append(record)

    # ── KG query logging ────────────────────────────────────────────────

    def log_kg_query(
        self,
        *,
        purpose: str,
        cypher: str,
        parameters: dict[str, Any] | None = None,
        row_count: int = 0,
        result_sample: Any = None,
        duration_ms: float = 0.0,
        error: str | None = None,
    ) -> None:
        step_name = self._current_step.name if self._current_step else "unknown"
        record = KGQueryRecord(
            step=step_name,
            purpose=purpose,
            cypher=cypher,
            parameters=parameters or {},
            row_count=row_count,
            result_sample=_safe_serialize(result_sample),
            duration_ms=duration_ms,
            error=error,
        )
        self._kg_queries.append(record)

    # ── Arbitrary data logging ──────────────────────────────────────────

    def log_data(self, key: str, value: Any) -> None:
        self._extra[key] = _safe_serialize(value)

    # ── Export ──────────────────────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        total_ms = (time.perf_counter() - self._start_time) * 1000
        return {
            "user_input": self._user_input,
            "final_output": _safe_serialize(self._final_output),
            "total_duration_ms": round(total_ms, 2),
            "total_llm_calls": len(self._llm_calls),
            "total_kg_queries": len(self._kg_queries),
            "total_steps": len(self._steps),
            "steps": [s.to_dict() for s in self._steps],
            "llm_calls": [c.to_dict() for c in self._llm_calls],
            "kg_queries": [q.to_dict() for q in self._kg_queries],
            "extra_data": self._extra,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False, default=str)


__all__ = [
    "PipelineTracer",
    "get_tracer",
    "set_tracer",
    "clear_tracer",
]
