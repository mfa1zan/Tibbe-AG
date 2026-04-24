"""Request / Response models for the chat API.

The ChatResponse fields are designed to match the frontend Zod schema in
``src/api.js`` exactly so that the zod `safeParse` never rejects the payload.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


# ── Request ──────────────────────────────────────────────────────────────────


class HistoryMessage(BaseModel):
    role: str = Field(..., pattern="^(user|bot)$")
    content: str


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1)
    history: list[HistoryMessage] = Field(default_factory=list, max_length=20)
    strict_mode: bool = Field(default=False, description="When true, LLM must only use DB evidence")


# ── Response ─────────────────────────────────────────────────────────────────


class ChatResponse(BaseModel):
    """Matches the Zod ``chatResponseSchema`` in ``src/api.js``."""

    final_answer: str
    evidence_strength: str | None = None
    graph_paths_used: int | None = None
    confidence_score: float | None = None
    safety: dict | None = None
    reasoning_trace: dict | None = None
    structured_fields: dict | None = None
    pipeline_debug_trace: dict | None = None
