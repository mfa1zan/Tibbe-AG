from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1)


class ConfidenceBreakdown(BaseModel):
    mapping_strength: float | None = None
    path_coverage: float | None = None
    hadith_presence: float | None = None
    faith_alignment: float | None = None
    causal_avg: float | None = None
    dosage_alignment: float | None = None


class ReasoningTrace(BaseModel):
    entity_detected: dict = Field(default_factory=dict)
    rationale_plan: list[dict] | None = None
    retrieved_paths: list[dict] = Field(default_factory=list)
    causal_ranking: list[dict] = Field(default_factory=list)
    causal_summary: dict = Field(default_factory=dict)
    dosage_validation: dict | None = None
    faith_alignment_notes: str = ""
    faith_alignment_score: float | None = None
    confidence_breakdown: ConfidenceBreakdown = Field(default_factory=ConfidenceBreakdown)
    multi_hop_activated: bool = False
    evaluation_metrics: dict | None = None
    pipeline_stages: list[str] = Field(default_factory=list)


class ChatResponse(BaseModel):
    final_answer: str
    evidence_strength: str
    graph_paths_used: int
    confidence_score: float | None = None
    reasoning_trace: ReasoningTrace | None = None
