from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1)


class ChatResponse(BaseModel):
    final_answer: str
    evidence_strength: str
    graph_paths_used: int
    confidence_score: float | None = None
