from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)
    session_id: str | None = None


class ChatResponse(BaseModel):
    reply: str
    provenance: list[str] = []
    generated_cypher: str | None = None
    kg_result_count: int = 0
    judge_score: int | None = None
