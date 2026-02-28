import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.logging_config import configure_logging
from app.schemas import ChatRequest, ChatResponse
from app.services import orchestrator, safety_service
from app.services.graph_service import close_graph_driver

logger = logging.getLogger(__name__)


def _build_services() -> orchestrator.GraphRAGOrchestrator:
    # Keep orchestrator construction isolated for easy future replacement.
    return orchestrator.GraphRAGOrchestrator()


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    configure_logging(settings.log_level)

    app.state.orchestrator = _build_services()

    logger.info("FastAPI app started")
    yield

    # Explicitly close shared graph driver used by graph_service singletons.
    close_graph_driver()
    logger.info("FastAPI app shutdown")


app = FastAPI(title="Biomedical KG Chat API", version="0.2.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/models")
async def models() -> dict[str, str]:
    """Return which Groq model is assigned to each pipeline stage."""
    from app.services.multi_model_service import get_model_map

    return get_model_map()


@app.get("/es")
async def evidence_strength_check() -> dict:
    """
    End-to-end pipeline smoke test.

    Sends a known biomedical query through the full orchestrator pipeline and
    returns a compact diagnostics object so you can verify:
      • Neo4j connectivity
      • LLM reachability (all 5 models)
      • Evidence strength / confidence produced
      • Pipeline stages executed

    Hit  GET /es  after deploy to confirm everything is wired correctly.
    """
    from app.services.multi_model_service import get_model_map

    pipeline: orchestrator.GraphRAGOrchestrator = app.state.orchestrator
    test_query = "What is the remedy for hypertension in Prophetic medicine?"
    models_map = get_model_map()

    try:
        result = await pipeline.process_user_query_with_context_async(test_query)
        output = result.get("output", {})
        trace = output.get("reasoning_trace") or result.get("reasoning_trace") or {}

        return {
            "status": "ok",
            "test_query": test_query,
            "evidence_strength": output.get("evidence_strength"),
            "graph_paths_used": output.get("graph_paths_used"),
            "confidence_score": output.get("confidence_score"),
            "pipeline_stages": trace.get("pipeline_stages", []),
            "entities_detected": result.get("entities"),
            "faith_alignment_score": (result.get("faith_alignment") or {}).get("faith_alignment_score"),
            "models": models_map,
            "neo4j": "connected",
            "llm": "reachable",
        }
    except Exception as exc:
        logger.exception("/es diagnostics failed")
        return {
            "status": "error",
            "test_query": test_query,
            "error": str(exc),
            "models": models_map,
            "neo4j": "unknown",
            "llm": "unknown",
        }


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    pipeline: orchestrator.GraphRAGOrchestrator = app.state.orchestrator

    try:
        # 1) Execute full GraphRAG orchestrator pipeline (safety already applied inside).
        pipeline_result = await pipeline.process_user_query_with_context_async(request.query)
        output = pipeline_result.get("output", {})

        # 2) Extract reasoning trace (attached by orchestrator stage 15).
        raw_trace = output.get("reasoning_trace") or pipeline_result.get("reasoning_trace")

        # 3) Return public-safe API fields + optional reasoning trace.
        return ChatResponse(
            final_answer=output.get("final_answer")
            or "PRO-MedGraph could not generate a final answer right now.",
            evidence_strength=output.get("evidence_strength") or "weak",
            graph_paths_used=int(output.get("graph_paths_used") or 0),
            confidence_score=output.get("confidence_score"),
            reasoning_trace=raw_trace,
        )
    except Exception:
        logger.exception("/api/chat failed")
        return ChatResponse(
            final_answer=(
                "PRO-MedGraph could not process your request safely at the moment. "
                "Please try again shortly with a clearer query."
            ),
            evidence_strength="weak",
            graph_paths_used=0,
            confidence_score=None,
        )
