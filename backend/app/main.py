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


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    pipeline: orchestrator.GraphRAGOrchestrator = app.state.orchestrator

    try:
        # 1) Execute GraphRAG orchestrator pipeline for the incoming query.
        pipeline_result = await pipeline.process_user_query_with_context_async(request.query)

        reasoning = pipeline_result.get("reasoning", {})
        llm_output = pipeline_result.get("output", {})

        # 2) Apply deterministic safety and confidence post-processing before response.
        safe_output = safety_service.apply_safety_checks(reasoning=reasoning, llm_output=llm_output)

        # 3) Return only public-safe API fields (no Cypher, raw records, or schema internals).
        return ChatResponse(
            final_answer=safe_output.get("final_answer")
            or "PRO-MedGraph could not generate a final answer right now.",
            evidence_strength=safe_output.get("evidence_strength") or "weak",
            graph_paths_used=int(safe_output.get("graph_paths_used") or 0),
            confidence_score=safe_output.get("confidence_score"),
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
