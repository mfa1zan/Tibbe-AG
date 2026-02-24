import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.logging_config import configure_logging
from app.schemas import ChatRequest, ChatResponse
from app.services.chat_service import ChatService
from app.services.kg_service import KGService
from app.services.llm_service import LLMService

logger = logging.getLogger(__name__)


def _build_services() -> tuple[KGService, LLMService, ChatService]:
    settings = get_settings()

    kg_service = KGService(
        uri=settings.neo4j_uri,
        username=settings.neo4j_username,
        password=settings.neo4j_password,
        ttl_seconds=settings.kg_cache_ttl_seconds,
        maxsize=settings.kg_cache_maxsize,
    )
    llm_service = LLMService(
        api_key=settings.groq_api_key,
        model=settings.groq_model,
        base_url=settings.groq_base_url,
    )
    chat_service = ChatService(kg_service=kg_service, llm_service=llm_service)
    return kg_service, llm_service, chat_service


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    configure_logging(settings.log_level)

    kg_service, llm_service, chat_service = _build_services()
    app.state.kg_service = kg_service
    app.state.llm_service = llm_service
    app.state.chat_service = chat_service

    logger.info("FastAPI app started")
    yield

    await kg_service.close()
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
    chat_service: ChatService = app.state.chat_service
    reply, provenance = await chat_service.process_message(
        user_message=request.message,
        session_id=request.session_id,
    )
    return ChatResponse(reply=reply, provenance=provenance)
