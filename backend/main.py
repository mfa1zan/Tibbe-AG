"""FastAPI entrypoint for the GraphRAG backend."""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.core.config import get_settings
from backend.services.graph_service import close_driver
from backend.utils.helpers import configure_logging

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    configure_logging(settings.log_level)
    logger.info("GraphRAG backend started (clean architecture)")
    # Create database tables
    from backend.db.database import create_tables
    create_tables()
    yield
    close_driver()
    logger.info("GraphRAG backend shutdown")


app = FastAPI(
    title="PRO-MedGraph API",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Mount API routes ─────────────────────────────────────────────────────────
from backend.api.auth import router as auth_router  # noqa: E402
from backend.api.chat import router as chat_router  # noqa: E402

app.include_router(auth_router)
app.include_router(chat_router)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}
