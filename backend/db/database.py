"""SQLAlchemy database connection and session management."""

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from backend.core.config import get_settings


def get_database_url() -> str:
    settings = get_settings()
    return settings.db_url


engine = create_engine(
    get_database_url(),
    pool_pre_ping=True,
    pool_recycle=3600,
    echo=False,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    pass


def get_db():
    """FastAPI dependency: yields a database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def create_tables():
    """Create all tables if they don't exist."""
    # Import models so SQLAlchemy registers them before create_all
    from backend.db import models  # noqa: F401
    Base.metadata.create_all(bind=engine)
