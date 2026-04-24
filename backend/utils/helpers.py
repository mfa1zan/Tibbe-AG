"""Utility helpers for the GraphRAG backend."""

from __future__ import annotations

import logging


def configure_logging(level: str = "INFO") -> None:
    """Set up structured logging for the application."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s │ %(levelname)-7s │ %(name)s │ %(message)s",
        datefmt="%H:%M:%S",
    )
