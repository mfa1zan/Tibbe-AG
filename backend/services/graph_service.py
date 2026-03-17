"""Neo4j graph service — execute predefined queries only."""

from __future__ import annotations

import logging
import time
from functools import lru_cache
from typing import Any

from neo4j import GraphDatabase

from backend.core.config import get_settings
from backend.queries.query_library import ALL_QUERIES, CypherQuery

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _get_driver():
    settings = get_settings()
    return GraphDatabase.driver(
        settings.neo4j_uri,
        auth=(settings.neo4j_username, settings.neo4j_password),
    )


def close_driver() -> None:
    """Explicitly close the Neo4j driver (called on app shutdown)."""
    driver = _get_driver()
    driver.close()
    _get_driver.cache_clear()


def execute_query(
    query_id: str,
    params: dict[str, Any],
) -> dict[str, Any]:
    """Execute a predefined query by its ID and return structured results.

    Returns a dict with:
        - ``rows``: list of record dicts
        - ``row_count``: number of rows returned
        - ``query_name``: human-readable name of the query
        - ``duration_ms``: execution time in milliseconds
        - ``cypher``: the Cypher text that was executed (for debug)
    """
    query_def: CypherQuery | None = ALL_QUERIES.get(query_id)
    if query_def is None:
        return {
            "error": f"Unknown query ID: {query_id}",
            "rows": [],
            "row_count": 0,
            "query_name": "unknown",
            "duration_ms": 0,
            "cypher": "",
        }

    # Validate all required params are present
    missing = [k for k in query_def.param_keys if k not in params]
    if missing:
        return {
            "error": f"Missing parameters for query {query_id}: {missing}",
            "rows": [],
            "row_count": 0,
            "query_name": query_def.name,
            "duration_ms": 0,
            "cypher": query_def.cypher,
        }

    driver = _get_driver()
    t0 = time.perf_counter()

    try:
        with driver.session() as session:
            result = session.run(query_def.cypher, **params)
            rows = [dict(record) for record in result]

        duration_ms = round((time.perf_counter() - t0) * 1000, 1)

        logger.info(
            "KG [%s] %s → %d rows (%.0fms)",
            query_def.id,
            query_def.name,
            len(rows),
            duration_ms,
        )

        return {
            "rows": rows,
            "row_count": len(rows),
            "query_name": query_def.name,
            "duration_ms": duration_ms,
            "cypher": query_def.cypher.strip(),
        }
    except Exception as exc:
        duration_ms = round((time.perf_counter() - t0) * 1000, 1)
        logger.exception(
            "KG QUERY FAILED [%s] %s (%.0fms): %s",
            query_def.id,
            query_def.name,
            duration_ms,
            exc,
        )
        return {
            "error": str(exc),
            "rows": [],
            "row_count": 0,
            "query_name": query_def.name,
            "duration_ms": duration_ms,
            "cypher": query_def.cypher.strip(),
        }
