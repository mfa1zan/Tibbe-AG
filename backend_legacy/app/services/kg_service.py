import logging
import json
from pathlib import Path
from typing import Any

from cachetools import TTLCache
from neo4j import AsyncGraphDatabase

logger = logging.getLogger(__name__)


class KGService:
    def __init__(
        self,
        uri: str,
        username: str,
        password: str,
        ttl_seconds: int,
        maxsize: int,
        schema_file_path: str,
    ) -> None:
        self._driver = AsyncGraphDatabase.driver(uri, auth=(username, password))
        self._query_cache = TTLCache(maxsize=maxsize, ttl=ttl_seconds)
        self._schema_file_path = schema_file_path
        self._schema_payload = self._load_schema_payload(Path(schema_file_path))

    @staticmethod
    def _load_schema_payload(schema_path: Path) -> Any:
        with schema_path.open("r", encoding="utf-8") as schema_file:
            return json.load(schema_file)

    async def close(self) -> None:
        await self._driver.close()

    def get_kg_schema(self) -> str:
        return json.dumps(self._schema_payload, ensure_ascii=False, indent=2)

    async def run_cypher_query(self, query: str) -> dict[str, Any]:
        if query.strip().upper() == "NONE":
            return {
                "records": [],
                "record_count": 0,
            }

        cache_key = query.strip()
        if cache_key in self._query_cache:
            cached_records = self._query_cache[cache_key]
            logger.debug("KG query cache hit")
            return {
                "records": cached_records,
                "record_count": len(cached_records),
            }

        async with self._driver.session() as session:
            result = await session.run(query)
            records = await result.data()

        normalized_records = [dict(record) for record in records]
        self._query_cache[cache_key] = normalized_records

        return {
            "records": normalized_records,
            "record_count": len(normalized_records),
        }
