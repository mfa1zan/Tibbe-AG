import logging
from collections import defaultdict
from typing import Any

from cachetools import TTLCache
from neo4j import AsyncGraphDatabase

logger = logging.getLogger(__name__)


class KGService:
    def __init__(self, uri: str, username: str, password: str, ttl_seconds: int, maxsize: int) -> None:
        self._driver = AsyncGraphDatabase.driver(uri, auth=(username, password))
        self._cache = TTLCache(maxsize=maxsize, ttl=ttl_seconds)

    async def close(self) -> None:
        await self._driver.close()

    async def fetch_context(self, disease_candidates: list[str]) -> tuple[str, list[str], list[dict[str, Any]]]:
        aggregated_rows: list[dict[str, Any]] = []

        for disease_name in disease_candidates:
            cache_key = disease_name.lower()

            if cache_key in self._cache:
                rows = self._cache[cache_key]
                logger.debug("KG cache hit for disease='%s'", disease_name)
            else:
                rows = await self._query_kg_for_disease(disease_name)
                self._cache[cache_key] = rows
                logger.debug("KG cache miss for disease='%s', fetched %s rows", disease_name, len(rows))

            if rows:
                aggregated_rows.extend(rows)

        context_text, provenance = self._format_context(aggregated_rows)
        return context_text, provenance, aggregated_rows

    async def _query_kg_for_disease(self, disease_name: str) -> list[dict[str, Any]]:
        query = """
        MATCH (d:Disease)
        WHERE toLower(d.name) = toLower($disease_name)
        OPTIONAL MATCH (i:Ingredient)-[:CURES]->(d)
        OPTIONAL MATCH (i)-[:CONTAINS]->(cc:ChemicalCompound)
        OPTIONAL MATCH (cc)-[:IS_IDENTICAL_TO|IS_LIKELY_EQUIVALENT_TO|IS_WEAK_MATCH_TO]->(dcc:DrugChemicalCompound)
        OPTIONAL MATCH (dcc)<-[:CONTAINS]-(drug:Drug)
        OPTIONAL MATCH (drug)-[:IS_IN_BOOK]->(book:Book)
        RETURN d.name AS disease, i.name AS ingredient, cc.name AS chemical_compound,
               dcc.name AS drug_compound, drug.name AS drug, book.name AS book
        LIMIT 120
        """

        async with self._driver.session() as session:
            result = await session.run(query, disease_name=disease_name)
            records = await result.data()

        cleaned = [
            {
                "disease": row.get("disease"),
                "ingredient": row.get("ingredient"),
                "chemical_compound": row.get("chemical_compound"),
                "drug_compound": row.get("drug_compound"),
                "drug": row.get("drug"),
                "book": row.get("book"),
            }
            for row in records
        ]

        return cleaned

    def _format_context(self, rows: list[dict[str, Any]]) -> tuple[str, list[str]]:
        if not rows:
            return "No specific graph context found.", []

        grouped: dict[str, set[str]] = defaultdict(set)

        for row in rows:
            for key in ("ingredient", "chemical_compound", "drug_compound", "drug", "book"):
                value = row.get(key)
                if value:
                    grouped[key].add(value)

        context_parts: list[str] = []
        for key in ("ingredient", "chemical_compound", "drug_compound", "drug", "book"):
            values = sorted(grouped.get(key, set()))
            if values:
                context_parts.append(f"{key.replace('_', ' ').title()}: {', '.join(values[:12])}")

        provenance: list[str] = sorted(
            set(
                list(grouped.get("ingredient", set()))
                + list(grouped.get("drug", set()))
                + list(grouped.get("book", set()))
            )
        )[:12]

        context = " | ".join(context_parts) if context_parts else "No specific graph context found."
        return context, provenance
