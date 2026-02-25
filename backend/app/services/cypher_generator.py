import logging

from app.services.llm_service import LLMService

logger = logging.getLogger(__name__)


class CypherGeneratorService:
    def __init__(self, llm_service: LLMService, schema_text: str, model: str | None = None) -> None:
        self._llm_service = llm_service
        self._schema_text = schema_text
        self._model = model

    async def generate_cypher_query(self, user_query: str) -> str:
        system_prompt = (
            "You are a Neo4j Cypher expert.\n"
            "Use ONLY the provided schema.\n"
            "Do not invent labels or relationships.\n"
            "Return ONLY a valid Cypher query.\n"
            "If query cannot be answered from schema, return: NONE."
        )

        user_prompt = (
            f"Knowledge Graph Schema:\n{self._schema_text}\n\n"
            f"User Question:\n{user_query}\n\n"
            "Output requirements:\n"
            "- Only Cypher or NONE\n"
            "- No markdown\n"
            "- No explanations\n"
            "- No backticks"
        )

        query = await self._llm_service.generate_completion(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.05,
            model=self._model,
        )

        cleaned = query.strip().strip("`").strip()
        logger.info("Generated Cypher: %s", cleaned)
        return cleaned or "NONE"