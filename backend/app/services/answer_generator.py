import json

from app.services.llm_service import LLMService


class AnswerGeneratorService:
    def __init__(self, llm_service: LLMService, model: str | None = None) -> None:
        self._llm_service = llm_service
        self._model = model

    async def generate_informed_answer(self, user_query: str, kg_results: dict) -> str:
        system_prompt = (
            "You are a medical knowledge assistant.\n"
            "You must answer ONLY using the provided knowledge graph results.\n"
            "If the results are empty, say you do not have enough structured knowledge.\n"
            "Do not hallucinate.\n"
            "Be clear and structured."
        )

        user_prompt = (
            f"User Query:\n{user_query}\n\n"
            f"Knowledge Graph Results (JSON):\n{json.dumps(kg_results, ensure_ascii=False, indent=2)}"
        )

        return await self._llm_service.generate_completion(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.2,
            model=self._model,
        )