import re

from app.services.llm_service import LLMService


class ValidatorService:
    def __init__(
        self,
        llm_service: LLMService,
        model: str | None = None,
        judge_model: str | None = None,
    ) -> None:
        self._llm_service = llm_service
        self._model = model
        self._judge_model = judge_model

    async def validate_with_science(self, informed_answer: str) -> str:
        system_prompt = (
            "You are a scientific medical reviewer.\n"
            "Refine the answer by:\n"
            "- Adding scientific reasoning\n"
            "- Explaining mechanisms (if relevant)\n"
            "- Improving clarity\n"
            "Do not contradict the original knowledge.\n"
            "Do not invent unsupported claims.\n"
            "Return improved final answer."
        )

        user_prompt = f"Informed Answer:\n{informed_answer}"

        return await self._llm_service.generate_completion(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.2,
            model=self._model,
        )

    async def judge_score(self, user_query: str, final_answer: str) -> int | None:
        system_prompt = (
            "You are an answer quality judge for medical Q&A. "
            "Score the response from 1 to 100 based on completeness and scientific grounding. "
            "Return ONLY an integer."
        )
        user_prompt = f"User Query:\n{user_query}\n\nFinal Answer:\n{final_answer}"

        raw = await self._llm_service.generate_completion(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.0,
            model=self._judge_model,
        )

        match = re.search(r"\b(100|[1-9]?\d)\b", raw)
        if not match:
            return None

        score = int(match.group(1))
        return max(1, min(100, score))