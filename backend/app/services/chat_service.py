import logging

from app.services.answer_generator import AnswerGeneratorService
from app.services.cypher_generator import CypherGeneratorService
from app.services.kg_service import KGService
from app.services.llm_service import LLMService
from app.services.validator import ValidatorService

logger = logging.getLogger(__name__)

FALLBACK_REPLY = "Sorry, I could not process your request."


class ChatService:
    def __init__(
        self,
        kg_service: KGService,
        llm_service: LLMService,
        cypher_generator: CypherGeneratorService,
        answer_generator: AnswerGeneratorService,
        validator_service: ValidatorService,
        enable_judge_scoring: bool = False,
    ) -> None:
        self._kg_service = kg_service
        self._llm_service = llm_service
        self._cypher_generator = cypher_generator
        self._answer_generator = answer_generator
        self._validator_service = validator_service
        self._enable_judge_scoring = enable_judge_scoring

    async def process_message(
        self,
        user_message: str,
        session_id: str | None = None,
    ) -> tuple[str, list[str], str | None, int, int | None]:
        del session_id

        try:
            generated_cypher = await self._cypher_generator.generate_cypher_query(user_message)
            logger.info("Pipeline Cypher='%s'", generated_cypher)

            try:
                kg_results = await self._kg_service.run_cypher_query(generated_cypher)
            except Exception:
                logger.exception("Cypher execution failed. Falling back to direct LLM response.")
                fallback = await self._llm_service.generate_fallback_answer(user_message)
                return fallback, [], generated_cypher, 0, None

            informed_answer = await self._answer_generator.generate_informed_answer(
                user_query=user_message,
                kg_results=kg_results,
            )
            final_answer = await self._validator_service.validate_with_science(informed_answer)

            judge_score = None
            if self._enable_judge_scoring:
                try:
                    judge_score = await self._validator_service.judge_score(
                        user_query=user_message,
                        final_answer=final_answer,
                    )
                except Exception:
                    logger.exception("Judge scoring failed")

            provenance = self._extract_provenance(kg_results.get("records", []))
            result_count = int(kg_results.get("record_count", 0))

            return final_answer, provenance, generated_cypher, result_count, judge_score
        except Exception:
            logger.exception("Failed processing query")
            try:
                fallback = await self._llm_service.generate_fallback_answer(user_message)
            except Exception:
                fallback = FALLBACK_REPLY

            return fallback, [], None, 0, None

    @staticmethod
    def _extract_provenance(records: list[dict]) -> list[str]:
        provenance_keys = (
            "ingredient",
            "drug",
            "book",
            "chemical_compound",
            "drug_compound",
            "disease",
            "name",
        )

        values: set[str] = set()
        for record in records:
            for key in provenance_keys:
                value = record.get(key)
                if isinstance(value, str) and value.strip():
                    values.add(value.strip())

        return sorted(values)[:20]
