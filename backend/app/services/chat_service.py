import logging
from collections import defaultdict

from app.services.kg_service import KGService
from app.services.llm_service import LLMService
from app.services.preprocess import extract_candidate_diseases, expand_terms, normalize_text

logger = logging.getLogger(__name__)

FALLBACK_REPLY = "Sorry, I could not process your request."


class ChatService:
    def __init__(self, kg_service: KGService, llm_service: LLMService) -> None:
        self._kg_service = kg_service
        self._llm_service = llm_service
        self._session_memory: dict[str, list[dict[str, str]]] = defaultdict(list)

    async def process_message(self, user_message: str, session_id: str | None = None) -> tuple[str, list[str]]:
        normalized = normalize_text(user_message)
        expanded = expand_terms(normalized)
        disease_candidates = extract_candidate_diseases(normalized, expanded)

        logger.info("User query='%s' normalized='%s' candidates=%s", user_message, normalized, disease_candidates)

        try:
            kg_context, provenance, kg_rows = await self._kg_service.fetch_context(disease_candidates)
            logger.info("KG context='%s' rows=%s", kg_context, len(kg_rows))

            session_history: list[dict[str, str]] = []
            if session_id:
                session_history = self._session_memory.get(session_id, [])

            reply = await self._llm_service.generate_reply(
                user_message=user_message,
                kg_context=kg_context,
                session_history=session_history,
            )
            logger.info("LLM response='%s'", reply)

            if session_id:
                self._session_memory[session_id].append({"role": "user", "content": user_message})
                self._session_memory[session_id].append({"role": "bot", "content": reply})
                self._session_memory[session_id] = self._session_memory[session_id][-12:]

            return reply, provenance
        except Exception:
            logger.exception("Failed processing query")
            return FALLBACK_REPLY, []
