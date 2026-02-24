import re
from typing import Iterable

SYNONYM_MAP: dict[str, list[str]] = {
    "fever": ["fever", "pyrexia"],
    "high blood pressure": ["high blood pressure", "hypertension"],
    "heart attack": ["heart attack", "myocardial infarction"],
    "diabetes": ["diabetes", "diabetes mellitus"],
}


def normalize_text(text: str) -> str:
    lowered = text.lower().strip()
    no_punct = re.sub(r"[^\w\s]", " ", lowered)
    normalized = re.sub(r"\s+", " ", no_punct).strip()
    return normalized


def expand_terms(normalized_text: str) -> list[str]:
    expanded: list[str] = [normalized_text]

    for phrase, synonyms in SYNONYM_MAP.items():
        if phrase in normalized_text:
            expanded.extend(synonyms)

    unique_terms = list(dict.fromkeys([term.strip() for term in expanded if term.strip()]))
    return unique_terms


def extract_candidate_diseases(normalized_text: str, expanded_terms: Iterable[str]) -> list[str]:
    candidates = [normalized_text]

    for term in expanded_terms:
        candidates.append(term)

    deduped = list(dict.fromkeys([term for term in candidates if term]))
    return deduped
