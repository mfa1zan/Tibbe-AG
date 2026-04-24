"""
Faith–Science Alignment Scoring for PRO-MedGraph.

Evaluates how well a pipeline response aligns with both Islamic faith
sources and modern scientific evidence.  The score is used:
1. In the final confidence formula as an additive component
2. In the evaluation framework as a white-box metric
3. In the reasoning trace for interpretability

Scoring dimensions:
    hadith_presence       – Is there a relevant Hadith reference?
    correct_framing       – Does the answer avoid divine exclusivity / miracle claims?
    scientific_backing    – Are there ChemicalCompound→Drug mappings present?
    no_miracle_claims     – Absence of unsupported miracle language?

    faith_alignment_score = 0.35 * hadith + 0.25 * framing + 0.25 * science + 0.15 * no_miracle

Pipeline position:
    ... → Af Validation → **Faith Alignment Scorer** → Safety Scorer
"""

from __future__ import annotations

import logging
import re
from dataclasses import asdict, dataclass
from typing import Any

logger = logging.getLogger(__name__)

# ── Miracle / problematic framing patterns ──────────────────────────────────────

_MIRACLE_PATTERNS = [
    re.compile(r"\b(guaranteed?\s+cure|miracle\s+cure|divine\s+healing)\b", re.IGNORECASE),
    re.compile(r"\b(will\s+definitely\s+cure|100%\s+effective|certain\s+to\s+heal)\b", re.IGNORECASE),
    re.compile(r"\b(god\s+will\s+cure|allah\s+guarantees|prophet\s+promised\s+cure)\b", re.IGNORECASE),
    re.compile(r"\b(replaces?\s+modern\s+medicine|no\s+need\s+for\s+doctor)\b", re.IGNORECASE),
    re.compile(r"\b(scientifically\s+proven\s+by\s+(quran|hadith|sunnah))\b", re.IGNORECASE),
]

_GOOD_FRAMING_PATTERNS = [
    re.compile(r"\b(consult\s+(a\s+)?doctor|medical\s+professional|healthcare)\b", re.IGNORECASE),
    re.compile(r"\b(not\s+a\s+substitute|complement(ary)?|alongside\s+modern)\b", re.IGNORECASE),
    re.compile(r"\b(with\s+caution|further\s+research|more\s+study)\b", re.IGNORECASE),
    re.compile(r"\b(traditional(ly)?|prophetic\s+medicine|tibb)\b", re.IGNORECASE),
]


# ── Data class ──────────────────────────────────────────────────────────────────


@dataclass
class FaithAlignmentResult:
    """Structured faith-science alignment evaluation."""

    hadith_present: bool = False
    hadith_count: int = 0
    hadith_score: float = 0.0

    correct_framing: bool = True
    framing_score: float = 0.5

    scientific_backing: bool = False
    scientific_backing_score: float = 0.0

    miracle_claims_found: bool = False
    no_miracle_score: float = 1.0

    faith_alignment_score: float = 0.0
    faith_alignment_notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ── Scoring functions ───────────────────────────────────────────────────────────


def _score_hadith_presence(reasoning: dict) -> tuple[bool, int, float]:
    """Score based on Hadith reference presence and count."""
    hadith_refs = reasoning.get("HadithReferences", [])
    if not isinstance(hadith_refs, list):
        hadith_refs = []

    count = len(hadith_refs)
    present = count > 0

    if count == 0:
        score = 0.0
    elif count == 1:
        score = 0.7
    elif count <= 3:
        score = 0.85
    else:
        score = 1.0

    return present, count, round(score, 2)


def _score_scientific_backing(reasoning: dict) -> tuple[bool, float]:
    """Score based on presence of compound→drug biochemical mappings."""
    mappings = reasoning.get("BiochemicalMappings", [])
    if not isinstance(mappings, list):
        mappings = []

    if not mappings:
        return False, 0.0

    # Check for strong / moderate mappings
    strengths = []
    for m in mappings:
        if isinstance(m, dict):
            s = m.get("mapping_strength")
            if isinstance(s, str):
                strengths.append(s)

    if not strengths:
        return False, 0.2

    identical = sum(1 for s in strengths if s == "IDENTICAL")
    likely = sum(1 for s in strengths if s == "LIKELY")

    if identical > 0:
        score = 1.0
    elif likely > 0:
        score = 0.75
    else:
        score = 0.4

    return True, round(score, 2)


def _score_answer_framing(answer_text: str) -> tuple[bool, float]:
    """
    Evaluate whether the answer uses correct framing:
    - Does it include appropriate caveats?
    - Does it mention consulting professionals?
    """
    if not isinstance(answer_text, str) or not answer_text.strip():
        return True, 0.5  # Neutral if no answer yet

    good_matches = sum(1 for p in _GOOD_FRAMING_PATTERNS if p.search(answer_text))

    if good_matches >= 3:
        return True, 1.0
    elif good_matches >= 2:
        return True, 0.85
    elif good_matches >= 1:
        return True, 0.7
    else:
        return False, 0.3


def _score_no_miracle_claims(answer_text: str) -> tuple[bool, float]:
    """Check for problematic miracle/exclusivity language."""
    if not isinstance(answer_text, str) or not answer_text.strip():
        return False, 1.0  # No claims if no text

    violations = sum(1 for p in _MIRACLE_PATTERNS if p.search(answer_text))

    if violations == 0:
        return False, 1.0
    elif violations == 1:
        return True, 0.4
    else:
        return True, 0.1


# ── Main entry point ───────────────────────────────────────────────────────────


def score_faith_alignment(
    reasoning: dict,
    answer_text: str = "",
) -> FaithAlignmentResult:
    """
    Compute faith-science alignment score.

    Formula:
        faith_alignment_score =
            0.35 * hadith_score +
            0.25 * framing_score +
            0.25 * scientific_backing_score +
            0.15 * no_miracle_score

    Args:
        reasoning: structured graph reasoning dict
        answer_text: the final answer text (A0 or Af) to check framing

    Returns:
        FaithAlignmentResult with all sub-scores and composite score
    """
    safe_reasoning = reasoning if isinstance(reasoning, dict) else {}

    hadith_present, hadith_count, hadith_score = _score_hadith_presence(safe_reasoning)
    sci_backed, sci_score = _score_scientific_backing(safe_reasoning)
    correct_frame, frame_score = _score_answer_framing(answer_text)
    miracle_found, no_miracle_score = _score_no_miracle_claims(answer_text)

    composite = round(
        0.35 * hadith_score
        + 0.25 * frame_score
        + 0.25 * sci_score
        + 0.15 * no_miracle_score,
        4,
    )

    # Build explanatory note
    notes_parts: list[str] = []
    if hadith_present:
        notes_parts.append(f"{hadith_count} Hadith reference(s) found")
    else:
        notes_parts.append("No Hadith references present")

    if sci_backed:
        notes_parts.append("Scientific compound-drug backing present")
    else:
        notes_parts.append("Limited scientific compound-drug backing")

    if not correct_frame:
        notes_parts.append("Answer framing could include more professional caveats")

    if miracle_found:
        notes_parts.append("WARNING: Potential miracle/exclusivity claims detected")

    result = FaithAlignmentResult(
        hadith_present=hadith_present,
        hadith_count=hadith_count,
        hadith_score=hadith_score,
        correct_framing=correct_frame,
        framing_score=frame_score,
        scientific_backing=sci_backed,
        scientific_backing_score=sci_score,
        miracle_claims_found=miracle_found,
        no_miracle_score=no_miracle_score,
        faith_alignment_score=composite,
        faith_alignment_notes="; ".join(notes_parts),
    )

    logger.info(
        "Faith alignment: score=%.3f hadith=%.2f framing=%.2f science=%.2f miracle=%.2f",
        composite, hadith_score, frame_score, sci_score, no_miracle_score,
    )

    return result


__all__ = [
    "score_faith_alignment",
    "FaithAlignmentResult",
]
