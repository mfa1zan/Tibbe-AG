"""
GPT-White / GPT-Black Evaluation Framework for PRO-MedGraph.

Implements formal evaluation metrics for thesis benchmarking:

White-Box Metrics (deterministic, computed from graph data):
    - path_coverage_score:     fraction of expected KG hops present
    - mapping_coherence:       consistency of mapping strengths in paths
    - multi_hop_depth:         average hop count across causal paths
    - causal_consistency:      variance of causal scores (lower = more consistent)

Black-Box Metrics (LLM-as-judge, using a separate evaluation LLM call):
    - logical_coherence:       does the answer follow from the evidence?
    - biomedical_validity:     are claims biomedically sound?
    - faith_consistency:       is Hadith framing respectful and accurate?
    - explanation_depth:       completeness and thoroughness of explanation

Results are stored as JSON log files for aggregation during thesis analysis.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_EVAL_LOG_DIR = Path(__file__).resolve().parents[2] / "logs" / "evaluations"


# ── Data classes ───────────────────────────────────────────────────────────────


@dataclass
class WhiteBoxMetrics:
    path_coverage_score: float = 0.0
    mapping_coherence: float = 0.0
    multi_hop_depth: float = 0.0
    causal_consistency: float = 0.0
    composite_white_score: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class BlackBoxMetrics:
    logical_coherence: float = 0.0
    biomedical_validity: float = 0.0
    faith_consistency: float = 0.0
    explanation_depth: float = 0.0
    composite_black_score: float = 0.0
    judge_raw_output: str = ""

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d.pop("judge_raw_output", None)  # Omit verbose raw from serialized output
        return d


@dataclass
class EvaluationResult:
    query: str = ""
    timestamp: str = ""
    white_box: WhiteBoxMetrics = field(default_factory=WhiteBoxMetrics)
    black_box: BlackBoxMetrics = field(default_factory=BlackBoxMetrics)
    combined_score: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "timestamp": self.timestamp,
            "white_box": self.white_box.to_dict(),
            "black_box": self.black_box.to_dict(),
            "combined_score": self.combined_score,
        }


# ── White-Box Metrics ──────────────────────────────────────────────────────────


def _compute_path_coverage(causal_paths: list[dict]) -> float:
    """
    Fraction of causal paths that have all 5 nodes (disease, ingredient,
    compound, drug_compound, drug) filled.  Full coverage → 1.0.
    """
    if not causal_paths:
        return 0.0

    full = 0
    for p in causal_paths:
        if not isinstance(p, dict):
            continue
        nodes = [p.get("disease"), p.get("ingredient"), p.get("compound"),
                 p.get("drug_compound"), p.get("drug")]
        if all(n for n in nodes):
            full += 1

    return round(full / len(causal_paths), 4) if causal_paths else 0.0


def _compute_mapping_coherence(causal_paths: list[dict]) -> float:
    """
    Coherence = 1 - normalised variance of mapping strengths.
    All-same-strength → 1.0.  Mixed → lower.
    """
    strength_map = {"IDENTICAL": 1.0, "LIKELY": 0.7, "WEAK": 0.35}
    scores = []
    for p in causal_paths:
        if isinstance(p, dict):
            label = p.get("mapping_strength_label", "WEAK")
            scores.append(strength_map.get(label, 0.35))

    if len(scores) <= 1:
        return 1.0 if scores else 0.0

    mean = sum(scores) / len(scores)
    variance = sum((s - mean) ** 2 for s in scores) / len(scores)
    # Normalize: max possible variance for [0.35, 1.0] range is ~0.105
    coherence = max(0.0, 1.0 - (variance / 0.11))
    return round(coherence, 4)


def _compute_multi_hop_depth(causal_paths: list[dict]) -> float:
    """Average hop count across all causal paths."""
    hops = [p.get("hop_count", 0) for p in causal_paths if isinstance(p, dict)]
    if not hops:
        return 0.0
    return round(sum(hops) / len(hops), 2)


def _compute_causal_consistency(causal_paths: list[dict]) -> float:
    """
    1 - normalised std-dev of causal scores.  
    Consistent scoring → 1.0.
    """
    scores = [p.get("causal_score", 0) for p in causal_paths if isinstance(p, dict)]
    if len(scores) <= 1:
        return 1.0 if scores else 0.0

    mean = sum(scores) / len(scores)
    std_dev = (sum((s - mean) ** 2 for s in scores) / len(scores)) ** 0.5
    # Max reasonable std_dev ≈ 0.5
    consistency = max(0.0, 1.0 - (std_dev / 0.5))
    return round(consistency, 4)


def compute_white_box_metrics(causal_paths: list[dict]) -> WhiteBoxMetrics:
    """Compute all white-box metrics from causal paths."""
    coverage = _compute_path_coverage(causal_paths)
    coherence = _compute_mapping_coherence(causal_paths)
    depth = _compute_multi_hop_depth(causal_paths)
    consistency = _compute_causal_consistency(causal_paths)

    # Composite: equal-weighted
    composite = round((coverage + coherence + min(depth / 4, 1.0) + consistency) / 4, 4)

    return WhiteBoxMetrics(
        path_coverage_score=coverage,
        mapping_coherence=coherence,
        multi_hop_depth=depth,
        causal_consistency=consistency,
        composite_white_score=composite,
    )


# ── Black-Box Metrics (LLM-as-Judge) ──────────────────────────────────────────


def _parse_judge_scores(raw: str) -> dict[str, float]:
    """Extract numeric scores from LLM judge output (expects JSON)."""
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            return {
                k: float(v) for k, v in parsed.items()
                if isinstance(v, (int, float)) and 0 <= float(v) <= 1
            }
    except (json.JSONDecodeError, ValueError):
        pass

    # Regex fallback: find "key": 0.X patterns
    scores: dict[str, float] = {}
    for match in re.finditer(r'"(\w+)"\s*:\s*([01]\.?\d*)', raw):
        key = match.group(1)
        try:
            val = float(match.group(2))
            if 0 <= val <= 1:
                scores[key] = val
        except ValueError:
            continue
    return scores


async def compute_black_box_metrics(
    query: str,
    answer: str,
    reasoning_summary: str,
    llm_service: Any,
    model: str,
) -> BlackBoxMetrics:
    """
    Use LLM-as-judge to rate the answer on 4 dimensions.

    Returns BlackBoxMetrics with scores in [0, 1].
    """
    system_prompt = (
        "You are an evaluation judge for a biomedical GraphRAG system.\n"
        "Rate the following answer on 4 dimensions, each from 0.0 to 1.0.\n"
        "Return ONLY a JSON object with these keys:\n"
        '  "logical_coherence", "biomedical_validity", "faith_consistency", "explanation_depth"\n'
        "No markdown, no explanation. JSON only."
    )

    user_prompt = (
        f"Query: {query}\n\n"
        f"Answer:\n{answer}\n\n"
        f"Evidence Summary:\n{reasoning_summary}\n\n"
        "Rate this answer now."
    )

    try:
        raw = await llm_service.generate_completion(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.1,
            model=model,
        )

        scores = _parse_judge_scores(raw)

        logical = scores.get("logical_coherence", 0.5)
        biomedical = scores.get("biomedical_validity", 0.5)
        faith = scores.get("faith_consistency", 0.5)
        depth = scores.get("explanation_depth", 0.5)

        composite = round((logical + biomedical + faith + depth) / 4, 4)

        return BlackBoxMetrics(
            logical_coherence=logical,
            biomedical_validity=biomedical,
            faith_consistency=faith,
            explanation_depth=depth,
            composite_black_score=composite,
            judge_raw_output=raw,
        )
    except Exception:
        logger.exception("Black-box evaluation failed")
        return BlackBoxMetrics(
            composite_black_score=0.5,
            judge_raw_output="evaluation_failed",
        )


# ── Full Evaluation ────────────────────────────────────────────────────────────


async def run_evaluation(
    query: str,
    answer: str,
    reasoning: dict,
    causal_paths: list[dict],
    llm_service: Any | None = None,
    model: str | None = None,
    enable_black_box: bool = False,
) -> EvaluationResult:
    """
    Run both white-box and optionally black-box evaluation.

    Args:
        enable_black_box: If True, uses LLM-as-judge (costs API tokens).
    """
    timestamp = datetime.now(timezone.utc).isoformat()

    white = compute_white_box_metrics(causal_paths)

    black = BlackBoxMetrics()
    if enable_black_box and llm_service and model:
        # Build a short reasoning summary for the judge
        summary_parts = []
        disease_name = reasoning.get("Disease", {}).get("name") if isinstance(reasoning.get("Disease"), dict) else None
        if disease_name:
            summary_parts.append(f"Disease: {disease_name}")
        n_ing = len(reasoning.get("Ingredients", []) or [])
        n_drugs = len(reasoning.get("Drugs", []) or [])
        n_hadith = len(reasoning.get("HadithReferences", []) or [])
        summary_parts.append(f"Ingredients: {n_ing}, Drugs: {n_drugs}, Hadith: {n_hadith}")
        summary_parts.append(f"Causal paths: {len(causal_paths)}")
        reasoning_summary = "; ".join(summary_parts)

        black = await compute_black_box_metrics(
            query=query,
            answer=answer,
            reasoning_summary=reasoning_summary,
            llm_service=llm_service,
            model=model,
        )

    # Combined score: 60% white-box + 40% black-box (if available)
    if enable_black_box and black.composite_black_score > 0:
        combined = round(0.6 * white.composite_white_score + 0.4 * black.composite_black_score, 4)
    else:
        combined = white.composite_white_score

    result = EvaluationResult(
        query=query,
        timestamp=timestamp,
        white_box=white,
        black_box=black,
        combined_score=combined,
    )

    logger.info(
        "Evaluation: white=%.3f black=%.3f combined=%.3f",
        white.composite_white_score,
        black.composite_black_score,
        combined,
    )

    return result


def save_evaluation(result: EvaluationResult) -> Path | None:
    """Persist evaluation result to JSON log file."""
    try:
        _EVAL_LOG_DIR.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
        path = _EVAL_LOG_DIR / f"eval_{ts}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)
        logger.info("Evaluation saved to %s", path)
        return path
    except Exception:
        logger.exception("Failed to save evaluation")
        return None


__all__ = [
    "compute_white_box_metrics",
    "compute_black_box_metrics",
    "run_evaluation",
    "save_evaluation",
    "EvaluationResult",
    "WhiteBoxMetrics",
    "BlackBoxMetrics",
]
