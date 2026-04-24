"""
Experimental Logging System for PRO-MedGraph Thesis Evaluation.

Persists a structured JSON log entry for every query processed through
the pipeline.  Logs are stored under backend/logs/experiments/ with
one file per query, timestamped for replay and aggregation.

Logged fields:
    query, entities, paths_used, causal_scores, rationale_plan,
    dosage_validation, faith_alignment, final_answer, confidence_score,
    evaluation_metrics, timestamp
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_LOG_DIR = Path(__file__).resolve().parents[2] / "logs" / "experiments"


def _ensure_log_dir() -> Path:
    _LOG_DIR.mkdir(parents=True, exist_ok=True)
    return _LOG_DIR


def log_experiment(
    query: str,
    entities: dict[str, Any],
    reasoning: dict[str, Any] | None = None,
    rationale_plan: dict[str, Any] | None = None,
    causal_analysis: dict[str, Any] | None = None,
    dosage_validation: dict[str, Any] | None = None,
    faith_alignment: dict[str, Any] | None = None,
    a0_answer: str = "",
    af_answer: str = "",
    final_answer: str = "",
    confidence_score: float | None = None,
    evidence_strength: str = "weak",
    graph_paths_used: int = 0,
    evaluation_metrics: dict[str, Any] | None = None,
    safety: dict[str, Any] | None = None,
    reasoning_trace: dict[str, Any] | None = None,
    pipeline_metadata: dict[str, Any] | None = None,
) -> Path | None:
    """
    Write one experiment log entry to logs/experiments/.

    Returns the path to the saved file, or None on failure.
    """
    timestamp = datetime.now(timezone.utc).isoformat()

    entry = {
        "timestamp": timestamp,
        "query": query,
        "entities": entities or {},
        "rationale_plan": rationale_plan,
        "paths_used": graph_paths_used,
        "causal_scores": (
            causal_analysis.get("causal_ranking")
            if isinstance(causal_analysis, dict) else None
        ),
        "causal_paths_count": (
            len(causal_analysis.get("causal_paths", []))
            if isinstance(causal_analysis, dict) else 0
        ),
        "dosage_validation": (
            dosage_validation if isinstance(dosage_validation, dict) else None
        ),
        "faith_alignment": (
            faith_alignment if isinstance(faith_alignment, dict) else None
        ),
        "a0_answer_length": len(a0_answer) if a0_answer else 0,
        "af_answer_length": len(af_answer) if af_answer else 0,
        "final_answer": final_answer,
        "confidence_score": confidence_score,
        "evidence_strength": evidence_strength,
        "evaluation_metrics": evaluation_metrics,
        "safety": safety,
        "reasoning_trace": reasoning_trace,
        "pipeline_metadata": pipeline_metadata,
    }

    try:
        log_dir = _ensure_log_dir()
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
        filename = f"exp_{ts}.json"
        path = log_dir / filename

        with open(path, "w", encoding="utf-8") as f:
            json.dump(entry, f, indent=2, ensure_ascii=False, default=str)

        logger.info("Experiment logged to %s", path)
        return path
    except Exception:
        logger.exception("Failed to write experiment log")
        return None


def load_all_experiments(log_dir: Path | None = None) -> list[dict]:
    """Load all experiment logs from disk for aggregation."""
    target = log_dir or _LOG_DIR
    if not target.exists():
        return []

    entries: list[dict] = []
    for file_path in sorted(target.glob("exp_*.json")):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                entries.append(json.load(f))
        except Exception:
            logger.warning("Skipping malformed log file: %s", file_path)
            continue

    return entries


def aggregate_experiment_stats(entries: list[dict] | None = None) -> dict[str, Any]:
    """
    Compute summary statistics across all logged experiments.
    Useful for thesis result tables.
    """
    logs = entries or load_all_experiments()
    if not logs:
        return {"total_queries": 0}

    confidences = [e.get("confidence_score") for e in logs if isinstance(e.get("confidence_score"), (int, float))]
    strengths = [e.get("evidence_strength", "weak") for e in logs]
    paths = [e.get("paths_used", 0) for e in logs if isinstance(e.get("paths_used"), (int, float))]

    causal_scores = []
    for e in logs:
        cs = e.get("causal_scores")
        if isinstance(cs, dict) and "avg_causal_score" in cs:
            causal_scores.append(cs["avg_causal_score"])

    faith_scores = []
    for e in logs:
        fa = e.get("faith_alignment")
        if isinstance(fa, dict) and "faith_alignment_score" in fa:
            faith_scores.append(fa["faith_alignment_score"])

    return {
        "total_queries": len(logs),
        "avg_confidence": round(sum(confidences) / len(confidences), 4) if confidences else None,
        "avg_paths_used": round(sum(paths) / len(paths), 2) if paths else 0,
        "avg_causal_score": round(sum(causal_scores) / len(causal_scores), 4) if causal_scores else None,
        "avg_faith_alignment": round(sum(faith_scores) / len(faith_scores), 4) if faith_scores else None,
        "strength_distribution": {
            "strong": sum(1 for s in strengths if s == "strong"),
            "moderate": sum(1 for s in strengths if s == "moderate"),
            "weak": sum(1 for s in strengths if s == "weak"),
        },
    }


__all__ = [
    "log_experiment",
    "load_all_experiments",
    "aggregate_experiment_stats",
]
