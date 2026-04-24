#!/usr/bin/env python3
"""
PRO-MedGraph Benchmark Runner

Usage:
    python run_benchmark.py                          # run all 50 questions
    python run_benchmark.py --limit 10               # run first 10 questions
    python run_benchmark.py --categories disease_treatment causal_reasoning
    python run_benchmark.py --enable-blackbox         # enable LLM-as-judge

Outputs:
    backend/logs/benchmark/benchmark_<timestamp>.csv
    backend/logs/benchmark/benchmark_<timestamp>_summary.json
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path

# Ensure backend is importable
sys.path.insert(0, str(Path(__file__).resolve().parent))

from app.config import get_settings  # noqa: E402
from app.logging_config import configure_logging  # noqa: E402
from app.services.orchestrator import GraphRAGOrchestrator  # noqa: E402


def load_questions(path: str | Path, limit: int | None = None, categories: list[str] | None = None) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        questions = json.load(f)
    if categories:
        questions = [q for q in questions if q.get("category") in categories]
    if limit:
        questions = questions[:limit]
    return questions


async def run_single(orch: GraphRAGOrchestrator, question: dict) -> dict:
    """Run a single benchmark question and collect metrics."""
    qid = question["id"]
    query = question["query"]
    category = question.get("category", "unknown")
    difficulty = question.get("difficulty", "unknown")

    start = time.perf_counter()
    try:
        result = await orch.process_user_query_with_context_async(query)
        elapsed = time.perf_counter() - start

        output = result.get("output", {})
        trace = output.get("reasoning_trace") or result.get("reasoning_trace", {})

        # Extract key metrics
        confidence_breakdown = trace.get("confidence_breakdown", {})
        eval_metrics = trace.get("evaluation_metrics") or {}

        return {
            "id": qid,
            "query": query,
            "category": category,
            "difficulty": difficulty,
            "latency_s": round(elapsed, 2),
            "evidence_strength": output.get("evidence_strength", "weak"),
            "graph_paths_used": output.get("graph_paths_used", 0),
            "confidence_score": output.get("confidence_score"),
            "faith_alignment_score": trace.get("faith_alignment_score"),
            "causal_avg": confidence_breakdown.get("causal_avg"),
            "dosage_alignment": confidence_breakdown.get("dosage_alignment"),
            "mapping_strength": confidence_breakdown.get("mapping_strength"),
            "multi_hop_activated": trace.get("multi_hop_activated", False),
            "pipeline_stages": len(trace.get("pipeline_stages", [])),
            "white_box_score": eval_metrics.get("white_box", {}).get("composite_white_score"),
            "black_box_score": eval_metrics.get("black_box", {}).get("composite_black_score"),
            "combined_score": eval_metrics.get("combined_score"),
            "answer_length": len(output.get("final_answer", "")),
            "error": None,
        }
    except Exception as e:
        elapsed = time.perf_counter() - start
        return {
            "id": qid,
            "query": query,
            "category": category,
            "difficulty": difficulty,
            "latency_s": round(elapsed, 2),
            "evidence_strength": "error",
            "graph_paths_used": 0,
            "confidence_score": None,
            "faith_alignment_score": None,
            "causal_avg": None,
            "dosage_alignment": None,
            "mapping_strength": None,
            "multi_hop_activated": False,
            "pipeline_stages": 0,
            "white_box_score": None,
            "black_box_score": None,
            "combined_score": None,
            "answer_length": 0,
            "error": str(e),
        }


def compute_summary(results: list[dict]) -> dict:
    """Aggregate benchmark statistics."""
    n = len(results)
    errors = sum(1 for r in results if r["error"])

    def safe_avg(key):
        vals = [r[key] for r in results if r[key] is not None]
        return round(sum(vals) / len(vals), 4) if vals else None

    def strength_dist():
        dist = {}
        for r in results:
            s = r["evidence_strength"]
            dist[s] = dist.get(s, 0) + 1
        return dist

    def category_breakdown():
        cats = {}
        for r in results:
            c = r["category"]
            if c not in cats:
                cats[c] = {"count": 0, "avg_confidence": [], "avg_latency": [], "errors": 0}
            cats[c]["count"] += 1
            if r["confidence_score"] is not None:
                cats[c]["avg_confidence"].append(r["confidence_score"])
            cats[c]["avg_latency"].append(r["latency_s"])
            if r["error"]:
                cats[c]["errors"] += 1
        # Reduce
        for c in cats:
            conf = cats[c]["avg_confidence"]
            cats[c]["avg_confidence"] = round(sum(conf) / len(conf), 4) if conf else None
            lat = cats[c]["avg_latency"]
            cats[c]["avg_latency"] = round(sum(lat) / len(lat), 2) if lat else None
        return cats

    return {
        "total_questions": n,
        "errors": errors,
        "avg_latency_s": safe_avg("latency_s"),
        "avg_confidence": safe_avg("confidence_score"),
        "avg_faith_alignment": safe_avg("faith_alignment_score"),
        "avg_causal": safe_avg("causal_avg"),
        "avg_dosage_alignment": safe_avg("dosage_alignment"),
        "avg_mapping_strength": safe_avg("mapping_strength"),
        "avg_graph_paths": safe_avg("graph_paths_used"),
        "avg_white_box_score": safe_avg("white_box_score"),
        "avg_black_box_score": safe_avg("black_box_score"),
        "avg_combined_score": safe_avg("combined_score"),
        "multi_hop_activated_count": sum(1 for r in results if r["multi_hop_activated"]),
        "evidence_strength_distribution": strength_dist(),
        "category_breakdown": category_breakdown(),
    }


async def main():
    parser = argparse.ArgumentParser(description="PRO-MedGraph Benchmark Runner")
    parser.add_argument("--questions", default="benchmark_questions.json", help="Path to questions JSON")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of questions")
    parser.add_argument("--categories", nargs="*", default=None, help="Filter by category")
    parser.add_argument("--enable-eval", action="store_true", help="Enable white-box evaluation")
    parser.add_argument("--enable-blackbox", action="store_true", help="Enable LLM-as-judge black-box eval")
    args = parser.parse_args()

    settings = get_settings()
    configure_logging(settings.log_level)

    questions = load_questions(args.questions, args.limit, args.categories)
    print(f"Loaded {len(questions)} benchmark questions")

    orch = GraphRAGOrchestrator(
        enable_evaluation=args.enable_eval or args.enable_blackbox,
        enable_black_box_eval=args.enable_blackbox,
    )

    results = []
    for i, q in enumerate(questions, 1):
        print(f"[{i}/{len(questions)}] Q{q['id']}: {q['query'][:60]}...", flush=True)
        row = await run_single(orch, q)
        results.append(row)
        status = "OK" if not row["error"] else f"ERR: {row['error'][:50]}"
        print(
            f"  -> {status} | conf={row['confidence_score']} | paths={row['graph_paths_used']} "
            f"| faith={row['faith_alignment_score']} | {row['latency_s']}s"
        )

    # Save CSV
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(__file__).resolve().parent / "logs" / "benchmark"
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / f"benchmark_{ts}.csv"
    if results:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        print(f"\nCSV saved: {csv_path}")

    # Save summary
    summary = compute_summary(results)
    summary_path = out_dir / f"benchmark_{ts}_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"Summary saved: {summary_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)
    print(f"  Total Questions:       {summary['total_questions']}")
    print(f"  Errors:                {summary['errors']}")
    print(f"  Avg Latency:           {summary['avg_latency_s']}s")
    print(f"  Avg Confidence:        {summary['avg_confidence']}")
    print(f"  Avg Faith Alignment:   {summary['avg_faith_alignment']}")
    print(f"  Avg Causal Score:      {summary['avg_causal']}")
    print(f"  Avg Dosage Alignment:  {summary['avg_dosage_alignment']}")
    print(f"  Avg White-Box Score:   {summary['avg_white_box_score']}")
    print(f"  Avg Black-Box Score:   {summary['avg_black_box_score']}")
    print(f"  Avg Combined Score:    {summary['avg_combined_score']}")
    print(f"  Multi-Hop Activated:   {summary['multi_hop_activated_count']}")
    print(f"\n  Evidence Distribution: {summary['evidence_strength_distribution']}")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
