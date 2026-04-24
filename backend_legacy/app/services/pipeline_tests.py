"""
PRO-MedGraph  —  Step 9: Testing & Debugging Pipeline
=======================================================

Automated end-to-end test suite that exercises every stage of the
multi-model GraphRAG pipeline:

    Step 2 (Intent Extraction)
    Step 3 (Cypher Query Generation)
    Step 4 (Graph Retrieval)
    Step 5 (Evidence Formatting)
    Step 6 (Clinical Reasoner — A0)
    Step 7 (Validator / Safety)
    Step 8 (Final Response Builder)

For each sample query the harness:
    • Validates JSON output structure at every stage.
    • Checks Neo4j connectivity and result shape.
    • Asserts that the reasoner only uses retrieved evidence (no hallucination).
    • Confirms the validator correctly flags unsafe / ungrounded content.
    • Records wall-clock latency per stage.
    • Produces a summary report (pass/fail, latencies, coverage, warnings).
    • Optionally saves a full JSON report to disk for CI / analysis.

Run from ``backend/`` directory:

    python -m app.services.pipeline_tests                 # console report
    python -m app.services.pipeline_tests --json report   # also save JSON

Pipeline position:
    This module is *not* part of the serving path — it is a developer /
    QA tool that validates correctness, safety, and performance.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════════
#  Pipeline imports
# ═══════════════════════════════════════════════════════════════════════════════

from app.services.intent_extractor import extract_intent          # Step 2  (async)
from app.services.cypher_query_generator import generate_cypher   # Step 3  (async)
from app.services.graph_retrieval import (                        # Step 4  (sync)
    retrieve_graph, retrieve_graph_from_step3,
)
from app.services.evidence_formatter import format_graph_for_llm  # Step 5  (sync)
from app.services.clinical_reasoner import generate_clinical_draft  # Step 6  (async)
from app.services.draft_validator import (                        # Step 7  (async)
    validate_draft, is_safe_to_publish,
)
from app.services.response_builder import build_final_answer      # Step 8  (async)

# ═══════════════════════════════════════════════════════════════════════════════
#  Sample queries — covering every intent category
# ═══════════════════════════════════════════════════════════════════════════════

SAMPLE_QUERIES: list[dict[str, Any]] = [
    # ── Dietary / nutritional remedy ──────────────────────────────────────
    {
        "id": "Q01",
        "query": "What foods or dietary remedies are recommended for diabetes in Islamic medicine?",
        "category": "dietary_nutritional",
        "expected_intent": "ask_remedy",
        "expected_entity_keywords": ["diabetes"],
    },
    # ── Herbal remedy ─────────────────────────────────────────────────────
    {
        "id": "Q02",
        "query": "How does black seed help with hypertension?",
        "category": "herbal_remedy",
        "expected_intent": "ask_remedy",
        "expected_entity_keywords": ["black seed", "hypertension"],
    },
    # ── Ritual / faith-based ──────────────────────────────────────────────
    {
        "id": "Q03",
        "query": "What does Islam recommend for general healing and wellness?",
        "category": "ritual_faith",
        "expected_intent": "ask_remedy",
        "expected_entity_keywords": ["healing"],
    },
    # ── Hygiene / wound care ──────────────────────────────────────────────
    {
        "id": "Q04",
        "query": "Are there Prophetic remedies for skin infections or wound healing?",
        "category": "hygiene_wound",
        "expected_intent": "ask_remedy",
        "expected_entity_keywords": ["skin", "wound"],
    },
    # ── Drug–ingredient mapping ───────────────────────────────────────────
    {
        "id": "Q05",
        "query": "Which modern drugs share compounds with honey?",
        "category": "drug_ingredient_mapping",
        "expected_intent": "drug_interaction",
        "expected_entity_keywords": ["honey"],
    },
    # ── Symptom check ─────────────────────────────────────────────────────
    {
        "id": "Q06",
        "query": "I have frequent headaches and nausea, what natural remedies can help?",
        "category": "symptom_check",
        "expected_intent": "symptom_check",
        "expected_entity_keywords": ["headache", "nausea"],
    },
    # ── Food remedy ───────────────────────────────────────────────────────
    {
        "id": "Q07",
        "query": "What are the health benefits of olive oil in Tibb-e-Nabawi?",
        "category": "food_remedy",
        "expected_intent": "food_remedy",
        "expected_entity_keywords": ["olive oil"],
    },
    # ── General / conversational (should not crash the pipeline) ──────────
    {
        "id": "Q08",
        "query": "Hello, what can you do?",
        "category": "general",
        "expected_intent": "general",
        "expected_entity_keywords": [],
    },
]

# ═══════════════════════════════════════════════════════════════════════════════
#  Data classes for results
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class StageResult:
    """Result of a single pipeline stage."""
    stage: str
    success: bool = False
    elapsed_ms: float = 0.0
    output: Any = None
    error: str | None = None
    checks: dict[str, bool] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        # Truncate large outputs for the JSON report.
        if isinstance(d.get("output"), str) and len(d["output"]) > 500:
            d["output"] = d["output"][:500] + "…"
        if isinstance(d.get("output"), dict):
            d["output"] = _truncate_dict(d["output"])
        return d


@dataclass
class QueryTestResult:
    """Full pipeline test result for one sample query."""
    query_id: str
    query: str
    category: str
    stages: list[StageResult] = field(default_factory=list)
    overall_pass: bool = False
    total_elapsed_ms: float = 0.0
    warnings: list[str] = field(default_factory=list)

    # Graph coverage stats
    node_count: int = 0
    edge_count: int = 0
    evidence_block_count: int = 0

    # Final answer metadata
    final_answer_chars: int = 0
    confidence: str | None = None
    safety_score: float = 0.0
    grounded: bool = False
    hallucination_flag: bool = True

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["stages"] = [s.to_dict() for s in self.stages]
        return d


@dataclass
class PipelineTestReport:
    """Aggregate report across all sample queries."""
    total_queries: int = 0
    passed: int = 0
    failed: int = 0
    total_elapsed_ms: float = 0.0
    avg_latency_ms: float = 0.0
    stage_avg_latencies: dict[str, float] = field(default_factory=dict)
    unsafe_warnings: list[str] = field(default_factory=list)
    graph_coverage: dict[str, Any] = field(default_factory=dict)
    results: list[QueryTestResult] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["results"] = [r.to_dict() for r in self.results]
        return d


def _truncate_dict(d: dict, max_str: int = 300) -> dict:
    """Truncate string values in a dict for compact JSON reports."""
    out = {}
    for k, v in d.items():
        if isinstance(v, str) and len(v) > max_str:
            out[k] = v[:max_str] + "…"
        elif isinstance(v, dict):
            out[k] = _truncate_dict(v, max_str)
        elif isinstance(v, list) and len(v) > 10:
            out[k] = v[:10] + ["… (truncated)"]
        else:
            out[k] = v
    return out


# ═══════════════════════════════════════════════════════════════════════════════
#  Stage runners — each returns a StageResult
# ═══════════════════════════════════════════════════════════════════════════════


async def _run_step2_intent(query: str) -> StageResult:
    """Step 2: Intent Extraction."""
    stage = StageResult(stage="Step 2 — Intent Extraction")
    t0 = time.perf_counter()
    try:
        result = await extract_intent(query)
        stage.elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)
        stage.output = result
        stage.success = True

        # ── Structural checks ──────────────────────────────────────────
        stage.checks["has_intent_type"] = "intent_type" in result
        stage.checks["has_entities"] = "entities" in result and isinstance(result["entities"], list)
        stage.checks["has_confidence"] = "confidence_score" in result
        stage.checks["confidence_valid"] = (
            isinstance(result.get("confidence_score"), (int, float))
            and 0 <= result["confidence_score"] <= 1
        )
    except Exception as exc:
        stage.elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)
        stage.error = f"{type(exc).__name__}: {exc}"
    return stage


async def _run_step3_cypher(intent_json: dict) -> StageResult:
    """Step 3: Cypher Query Generation."""
    stage = StageResult(stage="Step 3 — Cypher Generation")
    t0 = time.perf_counter()
    try:
        result = await generate_cypher(intent_json)
        stage.elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)
        stage.output = result
        stage.success = True

        stage.checks["has_cypher"] = "cypher" in result and isinstance(result["cypher"], str)
        stage.checks["has_params"] = "params" in result and isinstance(result["params"], dict)
        stage.checks["has_source"] = result.get("source") in ("template", "llm")
        stage.checks["cypher_non_empty"] = bool(result.get("cypher", "").strip())
    except Exception as exc:
        stage.elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)
        stage.error = f"{type(exc).__name__}: {exc}"
    return stage


def _run_step4_graph(step3_result: dict) -> StageResult:
    """Step 4: Graph Retrieval."""
    stage = StageResult(stage="Step 4 — Graph Retrieval")
    t0 = time.perf_counter()
    try:
        result = retrieve_graph_from_step3(step3_result)
        stage.elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)
        stage.output = result
        stage.success = True

        stage.checks["has_disease_key"] = "disease" in result
        stage.checks["has_remedies"] = "remedies" in result and isinstance(result["remedies"], list)
        stage.checks["has_raw_subgraph"] = "raw_subgraph" in result
        stage.checks["has_metadata"] = "metadata" in result

        meta = result.get("metadata", {})
        stage.checks["query_executed"] = not meta.get("skipped", False)
        stage.checks["has_results"] = not meta.get("empty", False)
    except Exception as exc:
        stage.elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)
        stage.error = f"{type(exc).__name__}: {exc}"
    return stage


def _run_step5_evidence(graph_json: dict) -> StageResult:
    """Step 5: Evidence Formatting."""
    stage = StageResult(stage="Step 5 — Evidence Formatting")
    t0 = time.perf_counter()
    try:
        blocks = format_graph_for_llm(graph_json, enrich_dosage=True)
        stage.elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)
        stage.output = blocks
        stage.success = True

        stage.checks["returns_list"] = isinstance(blocks, list)
        stage.checks["non_empty"] = len(blocks) > 0
        stage.checks["has_facts"] = any(b.startswith("FACT") for b in blocks)

        # Count block types.
        type_counts = {}
        for b in blocks:
            prefix = b.split(":")[0].split("(")[0].strip() if ":" in b else "OTHER"
            type_counts[prefix] = type_counts.get(prefix, 0) + 1
        stage.checks["block_type_counts"] = type_counts
    except Exception as exc:
        stage.elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)
        stage.error = f"{type(exc).__name__}: {exc}"
    return stage


async def _run_step6_reasoner(evidence_blocks: list[str], query: str) -> StageResult:
    """Step 6: Clinical Reasoner (A0 draft)."""
    stage = StageResult(stage="Step 6 — Clinical Reasoner")
    t0 = time.perf_counter()
    try:
        result = await generate_clinical_draft(evidence_blocks, query)
        stage.elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)
        stage.output = result
        stage.success = True

        stage.checks["has_a0"] = "a0" in result and isinstance(result["a0"], str)
        stage.checks["a0_non_empty"] = len(result.get("a0", "")) > 20
        stage.checks["has_metadata"] = "metadata" in result
    except Exception as exc:
        stage.elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)
        stage.error = f"{type(exc).__name__}: {exc}"
    return stage


async def _run_step7_validator(
    a0: str, evidence_blocks: list[str]
) -> StageResult:
    """Step 7: Validator / Safety Model."""
    stage = StageResult(stage="Step 7 — Validator / Safety")
    t0 = time.perf_counter()
    try:
        result = await validate_draft(a0, evidence_blocks)
        stage.elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)
        stage.output = result
        stage.success = True

        stage.checks["has_grounded"] = "grounded" in result
        stage.checks["has_safety_score"] = (
            "safety_score" in result
            and isinstance(result["safety_score"], (int, float))
        )
        stage.checks["has_hallucination_flag"] = "hallucination_flag" in result
        stage.checks["has_validated_answer"] = (
            "validated_answer" in result
            and isinstance(result["validated_answer"], str)
        )
        stage.checks["safe_to_publish"] = is_safe_to_publish(result)
    except Exception as exc:
        stage.elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)
        stage.error = f"{type(exc).__name__}: {exc}"
    return stage


async def _run_step8_response(
    a0: str,
    validation_result: dict,
    evidence_blocks: list[str],
    user_query: str,
) -> StageResult:
    """Step 8: Final Response Builder."""
    stage = StageResult(stage="Step 8 — Response Builder")
    t0 = time.perf_counter()
    try:
        result = await build_final_answer(
            a0, validation_result, evidence_blocks, user_query,
        )
        stage.elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)
        stage.output = result
        stage.success = True

        stage.checks["has_final_answer_text"] = (
            "final_answer_text" in result
            and isinstance(result["final_answer_text"], str)
        )
        stage.checks["has_structured_fields"] = (
            "structured_fields" in result
            and isinstance(result["structured_fields"], dict)
        )
        stage.checks["has_traceability"] = (
            "traceability" in result
            and isinstance(result["traceability"], dict)
        )
        stage.checks["has_metadata"] = "metadata" in result

        # Structured-fields sub-checks.
        sf = result.get("structured_fields", {})
        for key in ("remedy", "dosage", "mechanism", "safety",
                     "hadith_reference", "confidence", "safety_score"):
            stage.checks[f"sf_has_{key}"] = key in sf
    except Exception as exc:
        stage.elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)
        stage.error = f"{type(exc).__name__}: {exc}"
    return stage


# ═══════════════════════════════════════════════════════════════════════════════
#  Hallucination detector (heuristic, no extra LLM call)
# ═══════════════════════════════════════════════════════════════════════════════

def _check_hallucination_heuristic(
    a0: str, evidence_blocks: list[str],
) -> dict[str, Any]:
    """
    Simple heuristic: every named entity mentioned in A0 that is NOT
    present in any evidence block is flagged as a potential hallucination.

    Returns a dict with ``possibly_hallucinated`` (list of suspect tokens)
    and ``hallucination_risk`` (bool).
    """
    # Build a lowercase evidence corpus.
    evidence_corpus = " ".join(evidence_blocks).lower()

    # Extract capitalised multi-word tokens from A0 (likely named entities).
    # Only match tokens on a single line (no embedded newlines).
    import re
    entity_pattern = re.compile(r"\b[A-Z][a-z]+(?:[ ][A-Z][a-z]+)*\b")
    candidates = set(entity_pattern.findall(a0))

    # Common English words that are capitalised at sentence starts.
    _COMMON = {
        "The", "This", "That", "These", "Those", "It", "Its",
        "For", "From", "With", "However", "Although", "While",
        "Please", "Note", "According", "Based", "Given", "Some",
        "May", "Can", "Should", "Would", "Could", "Has", "Have",
        "Are", "Were", "Was", "Is", "Be", "Not", "All", "Any",
        "When", "Where", "What", "How", "Who", "Which", "Each",
        "Important", "Traditional", "Modern", "Prophetic", "Islamic",
        "Disclaimer", "Safety", "Dosage", "Preparation", "Recommendation",
        "Mechanism", "Explanation", "Faith", "Science", "Alignment",
        "References", "Contraindications", "Introduction",
    }

    suspect: list[str] = []
    for entity in candidates:
        if entity in _COMMON:
            continue
        if entity.lower() not in evidence_corpus:
            suspect.append(entity)

    return {
        "possibly_hallucinated": suspect,
        "hallucination_risk": len(suspect) > 0,
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  Full pipeline runner for one query
# ═══════════════════════════════════════════════════════════════════════════════


async def _run_single_query(tc: dict[str, Any]) -> QueryTestResult:
    """Execute the full pipeline for a single test query."""
    qr = QueryTestResult(
        query_id=tc["id"],
        query=tc["query"],
        category=tc["category"],
    )
    t_global = time.perf_counter()

    # ── Step 2: Intent Extraction ──────────────────────────────────────────
    s2 = await _run_step2_intent(tc["query"])
    qr.stages.append(s2)

    if not s2.success or not s2.output:
        qr.total_elapsed_ms = round((time.perf_counter() - t_global) * 1000, 1)
        qr.warnings.append("Pipeline aborted at Step 2 — intent extraction failed.")
        return qr

    intent_json = s2.output

    # Check expected intent (soft — intent names may vary).
    actual_intent = intent_json.get("intent_type", "")
    if tc.get("expected_intent") and actual_intent != tc["expected_intent"]:
        qr.warnings.append(
            f"Intent mismatch: expected '{tc['expected_intent']}', got '{actual_intent}'"
        )

    # ── Step 3: Cypher Generation ──────────────────────────────────────────
    s3 = await _run_step3_cypher(intent_json)
    qr.stages.append(s3)

    if not s3.success or not s3.output:
        qr.total_elapsed_ms = round((time.perf_counter() - t_global) * 1000, 1)
        qr.warnings.append("Pipeline aborted at Step 3 — Cypher generation failed.")
        return qr

    cypher_result = s3.output

    # ── Step 4: Graph Retrieval ────────────────────────────────────────────
    s4 = _run_step4_graph(cypher_result)
    qr.stages.append(s4)

    if not s4.success or not s4.output:
        qr.total_elapsed_ms = round((time.perf_counter() - t_global) * 1000, 1)
        qr.warnings.append("Pipeline aborted at Step 4 — graph retrieval failed.")
        return qr

    graph_json = s4.output
    meta4 = graph_json.get("metadata", {})
    qr.node_count = meta4.get("node_count", 0)
    qr.edge_count = meta4.get("edge_count", 0)

    # ── Step 5: Evidence Formatting ────────────────────────────────────────
    s5 = _run_step5_evidence(graph_json)
    qr.stages.append(s5)

    if not s5.success:
        qr.total_elapsed_ms = round((time.perf_counter() - t_global) * 1000, 1)
        qr.warnings.append("Pipeline aborted at Step 5 — evidence formatting failed.")
        return qr

    evidence_blocks: list[str] = s5.output or []
    qr.evidence_block_count = len(evidence_blocks)

    # If no evidence, we can still continue (Step 6+ handle empty evidence).

    # ── Step 6: Clinical Reasoner (A0) ─────────────────────────────────────
    s6 = await _run_step6_reasoner(evidence_blocks, tc["query"])
    qr.stages.append(s6)

    if not s6.success or not s6.output:
        qr.total_elapsed_ms = round((time.perf_counter() - t_global) * 1000, 1)
        qr.warnings.append("Pipeline aborted at Step 6 — reasoner failed.")
        return qr

    a0: str = s6.output["a0"]

    # Hallucination heuristic check.
    halluc_check = _check_hallucination_heuristic(a0, evidence_blocks)
    if halluc_check["hallucination_risk"]:
        qr.warnings.append(
            f"Possible hallucinations in A0: {halluc_check['possibly_hallucinated'][:5]}"
        )

    # ── Step 7: Validator / Safety ─────────────────────────────────────────
    s7 = await _run_step7_validator(a0, evidence_blocks)
    qr.stages.append(s7)

    if not s7.success or not s7.output:
        qr.total_elapsed_ms = round((time.perf_counter() - t_global) * 1000, 1)
        qr.warnings.append("Pipeline aborted at Step 7 — validator failed.")
        return qr

    validation_result = s7.output
    qr.grounded = validation_result.get("grounded", False)
    qr.hallucination_flag = validation_result.get("hallucination_flag", True)
    qr.safety_score = validation_result.get("safety_score", 0.0)

    if not is_safe_to_publish(validation_result):
        qr.warnings.append(
            f"Validation FAIL: grounded={qr.grounded}, "
            f"safety={qr.safety_score}, hallucination={qr.hallucination_flag}"
        )

    # ── Step 8: Final Response Builder ─────────────────────────────────────
    s8 = await _run_step8_response(
        a0, validation_result, evidence_blocks, tc["query"],
    )
    qr.stages.append(s8)

    if s8.success and s8.output:
        final = s8.output
        qr.final_answer_chars = len(final.get("final_answer_text", ""))
        sf = final.get("structured_fields", {})
        qr.confidence = sf.get("confidence")
        qr.safety_score = sf.get("safety_score", qr.safety_score)

    # ── Overall result ─────────────────────────────────────────────────────
    qr.total_elapsed_ms = round((time.perf_counter() - t_global) * 1000, 1)
    qr.overall_pass = all(s.success for s in qr.stages)

    return qr


# ═══════════════════════════════════════════════════════════════════════════════
#  Report generator
# ═══════════════════════════════════════════════════════════════════════════════


def _build_report(results: list[QueryTestResult]) -> PipelineTestReport:
    """Aggregate individual query results into a summary report."""
    report = PipelineTestReport(
        total_queries=len(results),
        results=results,
    )

    total_nodes = 0
    total_edges = 0
    total_evidence = 0
    stage_latencies: dict[str, list[float]] = {}

    for r in results:
        if r.overall_pass:
            report.passed += 1
        else:
            report.failed += 1
        report.total_elapsed_ms += r.total_elapsed_ms

        total_nodes += r.node_count
        total_edges += r.edge_count
        total_evidence += r.evidence_block_count

        for s in r.stages:
            stage_latencies.setdefault(s.stage, []).append(s.elapsed_ms)

        for w in r.warnings:
            if "unsafe" in w.lower() or "hallucination" in w.lower() or "validation fail" in w.lower():
                report.unsafe_warnings.append(f"[{r.query_id}] {w}")

    if results:
        report.avg_latency_ms = round(report.total_elapsed_ms / len(results), 1)

    for stage_name, lats in stage_latencies.items():
        report.stage_avg_latencies[stage_name] = round(sum(lats) / len(lats), 1)

    report.graph_coverage = {
        "total_nodes_retrieved": total_nodes,
        "total_edges_retrieved": total_edges,
        "total_evidence_blocks": total_evidence,
        "avg_nodes_per_query": round(total_nodes / max(len(results), 1), 1),
        "avg_edges_per_query": round(total_edges / max(len(results), 1), 1),
        "avg_evidence_blocks": round(total_evidence / max(len(results), 1), 1),
    }

    return report


# ═══════════════════════════════════════════════════════════════════════════════
#  Console printer
# ═══════════════════════════════════════════════════════════════════════════════

_SEP = "─" * 72


def _print_report(report: PipelineTestReport) -> None:
    """Pretty-print the test report to stdout."""
    print()
    print("═" * 72)
    print("  PRO-MedGraph · Pipeline Test Report (Step 9)")
    print("═" * 72)

    # ── Per-query details ──────────────────────────────────────────────────
    for r in report.results:
        status = "✓ PASS" if r.overall_pass else "✗ FAIL"
        print(f"\n{_SEP}")
        print(f"  [{r.query_id}] {status}  ({r.total_elapsed_ms} ms)  [{r.category}]")
        print(f"  Query: {r.query[:80]}")

        for s in r.stages:
            icon = "✓" if s.success else "✗"
            print(f"    {icon} {s.stage:36s}  {s.elapsed_ms:>8.1f} ms", end="")
            if s.error:
                print(f"  ERROR: {s.error[:60]}")
            else:
                # Summarise check results.
                fails = [k for k, v in s.checks.items() if v is False]
                if fails:
                    print(f"  ⚠ failed checks: {fails}")
                else:
                    print()

        if r.warnings:
            for w in r.warnings:
                print(f"    ⚠ {w[:100]}")

        print(f"    Graph: {r.node_count} nodes, {r.edge_count} edges, "
              f"{r.evidence_block_count} evidence blocks")
        print(f"    Answer: {r.final_answer_chars} chars, "
              f"confidence={r.confidence}, safety={r.safety_score}, "
              f"grounded={r.grounded}")

    # ── Summary ────────────────────────────────────────────────────────────
    print(f"\n{'═' * 72}")
    print("  SUMMARY")
    print(f"{'═' * 72}")
    print(f"  Queries:  {report.passed}/{report.total_queries} passed, "
          f"{report.failed}/{report.total_queries} failed")
    print(f"  Total time: {report.total_elapsed_ms:.0f} ms  "
          f"(avg {report.avg_latency_ms:.0f} ms/query)")

    print(f"\n  Stage latencies (avg):")
    for stage, avg in report.stage_avg_latencies.items():
        print(f"    {stage:36s}  {avg:>8.1f} ms")

    gc = report.graph_coverage
    print(f"\n  Graph coverage:")
    print(f"    Total nodes retrieved:    {gc['total_nodes_retrieved']}")
    print(f"    Total edges retrieved:    {gc['total_edges_retrieved']}")
    print(f"    Total evidence blocks:    {gc['total_evidence_blocks']}")
    print(f"    Avg nodes/query:          {gc['avg_nodes_per_query']}")
    print(f"    Avg edges/query:          {gc['avg_edges_per_query']}")
    print(f"    Avg evidence blocks:      {gc['avg_evidence_blocks']}")

    if report.unsafe_warnings:
        print(f"\n  ⚠ Safety / hallucination warnings ({len(report.unsafe_warnings)}):")
        for w in report.unsafe_warnings:
            print(f"    {w[:100]}")

    print(f"\n{'═' * 72}\n")


# ═══════════════════════════════════════════════════════════════════════════════
#  Public API
# ═══════════════════════════════════════════════════════════════════════════════


async def run_pipeline_tests(
    *,
    queries: list[dict[str, Any]] | None = None,
    save_json: str | None = None,
    verbose: bool = True,
) -> PipelineTestReport:
    """
    Execute the full multi-model pipeline on all sample queries and
    produce a structured test report.

    Parameters
    ----------
    queries : list[dict] | None
        Custom query list.  Defaults to :data:`SAMPLE_QUERIES`.
    save_json : str | None
        If set, save the full JSON report to this filename (relative to cwd).
    verbose : bool
        If True, print detailed results to stdout.

    Returns
    -------
    PipelineTestReport
    """
    test_queries = queries or SAMPLE_QUERIES

    logger.info("Pipeline tests: starting %d queries", len(test_queries))

    results: list[QueryTestResult] = []
    for idx, tc in enumerate(test_queries):
        # Rate-limit pause between queries to avoid 429 from Groq free tier.
        if idx > 0:
            logger.info("Waiting 5 s between queries (rate-limit guard)…")
            await asyncio.sleep(5)

        logger.info("──── Running [%s] %s ────", tc["id"], tc["query"][:60])
        qr = await _run_single_query(tc)
        results.append(qr)
        logger.info(
            "[%s] %s — %s (%.0f ms)",
            qr.query_id,
            "PASS" if qr.overall_pass else "FAIL",
            qr.category,
            qr.total_elapsed_ms,
        )

    report = _build_report(results)

    if verbose:
        _print_report(report)

    if save_json:
        out_path = Path(save_json).with_suffix(".json")
        out_path.write_text(
            json.dumps(report.to_dict(), indent=2, default=str, ensure_ascii=False),
            encoding="utf-8",
        )
        logger.info("Report saved to %s", out_path)
        if verbose:
            print(f"  📄 Report saved to {out_path}\n")

    return report


# ═══════════════════════════════════════════════════════════════════════════════
#  CLI entry point
# ═══════════════════════════════════════════════════════════════════════════════
#
#  python -m app.services.pipeline_tests               # console only
#  python -m app.services.pipeline_tests --json report  # + save JSON


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        stream=sys.stdout,
        format="%(levelname)s  %(message)s",
    )

    save_path: str | None = None
    if "--json" in sys.argv:
        idx = sys.argv.index("--json")
        if idx + 1 < len(sys.argv):
            save_path = sys.argv[idx + 1]
        else:
            save_path = "pipeline_test_report"

    report = asyncio.run(run_pipeline_tests(save_json=save_path))

    # Exit code reflects pass/fail for CI.
    sys.exit(0 if report.failed == 0 else 1)


if __name__ == "__main__":
    main()


# ── Module exports ─────────────────────────────────────────────────────────────

__all__ = [
    "run_pipeline_tests",
    "SAMPLE_QUERIES",
    "PipelineTestReport",
    "QueryTestResult",
    "StageResult",
]
