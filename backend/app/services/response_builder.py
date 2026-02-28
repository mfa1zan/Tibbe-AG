"""
PRO-MedGraph  —  Step 8: Final Response Builder
=================================================

Combines the validated draft (A0 + Step 7 validation report) with the
original evidence blocks and the user query to produce a **polished,
user-facing answer** via the chat LLM (``MODEL_CHAT``).

Pipeline position
-----------------
Step 7 (Validator) → **Step 8 (Final Response Builder)** → UI / API

The output is a JSON envelope containing:

    * ``final_answer_text`` — fully formatted markdown answer.
    * ``structured_fields`` — machine-readable dict with remedy, dosage,
      mechanism, safety, references, and confidence breakdowns.
    * ``traceability`` — list of evidence block indices that were used,
      so the frontend can highlight provenance.

Design choices
--------------
1. If the validation result says the draft is **safe to publish**
   (``grounded=true``, ``hallucination_flag=false``, ``safety_score ≥ 0.8``),
   we use the ``validated_answer`` from Step 7 as the primary source.
2. If the draft **failed** validation, we still generate an answer but
   prepend a safety disclaimer and lower the reported confidence.
3. The chat LLM is used purely for *formatting* — it is not allowed to
   add new facts.  The system prompt enforces this strictly.

Usage
-----
.. code-block:: python

    from app.services.response_builder import build_final_answer

    result = await build_final_answer(a0, validation_result, evidence_blocks, query)
    print(result["final_answer_text"])

Run standalone tests:  ``python -m app.services.response_builder``
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Any

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════════
#  LLM caller (reuses multi_model_service)
# ═══════════════════════════════════════════════════════════════════════════════

from app.services.multi_model_service import _call_groq, config  # noqa: E402

_CHAT_TIMEOUT = 60.0

# ═══════════════════════════════════════════════════════════════════════════════
#  System prompt
# ═══════════════════════════════════════════════════════════════════════════════

_SYSTEM_PROMPT = """\
You are **PRO-MedGraph Response Formatter**, the final stage of a \
biomedical GraphRAG pipeline that integrates Prophetic (Islamic) medicine \
with modern pharmacology.

You will receive:
  1. A validated draft answer (A_validated).
  2. The original user question.
  3. Structured evidence blocks from the knowledge graph.
  4. A validation summary (grounding, safety score, flags).

═══ YOUR TASK ═══
Rewrite A_validated into a polished, user-friendly answer AND produce a \
structured JSON object in a SINGLE response.

═══ STRICT RULES ═══
1. Use ONLY information present in A_validated and the evidence blocks.
2. Do NOT hallucinate, invent, or cite external studies / URLs / data.
3. If evidence is absent for a section, write "Information not available \
   in the current knowledge graph."
4. ALWAYS include a medical disclaimer at the end.

═══ REQUIRED JSON OUTPUT ═══
Return ONLY a JSON object (no markdown fences, no extra text outside the \
JSON) with these keys:

{
    "final_answer_text": "<markdown-formatted user-facing answer>",
    "structured_fields": {
        "remedy": "<name of prophetic remedy / ingredient>",
        "dosage": "<traditional dosage + preparation, or null>",
        "modern_drug_equivalent": "<mapped modern drug, or null>",
        "mechanism": "<mechanistic explanation linking ingredient → compound → drug effect>",
        "safety": "<contraindications / side-effects / warnings, or null>",
        "hadith_reference": "<verbatim hadith quote, or null>",
        "confidence": "<high | moderate | low>",
        "safety_score": <float 0.0–1.0 from validation>
    },
    "traceability": {
        "evidence_block_count": <int>,
        "used_block_indices": [<int>, ...],
        "grounded": <true|false>,
        "hallucination_flag": <true|false>,
        "faith_alignment_ok": <true|false>
    }
}

═══ FORMATTING GUIDELINES FOR final_answer_text ═══
Use this markdown template (skip any section that has no data):

## 🌿 Prophetic Remedy
<ingredient name and brief prophetic / historical context>

## 🔬 Mechanistic Explanation
<ingredient → compound → mechanism → modern drug equivalent>
<mapping confidence: high / moderate / low>

## 💊 Dosage & Preparation
**Traditional:** <traditional dosage>
**Modern equivalent:** <drug + standard dosage>

## ⚠️ Safety & Contraindications
<warnings, drug interactions, side effects>

## 📖 References
- **Hadith:** <verbatim quote>
- **Knowledge graph confidence:** <score>

## ❗ Disclaimer
This information is for educational purposes only and does not replace \
professional medical advice. Please consult a qualified healthcare \
professional before making any changes to your health regimen.
"""

# ═══════════════════════════════════════════════════════════════════════════════
#  JSON extraction (reused pattern from draft_validator)
# ═══════════════════════════════════════════════════════════════════════════════

_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*\n?(.*?)\n?\s*```", re.DOTALL)


def _extract_json(text: str) -> dict[str, Any] | None:
    """Try to extract a JSON object from the LLM response."""
    # Strategy 1: markdown code fence
    m = _JSON_FENCE_RE.search(text)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass

    # Strategy 2: brace-bounded substring
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            pass

    # Strategy 3: raw text
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


# ═══════════════════════════════════════════════════════════════════════════════
#  Evidence block classification helpers
# ═══════════════════════════════════════════════════════════════════════════════

_BLOCK_PREFIXES = ("FACT", "INFERENCE", "DOSAGE", "SAFETY", "HADITH", "NOTE")


def _classify_blocks(blocks: list[str]) -> dict[str, list[int]]:
    """
    Return a dict mapping block type → list of 0-based indices.

    Example: ``{"FACT": [0, 1, 2], "INFERENCE": [3], "DOSAGE": [4]}``
    """
    classification: dict[str, list[int]] = {}
    for i, block in enumerate(blocks):
        for prefix in _BLOCK_PREFIXES:
            if block.startswith(prefix):
                classification.setdefault(prefix, []).append(i)
                break
        else:
            classification.setdefault("OTHER", []).append(i)
    return classification


def _all_used_indices(blocks: list[str]) -> list[int]:
    """Return indices of all non-NOTE blocks (these are the evidence)."""
    return [i for i, b in enumerate(blocks) if not b.startswith("NOTE")]


# ═══════════════════════════════════════════════════════════════════════════════
#  Fallback builder (no LLM)
# ═══════════════════════════════════════════════════════════════════════════════

_DISCLAIMER = (
    "❗ **Disclaimer:** This information is for educational purposes only "
    "and does not replace professional medical advice.  Please consult a "
    "qualified healthcare professional before making any changes to your "
    "health regimen."
)


def _build_fallback(
    validated_answer: str,
    validation_result: dict[str, Any],
    evidence_blocks: list[str],
) -> dict[str, Any]:
    """
    Deterministic fallback when the chat LLM fails or returns un-parseable
    JSON.  Wraps the validated_answer in markdown and populates structured
    fields from the evidence blocks directly.
    """
    safety_score = validation_result.get("safety_score", 0.0)
    grounded = validation_result.get("grounded", False)
    hallucination = validation_result.get("hallucination_flag", True)
    faith_ok = validation_result.get("faith_alignment_ok", False)

    # Derive confidence label.
    if safety_score >= 0.85 and grounded:
        confidence = "high"
    elif safety_score >= 0.6:
        confidence = "moderate"
    else:
        confidence = "low"

    # Extract specific fields from evidence blocks.
    remedy = dosage = mechanism = safety = hadith = None
    for block in evidence_blocks:
        if block.startswith("FACT:") and "remedy for" in block:
            remedy = remedy or block.split("FACT:")[-1].strip()
        if block.startswith("DOSAGE"):
            dosage = (dosage or "") + block.split(":", 1)[-1].strip() + "; "
        if block.startswith("INFERENCE"):
            mechanism = mechanism or block.split("INFERENCE:")[-1].strip()
        if block.startswith("SAFETY"):
            safety = (safety or "") + block.split(":", 1)[-1].strip() + "; "
        if block.startswith("HADITH"):
            hadith = hadith or block.split("HADITH:")[-1].strip()

    if dosage:
        dosage = dosage.rstrip("; ")
    if safety:
        safety = safety.rstrip("; ")

    # Build markdown answer.
    md_parts: list[str] = []
    if not grounded or hallucination:
        md_parts.append(
            "> ⚠️ **Note:** This answer could not be fully validated against "
            "the knowledge graph.  Treat it as preliminary.\n"
        )
    md_parts.append(validated_answer)
    md_parts.append(f"\n\n{_DISCLAIMER}")

    return {
        "final_answer_text": "\n".join(md_parts),
        "structured_fields": {
            "remedy": remedy,
            "dosage": dosage,
            "modern_drug_equivalent": None,
            "mechanism": mechanism,
            "safety": safety,
            "hadith_reference": hadith,
            "confidence": confidence,
            "safety_score": safety_score,
        },
        "traceability": {
            "evidence_block_count": len(evidence_blocks),
            "used_block_indices": _all_used_indices(evidence_blocks),
            "grounded": grounded,
            "hallucination_flag": hallucination,
            "faith_alignment_ok": faith_ok,
        },
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  Normaliser for LLM JSON output
# ═══════════════════════════════════════════════════════════════════════════════


def _normalise(raw: dict[str, Any], fallback: dict[str, Any]) -> dict[str, Any]:
    """
    Ensure every expected top-level key exists and has the right shape.
    Falls back to the deterministic builder for any missing pieces.
    """
    if not isinstance(raw, dict):
        return fallback

    final_text = raw.get("final_answer_text")
    if not isinstance(final_text, str) or len(final_text.strip()) < 20:
        final_text = fallback["final_answer_text"]

    # structured_fields
    sf_raw = raw.get("structured_fields") if isinstance(raw.get("structured_fields"), dict) else {}
    sf_fb = fallback.get("structured_fields") or {}
    structured_fields: dict[str, Any] = {}
    for key in ("remedy", "dosage", "modern_drug_equivalent", "mechanism",
                "safety", "hadith_reference", "confidence"):
        val = sf_raw.get(key)
        structured_fields[key] = val if isinstance(val, str) and val.strip() else sf_fb.get(key)

    # safety_score — keep as float
    try:
        score_src = sf_raw.get("safety_score") if sf_raw.get("safety_score") is not None else sf_fb.get("safety_score", 0.0)
        structured_fields["safety_score"] = round(
            max(0.0, min(1.0, float(score_src))), 2
        )
    except (TypeError, ValueError):
        structured_fields["safety_score"] = sf_fb.get("safety_score", 0.0)

    # traceability
    tr_raw = raw.get("traceability") if isinstance(raw.get("traceability"), dict) else {}
    tr_fb = fallback.get("traceability") or {}
    traceability: dict[str, Any] = {
        "evidence_block_count": tr_raw.get("evidence_block_count", tr_fb.get("evidence_block_count", 0)),
        "used_block_indices": tr_raw.get("used_block_indices", tr_fb.get("used_block_indices", [])),
        "grounded": tr_raw.get("grounded", tr_fb.get("grounded", False)),
        "hallucination_flag": tr_raw.get("hallucination_flag", tr_fb.get("hallucination_flag", True)),
        "faith_alignment_ok": tr_raw.get("faith_alignment_ok", tr_fb.get("faith_alignment_ok", False)),
    }

    return {
        "final_answer_text": final_text,
        "structured_fields": structured_fields,
        "traceability": traceability,
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  Public API
# ═══════════════════════════════════════════════════════════════════════════════


async def build_final_answer(
    a0: str,
    validation_result: dict[str, Any],
    evidence_blocks: list[str],
    user_query: str,
    *,
    temperature: float = 0.20,
    max_retries: int = 1,
) -> dict[str, Any]:
    """
    Build the final, user-facing answer from all upstream pipeline outputs.

    Parameters
    ----------
    a0 : str
        The draft answer from Step 6 (Clinical Reasoner).
    validation_result : dict
        The JSON report from Step 7 (Validator / Safety Model).
        Expected keys: ``grounded``, ``safety_score``, ``hallucination_flag``,
        ``faith_alignment_ok``, ``validated_answer``, ``recommendations``.
    evidence_blocks : list[str]
        The FACT / INFERENCE / DOSAGE / SAFETY / HADITH strings from Step 5.
    user_query : str
        The original user question.
    temperature : float
        Sampling temperature for the chat LLM.
    max_retries : int
        Extra attempts if the LLM returns un-parseable JSON.

    Returns
    -------
    dict
        ``{"final_answer_text": str, "structured_fields": dict,
        "traceability": dict, "metadata": dict}``
    """
    t_start = time.perf_counter()

    # ── Choose the best available answer text ──────────────────────────────
    validated_answer = (
        validation_result.get("validated_answer") or a0
    ).strip()

    grounded = validation_result.get("grounded", False)
    safety_score = validation_result.get("safety_score", 0.0)
    halluc = validation_result.get("hallucination_flag", True)
    faith_ok = validation_result.get("faith_alignment_ok", False)

    # ── Deterministic fallback (always computed) ───────────────────────────
    fallback = _build_fallback(validated_answer, validation_result, evidence_blocks)

    # ── Build compact validation summary for the prompt ────────────────────
    validation_summary = (
        f"grounded={grounded}, safety_score={safety_score}, "
        f"hallucination_flag={halluc}, faith_alignment_ok={faith_ok}"
    )
    recs = validation_result.get("recommendations") or []
    if recs:
        validation_summary += f"\nRecommendations: {'; '.join(recs[:5])}"

    # ── Build evidence text ────────────────────────────────────────────────
    evidence_text = "\n".join(
        f"[{i}] {b}" for i, b in enumerate(evidence_blocks)
    ) if evidence_blocks else "(no evidence blocks)"

    block_classification = _classify_blocks(evidence_blocks)

    # ── Build user prompt ──────────────────────────────────────────────────
    user_prompt = (
        f"══ USER QUERY ══\n"
        f"{user_query}\n\n"
        f"══ VALIDATED DRAFT ANSWER (A_validated) ══\n"
        f"{validated_answer}\n\n"
        f"══ VALIDATION SUMMARY ══\n"
        f"{validation_summary}\n\n"
        f"══ EVIDENCE BLOCKS ({len(evidence_blocks)}) ══\n"
        f"{evidence_text}\n\n"
        f"══ EVIDENCE BLOCK CLASSIFICATION ══\n"
        f"{json.dumps(block_classification, indent=2)}"
    )

    model_id = config.chat

    logger.info(
        "Response builder: sending %d evidence blocks + validated answer "
        "(%d chars) to %s (temp=%.2f). Validation: %s",
        len(evidence_blocks), len(validated_answer), model_id,
        temperature, validation_summary[:120],
    )

    # ── Call LLM with retry ────────────────────────────────────────────────
    parsed: dict[str, Any] | None = None
    last_error: Exception | None = None

    for attempt in range(1 + max_retries):
        try:
            raw_response = await _call_groq(
                model=model_id,
                system_prompt=_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                temperature=temperature,
                timeout=_CHAT_TIMEOUT,
            )
        except Exception as exc:
            last_error = exc
            logger.warning(
                "Response builder LLM call failed (attempt %d/%d): %s",
                attempt + 1, 1 + max_retries, exc,
            )
            continue

        parsed = _extract_json(raw_response)
        if parsed is not None:
            break

        logger.warning(
            "Response builder: LLM returned non-JSON (attempt %d/%d). "
            "Preview: %.200s",
            attempt + 1, 1 + max_retries, raw_response,
        )

    elapsed_ms = round((time.perf_counter() - t_start) * 1000, 1)

    # ── Build result ───────────────────────────────────────────────────────
    if parsed is not None:
        try:
            result = _normalise(parsed, fallback)
        except Exception:
            logger.warning(
                "Response builder: normalisation failed; using fallback",
                exc_info=True,
            )
            result = fallback
    else:
        if last_error:
            logger.error(
                "Response builder: all LLM attempts failed (%.1f ms): %s",
                elapsed_ms, last_error,
            )
        else:
            logger.warning(
                "Response builder: could not parse LLM JSON; using "
                "deterministic fallback (%.1f ms)", elapsed_ms,
            )
        result = fallback

    # ── Attach metadata ────────────────────────────────────────────────────
    result["metadata"] = {
        "model": model_id,
        "elapsed_ms": elapsed_ms,
        "final_answer_chars": len(result["final_answer_text"]),
        "evidence_block_count": len(evidence_blocks),
        "validation_grounded": grounded,
        "validation_safety_score": safety_score,
        "used_fallback": parsed is None,
    }

    # ── Summary log ────────────────────────────────────────────────────────
    sf = result["structured_fields"]
    logger.info(
        "Response builder: final answer %d chars, confidence=%s, "
        "safety=%.2f, remedy=%s, fallback=%s (%.1f ms, model=%s)",
        len(result["final_answer_text"]),
        sf.get("confidence") or "unknown",
        sf.get("safety_score") or 0.0,
        (sf.get("remedy") or "?")[:40],
        result["metadata"]["used_fallback"],
        elapsed_ms,
        model_id,
    )

    return result


# ═══════════════════════════════════════════════════════════════════════════════
#  Convenience: full pipeline Step 4 → 8
# ═══════════════════════════════════════════════════════════════════════════════


async def build_answer_from_graph(
    graph_json: dict[str, Any],
    user_query: str,
    *,
    enrich_dosage: bool = True,
) -> dict[str, Any]:
    """
    End-to-end convenience: Step 4 output → Step 5 → Step 6 → Step 7 → Step 8.

    Useful for testing the full pipeline in one call.
    """
    from app.services.evidence_formatter import format_graph_for_llm
    from app.services.clinical_reasoner import generate_clinical_draft
    from app.services.draft_validator import validate_draft

    # Step 5
    blocks = format_graph_for_llm(graph_json, enrich_dosage=enrich_dosage)

    # Step 6
    draft_result = await generate_clinical_draft(blocks, user_query)
    a0 = draft_result["a0"]

    # Step 7
    validation = await validate_draft(a0, blocks)

    # Step 8
    return await build_final_answer(a0, validation, blocks, user_query)


# ═══════════════════════════════════════════════════════════════════════════════
#  Standalone test harness
# ═══════════════════════════════════════════════════════════════════════════════
#
#  Run:  python -m app.services.response_builder
#  (from backend/ dir with .env present and Groq API reachable)

import asyncio as _asyncio

_MOCK_EVIDENCE: list[str] = [
    "FACT: The query concerns the disease/condition 'hypertension'",
    "FACT: Black seed is a traditional remedy for hypertension",
    "FACT: Black seed contains Thymoquinone",
    "FACT: Black seed contains Nigellone",
    "FACT: Thymoquinone maps to pharmaceutical compound in Captopril",
    "FACT: Thymoquinone is biochemically identical to the active compound "
    "in Captopril (confidence: high)",
    "INFERENCE: Thymoquinone (from Black seed) is biochemically identical "
    "to the active compound in Captopril; this suggests Black seed may "
    "have therapeutic effects on hypertension comparable to Captopril",
    "DOSAGE (Black seed): 1 tsp/day orally",
    "DOSAGE (Black seed preparation): ground or cold-pressed oil",
    "DOSAGE (Captopril): 25-50 mg twice daily",
    "SAFETY (Captopril): Avoid with anticoagulants; not in pregnancy",
    "SAFETY (Captopril side-effects): Dry cough, dizziness",
    'HADITH: "In the black seed is healing for every disease except death"',
    "NOTE: All statements above are derived exclusively from the "
    "PRO-MedGraph knowledge graph.",
]

_MOCK_A0_GOOD = """\
### Prophetic Remedy Recommendation
Black seed (Nigella sativa) is identified as a traditional remedy for \
hypertension based on the PRO-MedGraph knowledge graph.

### Mechanistic Explanation
Black seed contains Thymoquinone and Nigellone. Thymoquinone is \
biochemically identical to the active compound in Captopril (confidence: \
high), suggesting comparable antihypertensive effects.

### Dosage & Preparation
Traditional dosage: 1 tsp/day orally, prepared as ground seeds or \
cold-pressed oil. Modern equivalent: Captopril 25-50 mg twice daily.

### Safety & Contraindications
Captopril should be avoided with anticoagulants and is not recommended \
during pregnancy. Side effects may include dry cough and dizziness.

### Faith-Science Alignment
The Hadith states: "In the black seed is healing for every disease except \
death." Modern biochemical analysis supports this tradition by identifying \
Thymoquinone as a compound with pharmacological activity relevant to blood \
pressure regulation.

### Important Disclaimer
This information is for educational purposes only. Please consult a \
qualified healthcare professional.
"""

_MOCK_VALIDATION_PASS: dict[str, Any] = {
    "grounded": True,
    "safety_score": 0.90,
    "hallucination_flag": False,
    "faith_alignment_ok": True,
    "flagged_claims": [],
    "recommendations": [],
    "validated_answer": _MOCK_A0_GOOD,
}

_MOCK_VALIDATION_FAIL: dict[str, Any] = {
    "grounded": False,
    "safety_score": 0.45,
    "hallucination_flag": True,
    "faith_alignment_ok": False,
    "flagged_claims": [
        "Black seed is a guaranteed cure proven by WHO",
        "Studies from Johns Hopkins show 95% efficacy",
    ],
    "recommendations": [
        "Remove ungrounded WHO claim",
        "Remove Johns Hopkins reference",
        "Add proper hedging language",
    ],
    "validated_answer": (
        "Black seed is traditionally used for hypertension.  "
        "It contains Thymoquinone which may have antihypertensive "
        "properties.  Consult a healthcare professional."
    ),
}

_TEST_CASES: list[dict[str, Any]] = [
    {
        "label": "Validated PASS — full evidence",
        "a0": _MOCK_A0_GOOD,
        "validation": _MOCK_VALIDATION_PASS,
        "blocks": _MOCK_EVIDENCE,
        "query": "What natural remedies help with hypertension?",
    },
    {
        "label": "Validated FAIL — hallucinated draft",
        "a0": "Black seed is a guaranteed cure for hypertension proven by WHO.",
        "validation": _MOCK_VALIDATION_FAIL,
        "blocks": _MOCK_EVIDENCE,
        "query": "What natural remedies help with hypertension?",
    },
    {
        "label": "Empty evidence — edge case",
        "a0": "No specific information available.",
        "validation": {
            "grounded": False,
            "safety_score": 0.5,
            "hallucination_flag": False,
            "faith_alignment_ok": True,
            "flagged_claims": [],
            "recommendations": ["No evidence was available."],
            "validated_answer": "No specific information available.",
        },
        "blocks": [],
        "query": "Hello, how are you?",
    },
]


async def _run_tests_async() -> None:
    import sys

    logging.basicConfig(
        level=logging.INFO,
        stream=sys.stdout,
        format="%(levelname)s  %(message)s",
    )

    print("=" * 72)
    print("  PRO-MedGraph · Response Builder (Step 8) Test Harness")
    print("=" * 72)

    passed = 0
    total = len(_TEST_CASES)

    for tc in _TEST_CASES:
        label = tc["label"]
        print(f"\n── {label} ──")
        print(f"   Query: {tc['query']}")
        print(f"   A0 chars: {len(tc['a0'])}")
        print(f"   Evidence blocks: {len(tc['blocks'])}")
        print(f"   Validation: grounded={tc['validation']['grounded']}  "
              f"safety={tc['validation']['safety_score']}")

        try:
            result = await build_final_answer(
                tc["a0"],
                tc["validation"],
                tc["blocks"],
                tc["query"],
            )
            meta = result["metadata"]
            sf = result["structured_fields"]

            print(f"   ✓ {meta['elapsed_ms']} ms | "
                  f"{meta['final_answer_chars']} chars | "
                  f"model={meta['model']} | "
                  f"fallback={meta['used_fallback']}")
            print(f"   remedy={sf.get('remedy', '?')}")
            print(f"   dosage={sf.get('dosage', '?')}")
            print(f"   confidence={sf.get('confidence')}  "
                  f"safety_score={sf.get('safety_score')}")

            # Preview first 400 chars of the final answer.
            preview = result["final_answer_text"][:400]
            print("   Final answer preview:")
            for line in preview.split("\n")[:8]:
                print(f"     │ {line}")

            t = result["traceability"]
            print(f"   Traceability: {t['evidence_block_count']} blocks, "
                  f"grounded={t['grounded']}")

            passed += 1

        except Exception as exc:
            print(f"   ✗ ERROR: {exc}")

    print("\n" + "=" * 72)
    print(f"  Results:  {passed}/{total} completed")
    print("=" * 72)


def _run_tests() -> None:
    _asyncio.run(_run_tests_async())


if __name__ == "__main__":
    _run_tests()


# ── Module exports ─────────────────────────────────────────────────────────────

__all__ = [
    "build_final_answer",
    "build_answer_from_graph",
]
