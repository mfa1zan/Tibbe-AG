"""
PRO-MedGraph  —  Step 6: Clinical Reasoner Model
==================================================

Accepts the ordered evidence blocks produced by Step 5
(:mod:`evidence_formatter`) and the original user query, then calls the
**large reasoning LLM** (``MODEL_REASONER``) to generate a medically
cautious draft answer (**A0**).

Pipeline position
-----------------
Step 5 (Evidence Formatter) → **Step 6 (Clinical Reasoner)** → Step 7 (Agentic Refinement)

The draft A0 is:
    • Grounded exclusively in the FACT / INFERENCE / DOSAGE / SAFETY /
      HADITH blocks from the knowledge graph — no invented claims.
    • Structured with clear sections: Recommendation, Mechanism, Dosage,
      Safety, and Faith-Science Alignment.
    • Explicitly uncertain where evidence is weak, and always recommends
      professional medical consultation.

A0 is deliberately **not** the final answer.  It is designed to be
re-evaluated and refined by Step 7 (Agentic Refinement / Validator).

Usage
-----
.. code-block:: python

    from app.services.clinical_reasoner import generate_clinical_draft

    blocks = format_graph_for_llm(graph_json)      # Step 5
    a0     = await generate_clinical_draft(blocks, query="What helps with hypertension?")

Run standalone tests:  ``python -m app.services.clinical_reasoner``
"""

from __future__ import annotations

import logging
import time
from typing import Any

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════════
#  LLM caller  (reuses multi_model_service)
# ═══════════════════════════════════════════════════════════════════════════════

from app.services.multi_model_service import _call_groq, config  # noqa: E402

# Extra time for the large reasoner model.
_REASONER_TIMEOUT = 90.0

# ═══════════════════════════════════════════════════════════════════════════════
#  System prompt
# ═══════════════════════════════════════════════════════════════════════════════

_SYSTEM_PROMPT = """\
You are **PRO-MedGraph Clinical Reasoner**, a biomedical assistant that \
integrates Prophetic (Islamic) medicine with modern pharmacology.

═══ STRICT RULES ═══
1. Use ONLY the evidence blocks provided below.  Every claim you make must \
   trace to at least one FACT or INFERENCE block.
2. Do NOT hallucinate, invent, or cite external studies, URLs, or data \
   that are not present in the evidence.
3. Where the evidence is weak or absent, say so explicitly (e.g. "limited \
   evidence", "weak structural match").
4. ALWAYS end with a recommendation to consult a qualified healthcare \
   professional.

═══ RESPONSE STRUCTURE ═══
Organise your answer under these headings (skip a section only if the \
evidence contains no relevant data for it):

### Prophetic Remedy Recommendation
- Which traditional ingredient(s) the knowledge graph identifies for the \
  condition, and their historical/prophetic context.

### Mechanistic Explanation
- The biochemical pathway: Ingredient → Active Compound → Mechanism → \
  Modern Drug Equivalent.
- State the mapping confidence (high / moderate / low) as given in the \
  evidence.

### Dosage & Preparation
- Traditional dosage and preparation method (if provided).
- Corresponding modern drug dosage for context (if provided).

### Safety & Contraindications
- Any contraindications, side-effects, or drug interactions from the \
  evidence.
- If none are available, note that safety data is limited.

### Faith-Science Alignment
- Respectfully connect the Prophetic tradition with the mechanistic \
  evidence.  Use scholarly, non-exclusivist language.  Avoid miracle or \
  guarantee claims.
- Include Hadith references from the evidence verbatim, if present.

### Important Disclaimer
- Remind the user that this information is for educational purposes and \
  does not replace professional medical advice.
"""

# ═══════════════════════════════════════════════════════════════════════════════
#  Public API
# ═══════════════════════════════════════════════════════════════════════════════


async def generate_clinical_draft(
    evidence_blocks: list[str],
    query: str,
    *,
    temperature: float = 0.15,
    extra_instructions: str | None = None,
) -> dict[str, Any]:
    """
    Generate the draft medically cautious answer (A0).

    Parameters
    ----------
    evidence_blocks : list[str]
        The ordered FACT / INFERENCE / DOSAGE / SAFETY / HADITH strings
        produced by :func:`evidence_formatter.format_graph_for_llm`.
    query : str
        The original user question (used as context for the LLM).
    temperature : float
        Sampling temperature.  Kept low (0.15) to favour factual output.
    extra_instructions : str | None
        Optional additional instructions appended to the system prompt
        (e.g. "focus on drug interactions").

    Returns
    -------
    dict
        ``{"a0": str, "metadata": dict}``

        *  ``a0`` — the draft answer text.
        *  ``metadata`` — timing, block count, char count, model used.

    Raises
    ------
    RuntimeError
        If the LLM call fails after internal retries in ``_call_groq``.
    """
    t_start = time.perf_counter()

    # ── Build the user prompt ──────────────────────────────────────────────
    evidence_text = "\n".join(evidence_blocks) if evidence_blocks else "(no evidence available)"

    user_prompt = (
        f"══ USER QUERY ══\n"
        f"{query}\n\n"
        f"══ KNOWLEDGE-GRAPH EVIDENCE ({len(evidence_blocks)} blocks) ══\n"
        f"{evidence_text}"
    )

    # ── Optionally extend system prompt ────────────────────────────────────
    system = _SYSTEM_PROMPT
    if extra_instructions:
        system += f"\n\n═══ ADDITIONAL INSTRUCTIONS ═══\n{extra_instructions}\n"

    # ── Call the reasoner LLM ──────────────────────────────────────────────
    model_id = config.reasoner

    logger.info(
        "Clinical reasoner: sending %d evidence blocks to %s (temp=%.2f)",
        len(evidence_blocks), model_id, temperature,
    )

    try:
        a0 = await _call_groq(
            model=model_id,
            system_prompt=system,
            user_prompt=user_prompt,
            temperature=temperature,
            timeout=_REASONER_TIMEOUT,
        )
    except Exception as exc:
        elapsed = round((time.perf_counter() - t_start) * 1000, 1)
        logger.error(
            "Clinical reasoner failed after %.1f ms: %s — %s",
            elapsed, type(exc).__name__, exc,
        )
        raise RuntimeError(
            f"Clinical reasoner LLM call failed: {exc}"
        ) from exc

    elapsed_ms = round((time.perf_counter() - t_start) * 1000, 1)

    logger.info(
        "Clinical reasoner: A0 generated in %.1f ms — %d chars, model=%s",
        elapsed_ms, len(a0), model_id,
    )

    return {
        "a0": a0,
        "metadata": {
            "model": model_id,
            "evidence_block_count": len(evidence_blocks),
            "a0_char_count": len(a0),
            "temperature": temperature,
            "elapsed_ms": elapsed_ms,
            "query_preview": query[:120],
        },
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  Convenience: generate from Step 4 output directly
# ═══════════════════════════════════════════════════════════════════════════════


async def generate_clinical_draft_from_graph(
    graph_json: dict[str, Any],
    query: str,
    *,
    enrich_dosage: bool = True,
    temperature: float = 0.15,
) -> dict[str, Any]:
    """
    End-to-end convenience: Step 4 output → Step 5 formatting → Step 6 A0.

    Parameters
    ----------
    graph_json : dict
        The canonical output from :func:`graph_retrieval.retrieve_graph`.
    query : str
        The original user question.
    enrich_dosage : bool
        Passed through to the evidence formatter.
    temperature : float
        Sampling temperature for the reasoner LLM.

    Returns
    -------
    dict
        Same shape as :func:`generate_clinical_draft` output, with an
        extra ``evidence_blocks`` key for traceability.
    """
    from app.services.evidence_formatter import format_graph_for_llm

    blocks = format_graph_for_llm(graph_json, enrich_dosage=enrich_dosage)
    result = await generate_clinical_draft(blocks, query, temperature=temperature)
    result["evidence_blocks"] = blocks
    return result


# ═══════════════════════════════════════════════════════════════════════════════
#  Standalone test harness
# ═══════════════════════════════════════════════════════════════════════════════
#
#  Run:  python -m app.services.clinical_reasoner
#  (from backend/ dir with .env present and Groq API reachable)

import asyncio as _asyncio

_MOCK_EVIDENCE_BLOCKS: list[str] = [
    "FACT: The query concerns the disease/condition 'hypertension'",
    "FACT: Black seed is a traditional remedy for hypertension",
    "FACT: Black seed contains Thymoquinone",
    "FACT: Black seed contains Nigellone",
    "FACT: Thymoquinone maps to pharmaceutical compound in Captopril",
    "FACT: Thymoquinone is biochemically identical to the active compound in Captopril (confidence: high)",
    "INFERENCE: Thymoquinone (from Black seed) is biochemically identical to "
    "the active compound in Captopril; this suggests Black seed may have "
    "therapeutic effects on hypertension comparable to Captopril",
    "INFERENCE: Nigellone (from Black seed) is biochemically identical to "
    "the active compound in Captopril; this suggests Black seed may have "
    "therapeutic effects on hypertension comparable to Captopril",
    "DOSAGE (Black seed): 1 tsp/day orally",
    "DOSAGE (Black seed preparation): ground or cold-pressed oil",
    "DOSAGE (Captopril): 25-50 mg twice daily",
    "SAFETY (Captopril): Avoid with anticoagulants; not in pregnancy",
    "SAFETY (Captopril side-effects): Dry cough, dizziness",
    'HADITH: "In the black seed is healing for every disease except death"',
    "NOTE: All statements above are derived exclusively from the PRO-MedGraph "
    "knowledge graph. No external or speculative claims are included.",
]

_TEST_QUERIES: list[dict[str, Any]] = [
    {
        "label": "Hypertension — full evidence",
        "query": "What natural remedies can help with hypertension?",
        "blocks": _MOCK_EVIDENCE_BLOCKS,
    },
    {
        "label": "Minimal evidence (no dosage/safety)",
        "query": "Are there any Prophetic remedies for migraines?",
        "blocks": [
            "FACT: The query concerns the disease/condition 'migraine'",
            "FACT: Peppermint is a traditional remedy for migraine",
            "INFERENCE: Peppermint is traditionally used for migraine; "
            "no modern drug mapping found in the knowledge graph",
            "NOTE: All statements above are derived exclusively from the "
            "PRO-MedGraph knowledge graph.",
        ],
    },
    {
        "label": "Empty evidence (skipped query)",
        "query": "Hello, how are you?",
        "blocks": [],
    },
]


async def _run_tests_async() -> None:
    """Execute test queries against the live Groq API and print results."""
    import sys

    logging.basicConfig(
        level=logging.INFO,
        stream=sys.stdout,
        format="%(levelname)s  %(message)s",
    )

    print("=" * 72)
    print("  PRO-MedGraph · Clinical Reasoner (Step 6) Test Harness")
    print("=" * 72)

    passed = 0
    total = len(_TEST_QUERIES)

    for tc in _TEST_QUERIES:
        label = tc["label"]
        print(f"\n── {label} ──")
        print(f"   Query:  {tc['query']}")
        print(f"   Blocks: {len(tc['blocks'])}")

        try:
            result = await generate_clinical_draft(
                tc["blocks"],
                tc["query"],
            )
            meta = result["metadata"]
            a0 = result["a0"]

            print(f"   ✓ {meta['elapsed_ms']} ms | {meta['a0_char_count']} chars | model={meta['model']}")
            # Print first 500 chars of A0 for inspection.
            preview = a0[:500] + ("…" if len(a0) > 500 else "")
            for line in preview.split("\n"):
                print(f"   │ {line}")
            passed += 1

        except Exception as exc:
            print(f"   ✗ FAILED: {exc}")

    print("\n" + "=" * 72)
    print(f"  Results:  {passed}/{total} passed")
    print("=" * 72)


def _run_tests() -> None:
    _asyncio.run(_run_tests_async())


if __name__ == "__main__":
    _run_tests()


# ── Module exports ─────────────────────────────────────────────────────────────

__all__ = [
    "generate_clinical_draft",
    "generate_clinical_draft_from_graph",
]
