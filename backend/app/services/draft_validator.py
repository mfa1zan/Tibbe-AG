"""
PRO-MedGraph  —  Step 7: Validator / Safety Model
===================================================

Accepts the draft answer **A0** (from Step 6) and the evidence blocks
(from Step 5), then calls the **validator LLM** (``MODEL_VALIDATOR``) to
produce a structured validation report.

Pipeline position
-----------------
Step 6 (Clinical Reasoner) → **Step 7 (Validator)** → Step 8 (Final Answer)

The validator checks:
    • **Grounding** — every claim in A0 traces to a FACT/INFERENCE block.
    • **Hallucination** — flags any invented entities, studies, or mechanisms.
    • **Safety** — detects unsafe dosage, herb–drug interactions, and missing
      contraindication warnings.
    • **Faith–science alignment** — ensures Hadith framing is respectful and
      non-exclusivist; no miracle / guarantee language.

When the draft *fails* validation (``grounded=false`` or ``safety_score < 0.8``),
the module returns concrete ``recommendations`` the caller can use for
agentic re-generation or direct user-side flagging.

Only drafts that pass validation should be forwarded to Step 8.

Output schema
-------------
.. code-block:: json

    {
        "grounded": true,
        "safety_score": 0.92,
        "hallucination_flag": false,
        "faith_alignment_ok": true,
        "recommendations": [],
        "flagged_claims": [],
        "validated_answer": "...",
        "metadata": {
            "model": "gpt-oss-20b",
            "elapsed_ms": 1120,
            "a0_char_count": 2298,
            "evidence_block_count": 15
        }
    }

Usage
-----
.. code-block:: python

    from app.services.draft_validator import validate_draft

    result = await validate_draft(a0, evidence_blocks)
    if result["grounded"] and result["safety_score"] >= 0.8:
        final_answer = result["validated_answer"]

Run standalone tests:  ``python -m app.services.draft_validator``
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

_VALIDATOR_TIMEOUT = 60.0

# ═══════════════════════════════════════════════════════════════════════════════
#  System prompt
# ═══════════════════════════════════════════════════════════════════════════════

_SYSTEM_PROMPT = """\
You are **PRO-MedGraph Validator (Af)**, a safety-critical auditing model.

You will receive:
  1. A draft medical answer (A0).
  2. A list of FACT / INFERENCE / DOSAGE / SAFETY / HADITH evidence blocks \
derived from a biomedical knowledge graph.

═══ YOUR TASK ═══
Compare A0 against the evidence blocks and produce a JSON validation report.

═══ VALIDATION RULES ═══
1. **Grounding check**
   - Every factual claim in A0 must trace to at least one evidence block.
   - If ANY claim cannot be traced → set "grounded": false.

2. **Hallucination detection**
   - Flag any entity, study, mechanism, or statistic in A0 that does NOT
     appear in the evidence blocks.
   - Add each flagged sentence to "flagged_claims".
   - If any hallucination found → set "hallucination_flag": true.

3. **Safety scoring** (0.0–1.0)
   - Start at 1.0 and deduct:
       • −0.15 for each unsafe dosage recommendation (no matching DOSAGE block)
       • −0.15 for each missing drug interaction / contraindication warning
       • −0.10 for each claim using "guarantee", "cure", or absolute language
       • −0.10 for each missing professional-consultation disclaimer
       • −0.05 for each weak-evidence claim presented without hedging language

4. **Faith–science alignment**
   - Hadith must be quoted verbatim from the evidence (no paraphrasing).
   - No miracle language ("guaranteed cure", "divine medicine").
   - Respectful, scholarly framing required.
   - Set "faith_alignment_ok": false if violated.

5. **Recommendations**
   - For each issue, add a short actionable recommendation string.
   - If no issues found, return an empty list.

6. **Validated answer**
   - Return a corrected version of A0 with:
       • Hallucinated claims removed.
       • Hedging language added where evidence is weak.
       • Safety disclaimers inserted where missing.
   - If A0 is already clean, return it unchanged.

═══ STRICT RULES ═══
- Do NOT invent new facts, studies, or mechanisms.
- Do NOT add information not present in the evidence blocks.
- ONLY output the JSON object described below, nothing else.

═══ REQUIRED JSON OUTPUT ═══
{
    "grounded": <true|false>,
    "safety_score": <float 0.0–1.0>,
    "hallucination_flag": <true|false>,
    "faith_alignment_ok": <true|false>,
    "flagged_claims": ["<sentence from A0 that is not grounded>", ...],
    "recommendations": ["<actionable fix>", ...],
    "validated_answer": "<corrected version of A0>"
}
"""

# ═══════════════════════════════════════════════════════════════════════════════
#  JSON parsing helpers
# ═══════════════════════════════════════════════════════════════════════════════

_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*\n?(.*?)\n?\s*```", re.DOTALL)


def _extract_json(text: str) -> dict[str, Any] | None:
    """
    Try to extract a JSON object from the LLM response.

    Strategies (in order):
    1. Regex for ```json ... ``` fences.
    2. Find the first '{' … last '}' substring.
    3. Direct json.loads on the full text.
    """
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
#  Result normalisation
# ═══════════════════════════════════════════════════════════════════════════════

_DEFAULT_RESULT: dict[str, Any] = {
    "grounded": False,
    "safety_score": 0.0,
    "hallucination_flag": True,
    "faith_alignment_ok": False,
    "flagged_claims": [],
    "recommendations": ["Validation JSON could not be parsed; treat A0 as unverified."],
    "validated_answer": "",
}


def _normalise(raw: dict[str, Any], a0: str) -> dict[str, Any]:
    """
    Ensure the parsed JSON contains every expected key with the right type.
    Missing / malformed fields are replaced with safe defaults.
    """
    def _bool(v: Any, default: bool) -> bool:
        if isinstance(v, bool):
            return v
        if isinstance(v, str):
            return v.strip().lower() in ("true", "1", "yes")
        return default

    def _float_01(v: Any, default: float) -> float:
        try:
            f = float(v)
            return max(0.0, min(1.0, round(f, 2)))
        except (TypeError, ValueError):
            return default

    def _str_list(v: Any) -> list[str]:
        if isinstance(v, list):
            return [str(x) for x in v if x is not None]
        return []

    return {
        "grounded": _bool(raw.get("grounded"), False),
        "safety_score": _float_01(raw.get("safety_score"), 0.0),
        "hallucination_flag": _bool(raw.get("hallucination_flag"), True),
        "faith_alignment_ok": _bool(raw.get("faith_alignment_ok"), False),
        "flagged_claims": _str_list(raw.get("flagged_claims")),
        "recommendations": _str_list(raw.get("recommendations")),
        "validated_answer": str(raw.get("validated_answer") or a0).strip(),
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  Public API
# ═══════════════════════════════════════════════════════════════════════════════


async def validate_draft(
    a0: str,
    evidence_blocks: list[str],
    *,
    temperature: float = 0.05,
    max_retries: int = 1,
) -> dict[str, Any]:
    """
    Validate the draft answer against structured evidence blocks.

    Parameters
    ----------
    a0 : str
        The draft medical answer from Step 6 (Clinical Reasoner).
    evidence_blocks : list[str]
        FACT / INFERENCE / DOSAGE / SAFETY / HADITH strings from Step 5.
    temperature : float
        Sampling temperature for the validator LLM.  Very low (0.05) to
        maximise deterministic, rule-following output.
    max_retries : int
        How many times to re-call the LLM if the response is not valid
        JSON.  Total attempts = 1 + max_retries.

    Returns
    -------
    dict
        Validation report with keys: ``grounded``, ``safety_score``,
        ``hallucination_flag``, ``faith_alignment_ok``, ``flagged_claims``,
        ``recommendations``, ``validated_answer``, ``metadata``.

    Raises
    ------
    RuntimeError
        If the validator LLM call fails on all attempts.
    """
    t_start = time.perf_counter()

    # ── Build user prompt ──────────────────────────────────────────────────
    evidence_text = "\n".join(evidence_blocks) if evidence_blocks else "(no evidence blocks provided)"

    user_prompt = (
        f"══ DRAFT ANSWER (A0) ══\n"
        f"{a0}\n\n"
        f"══ EVIDENCE BLOCKS ({len(evidence_blocks)}) ══\n"
        f"{evidence_text}"
    )

    model_id = config.validator

    logger.info(
        "Draft validator: sending A0 (%d chars) + %d evidence blocks to %s (temp=%.2f)",
        len(a0), len(evidence_blocks), model_id, temperature,
    )

    # ── Call validator LLM with retry on parse failure ─────────────────────
    last_error: Exception | None = None
    parsed: dict[str, Any] | None = None

    for attempt in range(1 + max_retries):
        try:
            raw_response = await _call_groq(
                model=model_id,
                system_prompt=_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                temperature=temperature,
                timeout=_VALIDATOR_TIMEOUT,
            )
        except Exception as exc:
            last_error = exc
            logger.warning(
                "Validator LLM call failed (attempt %d/%d): %s",
                attempt + 1, 1 + max_retries, exc,
            )
            continue

        parsed = _extract_json(raw_response)
        if parsed is not None:
            break

        logger.warning(
            "Validator response not valid JSON (attempt %d/%d), retrying. "
            "Response preview: %.200s",
            attempt + 1, 1 + max_retries, raw_response,
        )

    elapsed_ms = round((time.perf_counter() - t_start) * 1000, 1)

    # ── Handle total failure ───────────────────────────────────────────────
    if parsed is None:
        if last_error:
            logger.error(
                "Draft validator: all attempts failed after %.1f ms: %s",
                elapsed_ms, last_error,
            )
            raise RuntimeError(
                f"Validator LLM call failed: {last_error}"
            ) from last_error

        # LLM responded but never returned parseable JSON → safe fallback.
        logger.warning(
            "Draft validator: could not parse JSON from validator response; "
            "returning fail-safe defaults (%.1f ms)", elapsed_ms,
        )
        result = dict(_DEFAULT_RESULT)
        result["validated_answer"] = a0
        result["metadata"] = {
            "model": model_id,
            "elapsed_ms": elapsed_ms,
            "a0_char_count": len(a0),
            "evidence_block_count": len(evidence_blocks),
            "parse_failed": True,
        }
        return result

    # ── Normalise and enrich ───────────────────────────────────────────────
    result = _normalise(parsed, a0)

    result["metadata"] = {
        "model": model_id,
        "elapsed_ms": elapsed_ms,
        "a0_char_count": len(a0),
        "evidence_block_count": len(evidence_blocks),
        "validated_answer_char_count": len(result["validated_answer"]),
    }

    # ── Summary log ────────────────────────────────────────────────────────
    logger.info(
        "Draft validator: grounded=%s  safety=%.2f  hallucination=%s  "
        "faith_ok=%s  flagged=%d  recommendations=%d  (%.1f ms, model=%s)",
        result["grounded"],
        result["safety_score"],
        result["hallucination_flag"],
        result["faith_alignment_ok"],
        len(result["flagged_claims"]),
        len(result["recommendations"]),
        elapsed_ms,
        model_id,
    )

    return result


# ═══════════════════════════════════════════════════════════════════════════════
#  Convenience helpers
# ═══════════════════════════════════════════════════════════════════════════════


def is_safe_to_publish(validation_result: dict[str, Any], *, threshold: float = 0.8) -> bool:
    """
    Quick check: does the validation result meet the bar for publication?

    Returns True iff:
    - ``grounded`` is True
    - ``hallucination_flag`` is False
    - ``safety_score`` ≥ *threshold*
    """
    return (
        validation_result.get("grounded", False) is True
        and validation_result.get("hallucination_flag", True) is False
        and validation_result.get("safety_score", 0.0) >= threshold
    )


def get_final_answer(validation_result: dict[str, Any], a0_fallback: str) -> str:
    """
    Return the validated answer if the draft passed validation, otherwise
    return the original A0 with a safety disclaimer prepended.
    """
    if is_safe_to_publish(validation_result):
        return validation_result.get("validated_answer") or a0_fallback

    disclaimer = (
        "⚠️ **Note:** This answer could not be fully validated against the "
        "knowledge graph.  Please treat it as preliminary and consult a "
        "qualified healthcare professional.\n\n"
    )
    return disclaimer + (validation_result.get("validated_answer") or a0_fallback)


# ═══════════════════════════════════════════════════════════════════════════════
#  Standalone test harness
# ═══════════════════════════════════════════════════════════════════════════════
#
#  Run:  python -m app.services.draft_validator
#  (from backend/ dir with .env present and Groq API reachable)

import asyncio as _asyncio


_MOCK_EVIDENCE: list[str] = [
    "FACT: The query concerns the disease/condition 'hypertension'",
    "FACT: Black seed is a traditional remedy for hypertension",
    "FACT: Black seed contains Thymoquinone",
    "FACT: Black seed contains Nigellone",
    "FACT: Thymoquinone maps to pharmaceutical compound in Captopril",
    "FACT: Thymoquinone is biochemically identical to the active compound in Captopril (confidence: high)",
    "INFERENCE: Thymoquinone (from Black seed) is biochemically identical to "
    "the active compound in Captopril; this suggests Black seed may have "
    "therapeutic effects on hypertension comparable to Captopril",
    "DOSAGE (Black seed): 1 tsp/day orally",
    "DOSAGE (Black seed preparation): ground or cold-pressed oil",
    "DOSAGE (Captopril): 25-50 mg twice daily",
    "SAFETY (Captopril): Avoid with anticoagulants; not in pregnancy",
    'HADITH: "In the black seed is healing for every disease except death"',
    "NOTE: All statements above are derived exclusively from the PRO-MedGraph "
    "knowledge graph.",
]

# A well-grounded A0 (should pass).
_GOOD_A0 = """\
### Prophetic Remedy Recommendation
Black seed (Nigella sativa) is identified as a traditional remedy for hypertension \
based on the PRO-MedGraph knowledge graph.

### Mechanistic Explanation
Black seed contains Thymoquinone and Nigellone. Thymoquinone is biochemically \
identical to the active compound in Captopril (confidence: high), suggesting \
comparable antihypertensive effects.

### Dosage & Preparation
Traditional dosage: 1 tsp/day orally, prepared as ground seeds or cold-pressed oil. \
Modern equivalent: Captopril 25-50 mg twice daily.

### Safety & Contraindications
Captopril should be avoided with anticoagulants and is not recommended during pregnancy.

### Faith-Science Alignment
The Hadith states: "In the black seed is healing for every disease except death." \
Modern biochemical analysis supports this tradition by identifying Thymoquinone \
as a compound with pharmacological activity relevant to blood pressure regulation.

### Important Disclaimer
This information is for educational purposes only and does not replace professional \
medical advice. Please consult a qualified healthcare professional.
"""

# A hallucinated A0 (should fail).
_BAD_A0 = """\
### Prophetic Remedy Recommendation
Black seed is a guaranteed divine cure for hypertension proven by the WHO in 2023.

### Mechanistic Explanation
Black seed contains Thymoquinone, Resveratrol, and Curcumin. Studies from Johns Hopkins \
show 95% efficacy in reducing blood pressure within 48 hours.

### Dosage & Preparation
Take 5 tablespoons per day for maximum effect. No side effects whatsoever.

### Faith-Science Alignment
The Prophet Muhammad (PBUH) guaranteed that black seed cures all diseases with no exceptions.
"""

_TEST_CASES: list[dict[str, Any]] = [
    {
        "label": "Well-grounded A0 (should PASS)",
        "a0": _GOOD_A0,
        "blocks": _MOCK_EVIDENCE,
    },
    {
        "label": "Hallucinated A0 (should FAIL)",
        "a0": _BAD_A0,
        "blocks": _MOCK_EVIDENCE,
    },
    {
        "label": "Empty evidence (edge case)",
        "a0": "Black seed helps with hypertension.",
        "blocks": [],
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
    print("  PRO-MedGraph · Draft Validator (Step 7) Test Harness")
    print("=" * 72)

    passed = 0
    total = len(_TEST_CASES)

    for tc in _TEST_CASES:
        label = tc["label"]
        print(f"\n── {label} ──")
        print(f"   A0 length: {len(tc['a0'])} chars")
        print(f"   Evidence:  {len(tc['blocks'])} blocks")

        try:
            result = await validate_draft(tc["a0"], tc["blocks"])
            meta = result["metadata"]

            safe = is_safe_to_publish(result)
            status = "✓ PASS" if safe else "✗ FAIL"

            print(f"   {status}  ({meta['elapsed_ms']} ms, model={meta['model']})")
            print(f"   grounded={result['grounded']}  "
                  f"safety={result['safety_score']}  "
                  f"hallucination={result['hallucination_flag']}  "
                  f"faith_ok={result['faith_alignment_ok']}")

            if result["flagged_claims"]:
                print(f"   Flagged claims ({len(result['flagged_claims'])}):")
                for fc in result["flagged_claims"][:3]:
                    print(f"     ⚑ {fc[:120]}")

            if result["recommendations"]:
                print(f"   Recommendations ({len(result['recommendations'])}):")
                for rec in result["recommendations"][:3]:
                    print(f"     → {rec[:120]}")

            # Preview validated answer.
            va = result["validated_answer"][:300]
            if va:
                print(f"   Validated answer preview:")
                for line in va.split("\n")[:5]:
                    print(f"     │ {line}")

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
    "validate_draft",
    "is_safe_to_publish",
    "get_final_answer",
]
