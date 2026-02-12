"""Post-hoc processing for WattBot experiment results.

Reads a results.json produced by run_experiment.py, applies answer-value
and ref-id normalization, re-scores against ground truth, and writes a
normalised submission.csv ready for Kaggle upload.

Usage
-----
    python scripts/posthoc.py path/to/results.json              # process one
    python scripts/posthoc.py path/to/experiment_dir             # auto-find results.json
    python scripts/posthoc.py path/to/results.json --dry-run     # print score, don't write

This is the **single canonical location** for all answer normalisation.
Neither the experiment runner nor the vendor pipeline should duplicate this
logic — they save raw model output, and this script cleans it up.
"""

from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path
from typing import Sequence

# ============================================================================
# BLANK / NA helpers
# ============================================================================

_BLANK_TOKENS = {"", "is_blank", "na", "n/a", "none", "null"}
_FALLBACK_PHRASE = "unable to answer with confidence based on the provided documents"


def is_blank(x) -> bool:
    val = str(x).strip().lower() if x is not None else ""
    return val in _BLANK_TOKENS or val.startswith(_FALLBACK_PHRASE)


# ============================================================================
# Answer-value normalisation
# ============================================================================

_TRUE_TOKENS = {"true", "yes"}
_FALSE_TOKENS = {"false", "no"}

# Numeric ranges: "80-90", "80 – 90", "80 to 90", "from 80 to 90"
_RANGE_RE = re.compile(
    r"^(?:from\s+)?"
    r"(-?[\d.]+)"
    r"\s*(?:[-–—]|to)\s*"
    r"(-?[\d.]+)$",
    re.IGNORECASE,
)

# Magnitude suffix multipliers: 2B → 2_000_000_000
_MAGNITUDE_MAP = {"k": 1e3, "m": 1e6, "b": 1e9, "t": 1e12}
_MAGNITUDE_RE = re.compile(r"^(-?\d+(?:\.\d+)?)\s*([KkMmBbTt])$")

# Hedging / qualifier prefixes
_HEDGING_RE = re.compile(
    r"^(?:(?:more\s+than|less\s+than|greater\s+than|fewer\s+than"
    r"|approximately|about|around|over|under|nearly|roughly"
    r"|up\s+to|at\s+least|at\s+most)\s+"
    r"|[~≈><]\s*)",
    re.IGNORECASE,
)

# Trailing parenthetical abbreviation: "Function (CTCF)" → "Function"
_TRAILING_PAREN_RE = re.compile(r"\s*\([^()]{1,15}\)\s*$")


def _fmt_num(v: float) -> str:
    """Format a number: drop trailing .0 for integers."""
    return str(int(v)) if v == int(v) else str(v)


def _strip_commas(s: str) -> str:
    """Remove thousand-separator commas from a numeric-looking string."""
    candidate = s.replace(",", "")
    try:
        float(candidate)
        return candidate
    except ValueError:
        return s


def normalize_answer_value(raw: str, question: str = "") -> str:
    """Normalise an LLM answer_value to match expected ground-truth formats.

    Transformations applied (in order):
      1. Strip thousand-separator commas   ("10,000"  → "10000")
      2. True / False / Yes / No           → "1" / "0"
      3. Strip hedging prefixes            ("more than 10000" → "10000")
      4. Expand magnitude suffixes         ("2B" → "2000000000")
      5. Numeric range normalisation       ("80-90" → "[80,90]")
      6. Strip trailing parenthetical abbr ("Function (CTCF)" → "Function")

    Parameters
    ----------
    raw : str
        The raw answer_value string from the LLM.
    question : str, optional
        The question text.  When provided and the question starts with
        "True or False", fuzzy boolean detection is applied (e.g. an answer
        containing "true" but not "false" is mapped to "1").
    """
    s = str(raw).strip()
    if not s or is_blank(s):
        return s if s else "is_blank"

    # ── Step 1: Strip thousand-separator commas ──────────────────────────
    s = _strip_commas(s)

    low = s.lower()

    # ── Step 2: Boolean normalisation ────────────────────────────────────
    if low in _TRUE_TOKENS:
        return "1"
    if low in _FALSE_TOKENS:
        return "0"

    # Question-aware fuzzy boolean (from vendor logic)
    if question and question.strip().lower().startswith("true or false"):
        if "true" in low and "false" not in low:
            return "1"
        if "false" in low and "true" not in low:
            return "0"

    # ── Step 3: Strip hedging prefixes ───────────────────────────────────
    m_hedge = _HEDGING_RE.match(s)
    if m_hedge:
        s = s[m_hedge.end():].strip()
        s = _strip_commas(s)  # re-strip after removing prefix

    # ── Step 4: Expand magnitude suffixes (2B → 2000000000) ─────────────
    m_mag = _MAGNITUDE_RE.match(s)
    if m_mag:
        num = float(m_mag.group(1))
        multiplier = _MAGNITUDE_MAP[m_mag.group(2).lower()]
        return _fmt_num(num * multiplier)

    # ── Step 5: Numeric range normalisation ──────────────────────────────
    m = _RANGE_RE.match(s)
    if m:
        try:
            a, b = float(m.group(1)), float(m.group(2))
            lo, hi = (a, b) if a <= b else (b, a)
            return f"[{_fmt_num(lo)},{_fmt_num(hi)}]"
        except ValueError:
            pass

    # ── Step 6: Strip trailing parenthetical abbreviation ────────────────
    # Only for categorical (non-numeric) answers.
    try:
        float(s)
    except ValueError:
        stripped = _TRAILING_PAREN_RE.sub("", s).strip()
        if stripped and stripped != s:
            s = stripped

    return s


# ============================================================================
# Ref-id normalisation
# ============================================================================

_REF_ID_MARKER_RE = re.compile(r"\[ref_id=([^\]]+)\]", re.IGNORECASE)
_BLANK_REF_TOKENS = {"", "is_blank", "na", "n/a", "none", "null"}


def _extract_refs_from_string(s: str) -> list[str]:
    """Extract clean ref IDs from a single (possibly messy) string."""
    s = s.strip()
    if not s or s.lower() in _BLANK_REF_TOKENS:
        return []

    # [ref_id=xxx] markers
    markers = _REF_ID_MARKER_RE.findall(s)
    if markers:
        return [m.strip() for m in markers if m.strip()]

    # Comma-separated
    if "," in s:
        return [
            p.strip()
            for p in s.split(",")
            if p.strip() and p.strip().lower() not in _BLANK_REF_TOKENS
        ]

    return [s]


def normalize_ref_id(raw_ref) -> list | str:
    """Normalise LLM ref_id output to a clean list of document IDs.

    Returns a list of strings, or the scalar ``"is_blank"``.
    """
    if isinstance(raw_ref, list):
        cleaned: list[str] = []
        for item in raw_ref:
            cleaned.extend(_extract_refs_from_string(str(item)))
        return cleaned if cleaned else "is_blank"

    s = str(raw_ref).strip()
    if s.lower() in _BLANK_REF_TOKENS:
        return "is_blank"

    extracted = _extract_refs_from_string(s)
    if not extracted:
        return "is_blank"
    if len(extracted) == 1:
        return extracted[0]
    return extracted


# ============================================================================
# CLI: process a results.json
# ============================================================================

def _ref_to_str(ref) -> str:
    """Serialise a normalised ref to the CSV string format."""
    if isinstance(ref, list):
        return json.dumps(ref)
    if ref == "is_blank" or not ref:
        return "is_blank"
    return json.dumps([ref])


def apply_posthoc(results_path: Path, *, dry_run: bool = False) -> None:
    """Read results.json, normalise, re-score, and write submission.csv.

    The results.json is expected to contain raw (un-normalised) pred_value
    and pred_ref fields.  This function:
      1. Applies normalize_answer_value and normalize_ref_id.
      2. Re-computes per-row scores using the official WattBot rubric.
      3. Writes a normalised submission.csv alongside results.json.
      4. Prints component scores and the overall WattBot Score.
    """
    results_path = Path(results_path)
    if results_path.is_dir():
        results_path = results_path / "results.json"
    if not results_path.exists():
        raise FileNotFoundError(results_path)

    with open(results_path) as f:
        results = json.load(f)

    # Lazy-import score helpers (avoids hard pandas dep at import time)
    sys.path.insert(0, str(results_path.parent.parent.parent.parent / "scripts"))
    from score import row_bits, is_blank as score_is_blank  # noqa: E402

    total = len(results)
    val_correct = 0
    ref_total = 0.0
    na_gt: list[bool] = []  # True for truly-NA questions

    submission_rows: list[dict] = []

    for r in results:
        question = r.get("question", "")
        raw_val = r.get("pred_value", "is_blank")
        raw_ref = r.get("pred_ref", "is_blank")

        norm_val = normalize_answer_value(raw_val, question)
        norm_ref = normalize_ref_id(raw_ref)
        norm_ref_str = _ref_to_str(norm_ref)

        gt_value = str(r.get("gt_value", "is_blank"))
        gt_unit = str(r.get("gt_unit", ""))
        gt_ref = str(r.get("gt_ref", "is_blank"))

        bits = row_bits(
            sol={"answer_value": gt_value, "answer_unit": gt_unit, "ref_id": gt_ref},
            sub={"answer_value": norm_val, "answer_unit": gt_unit, "ref_id": norm_ref_str},
        )

        val_ok = bool(bits["val"])
        ref_sc = float(bits["ref"])

        if val_ok:
            val_correct += 1
        ref_total += ref_sc

        if score_is_blank(gt_value):
            na_gt.append(bool(bits["na"]))

        submission_rows.append({
            "id": r["id"],
            "question": question,
            "answer": r.get("pred_explanation", "is_blank"),
            "answer_value": norm_val,
            "answer_unit": gt_unit,
            "ref_id": norm_ref_str,
            "ref_url": json.dumps(r.get("pred_ref_url", [])) if r.get("pred_ref_url") else "is_blank",
            "supporting_materials": r.get("pred_supporting_materials", "") or "is_blank",
            "explanation": r.get("pred_explanation", "is_blank"),
        })

    # Compute aggregate scores
    value_acc = val_correct / total if total else 0.0
    ref_overlap = ref_total / total if total else 0.0
    na_recall = (sum(na_gt) / len(na_gt)) if na_gt else 1.0
    overall = 0.75 * value_acc + 0.15 * ref_overlap + 0.10 * na_recall

    print(f"\n{'=' * 60}")
    print("POST-HOC PROCESSING RESULTS")
    print(f"{'=' * 60}")
    print(f"Questions:     {total}")
    print(f"Value correct: {val_correct}/{total} ({value_acc:.3f})")
    print(f"Ref overlap:   {ref_overlap:.3f}")
    print(f"NA recall:     {na_recall:.3f} (n={len(na_gt)} truly-NA)")
    print(f"\nWATTBOT SCORE: {overall:.4f}")
    print(f"{'=' * 60}")

    if dry_run:
        print("\n[dry-run] No files written.")
        return

    # Write normalised submission.csv
    out_dir = results_path.parent
    sub_path = out_dir / "submission.csv"
    with open(sub_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["id", "question", "answer", "answer_value", "answer_unit",
                        "ref_id", "ref_url", "supporting_materials", "explanation"],
            quoting=csv.QUOTE_ALL,
        )
        writer.writeheader()
        writer.writerows(submission_rows)

    print(f"\nWrote normalised submission: {sub_path}")


# ============================================================================
# Entry point
# ============================================================================

if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    flags = {a for a in sys.argv[1:] if a.startswith("-")}

    if not args:
        print("Usage: python scripts/posthoc.py <results.json|experiment_dir> [--dry-run]")
        sys.exit(1)

    apply_posthoc(Path(args[0]), dry_run="--dry-run" in flags)
