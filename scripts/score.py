import pandas as pd, numpy as np, json, math, re, unicodedata

# Default for import-time use (Kaggle will override in __main__)
IS_KAGGLE = False

# ───────────────────────────────────────────────────────── helpers ──────
class ParticipantVisibleError(Exception):
    '''Shown to competitors when their CSV is malformed.'''
    pass

def _s(x):
    return "" if x is None else str(x)

canon_unit = lambda u: _s(u).strip().lower()   # plug in real mapping if needed

def canon_refs(r):
    """Return sorted, lowercased list of refs. Robust to NaN/singletons."""
    s = _s(r).strip()
    if not s:
        return []
    try:
        if s.startswith("["):
            arr = json.loads(s.replace("'", '"'))
            return sorted([_s(x).strip().lower() for x in arr])
    except Exception:
        pass
    return [s.lower()]

def ref_overlap_score(sol_ref, sub_ref):
    """
    Jaccard overlap on normalized ref-id sets.
    - Exact match → 1.0
    - Partial overlap → (|A∩B| / |A∪B|)
    - Both empty → 1.0
    - One empty, one non-empty → 0.0
    """
    A = set(canon_refs(sol_ref))
    B = set(canon_refs(sub_ref))
    if not A and not B:
        return 1.0
    if not A or not B:
        return 0.0
    inter = len(A & B)
    union = len(A | B)
    return inter / union if union else 0.0

# tokens that mean “no answer supplied”
_BLANK_TOKENS = {"", "na", "n/a", "is_blank"}

_FALLBACK_PHRASE = "Unable to answer with confidence based on the provided documents"

def is_blank(x) -> bool:
    val = _s(x).strip().lower()
    return val in _BLANK_TOKENS or val.startswith(_FALLBACK_PHRASE.lower())

def both_numeric(a: str, b: str) -> bool:
    '''Quick check before math.isclose to avoid ValueError spam.'''
    try:
        float(a); float(b)
        return True
    except ValueError:
        return False

# ───────── lightweight submission-side normalization ──────────
# Applied inside row_bits so that format-only differences
# (commas, magnitude suffixes, range strings) don't cause spurious failures.

_TRUE_TOKENS  = {"true", "yes"}
_FALSE_TOKENS = {"false", "no"}

_MAGNITUDE_MAP = {"k": 1e3, "m": 1e6, "b": 1e9, "t": 1e12}
_MAG_RE   = re.compile(r"^(-?\d+(?:\.\d+)?)\s*([KkMmBbTt])$")
_RANGE_RE = re.compile(
    r"^(?:from\s+)?(-?[\d.]+)\s*(?:[-–—]|to)\s*(-?[\d.]+)$",
    re.IGNORECASE,
)
_HEDGE_RE = re.compile(
    r"^(?:(?:more\s+than|less\s+than|greater\s+than|fewer\s+than"
    r"|approximately|about|around|over|under|nearly|roughly"
    r"|up\s+to|at\s+least|at\s+most)\s+"
    r"|[~≈><]\s*)",
    re.IGNORECASE,
)

def _normalize_sub_value(raw: str) -> str:
    """Best-effort normalization of a submission answer_value.

    Handles boolean mapping (True→1, False→0), comma-thousands,
    magnitude suffixes (2B→2000000000), hedging prefixes, and
    range strings ("80-90"→"[80,90]").
    """
    s = _s(raw).strip()
    if not s or is_blank(s):
        return s

    # boolean mapping (True/False/Yes/No → 1/0)
    low = s.lower()
    if low in _TRUE_TOKENS:
        return "1"
    if low in _FALSE_TOKENS:
        return "0"

    # strip commas
    no_comma = s.replace(",", "")
    try:
        float(no_comma)
        s = no_comma
    except ValueError:
        pass

    # strip hedging
    hm = _HEDGE_RE.match(s)
    if hm:
        s = s[hm.end():].strip()
        nc = s.replace(",", "")
        try:
            float(nc); s = nc
        except ValueError:
            pass

    # magnitude suffix
    mm = _MAG_RE.match(s)
    if mm:
        val = float(mm.group(1)) * _MAGNITUDE_MAP[mm.group(2).lower()]
        return str(int(val)) if val == int(val) else str(val)

    # range string → bracket list
    rm = _RANGE_RE.match(s)
    if rm:
        try:
            a, b = float(rm.group(1)), float(rm.group(2))
            lo, hi = (a, b) if a <= b else (b, a)
            fmt = lambda v: str(int(v)) if v == int(v) else str(v)
            return f"[{fmt(lo)},{fmt(hi)}]"
        except ValueError:
            pass

    return s

def parse_listish(x):
    """Parse list/set-like strings into Python lists.
    - [a, b] : list, can be numeric range or categorical list
    - {a, b} : set of terms, order-insensitive, lowercase
    Returns None if not a collection.
    """
    s = _s(x).strip()
    if not s:
        return None

    # square bracket list
    if s.startswith("[") and s.endswith("]"):
        try:
            arr = json.loads(s.replace("'", '"'))
        except Exception:
            arr = [t.strip() for t in s[1:-1].split(",")]
        out = []
        for t in arr:
            t = _s(t).strip()
            try:
                out.append(float(t))
            except ValueError:
                out.append(t.lower())
        return out

    # curly brace set of terms
    if s.startswith("{") and s.endswith("}"):
        inner = s[1:-1]
        terms = [t.strip().lower() for t in inner.split(",") if t.strip()]
        return sorted(set(terms))  # normalized set

    return None

def compare_lists(a, b, tol=1e-3):
    """Order-insensitive list equality, numeric with tolerance."""
    a_nums = all(isinstance(v, (int, float)) for v in a)
    b_nums = all(isinstance(v, (int, float)) for v in b)
    if a_nums and b_nums:
        A = sorted(float(x) for x in a)
        B = sorted(float(x) for x in b)
        if len(A) != len(B):
            return False
        return all(math.isclose(x, y, rel_tol=tol, abs_tol=tol) for x, y in zip(A, B))
    # string-ish compare
    A = sorted(_s(x).strip().lower() for x in a)
    B = sorted(_s(x).strip().lower() for x in b)
    return A == B

def row_bits(sol, sub, tol: float = 1e-3):
    """
    Per-row rubric components:
      val / unit / ref / na → booleans/floats used later for weighting.

    Changes:
    - ref: now a fractional Jaccard overlap in [0,1] instead of binary.
    """
    # blank / NA component
    if is_blank(sol["answer_value"]):
        na_ok = is_blank(sub["answer_value"])
    else:
        na_ok = True  # not a 'NA' case, don't penalize

    # value component (supports bracketed lists/ranges)
    if is_blank(sol["answer_value"]):
        val_ok = True  # nothing expected
    else:
        # Normalize submission value to handle commas, abbreviations, range strings
        sub_val = _normalize_sub_value(sub["answer_value"])

        sol_list = parse_listish(sol["answer_value"])
        sub_list = parse_listish(sub_val)

        if sol_list is not None:
            if sub_list is not None:
                # list-to-list comparison (order-insensitive, numeric tol)
                val_ok = compare_lists(sol_list, sub_list, tol=tol)
            else:
                # If solution is a numeric 2-tuple (treat as range), allow single number within range
                if len(sol_list) == 2 and all(isinstance(v, (int, float)) for v in sol_list):
                    lo, hi = sorted(sol_list)
                    try:
                        x = float(sub_val)
                        val_ok = (lo - tol) <= x <= (hi + tol)
                    except ValueError:
                        val_ok = False
                else:
                    # solution wanted a categorical list; submission didn't provide a list
                    val_ok = False
        else:
            if both_numeric(sol["answer_value"], sub_val):
                val_ok = math.isclose(
                    float(sol["answer_value"]),
                    float(sub_val),
                    rel_tol=tol, abs_tol=tol
                )
            else:  # named entity or free-text (normalized)
                val_ok = (
                    _s(sol["answer_value"]).strip().lower()
                    == _s(sub_val).strip().lower()
                )

    # unit + ref components (case/whitespace insensitive)
    unit_ok = canon_unit(sol["answer_unit"]) == canon_unit(sub["answer_unit"])
    ref_score  = ref_overlap_score(sol["ref_id"], sub["ref_id"])  # ← fractional

    return {"val": val_ok, "unit": unit_ok, "ref": ref_score, "na": na_ok}

# ───────────── explanation-only evidence classification ─────
def _norm(x: str) -> str:
    return unicodedata.normalize("NFKC", _s(x)).lower()

# Regexes tuned for phrasing that tends to appear in EXPLANATION
RE_TABLE = re.compile(
    r"\btable(?:s)?\b|\btab\.\b|\btbl\b|\bappendix\s+table\b|\bsupp(?:lement(?:ary)?)?\s+table\b|\btable\s*s?\d+\b",
    re.IGNORECASE,
)
RE_FIG = re.compile(
    r"\bfig(?:ure)?s?\b|\bfig\.\b|\bfigure\s*s?\d+\b|\bpanel\s*[a-z]\b|\bexhibit\s*\d+\b|\bchart\b|\bdiagram\b|\bplot\b|\bgraph\b",
    re.IGNORECASE,
)
RE_QUOTE = re.compile(
    r"\bquote[sd]?\b|“[^”]{10,}”|\"[^\"]{10,}\"|‘[^’]{10,}’",
    re.IGNORECASE,
)

def classify_from_explanation(expl: str) -> str:
    """
    Classify primary evidence type based ONLY on EXPLANATION text.
    Returns one of: {'Tables','Figures','Quotes','Other'}.
    Priority: Tables > Figures > Quotes > Other (assume tab/fig mentions are primary).
    """
    t = _norm(expl)
    if not t or t == "is_blank":
        return "Other"
    if RE_TABLE.search(t):
        return "Tables"
    if RE_FIG.search(t):
        return "Figures"
    if RE_QUOTE.search(t):
        return "Quotes"
    return "Other"

def classify_from_boolean(row: pd.Series) -> str:
    """Map the SOLUTION’s boolean flags to a single evidence label.
       Priority: Tables > Figures > Quotes > Math > NA > Other.
    """
    def truthy(x):
        s = str(x).strip().lower()
        return s in ("1", "true", "t", "yes", "y")

    if truthy(row.get("Quote", 0)):   return "Quotes"
    if truthy(row.get("Table", 0)):   return "Tables"
    if truthy(row.get("Figure", 0)):  return "Figures"
    if truthy(row.get("Math", 0)):    return "Math"
    if truthy(row.get("is_NA", 0)):   return "NA"
    return "Other"

# ───────────────────────────────────────────────── metric entry point ──
def score(solution: pd.DataFrame,
          submission: pd.DataFrame) -> float:
    """
    WattBot Score (0–1)

    Submissions are scored with a custom WattBot Score that evaluates four fields
    for every question and returns a weighted accuracy:

    Component      Weight  What counts as correct
    ------------------------------------------------------------
    answer_value    0.75   Matches the ground truth. Numeric answers must be within
                           ±0.1% relative tolerance (and ~1e-3 absolute fallback);
                           categorical values must match exactly after normalization.
                           If a question is unanswerable, this column must contain
                           'is_blank'.

    ref_id          0.15   **Partial credit via Jaccard overlap** between your ref_id
                           set and the ground-truth set (order ignored, case-insensitive).
                           Use 'is_blank' if no evidence is available.

    is_NA (NA)      0.10   **Recall** over truly unanswerable questions: of all
                           ground-truth NA questions, what fraction did the model
                           correctly mark as 'is_blank'?  This avoids inflating
                           the score when most questions are answerable (the value
                           component already penalizes false NAs).

    Notes:
    - Output: the scorer prints per-component accuracies/scores and the final score; it
      returns the final score as a float in [0, 1].
    """

    # Required columns: solution doesn't need 'explanation'; submission does.
    required_sol = ["id", "answer_value", "answer_unit", "ref_id"]
    required_sub = ["id", "answer_value", "answer_unit", "ref_id", "explanation"]
    payload_cols = ["answer_value", "answer_unit", "ref_id"]  # used after set_index

    miss_sol = set(required_sol) - set(solution.columns)
    if miss_sol:
        raise ParticipantVisibleError(f"solution missing columns: {sorted(miss_sol)}")

    miss_sub = set(required_sub) - set(submission.columns)
    if miss_sub:
        raise ParticipantVisibleError(f"submission missing columns: {sorted(miss_sub)}")

    # Normalize to strings for stability (only columns that exist)
    for col in required_sol:
        solution[col] = solution[col].astype(str)
    for col in required_sub:
        submission[col] = submission[col].astype(str)

    if submission["explanation"].str.strip().eq("").any():
        raise ParticipantVisibleError("Each row needs a non-empty explanation.")

    # Index on id
    sol = solution.set_index("id")
    sub = submission.set_index("id")

    # IMPORTANT: don't try to select 'id' as a column after set_index
    merged = sol.join(sub[payload_cols], lsuffix="_sol", rsuffix="_sub", how="inner")

    if merged.empty:
        raise ParticipantVisibleError("No matching IDs between submission and solution.")

    bits = merged.apply(
        lambda r: row_bits(
            sol=r.filter(like="_sol").rename(lambda c: c[:-4]),
            sub=r.filter(like="_sub").rename(lambda c: c[:-4])
        ),
        axis=1, result_type="expand"
    )

    comp = bits.mean(numeric_only=True)

    # NA component: use RECALL over truly-NA questions only.
    # The old approach averaged na_ok over ALL rows, where non-NA rows always
    # score True — washing out the signal (e.g., 240/250 = 0.96 even if every
    # NA question is wrong).  Recall = (correctly abstained) / (truly NA).
    na_gt_mask = merged["answer_value_sol"].apply(is_blank)
    na_gt_count = int(na_gt_mask.sum())
    if na_gt_count > 0:
        na_recall = float(bits.loc[na_gt_mask, "na"].mean())
    else:
        na_recall = 1.0  # no NA questions in this set → perfect by default

    overall = (
        0.75*comp["val"] +
        0.00*comp["unit"] +  # unit remains binary & weight=0
        0.15*comp["ref"] +   # now the mean Jaccard score
        0.10*na_recall       # recall over truly-NA questions only
    )

    # ----- Component accuracy / score
    print("Component scores (means):")
    print(f"  value match  : {comp['val']:.3f}")
    print(f"  unit match   : {comp['unit']:.3f}")
    print(f"  ref overlap  : {comp['ref']:.3f}")  # renamed
    print(f"  NA recall    : {na_recall:.3f}  (n={na_gt_count} truly-NA questions)")
    print(f"OVERALL SCORE  : {overall:.3f}\n")

    # ----- Answer-value accuracy by coded evidence flags (multi-label; SOLUTION ONLY, NO FALLBACK)
    required_flags = ["Quote", "Table", "Figure", "Math", "is_NA"]
    missing_flags = [c for c in required_flags if c not in sol.columns]
    if missing_flags:
        raise ParticipantVisibleError(f"solution missing coded columns: {sorted(missing_flags)}")

    def _truthy(x):
        s = str(x).strip().lower()
        return s in ("1", "true", "t", "yes", "y")

    print("Answer-value accuracy by coded evidence flags (multi-label; from SOLUTION):")
    for label in ["Quote", "Table", "Figure",  "Math", "is_NA"]:  # omit "Other" on purpose
        mask = sol.loc[bits.index, label].apply(_truthy)
        if mask.any():
            mean = float(bits.loc[mask, "val"].mean())
            total = int(mask.sum())
            correct = int(bits.loc[mask, "val"].sum())
            print(f"  {label:<7} : {mean:.3f}  (n={total}, correct={correct})")
    print()

    # ─── Minimal diagnostics: print a few representative incorrect rows ───
    try:
        show = 0 if IS_KAGGLE else 5  # how many rows to show locally
        # Treat "ref" < 1.0 as a mismatch; others must be strictly True
        mismatch_mask = (~bits["val"]) | (~bits["unit"]) | (bits["ref"] < 1.0) | (~bits["na"])
        bad_ids = bits.index[mismatch_mask].tolist()[:show]

        if bad_ids:
            print("Examples of mismatches (expected vs. got):")
            for rid in bad_ids:
                s = sol.loc[rid]
                u = sub.loc[rid]

                reasons = []
                # value reason (mirror logic, but summarized)
                if not is_blank(s["answer_value"]):
                    s_list = parse_listish(s["answer_value"])
                    u_list = parse_listish(u["answer_value"])
                    if s_list is not None:
                        if u_list is not None:
                            if not compare_lists(s_list, u_list, tol=1e-3):
                                reasons.append(f"value list mismatch: expected {s_list}, got {u_list}")
                        else:
                            if len(s_list) == 2 and all(isinstance(v, (int, float)) for v in s_list):
                                lo, hi = sorted(s_list)
                                try:
                                    x = float(u["answer_value"])
                                    if not ((lo - 1e-3) <= x <= (hi + 1e-3)):
                                        reasons.append(f"value out of range: expected [{lo}, {hi}], got {x}")
                                except ValueError:
                                    reasons.append(f"value type mismatch: expected numeric in range, got '{u['answer_value']}'")
                            else:
                                reasons.append("value type mismatch: expected list, got scalar/string")
                else:
                    if not is_blank(u["answer_value"]):
                        reasons.append(f"NA agreement failed: expected blank/NA, got '{u['answer_value']}'")

                # unit reason (still shown even though weight=0)
                if canon_unit(s["answer_unit"]) != canon_unit(u["answer_unit"]):
                    reasons.append(f"unit mismatch: expected '{s['answer_unit']}', got '{u['answer_unit']}'")

                # ref reason (now fractional)
                expected_refs = canon_refs(s["ref_id"])
                got_refs = canon_refs(u["ref_id"])
                if ref_overlap_score(s["ref_id"], u["ref_id"]) < 1.0:
                    reasons.append(f"ref partial/none: expected {expected_refs}, got {got_refs}")

                print(f"- id={rid}")
                print(f"  expected: value={s['answer_value']} | unit={s['answer_unit']} | ref={expected_refs}")
                print(f"  got     : value={u['answer_value']} | unit={u['answer_unit']} | ref={got_refs}")
                for why in reasons:
                    print(f"  why     : {why}")
    except Exception as _e:
        print(f"[warn] diagnostics printing failed: {_e}")

    return float(overall)

# ────────────────────────────────────────────────────── CLI hook ──────
if __name__ == "__main__":
    import sys, os
    from pathlib import Path
    import pandas as pd  # already imported above, but safe

    # Kaggle kernels/validate generally set /kaggle paths or env vars
    IS_KAGGLE = os.path.exists("/kaggle") or "KAGGLE_URL_BASE" in os.environ

    # Strip notebook noise like "-f"
    args = [a for a in sys.argv[1:] if not a.startswith("-")]

    if len(args) == 2:
        # Explicit local run with two files
        sol_path, sub_path = Path(args[0]), Path(args[1])
        print(score(pd.read_csv(sol_path), pd.read_csv(sub_path)))
    elif not IS_KAGGLE:
        # Local convenience: use defaults *only on your machine*
        here = Path(__file__).resolve().parent if "__file__" in globals() else Path(os.getcwd())
        default_sol = here / "solutions_small.csv"
        default_sub = here / "sandbox_submission_small.csv"

        if default_sol.exists() and default_sub.exists():
            print(score(pd.read_csv(default_sol), pd.read_csv(default_sub)))
        else:
            sys.stderr.write(
                "Usage: python Score.py solution.csv submission.csv\n"
                f"(Or place local defaults at {default_sol.name} and {default_sub.name})\n"
            )
    else:
        # In Kaggle Save & Validate: NO-OP.
        # Kaggle will import this module and call score(solution, submission) directly.
        pass
