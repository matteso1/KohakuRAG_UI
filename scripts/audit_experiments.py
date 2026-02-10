#!/usr/bin/env python3
"""
Audit all experiment runs for data quality issues.

Checks:
  1. Missing or zero token counts (cost estimates will be wrong)
  2. Suspiciously high latency (>60s avg = probably timeout/retry issues)
  3. Zero cost estimates (pricing lookup failed)
  4. High error rates
  5. Very low scores (likely broken pipeline run)
  6. Duplicate models (multiple runs, which is best?)
  7. Score consistency (recalculate from results.json vs summary.json)
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from score import row_bits


def audit():
    experiments_dir = Path(__file__).parent.parent / "artifacts" / "experiments"
    gt_path = Path(__file__).parent.parent / "data" / "train_QA.csv"

    if not experiments_dir.exists():
        print("ERROR: No experiments directory found")
        return

    # Load ground truth for score verification
    gt = None
    if gt_path.exists():
        import pandas as pd
        gt = pd.read_csv(gt_path)
        print(f"Ground truth: {len(gt)} questions from {gt_path.name}")
    else:
        print(f"WARNING: Ground truth not found at {gt_path}")

    print(f"\n{'='*110}")
    print(f"{'Experiment':<25s} {'Model ID':<45s} {'Score':>6s} {'Val':>5s} {'Ref':>5s} {'Lat':>8s} {'InTok':>8s} {'OutTok':>7s} {'Cost':>8s} {'Err':>4s} {'FLAGS'}")
    print(f"{'-'*110}")

    issues = []
    clean_experiments = []

    # Find all experiment dirs (supports both flat and <env>/<name>/ layouts)
    exp_dirs = sorted(
        p.parent for p in experiments_dir.glob("**/summary.json")
    )

    for exp_dir in exp_dirs:
        summary_path = exp_dir / "summary.json"
        results_path = exp_dir / "results.json"

        with open(summary_path) as f:
            s = json.load(f)

        name = s.get("name", exp_dir.name)
        model_id = s.get("model_id", "?")
        score = s.get("overall_score", 0)
        val_acc = s.get("value_accuracy", 0)
        ref_ovl = s.get("ref_overlap", 0)
        na_acc = s.get("na_accuracy", 0)
        latency = s.get("avg_latency_seconds", 0)
        in_tok = s.get("input_tokens", 0)
        out_tok = s.get("output_tokens", 0)
        cost = s.get("estimated_cost_usd", 0)
        errors = s.get("error_count", 0)
        n_questions = s.get("num_questions", 0)

        flags = []

        # Check 1: Missing token counts
        if in_tok == 0 and out_tok == 0:
            flags.append("NO_TOKENS")

        # Check 2: Suspicious latency
        if latency > 60:
            flags.append(f"HIGH_LAT({latency:.0f}s)")
        if latency == 0:
            flags.append("ZERO_LAT")

        # Check 3: Zero cost
        if cost == 0 and in_tok > 0:
            flags.append("ZERO_COST")
        if cost == 0 and in_tok == 0:
            flags.append("NO_COST_DATA")

        # Check 4: High error rate
        if n_questions > 0 and errors > 0:
            error_rate = errors / n_questions
            if error_rate > 0.1:
                flags.append(f"ERRORS({errors}/{n_questions})")

        # Check 5: Very low score
        if score < 0.2:
            flags.append("BROKEN_RUN")
        elif score < 0.4:
            flags.append("LOW_SCORE")

        # Check 6: Ensemble (no single model)
        if not model_id or model_id == "?":
            flags.append("NO_MODEL")

        # Check 7: Verify score from results.json
        if results_path.exists() and gt is not None:
            try:
                with open(results_path) as f:
                    results = json.load(f)
                # Recalculate score
                recalc_val = sum(1 for r in results if r.get("value_correct")) / len(results) if results else 0
                if abs(recalc_val - val_acc) > 0.01:
                    flags.append(f"SCORE_MISMATCH(recalc={recalc_val:.3f})")
            except Exception:
                flags.append("BAD_RESULTS_JSON")

        flag_str = ", ".join(flags) if flags else "OK"
        if flags:
            issues.append((name, flags))

        # Determine if this is a clean experiment
        is_clean = (
            model_id and model_id != "?"
            and score >= 0.2
            and latency > 0
            and latency < 60
            and (errors == 0 or (n_questions > 0 and errors / n_questions < 0.1))
        )

        marker = "  " if is_clean else "! "

        print(
            f"{marker}{name:<23s} "
            f"{model_id:<45s} "
            f"{score:>5.3f} "
            f"{val_acc:>5.3f} "
            f"{ref_ovl:>5.3f} "
            f"{latency:>7.2f}s "
            f"{in_tok:>8,} "
            f"{out_tok:>7,} "
            f"${cost:>6.4f} "
            f"{errors:>4d} "
            f"{flag_str}"
        )

        if is_clean:
            clean_experiments.append({
                "name": name,
                "model_id": model_id,
                "score": score,
                "val_acc": val_acc,
                "ref_ovl": ref_ovl,
                "latency": latency,
                "cost": cost,
                "in_tok": in_tok,
                "out_tok": out_tok,
            })

    print(f"{'='*110}")
    print(f"\nTotal experiments: {len(exp_dirs)}")
    print(f"Clean experiments: {len(clean_experiments)}")
    print(f"Experiments with issues: {len(issues)}")

    if issues:
        print(f"\n--- ISSUES FOUND ---")
        for name, flags in issues:
            print(f"  {name}: {', '.join(flags)}")

    # Check for duplicate models in clean set
    print(f"\n--- DUPLICATE MODEL CHECK ---")
    model_runs = {}
    for exp in clean_experiments:
        mid = exp["model_id"]
        if mid not in model_runs:
            model_runs[mid] = []
        model_runs[mid].append(exp)

    for mid, runs in model_runs.items():
        if len(runs) > 1:
            print(f"  {mid}: {len(runs)} runs")
            for r in runs:
                print(f"    {r['name']}: score={r['score']:.3f}, latency={r['latency']:.2f}s")
            best = max(runs, key=lambda x: x["score"])
            print(f"    -> Best: {best['name']} ({best['score']:.3f})")

    # Summary of clean data
    print(f"\n--- CLEAN EXPERIMENTS (for plotting) ---")
    seen = set()
    for exp in sorted(clean_experiments, key=lambda x: x["score"], reverse=True):
        if exp["model_id"] in seen:
            continue
        seen.add(exp["model_id"])
        tok_status = "tokens OK" if exp["in_tok"] > 0 else "NO TOKENS"
        cost_status = f"${exp['cost']:.4f}" if exp["cost"] > 0 else "NO COST"
        print(f"  {exp['name']:<25s} score={exp['score']:.3f}  lat={exp['latency']:.2f}s  {tok_status}  {cost_status}")


if __name__ == "__main__":
    audit()
