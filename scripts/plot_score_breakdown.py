#!/usr/bin/env python3
"""
Generate Score Component Breakdown Chart

Shows Value Accuracy, Ref Overlap, and NA Recall side-by-side for each model
to understand WHY certain models perform better.

Usage:
    python scripts/plot_score_breakdown.py --output artifacts/plots/score_breakdown.png
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import numpy as np

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
except ImportError:
    print("matplotlib required: pip install matplotlib")
    sys.exit(1)

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from scripts.score import row_bits, is_blank
from results_io import load_results


def wilson_ci(successes, n, confidence=0.95):
    """Wilson score interval for binomial proportion.

    Returns (lower, upper) bounds for the confidence interval.
    """
    if n == 0:
        return 0.0, 0.0

    try:
        from scipy import stats as sp_stats
        z = sp_stats.norm.ppf(1 - (1 - confidence) / 2)
    except ImportError:
        z = 1.96 if confidence == 0.95 else 1.645

    p = successes / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    spread = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return max(0, center - spread), min(1, center + spread)


def _compute_ci(val_scores, ref_scores, na_gt_scores, val_acc, ref_acc, na_recall, overall):
    """Compute Wilson CIs for score components and propagated overall CI."""
    n_val = len(val_scores)
    n_ref = len(ref_scores)
    n_na = len(na_gt_scores)
    val_ci = wilson_ci(sum(val_scores), n_val) if n_val > 0 else (0, 0)
    if n_ref > 1:
        ref_se_raw = np.std(ref_scores) / np.sqrt(n_ref)
        ref_ci = (max(0, ref_acc - 1.96 * ref_se_raw), min(1, ref_acc + 1.96 * ref_se_raw))
    else:
        ref_ci = (ref_acc, ref_acc)
    na_ci = wilson_ci(sum(na_gt_scores), n_na) if n_na > 0 else (1, 1)

    val_se = (val_ci[1] - val_ci[0]) / (2 * 1.96) if n_val > 0 else 0
    ref_se = (ref_ci[1] - ref_ci[0]) / (2 * 1.96) if n_ref > 0 else 0
    na_se = (na_ci[1] - na_ci[0]) / (2 * 1.96) if n_na > 0 else 0
    overall_se = np.sqrt((0.75 * val_se)**2 + (0.15 * ref_se)**2 + (0.10 * na_se)**2)

    return {
        "val_ci": val_ci,
        "ref_ci": ref_ci,
        "na_ci": na_ci,
        "overall_ci": (overall - 1.96 * overall_se, overall + 1.96 * overall_se),
    }


def _score_from_results_json(results_path: Path):
    """Compute component scores from per-question results (chunked or monolithic).

    This is the preferred source because it contains data from the full
    experiment run (same ground truth the summary was computed against),
    avoiding mismatches when a smaller GT CSV is used for plotting.
    """
    experiment_dir = results_path.parent if results_path.is_file() else results_path
    items = load_results(experiment_dir)

    val_scores = []
    ref_scores = []
    na_gt_scores = []

    for item in items:
        if item.get("error"):
            continue
        val_scores.append(bool(item["value_correct"]))
        ref_scores.append(float(item["ref_score"]))
        gt_val = str(item.get("gt_value", ""))
        if is_blank(gt_val):
            na_gt_scores.append(bool(item["na_correct"]))

    if not val_scores:
        return None

    val_acc = float(np.mean(val_scores))
    ref_acc = float(np.mean(ref_scores))
    na_recall = float(np.mean(na_gt_scores)) if na_gt_scores else 1.0
    overall = 0.75 * val_acc + 0.15 * ref_acc + 0.10 * na_recall

    result = {
        "Value Accuracy": val_acc,
        "Ref Overlap": ref_acc,
        "NA Recall": na_recall,
        "Overall": overall,
    }
    result.update(_compute_ci(val_scores, ref_scores, na_gt_scores,
                              val_acc, ref_acc, na_recall, overall))
    return result


def _score_from_submission(gt_df: pd.DataFrame, sub_path: Path):
    """Fallback: re-score a submission against a GT DataFrame."""
    sub_df = pd.read_csv(sub_path)
    sub_df["id"] = sub_df["id"].astype(str)
    sub_df = sub_df.set_index("id")

    common_ids = gt_df.index.intersection(sub_df.index)

    val_scores = []
    ref_scores = []
    na_gt_scores = []

    for qid in common_ids:
        gt_row = gt_df.loc[qid]
        sub_row = sub_df.loc[qid]

        gt_val = str(gt_row.get("answer_value", ""))
        bits = row_bits(
            sol={
                "answer_value": gt_val,
                "answer_unit": str(gt_row.get("answer_unit", "")),
                "ref_id": str(gt_row.get("ref_id", "")),
            },
            sub={
                "answer_value": str(sub_row.get("answer_value", "")),
                "answer_unit": str(sub_row.get("answer_unit", "")),
                "ref_id": str(sub_row.get("ref_id", "")),
            },
        )
        val_scores.append(bits["val"])
        ref_scores.append(bits["ref"])
        if is_blank(gt_val):
            na_gt_scores.append(bits["na"])

    if not val_scores:
        return None

    val_acc = float(np.mean(val_scores))
    ref_acc = float(np.mean(ref_scores))
    na_recall = float(np.mean(na_gt_scores)) if na_gt_scores else 1.0
    overall = 0.75 * val_acc + 0.15 * ref_acc + 0.10 * na_recall

    result = {
        "Value Accuracy": val_acc,
        "Ref Overlap": ref_acc,
        "NA Recall": na_recall,
        "Overall": overall,
    }
    result.update(_compute_ci(val_scores, ref_scores, na_gt_scores,
                              val_acc, ref_acc, na_recall, overall))
    return result


def load_and_score(gt_path: Path, experiments_dir: Path, datafile: str | None = None):
    """Load and calculate component scores for each model.

    Prefers per-question ``results.json`` from the experiment run so that
    scores are computed over the *same* question set as the original
    experiment.  Falls back to re-scoring ``submission.csv`` against
    ``gt_path`` when ``results.json`` is unavailable.

    When *datafile* is given (e.g. ``"train_QA"``), only experiments
    whose path contains that subfolder are included.
    """
    # Only load GT if we actually need it (fallback path)
    gt_df = None

    results = {}

    # Find all experiment dirs
    all_exp_dirs = sorted(
        p.parent for p in experiments_dir.glob("**/submission.csv")
        if datafile is None or datafile in p.parts
    )

    for exp_dir in all_exp_dirs:
        model_name = exp_dir.name

        # Skip v1 versions if v2 exists (check sibling directory)
        if model_name.endswith("-v1"):
            v2_name = model_name.replace("-v1", "-v2")
            if (exp_dir.parent / v2_name).exists():
                continue

        try:
            scored = _score_from_results_json(exp_dir)
        except FileNotFoundError:
            scored = None
        if scored is None:
            # Lazy-load GT only when needed
            if gt_df is None:
                gt_df = pd.read_csv(gt_path)
                gt_df["id"] = gt_df["id"].astype(str)
                gt_df = gt_df.set_index("id")
            scored = _score_from_submission(gt_df, exp_dir / "submission.csv")

        if scored is not None:
            results[model_name] = scored

    return results


def plot_breakdown(results: dict, output_path: Path):
    """Create grouped bar chart showing score components."""
    # Sort by overall score
    sorted_models = sorted(results.keys(), key=lambda m: results[m]["Overall"], reverse=True)

    # Limit to top 10 for readability
    sorted_models = sorted_models[:10]

    fig, ax = plt.subplots(figsize=(14, 8))

    x = np.arange(len(sorted_models))
    width = 0.22

    val_scores = [results[m]["Value Accuracy"] for m in sorted_models]
    ref_scores = [results[m]["Ref Overlap"] for m in sorted_models]
    na_scores = [results[m]["NA Recall"] for m in sorted_models]
    overall_scores = [results[m]["Overall"] for m in sorted_models]

    # Error bars (distance from score to CI bounds)
    val_yerr = [[s - results[m]["val_ci"][0] for m, s in zip(sorted_models, val_scores)],
                [results[m]["val_ci"][1] - s for m, s in zip(sorted_models, val_scores)]]
    ref_yerr = [[s - results[m]["ref_ci"][0] for m, s in zip(sorted_models, ref_scores)],
                [results[m]["ref_ci"][1] - s for m, s in zip(sorted_models, ref_scores)]]
    na_yerr = [[s - results[m]["na_ci"][0] for m, s in zip(sorted_models, na_scores)],
               [results[m]["na_ci"][1] - s for m, s in zip(sorted_models, na_scores)]]
    overall_yerr = [[s - results[m]["overall_ci"][0] for m, s in zip(sorted_models, overall_scores)],
                    [results[m]["overall_ci"][1] - s for m, s in zip(sorted_models, overall_scores)]]

    # Colors
    colors = {
        "Value": "#2ecc71",      # Green
        "Ref": "#3498db",        # Blue
        "NA": "#9b59b6",         # Purple
        "Overall": "#e74c3c",    # Red
    }

    err_kw = {'linewidth': 1, 'color': '#333'}
    bars1 = ax.bar(x - 1.5*width, val_scores, width, label=f"Value Acc (75%)", color=colors["Value"],
                   yerr=val_yerr, capsize=2, error_kw=err_kw)
    bars2 = ax.bar(x - 0.5*width, ref_scores, width, label=f"Ref Overlap (15%)", color=colors["Ref"],
                   yerr=ref_yerr, capsize=2, error_kw=err_kw)
    bars3 = ax.bar(x + 0.5*width, na_scores, width, label=f"NA Recall (10%)", color=colors["NA"],
                   yerr=na_yerr, capsize=2, error_kw=err_kw)
    bars4 = ax.bar(x + 1.5*width, overall_scores, width, label=f"Overall", color=colors["Overall"], alpha=0.8,
                   yerr=overall_yerr, capsize=2, error_kw=err_kw)

    # Add value labels on bars
    def add_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f"{height:.2f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=7,
                rotation=90,
            )

    add_labels(bars4)  # Only label overall to avoid clutter

    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("WattBot Score Breakdown by Component\n(Weight: 75% Value + 15% Ref + 10% NA | 95% Wilson CI)", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(sorted_models, rotation=45, ha="right", fontsize=10)
    ax.legend(loc="upper right")
    ax.set_ylim(0, 1.15)
    ax.grid(axis="y", alpha=0.3)

    # Add note about NA questions
    ax.text(
        0.02, 0.98,
        "NA = Unanswerable questions (GT answer_value='is_blank')",
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="top",
        style="italic",
        color="gray",
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate score breakdown chart")
    parser.add_argument(
        "--ground-truth", "-g",
        default=None,
        help="Path to ground truth CSV (auto-detects test_solutions.csv or train_QA.csv)",
    )
    parser.add_argument(
        "--experiments", "-e",
        default="artifacts/experiments",
        help="Path to experiments directory",
    )
    parser.add_argument(
        "--datafile", "-d",
        default=None,
        help="Filter to experiments from this datafile subfolder "
             "(e.g. 'train_QA', 'test_solutions'). Default: include all.",
    )
    parser.add_argument(
        "--output", "-o",
        default="artifacts/plots/score_breakdown.png",
        help="Output path for chart",
    )

    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    experiments_dir = project_root / args.experiments
    output_path = project_root / args.output
    if args.datafile:
        output_path = output_path.parent / args.datafile / output_path.name

    # Auto-detect ground truth: prefer test_solutions.csv, fall back to train_QA.csv
    if args.ground_truth:
        gt_path = project_root / args.ground_truth
    else:
        gt_path = project_root / "data" / "test_solutions.csv"
        if not gt_path.exists():
            gt_path = project_root / "data" / "train_QA.csv"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.datafile:
        print(f"Filtering to datafile: {args.datafile}")
    print("Loading and scoring experiments...")
    results = load_and_score(gt_path, experiments_dir, datafile=args.datafile)

    print(f"\nResults ({len(results)} models):")
    print("-" * 70)
    print(f"{'Model':<25} {'Value':>8} {'Ref':>8} {'NA':>8} {'Overall':>8}")
    print("-" * 70)
    for model in sorted(results.keys(), key=lambda m: results[m]["Overall"], reverse=True):
        r = results[model]
        print(f"{model:<25} {r['Value Accuracy']:>8.3f} {r['Ref Overlap']:>8.3f} {r['NA Recall']:>8.3f} {r['Overall']:>8.3f}")

    print("\nGenerating breakdown chart...")
    plot_breakdown(results, output_path)


if __name__ == "__main__":
    main()
