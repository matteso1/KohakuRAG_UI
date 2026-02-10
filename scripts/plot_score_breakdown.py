#!/usr/bin/env python3
"""
Generate Score Component Breakdown Chart

Shows Value Accuracy, Ref Overlap, and NA Accuracy side-by-side for each model
to understand WHY certain models perform better.

Usage:
    python scripts/plot_score_breakdown.py --output artifacts/plots/score_breakdown.png
"""

import argparse
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
from scripts.score import row_bits, is_blank


def load_and_score(gt_path: Path, experiments_dir: Path):
    """Load ground truth and calculate component scores for each model."""
    gt_df = pd.read_csv(gt_path)
    gt_df["id"] = gt_df["id"].astype(str)
    gt_df = gt_df.set_index("id")

    results = {}

    for exp_dir in experiments_dir.iterdir():
        if not exp_dir.is_dir():
            continue

        sub_path = exp_dir / "submission.csv"
        if not sub_path.exists():
            continue

        model_name = exp_dir.name

        # Skip v1 versions if v2 exists
        if model_name.endswith("-v1"):
            v2_name = model_name.replace("-v1", "-v2")
            if (experiments_dir / v2_name).exists():
                continue

        sub_df = pd.read_csv(sub_path)
        sub_df["id"] = sub_df["id"].astype(str)
        sub_df = sub_df.set_index("id")

        common_ids = gt_df.index.intersection(sub_df.index)

        val_scores = []
        ref_scores = []
        na_scores = []

        for qid in common_ids:
            gt_row = gt_df.loc[qid]
            sub_row = sub_df.loc[qid]

            bits = row_bits(
                sol={
                    "answer_value": str(gt_row.get("answer_value", "")),
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
            na_scores.append(bits["na"])

        val_acc = np.mean(val_scores)
        ref_acc = np.mean(ref_scores)
        na_acc = np.mean(na_scores)
        overall = 0.75 * val_acc + 0.15 * ref_acc + 0.10 * na_acc

        results[model_name] = {
            "Value Accuracy": val_acc,
            "Ref Overlap": ref_acc,
            "NA Accuracy": na_acc,
            "Overall": overall,
        }

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
    na_scores = [results[m]["NA Accuracy"] for m in sorted_models]
    overall_scores = [results[m]["Overall"] for m in sorted_models]

    # Colors
    colors = {
        "Value": "#2ecc71",      # Green
        "Ref": "#3498db",        # Blue
        "NA": "#9b59b6",         # Purple
        "Overall": "#e74c3c",    # Red
    }

    bars1 = ax.bar(x - 1.5*width, val_scores, width, label=f"Value Acc (75%)", color=colors["Value"])
    bars2 = ax.bar(x - 0.5*width, ref_scores, width, label=f"Ref Overlap (15%)", color=colors["Ref"])
    bars3 = ax.bar(x + 0.5*width, na_scores, width, label=f"NA Acc (10%)", color=colors["NA"])
    bars4 = ax.bar(x + 1.5*width, overall_scores, width, label=f"Overall", color=colors["Overall"], alpha=0.8)

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
    ax.set_title("WattBot Score Breakdown by Component\n(Weight: 75% Value + 15% Ref + 10% NA)", fontsize=14)
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
        default="data/train_QA.csv",
        help="Path to ground truth CSV",
    )
    parser.add_argument(
        "--experiments", "-e",
        default="artifacts/experiments",
        help="Path to experiments directory",
    )
    parser.add_argument(
        "--output", "-o",
        default="artifacts/plots/score_breakdown.png",
        help="Output path for chart",
    )

    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    gt_path = project_root / args.ground_truth
    experiments_dir = project_root / args.experiments
    output_path = project_root / args.output

    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("Loading and scoring experiments...")
    results = load_and_score(gt_path, experiments_dir)

    print(f"\nResults ({len(results)} models):")
    print("-" * 70)
    print(f"{'Model':<25} {'Value':>8} {'Ref':>8} {'NA':>8} {'Overall':>8}")
    print("-" * 70)
    for model in sorted(results.keys(), key=lambda m: results[m]["Overall"], reverse=True):
        r = results[model]
        print(f"{model:<25} {r['Value Accuracy']:>8.3f} {r['Ref Overlap']:>8.3f} {r['NA Accuracy']:>8.3f} {r['Overall']:>8.3f}")

    print("\nGenerating breakdown chart...")
    plot_breakdown(results, output_path)


if __name__ == "__main__":
    main()
