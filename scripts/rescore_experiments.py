#!/usr/bin/env python3
"""
Re-score existing experiments with updated NA metric.

Reads results.json for each experiment and recalculates scores using:
- NA Recall: of all truly NA questions, how many did the model correctly abstain on?
  (Previously: na_correct / total_questions, which inflated to ~99%)

Usage:
    python scripts/rescore_experiments.py --filter test
    python scripts/rescore_experiments.py --filter bench
    python scripts/rescore_experiments.py  # all experiments
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from score import is_blank


def rescore(experiments_dir: Path, name_filter: str | None = None) -> None:
    """Re-score all experiments with updated NA recall metric."""

    for exp_dir in sorted(experiments_dir.iterdir()):
        if not exp_dir.is_dir():
            continue

        summary_path = exp_dir / "summary.json"
        results_path = exp_dir / "results.json"

        if not summary_path.exists() or not results_path.exists():
            continue

        name = exp_dir.name
        if name_filter and name_filter not in name:
            continue

        summary = json.loads(summary_path.read_text())
        results = json.loads(results_path.read_text())

        total = len(results)
        if total == 0:
            continue

        # Recalculate value accuracy
        value_acc = sum(1 for r in results if r.get("value_correct")) / total

        # Recalculate ref overlap
        ref_overlap = sum(r.get("ref_score", 0) for r in results) / total

        # NEW: NA Recall (of truly NA questions, how many correctly abstained)
        truly_na = [r for r in results if is_blank(r.get("gt_value", ""))]
        if truly_na:
            na_recall = sum(1 for r in truly_na if r.get("na_correct")) / len(truly_na)
        else:
            na_recall = 1.0

        # Recalculate overall score
        overall = 0.75 * value_acc + 0.15 * ref_overlap + 0.10 * na_recall

        # Store old values for comparison
        old_na = summary.get("na_accuracy", 0)
        old_overall = summary.get("overall_score", 0)

        # Update summary
        summary["value_accuracy"] = value_acc
        summary["ref_overlap"] = ref_overlap
        summary["na_accuracy"] = na_recall
        summary["overall_score"] = overall
        summary["na_total_truly_na"] = len(truly_na)
        summary["na_correctly_abstained"] = sum(1 for r in truly_na if r.get("na_correct"))

        # Write back
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        delta = overall - old_overall
        delta_str = f"({delta:+.3f})" if abs(delta) > 0.001 else "(unchanged)"
        print(
            f"  {name:<30s}  "
            f"NA: {old_na:.3f} -> {na_recall:.3f} ({len(truly_na)} NA qs)  "
            f"Overall: {old_overall:.3f} -> {overall:.3f} {delta_str}"
        )


def main():
    parser = argparse.ArgumentParser(description="Re-score experiments with NA recall")
    parser.add_argument("--filter", "-f", default=None, help="Filter experiments by name")
    parser.add_argument(
        "--experiments", "-e",
        default="artifacts/experiments",
        help="Experiments directory",
    )
    args = parser.parse_args()

    experiments_dir = Path(args.experiments)
    if not experiments_dir.exists():
        print(f"Error: {experiments_dir} not found")
        sys.exit(1)

    print(f"Re-scoring experiments with NA recall metric...")
    if args.filter:
        print(f"  Filter: '{args.filter}'")
    print()

    rescore(experiments_dir, args.filter)
    print("\nDone! Summary.json files updated in-place.")


if __name__ == "__main__":
    main()
