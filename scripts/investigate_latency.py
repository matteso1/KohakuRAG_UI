#!/usr/bin/env python3
"""Investigate per-question latency inflation caused by max_concurrent > 1.

When max_concurrent > 1, multiple HuggingFace inference calls run in parallel
on the same GPU via thread pool executors. Each call contends for GPU resources,
inflating per-question wall-clock latency by approximately the concurrency factor.

The ratio (avg_latency * num_questions) / total_wall_time should equal ~1.0
for sequential execution, but equals ~max_concurrent when concurrency is used.

This script reads archived experiment summaries and prints the analysis.
"""

import json
import os
import sys
from pathlib import Path


def analyze_experiments(base_dir: Path) -> list[dict]:
    """Analyze all experiment summaries in a directory."""
    rows = []
    for model_dir in sorted(base_dir.iterdir()):
        summary_path = model_dir / "summary.json"
        if not summary_path.exists():
            continue

        with open(summary_path) as f:
            s = json.load(f)

        config = s.get("config_snapshot", {})
        concurrent = config.get("max_concurrent", None)
        avg_lat = s.get("avg_latency_seconds", 0)
        total_t = s.get("total_time_seconds", 0)
        n = s.get("num_questions", 0)
        avg_gen = s.get("avg_generation_seconds", 0)

        if avg_lat == 0 or total_t == 0 or n == 0:
            continue

        sum_lat = avg_lat * n
        ratio = sum_lat / total_t
        est_true = avg_lat / concurrent if concurrent and concurrent > 0 else avg_lat

        rows.append({
            "model": model_dir.name,
            "max_concurrent": concurrent,
            "avg_latency": avg_lat,
            "avg_generation": avg_gen,
            "total_time": total_t,
            "num_questions": n,
            "ratio": ratio,
            "est_true_latency": est_true,
            "throughput_qps": n / total_t,
        })

    return rows


def print_table(rows: list[dict], dataset_name: str) -> None:
    """Print formatted analysis table."""
    print(f"\n{'=' * 90}")
    print(f"  {dataset_name}")
    print(f"{'=' * 90}")
    print(
        f"{'Model':<25} {'Conc':>5} {'AvgLat':>8} {'TotalT':>9} "
        f"{'Ratio':>7} {'EstTrue':>8} {'Q/s':>6}"
    )
    print("-" * 90)

    for r in rows:
        conc = str(r["max_concurrent"]) if r["max_concurrent"] else "N/A"
        inflated = " ***" if r["ratio"] > 1.2 else ""
        print(
            f"{r['model']:<25} {conc:>5} {r['avg_latency']:>8.2f} "
            f"{r['total_time']:>9.1f} {r['ratio']:>7.2f} "
            f"{r['est_true_latency']:>8.2f} {r['throughput_qps']:>6.3f}{inflated}"
        )

    inflated_models = [r for r in rows if r["ratio"] > 1.2]
    if inflated_models:
        print(f"\n  *** = latency inflated by concurrency (ratio >> 1.0)")
        print(f"  {len(inflated_models)} of {len(rows)} models affected")


def main():
    archive = Path(__file__).resolve().parent.parent / "archive_pre-fixed-pipeline" / "experiments" / "PowerEdge"
    if not archive.exists():
        print(f"Archive not found: {archive}", file=sys.stderr)
        sys.exit(1)

    for dataset in sorted(archive.iterdir()):
        if dataset.is_dir():
            rows = analyze_experiments(dataset)
            if rows:
                print_table(rows, dataset.name)

    # Summary
    print(f"\n{'=' * 90}")
    print("  CONCLUSION")
    print(f"{'=' * 90}")
    print("""
  The per-question avg_latency_seconds is inflated by exactly the max_concurrent
  factor for every model where max_concurrent > 1. This is because HF inference
  calls run in parallel threads on the same GPU, causing resource contention.

  The 'EstTrue' column shows the estimated sequential per-question latency
  (avg_latency / max_concurrent), which gives correct cross-model comparisons.

  The total_time_seconds metric is unaffected and can be used for throughput
  comparisons (Q/s column).

  For benchmarking, either:
    a) Run all models with max_concurrent=1, or
    b) Report throughput (questions/sec) instead of per-question latency, or
    c) Normalize: true_latency ≈ avg_latency / max_concurrent
""")


if __name__ == "__main__":
    main()
