#!/usr/bin/env python3
"""
Qwen Model Size Scaling Experiment

Runs the WattBot RAG pipeline with Qwen 2.5 models at different sizes
(1.5B, 3B, 7B, 14B, 32B) to measure how performance scales with model
size. Collects hardware metrics (VRAM, energy, disk) at each point.

Each model is run sequentially (GPU memory is freed between runs) to get
clean measurements. Results are saved per-model AND as a combined
comparison CSV for easy analysis.

Usage:
    # Run all Qwen sizes that fit on your GPU
    python scripts/run_qwen_scaling.py

    # Specify which sizes to run
    python scripts/run_qwen_scaling.py --sizes 1.5 3 7

    # Skip models already run (resume mode)
    python scripts/run_qwen_scaling.py --skip-existing

    # Dry run (just show what would run)
    python scripts/run_qwen_scaling.py --dry-run

Output:
    artifacts/experiments/qwen-scaling/
    ├── qwen1.5b/             # Per-model experiment dirs
    │   ├── submission.csv
    │   ├── results.json
    │   └── summary.json      # Includes hardware metrics
    ├── qwen3b/
    ├── qwen7b/
    ├── qwen14b/
    ├── qwen32b/
    └── scaling_comparison.csv  # Combined comparison table
"""

import argparse
import csv
import gc
import json
import subprocess
import sys
import time
from pathlib import Path

# =============================================================================
# Qwen Model Registry
# =============================================================================

QWEN_MODELS = {
    "1.5": {
        "config": "vendor/KohakuRAG/configs/hf_qwen1_5b.py",
        "name": "qwen1.5b",
        "model_id": "Qwen/Qwen2.5-1.5B-Instruct",
        "params_b": 1.5,
        "approx_vram_gb": 4,
    },
    "3": {
        "config": "vendor/KohakuRAG/configs/hf_qwen3b.py",
        "name": "qwen3b",
        "model_id": "Qwen/Qwen2.5-3B-Instruct",
        "params_b": 3,
        "approx_vram_gb": 8,
    },
    "7": {
        "config": "vendor/KohakuRAG/configs/hf_qwen7b.py",
        "name": "qwen7b",
        "model_id": "Qwen/Qwen2.5-7B-Instruct",
        "params_b": 7,
        "approx_vram_gb": 16,
    },
    "14": {
        "config": "vendor/KohakuRAG/configs/hf_qwen14b.py",
        "name": "qwen14b",
        "model_id": "Qwen/Qwen2.5-14B-Instruct",
        "params_b": 14,
        "approx_vram_gb": 30,
    },
    "32": {
        "config": "vendor/KohakuRAG/configs/hf_qwen32b.py",
        "name": "qwen32b",
        "model_id": "Qwen/Qwen2.5-32B-Instruct",
        "params_b": 32,
        "approx_vram_gb": 65,
    },
    "72": {
        "config": "vendor/KohakuRAG/configs/hf_qwen72b.py",
        "name": "qwen72b",
        "model_id": "Qwen/Qwen2.5-72B-Instruct",
        "params_b": 72,
        "approx_vram_gb": 140,
    },
    "72-4bit": {
        "config": "vendor/KohakuRAG/configs/hf_qwen72b_4bit.py",
        "name": "qwen72b-4bit",
        "model_id": "Qwen/Qwen2.5-72B-Instruct (4-bit)",
        "params_b": 72,
        "approx_vram_gb": 40,
    },
}


def _sort_key(size_key: str) -> tuple[float, str]:
    """Sort key for model size keys: numeric part first, then suffix."""
    parts = size_key.split("-", 1)
    return (float(parts[0]), parts[1] if len(parts) > 1 else "")


def get_available_vram_gb() -> float:
    """Query nvidia-smi for available VRAM (free memory on GPU 0)."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--id=0", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            return float(result.stdout.strip()) / 1024  # MiB -> GiB
    except (subprocess.TimeoutExpired, ValueError, FileNotFoundError):
        pass
    return 0.0


def run_single_model(config_path: str, experiment_name: str, env: str = "") -> tuple[bool, dict | None]:
    """Run a single experiment via subprocess, return (success, summary_dict)."""
    output_dir = Path("artifacts/experiments/qwen-scaling") / experiment_name

    cmd = [
        sys.executable, "scripts/run_experiment.py",
        "--config", config_path,
        "--name", f"qwen-scaling/{experiment_name}",
    ]
    if env:
        cmd.extend(["--env", env])

    print(f"\n{'='*70}")
    print(f"Running: {experiment_name} ({config_path})")
    print(f"{'='*70}")

    start = time.time()
    try:
        result = subprocess.run(
            cmd,
            timeout=3600,  # 1 hour timeout
            text=True,
        )
        elapsed = time.time() - start
        success = result.returncode == 0

        if success:
            summary_path = output_dir / "summary.json"
            if summary_path.exists():
                with open(summary_path) as f:
                    summary = json.load(f)
                print(f"\n[OK] {experiment_name} completed in {elapsed:.0f}s (score: {summary.get('overall_score', 0):.3f})")
                return True, summary
            else:
                print(f"\n[WARN] {experiment_name} completed but no summary.json found")
                return True, None
        else:
            print(f"\n[FAIL] {experiment_name} failed after {elapsed:.0f}s")
            return False, None

    except subprocess.TimeoutExpired:
        print(f"\n[TIMEOUT] {experiment_name} exceeded 1 hour")
        return False, None


def generate_comparison_csv(summaries: dict, output_path: Path):
    """Generate a CSV comparing all model sizes."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "model", "params_b", "quantization", "run_environment", "hostname",
        "overall_score", "value_accuracy", "ref_overlap", "na_accuracy",
        "avg_latency_s", "avg_retrieval_s", "avg_generation_s", "total_time_s",
        "vram_allocated_gb", "vram_reserved_gb", "vram_total_gb",
        "gpu_name", "gpu_count",
        "model_disk_gb", "model_load_time_s", "llm_load_time_s", "embedder_load_time_s",
        "energy_wh", "energy_method", "avg_power_w", "peak_power_w",
        "cpu_rss_peak_gb",
        "questions", "correct", "wrong", "errors",
        "input_tokens", "output_tokens", "est_cost_usd",
    ]

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for size_key, summary in sorted(summaries.items(), key=lambda x: _sort_key(x[0])):
            if summary is None:
                continue

            hw = summary.get("hardware", {})
            model_info = QWEN_MODELS.get(size_key, {})

            writer.writerow({
                "model": summary.get("model_id", model_info.get("model_id", "")),
                "params_b": model_info.get("params_b", ""),
                "quantization": summary.get("quantization", summary.get("config_snapshot", {}).get("hf_dtype", "bf16")),
                "run_environment": summary.get("run_environment", ""),
                "hostname": hw.get("hostname", ""),
                "overall_score": f"{summary.get('overall_score', 0):.4f}",
                "value_accuracy": f"{summary.get('value_accuracy', 0):.4f}",
                "ref_overlap": f"{summary.get('ref_overlap', 0):.4f}",
                "na_accuracy": f"{summary.get('na_accuracy', 0):.4f}",
                "avg_latency_s": f"{summary.get('avg_latency_seconds', 0):.2f}",
                "avg_retrieval_s": f"{summary.get('avg_retrieval_seconds', 0):.2f}",
                "avg_generation_s": f"{summary.get('avg_generation_seconds', 0):.2f}",
                "total_time_s": f"{summary.get('total_time_seconds', 0):.1f}",
                "vram_allocated_gb": f"{hw.get('gpu_vram_allocated_gb', 0):.2f}",
                "vram_reserved_gb": f"{hw.get('gpu_vram_reserved_gb', 0):.2f}",
                "vram_total_gb": f"{hw.get('gpu_vram_total_gb', 0):.1f}",
                "gpu_name": hw.get("gpu_name", ""),
                "gpu_count": hw.get("gpu_count", 0),
                "model_disk_gb": f"{hw.get('model_disk_size_gb', 0):.2f}",
                "model_load_time_s": f"{hw.get('model_load_time_seconds', 0):.1f}",
                "llm_load_time_s": f"{hw.get('llm_load_time_seconds', 0):.1f}",
                "embedder_load_time_s": f"{hw.get('embedder_load_time_seconds', 0):.1f}",
                "energy_wh": f"{hw.get('gpu_energy_wh', 0):.3f}",
                "energy_method": hw.get("gpu_energy_method", ""),
                "avg_power_w": f"{hw.get('gpu_avg_power_watts', 0):.1f}",
                "peak_power_w": f"{hw.get('gpu_peak_power_watts', 0):.1f}",
                "cpu_rss_peak_gb": f"{hw.get('cpu_rss_peak_gb', 0):.2f}",
                "questions": summary.get("num_questions", 0),
                "correct": summary.get("questions_correct", 0),
                "wrong": summary.get("questions_wrong", 0),
                "errors": summary.get("error_count", 0),
                "input_tokens": summary.get("input_tokens", 0),
                "output_tokens": summary.get("output_tokens", 0),
                "est_cost_usd": f"{summary.get('estimated_cost_usd', 0):.4f}",
            })

    print(f"\nComparison CSV saved to: {output_path}")


def print_comparison_table(summaries: dict):
    """Print a formatted comparison table."""
    print(f"\n{'='*100}")
    print("QWEN MODEL SIZE SCALING COMPARISON")
    print(f"{'='*100}")
    print(f"{'Model':<30s} {'Params':>7s} {'Score':>7s} {'Value':>7s} {'VRAM':>8s} {'Disk':>7s} {'Energy':>8s} {'Latency':>9s}")
    print(f"{'-'*100}")

    for size_key in sorted(summaries.keys(), key=_sort_key):
        s = summaries[size_key]
        if s is None:
            model_name = QWEN_MODELS.get(size_key, {}).get("model_id", f"Qwen {size_key}B")
            print(f"{model_name:<30s} {'FAILED':>7s}")
            continue

        hw = s.get("hardware", {})
        model_info = QWEN_MODELS.get(size_key, {})
        print(
            f"{s.get('model_id', ''):<30s} "
            f"{model_info.get('params_b', '?'):>6}B "
            f"{s.get('overall_score', 0):>6.3f} "
            f"{s.get('value_accuracy', 0):>6.3f} "
            f"{hw.get('gpu_vram_allocated_gb', 0):>6.1f}GB "
            f"{hw.get('model_disk_size_gb', 0):>5.1f}GB "
            f"{hw.get('gpu_energy_wh', 0):>6.3f}Wh "
            f"{s.get('avg_latency_seconds', 0):>7.1f}s "
        )

    print(f"{'='*100}")


def main():
    parser = argparse.ArgumentParser(
        description="Run Qwen model size scaling experiment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--sizes", nargs="+", default=None,
        help="Which sizes to run (e.g., 1.5 3 7). Default: all that fit in VRAM.",
    )
    parser.add_argument(
        "--skip-existing", action="store_true",
        help="Skip models that already have a summary.json",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would run without actually running",
    )
    parser.add_argument(
        "--force-all", action="store_true",
        help="Run all sizes even if they might not fit in VRAM",
    )
    parser.add_argument(
        "--env", "-e", default="",
        help="Run environment label (e.g. 'GB10', 'PowerEdge') for cross-machine comparison",
    )

    args = parser.parse_args()

    # Determine which sizes to run
    if args.sizes:
        sizes_to_run = [s for s in args.sizes if s in QWEN_MODELS]
        if not sizes_to_run:
            print(f"No valid sizes. Choose from: {list(QWEN_MODELS.keys())}")
            return
    else:
        sizes_to_run = list(QWEN_MODELS.keys())

    # Check available VRAM
    available_vram = get_available_vram_gb()
    if available_vram > 0 and not args.force_all:
        print(f"Available VRAM: ~{available_vram:.1f} GB")
        # Filter to models that likely fit (with some headroom)
        original = len(sizes_to_run)
        sizes_to_run = [
            s for s in sizes_to_run
            if QWEN_MODELS[s]["approx_vram_gb"] <= available_vram * 0.95
        ]
        if len(sizes_to_run) < original:
            skipped = original - len(sizes_to_run)
            print(f"Skipping {skipped} model(s) that likely won't fit. Use --force-all to override.")

    # Check for existing results
    scaling_dir = Path("artifacts/experiments/qwen-scaling")
    if args.skip_existing:
        original = len(sizes_to_run)
        sizes_to_run = [
            s for s in sizes_to_run
            if not (scaling_dir / QWEN_MODELS[s]["name"] / "summary.json").exists()
        ]
        skipped = original - len(sizes_to_run)
        if skipped:
            print(f"Skipping {skipped} model(s) with existing results.")

    # Check configs exist
    missing = [s for s in sizes_to_run if not Path(QWEN_MODELS[s]["config"]).exists()]
    if missing:
        print(f"Missing config files for sizes: {missing}")
        sizes_to_run = [s for s in sizes_to_run if s not in missing]

    if not sizes_to_run:
        print("No models to run!")
        return

    print(f"\nModels to run: {[QWEN_MODELS[s]['name'] for s in sizes_to_run]}")
    print(f"Results will be saved to: {scaling_dir}/")

    if args.dry_run:
        print("\n[DRY RUN] Would run the above models. Exiting.")
        return

    # Run each model sequentially
    summaries = {}
    total_start = time.time()

    for size_key in sorted(sizes_to_run, key=_sort_key):
        model_info = QWEN_MODELS[size_key]
        success, summary = run_single_model(model_info["config"], model_info["name"], env=args.env)
        summaries[size_key] = summary

        # Force garbage collection between models to free VRAM
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except (ImportError, RuntimeError):
            pass

    total_elapsed = time.time() - total_start

    # Load any existing results we skipped
    if args.skip_existing:
        for size_key, model_info in QWEN_MODELS.items():
            if size_key not in summaries:
                summary_path = scaling_dir / model_info["name"] / "summary.json"
                if summary_path.exists():
                    with open(summary_path) as f:
                        summaries[size_key] = json.load(f)

    # Generate comparison outputs
    comparison_csv = scaling_dir / "scaling_comparison.csv"
    generate_comparison_csv(summaries, comparison_csv)

    # Also save as JSON for programmatic access
    comparison_json = scaling_dir / "scaling_comparison.json"
    with open(comparison_json, "w") as f:
        json.dump(summaries, f, indent=2)
    print(f"Comparison JSON saved to: {comparison_json}")

    # Print table
    print_comparison_table(summaries)

    passed = sum(1 for s in summaries.values() if s is not None)
    failed = sum(1 for s in summaries.values() if s is None)
    print(f"\nTotal time: {total_elapsed:.0f}s ({total_elapsed/60:.1f} min)")
    print(f"Passed: {passed}, Failed: {failed}")

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
