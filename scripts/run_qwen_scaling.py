#!/usr/bin/env python3
"""
Qwen Model Size Scaling Experiment

Collects existing Qwen experiment results and generates a scaling comparison
table. Optionally runs missing sizes that haven't been benchmarked yet.

Results are read from the standard experiment directory layout
(artifacts/experiments/ or artifacts/experiments/<env>/) — the same results
produced by run_experiment.py and run_full_benchmark.py. No duplicate
experiment directory is created.

Usage:
    # Collect existing results and produce comparison table
    python scripts/run_qwen_scaling.py --env PowerEdge

    # Also run any missing sizes that fit in VRAM
    python scripts/run_qwen_scaling.py --env PowerEdge --run-missing

    # Specify which sizes to include
    python scripts/run_qwen_scaling.py --sizes 1.5 3 7 --env PowerEdge

    # Dry run (show what would run)
    python scripts/run_qwen_scaling.py --run-missing --dry-run --env PowerEdge

Output:
    artifacts/experiments/<env>/qwen_scaling_comparison.csv
    artifacts/experiments/<env>/qwen_scaling_comparison.json
"""

import argparse
import csv
import gc
import json
import subprocess
import sys
import time
from pathlib import Path

EXPERIMENTS_BASE = Path("artifacts/experiments")

# =============================================================================
# Qwen Model Registry
# =============================================================================

QWEN_MODELS = {
    "1.5": {
        "config": "vendor/KohakuRAG/configs/hf_qwen1_5b.py",
        "bench_name": "qwen1.5b-bench",
        "model_id": "Qwen/Qwen2.5-1.5B-Instruct",
        "params_b": 1.5,
        "approx_vram_gb": 2,
    },
    "3": {
        "config": "vendor/KohakuRAG/configs/hf_qwen3b.py",
        "bench_name": "qwen3b-bench",
        "model_id": "Qwen/Qwen2.5-3B-Instruct",
        "params_b": 3,
        "approx_vram_gb": 3,
    },
    "7": {
        "config": "vendor/KohakuRAG/configs/hf_qwen7b.py",
        "bench_name": "qwen7b-bench",
        "model_id": "Qwen/Qwen2.5-7B-Instruct",
        "params_b": 7,
        "approx_vram_gb": 6,
    },
    "14": {
        "config": "vendor/KohakuRAG/configs/hf_qwen14b.py",
        "bench_name": "qwen14b-bench",
        "model_id": "Qwen/Qwen2.5-14B-Instruct",
        "params_b": 14,
        "approx_vram_gb": 10,
    },
    "32": {
        "config": "vendor/KohakuRAG/configs/hf_qwen32b.py",
        "bench_name": "qwen32b-bench",
        "model_id": "Qwen/Qwen2.5-32B-Instruct",
        "params_b": 32,
        "approx_vram_gb": 20,
    },
    "72": {
        "config": "vendor/KohakuRAG/configs/hf_qwen72b.py",
        "bench_name": "qwen72b-bench",
        "model_id": "Qwen/Qwen2.5-72B-Instruct",
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


def find_existing_summary(bench_name: str, env: str = "", datafile_stem: str = "train_QA") -> dict | None:
    """Search for an existing summary.json matching this bench name."""
    # Try explicit paths in order of preference (new structure first, then legacy)
    candidates = []
    if env:
        candidates.append(EXPERIMENTS_BASE / env / datafile_stem / bench_name / "summary.json")
        candidates.append(EXPERIMENTS_BASE / env / bench_name / "summary.json")
    candidates.append(EXPERIMENTS_BASE / datafile_stem / bench_name / "summary.json")
    candidates.append(EXPERIMENTS_BASE / bench_name / "summary.json")

    for path in candidates:
        if path.exists():
            with open(path) as f:
                return json.load(f)

    # Fallback: search recursively for any match
    for path in EXPERIMENTS_BASE.glob(f"**/{bench_name}/summary.json"):
        with open(path) as f:
            return json.load(f)

    return None


def run_single_model(config_path: str, bench_name: str, env: str = "",
                     precision: str = "4bit") -> tuple[bool, dict | None]:
    """Run a single experiment via subprocess, return (success, summary_dict)."""
    cmd = [
        sys.executable, "scripts/run_experiment.py",
        "--config", config_path,
        "--name", bench_name,
        "--precision", precision,
    ]
    if env:
        cmd.extend(["--env", env])

    print(f"\n{'='*70}")
    print(f"Running: {bench_name} ({config_path})")
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
            summary = find_existing_summary(bench_name, env)
            if summary:
                print(f"\n[OK] {bench_name} completed in {elapsed:.0f}s (score: {summary.get('overall_score', 0):.3f})")
                return True, summary
            else:
                print(f"\n[WARN] {bench_name} completed but no summary.json found")
                return True, None
        else:
            print(f"\n[FAIL] {bench_name} failed after {elapsed:.0f}s")
            return False, None

    except subprocess.TimeoutExpired:
        print(f"\n[TIMEOUT] {bench_name} exceeded 1 hour")
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
                "quantization": summary.get("quantization", summary.get("config_snapshot", {}).get("hf_dtype", "4bit")),
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
            print(f"{model_name:<30s} {'--':>7s}")
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
        description="Qwen model size scaling comparison",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--sizes", nargs="+", default=None,
        help="Which sizes to include (e.g., 1.5 3 7). Default: all.",
    )
    parser.add_argument(
        "--run-missing", action="store_true",
        help="Run experiments for sizes that don't have existing results",
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
        help="Run environment label (e.g. 'GB10', 'PowerEdge')",
    )
    parser.add_argument(
        "--precision", "-p", default="4bit",
        choices=["4bit", "bf16", "fp16", "auto"],
        help="Model precision/quantization (default: 4bit)",
    )
    parser.add_argument(
        "--datafile", "-d", default="train_QA",
        help="Datafile subfolder name (default: train_QA). "
             "Set to match the questions CSV stem (e.g. 'test_solutions').",
    )

    args = parser.parse_args()

    # Determine which sizes to include
    if args.sizes:
        sizes = [s for s in args.sizes if s in QWEN_MODELS]
        if not sizes:
            print(f"No valid sizes. Choose from: {list(QWEN_MODELS.keys())}")
            return
    else:
        sizes = list(QWEN_MODELS.keys())

    # Collect existing results
    summaries = {}
    missing_sizes = []

    for size_key in sizes:
        model_info = QWEN_MODELS[size_key]
        summary = find_existing_summary(model_info["bench_name"], args.env, args.datafile)
        if summary:
            summaries[size_key] = summary
            print(f"  [FOUND] {model_info['bench_name']}: score={summary.get('overall_score', 0):.3f}")
        else:
            summaries[size_key] = None
            missing_sizes.append(size_key)
            print(f"  [MISSING] {model_info['bench_name']}")

    print(f"\nFound {len(sizes) - len(missing_sizes)}/{len(sizes)} Qwen results")

    # Optionally run missing sizes
    if missing_sizes and args.run_missing:
        # Check available VRAM
        available_vram = get_available_vram_gb()
        if available_vram > 0 and not args.force_all:
            print(f"Available VRAM: ~{available_vram:.1f} GB")
            runnable = [
                s for s in missing_sizes
                if QWEN_MODELS[s]["approx_vram_gb"] <= available_vram * 0.95
            ]
            skipped_vram = len(missing_sizes) - len(runnable)
            if skipped_vram:
                print(f"Skipping {skipped_vram} model(s) that likely won't fit. Use --force-all to override.")
            missing_sizes = runnable

        # Check configs exist
        missing_configs = [s for s in missing_sizes if not Path(QWEN_MODELS[s]["config"]).exists()]
        if missing_configs:
            print(f"Missing config files for sizes: {missing_configs}")
            missing_sizes = [s for s in missing_sizes if s not in missing_configs]

        if missing_sizes:
            print(f"\nWill run: {[QWEN_MODELS[s]['bench_name'] for s in missing_sizes]}")

            if args.dry_run:
                print("[DRY RUN] Would run the above models. Exiting.")
            else:
                for size_key in sorted(missing_sizes, key=_sort_key):
                    model_info = QWEN_MODELS[size_key]
                    success, summary = run_single_model(
                        model_info["config"], model_info["bench_name"],
                        env=args.env, precision=args.precision,
                    )
                    summaries[size_key] = summary

                    # Free VRAM between models
                    gc.collect()
                    try:
                        import torch
                        torch.cuda.empty_cache()
                    except (ImportError, RuntimeError):
                        pass
    elif missing_sizes and not args.run_missing:
        print(f"\n{len(missing_sizes)} size(s) missing. Use --run-missing to run them.")

    # Generate comparison outputs
    found = {k: v for k, v in summaries.items() if v is not None}
    if not found:
        print("\nNo results to compare.")
        return

    if args.env:
        output_dir = EXPERIMENTS_BASE / args.env / args.datafile
    else:
        output_dir = EXPERIMENTS_BASE / args.datafile
    output_dir.mkdir(parents=True, exist_ok=True)

    comparison_csv = output_dir / "qwen_scaling_comparison.csv"
    generate_comparison_csv(summaries, comparison_csv)

    comparison_json = output_dir / "qwen_scaling_comparison.json"
    with open(comparison_json, "w") as f:
        json.dump(summaries, f, indent=2)
    print(f"Comparison JSON saved to: {comparison_json}")

    print_comparison_table(summaries)

    found_count = len(found)
    missing_count = sum(1 for v in summaries.values() if v is None)
    print(f"\nFound: {found_count}, Missing: {missing_count}")


if __name__ == "__main__":
    main()
