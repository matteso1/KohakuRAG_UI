#!/usr/bin/env python3
"""
Full Benchmark Runner (Provider-Agnostic)

Runs ALL configured models on the training set. Supports both API-based models
(Bedrock, OpenRouter) and local HuggingFace models.

Includes a smoke test (1 question) before each full run to catch
failures early without burning through the whole question set.

Usage:
    # Smoke test only (1 question per model, fast)
    python scripts/run_full_benchmark.py --smoke-test

    # Full benchmark (all questions, all models)
    python scripts/run_full_benchmark.py

    # Single model only
    python scripts/run_full_benchmark.py --model qwen7b

    # Only local models
    python scripts/run_full_benchmark.py --provider hf_local

    # Only bedrock models
    python scripts/run_full_benchmark.py --provider bedrock
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path


# ============================================================================
# Model Registry
# Maps config filename (without .py) -> experiment name
# ============================================================================

# Local HuggingFace models
HF_LOCAL_MODELS = {
    "hf_qwen1_5b": "qwen1.5b-bench",
    "hf_qwen3b": "qwen3b-bench",
    "hf_qwen7b": "qwen7b-bench",
    "hf_qwen14b": "qwen14b-bench",
    "hf_qwen32b": "qwen32b-bench",
    "hf_llama3_8b": "llama3-8b-bench",
    "hf_mistral7b": "mistral7b-bench",
    "hf_phi3_mini": "phi3-mini-bench",
    "hf_qwen72b": "qwen72b-bench",
    "hf_qwen72b_4bit": "qwen72b-4bit-bench",
    "hf_gemma2_9b": "gemma2-9b-bench",
    "hf_gemma2_27b": "gemma2-27b-bench",
    "hf_llama3_70b_4bit": "llama3-70b-4bit-bench",
    "hf_mixtral_8x7b": "mixtral-8x7b-bench",
}

# AWS Bedrock models (requires llm_bedrock module + AWS credentials)
BEDROCK_MODELS = {
    "bedrock_haiku": "haiku-bench",
    "bedrock_claude35_haiku": "claude35-haiku-bench",
    "bedrock_sonnet": "sonnet-bench",
    "bedrock_claude37_sonnet": "claude37-sonnet-bench",
    "bedrock_llama3_70b": "llama3-70b-bench",
    "bedrock_llama4_scout": "llama4-scout-bench",
    "bedrock_llama4_maverick": "llama4-maverick-bench",
    "bedrock_nova_pro": "nova-pro-bench",
    "bedrock_deepseek_v3": "deepseek-r1-bench",
}

# All models combined
ALL_MODELS = {**HF_LOCAL_MODELS, **BEDROCK_MODELS}


def run_experiment(config_name: str, experiment_name: str, env: str = "") -> tuple[bool, str]:
    """Run a single experiment. Returns (success, output_summary)."""
    config_path = f"vendor/KohakuRAG/configs/{config_name}.py"

    if not Path(config_path).exists():
        return False, f"Config not found: {config_path}"

    cmd = [
        sys.executable, "scripts/run_experiment.py",
        "--config", config_path,
        "--name", experiment_name,
    ]
    if env:
        cmd.extend(["--env", env])

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=1800,  # 30 minute timeout (local models can be slow)
        )

        output = result.stdout + result.stderr
        success = result.returncode == 0

        # Look for the score line
        score_line = ""
        for line in output.split("\n"):
            if "OVERALL SCORE" in line:
                score_line = line.strip()
            elif "Error" in line and "Traceback" not in line:
                score_line = line.strip()[:120]

        if not success:
            error_lines = [l for l in output.split("\n") if "Error" in l or "error" in l]
            error_summary = error_lines[-1][:150] if error_lines else "Unknown error"
            return False, error_summary

        return True, score_line

    except subprocess.TimeoutExpired:
        return False, "TIMEOUT (>30 min)"
    except Exception as e:
        return False, str(e)[:150]


def main():
    parser = argparse.ArgumentParser(description="Run full benchmark across all models")
    parser.add_argument(
        "--smoke-test", action="store_true",
        help="Only run 1 question per model to verify they work",
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="Run only a specific model (e.g., 'qwen7b', 'haiku')",
    )
    parser.add_argument(
        "--provider", type=str, default=None,
        choices=["hf_local", "bedrock", "all"],
        help="Run only models from a specific provider",
    )
    parser.add_argument(
        "--env", "-e", default="",
        help="Run environment label (e.g. 'GB10', 'PowerEdge') for cross-machine comparison",
    )
    args = parser.parse_args()

    # Select model set
    if args.provider == "hf_local":
        models = HF_LOCAL_MODELS
    elif args.provider == "bedrock":
        models = BEDROCK_MODELS
    else:
        models = ALL_MODELS

    # Filter by specific model name
    if args.model:
        models = {
            k: v for k, v in models.items()
            if args.model.lower() in k.lower() or args.model.lower() in v.lower()
        }
        if not models:
            print(f"No model matching '{args.model}'. Available:")
            for k in ALL_MODELS:
                print(f"  {k}")
            return

    # Filter out models without config files
    available_models = {}
    skipped = []
    for config_name, exp_name in models.items():
        if Path(f"vendor/KohakuRAG/configs/{config_name}.py").exists():
            available_models[config_name] = exp_name
        else:
            skipped.append(config_name)

    if skipped:
        print(f"Skipping {len(skipped)} models (no config file): {', '.join(skipped)}")

    if not available_models:
        print("No models available to benchmark. Create config files in vendor/KohakuRAG/configs/")
        return

    mode = "SMOKE TEST" if args.smoke_test else "FULL BENCHMARK"
    print(f"{'='*70}")
    print(f"{mode}: {len(available_models)} models")
    print(f"{'='*70}\n")

    results = {}
    total_start = time.time()

    for i, (config_name, exp_name) in enumerate(available_models.items(), 1):
        if args.smoke_test:
            exp_name = f"{exp_name}-smoke"

        print(f"[{i}/{len(available_models)}] {config_name} -> {exp_name}")
        start = time.time()

        success, summary = run_experiment(config_name, exp_name, env=args.env)
        elapsed = time.time() - start

        status = "PASS" if success else "FAIL"
        print(f"  [{status}] {elapsed:.1f}s - {summary}")
        print()

        results[config_name] = {
            "experiment": exp_name,
            "success": success,
            "summary": summary,
            "time": elapsed,
        }

    # Final summary
    total_time = time.time() - total_start
    passed = sum(1 for r in results.values() if r["success"])
    failed = sum(1 for r in results.values() if not r["success"])

    print(f"\n{'='*70}")
    print(f"RESULTS: {passed} passed, {failed} failed ({total_time:.0f}s total)")
    print(f"{'='*70}")

    for config_name, r in results.items():
        status = "PASS" if r["success"] else "FAIL"
        print(f"  [{status}] {config_name:<35s} {r['time']:>6.1f}s  {r['summary'][:60]}")

    if failed > 0:
        print(f"\n!!! {failed} model(s) FAILED -- fix before running full benchmark !!!")
        sys.exit(1)
    else:
        print(f"\nAll {passed} models passed!")
        if args.smoke_test:
            print("\nRun without --smoke-test for the full benchmark.")


if __name__ == "__main__":
    main()
