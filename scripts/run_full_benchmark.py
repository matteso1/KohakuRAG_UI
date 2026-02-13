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
import json
import re
import subprocess
import sys
import time
from pathlib import Path

EXPERIMENTS_BASE = Path(__file__).parent.parent / "artifacts" / "experiments"


# ============================================================================
# Model Registry
# Maps config filename (without .py) -> experiment name
# ============================================================================

# Local HuggingFace models
HF_LOCAL_MODELS = {
    # "hf_qwen1_5b": "qwen1.5b-bench",       # too small to be useful
    "hf_qwen3b": "qwen3b-bench",
    "hf_qwen7b": "qwen7b-bench",
    "hf_qwen14b": "qwen14b-bench",
    "hf_qwen32b": "qwen32b-bench",
    # "hf_llama3_8b": "llama3-8b-bench",     # gated — needs HF_TOKEN
    "hf_mistral7b": "mistral7b-bench",
    "hf_phi3_mini": "phi3-mini-bench",
    "hf_qwen72b": "qwen72b-bench",
    "hf_qwen1_5_110b": "qwen1.5-110b-bench",
    # "hf_gemma2_9b": "gemma2-9b-bench",     # gated — needs HF_TOKEN
    # "hf_gemma2_27b": "gemma2-27b-bench",   # gated — needs HF_TOKEN
    "hf_mixtral_8x7b": "mixtral-8x7b-bench",
    "hf_mixtral_8x22b": "mixtral-8x22b-bench",
    "hf_qwen3_30b_a3b": "qwen3-30b-a3b-bench",
    "hf_qwen3_next_80b_a3b": "qwen3-next-80b-a3b-bench",
    "hf_qwen3_next_80b_a3b_thinking": "qwen3-next-80b-a3b-thinking-bench",
    "hf_olmoe_1b7b": "olmoe-1b7b-bench",
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


def datafile_stem_from_questions(questions_path: str | None) -> str:
    """Derive the datafile subfolder name from the questions CSV path."""
    if questions_path:
        return Path(questions_path).stem
    return "train_QA"  # default questions file used by configs


def experiment_dir(experiment_name: str, env: str = "", datafile_stem: str = "train_QA") -> Path:
    """Return the expected output directory for an experiment."""
    if env:
        return EXPERIMENTS_BASE / env / datafile_stem / experiment_name
    return EXPERIMENTS_BASE / datafile_stem / experiment_name


def check_existing(experiment_name: str, env: str = "", datafile_stem: str = "train_QA") -> str | None:
    """If summary.json exists for this experiment, return the score line. Else None."""
    summary_path = experiment_dir(experiment_name, env, datafile_stem) / "summary.json"
    if not summary_path.exists():
        return None
    try:
        with open(summary_path) as f:
            data = json.load(f)
        score = data.get("overall_score")
        if score is not None:
            return f"OVERALL SCORE  : {score:.3f} (cached)"
        return "completed (cached)"
    except (json.JSONDecodeError, KeyError):
        return None


def run_experiment(config_name: str, experiment_name: str, env: str = "",
                    questions: str | None = None,
                    precision: str = "4bit") -> tuple[bool, str]:
    """Run a single experiment. Returns (success, output_summary)."""
    config_path = f"vendor/KohakuRAG/configs/{config_name}.py"

    if not Path(config_path).exists():
        return False, f"Config not found: {config_path}"

    cmd = [
        sys.executable, "-u", "scripts/run_experiment.py",
        "--config", config_path,
        "--name", experiment_name,
        "--precision", precision,
    ]
    if env:
        cmd.extend(["--env", env])
    if questions:
        cmd.extend(["--questions", questions])

    _noise_re = re.compile(
        r"Loading checkpoint shards|Fetching \d+ files|Encoding texts"
        r"|FutureWarning|warnings\.warn|torch_dtype|TRANSFORMERS_CACHE"
        r"|malicious code|downloaded from https://huggingface"
        r"|^\s*- (configuration|modeling)_"
        r"|Setting `pad_token_id`"
        r"|\d+%\|[█▏▎▍▌▋▊▉ ]*\|"
    )
    # Matches e.g. "[3/41] Q123: some answer [OK] (8.2s | ret=0.5s gen=7.7s)"
    _progress_re = re.compile(r"^\[(\d+)/(\d+)\].*\[(?:OK|WRONG)\]\s*\((\d+\.\d+)s")
    # Matches "[1/282] Q123: processing..." or "[1/282] Q123: TIMEOUT ..."
    _status_re = re.compile(r"^\[(\d+)/(\d+)\].*(?:processing\.\.\.|TIMEOUT|ERROR)")
    # Stage prefixes to forward so the user sees loading/run progress
    _stage_prefixes = ("[init]", "[run]", "[resume]", "[monitor]", "[checkpoint]", "Loaded ")

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

        all_lines = []
        in_report = False
        score_line = ""
        current_q = 0
        total_q = 0
        elapsed_sum = 0.0
        start_time = time.time()

        for line in iter(proc.stdout.readline, ""):
            line = line.rstrip("\n")
            all_lines.append(line)

            # Forward key stage lines so the user sees loading progress
            if any(line.startswith(p) for p in _stage_prefixes):
                print(f"  {line}", flush=True)
                continue

            # Forward "processing..." / TIMEOUT / ERROR status lines
            if _status_re.match(line):
                elapsed = time.time() - start_time
                print(f"\r  {line}  [{elapsed:.0f}s elapsed]", end="", flush=True)
                continue

            # Skip noisy lines early (before progress/report checks)
            if _noise_re.search(line):
                continue

            # Extract progress from per-question output
            m = _progress_re.match(line)
            if m:
                current_q = int(m.group(1))
                total_q = int(m.group(2))
                q_time = float(m.group(3))
                elapsed_sum += q_time
                avg = elapsed_sum / current_q
                eta = avg * (total_q - current_q)
                bar_len = 20
                filled = int(bar_len * current_q / total_q) if total_q else 0
                bar = "█" * filled + "░" * (bar_len - filled)
                print(f"\r  {bar} {current_q}/{total_q}  avg={avg:.1f}s  ETA={eta:.0f}s  ", end="", flush=True)
                continue

            # Capture report section
            if "EXPERIMENT COMPLETE" in line:
                if current_q > 0:
                    print()  # newline after progress bar
                in_report = True
            if in_report:
                stripped = line.strip()
                if not stripped or _noise_re.search(line):
                    continue
                print(f"  {line}")

            # Capture score line
            if "OVERALL SCORE" in line:
                score_line = line.strip()

        proc.wait(timeout=1800)
        success = proc.returncode == 0

        if not success:
            full_output = "\n".join(all_lines)
            if "401" in full_output or "Access denied" in full_output:
                return False, "Gated model — accept license on HuggingFace + set HF_TOKEN"
            error_lines = [l for l in all_lines if "Error" in l or "error" in l]
            error_summary = error_lines[-1][:150] if error_lines else "Unknown error"
            # Clear progress bar before error output
            if current_q > 0:
                print()
            return False, error_summary

        return True, score_line

    except subprocess.TimeoutExpired:
        proc.kill()
        if current_q > 0:
            print()
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
    parser.add_argument(
        "--questions", "-q", default=None,
        help="Override questions CSV path (e.g. data/test_solutions.csv)",
    )
    parser.add_argument(
        "--precision", "-p", default="4bit",
        choices=["4bit", "bf16", "fp16", "auto"],
        help="Model precision/quantization for HF local models (default: 4bit)",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-run experiments even if results already exist",
    )
    parser.add_argument(
        "--split", type=str, default=None,
        help="Append a suffix to experiment names (e.g. --split test → qwen7b-bench-test). "
             "Use this to run on a different question set without overwriting existing results.",
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

    # Derive datafile subfolder from --questions arg
    datafile_stem = datafile_stem_from_questions(args.questions)

    mode = "SMOKE TEST" if args.smoke_test else "FULL BENCHMARK"
    print(f"{'='*70}")
    print(f"{mode}: {len(available_models)} models")
    print(f"{'='*70}\n")

    results = {}
    total_start = time.time()

    for i, (config_name, exp_name) in enumerate(available_models.items(), 1):
        if args.smoke_test:
            exp_name = f"{exp_name}-smoke"
        if args.split:
            exp_name = f"{exp_name}-{args.split}"

        print(f"[{i}/{len(available_models)}] {config_name} -> {exp_name}")

        # Skip if results already exist (unless --force)
        if not args.force:
            cached = check_existing(exp_name, args.env, datafile_stem)
            if cached:
                print(f"  [SKIP] already exists - {cached}")
                print()
                results[config_name] = {
                    "experiment": exp_name,
                    "success": True,
                    "summary": cached,
                    "time": 0,
                    "skipped": True,
                }
                continue

        start = time.time()

        success, summary = run_experiment(config_name, exp_name, env=args.env,
                                           questions=args.questions,
                                           precision=args.precision)
        elapsed = time.time() - start

        status = "PASS" if success else "FAIL"
        print(f"  [{status}] {elapsed:.1f}s - {summary}")
        print()

        results[config_name] = {
            "experiment": exp_name,
            "success": success,
            "summary": summary,
            "time": elapsed,
            "skipped": False,
        }

    # Final summary
    total_time = time.time() - total_start
    skipped = sum(1 for r in results.values() if r.get("skipped"))
    passed = sum(1 for r in results.values() if r["success"] and not r.get("skipped"))
    failed = sum(1 for r in results.values() if not r["success"])

    skip_msg = f", {skipped} skipped" if skipped else ""
    print(f"\n{'='*70}")
    print(f"RESULTS: {passed} passed, {failed} failed{skip_msg} ({total_time:.0f}s total)")
    print(f"{'='*70}")

    for config_name, r in results.items():
        if r.get("skipped"):
            status = "SKIP"
        elif r["success"]:
            status = "PASS"
        else:
            status = "FAIL"
        print(f"  [{status}] {config_name:<35s} {r['time']:>6.1f}s  {r['summary'][:60]}")

    if failed > 0:
        print(f"\n!!! {failed} model(s) FAILED -- fix before running full benchmark !!!")
        sys.exit(1)
    else:
        print(f"\nAll {passed} models passed!")
        if skipped:
            print(f"({skipped} skipped — use --force to re-run)")
        if args.smoke_test:
            print("\nRun without --smoke-test for the full benchmark.")


if __name__ == "__main__":
    main()
