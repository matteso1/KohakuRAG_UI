#!/usr/bin/env python3
"""
Ensemble Experiment Runner

Supports two modes:

1. **Same-model ensemble** (``--config`` + ``--num-runs``):
   Runs *m* independent inference passes of the **same model** and aggregates
   results via voting.  This is the KohakuRAG competition strategy that proved
   most robust across public and private test partitions.

2. **Cross-model ensemble** (``--experiments``):
   Aggregates results from previously completed experiments (different models
   or different runs).

Aggregation strategies (matching vendor wattbot_aggregate.py):
- majority:        Most common answer wins (ties → first occurrence)
- answer_priority: Vote answer first, then collect refs only from matching
                   runs — ensures citation consistency (KohakuRAG default)
- first_non_blank: First non-blank answer wins (good for NA detection)

Abstention-aware voting (``--ignore-blank``):
  If any run produces a non-blank answer, blank ("is_blank") runs are filtered
  out before voting.  Enabled by default for same-model ensembles.

Usage:
    # Same-model ensemble: 5 runs of qwen7b, answer_priority voting
    python scripts/run_ensemble.py \\
        --config vendor/KohakuRAG/configs/hf_qwen7b.py \\
        --num-runs 5 --name qwen7b-ensemble-5x --env GB10

    # Cross-model ensemble from existing experiments
    python scripts/run_ensemble.py \\
        --experiments qwen7b-v1 llama3-8b-v1 mistral7b-v1 \\
        --name ensemble-3way --env PowerEdge

    # Cross-model with answer_priority voting
    python scripts/run_ensemble.py \\
        --experiments qwen7b-v1 llama3-8b-v1 \\
        --strategy answer_priority --name ensemble-ap --env GB10
"""

import argparse
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from score import row_bits, is_blank, ref_overlap_score


# =============================================================================
# Loading helpers
# =============================================================================


def load_experiment_results(
    experiments_dir: Path,
    experiment_names: list[str],
    env: str = "",
    datafile: str = "",
) -> dict[str, list[dict]]:
    """Load results.json from each experiment.

    Supports flat, env-nested, and datafile-nested directory layouts.
    When env and/or datafile are provided, the search is scoped to
    experiments_dir/<env>/<datafile>/ so that identically-named experiments
    under different datafile subfolders don't collide.
    """
    all_results = {}

    # Narrow the search root when env/datafile are known
    search_root = experiments_dir
    if env:
        search_root = search_root / env
    if datafile:
        search_root = search_root / datafile

    # Build lookup: experiment name -> results.json path
    all_results_paths = {
        p.parent.name: p
        for p in search_root.glob("**/results.json")
    }

    for name in experiment_names:
        # Try scoped path first, then fallback to lookup
        results_path = search_root / name / "results.json"
        if not results_path.exists():
            results_path = all_results_paths.get(name)
        if results_path is None or not results_path.exists():
            print(f"Warning: No results found for {name} under {search_root}")
            continue

        with open(results_path) as f:
            all_results[name] = json.load(f)

    return all_results


def infer_datafile_stem(experiments_dir: Path, experiment_names: list[str], env: str = "") -> str:
    """Infer the datafile subfolder from source experiments' summary.json."""
    search_root = experiments_dir / env if env else experiments_dir
    for p in search_root.glob("**/summary.json"):
        if p.parent.name in experiment_names:
            try:
                with open(p) as f:
                    data = json.load(f)
                qf = data.get("questions_file", "")
                if qf:
                    return Path(qf).stem
            except (json.JSONDecodeError, OSError):
                continue
    return "train_QA"


def load_ground_truth(gt_path: Path) -> pd.DataFrame:
    """Load ground truth CSV."""
    return pd.read_csv(gt_path)


# =============================================================================
# Voting / aggregation
# =============================================================================


def aggregate_majority(answers: list[str], ignore_blank: bool = False) -> str:
    """Return most common answer.  Ties go to first occurrence.

    If *ignore_blank* is True, "is_blank" values are filtered out before voting
    (only when non-blank answers exist).
    """
    if not answers:
        return "is_blank"

    # Filter out empty strings
    valid_answers = [a for a in answers if a and a.strip()]

    # Optionally filter out is_blank / refusal answers so they can't outvote
    # real answers from stronger models
    if ignore_blank:
        non_blank = [a for a in valid_answers if not is_blank(a)]
        if non_blank:
            valid_answers = non_blank

    if not valid_answers:
        return "is_blank"

    if ignore_blank:
        non_blank = [a for a in valid_answers if a.lower() != "is_blank"]
        if non_blank:
            valid_answers = non_blank

    counter = Counter(valid_answers)
    max_count = counter.most_common(1)[0][1]
    tied = [v for v, c in counter.items() if c == max_count]
    if len(tied) == 1:
        return tied[0]
    # First-occurrence tiebreak
    for v in valid_answers:
        if v in tied:
            return v
    return tied[0]


def aggregate_first_non_blank(answers: list[str]) -> str:
    """Return first non-blank answer."""
    for a in answers:
        if a and not is_blank(a):
            return a
    return "is_blank"


def _parse_ref_ids(ref_str: str) -> set[str]:
    """Parse a ref_id string/list into a set of document IDs."""
    if not ref_str or ref_str == "is_blank":
        return set()
    try:
        parsed = json.loads(ref_str.replace("'", '"'))
        if isinstance(parsed, list):
            return {str(x).strip() for x in parsed if x}
    except (json.JSONDecodeError, ValueError):
        pass
    return {ref_str.strip()} if ref_str.strip() else set()


def _format_ref_ids(ref_set: set[str]) -> str:
    """Format ref_id set back to JSON list string."""
    if not ref_set:
        return "is_blank"
    return json.dumps(sorted(ref_set))


def aggregate_refs_union(ref_lists: list[str]) -> str:
    """Aggregate reference IDs — union of all refs."""
    all_refs: set[str] = set()
    for ref_str in ref_lists:
        all_refs.update(_parse_ref_ids(ref_str))
    return _format_ref_ids(all_refs)


def aggregate_answer_priority(
    answers: list[str],
    refs: list[str],
    ignore_blank: bool = False,
) -> tuple[str, str]:
    """Vote on answer_value first, then collect refs only from matching runs.

    This is the AnswerPriority strategy from the KohakuRAG competition that
    ensures citation consistency — refs are only drawn from runs that agree
    with the winning answer.
    """
    best_val = aggregate_majority(answers, ignore_blank=ignore_blank)

    # Collect refs only from runs whose answer matches the winning value
    matching_refs: list[str] = []
    for ans, ref in zip(answers, refs):
        if ans == best_val:
            matching_refs.append(ref)

    # Union refs from matching runs
    combined_refs: set[str] = set()
    for ref_str in matching_refs:
        combined_refs.update(_parse_ref_ids(ref_str))

    return best_val, _format_ref_ids(combined_refs)


# =============================================================================
# Ensemble core
# =============================================================================


def run_ensemble(
    all_results: dict[str, list[dict]],
    strategy: str = "majority",
    ignore_blank: bool = False,
    model_weights: dict[str, float] | None = None,
) -> list[dict]:
    """Combine results from multiple runs/models.

    Args:
        all_results: Dict mapping run name -> list of per-question result dicts
        strategy: "majority", "answer_priority", or "first_non_blank"
        ignore_blank: If True, filter "is_blank" before voting when non-blank
                      answers exist (abstention-aware voting)
    """
    if not all_results:
        return []

    # Get all question IDs from first model
    first_model = list(all_results.keys())[0]
    question_ids = [r["id"] for r in all_results[first_model]]

    # Build lookup by question ID for each model
    results_by_id = {}
    for model_name, results in all_results.items():
        results_by_id[model_name] = {r["id"]: r for r in results}

    ensemble_results = []
    model_names = list(all_results.keys())

    for qid in question_ids:
        # Collect answers from all runs for this question
        answers = []
        refs = []
        explanations = []

        for model_name in model_names:
            if qid in results_by_id[model_name]:
                r = results_by_id[model_name][qid]
                answers.append(r.get("pred_value", "is_blank"))
                refs.append(r.get("pred_ref", "is_blank"))
                explanations.append(r.get("pred_explanation", ""))

        # Aggregate based on strategy
        if strategy == "answer_priority":
            final_value, final_ref = aggregate_answer_priority(
                answers, refs, ignore_blank=ignore_blank,
            )
        elif strategy == "first_non_blank":
            final_value = aggregate_first_non_blank(answers)
            final_ref = aggregate_refs_union(refs)
        else:
            # majority (default) — scope refs to winning voters only
            final_value = aggregate_majority(answers, ignore_blank=ignore_blank)
            winning_refs = [
                ref for ans, ref in zip(answers, refs)
                if ans == final_value
            ]
            final_ref = aggregate_refs_union(winning_refs)

        # Get metadata from first run's result
        first_result = results_by_id[first_model].get(qid, {})

        ensemble_results.append({
            "id": qid,
            "question": first_result.get("question", ""),
            "gt_value": first_result.get("gt_value", ""),
            "gt_unit": first_result.get("gt_unit", ""),
            "gt_ref": first_result.get("gt_ref", ""),
            "pred_value": final_value,
            "pred_unit": first_result.get("pred_unit", ""),
            "pred_ref": final_ref,
            "pred_explanation": f"Ensemble ({strategy}) of {len(model_names)} runs: {', '.join(model_names)}",
            "raw_response": "",
            "individual_answers": dict(zip(model_names, answers)),
            "individual_refs": dict(zip(model_names, refs)),
        })

    return ensemble_results


# =============================================================================
# Scoring
# =============================================================================


def score_ensemble_results(results: list[dict]) -> dict:
    """Score the ensemble results."""
    total = len(results)
    value_correct = 0
    ref_scores = []
    na_correct = 0

    for r in results:
        bits = row_bits(
            sol={"answer_value": r["gt_value"], "answer_unit": r["gt_unit"], "ref_id": r["gt_ref"]},
            sub={"answer_value": r["pred_value"], "answer_unit": r["pred_unit"], "ref_id": r["pred_ref"]},
        )

        if bits["val"]:
            value_correct += 1
        ref_scores.append(bits["ref"])
        if bits["na"]:
            na_correct += 1

        # Add scores to result
        r["value_correct"] = bool(bits["val"])
        r["ref_score"] = float(bits["ref"])
        r["na_correct"] = bool(bits["na"])

    value_accuracy = value_correct / total
    ref_overlap = sum(ref_scores) / total

    # NA component: recall over truly-NA questions only
    na_gt = [r for r in results if is_blank(r.get("gt_value", ""))]
    if na_gt:
        na_recall = sum(1 for r in na_gt if r.get("na_correct", False)) / len(na_gt)
    else:
        na_recall = 1.0
    overall = 0.75 * value_accuracy + 0.15 * ref_overlap + 0.10 * na_recall

    return {
        "value_accuracy": value_accuracy,
        "ref_overlap": ref_overlap,
        "na_accuracy": na_recall,
        "overall_score": overall,
        "questions_correct": value_correct,
        "questions_wrong": total - value_correct,
    }


# =============================================================================
# Save helpers
# =============================================================================


def save_ensemble_results(
    results: list[dict],
    scores: dict,
    output_dir: Path,
    model_names: list[str],
    strategy: str,
    *,
    ensemble_type: str = "cross-model",
    ignore_blank: bool = False,
    num_runs: int | None = None,
    config_path: str | None = None,
):
    """Save ensemble results."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save results JSON
    results_path = output_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved results: {results_path}")

    # Save summary JSON
    summary = {
        "name": output_dir.name,
        "type": "ensemble",
        "ensemble_type": ensemble_type,
        "strategy": strategy,
        "ignore_blank": ignore_blank,
        "models": model_names,
        "num_models": len(model_names),
        "num_questions": len(results),
        **scores,
    }
    if num_runs is not None:
        summary["num_runs"] = num_runs
    if config_path is not None:
        summary["config_path"] = config_path

    summary_path = output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary: {summary_path}")

    # Save submission CSV
    submission_rows = []
    for r in results:
        submission_rows.append({
            "id": r["id"],
            "question": r["question"],
            "answer": r["pred_explanation"],
            "answer_value": r["pred_value"],
            "answer_unit": r["pred_unit"],
            "ref_id": r["pred_ref"],
            "ref_url": "is_blank",
            "supporting_materials": "is_blank",
            "explanation": r["pred_explanation"],
        })

    sub_df = pd.DataFrame(submission_rows)
    sub_path = output_dir / "submission.csv"
    sub_df.to_csv(sub_path, index=False)
    print(f"Saved submission: {sub_path}")


# =============================================================================
# Same-model ensemble: run m independent inference passes
# =============================================================================


def run_same_model_inference(
    config_path: str,
    num_runs: int,
    ensemble_name: str,
    env: str,
    questions: str | None,
    precision: str,
) -> list[str]:
    """Run *num_runs* independent inference passes using run_experiment.py.

    Each run is executed as a separate subprocess to ensure completely
    independent model state (fresh random seeds, no shared GPU state).

    Returns a list of experiment names that completed successfully.
    """
    run_experiment_script = Path(__file__).parent / "run_experiment.py"
    completed_names: list[str] = []

    for i in range(num_runs):
        run_name = f"{ensemble_name}_run{i}"
        cmd = [
            sys.executable, str(run_experiment_script),
            "--config", config_path,
            "--name", run_name,
            "--precision", precision,
        ]
        if env:
            cmd.extend(["--env", env])
        if questions:
            cmd.extend(["--questions", questions])

        print(f"\n{'='*60}")
        print(f"ENSEMBLE RUN {i+1}/{num_runs}: {run_name}")
        print(f"{'='*60}")

        result = subprocess.run(cmd, capture_output=False)
        if result.returncode == 0:
            completed_names.append(run_name)
            print(f"  Run {i+1}/{num_runs} complete: {run_name}")
        else:
            print(f"  WARNING: Run {i+1}/{num_runs} failed (exit code {result.returncode})")

    return completed_names


# =============================================================================
# CLI entry point
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Ensemble: same-model (--config) or cross-model (--experiments)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Same-model ensemble: 5 independent runs, answer_priority voting
  python scripts/run_ensemble.py \\
      --config vendor/KohakuRAG/configs/hf_qwen7b.py \\
      --num-runs 5 --name qwen7b-ens5 --env GB10

  # Cross-model ensemble from existing experiments
  python scripts/run_ensemble.py \\
      --experiments qwen7b-v1 llama3-8b-v1 \\
      --name ensemble-2way --env PowerEdge
""",
    )

    # Mode 1: same-model ensemble
    parser.add_argument(
        "--config", "-c",
        default=None,
        help="Config file for same-model ensemble (runs m independent passes)",
    )
    parser.add_argument(
        "--num-runs", "-m",
        type=int,
        default=5,
        help="Number of independent inference runs for same-model ensemble (default: 5)",
    )
    parser.add_argument(
        "--questions", "-q",
        default=None,
        help="Override questions CSV path (same-model mode)",
    )
    parser.add_argument(
        "--precision", "-p",
        default="4bit",
        choices=["4bit", "bf16", "fp16", "auto"],
        help="Model precision (same-model mode, default: 4bit)",
    )

    # Mode 2: cross-model ensemble
    parser.add_argument(
        "--experiments", "-e",
        nargs="+",
        default=None,
        help="Names of existing experiments to ensemble",
    )

    # Shared options
    parser.add_argument(
        "--name", "-n",
        required=True,
        help="Name for the ensemble experiment",
    )
    parser.add_argument(
        "--strategy", "-s",
        choices=["majority", "answer_priority", "first_non_blank"],
        default="answer_priority",
        help="Aggregation strategy (default: answer_priority)",
    )
    parser.add_argument(
        "--env", "-v",
        default="",
        help="Run environment label (e.g. 'GB10', 'PowerEdge')",
    )
    parser.add_argument(
        "--datafile", "-d",
        default=None,
        help="Datafile subfolder (e.g. 'train_QA', 'test_solutions'). "
             "Auto-detected if not specified.",
    )
    parser.add_argument(
        "--ignore-blank",
        action="store_true",
        default=None,
        help="Filter 'is_blank' before voting if non-blank answers exist "
             "(enabled by default for same-model ensemble)",
    )
    parser.add_argument(
        "--no-ignore-blank",
        action="store_true",
        default=False,
        help="Disable abstention-aware voting",
    )

    args = parser.parse_args()

    # Validate mode
    if args.config is None and args.experiments is None:
        parser.error("Must specify either --config (same-model) or --experiments (cross-model)")
    if args.config is not None and args.experiments is not None:
        parser.error("Cannot specify both --config and --experiments")

    experiments_dir = Path(__file__).parent.parent / "artifacts" / "experiments"

    # Determine ignore_blank default
    if args.no_ignore_blank:
        ignore_blank = False
    elif args.ignore_blank is not None:
        ignore_blank = args.ignore_blank
    else:
        # Default: True for same-model, False for cross-model
        ignore_blank = args.config is not None

    # ---- Same-model ensemble ----
    if args.config is not None:
        print(f"\n{'='*60}")
        print(f"SAME-MODEL ENSEMBLE: {args.name}")
        print(f"Config: {args.config}")
        print(f"Runs: {args.num_runs}")
        print(f"Strategy: {args.strategy}")
        print(f"Ignore blank: {ignore_blank}")
        print(f"{'='*60}\n")

        # Run m independent inference passes
        completed_runs = run_same_model_inference(
            config_path=args.config,
            num_runs=args.num_runs,
            ensemble_name=args.name,
            env=args.env,
            questions=args.questions,
            precision=args.precision,
        )

        if len(completed_runs) < 2:
            print(f"Error: Only {len(completed_runs)} runs completed, need at least 2")
            sys.exit(1)

        # Determine datafile stem (needed to scope the load)
        if args.datafile:
            datafile_stem = args.datafile
        else:
            datafile_stem = infer_datafile_stem(experiments_dir, completed_runs)

        # Load results from completed runs
        all_results = load_experiment_results(
            experiments_dir, completed_runs, env=args.env, datafile=datafile_stem,
        )

        if len(all_results) < 2:
            print(f"Error: Only {len(all_results)} results loaded, need at least 2")
            sys.exit(1)

        ensemble_type = "same-model"
        config_path = args.config

    # ---- Cross-model ensemble ----
    else:
        # Determine datafile stem (needed to scope the load)
        if args.datafile:
            datafile_stem = args.datafile
        else:
            datafile_stem = infer_datafile_stem(experiments_dir, args.experiments)

        # Load results from existing experiments (scoped to env/datafile)
        all_results = load_experiment_results(
            experiments_dir, args.experiments, env=args.env, datafile=datafile_stem,
        )

        if len(all_results) < 2:
            print("Error: Need at least 2 experiments for ensemble")
            sys.exit(1)

        ensemble_type = "cross-model"
        config_path = None

    # Build output path
    if args.env:
        output_dir = experiments_dir / args.env / datafile_stem / args.name
    else:
        output_dir = experiments_dir / datafile_stem / args.name

    print(f"\n{'='*60}")
    print(f"AGGREGATING {len(all_results)} runs")
    print(f"Strategy: {args.strategy}")
    print(f"Ignore blank: {ignore_blank}")
    print(f"{'='*60}\n")

    # Run ensemble voting
    ensemble_results = run_ensemble(
        all_results,
        strategy=args.strategy,
        ignore_blank=ignore_blank,
    )

    # Score ensemble
    scores = score_ensemble_results(ensemble_results)

    # Save results
    save_ensemble_results(
        ensemble_results,
        scores,
        output_dir,
        list(all_results.keys()),
        args.strategy,
        ensemble_type=ensemble_type,
        ignore_blank=ignore_blank,
        num_runs=args.num_runs if args.config else None,
        config_path=config_path,
    )

    # Print summary
    print(f"\n{'='*60}")
    print("ENSEMBLE COMPLETE")
    print(f"{'='*60}")
    print(f"Type         : {ensemble_type}")
    print(f"Runs         : {', '.join(all_results.keys())}")
    print(f"Strategy     : {args.strategy}")
    print(f"Ignore blank : {ignore_blank}")
    print(f"Questions    : {len(ensemble_results)}")
    print(f"Correct      : {scores['questions_correct']} ({100*scores['value_accuracy']:.1f}%)")
    print(f"\nComponent Scores:")
    print(f"  Value match  : {scores['value_accuracy']:.3f}")
    print(f"  Ref overlap  : {scores['ref_overlap']:.3f}")
    print(f"  NA recall    : {scores['na_accuracy']:.3f}")
    print(f"\nOVERALL SCORE  : {scores['overall_score']:.3f}")
    print(f"\nOutput dir     : {output_dir}")


if __name__ == "__main__":
    main()
