#!/usr/bin/env python3
"""
Ensemble Experiment Runner

Runs multiple models and combines their outputs using voting/aggregation.

Usage:
    python scripts/run_ensemble.py --models llama4-scout-v1 haiku-baseline sonnet-v1 --name ensemble-3way
    python scripts/run_ensemble.py --experiments llama4-scout-v1 haiku-baseline --name ensemble-test

Aggregation strategies:
- majority: Most common answer wins (ties go to first model)
- confidence: Weighted by model's overall accuracy (requires pre-computed scores)
- first_non_blank: First non-blank answer wins (good for NA detection)
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from score import row_bits, is_blank, ref_overlap_score


def load_experiment_results(experiments_dir: Path, experiment_names: list[str]) -> dict[str, list[dict]]:
    """Load results.json from each experiment.

    Supports both flat (experiments/<name>/) and env-nested
    (experiments/<env>/<name>/) directory layouts.
    """
    all_results = {}

    # Build lookup: experiment name -> results.json path
    all_results_paths = {
        p.parent.name: p
        for p in experiments_dir.glob("**/results.json")
    }

    for name in experiment_names:
        # Try direct path first (supports <env>/<name> syntax), then name-only lookup
        results_path = experiments_dir / name / "results.json"
        if not results_path.exists():
            results_path = all_results_paths.get(name)
        if results_path is None or not results_path.exists():
            print(f"Warning: No results found for {name}")
            continue

        with open(results_path) as f:
            all_results[name] = json.load(f)

    return all_results


def load_ground_truth(gt_path: Path) -> pd.DataFrame:
    """Load ground truth CSV."""
    return pd.read_csv(gt_path)


def aggregate_majority(answers: list[str]) -> str:
    """Return most common answer. Ties go to first occurrence."""
    if not answers:
        return "is_blank"

    # Filter out empty strings
    valid_answers = [a for a in answers if a and a.strip()]
    if not valid_answers:
        return "is_blank"

    counter = Counter(valid_answers)
    most_common = counter.most_common(1)[0][0]
    return most_common


def aggregate_first_non_blank(answers: list[str]) -> str:
    """Return first non-blank answer."""
    for a in answers:
        if a and not is_blank(a):
            return a
    return "is_blank"


def aggregate_refs(ref_lists: list[str]) -> str:
    """Aggregate reference IDs - union of all refs."""
    all_refs = set()
    for ref_str in ref_lists:
        if ref_str and ref_str != "is_blank":
            try:
                if ref_str.startswith("["):
                    refs = json.loads(ref_str.replace("'", '"'))
                    all_refs.update(refs)
                else:
                    all_refs.add(ref_str)
            except:
                all_refs.add(ref_str)

    if not all_refs:
        return "is_blank"
    return json.dumps(sorted(all_refs))


def run_ensemble(
    all_results: dict[str, list[dict]],
    strategy: str = "majority",
    model_weights: dict[str, float] | None = None,
) -> list[dict]:
    """Combine results from multiple models."""
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
        # Collect answers from all models for this question
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
        if strategy == "majority":
            final_value = aggregate_majority(answers)
        elif strategy == "first_non_blank":
            final_value = aggregate_first_non_blank(answers)
        else:
            final_value = aggregate_majority(answers)

        final_ref = aggregate_refs(refs)

        # Get metadata from first model's result
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
            "pred_explanation": f"Ensemble of {len(model_names)} models: {', '.join(model_names)}",
            "raw_response": "",
            "individual_answers": dict(zip(model_names, answers)),
            "individual_refs": dict(zip(model_names, refs)),
        })

    return ensemble_results


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


def save_ensemble_results(
    results: list[dict],
    scores: dict,
    output_dir: Path,
    model_names: list[str],
    strategy: str,
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
        "strategy": strategy,
        "models": model_names,
        "num_models": len(model_names),
        "num_questions": len(results),
        **scores,
    }
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


def main():
    parser = argparse.ArgumentParser(description="Run ensemble from existing experiment results")
    parser.add_argument(
        "--experiments", "-e",
        nargs="+",
        required=True,
        help="Names of experiments to ensemble (e.g., haiku-baseline sonnet-v1)"
    )
    parser.add_argument(
        "--name", "-n",
        required=True,
        help="Name for the ensemble experiment"
    )
    parser.add_argument(
        "--strategy", "-s",
        choices=["majority", "first_non_blank"],
        default="majority",
        help="Aggregation strategy (default: majority)"
    )

    args = parser.parse_args()

    experiments_dir = Path(__file__).parent.parent / "artifacts" / "experiments"
    output_dir = experiments_dir / args.name

    print(f"\n{'='*60}")
    print(f"ENSEMBLE: {args.name}")
    print(f"Models: {', '.join(args.experiments)}")
    print(f"Strategy: {args.strategy}")
    print(f"{'='*60}\n")

    # Load results from each experiment
    all_results = load_experiment_results(experiments_dir, args.experiments)

    if len(all_results) < 2:
        print("Error: Need at least 2 experiments for ensemble")
        sys.exit(1)

    print(f"Loaded results from {len(all_results)} models")

    # Run ensemble
    ensemble_results = run_ensemble(all_results, strategy=args.strategy)

    # Score ensemble
    scores = score_ensemble_results(ensemble_results)

    # Save results
    save_ensemble_results(
        ensemble_results,
        scores,
        output_dir,
        list(all_results.keys()),
        args.strategy,
    )

    # Print summary
    print(f"\n{'='*60}")
    print("ENSEMBLE COMPLETE")
    print(f"{'='*60}")
    print(f"Models       : {', '.join(all_results.keys())}")
    print(f"Strategy     : {args.strategy}")
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
