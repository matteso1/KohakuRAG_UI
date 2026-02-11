#!/usr/bin/env python3
"""
Generate Results Matrix

Creates a unified CSV/Excel file where each row is a question and columns show
the answers and checks from multiple models side-by-side.

Usage:
    python scripts/generate_results_matrix.py \
        --submissions artifacts/submission_*.csv \
        --output artifacts/results_matrix.csv
"""

import argparse
import glob
import json
import sys
from pathlib import Path
import pandas as pd

# Add project root for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.score import row_bits

def load_ground_truth(path: Path) -> pd.DataFrame:
    """Load ground truth with question type metadata."""
    if not path.exists():
        raise FileNotFoundError(f"Ground truth file not found: {path}")
    
    df = pd.read_csv(path, encoding="utf-8-sig")

    # Normalize ID column
    if "id" not in df.columns:
        # Try to infer ID column or create one
        print("Warning: 'id' column not found in ground truth. Using index.")
        df["id"] = df.index.astype(str)

    df["id"] = df["id"].astype(str)
    return df.set_index("id")

def load_submission(path: Path) -> pd.DataFrame:
    """Load a submission file."""
    df = pd.read_csv(path, encoding="utf-8-sig")
    # Ensure ID column is string
    if "id" in df.columns:
        df["id"] = df["id"].astype(str)
        return df.set_index("id")
    return None

def build_gt_from_results_json(submission_files: list) -> pd.DataFrame:
    """Build a ground truth DataFrame from results.json files next to submissions.

    Each results.json entry has gt_value, gt_unit, gt_ref, question, and id fields
    that can reconstruct the ground truth when the CSV ground truth file is missing
    or has non-overlapping IDs (e.g. train vs test split).
    """
    rows = {}
    for sub_path in submission_files:
        results_path = sub_path.parent / "results.json"
        if not results_path.exists():
            continue
        with open(results_path) as f:
            results = json.load(f)
        for entry in results:
            qid = str(entry.get("id", ""))
            if not qid or qid in rows:
                continue
            rows[qid] = {
                "question": entry.get("question", ""),
                "answer_value": entry.get("gt_value", ""),
                "answer_unit": entry.get("gt_unit", ""),
                "ref_id": entry.get("gt_ref", ""),
                "explanation": "",
            }
    if not rows:
        return None
    df = pd.DataFrame.from_dict(rows, orient="index")
    df.index.name = "id"
    return df


def main():
    parser = argparse.ArgumentParser(description="Generate Results Matrix")

    parser.add_argument(
        "--submissions", "-s",
        nargs="+",
        default=[
            "artifacts/submission_*.csv",
            "artifacts/experiments/*/submission.csv",
            "artifacts/experiments/*/*.csv",
            "artifacts/experiments/*/*/submission.csv",
            "artifacts/experiments/*/*/*.csv",
            "artifacts/experiments/*/*/*/submission.csv",
            "artifacts/experiments/*/*/*/*.csv",
        ],
        help="List of submission CSV files (supports glob patterns)"
    )
    parser.add_argument(
        "--ground-truth", "-g",
        default=None,
        help="Path to ground truth CSV (auto-detects test_solutions.csv or train_QA.csv)"
    )
    parser.add_argument(
        "--output", "-o",
        default="artifacts/results_matrix.csv",
        help="Output CSV path"
    )

    args = parser.parse_args()

    # Expand globs
    submission_files = []
    for pattern in args.submissions:
        expanded = glob.glob(pattern)
        if expanded:
            submission_files.extend([Path(p) for p in expanded])
        else:
            # Maybe it's a direct file path that doesn't need glob expansion
            p = Path(pattern)
            if p.exists():
                submission_files.append(p)

    unique_files = sorted(list(set(submission_files)))
    if not unique_files:
        print("No submission files found.")
        sys.exit(1)

    print(f"Loading {len(unique_files)} submission files...")

    # Load Ground Truth - try multiple sources in order of preference
    project_root = Path(__file__).parent.parent
    gt_df = None
    csv_gt_df = None  # keep CSV GT for explanation metadata

    if args.ground_truth:
        gt_candidates = [Path(args.ground_truth)]
    else:
        gt_candidates = [
            project_root / "data" / "test_solutions.csv",
            project_root / "data" / "train_QA.csv",
        ]

    for gt_path in gt_candidates:
        if gt_path.exists():
            try:
                csv_gt_df = load_ground_truth(gt_path)
                print(f"Loaded ground truth from {gt_path} ({len(csv_gt_df)} questions).")
                break
            except Exception as e:
                print(f"Warning: Could not load {gt_path}: {e}")

    # Check if CSV GT covers the submission IDs and pick best GT source
    if csv_gt_df is not None:
        sample_sub = load_submission(unique_files[0])
        if sample_sub is not None:
            overlap = len(csv_gt_df.index.intersection(sample_sub.index))
            sub_count = len(sample_sub)
            if overlap == 0:
                print(f"CSV ground truth has no overlapping IDs with submissions.")
            elif overlap < sub_count:
                print(f"CSV ground truth covers {overlap}/{sub_count} submission IDs.")

            # Use CSV GT directly if it covers all (or most) submissions
            if overlap >= sub_count * 0.9:
                gt_df = csv_gt_df
            else:
                # Try results.json for better ID coverage, merge CSV explanations
                json_gt = build_gt_from_results_json(unique_files)
                if json_gt is not None:
                    if "explanation" in csv_gt_df.columns:
                        common = json_gt.index.intersection(csv_gt_df.index)
                        if len(common) > 0:
                            json_gt.loc[common, "explanation"] = csv_gt_df.loc[common, "explanation"]
                            print(f"Merged explanations for {len(common)} questions from CSV ground truth.")
                    gt_df = json_gt
                    print(f"Using results.json ground truth ({len(json_gt)} questions).")
                else:
                    # CSV GT is still better than nothing
                    gt_df = csv_gt_df
                    print(f"Using CSV ground truth despite low overlap ({overlap}/{sub_count}).")
        else:
            gt_df = csv_gt_df

    if gt_df is None:
        gt_df = build_gt_from_results_json(unique_files)
        if gt_df is None:
            print("Error: No ground truth found (neither CSV nor results.json).")
            sys.exit(1)
        print(f"Reconstructed ground truth for {len(gt_df)} questions from results.json files.")

    # Initialize Master DataFrame with GT
    # Select relevant GT columns
    cols_to_keep = ["question", "answer_value", "ref_id"]
    # Add other metadata columns if they exist
    for c in ["explanation", "answer_unit", "question_type"]:
        if c in gt_df.columns:
            cols_to_keep.append(c)

    master_df = gt_df[cols_to_keep].copy()
    rename_map = {"answer_value": "GT_Value", "ref_id": "GT_Ref"}
    if "explanation" in cols_to_keep:
        rename_map["explanation"] = "GT_Explanation"
    master_df = master_df.rename(columns=rename_map)

    # Iterate through submissions and merge
    processed_models = set()

    for sub_path in unique_files:
        # Derive model name
        if sub_path.stem == "submission":
            # Use parent folder name: experiments/deepseek-r1-v1/submission.csv -> deepseek-r1
            model_name = sub_path.parent.name
        elif sub_path.stem.startswith("submission_"):
            # artifacts/submission_llama4_scout.csv -> llama4_scout
            model_name = sub_path.stem.replace("submission_", "")
        else:
            model_name = sub_path.stem

        # Clean up tags
        model_name = model_name.replace("-v1", "").replace("bedrock_", "")

        if model_name in processed_models:
            print(f"Skipping duplicate model: {model_name} ({sub_path})")
            continue

        processed_models.add(model_name)
        print(f"Processing {model_name}...")

        sub_df = load_submission(sub_path)
        if sub_df is None:
            print(f"Skipping {sub_path}: could not read")
            continue

        # Join with master
        # We only care about matching IDs
        common_ids = master_df.index.intersection(sub_df.index)

        if len(common_ids) == 0:
             print(f"Warning: No matching IDs for {model_name}")
             continue


        # Calculate scores for each row
        val_correct = []
        ref_correct = []
        na_correct = []

        for qid in common_ids:
            gt_row = gt_df.loc[qid]
            sub_row = sub_df.loc[qid]

            # Use row_bits from score.py logic
            bits = row_bits(
                sol={"answer_value": str(gt_row.get("answer_value", "")),
                     "answer_unit": str(gt_row.get("answer_unit", "")),
                     "ref_id": str(gt_row.get("ref_id", ""))},
                sub={"answer_value": str(sub_row.get("answer_value", "")),
                     "answer_unit": str(sub_row.get("answer_unit", "")),
                     "ref_id": str(sub_row.get("ref_id", ""))}
            )
            val_correct.append(bits["val"])
            ref_correct.append(bits["ref"])
            na_correct.append(bits["na"])

        # Create temporary DF for this model's data
        temp_df = pd.DataFrame(index=common_ids)
        temp_df[f"{model_name}_Value"] = sub_df.loc[common_ids, "answer_value"]
        temp_df[f"{model_name}_Ref"] = sub_df.loc[common_ids, "ref_id"]
        temp_df[f"{model_name}_ValCorrect"] = val_correct
        temp_df[f"{model_name}_RefScore"] = ref_correct
        temp_df[f"{model_name}_NACorrect"] = na_correct

        # Merge into master
        master_df = master_df.join(temp_df, how="left")

    # Output
    print(f"Writing matrix to {args.output}...")
    master_df.to_csv(args.output)
    print("Done!")

if __name__ == "__main__":
    main()
