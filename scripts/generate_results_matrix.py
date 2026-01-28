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
    
    df = pd.read_csv(path)
    
    # Normalize ID column
    if "id" not in df.columns:
        # Try to infer ID column or create one
        print("Warning: 'id' column not found in ground truth. Using index.")
        df["id"] = df.index.astype(str)
    
    df["id"] = df["id"].astype(str)
    return df.set_index("id")

def load_submission(path: Path) -> pd.DataFrame:
    """Load a submission file."""
    df = pd.read_csv(path)
    # Ensure ID column is string
    if "id" in df.columns:
        df["id"] = df["id"].astype(str)
        return df.set_index("id")
    return None

def main():
    parser = argparse.ArgumentParser(description="Generate Results Matrix")

    parser.add_argument(
        "--submissions", "-s",
        nargs="+",
        default=[
            "artifacts/submission_*.csv",
            "artifacts/experiments/*/submission.csv",
            "artifacts/experiments/*/*.csv"
        ],
        help="List of submission CSV files (supports glob patterns)"
    )
    parser.add_argument(
        "--ground-truth", "-g",
        default="data/train_QA.csv",
        help="Path to ground truth CSV"
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
    
    # Load Ground Truth
    try:
        gt_df = load_ground_truth(Path(args.ground_truth))
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)

    # Initialize Master DataFrame with GT
    # Select relevant GT columns
    cols_to_keep = ["question", "answer_value", "ref_id", "explanation"] 
    # Add other metadata columns if they exist
    for c in ["answer_unit", "question_type"]: 
        if c in gt_df.columns:
            cols_to_keep.append(c)
            
    master_df = gt_df[cols_to_keep].copy()
    master_df = master_df.rename(columns={
        "answer_value": "GT_Value",
        "ref_id": "GT_Ref",
        "explanation": "GT_Explanation"
    })
    
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
