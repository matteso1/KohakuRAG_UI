"""
Aggregation Config - Ensemble of 3 Small Models

Aggregates results from Scout, Maverick, and Mistral Small.
Usage:
    cd KohakuRAG && kogine run scripts/wattbot_aggregate.py --config ../configs/ensemble_small.py
"""

from kohakuengine import Config

# Input files (outputs from individual model runs)
inputs = [
    "artifacts/submission_llama4_scout.csv",
    "artifacts/submission_llama4_maverick.csv",
    "artifacts/submission_mistral_small.csv",
]

# Output file
output = "artifacts/submission_ensemble_small.csv"

# Aggregation strategy
ref_mode = "union"      # Union all references found by models that agree on the answer
tiebreak = "first"      # If no majority, pick the first model's answer (Scout)
ignore_blank = True     # If a model returns "is_blank" but others have an answer, ignore the blank


def config_gen():
    return Config.from_globals()
