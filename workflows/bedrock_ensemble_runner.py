"""Bedrock Ensemble Runner: replicate winning KohakuRAG config with AWS Bedrock.

Mirrors the exact winning configuration from jinav4_ensemble_runner_test-gpt-oss-120b.py
but uses Claude 3.5 Sonnet on AWS Bedrock instead of GPT-OSS-120B on OpenRouter.

Usage:
    python workflows/bedrock_ensemble_runner.py
"""

from pathlib import Path
from typing import Any

from kohakuengine import Config, Flow, Script, capture_globals

# Shared settings - EXACTLY matching winning config
DB = "artifacts/wattbot_jinav4.db"
TABLE_PREFIX = "wattbot_jv4"
QUESTIONS = "data/train_QA.csv"  # Use train for validation
METADATA = "data/metadata.csv"

# Bedrock model - Claude 3.7 Sonnet (newest, best context handling)
BEDROCK_MODEL = "us.anthropic.claude-3-7-sonnet-20250219-v1:0"
BEDROCK_PROFILE = "bedrock_nils"
BEDROCK_REGION = "us-east-2"

OUTPUT_DIR = Path("outputs/bedrock-sonnet-ensemble")
NUM_RUNS = 5  # Same as winning solution

# Models to run in parallel
MODELS = [
    {
        "model": BEDROCK_MODEL,
        "output": (OUTPUT_DIR / f"single_preds{i}.csv").as_posix(),
    }
    for i in range(NUM_RUNS)
]

# Aggregation settings - EXACTLY matching winning config
AGGREGATED_OUTPUT = (OUTPUT_DIR / "ensemble_preds.csv").as_posix()
REF_MODE = "answer_priority"
TIEBREAK = "first"
IGNORE_BLANK = True

# Base config - EXACTLY matching winning config
with capture_globals() as ctx:
    db = DB
    table_prefix = TABLE_PREFIX
    questions = QUESTIONS
    metadata = METADATA

    # LLM settings - Bedrock instead of OpenRouter
    llm_provider = "bedrock"
    bedrock_model = BEDROCK_MODEL
    bedrock_profile = BEDROCK_PROFILE
    bedrock_region = BEDROCK_REGION
    planner_model = None

    # Retrieval settings - adapted for Bedrock context limits
    top_k = 10  # Reduced from 16 (Claude context limits)
    bm25_top_k = 4
    planner_max_queries = 4
    deduplicate_retrieval = True
    rerank_strategy = "combined"
    top_k_final = None  # No truncation
    paragraph_search_mode = "full"

    # JinaV4 settings - EXACTLY matching winning config
    embedding_model = "jinav4"
    embedding_dim = 512
    embedding_task = "retrieval"

    # Image settings - EXACTLY matching winning config
    with_images = True
    top_k_images = 2
    send_images_to_llm = False

    # Prompt ordering - EXACTLY matching winning config
    use_reordered_prompt = True

    # Other settings
    max_retries = 2
    max_concurrent = 1  # Serialize requests to avoid token limits
    single_run_debug = False
    question_id = None


def create_answer_config(cfg: dict[str, Any]) -> Config:
    """Create answer config for a specific model run."""
    base_config = Config.from_context(ctx)
    base_config.globals_dict.update(cfg)
    return base_config


if __name__ == "__main__":
    # Ensure output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Create answer scripts for each model run
    answer_scripts = [
        Script(
            "KohakuRAG/scripts/wattbot_answer.py",
            config=create_answer_config(cfg),
        )
        for cfg in MODELS
    ]

    # Run answer scripts in parallel
    print("=" * 70)
    print(f"Running {len(MODELS)} ensemble runs in parallel (Bedrock + JinaV4)...")
    print("=" * 70)

    # Run sequentially to avoid Bedrock rate limits
    # (winning solution used parallel but OpenRouter has higher limits)
    answer_flow = Flow(answer_scripts, use_subprocess=True, max_workers=1)
    answer_flow.run()

    # Aggregate results
    print("\n" + "=" * 70)
    print("Aggregating results...")
    print("=" * 70)

    aggregate_config = Config(
        globals_dict={
            "inputs": [cfg["output"] for cfg in MODELS],
            "output": AGGREGATED_OUTPUT,
            "ref_mode": REF_MODE,
            "tiebreak": TIEBREAK,
            "ignore_blank": IGNORE_BLANK,
        }
    )

    aggregate_script = Script("KohakuRAG/scripts/wattbot_aggregate.py", config=aggregate_config)
    aggregate_script.run()

    print("\n" + "=" * 70)
    print("Bedrock Ensemble Complete!")
    print("=" * 70)

    print("\nSettings (matching winning config):")
    print(f"  Model: {BEDROCK_MODEL}")
    print(f"  Num runs: {NUM_RUNS}")
    print(f"  Ref mode: {REF_MODE}")
    print(f"  Tiebreak: {TIEBREAK}")
    print(f"  LLM provider: bedrock")
    print(f"  Planner max queries: {planner_max_queries}")
    print(f"  Top k: {top_k}")
    print(f"  BM25 top k: {bm25_top_k}")
    print(f"  Top k final: {top_k_final}")
    print(f"  Rerank strategy: {rerank_strategy}")
    print(f"  Deduplicate: {deduplicate_retrieval}")
    print(f"  Embedding: {embedding_model} (dim={embedding_dim})")
    print(f"  Top k images: {top_k_images}")
    print(f"  Send images to LLM: {send_images_to_llm}")
    print(f"  Reordered prompt: {use_reordered_prompt}")
    print(f"  Max retries: {max_retries}")
    print(f"  Ignore blank: {IGNORE_BLANK}")
    print(f"\n  Aggregated output: {AGGREGATED_OUTPUT}")
    print("=" * 70)
