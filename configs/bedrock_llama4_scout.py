"""
WattBot Evaluation Config - Meta Llama 4 Scout 17B

Meta Llama 4 Scout is a 17B parameter model - good mid-size option.
Closest to the meeting's 20-30B parameter target for cost/quality tradeoff.

NOTE: Verify the exact model ID in AWS Bedrock console before use.
The model ID pattern may be: us.meta.llama4-scout-17b-instruct-v1:0

Usage:
    cd KohakuRAG && kogine run scripts/wattbot_answer.py --config ../configs/bedrock_llama4_scout.py
"""

from kohakuengine import Config

# Database settings
db = "../artifacts/wattbot_jinav4.db"
table_prefix = "wattbot_jv4"

# Input/output
questions = "../data/train_QA.csv"
output = "../artifacts/submission_llama4_scout.csv"
metadata = "../data/metadata.csv"

# LLM settings - Meta Llama 4 Scout 17B
# MUST use "us." prefix for cross-region inference
llm_provider = "bedrock"
bedrock_profile = "bedrock_nils"
bedrock_region = "us-east-2"
bedrock_model = "us.meta.llama4-scout-17b-instruct-v1:0"

# Embedding settings (JinaV4)
embedding_model = "jinav4"
embedding_dim = 512
embedding_task = "retrieval"

# Retrieval settings
top_k = 8
planner_max_queries = 3
deduplicate_retrieval = True
rerank_strategy = "combined"
top_k_final = 10

# Unanswerable detection
retrieval_threshold = 0.25

# Other settings
max_retries = 3
max_concurrent = 3


def config_gen():
    return Config.from_globals()
