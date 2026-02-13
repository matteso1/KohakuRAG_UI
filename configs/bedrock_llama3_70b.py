"""
WattBot Evaluation Config - Meta Llama 3.3 70B

Llama 3.3 70B is a larger model for comparison against smaller Llama 4 Scout.

Usage:
    python scripts/run_experiment.py --config configs/bedrock_llama3_70b.py --name llama3-70b-v1
"""

from kohakuengine import Config

# Database settings
db = "../artifacts/wattbot_jinav4.db"
table_prefix = "wattbot_jv4"

# Input/output
questions = "../data/train_QA.csv"
output = "../artifacts/submission_llama3_70b.csv"
metadata = "../data/metadata.csv"

# LLM settings - Meta Llama 3.3 70B
llm_provider = "bedrock"
bedrock_profile = "bedrock_nils"
bedrock_region = "us-east-2"
bedrock_model = "us.meta.llama3-3-70b-instruct-v1:0"

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

# Other settings - lower concurrency for larger model
max_retries = 3
max_concurrent = 2


def config_gen():
    return Config.from_globals()
