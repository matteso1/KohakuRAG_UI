"""
WattBot Evaluation Config - Claude 3.7 Sonnet

Claude 3.7 Sonnet is the newest Sonnet model (as of early 2025).
Expected to be better than 3.5 Sonnet on reasoning tasks.

Usage:
    python scripts/run_experiment.py --config configs/bedrock_claude37_sonnet.py --name claude37-sonnet-v1
"""

from kohakuengine import Config

# Database settings
db = "../artifacts/wattbot_jinav4.db"
table_prefix = "wattbot_jv4"

# Input/output
questions = "../data/train_QA.csv"
output = "../artifacts/submission_claude37_sonnet.csv"
metadata = "../data/metadata.csv"

# LLM settings - Claude 3.7 Sonnet
llm_provider = "bedrock"
bedrock_profile = "bedrock_nils"
bedrock_region = "us-east-2"
bedrock_model = "us.anthropic.claude-3-7-sonnet-20250219-v1:0"

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
max_concurrent = 2


def config_gen():
    return Config.from_globals()
