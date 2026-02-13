"""
WattBot Evaluation Config - Amazon Nova Pro

Amazon Nova Pro - Mid-tier model with good general capabilities.
Part of the ensemble tests.

Usage:
    python scripts/run_experiment.py --config configs/bedrock_nova_pro.py --name nova-pro-v1
"""

from kohakuengine import Config

# Database settings
db = "../artifacts/wattbot_jinav4.db"
table_prefix = "wattbot_jv4"

# Input/output
questions = "../data/train_QA.csv"
output = "../artifacts/submission_nova_pro.csv"
metadata = "../data/metadata.csv"

# LLM settings - Amazon Nova Pro
llm_provider = "bedrock"
bedrock_profile = "bedrock_nils"
bedrock_region = "us-east-2"
bedrock_model = "us.amazon.nova-pro-v1:0"

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
