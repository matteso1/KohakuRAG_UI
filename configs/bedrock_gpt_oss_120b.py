"""
WattBot Evaluation Config - OpenAI GPT-OSS 120B

OpenAI's open-weight 120B model, available on AWS Bedrock since Aug 2025.
Optimized for production, general-purpose, and high-reasoning tasks. 128K context.
Pricing: ~$0.0006/1K output tokens (roughly half Claude 3 Haiku).

Usage:
    cd KohakuRAG && kogine run scripts/wattbot_answer.py --config ../configs/bedrock_gpt_oss_120b.py
"""

from kohakuengine import Config

# Database settings
db = "../artifacts/wattbot_jinav4.db"
table_prefix = "wattbot_jv4"

# Input/output
questions = "../data/train_QA.csv"
output = "../artifacts/submission_gpt_oss_120b.csv"
metadata = "../data/metadata.csv"

# LLM settings - OpenAI GPT-OSS 120B
llm_provider = "bedrock"
bedrock_profile = "bedrock_nils"
bedrock_region = "us-east-2"
bedrock_model = "openai.gpt-oss-120b-1:0"

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
max_concurrent = 3


def config_gen():
    return Config.from_globals()
