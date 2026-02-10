"""
WattBot Evaluation Config - DeepSeek-V3.1

DeepSeek-V3.1 is a 685B parameter model with hybrid reasoning capabilities.
Can toggle "thinking" mode on/off for different use cases.

NOTE: Verify the exact model ID in AWS Bedrock console before use.
The model ID pattern may be: us.deepseek.deepseek-v3-1-v1:0

Key features:
- Extended thinking mode for complex reasoning
- May help with NA detection (knowing what it doesn't know)
- Higher cost per query, use strategically

Usage:
    cd KohakuRAG && kogine run scripts/wattbot_answer.py --config ../configs/bedrock_deepseek_v3.py
"""

from kohakuengine import Config

# Database settings
db = "../artifacts/wattbot_jinav4.db"
table_prefix = "wattbot_jv4"

# Input/output
questions = "../data/train_QA.csv"
output = "../artifacts/submission_deepseek_v3.csv"
metadata = "../data/metadata.csv"

# LLM settings - DeepSeek-V3.1
# NOTE: Verify exact model ID in AWS console
llm_provider = "bedrock"
bedrock_profile = "bedrock_nils"
bedrock_region = "us-east-2"
bedrock_model = "us.deepseek.r1-v1:0"

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

# Other settings - lower concurrency for large model
max_retries = 3
max_concurrent = 1


def config_gen():
    return Config.from_globals()
