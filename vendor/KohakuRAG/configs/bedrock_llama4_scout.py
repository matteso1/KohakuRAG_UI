"""
WattBot Evaluation Config - Meta Llama 4 Scout via AWS Bedrock

Meta's Llama 4 Scout model, cost-effective with strong performance.
~$0.17/M input tokens, $0.17/M output tokens.

Usage:
    python scripts/run_experiment.py --config vendor/KohakuRAG/configs/bedrock_llama4_scout.py
"""

# Database settings
db = "../../data/embeddings/wattbot_jinav4.db"
table_prefix = "wattbot_jv4"

# Input/output
questions = "../../data/train_QA.csv"
output = "../../artifacts/submission_bedrock_llama4_scout.csv"
metadata = "../../data/metadata.csv"

# LLM settings - Llama 4 Scout via Bedrock
llm_provider = "bedrock"
bedrock_model = "us.meta.llama4-scout-17b-instruct-v1:0"
bedrock_region = "us-east-2"
# bedrock_profile = "your-sso-profile"  # Uncomment and set your AWS SSO profile

# Embedding settings (Jina v4 - must match index)
embedding_model = "jinav4"
embedding_dim = 1024
embedding_task = "retrieval"

# Retrieval settings (same as local configs for fair comparison)
top_k = 8
planner_max_queries = 4
deduplicate_retrieval = True
rerank_strategy = "combined"
top_k_final = 10
use_reordered_prompt = True  # C→Q: place context before question

# Unanswerable detection
retrieval_threshold = 0.25

# Other settings
max_retries = 3
max_concurrent = 5
