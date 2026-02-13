"""
WattBot Evaluation Config - Amazon Nova Pro via AWS Bedrock

Amazon's own foundation model, available natively in Bedrock.
~$0.80/M input tokens, $3.20/M output tokens.

Usage:
    python scripts/run_experiment.py --config vendor/KohakuRAG/configs/bedrock_nova_pro.py
"""

# Database settings
db = "../../data/embeddings/wattbot_jinav4.db"
table_prefix = "wattbot_jv4"

# Input/output
questions = "../../data/train_QA.csv"
output = "../../artifacts/submission_bedrock_nova_pro.csv"
metadata = "../../data/metadata.csv"

# LLM settings - Amazon Nova Pro via Bedrock
llm_provider = "bedrock"
bedrock_model = "amazon.nova-pro-v1:0"
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
