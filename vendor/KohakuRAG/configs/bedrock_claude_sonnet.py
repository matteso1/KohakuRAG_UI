"""
WattBot Evaluation Config - Claude 3.5 Sonnet via AWS Bedrock

Balanced quality/cost model. Strong reasoning for technical QA.
~$3.00/M input tokens, $15.00/M output tokens.

Usage:
    python scripts/run_experiment.py --config vendor/KohakuRAG/configs/bedrock_claude_sonnet.py
"""

# Database settings
db = "../../data/embeddings/wattbot_jinav4.db"
table_prefix = "wattbot_jv4"

# Input/output
questions = "../../data/train_QA.csv"
output = "../../artifacts/submission_bedrock_sonnet.csv"
metadata = "../../data/metadata.csv"

# LLM settings - Claude 3.5 Sonnet via Bedrock
llm_provider = "bedrock"
bedrock_model = "us.anthropic.claude-3-5-sonnet-20241022-v2:0"
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
