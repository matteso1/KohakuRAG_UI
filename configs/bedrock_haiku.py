"""
WattBot Evaluation Config - Claude 3 Haiku (Baseline)

Claude 3 Haiku is a ~8B parameter model optimized for speed and cost.
This is our current baseline model.

Known limitations (from analysis):
- Hallucinates on truly unanswerable questions (q062, q164)
- Over-triggers is_blank on True/False questions (q075, q200, q230, q231)
- Some reasoning errors on complex questions

Usage:
    cd KohakuRAG && kogine run scripts/wattbot_answer.py --config ../configs/bedrock_haiku.py
"""

from kohakuengine import Config

# Database settings
db = "../artifacts/wattbot_jinav4.db"
table_prefix = "wattbot_jv4"

# Input/output
questions = "../data/train_QA.csv"
output = "../artifacts/submission_haiku.csv"
metadata = "../data/metadata.csv"

# LLM settings - Claude 3 Haiku (baseline)
llm_provider = "bedrock"
bedrock_profile = "bedrock_nils"
bedrock_region = "us-east-2"
bedrock_model = "us.anthropic.claude-3-haiku-20240307-v1:0"

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
max_retries = 2
max_concurrent = 5


def config_gen():
    return Config.from_globals()
