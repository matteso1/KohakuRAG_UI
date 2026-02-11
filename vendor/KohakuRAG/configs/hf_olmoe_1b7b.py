"""
WattBot Evaluation Config - OLMoE 1B-7B Instruct (Local HF, MoE)

Allen AI's fully open Mixture-of-Experts model: 7B total parameters
with only ~1B active per token. Very lightweight (~4GB VRAM at 4-bit).

Interesting as a "small MoE vs small dense" comparison against Qwen 1.5B/3B.
Apache 2.0 license, ungated, 100% open-source (code, data, and weights).

Usage:
    python scripts/run_experiment.py --config vendor/KohakuRAG/configs/hf_olmoe_1b7b.py
"""

# Database settings
db = "../../data/embeddings/wattbot_jinav4.db"
table_prefix = "wattbot_jv4"

# Input/output
questions = "../../data/train_QA.csv"
output = "../../artifacts/submission_olmoe_1b7b.csv"
metadata = "../../data/metadata.csv"

# LLM settings - OLMoE 1B-7B Instruct (local MoE)
llm_provider = "hf_local"
hf_model_id = "allenai/OLMoE-1B-7B-0125-Instruct"
hf_max_new_tokens = 512
hf_temperature = 0.2

# Embedding settings (Jina v4 - must match index)
embedding_model = "jinav4"
embedding_dim = 1024
embedding_task = "retrieval"

# Retrieval settings
top_k = 8
planner_max_queries = 4
deduplicate_retrieval = True
rerank_strategy = "combined"
top_k_final = 10
use_reordered_prompt = True  # C→Q: place context before question

# Unanswerable detection
retrieval_threshold = 0.25

# Other settings
max_retries = 2
max_concurrent = 2
