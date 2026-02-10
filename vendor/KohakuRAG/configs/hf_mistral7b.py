"""
WattBot Evaluation Config - Mistral 7B Instruct v0.3 (Local HF)

Mistral 7B Instruct v0.3. Runs on a single GPU with ~16GB VRAM (bf16).

Usage:
    python scripts/run_experiment.py --config vendor/KohakuRAG/configs/hf_mistral7b.py
"""

# Database settings
db = "../../data/embeddings/wattbot_jinav4.db"
table_prefix = "wattbot_jv4"

# Input/output
questions = "../../data/train_QA.csv"
output = "../../artifacts/submission_mistral7b.csv"
metadata = "../../data/metadata.csv"

# LLM settings - Mistral 7B Instruct (local)
llm_provider = "hf_local"
hf_model_id = "mistralai/Mistral-7B-Instruct-v0.3"
hf_dtype = "bf16"
hf_max_new_tokens = 512
hf_temperature = 0.2

# Embedding settings (Jina v4 - must match index)
embedding_model = "jinav4"
embedding_dim = 1024
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
max_concurrent = 2
