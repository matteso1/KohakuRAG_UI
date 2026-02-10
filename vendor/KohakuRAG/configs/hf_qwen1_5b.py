"""
WattBot Evaluation Config - Qwen 2.5 1.5B Instruct (Local HF)

Qwen 2.5 1.5B is a lightweight model that can run on limited GPU VRAM (~4GB bf16).
Good as a fast baseline or for testing the pipeline quickly.

Usage:
    python scripts/run_experiment.py --config vendor/KohakuRAG/configs/hf_qwen1_5b.py
    python scripts/run_wattbot_eval.py --config vendor/KohakuRAG/configs/hf_qwen1_5b.py
"""

# Database settings
db = "../../data/embeddings/wattbot_jinav4.db"
table_prefix = "wattbot_jv4"

# Input/output
questions = "../../data/train_QA.csv"
output = "../../artifacts/submission_qwen1_5b.csv"
metadata = "../../data/metadata.csv"

# LLM settings - Qwen 2.5 1.5B Instruct (local)
llm_provider = "hf_local"
hf_model_id = "Qwen/Qwen2.5-1.5B-Instruct"
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
max_concurrent = 4  # Small model can handle more concurrency
