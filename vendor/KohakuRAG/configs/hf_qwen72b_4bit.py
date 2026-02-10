"""
WattBot Evaluation Config - Qwen 2.5 72B Instruct (4-bit Quantized)

4-bit quantized variant of Qwen 2.5 72B. Fits in ~40GB VRAM so it works
on 1x A100 or 2x A6000 without needing 4+ GPUs.

Requires bitsandbytes: uv pip install bitsandbytes

Usage:
    python scripts/run_experiment.py --config vendor/KohakuRAG/configs/hf_qwen72b_4bit.py
"""

# Database settings
db = "../../data/embeddings/wattbot_jinav4.db"
table_prefix = "wattbot_jv4"

# Input/output
questions = "../../data/train_QA.csv"
output = "../../artifacts/submission_qwen72b_4bit.csv"
metadata = "../../data/metadata.csv"

# LLM settings - Qwen 2.5 72B Instruct (4-bit quantized)
llm_provider = "hf_local"
hf_model_id = "Qwen/Qwen2.5-72B-Instruct"
hf_dtype = "4bit"
hf_max_new_tokens = 512
hf_temperature = 0.2

# Embedding settings (local HF)
embedding_model = "hf_local"
embedding_model_id = "BAAI/bge-base-en-v1.5"

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
max_concurrent = 1  # Very large model - conservative concurrency
