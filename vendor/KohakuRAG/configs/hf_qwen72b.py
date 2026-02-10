"""
WattBot Evaluation Config - Qwen 2.5 72B Instruct (Local HF)

Qwen 2.5 72B is the largest dense model in the Qwen 2.5 family.
In bf16 it requires ~140GB VRAM, so it needs multi-GPU (e.g., 2x A100-80GB
or 4x A6000-48GB). With 4-bit quantization it fits in ~40GB (1x A100 or
2x A6000).

device_map="auto" handles multi-GPU sharding automatically.

Usage:
    # bf16 (needs ~140GB total VRAM)
    python scripts/run_experiment.py --config vendor/KohakuRAG/configs/hf_qwen72b.py

    # 4-bit quantized (needs ~40GB total VRAM)
    python scripts/run_experiment.py --config vendor/KohakuRAG/configs/hf_qwen72b_4bit.py
"""

# Database settings
db = "../../data/embeddings/wattbot_jinav4.db"
table_prefix = "wattbot_jv4"

# Input/output
questions = "../../data/train_QA.csv"
output = "../../artifacts/submission_qwen72b.csv"
metadata = "../../data/metadata.csv"

# LLM settings - Qwen 2.5 72B Instruct (local)
llm_provider = "hf_local"
hf_model_id = "Qwen/Qwen2.5-72B-Instruct"
hf_dtype = "bf16"
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
