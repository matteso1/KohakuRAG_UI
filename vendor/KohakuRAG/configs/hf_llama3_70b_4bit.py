"""
WattBot Evaluation Config - Llama 3.1 70B Instruct (4-bit Quantized)

Meta's Llama 3.1 70B with 4-bit quantization. Fits in ~40GB VRAM
(1x A100 or 2x A6000). Full bf16 would need ~140GB.

Requires bitsandbytes: uv pip install bitsandbytes

Usage:
    python scripts/run_experiment.py --config vendor/KohakuRAG/configs/hf_llama3_70b_4bit.py
"""

# Database settings
db = "../../data/embeddings/wattbot_jinav4.db"
table_prefix = "wattbot_jv4"

# Input/output
questions = "../../data/train_QA.csv"
output = "../../artifacts/submission_llama3_70b_4bit.csv"
metadata = "../../data/metadata.csv"

# LLM settings - Llama 3.1 70B Instruct (4-bit quantized)
llm_provider = "hf_local"
hf_model_id = "meta-llama/Llama-3.1-70B-Instruct"
hf_dtype = "4bit"
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
max_concurrent = 1  # Very large model - conservative concurrency
