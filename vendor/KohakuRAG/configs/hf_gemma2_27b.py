"""
WattBot Evaluation Config - Gemma 2 27B Instruct (Local HF)

Google's Gemma 2 27B instruction-tuned model. Requires ~55GB VRAM (bf16),
so needs multi-GPU (2x A6000) or a single 80GB GPU. device_map="auto"
handles GPU sharding.

Usage:
    python scripts/run_experiment.py --config vendor/KohakuRAG/configs/hf_gemma2_27b.py
"""

# Database settings
db = "../../artifacts/wattbot_jinav4.db"
table_prefix = "wattbot_jv4"

# Input/output
questions = "../../data/train_QA.csv"
output = "../../artifacts/submission_gemma2_27b.csv"
metadata = "../../data/metadata.csv"

# LLM settings - Gemma 2 27B Instruct (local)
llm_provider = "hf_local"
hf_model_id = "google/gemma-2-27b-it"
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
max_concurrent = 1  # Large model - conservative concurrency
