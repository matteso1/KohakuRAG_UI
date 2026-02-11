"""
WattBot Evaluation Config - Mixtral 8x7B Instruct (Local HF)

Mistral's Mixture-of-Experts model (8 experts, 2 active per token).
46.7B total parameters but only ~13B active per inference.
Requires ~14GB VRAM (4-bit NF4) due to loading all expert weights.

Usage:
    python scripts/run_experiment.py --config vendor/KohakuRAG/configs/hf_mixtral_8x7b.py
"""

# Database settings
db = "../../data/embeddings/wattbot_jinav4.db"
table_prefix = "wattbot_jv4"

# Input/output
questions = "../../data/train_QA.csv"
output = "../../artifacts/submission_mixtral_8x7b.csv"
metadata = "../../data/metadata.csv"

# LLM settings - Mixtral 8x7B Instruct (local)
llm_provider = "hf_local"
hf_model_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
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
