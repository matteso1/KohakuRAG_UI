"""
WattBot Evaluation Config - Mixtral 8x22B Instruct (Local HF, MoE)

Mistral's large Mixture-of-Experts model: 141B total parameters with
39B active per token (8 experts, 2 active). At 4-bit NF4 quantization
it needs ~73-80GB VRAM — fits on a single 80-96GB GPU.

Apache 2.0 license, ungated on HuggingFace.

Usage:
    python scripts/run_experiment.py --config vendor/KohakuRAG/configs/hf_mixtral_8x22b.py
"""

# Database settings
db = "../../data/embeddings/wattbot_jinav4.db"
table_prefix = "wattbot_jv4"

# Input/output
questions = "../../data/train_QA.csv"
output = "../../artifacts/submission_mixtral_8x22b.csv"
metadata = "../../data/metadata.csv"

# LLM settings - Mixtral 8x22B Instruct (local MoE)
llm_provider = "hf_local"
hf_model_id = "mistralai/Mixtral-8x22B-Instruct-v0.1"
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
max_concurrent = 1  # Large MoE model - conservative concurrency
