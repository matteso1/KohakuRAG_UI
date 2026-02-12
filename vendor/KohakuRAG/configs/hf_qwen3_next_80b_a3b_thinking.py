"""
WattBot Evaluation Config - Qwen3-Next 80B-A3B Thinking FP8 (Local HF, MoE)

Thinking (reasoning) variant of Qwen3-Next 80B-A3B. Same architecture
(80B total / 3B active, 512 experts) but trained with GSPO for chain-of-
thought reasoning. Wraps its reasoning in <think>...</think> before the
final answer. Pre-quantized to FP8; at 4-bit NF4 it needs ~40GB VRAM.

Outperforms Qwen3-30B-A3B-Thinking and Qwen3-32B-Thinking, and beats
Gemini-2.5-Flash-Thinking on multiple benchmarks.

Apache 2.0 license, ungated on HuggingFace.

NOTE: Requires transformers from main branch:
    pip install git+https://github.com/huggingface/transformers.git@main

CAVEAT: This model always generates <think> blocks. The extra reasoning
tokens mean hf_max_new_tokens must be large enough to fit both the
thinking trace AND the final JSON answer. If the JSON parser sees stray
braces inside the thinking content, the answer may score as empty.

Usage:
    python scripts/run_experiment.py --config vendor/KohakuRAG/configs/hf_qwen3_next_80b_a3b_thinking.py
"""

# Database settings
db = "../../data/embeddings/wattbot_jinav4.db"
table_prefix = "wattbot_jv4"

# Input/output
questions = "../../data/train_QA.csv"
output = "../../artifacts/submission_qwen3_next_80b_a3b_thinking.csv"
metadata = "../../data/metadata.csv"

# LLM settings - Qwen3-Next 80B-A3B Thinking FP8 (local MoE, reasoning)
llm_provider = "hf_local"
hf_model_id = "Qwen/Qwen3-Next-80B-A3B-Thinking-FP8"
hf_max_new_tokens = 8192  # thinking models need headroom for reasoning + answer
hf_temperature = 0.6      # model card recommends 0.6 for thinking variant

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
max_concurrent = 1
