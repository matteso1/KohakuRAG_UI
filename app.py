"""
WattBot RAG — Streamlit App

Interactive UI for querying the WattBot RAG pipeline.
Supports both local HuggingFace models (GPU) and AWS Bedrock models (API).

Launch (Bedrock — no GPU required):
    streamlit run app.py -- --mode bedrock

Launch (Local — requires GPU):
    streamlit run app.py -- --mode local

If no --mode is given, defaults to "bedrock".
When a GPU is detected, the sidebar offers a toggle to switch modes at runtime.
"""

import argparse
import asyncio
import csv
import gc
import importlib.util
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import time
import traceback
from collections import Counter
from pathlib import Path

import streamlit as st

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# CLI argument parsing (works with `streamlit run app.py -- --mode bedrock`)
# ---------------------------------------------------------------------------
def _parse_run_mode() -> str:
    """Parse --mode from CLI args. Returns 'bedrock' or 'local'."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--mode", choices=["bedrock", "local"], default="bedrock")
    args, _ = parser.parse_known_args()
    return args.mode

_CLI_MODE = _parse_run_mode()

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_repo_root = Path(__file__).resolve().parent
sys.path.insert(0, str(_repo_root / "vendor" / "KohakuRAG" / "src"))
sys.path.insert(0, str(_repo_root / "scripts"))

from kohakurag import RAGPipeline
from kohakurag.datastore import KVaultNodeStore
from kohakurag.pipeline import LLMQueryPlanner, SimpleQueryPlanner

# Conditional imports — local-mode deps may not be installed in bedrock envs
_HAS_LOCAL_DEPS = False
try:
    from kohakurag.embeddings import JinaV4EmbeddingModel
    from kohakurag.llm import HuggingFaceLocalChatModel
    _HAS_LOCAL_DEPS = True
except ImportError:
    JinaV4EmbeddingModel = None  # type: ignore[assignment,misc]
    HuggingFaceLocalChatModel = None  # type: ignore[assignment,misc]

_HAS_BEDROCK_DEPS = False
try:
    from llm_bedrock import BedrockChatModel, BedrockEmbeddingModel
    _HAS_BEDROCK_DEPS = True
except ImportError:
    BedrockChatModel = None  # type: ignore[assignment,misc]
    BedrockEmbeddingModel = None  # type: ignore[assignment,misc]

# ---------------------------------------------------------------------------
# Prompts (shared with run_experiment.py)
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """
You must answer strictly based on the provided context snippets.
Do NOT use external knowledge or assumptions.
If the context does not clearly support an answer, you must output the literal string "is_blank" for both answer_value and ref_id.
For True/False questions, you MUST output "1" for True and "0" for False in answer_value. Do NOT output the words "True" or "False".
""".strip()

SYSTEM_PROMPT_BEST_GUESS = """
You must answer based on the provided context snippets.
If the context strongly supports an answer, answer normally.
If the context only partially or weakly supports an answer, still provide your best guess but set confidence to "low".
Set confidence to "high" when the context clearly and directly answers the question.
For True/False questions, you MUST output "1" for True and "0" for False in answer_value. Do NOT output the words "True" or "False".
""".strip()

USER_TEMPLATE = """
You will be given a question and context snippets taken from documents.
You must follow these rules:
- Use only the provided context; do not rely on external knowledge.
- If the context does not clearly support an answer, use "is_blank" for all fields except explanation.
- For unanswerable questions, set answer to "Unable to answer with confidence based on the provided documents."
- For True/False questions: answer_value must be "1" for True or "0" for False (not the words "True" or "False").

Question: {question}

Context:
{context}

Return STRICT JSON with the following keys, in this order:
- explanation          (1-3 sentences that directly answer the question. Cite sources by ref_id, e.g. "According to [wu2021a], ...". Do NOT use vague phrases like "the context states" or "the passage mentions".)
- answer               (short natural-language response, e.g. "1438 lbs", "Water consumption", "TRUE")
- answer_value         (ONLY the numeric or categorical value, e.g. "1438", "Water consumption", "1"; or "is_blank")
- ref_id               (list of document ids from the context used as evidence; or "is_blank")
- ref_url              (list of URLs for the cited documents; or "is_blank")
- supporting_materials (verbatim quote, table reference, or figure reference from the cited document; or "is_blank")

JSON Answer:
""".strip()

USER_TEMPLATE_BEST_GUESS = """
You will be given a question and context snippets taken from documents.
You must follow these rules:
- Use the provided context as your primary source.
- If the context clearly answers the question, answer normally with confidence "high".
- If the context only partially relates, provide your best-effort answer with confidence "low".
- For True/False questions: answer_value must be "1" for True or "0" for False (not the words "True" or "False").

Question: {question}

Context:
{context}

Return STRICT JSON with the following keys, in this order:
- explanation          (1-3 sentences that directly answer the question. Cite sources by ref_id, e.g. "According to [wu2021a], ...". Do NOT use vague phrases like "the context states" or "the passage mentions".)
- answer               (short natural-language response, e.g. "1438 lbs", "Water consumption", "TRUE")
- answer_value         (ONLY the numeric or categorical value, e.g. "1438", "Water consumption", "1"; or "is_blank")
- confidence           ("high" if the context clearly supports the answer, "low" if this is a best guess)
- ref_id               (list of document ids from the context used as evidence; or "is_blank")
- ref_url              (list of URLs for the cited documents; or "is_blank")
- supporting_materials (verbatim quote, table reference, or figure reference from the cited document; or "is_blank")

JSON Answer:
""".strip()


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CONFIGS_DIR = _repo_root / "vendor" / "KohakuRAG" / "configs"

# Approximate 4-bit NF4 VRAM (GB) per config. Used for planning only.
VRAM_4BIT_GB = {
    "hf_qwen1_5b": 2, "hf_qwen3b": 3, "hf_qwen7b": 6, "hf_qwen14b": 10,
    "hf_qwen32b": 20, "hf_qwen72b": 40, "hf_llama3_8b": 6, "hf_gemma2_9b": 7,
    "hf_gemma2_27b": 17, "hf_mixtral_8x7b": 26, "hf_mixtral_8x22b": 80,
    "hf_mistral7b": 6, "hf_phi3_mini": 3, "hf_qwen3_30b_a3b": 18,
    "hf_qwen3_next_80b_a3b": 40, "hf_qwen3_next_80b_a3b_thinking": 40,
    "hf_olmoe_1b7b": 4, "hf_qwen1_5_110b": 60,
}
EMBEDDER_OVERHEAD_GB = 3  # Jina V4 embedder + store + misc
PRECISION_MULTIPLIER = {"4bit": 1.0, "bf16": 4.0, "fp16": 4.0, "auto": 4.0}

# Approximate Bedrock pricing ($/M tokens).  Keyed by model-ID prefix so
# versioned IDs like "...v2:0" still match.  Prices are for cross-region
# inference profiles (us.*) as of 2025-Q2.
BEDROCK_PRICING: dict[str, tuple[float, float]] = {
    # (input $/M tokens, output $/M tokens)
    "us.anthropic.claude-3-haiku":       (0.25,  1.25),
    "us.anthropic.claude-3-5-haiku":     (0.80,  4.00),
    "us.anthropic.claude-3-5-sonnet":    (3.00, 15.00),
    "us.anthropic.claude-3-7-sonnet":    (3.00, 15.00),
    "us.meta.llama3-3-70b":              (0.72,  0.72),
    "us.meta.llama4-scout":              (0.17,  0.17),
    "us.meta.llama4-maverick":           (0.20,  0.60),
    "us.amazon.nova-pro":                (0.80,  3.20),
    "us.deepseek.r1":                    (1.35,  5.40),
}


def _get_bedrock_cost(model_id: str, input_tokens: int, output_tokens: int) -> float | None:
    """Estimate USD cost for a Bedrock call. Returns None if pricing unknown."""
    for prefix, (inp_price, out_price) in BEDROCK_PRICING.items():
        if model_id.startswith(prefix):
            return (input_tokens * inp_price + output_tokens * out_price) / 1_000_000
    return None


def _get_pipeline_token_usage(pipeline: "RAGPipeline") -> tuple[int, int, str]:
    """Extract token usage from a pipeline's chat model.

    Returns (input_tokens, output_tokens, model_id).
    """
    chat = pipeline._chat
    if hasattr(chat, "token_usage"):
        return chat.token_usage.input_tokens, chat.token_usage.output_tokens, getattr(chat, "_model_id", "")
    return 0, 0, ""


def _reset_pipeline_token_usage(pipeline: "RAGPipeline") -> None:
    """Reset accumulated token counts before a new query."""
    chat = pipeline._chat
    if hasattr(chat, "token_usage"):
        chat.token_usage.reset()

# ---------------------------------------------------------------------------
# Metadata URL lookup  (ref_id → URL from metadata.csv)
# ---------------------------------------------------------------------------
_METADATA_CSV = _repo_root / "data" / "metadata.csv"

def _load_metadata_urls() -> dict[str, str]:
    """Build a ref_id → url mapping from metadata.csv."""
    mapping: dict[str, str] = {}
    if not _METADATA_CSV.exists():
        return mapping
    with open(_METADATA_CSV, newline="", encoding="utf-8", errors="replace") as f:
        for row in csv.DictReader(f):
            doc_id = row.get("id", "").strip()
            url = row.get("url", "").strip()
            if doc_id and url:
                mapping[doc_id] = url
    return mapping

METADATA_URLS: dict[str, str] = _load_metadata_urls()


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------
def _debug(msg: str) -> None:
    """Print debug info to terminal and, if debug mode is on, to the Streamlit UI."""
    logger.info(msg)
    print(f"[DEBUG] {msg}", flush=True)


def discover_configs(provider: str = "local") -> dict[str, Path]:
    """Find config files and return {display_name: path}.

    Args:
        provider: "local" for hf_*.py, "bedrock" for bedrock_*.py, "all" for both.
    """
    if provider == "bedrock":
        return {p.stem: p for p in sorted(CONFIGS_DIR.glob("bedrock_*.py"))}
    elif provider == "all":
        hf = {p.stem: p for p in sorted(CONFIGS_DIR.glob("hf_*.py"))}
        br = {p.stem: p for p in sorted(CONFIGS_DIR.glob("bedrock_*.py"))}
        return {**hf, **br}
    else:
        return {p.stem: p for p in sorted(CONFIGS_DIR.glob("hf_*.py"))}


def load_config(config_path: Path) -> dict:
    """Load a Python config file into a dict."""
    spec = importlib.util.spec_from_file_location("config", config_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config = {}
    for key in [
        "db", "table_prefix", "questions", "output", "metadata",
        "llm_provider", "top_k", "planner_max_queries", "deduplicate_retrieval",
        "rerank_strategy", "top_k_final", "retrieval_threshold",
        "max_retries", "max_concurrent",
        "embedding_model", "embedding_dim", "embedding_task", "embedding_model_id",
        "hf_model_id", "hf_dtype", "hf_max_new_tokens", "hf_temperature",
        # Bedrock-specific keys
        "bedrock_model", "bedrock_region", "bedrock_profile",
        "bedrock_embedding_model",
    ]:
        if hasattr(module, key):
            config[key] = getattr(module, key)
    return config


def estimate_vram(config_name: str, precision: str) -> float:
    """Estimate VRAM (GB) for a model at given precision."""
    base = VRAM_4BIT_GB.get(config_name, 8)
    return base * PRECISION_MULTIPLIER.get(precision, 1.0)


# ---------------------------------------------------------------------------
# GPU detection
# ---------------------------------------------------------------------------
def get_gpu_info() -> dict:
    """Detect GPU count, names, and free VRAM per GPU."""
    try:
        import torch
        if not torch.cuda.is_available():
            return {"gpu_count": 0, "gpus": [], "total_free_gb": 0}
    except ImportError:
        return {"gpu_count": 0, "gpus": [], "total_free_gb": 0}

    gpus = []
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        free, total = torch.cuda.mem_get_info(i)
        gpus.append({
            "index": i,
            "name": props.name,
            "total_gb": total / (1024**3),
            "free_gb": free / (1024**3),
        })
    total_free = sum(g["free_gb"] for g in gpus)
    return {"gpu_count": len(gpus), "gpus": gpus, "total_free_gb": total_free}


def plan_ensemble(config_names: list[str], precision: str, gpu_info: dict) -> dict:
    """Decide parallel vs sequential execution based on available VRAM.

    Returns:
        {"mode": "parallel"|"sequential"|"error", "model_vrams": [...], ...}
    """
    model_vrams = [estimate_vram(n, precision) for n in config_names]
    total_needed = sum(model_vrams) + EMBEDDER_OVERHEAD_GB
    total_free = gpu_info["total_free_gb"]

    if total_free == 0:
        return {"mode": "error", "model_vrams": model_vrams,
                "reason": "No GPU detected"}

    max_single_gpu = max(g["free_gb"] for g in gpu_info["gpus"])
    largest_model = max(model_vrams)

    if largest_model + EMBEDDER_OVERHEAD_GB > max_single_gpu:
        return {"mode": "error", "model_vrams": model_vrams,
                "reason": (f"Largest model needs ~{largest_model + EMBEDDER_OVERHEAD_GB:.0f} GB "
                           f"but largest GPU only has {max_single_gpu:.0f} GB free")}

    if total_needed <= total_free:
        return {"mode": "parallel", "model_vrams": model_vrams}
    return {"mode": "sequential", "model_vrams": model_vrams}


# ---------------------------------------------------------------------------
# Pipeline init
# ---------------------------------------------------------------------------
def _load_shared_resources(config: dict) -> tuple[JinaV4EmbeddingModel, KVaultNodeStore]:
    """Load embedder and vector store from config."""
    embedding_dim = config.get("embedding_dim", 1024)
    embedding_task = config.get("embedding_task", "retrieval")
    db_raw = config.get("db", "data/embeddings/wattbot_jinav4.db")
    db_path = _repo_root / db_raw.removeprefix("../").removeprefix("../")
    table_prefix = config.get("table_prefix", "wattbot_jv4")

    _debug(
        f"Loading shared resources:\n"
        f"  db_path       = {db_path} (exists={db_path.exists()})\n"
        f"  table_prefix  = {table_prefix}\n"
        f"  embedding_dim = {embedding_dim}\n"
        f"  embedding_task= {embedding_task}"
    )

    embedder = JinaV4EmbeddingModel(
        task=embedding_task,
        truncate_dim=embedding_dim,
    )
    _debug(f"Embedder loaded: dimension={embedder.dimension}")

    store = KVaultNodeStore(
        db_path,
        table_prefix=table_prefix,
        dimensions=embedding_dim,
        paragraph_search_mode="averaged",
    )
    _debug(
        f"Store opened: dimensions={store._dimensions}, "
        f"vec_count={store._vectors.info().get('count', '?')}"
    )
    return embedder, store


def _load_chat_model(config: dict, precision: str) -> HuggingFaceLocalChatModel:
    """Create a HuggingFaceLocalChatModel from config."""
    return HuggingFaceLocalChatModel(
        model=config.get("hf_model_id", "Qwen/Qwen2.5-7B-Instruct"),
        system_prompt=SYSTEM_PROMPT,
        dtype=precision,
        max_new_tokens=config.get("hf_max_new_tokens", 512),
        temperature=config.get("hf_temperature", 0.2),
        max_concurrent=config.get("max_concurrent", 2),
    )


def _unload_chat_model(chat_model: HuggingFaceLocalChatModel) -> None:
    """Free GPU memory from a loaded model."""
    import torch
    if hasattr(chat_model, "_model"):
        del chat_model._model
    if hasattr(chat_model, "_tokenizer"):
        del chat_model._tokenizer
    del chat_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _apply_planner(
    pipeline: RAGPipeline, use_planner: bool, planner_queries: int,
) -> None:
    """Configure the query planner on an already-built pipeline.

    Swaps the lightweight planner object without reloading model weights.
    """
    if use_planner:
        pipeline._planner = LLMQueryPlanner(
            pipeline._chat, max_queries=planner_queries,
        )
    else:
        pipeline._planner = SimpleQueryPlanner()


@st.cache_resource(show_spinner="Loading model and vector store...")
def init_single_pipeline(config_name: str, precision: str) -> RAGPipeline:
    """Load a single-model pipeline. Cached across reruns."""
    config = load_config(CONFIGS_DIR / f"{config_name}.py")
    embedder, store = _load_shared_resources(config)
    chat_model = _load_chat_model(config, precision)
    return RAGPipeline(store=store, embedder=embedder, chat_model=chat_model, planner=None)


@st.cache_resource(show_spinner="Loading ensemble models...")
def init_ensemble_parallel(config_names: tuple[str, ...], precision: str) -> dict[str, RAGPipeline]:
    """Load all ensemble models into memory (parallel mode). Cached."""
    # Use first config for shared resources (db/embedder are the same across configs)
    ref_config = load_config(CONFIGS_DIR / f"{config_names[0]}.py")
    embedder, store = _load_shared_resources(ref_config)

    pipelines = {}
    for name in config_names:
        config = load_config(CONFIGS_DIR / f"{name}.py")
        chat_model = _load_chat_model(config, precision)
        pipelines[name] = RAGPipeline(
            store=store, embedder=embedder, chat_model=chat_model, planner=None,
        )
    return pipelines


@st.cache_resource(show_spinner="Loading embedder and vector store...")
def init_shared_only() -> tuple:
    """Load only the embedder + store (for sequential ensemble). Cached."""
    ref_config = load_config(next(CONFIGS_DIR.glob("hf_*.py")))
    return _load_shared_resources(ref_config)


# ---------------------------------------------------------------------------
# Bedrock pipeline init
# ---------------------------------------------------------------------------
def _load_bedrock_shared_resources(
    config: dict, aws_profile: str | None = None, aws_region: str = "us-east-2",
) -> tuple:
    """Load embedder and vector store for a Bedrock config."""
    embedding_dim = config.get("embedding_dim", 1024)
    db_raw = config.get("db", "data/embeddings/wattbot_titan_v2.db")
    db_path = _repo_root / db_raw.removeprefix("../").removeprefix("../")
    table_prefix = config.get("table_prefix", "wattbot_tv2")

    emb_model = config.get("embedding_model", "bedrock")
    if emb_model == "bedrock":
        embedder = BedrockEmbeddingModel(
            model_id=config.get("bedrock_embedding_model", "amazon.titan-embed-text-v2:0"),
            profile_name=aws_profile,
            region_name=aws_region,
            dimensions=embedding_dim,
        )
    else:
        # Bedrock LLM but local Jina V4 embeddings (GPU server scenario)
        embedder = JinaV4EmbeddingModel(
            task=config.get("embedding_task", "retrieval"),
            truncate_dim=embedding_dim,
        )

    _debug(
        f"Loading Bedrock shared resources:\n"
        f"  db_path       = {db_path} (exists={db_path.exists()})\n"
        f"  table_prefix  = {table_prefix}\n"
        f"  embedding_dim = {embedding_dim}\n"
        f"  embedding     = {emb_model}"
    )

    store = KVaultNodeStore(
        db_path,
        table_prefix=table_prefix,
        dimensions=embedding_dim,
        paragraph_search_mode="averaged",
    )
    return embedder, store


def _load_bedrock_chat_model(
    config: dict, aws_profile: str | None = None, aws_region: str = "us-east-2",
) -> "BedrockChatModel":
    """Create a BedrockChatModel from config."""
    return BedrockChatModel(
        model_id=config.get("bedrock_model", "us.anthropic.claude-3-haiku-20240307-v1:0"),
        profile_name=aws_profile,
        region_name=aws_region,
        system_prompt=SYSTEM_PROMPT,
        max_tokens=config.get("hf_max_new_tokens", 4096),
        max_retries=config.get("max_retries", 5),
        max_concurrent=config.get("max_concurrent", 5),
    )


@st.cache_resource(show_spinner="Connecting to AWS Bedrock...")
def init_bedrock_pipeline(
    config_name: str, aws_profile: str | None = None, aws_region: str = "us-east-2",
) -> RAGPipeline:
    """Load a single Bedrock-backed pipeline. Cached across reruns."""
    config = load_config(CONFIGS_DIR / f"{config_name}.py")
    embedder, store = _load_bedrock_shared_resources(config, aws_profile, aws_region)
    chat_model = _load_bedrock_chat_model(config, aws_profile, aws_region)
    return RAGPipeline(store=store, embedder=embedder, chat_model=chat_model, planner=None)


@st.cache_resource(show_spinner="Connecting ensemble models to Bedrock...")
def init_bedrock_ensemble_parallel(
    config_names: tuple[str, ...],
    aws_profile: str | None = None,
    aws_region: str = "us-east-2",
) -> dict[str, RAGPipeline]:
    """Load all Bedrock ensemble models. Cached."""
    ref_config = load_config(CONFIGS_DIR / f"{config_names[0]}.py")
    embedder, store = _load_bedrock_shared_resources(ref_config, aws_profile, aws_region)

    pipelines = {}
    for name in config_names:
        config = load_config(CONFIGS_DIR / f"{name}.py")
        chat_model = _load_bedrock_chat_model(config, aws_profile, aws_region)
        pipelines[name] = RAGPipeline(
            store=store, embedder=embedder, chat_model=chat_model, planner=None,
        )
    return pipelines


@st.cache_resource(show_spinner="Loading Bedrock embedder and vector store...")
def init_bedrock_shared_only(
    aws_profile: str | None = None, aws_region: str = "us-east-2",
) -> tuple:
    """Load only embedder + store for sequential Bedrock ensemble. Cached."""
    ref_config = load_config(next(CONFIGS_DIR.glob("bedrock_*.py")))
    return _load_bedrock_shared_resources(ref_config, aws_profile, aws_region)


def run_bedrock_ensemble_sequential_query(
    config_names: list[str],
    question: str,
    top_k: int,
    aws_profile: str | None = None,
    aws_region: str = "us-east-2",
    progress_callback=None,
    best_guess: bool = False,
    max_retries: int = 0,
    use_planner: bool = False,
    planner_queries: int = 3,
) -> dict[str, object]:
    """Query Bedrock models one at a time (sequential ensemble)."""
    embedder, store = init_bedrock_shared_only(aws_profile, aws_region)
    results = {}

    for i, name in enumerate(config_names):
        if progress_callback:
            progress_callback(i, len(config_names), name)

        config = load_config(CONFIGS_DIR / f"{name}.py")
        chat_model = _load_bedrock_chat_model(config, aws_profile, aws_region)
        pipeline = RAGPipeline(
            store=store, embedder=embedder, chat_model=chat_model, planner=None,
        )
        _apply_planner(pipeline, use_planner, planner_queries)

        _reset_pipeline_token_usage(pipeline)
        t0 = time.time()
        result = _run_qa_sync(
            pipeline, question, top_k,
            best_guess=best_guess, max_retries=max_retries,
        )
        elapsed = time.time() - t0
        inp_tok, out_tok, mid = _get_pipeline_token_usage(pipeline)
        cost = _get_bedrock_cost(mid, inp_tok, out_tok)
        results[name] = {
            "result": result, "time": elapsed,
            "cost_info": {"input_tokens": inp_tok, "output_tokens": out_tok,
                          "model_id": mid, "cost": cost},
        }

    return results


# ---------------------------------------------------------------------------
# Query helpers
# ---------------------------------------------------------------------------
def _run_qa_sync(
    pipeline: RAGPipeline,
    question: str,
    top_k: int,
    best_guess: bool = False,
    max_retries: int = 0,
):
    """Run pipeline.run_qa synchronously, retrying on failures.

    Args:
        max_retries: Number of additional attempts after the first failure.
                     0 means no retries (single attempt).
    """
    sys_prompt = SYSTEM_PROMPT_BEST_GUESS if best_guess else SYSTEM_PROMPT
    usr_template = USER_TEMPLATE_BEST_GUESS if best_guess else USER_TEMPLATE
    last_exc: Exception | None = None
    for attempt in range(max_retries + 1):
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(
                pipeline.run_qa(
                    question,
                    system_prompt=sys_prompt,
                    user_template=usr_template,
                    top_k=top_k,
                )
            )
        except Exception as exc:
            last_exc = exc
            _debug(f"Attempt {attempt + 1}/{max_retries + 1} failed: {exc}")
            if attempt < max_retries:
                time.sleep(1)  # brief pause before retry
        finally:
            loop.close()
    raise last_exc  # type: ignore[misc]


def run_single_query(
    pipeline: RAGPipeline, question: str, top_k: int,
    best_guess: bool = False, max_retries: int = 0,
):
    """Run a single model query."""
    return _run_qa_sync(
        pipeline, question, top_k,
        best_guess=best_guess, max_retries=max_retries,
    )


def run_ensemble_parallel_query(
    pipelines: dict[str, RAGPipeline], question: str, top_k: int,
    best_guess: bool = False, max_retries: int = 0,
) -> dict[str, object]:
    """Query all pre-loaded models concurrently."""
    results = {}
    for name, pipeline in pipelines.items():
        t0 = time.time()
        result = _run_qa_sync(
            pipeline, question, top_k,
            best_guess=best_guess, max_retries=max_retries,
        )
        results[name] = {"result": result, "time": time.time() - t0}
    return results


def run_ensemble_sequential_query(
    config_names: list[str],
    precision: str,
    question: str,
    top_k: int,
    progress_callback=None,
    best_guess: bool = False,
    max_retries: int = 0,
    use_planner: bool = False,
    planner_queries: int = 3,
) -> dict[str, object]:
    """Load each model one at a time, query, unload. Saves VRAM."""
    embedder, store = init_shared_only()
    results = {}

    for i, name in enumerate(config_names):
        if progress_callback:
            progress_callback(i, len(config_names), name)

        config = load_config(CONFIGS_DIR / f"{name}.py")
        chat_model = _load_chat_model(config, precision)
        pipeline = RAGPipeline(
            store=store, embedder=embedder, chat_model=chat_model, planner=None,
        )
        _apply_planner(pipeline, use_planner, planner_queries)

        t0 = time.time()
        result = _run_qa_sync(
            pipeline, question, top_k,
            best_guess=best_guess, max_retries=max_retries,
        )
        elapsed = time.time() - t0
        results[name] = {"result": result, "time": elapsed}

        # Free model memory before loading next
        _unload_chat_model(chat_model)
        del pipeline

    return results


# ---------------------------------------------------------------------------
# Ensemble aggregation
# ---------------------------------------------------------------------------
def aggregate_majority(answers: list[str]) -> str:
    """Most common answer. Ties go to first occurrence."""
    valid = [a for a in answers if a and a.strip() and a != "is_blank"]
    if not valid:
        return "is_blank"
    return Counter(valid).most_common(1)[0][0]


def aggregate_first_non_blank(answers: list[str]) -> str:
    """First non-blank answer in model order."""
    for a in answers:
        if a and a.strip() and a != "is_blank":
            return a
    return "is_blank"


def aggregate_refs(ref_lists: list) -> list[str]:
    """Union of all reference IDs across models."""
    all_refs = set()
    for refs in ref_lists:
        if isinstance(refs, list):
            all_refs.update(r for r in refs if r and r != "is_blank")
        elif isinstance(refs, str) and refs != "is_blank":
            try:
                parsed = json.loads(refs.replace("'", '"'))
                all_refs.update(parsed)
            except (json.JSONDecodeError, TypeError):
                all_refs.add(refs)
    return sorted(all_refs) if all_refs else []


def build_ensemble_answer(
    model_results: dict[str, object], strategy: str,
) -> dict:
    """Aggregate individual model results into an ensemble answer."""
    answers = []
    values = []
    explanations = []
    ref_lists = []
    ref_url_lists = []

    for name, entry in model_results.items():
        ans = entry["result"].answer
        answers.append(ans.answer)
        values.append(ans.answer_value)
        explanations.append(ans.explanation)
        ref_lists.append(ans.ref_id)
        ref_url_lists.append(ans.ref_url)

    agg_fn = aggregate_majority if strategy == "majority" else aggregate_first_non_blank

    best_answer = agg_fn(answers)
    best_value = agg_fn(values)
    best_explanation = agg_fn(explanations)

    # Scope refs to runs that agree with the winning answer
    winning_refs = [r for a, r in zip(answers, ref_lists) if a == best_answer]
    winning_ref_urls = [r for a, r in zip(answers, ref_url_lists) if a == best_answer]

    return {
        "answer": best_answer,
        "answer_value": best_value,
        "explanation": best_explanation,
        "ref_id": aggregate_refs(winning_refs),
        "ref_url": aggregate_refs(winning_ref_urls),
        "individual": {
            name: {
                "answer": entry["result"].answer.answer,
                "answer_value": entry["result"].answer.answer_value,
                "explanation": entry["result"].answer.explanation,
                "ref_id": entry["result"].answer.ref_id,
                "time": entry["time"],
                "raw_response": entry["result"].raw_response,
            }
            for name, entry in model_results.items()
        },
    }


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------
def _detect_gpu_available() -> bool:
    """Return True if at least one CUDA GPU is available."""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def _check_aws_credentials(
    profile_name: str | None, region_name: str,
) -> tuple[bool, str]:
    """Verify AWS credentials are valid before using Bedrock.

    Returns (ok, message). If *ok* is False the message explains the
    problem and what command to run.
    """
    try:
        import boto3
        session = boto3.Session(
            profile_name=profile_name,
            region_name=region_name,
        )
        sts = session.client("sts")
        identity = sts.get_caller_identity()
        account = identity.get("Account", "")
        return True, f"AWS OK (account {account})"
    except Exception as exc:
        msg = str(exc)
        if "Token has expired" in msg or "refresh failed" in msg:
            return False, "AWS SSO session expired — please re-login."
        if "NoCredentialProviders" in msg or "Unable to locate credentials" in msg:
            if profile_name:
                return False, (
                    f"No valid credentials for profile **{profile_name}**. "
                    "Please login first."
                )
            return False, (
                "No AWS credentials found. Set the **AWS profile** field above, "
                "or configure environment variables."
            )
        if "InvalidClientTokenId" in msg or "SignatureDoesNotMatch" in msg:
            return False, "AWS credentials are invalid. Re-configure your SSO profile."
        return False, f"AWS credential check failed: {msg}"


def _run_sso_login(profile_name: str | None) -> tuple[bool, str]:
    """Run ``aws sso login`` and return (success, output).

    Opens a browser for authentication.  Blocks until the user completes
    the flow or the 5-minute timeout expires.
    """
    if not shutil.which("aws"):
        return False, (
            "The `aws` CLI is not installed or not on PATH.\n"
            "Install it from https://docs.aws.amazon.com/cli/latest/userguide/"
            "getting-started-install.html"
        )
    cmd = ["aws", "sso", "login"]
    if profile_name:
        cmd += ["--profile", profile_name]
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=300,
        )
        output = (proc.stdout + "\n" + proc.stderr).strip()
        if proc.returncode == 0:
            return True, output
        return False, output or "Login process exited with an error."
    except subprocess.TimeoutExpired:
        return False, "Login timed out after 5 minutes. Please try again."
    except Exception as exc:
        return False, str(exc)


def main():
    st.set_page_config(page_title="WattBot RAG", page_icon="lightning", layout="wide")
    st.title("WattBot RAG Pipeline")

    # Determine effective run mode (CLI default + optional sidebar toggle)
    gpu_available = _detect_gpu_available()

    # ---- Sidebar ----
    with st.sidebar:
        st.header("Settings")

        # --- Provider mode selection ---
        # Show toggle only when both backends are possible
        can_toggle = gpu_available and _HAS_LOCAL_DEPS and _HAS_BEDROCK_DEPS
        if can_toggle:
            use_local = st.toggle(
                "Use local GPU models",
                value=(_CLI_MODE == "local"),
                help=(
                    "GPU detected on this machine. Toggle ON to use local "
                    "HuggingFace models; toggle OFF for AWS Bedrock API models."
                ),
            )
            provider = "local" if use_local else "bedrock"
        elif _CLI_MODE == "local" and _HAS_LOCAL_DEPS:
            provider = "local"
        else:
            provider = "bedrock"

        is_bedrock = provider == "bedrock"

        if is_bedrock:
            st.caption("Backend: **AWS Bedrock** (API)")
        else:
            st.caption("Backend: **Local HuggingFace** (GPU)")

        # --- Setup Bedrock section (bedrock mode only) ---
        if is_bedrock:
            st.divider()
            st.subheader("Setup Bedrock")
            if not _HAS_BEDROCK_DEPS:
                st.error(
                    "Bedrock dependencies not installed. Run:\n\n"
                    "```\npip install -r bedrock_requirements.txt\n```"
                )
                return
            aws_profile = st.text_input(
                "AWS profile",
                value=os.environ.get("AWS_PROFILE", ""),
                help=(
                    "Your AWS SSO profile name (e.g. bedrock_yourname). "
                    "Leave blank to use environment variables or instance role."
                ),
            )
            aws_region = st.text_input(
                "AWS region",
                value=os.environ.get("AWS_REGION", "us-east-2"),
                help="AWS region where Bedrock models are enabled.",
            )
            # Normalize empty string to None for boto3
            aws_profile = aws_profile.strip() or None

            # --- Validate AWS credentials upfront ---
            cred_ok, cred_msg = _check_aws_credentials(aws_profile, aws_region)
            if not cred_ok:
                st.error(cred_msg)
                if st.button("Login to AWS SSO"):
                    with st.spinner(
                        "Waiting for browser authentication... "
                        "Complete the login in your browser."
                    ):
                        ok, output = _run_sso_login(aws_profile)
                    if ok:
                        st.success("Logged in successfully!")
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error(output)
                return
            else:
                st.caption(cred_msg)

        st.divider()
        mode = st.radio("Mode", ["Single model", "Ensemble"], horizontal=True)

        # --- Config discovery + model selection (right after Mode) ---
        configs = discover_configs(provider)
        if not configs:
            prefix = "bedrock_*" if is_bedrock else "hf_*"
            st.error(f"No {prefix}.py config files found in vendor/KohakuRAG/configs/")
            return

        config_list = list(configs.keys())

        if mode == "Single model":
            if is_bedrock:
                default_key = "bedrock_deepseek_v3"
            else:
                default_key = "hf_qwen7b"
            default_idx = config_list.index(default_key) if default_key in config_list else 0
            selected_config = st.selectbox("Model config", config_list, index=default_idx)
            selected_configs = [selected_config]
            ensemble_strategy = None
        else:
            if is_bedrock:
                default_models = ["bedrock_deepseek_v3", "bedrock_sonnet"]
            else:
                default_models = ["hf_qwen7b", "hf_llama3_8b"]
            selected_configs = st.multiselect(
                "Ensemble models (pick 2+)", config_list,
                default=[m for m in default_models if m in config_list] or config_list[:2],
            )
            ensemble_strategy = st.selectbox(
                "Aggregation", ["majority", "first_non_blank"],
            )

        # Precision only matters for local models
        precision = "4bit"
        if not is_bedrock:
            precision = st.selectbox("Precision", ["4bit", "bf16", "fp16", "auto"], index=0)

        st.divider()
        top_k = st.slider("Retrieved chunks (top_k)", min_value=1, max_value=20, value=8)
        best_guess = st.toggle("Allow best-guess answers", value=False,
                               help="When enabled, out-of-scope questions get a best-effort answer labelled as a guess.")

        st.divider()
        st.subheader("Query planner & retries")
        use_planner = st.toggle(
            "Enable query planner", value=False,
            help=(
                "Expands each question into multiple diverse search queries "
                "via the LLM for better retrieval coverage."
            ),
        )
        planner_queries = 3
        if use_planner:
            planner_queries = st.slider(
                "Planner queries", min_value=2, max_value=10, value=3,
                help="Number of diverse search queries the LLM generates per question.",
            )
        max_retries = st.number_input(
            "Max retries", min_value=0, max_value=10, value=2,
            help="Maximum retry attempts when the LLM response cannot be parsed.",
        )
        st.caption(
            "Tip: Disabling the query planner skips an extra LLM inference "
            "call, and lowering retries caps worst-case wait time. Both "
            "reduce end-to-end latency per question."
        )

        # --- GPU / VRAM info (local mode only) ---
        if not is_bedrock:
            st.divider()
            gpu_info = get_gpu_info()
            if gpu_info["gpu_count"] > 0:
                st.caption(f"**{gpu_info['gpu_count']} GPU(s)** detected")
                for g in gpu_info["gpus"]:
                    st.caption(f"  GPU {g['index']}: {g['name']} — "
                               f"{g['free_gb']:.1f} / {g['total_gb']:.1f} GB free")
            else:
                st.caption("No GPU detected")

            # Ensemble VRAM plan
            if mode == "Ensemble" and len(selected_configs) >= 2:
                plan = plan_ensemble(selected_configs, precision, gpu_info)
                vram_list = [f"{n}: ~{v:.0f}GB" for n, v in
                             zip(selected_configs, plan["model_vrams"])]
                st.caption(f"VRAM: {', '.join(vram_list)}")
                if plan["mode"] == "parallel":
                    st.caption("Strategy: **parallel** (all models in memory)")
                elif plan["mode"] == "sequential":
                    st.caption("Strategy: **sequential** (load one at a time)")
                else:
                    st.warning(plan["reason"])
        else:
            # Bedrock ensemble always runs parallel (API-based, no VRAM constraint)
            gpu_info = {"gpu_count": 0, "gpus": [], "total_free_gb": 0}
            if mode == "Ensemble" and len(selected_configs) >= 2:
                st.divider()
                st.caption("Strategy: **parallel** (API mode — no GPU needed)")

    # ---- Validate ensemble selection ----
    if mode == "Ensemble" and len(selected_configs) < 2:
        st.info("Select at least 2 models for ensemble mode.")
        return

    # ---- Load pipelines ----
    try:
        if is_bedrock:
            # Bedrock pipelines
            if mode == "Single model":
                pipeline = init_bedrock_pipeline(
                    selected_configs[0], aws_profile, aws_region,
                )
                _apply_planner(pipeline, use_planner, planner_queries)
            elif mode == "Ensemble":
                # Bedrock ensembles always run parallel (no VRAM constraint)
                ensemble_pipelines = init_bedrock_ensemble_parallel(
                    tuple(selected_configs), aws_profile, aws_region,
                )
                for _p in ensemble_pipelines.values():
                    _apply_planner(_p, use_planner, planner_queries)
        else:
            # Local pipelines
            if mode == "Single model":
                pipeline = init_single_pipeline(selected_configs[0], precision)
                _apply_planner(pipeline, use_planner, planner_queries)
            elif mode == "Ensemble":
                plan = plan_ensemble(selected_configs, precision, gpu_info)
                if plan["mode"] == "error":
                    st.error(f"Cannot run ensemble: {plan['reason']}")
                    return
                if plan["mode"] == "parallel":
                    ensemble_pipelines = init_ensemble_parallel(
                        tuple(selected_configs), precision,
                    )
                    for _p in ensemble_pipelines.values():
                        _apply_planner(_p, use_planner, planner_queries)
                # sequential doesn't pre-load models (planner set inside query fn)
    except Exception as e:
        st.error(f"Failed to load pipeline: {e}")
        tb = traceback.format_exc()
        _debug(f"Load error:\n{tb}")
        with st.expander("Full traceback"):
            st.code(tb, language="python")
        return

    # ---- Chat interface ----
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Render history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if msg["role"] == "assistant":
                details = msg.get("details", {})
                linked = _linkify_citations(
                    msg["content"],
                    ref_ids=details.get("ref_id"),
                    ref_urls=details.get("ref_url"),
                )
                st.markdown(f"**{linked}**")
            else:
                st.markdown(msg["content"])
            if msg["role"] == "assistant" and "details" in msg:
                _render_details(msg["details"])

    # User input
    if question := st.chat_input("Ask a question about the WattBot documents..."):
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            t0 = time.time()

            if mode == "Single model":
                spinner_msg = (
                    "Querying Bedrock..." if is_bedrock
                    else "Retrieving and generating..."
                )
                if is_bedrock:
                    _reset_pipeline_token_usage(pipeline)
                with st.spinner(spinner_msg):
                    try:
                        result = run_single_query(
                            pipeline, question, top_k,
                            best_guess=best_guess, max_retries=max_retries,
                        )
                    except Exception as e:
                        st.error(f"Pipeline error: {e}")
                        tb = traceback.format_exc()
                        _debug(f"Pipeline error:\n{tb}")
                        with st.expander("Full traceback"):
                            st.code(tb, language="python")
                        return
                elapsed = time.time() - t0
                cost_info = None
                if is_bedrock:
                    inp_tok, out_tok, model_id = _get_pipeline_token_usage(pipeline)
                    cost = _get_bedrock_cost(model_id, inp_tok, out_tok)
                    cost_info = {"input_tokens": inp_tok, "output_tokens": out_tok,
                                 "model_id": model_id, "cost": cost}
                _display_single_result(result, elapsed, cost_info=cost_info)

            else:  # Ensemble
                try:
                    is_sequential = (
                        not is_bedrock
                        and plan["mode"] == "sequential"
                    )
                    if is_bedrock and not is_sequential:
                        # Reset token counters on all ensemble pipelines
                        for _p in ensemble_pipelines.values():
                            _reset_pipeline_token_usage(_p)
                    if not is_sequential:
                        with st.spinner(
                            f"Querying {len(selected_configs)} models in parallel..."
                        ):
                            model_results = run_ensemble_parallel_query(
                                ensemble_pipelines, question, top_k,
                                best_guess=best_guess,
                                max_retries=max_retries,
                            )
                    else:
                        status = st.status(
                            f"Querying {len(selected_configs)} models sequentially...",
                            expanded=True,
                        )
                        def _progress(i, total, name):
                            status.update(label=f"[{i+1}/{total}] Loading {name}...")
                        model_results = run_ensemble_sequential_query(
                            selected_configs, precision, question, top_k,
                            progress_callback=_progress,
                            best_guess=best_guess,
                            max_retries=max_retries,
                            use_planner=use_planner,
                            planner_queries=planner_queries,
                        )
                        status.update(label="Aggregating results...", state="complete")
                except Exception as e:
                    st.error(f"Ensemble error: {e}")
                    tb = traceback.format_exc()
                    _debug(f"Ensemble error:\n{tb}")
                    with st.expander("Full traceback"):
                        st.code(tb, language="python")
                    return

                elapsed = time.time() - t0
                # Collect per-model cost info for bedrock ensembles
                ensemble_cost_info = None
                if is_bedrock and not is_sequential:
                    ensemble_cost_info = {}
                    for name, _p in ensemble_pipelines.items():
                        inp_tok, out_tok, mid = _get_pipeline_token_usage(_p)
                        cost = _get_bedrock_cost(mid, inp_tok, out_tok)
                        ensemble_cost_info[name] = {
                            "input_tokens": inp_tok, "output_tokens": out_tok,
                            "model_id": mid, "cost": cost,
                        }
                agg = build_ensemble_answer(model_results, ensemble_strategy)
                _display_ensemble_result(
                    agg, model_results, elapsed, ensemble_strategy,
                    cost_info=ensemble_cost_info,
                )


def _extract_confidence(raw_response: str) -> str:
    """Extract confidence field from raw JSON or bullet-list response."""
    # Try JSON first
    try:
        start = raw_response.index("{")
        end = raw_response.rindex("}") + 1
        data = json.loads(raw_response[start:end])
        return str(data.get("confidence", "")).strip().lower()
    except Exception:
        pass
    # Fallback: bullet-list format (- confidence   high/low)
    m = re.search(r"-\s*confidence\s{2,}(\S+)", raw_response)
    if m:
        return m.group(1).strip().strip('"').lower()
    return ""


def _humanize_ref_id(rid: str) -> str:
    """Convert a ref_id like ``luccioni2025c`` to ``Luccioni et al., 2025``.

    Expects the common ``<surname><4-digit-year>[suffix]`` pattern.
    Falls back to the raw id if the pattern doesn't match.
    """
    m = re.match(r"([a-zA-Z]+)(\d{4})", rid)
    if m:
        author = m.group(1).capitalize()
        year = m.group(2)
        return f"{author} et al., {year}"
    return rid


def _linkify_citations(
    text: str,
    ref_ids=None,
    ref_urls=None,
) -> str:
    """Replace ``[ref_id]`` citations in *text* with clickable markdown links.

    * Converts raw ids to human-readable labels (``Luccioni et al., 2025``).
    * Inserts comma separators between adjacent citations so they don't
      render as a single run-on string.
    * Looks up each ``[...]`` token against METADATA_URLS (primary) and the
      answer's own ref_url list (fallback).  Already-linked references
      (``[id](url)``) are left untouched.
    """
    if not text:
        return text

    # Build fallback url map from the answer's own ref data
    answer_urls: dict[str, str] = {}
    if ref_ids and ref_ids != "is_blank":
        ids = ref_ids if isinstance(ref_ids, list) else [ref_ids]
        urls = ref_urls if isinstance(ref_urls, list) else ([ref_urls] if ref_urls else [])
        for i, rid in enumerate(ids):
            if not METADATA_URLS.get(rid) and i < len(urls):
                u = urls[i]
                if u and u != "is_blank":
                    answer_urls[rid] = u

    def _replace(match: re.Match) -> str:
        rid = match.group(1)
        url = METADATA_URLS.get(rid) or answer_urls.get(rid)
        label = _humanize_ref_id(rid)
        if url:
            return f"[{label}]({url})"
        # No URL — still humanize if it looks like a ref_id
        if label != rid:
            return f"({label})"
        return match.group(0)

    # Match [something] NOT already followed by '(' (avoids double-linking)
    text = re.sub(r"\[([^\]]+)\](?!\()", _replace, text)

    # Insert ", " between adjacent markdown links: ...](url)[... → ...](url), [...
    text = re.sub(r"\]\(([^)]+)\)\[", r"](\1), [", text)

    return text


def _display_single_result(result, elapsed: float, cost_info: dict | None = None):
    """Display a single-model answer."""
    answer = result.answer
    timing = result.timing
    confidence = _extract_confidence(result.raw_response)

    # Linkify inline [ref_id] citations so they match the Sources section
    linked_explanation = _linkify_citations(
        answer.explanation, ref_ids=answer.ref_id, ref_urls=answer.ref_url,
    )

    if linked_explanation and linked_explanation != "is_blank":
        st.markdown(f"**{linked_explanation}**")
        if confidence == "low":
            st.warning("Best guess — the retrieved context only partially supports this answer.")
    elif answer.answer and answer.answer != "is_blank":
        st.markdown(f"**{answer.answer}**")
    else:
        st.markdown("**Out-of-scope** — the provided documents do not contain enough information to answer this question.")
    if answer.answer_value and answer.answer_value != "is_blank":
        st.markdown(f"Value: `{answer.answer_value}`")

    # Clickable reference links (shown directly, not inside an expander)
    ref_ids = answer.ref_id
    ref_urls = answer.ref_url
    if ref_ids and ref_ids != "is_blank":
        links = []
        for i, rid in enumerate(ref_ids if isinstance(ref_ids, list) else [ref_ids]):
            url = METADATA_URLS.get(rid)
            if not url:
                url = ref_urls[i] if isinstance(ref_urls, list) and i < len(ref_urls) else None
            label = _humanize_ref_id(rid)
            if url and url != "is_blank":
                links.append(f"[{label}]({url})")
            else:
                links.append(label)
        st.markdown("Sources: " + " · ".join(links))

    details = {
        "timing": timing,
        "elapsed": elapsed,
        "ref_id": answer.ref_id,
        "ref_url": answer.ref_url,
        "supporting_materials": answer.supporting_materials,
        "snippets": [
            {"rank": s.rank, "score": s.score, "title": s.document_title, "text": s.text}
            for s in result.retrieval.snippets
        ],
        "raw_response": result.raw_response,
    }
    if cost_info:
        details["cost_info"] = cost_info
    _render_details(details)

    if answer.explanation and answer.explanation != "is_blank":
        display_answer = answer.explanation
    elif answer.answer and answer.answer != "is_blank":
        display_answer = answer.answer
    else:
        display_answer = "Out-of-scope"
    st.session_state.messages.append({
        "role": "assistant", "content": display_answer, "details": details,
    })


def _display_ensemble_result(
    agg: dict, model_results: dict, elapsed: float, strategy: str,
    cost_info: dict | None = None,
):
    """Display aggregated ensemble answer + per-model breakdown."""
    linked_explanation = _linkify_citations(
        agg["explanation"], ref_ids=agg.get("ref_id"), ref_urls=agg.get("ref_url"),
    )

    if linked_explanation and linked_explanation != "is_blank":
        st.markdown(f"**{linked_explanation}**")
    elif agg["answer"] and agg["answer"] != "is_blank":
        st.markdown(f"**{agg['answer']}**")
    else:
        st.markdown("**Out-of-scope** — the provided documents do not contain enough information to answer this question.")
    if agg["answer_value"] and agg["answer_value"] != "is_blank":
        st.markdown(f"Value: `{agg['answer_value']}`")

    n_models = len(model_results)
    model_times = [e["time"] for e in model_results.values()]
    total_gen = sum(model_times)

    # Merge cost_info from sequential results into the top-level dict
    if cost_info is None:
        cost_info = {}
        for name, entry in model_results.items():
            if "cost_info" in entry:
                cost_info[name] = entry["cost_info"]

    # Compute total estimated cost across all models
    total_cost = None
    if cost_info:
        costs = [c["cost"] for c in cost_info.values() if c.get("cost") is not None]
        if costs:
            total_cost = sum(costs)

    if total_cost is not None:
        cols = st.columns(4)
        cols[0].metric("Models", n_models)
        cols[1].metric("Aggregation", strategy)
        cols[2].metric("Total", f"{elapsed:.1f}s")
        cols[3].metric("Est. cost", f"${total_cost:.4f}")
    else:
        cols = st.columns(3)
        cols[0].metric("Models", n_models)
        cols[1].metric("Aggregation", strategy)
        cols[2].metric("Total", f"{elapsed:.1f}s")

    # Per-model answers
    with st.expander(f"Individual model answers ({n_models} models)"):
        for name, info in agg["individual"].items():
            agreed = info["answer_value"] == agg["answer_value"]
            marker = "+" if agreed else "-"
            val = info["answer_value"] if info["answer_value"] and info["answer_value"] != "is_blank" else "Out-of-scope"
            ans = info["answer"] if info["answer"] and info["answer"] != "is_blank" else "Out-of-scope"
            cost_str = ""
            if name in cost_info and cost_info[name].get("cost") is not None:
                cost_str = f" · ${cost_info[name]['cost']:.4f}"
            st.markdown(
                f"**{name}** ({info['time']:.1f}s{cost_str}) [{marker}]  \n"
                f"Answer: `{val}` — {ans}"
            )
            if info["explanation"] and info["explanation"] != "is_blank":
                st.caption(_linkify_citations(
                    info["explanation"], ref_ids=info.get("ref_id"),
                ))
            st.divider()

    # Clickable reference links
    if agg["ref_id"]:
        links = []
        for rid in agg["ref_id"]:
            url = METADATA_URLS.get(rid)
            label = _humanize_ref_id(rid)
            if url:
                links.append(f"[{label}]({url})")
            else:
                links.append(label)
        st.markdown("Sources: " + " · ".join(links))

    # First model's retrieval context (shared across models since same embedder+store)
    first_result = next(iter(model_results.values()))["result"]
    snippets = first_result.retrieval.snippets
    if snippets:
        display_snippets = snippets[:10]
        label = f"Retrieved context ({len(display_snippets)} of {len(snippets)} chunks)"
        with st.expander(label):
            for s in display_snippets:
                st.markdown(f"**#{s.rank}** _{s.document_title}_ (score: {s.score:.3f})")
                st.text(s.text[:500] + ("..." if len(s.text) > 500 else ""))
                st.divider()

    # Raw responses per model
    with st.expander("Raw LLM responses"):
        for name, info in agg["individual"].items():
            st.markdown(f"**{name}**")
            st.code(info["raw_response"], language="json")

    details = {
        "elapsed": elapsed,
        "ensemble": True,
        "strategy": strategy,
        "models": list(model_results.keys()),
        "answer": agg["answer"],
        "answer_value": agg["answer_value"],
    }
    if total_cost is not None:
        details["total_cost"] = total_cost
    if agg["explanation"] and agg["explanation"] != "is_blank":
        display_answer = agg["explanation"]
    elif agg["answer"] and agg["answer"] != "is_blank":
        display_answer = agg["answer"]
    else:
        display_answer = "Out-of-scope"
    st.session_state.messages.append({
        "role": "assistant", "content": display_answer, "details": details,
    })


def _render_details(details: dict):
    """Render expandable sections for a stored message (history replay)."""
    if details.get("ensemble"):
        # Minimal replay for ensemble messages
        total_cost = details.get("total_cost")
        if total_cost is not None:
            cols = st.columns(4)
            cols[0].metric("Models", len(details.get("models", [])))
            cols[1].metric("Aggregation", details.get("strategy", ""))
            cols[2].metric("Total", f"{details.get('elapsed', 0):.1f}s")
            cols[3].metric("Est. cost", f"${total_cost:.4f}")
        else:
            cols = st.columns(3)
            cols[0].metric("Models", len(details.get("models", [])))
            cols[1].metric("Aggregation", details.get("strategy", ""))
            cols[2].metric("Total", f"{details.get('elapsed', 0):.1f}s")
        return

    timing = details.get("timing", {})
    elapsed = details.get("elapsed", 0)
    cost_info = details.get("cost_info")

    if cost_info and cost_info.get("cost") is not None:
        cols = st.columns(4)
        cols[0].metric("Retrieval", f"{timing.get('retrieval_s', 0):.1f}s")
        cols[1].metric("Generation", f"{timing.get('generation_s', 0):.1f}s")
        cols[2].metric("Total", f"{elapsed:.1f}s")
        cols[3].metric("Est. cost", f"${cost_info['cost']:.4f}")
    else:
        cols = st.columns(3)
        cols[0].metric("Retrieval", f"{timing.get('retrieval_s', 0):.1f}s")
        cols[1].metric("Generation", f"{timing.get('generation_s', 0):.1f}s")
        cols[2].metric("Total", f"{elapsed:.1f}s")

    ref_ids = details.get("ref_id", [])
    ref_urls = details.get("ref_url", [])
    if ref_ids and ref_ids != "is_blank":
        links = []
        for i, rid in enumerate(ref_ids if isinstance(ref_ids, list) else [ref_ids]):
            url = METADATA_URLS.get(rid)
            if not url:
                url = ref_urls[i] if isinstance(ref_urls, list) and i < len(ref_urls) else None
            label = _humanize_ref_id(rid)
            if url and url != "is_blank":
                links.append(f"[{label}]({url})")
            else:
                links.append(label)
        st.markdown("Sources: " + " · ".join(links))
        sm = details.get("supporting_materials", "")
        if sm and sm != "is_blank":
            st.caption(f"Supporting: {sm}")

    snippets = details.get("snippets", [])
    if snippets:
        display_snippets = snippets[:10]
        label = f"Retrieved context ({len(display_snippets)} of {len(snippets)} chunks)"
        with st.expander(label):
            for s in display_snippets:
                st.markdown(f"**#{s['rank']}** _{s['title']}_ (score: {s['score']:.3f})")
                st.text(s["text"][:500] + ("..." if len(s["text"]) > 500 else ""))
                st.divider()

    raw = details.get("raw_response", "")
    if raw:
        with st.expander("Raw LLM response"):
            st.code(raw, language="json")


if __name__ == "__main__":
    main()
