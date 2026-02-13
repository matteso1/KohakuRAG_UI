#!/usr/bin/env python3
"""
Automated Model Experiment Runner (Provider-Agnostic)

Runs experiments with different LLM providers (hf_local, openrouter, openai, bedrock)
and automatically scores results. Saves detailed outputs including raw responses,
per-question scores, and timing info.

Supports:
    - hf_local: Local HuggingFace models (Qwen, Llama, Mistral, etc.)
    - openrouter: OpenRouter API models
    - openai: OpenAI API models
    - bedrock: AWS Bedrock models (requires llm_bedrock module)

Usage:
    python scripts/run_experiment.py --config vendor/KohakuRAG/configs/hf_qwen7b.py
    python scripts/run_experiment.py --config vendor/KohakuRAG/configs/hf_qwen7b.py --name "qwen7b-test"

Output:
    - artifacts/experiments/<datafile>/<name>/results.json - Raw per-question results (un-normalised)
    - artifacts/experiments/<datafile>/<name>/summary.json - Overall metrics and timing

    Where <datafile> is the stem of the questions CSV (e.g. "train_QA", "test_solutions").

    results.json stores the raw LLM output.  To produce a normalised submission.csv
    for Kaggle, run the post-hoc processing step:

        python scripts/posthoc.py artifacts/experiments/<datafile>/<name>/results.json

    See scripts/posthoc.py for the canonical answer normalisation logic.
"""

import argparse
import asyncio
import csv
import importlib.util
import json
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

# Add paths for imports
_repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_repo_root / "vendor" / "KohakuRAG" / "src"))
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd

from kohakurag import RAGPipeline, LLMQueryPlanner
from kohakurag.datastore import KVaultNodeStore
from kohakurag.embeddings import JinaEmbeddingModel, JinaV4EmbeddingModel, LocalHFEmbeddingModel
from kohakurag.llm import HuggingFaceLocalChatModel, OpenAIChatModel, OpenRouterChatModel

# Optional: Bedrock support (only available if llm_bedrock module exists)
try:
    from llm_bedrock import BedrockChatModel
    HAS_BEDROCK = True
except ImportError:
    HAS_BEDROCK = False

from score import score as compute_wattbot_score, row_bits, is_blank
from hardware_metrics import (
    HardwareMetrics,
    GPUPowerMonitor,
    NVMLEnergyCounter,
    CPURSSMonitor,
    collect_post_experiment_metrics,
    reset_gpu_peak_stats,
)


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class QuestionResult:
    """Detailed result for a single question."""
    id: str
    question: str
    gt_value: str
    gt_unit: str
    gt_ref: str
    pred_value: str
    pred_unit: str
    pred_ref: str
    pred_explanation: str
    raw_response: str
    value_correct: bool
    ref_score: float
    na_correct: bool
    weighted_score: float
    latency_seconds: float
    retrieval_seconds: float = 0.0
    generation_seconds: float = 0.0
    error: str | None = None
    # LLM-generated evidence fields
    pred_ref_url: list = field(default_factory=list)  # URLs cited by the model
    pred_supporting_materials: str = ""  # Verbatim quote / table ref from the model
    # Full context for debugging / analysis
    rendered_prompt: str = ""  # The complete prompt sent to the LLM
    retrieved_snippets: list = field(default_factory=list)  # [{node_id, doc_title, text, score, rank}]
    num_snippets: int = 0
    retry_count: int = 0  # Number of iterative-deepening retries used (0 = answered on first attempt)


@dataclass
class ExperimentSummary:
    """Summary of an experiment run."""
    name: str
    config_path: str
    model_id: str
    llm_provider: str
    quantization: str  # "bf16", "fp16", "4bit", "auto", or "api" for API providers
    timestamp: str
    num_questions: int
    total_time_seconds: float
    avg_latency_seconds: float
    avg_retrieval_seconds: float
    avg_generation_seconds: float
    value_accuracy: float
    ref_overlap: float
    na_accuracy: float
    overall_score: float
    questions_correct: int
    questions_wrong: int
    error_count: int
    # Cost tracking (API providers only)
    input_tokens: int = 0
    output_tokens: int = 0
    estimated_cost_usd: float = 0.0
    # Hardware metrics (local models)
    hardware: dict = field(default_factory=dict)
    config_snapshot: dict = field(default_factory=dict)
    # Run environment (machine label for cross-machine comparison)
    run_environment: str = ""  # e.g. "GB10", "PowerEdge", "Bedrock-us-east-1"
    # Retry stats
    total_retries: int = 0  # Sum of retry_count across all questions
    questions_retried: int = 0  # Number of questions that needed at least 1 retry
    avg_retries: float = 0.0  # Average retry_count per question
    # Dataset info
    questions_file: str = ""  # path to the questions CSV used for this run


# API pricing per 1M tokens (for cost estimation)
API_PRICING = {
    # Bedrock - Anthropic Claude family
    "claude-3-haiku": {"input": 0.25, "output": 1.25},
    "claude-3-5-haiku": {"input": 0.80, "output": 4.00},
    "claude-3-sonnet": {"input": 3.00, "output": 15.00},
    "claude-3-5-sonnet": {"input": 3.00, "output": 15.00},
    "claude-3-7-sonnet": {"input": 3.00, "output": 15.00},
    "claude-3-opus": {"input": 15.00, "output": 75.00},
    # Bedrock - Meta Llama family
    "llama3-70b": {"input": 0.72, "output": 0.72},
    "llama4-scout": {"input": 0.17, "output": 0.17},
    "llama4-maverick": {"input": 0.49, "output": 0.49},
    # Bedrock - Amazon Nova
    "nova-pro": {"input": 0.80, "output": 3.20},
    # Bedrock - Mistral
    "mistral-small": {"input": 0.10, "output": 0.30},
    # Bedrock - DeepSeek
    "deepseek": {"input": 1.35, "output": 5.40},
    # OpenRouter / OpenAI (approximate)
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-5-nano": {"input": 0.10, "output": 0.40},
}


def estimate_cost(model_id: str, input_tokens: int, output_tokens: int) -> float:
    """Estimate cost in USD based on model and token counts."""
    model_lower = model_id.lower()
    pricing = None

    for key, prices in API_PRICING.items():
        if key in model_lower:
            pricing = prices
            break

    if pricing is None:
        return 0.0  # Unknown model or local (no API cost)

    input_cost = (input_tokens / 1_000_000) * pricing["input"]
    output_cost = (output_tokens / 1_000_000) * pricing["output"]
    return input_cost + output_cost


# =============================================================================
# Answer / Ref Normalisation  (canonical logic lives in posthoc.py)
# =============================================================================
# Imported here so that callers that historically relied on these names from
# run_experiment still work (e.g. ensemble scripts).  The experiment runner
# itself no longer calls them inline — raw model output is saved to
# results.json, and posthoc.py normalises + scores in a separate step.

from posthoc import normalize_answer_value, normalize_ref_id  # noqa: F401
from results_io import CHUNK_SIZE, load_partial_progress


# =============================================================================
# Prompts — Q→C ordering (question before context, the default)
# =============================================================================

SYSTEM_PROMPT = """
You must answer strictly based on the provided context snippets.
Do NOT use external knowledge or assumptions.
If the context does not clearly support an answer, you must output the literal string "is_blank" for both answer_value and ref_id.
For True/False questions, you MUST output "1" for True and "0" for False in answer_value. Do NOT output the words "True" or "False".
""".strip()

USER_TEMPLATE = """
You will be given a question and context snippets taken from documents.
You must follow these rules:
- Use only the provided context; do not rely on external knowledge.
- If the context does not clearly support an answer, use "is_blank" for all fields except explanation.
- For unanswerable questions, set answer to "Unable to answer with confidence based on the provided documents."
- For True/False questions: answer_value must be "1" for True or "0" for False (not the words "True" or "False").

Additional info (JSON): {additional_info_json}

Question: {question}

Context:
{context}

Return STRICT JSON with the following keys, in this order:
- explanation          (1-3 sentences explaining how the context supports the answer; or "is_blank")
- answer               (short natural-language response, e.g. "1438 lbs", "Water consumption", "TRUE")
- answer_value         (ONLY the numeric or categorical value, e.g. "1438", "Water consumption", "1"; or "is_blank")
- ref_id               (list of document ids from the context used as evidence; or "is_blank")
- ref_url              (list of URLs for the cited documents; or "is_blank")
- supporting_materials (verbatim quote, table reference, or figure reference from the cited document; or "is_blank")

CRITICAL formatting rules for answer_value:
- Write full numbers without commas or abbreviations: "2000000000" not "2B" or "2,000,000,000"
- For numeric ranges, use bracket notation: "[80,90]" not "80-90" or "80 to 90"
- Do NOT include units in answer_value — units belong in answer_unit only
- Do NOT include hedging words like "approximately", "more than", "~", etc.
- Do NOT add parenthetical abbreviations: "Compute Time Calibration Function" not "Compute Time Calibration Function (CTCF)"
- For percentages, give just the number: "4" not "4%" (the unit field carries "percent")

JSON Answer:
""".strip()


# =============================================================================
# Prompts — C→Q ordering (context BEFORE question to combat "lost in the middle")
# Placing retrieved context first yields ~80% relative improvement in some tests.
# =============================================================================

SYSTEM_PROMPT_REORDERED = """
You must answer strictly based on the provided context snippets.
Do NOT use external knowledge or assumptions.
If the context does not clearly support an answer, output "is_blank" for both answer_value and ref_id.

CRITICAL: Match your answer_value to what the question asks for:
- For "Which X..." questions expecting an identifier, return the NAME/identifier, not a numeric value.
- For numeric questions with a unit, return only the number in that unit.
- The "answer_unit" field in additional info tells you the expected format.
- For True/False questions, you MUST output "1" for True and "0" for False.
""".strip()

USER_TEMPLATE_REORDERED = """
You will answer a question using ONLY the provided context snippets.
If the context does not clearly support an answer, use "is_blank".

Context snippets from documents:
{context}

---

Now answer the following question based ONLY on the context above.

Question: {question}

Additional info (JSON): {additional_info_json}

IMPORTANT: The "answer_unit" field specifies the expected format/unit for answer_value.
- If answer_unit is a unit (e.g., "kW", "USD"), express answer_value as a number in that unit (no unit name).
- If answer_unit is "is_blank", answer_value should be the exact identifier/name from context that answers the question.
- If the answer is a numeric range, format as [lower,upper].

Return STRICT JSON with these keys in order:
- explanation          (1-3 sentences explaining how context supports the answer and how you applied answer_unit; or "is_blank")
- answer               (short natural-language response)
- answer_value         (the value matching the expected format; or "is_blank")
- ref_id               (list of document ids from context used as evidence; or "is_blank")
- ref_url              (list of URLs for the cited documents; or "is_blank")
- supporting_materials (verbatim quote, table reference, or figure reference from the cited document; or "is_blank")

JSON Answer:
""".strip()


# =============================================================================
# Config Loading
# =============================================================================

def load_config(config_path: str) -> dict:
    """Load configuration from a Python config file."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    spec = importlib.util.spec_from_file_location("config", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # Extract relevant config values
    config = {}
    config_keys = [
        # Common settings
        "db", "table_prefix", "questions", "output", "metadata",
        "llm_provider", "top_k", "planner_max_queries", "deduplicate_retrieval",
        "rerank_strategy", "top_k_final", "retrieval_threshold",
        "max_retries", "max_concurrent",
        # Prompt ordering: if True, place context BEFORE question (C→Q)
        "use_reordered_prompt",
        # Embedding settings
        "embedding_model", "embedding_dim", "embedding_task", "embedding_model_id",
        # HF local settings
        "hf_model_id", "hf_dtype", "hf_max_new_tokens", "hf_temperature",
        # OpenRouter/OpenAI settings
        "model", "openrouter_api_key", "site_url", "app_name",
        # Bedrock settings
        "bedrock_profile", "bedrock_region", "bedrock_model",
    ]

    for key in config_keys:
        if hasattr(module, key):
            config[key] = getattr(module, key)

    return config


# =============================================================================
# Provider Factory
# =============================================================================

def create_chat_model_from_config(config: dict, system_prompt: str):
    """Create a chat model based on the provider specified in config.

    Supports: hf_local, openrouter, openai, bedrock
    """
    provider = config.get("llm_provider", "openrouter")

    if provider == "hf_local":
        return HuggingFaceLocalChatModel(
            model=config.get("hf_model_id", "Qwen/Qwen2.5-7B-Instruct"),
            system_prompt=system_prompt,
            dtype=config.get("hf_dtype", "4bit"),
            max_new_tokens=config.get("hf_max_new_tokens", 512),
            temperature=config.get("hf_temperature", 0.2),
            max_concurrent=config.get("max_concurrent", 2),
        )

    elif provider == "bedrock":
        if not HAS_BEDROCK:
            raise ImportError(
                "Bedrock provider requires llm_bedrock module. "
                "Copy llm_bedrock.py from the bedrock branch or install AWS dependencies."
            )
        model_id = config.get("bedrock_model", "us.anthropic.claude-3-haiku-20240307-v1:0")
        return BedrockChatModel(
            model_id=model_id,
            profile_name=config.get("bedrock_profile", "bedrock_nils"),
            region_name=config.get("bedrock_region", "us-east-2"),
            system_prompt=system_prompt,
            max_retries=config.get("max_retries", 3),
        )

    elif provider == "openrouter":
        return OpenRouterChatModel(
            model=config.get("model", "openai/gpt-5-nano"),
            api_key=config.get("openrouter_api_key"),
            site_url=config.get("site_url"),
            app_name=config.get("app_name"),
            system_prompt=system_prompt,
            max_concurrent=config.get("max_concurrent", 5),
        )

    else:
        # Default: OpenAI
        return OpenAIChatModel(
            model=config.get("model", "gpt-4o-mini"),
            system_prompt=system_prompt,
            max_concurrent=config.get("max_concurrent", 5),
        )


def create_embedder_from_config(config: dict):
    """Create an embedding model based on config settings."""
    model_type = config.get("embedding_model", "jina")

    if model_type == "hf_local":
        model_id = config.get("embedding_model_id", "BAAI/bge-base-en-v1.5")
        return LocalHFEmbeddingModel(model_name=model_id)
    elif model_type == "jinav4":
        dim = config.get("embedding_dim", 1024)
        task = config.get("embedding_task", "retrieval")
        return JinaV4EmbeddingModel(task=task, truncate_dim=dim)
    else:
        return JinaEmbeddingModel()


def get_model_display_id(config: dict) -> str:
    """Get a human-readable model identifier from config."""
    provider = config.get("llm_provider", "openrouter")
    if provider == "hf_local":
        return config.get("hf_model_id", "unknown-hf")
    elif provider == "bedrock":
        return config.get("bedrock_model", "unknown-bedrock")
    else:
        return config.get("model", "unknown-model")


# =============================================================================
# Experiment Runner
# =============================================================================

class ExperimentRunner:
    """Runs experiments with configurable models and saves detailed results."""

    def __init__(
        self,
        config: dict,
        experiment_name: str,
        output_dir: Path,
    ):
        self.config = config
        self.experiment_name = experiment_name
        self.output_dir = output_dir

        self.pipeline: RAGPipeline | None = None
        self.chat_model = None
        self.semaphore: asyncio.Semaphore | None = None
        self.results: list[QuestionResult] = []
        self.model_load_time: float = 0.0
        self.power_monitor: GPUPowerMonitor | None = None
        self.nvml_energy: NVMLEnergyCounter | None = None
        self.cpu_monitor: CPURSSMonitor | None = None

        # Prompt ordering (C→Q vs Q→C)
        self.use_reordered_prompt: bool = config.get("use_reordered_prompt", False)

        # Retry config for iterative deepening on blank answers
        self.max_retries: int = config.get("max_retries", 3)

    async def initialize(self) -> None:
        """Initialize the RAG pipeline with config settings."""
        project_root = Path(__file__).parent.parent

        db_raw = self.config.get("db", "data/embeddings/wattbot_jinav4.db")
        db_path = project_root / db_raw.removeprefix("../").removeprefix("../")
        if not db_path.exists():
            raise FileNotFoundError(
                f"Database not found: {db_path}\n\n"
                f"The vector database must be built before running experiments.\n"
                f"Run from the repo root:\n\n"
                f"  cd vendor/KohakuRAG\n"
                f"  kogine run scripts/wattbot_build_index.py --config configs/jinav4/index.py\n\n"
                f"See docs/Benchmarking_Guide.md section 0 for full details."
            )

        model_id = get_model_display_id(self.config)
        provider = self.config.get("llm_provider", "openrouter")

        print(f"[init] Provider: {provider}")
        print(f"[init] Model: {model_id}")

        # Reset GPU peak stats before loading (for accurate VRAM measurement)
        reset_gpu_peak_stats()

        dtype = self.config.get("hf_dtype", "4bit") if provider == "hf_local" else "api"
        print(f"[init] Precision: {dtype}", flush=True)

        llm_load_start = time.time()
        self.chat_model = create_chat_model_from_config(self.config, SYSTEM_PROMPT)
        self.llm_load_time = time.time() - llm_load_start
        print(f"[init] LLM loaded in {self.llm_load_time:.1f}s", flush=True)

        embed_model_type = self.config.get("embedding_model", "jina")
        print(f"[init] Loading embedder ({embed_model_type})...", flush=True)
        embed_load_start = time.time()
        embedder = create_embedder_from_config(self.config)
        self.embedder_load_time = time.time() - embed_load_start
        print(f"[init] Embedder loaded in {self.embedder_load_time:.1f}s", flush=True)

        self.model_load_time = self.llm_load_time + self.embedder_load_time

        table_prefix = self.config.get("table_prefix", "wattbot_jv4")
        print(f"[init] Loading vector store from {db_path} (prefix: {table_prefix})...")
        store = KVaultNodeStore(
            db_path,
            table_prefix=table_prefix,
            dimensions=None,
            paragraph_search_mode="averaged",
        )

        # Query planning: expand each question into multiple diverse retrieval
        # queries (default 3) to capture different terminologies / sub-questions.
        max_queries = self.config.get("planner_max_queries", 3)
        planner = LLMQueryPlanner(self.chat_model, max_queries=max_queries)
        print(f"[init] Query planner: LLMQueryPlanner (max_queries={max_queries})")
        prompt_order = "C→Q (reordered)" if self.use_reordered_prompt else "Q→C (default)"
        print(f"[init] Prompt ordering: {prompt_order}")
        print(f"[init] Max retries (iterative deepening): {self.max_retries}")

        print("[init] Building RAG pipeline...")
        self.pipeline = RAGPipeline(
            store=store,
            embedder=embedder,
            chat_model=self.chat_model,
            planner=planner,
            deduplicate_retrieval=self.config.get("deduplicate_retrieval", True),
            rerank_strategy=self.config.get("rerank_strategy", "combined"),
            top_k_final=self.config.get("top_k_final", None),
        )

        max_concurrent = self.config.get("max_concurrent", 5)
        self.semaphore = asyncio.Semaphore(max_concurrent)
        print(f"[init] Ready! (max_concurrent={max_concurrent})")

    async def process_question(
        self,
        row: pd.Series,
        index: int,
        total: int,
    ) -> QuestionResult:
        """Process a single question with retry (iterative deepening) on blank answers.

        Strategy (mirrors vendor wattbot_answer.py):
        - Start with top_k context snippets
        - If the LLM answer is blank, retry with 2*top_k, then 3*top_k, etc.
        - Stops after max_retries additional attempts or on a non-blank answer.
        """
        async with self.semaphore:
            print(f"[{index}/{total}] {row['id']}: processing...", flush=True)
            start_time = time.time()
            error_msg = None
            raw_response = ""

            # Select prompts based on config (C→Q vs Q→C)
            if self.use_reordered_prompt:
                sys_prompt = SYSTEM_PROMPT_REORDERED
                usr_template = USER_TEMPLATE_REORDERED
            else:
                sys_prompt = SYSTEM_PROMPT
                usr_template = USER_TEMPLATE

            try:
                additional_info = {
                    "answer_unit": row.get("answer_unit", ""),
                    "question_id": row["id"],
                }

                base_top_k = self.config.get("top_k", 5)
                base_top_k_final = self.config.get("top_k_final", None)
                result = None
                answer_is_blank = True
                retry_count = 0

                # Retry loop: increase retrieval depth on blank answers
                for attempt in range(self.max_retries + 1):
                    current_top_k = base_top_k * (attempt + 1)
                    current_top_k_final = (
                        base_top_k_final * (attempt + 1)
                        if base_top_k_final is not None
                        else None
                    )

                    try:
                        result = await self.pipeline.run_qa(
                            question=row["question"],
                            top_k=current_top_k,
                            system_prompt=sys_prompt,
                            user_template=usr_template,
                            additional_info=additional_info,
                            top_k_final=current_top_k_final,
                        )
                    except Exception as retry_err:
                        # Context overflow — reduce top_k and try once more
                        err_msg = str(retry_err).lower()
                        if "maximum context length" in err_msg or "context_length_exceeded" in err_msg:
                            reduced_k = max(current_top_k - 2, 1)
                            print(f"  [{row['id']}] Context overflow at top_k={current_top_k}, retrying with {reduced_k}", flush=True)
                            result = await self.pipeline.run_qa(
                                question=row["question"],
                                top_k=reduced_k,
                                system_prompt=sys_prompt,
                                user_template=usr_template,
                                additional_info=additional_info,
                            )
                        else:
                            raise

                    # Check if the answer is blank
                    answer_is_blank = (
                        result.answer.answer_value.strip().lower() == "is_blank"
                        or not result.answer.ref_id
                    )
                    if not answer_is_blank:
                        retry_count = attempt
                        break  # Got a valid answer
                    if attempt < self.max_retries:
                        print(f"  [{row['id']}] Blank answer at top_k={current_top_k}, deepening retrieval...", flush=True)
                else:
                    # All retries exhausted — record the full count
                    retry_count = self.max_retries

                # Store raw model output — normalisation is applied post-hoc
                # by scripts/posthoc.py (single source of truth).
                pred_value = str(result.answer.answer_value).strip()
                pred_ref = result.answer.ref_id  # raw list from pipeline
                pred_explanation = result.answer.explanation
                raw_response = getattr(result, "raw_response", "")
                timing = getattr(result, "timing", {})
                rendered_prompt = getattr(result, "prompt", "")

                # Capture LLM evidence fields
                pred_ref_url = getattr(result.answer, "ref_url", []) or []
                pred_supporting_materials = getattr(result.answer, "supporting_materials", "") or ""

                # Serialize retrieval snippets for debugging
                snippets_data = []
                retrieval = getattr(result, "retrieval", None)
                if retrieval and hasattr(retrieval, "snippets"):
                    for s in retrieval.snippets:
                        snippets_data.append({
                            "node_id": s.node_id,
                            "document_title": s.document_title,
                            "text": s.text,
                            "score": round(s.score, 4),
                            "rank": s.rank,
                            "metadata": s.metadata,
                        })

                if isinstance(pred_ref, list):
                    pred_ref_str = json.dumps(pred_ref)
                elif pred_ref == "is_blank" or not pred_ref:
                    pred_ref_str = "is_blank"
                else:
                    pred_ref_str = json.dumps([pred_ref])

            except Exception as e:
                import traceback
                traceback.print_exc()
                error_msg = str(e)
                pred_value = "is_blank"
                pred_ref_str = "is_blank"
                pred_explanation = f"Error: {error_msg}"
                timing = {}
                rendered_prompt = ""
                snippets_data = []
                pred_ref_url = []
                pred_supporting_materials = ""

            latency = time.time() - start_time
            retrieval_s = timing.get("retrieval_s", 0.0)
            generation_s = timing.get("generation_s", 0.0)

            # Compute scores
            gt_value = str(row.get("answer_value", "is_blank"))
            gt_ref = str(row.get("ref_id", "is_blank"))

            bits = row_bits(
                sol={"answer_value": gt_value, "answer_unit": row.get("answer_unit", ""), "ref_id": gt_ref},
                sub={"answer_value": pred_value, "answer_unit": row.get("answer_unit", ""), "ref_id": pred_ref_str},
            )

            weighted = 0.75 * float(bits["val"]) + 0.15 * float(bits["ref"]) + 0.10 * float(bits["na"])

            status = "OK" if bits["val"] else "WRONG"
            preview = str(pred_value)[:40]
            print(f"[{index}/{total}] {row['id']}: {preview} [{status}] ({latency:.1f}s | ret={retrieval_s:.1f}s gen={generation_s:.1f}s)", flush=True)

            return QuestionResult(
                id=row["id"],
                question=row["question"],
                gt_value=gt_value,
                gt_unit=str(row.get("answer_unit", "")),
                gt_ref=gt_ref,
                pred_value=str(pred_value),
                pred_unit=str(row.get("answer_unit", "")),
                pred_ref=pred_ref_str,
                pred_explanation=pred_explanation,
                raw_response=raw_response,
                value_correct=bool(bits["val"]),
                ref_score=float(bits["ref"]),
                na_correct=bool(bits["na"]),
                weighted_score=weighted,
                latency_seconds=latency,
                retrieval_seconds=retrieval_s,
                generation_seconds=generation_s,
                error=error_msg,
                pred_ref_url=pred_ref_url,
                pred_supporting_materials=pred_supporting_materials,
                rendered_prompt=rendered_prompt,
                retrieved_snippets=snippets_data,
                num_snippets=len(snippets_data),
                retry_count=retry_count,
            )

    async def run(self, questions_df: pd.DataFrame) -> ExperimentSummary:
        """Run the full experiment."""
        if self.pipeline is None:
            raise RuntimeError("Pipeline not initialized")

        total = len(questions_df)
        start_time = time.time()
        model_id = get_model_display_id(self.config)
        provider = self.config.get("llm_provider", "openrouter")

        print(f"\n{'='*60}")
        print(f"EXPERIMENT: {self.experiment_name}")
        print(f"Provider: {provider}")
        print(f"Model: {model_id}")
        print(f"Questions: {total}")
        print(f"{'='*60}\n")

        # Start hardware monitors
        # NVML energy counter (preferred, hardware-level accuracy)
        self.nvml_energy = NVMLEnergyCounter()
        if self.nvml_energy.available:
            self.nvml_energy.start()
            print("[monitor] Using NVML energy counter")

        # nvidia-smi power sampling (fallback for energy, also provides avg/peak power)
        self.power_monitor = GPUPowerMonitor(device_id=0, interval=1.0)
        self.power_monitor.start()

        # CPU RSS monitor
        self.cpu_monitor = CPURSSMonitor(interval=0.5)
        self.cpu_monitor.start()

        # ── Resume from partial progress ────────────────────────────────
        # If a previous run crashed, chunk files already on disk are loaded
        # and those question IDs are skipped.
        self.output_dir.mkdir(parents=True, exist_ok=True)
        prior_dicts, next_chunk_idx = load_partial_progress(self.output_dir)
        completed_ids: set[str] = set()

        self.results = []
        if prior_dicts:
            completed_ids = {r["id"] for r in prior_dicts}
            # Reconstruct QuestionResult objects so aggregate scoring works
            for d in prior_dicts:
                self.results.append(QuestionResult(**{
                    k: v for k, v in d.items()
                    if k in QuestionResult.__dataclass_fields__
                }))
            print(f"[resume] Loaded {len(prior_dicts)} completed questions "
                  f"from {next_chunk_idx} chunk(s) — skipping those")

        # Filter to only questions that still need answering
        remaining_rows = [
            (_idx, row) for _idx, row in questions_df.iterrows()
            if row["id"] not in completed_ids
        ]

        if not remaining_rows:
            print("[resume] All questions already completed!")
        else:
            print(f"[run] {len(remaining_rows)} questions to process")

        # ── Process remaining questions in batches ────────────────────
        chunk_idx = next_chunk_idx
        for batch_start in range(0, len(remaining_rows), CHUNK_SIZE):
            batch_rows = remaining_rows[batch_start : batch_start + CHUNK_SIZE]

            tasks = []
            for i, (_idx, row) in enumerate(batch_rows):
                done_so_far = len(completed_ids) + batch_start + i + 1
                tasks.append(
                    self.process_question(row, done_so_far, total)
                )

            batch_results_raw = await asyncio.gather(*tasks, return_exceptions=True)

            # Convert exceptions to error QuestionResult objects
            batch_results = []
            for j, result in enumerate(batch_results_raw):
                if isinstance(result, Exception):
                    _idx, row = batch_rows[j]
                    done_so_far = len(completed_ids) + batch_start + j + 1
                    print(f"[{done_so_far}/{total}] {row['id']}: ERROR - {str(result)[:80]}", flush=True)
                    batch_results.append(QuestionResult(
                        id=row["id"],
                        question=row["question"],
                        gt_value=str(row.get("answer_value", "is_blank")),
                        gt_unit=str(row.get("answer_unit", "")),
                        gt_ref=str(row.get("ref_id", "is_blank")),
                        pred_value="is_blank",
                        pred_unit=str(row.get("answer_unit", "")),
                        pred_ref="is_blank",
                        pred_explanation=f"Error: {result!s}",
                        raw_response="",
                        value_correct=False,
                        ref_score=0.0,
                        na_correct=False,
                        weighted_score=0.0,
                        latency_seconds=0.0,
                        error=str(result),
                    ))
                else:
                    batch_results.append(result)
            self.results.extend(batch_results)

            # Flush this batch to its own chunk file
            chunk_data = [asdict(r) for r in batch_results]
            chunk_path = self.output_dir / f"results_chunk_{chunk_idx:03d}.json"
            with open(chunk_path, "w") as f:
                json.dump(chunk_data, f, indent=2)
            print(f"[checkpoint] Saved {len(chunk_data)} results to {chunk_path.name}", flush=True)
            chunk_idx += 1
        total_time = time.time() - start_time

        # Stop all monitors
        self.power_monitor.stop()

        # Compute aggregate metrics
        value_acc = sum(1 for r in self.results if r.value_correct) / total
        ref_overlap = sum(r.ref_score for r in self.results) / total

        # NA component: recall over truly-NA questions only
        # (avoids inflating the score when most questions are answerable)
        na_questions = [r for r in self.results if is_blank(r.gt_value)]
        if na_questions:
            na_recall = sum(1 for r in na_questions if r.na_correct) / len(na_questions)
        else:
            na_recall = 1.0  # no NA questions → perfect by default
        overall = 0.75 * value_acc + 0.15 * ref_overlap + 0.10 * na_recall
        avg_latency = sum(r.latency_seconds for r in self.results) / total
        avg_retrieval = sum(r.retrieval_seconds for r in self.results) / total
        avg_generation = sum(r.generation_seconds for r in self.results) / total
        error_count = sum(1 for r in self.results if r.error)
        total_retries = sum(r.retry_count for r in self.results)
        questions_retried = sum(1 for r in self.results if r.retry_count > 0)
        avg_retries = total_retries / total if total else 0.0

        # Get token usage from chat model (if supported)
        input_tokens = 0
        output_tokens = 0
        if hasattr(self.chat_model, 'token_usage'):
            input_tokens = self.chat_model.token_usage.input_tokens
            output_tokens = self.chat_model.token_usage.output_tokens

        estimated_cost = estimate_cost(model_id, input_tokens, output_tokens)

        # Collect hardware metrics (VRAM, disk size, energy, CPU RSS)
        hw_metrics = collect_post_experiment_metrics(
            model_id=model_id,
            device_id=0,
            power_monitor=self.power_monitor,
            nvml_energy=self.nvml_energy,
            cpu_monitor=self.cpu_monitor,
            model_load_time=self.model_load_time,
            llm_load_time=getattr(self, "llm_load_time", 0.0),
            embedder_load_time=getattr(self, "embedder_load_time", 0.0),
        )
        hw_dict = asdict(hw_metrics)

        # Determine quantization/dtype
        if provider == "hf_local":
            quantization = self.config.get("hf_dtype", "4bit")
        else:
            quantization = "api"

        return ExperimentSummary(
            name=self.experiment_name,
            config_path=str(self.config.get("_config_path", "unknown")),
            model_id=model_id,
            llm_provider=provider,
            quantization=quantization,
            timestamp=datetime.now().isoformat(),
            num_questions=total,
            total_time_seconds=total_time,
            avg_latency_seconds=avg_latency,
            avg_retrieval_seconds=avg_retrieval,
            avg_generation_seconds=avg_generation,
            value_accuracy=value_acc,
            ref_overlap=ref_overlap,
            na_accuracy=na_recall,
            overall_score=overall,
            questions_correct=sum(1 for r in self.results if r.value_correct),
            questions_wrong=sum(1 for r in self.results if not r.value_correct),
            error_count=error_count,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            estimated_cost_usd=estimated_cost,
            hardware=hw_dict,
            config_snapshot=self.config,
            total_retries=total_retries,
            questions_retried=questions_retried,
            avg_retries=avg_retries,
            run_environment=self.config.get("_run_environment", ""),
            questions_file=self.config.get("_questions_file", ""),
        )

    def save_results(self, summary: ExperimentSummary) -> None:
        """Save submission CSV and summary JSON.

        Results chunk files (results_chunk_NNN.json) are already written
        incrementally during :meth:`run`, so this only handles the final
        artefacts that depend on the complete result set.
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Save Kaggle-format submission CSV
        submission_rows = []
        for r in self.results:
            submission_rows.append({
                "id": r.id,
                "question": r.question,
                "answer": r.pred_explanation,
                "answer_value": r.pred_value,
                "answer_unit": r.pred_unit,
                "ref_id": r.pred_ref,
                "ref_url": json.dumps(r.pred_ref_url) if r.pred_ref_url else "is_blank",
                "supporting_materials": r.pred_supporting_materials if r.pred_supporting_materials else "is_blank",
                "explanation": r.pred_explanation,
            })

        sub_df = pd.DataFrame(submission_rows)
        sub_path = self.output_dir / "submission.csv"
        sub_df.to_csv(sub_path, index=False, quoting=csv.QUOTE_ALL)
        print(f"\nSaved submission: {sub_path}")

        # Save summary JSON
        summary_path = self.output_dir / "summary.json"
        with open(summary_path, "w") as f:
            json.dump(asdict(summary), f, indent=2)
        print(f"Saved summary: {summary_path}")


# =============================================================================
# Main
# =============================================================================

async def main(config_path: str, experiment_name: str | None = None, run_environment: str = "", questions_override: str | None = None, precision: str = "4bit") -> None:
    """Run an experiment with the given config."""
    # Print CUDA status upfront so GPU issues are caught immediately
    import torch
    print(f"[env] PyTorch {torch.__version__} | CUDA available: {torch.cuda.is_available()}"
          f"{f' | Device: {torch.cuda.get_device_name(0)}' if torch.cuda.is_available() else ' | *** WARNING: running on CPU ***'}")

    config = load_config(config_path)
    config["_config_path"] = config_path
    config["_run_environment"] = run_environment

    # CLI --precision overrides any hf_dtype in config (only relevant for hf_local)
    if config.get("llm_provider") == "hf_local":
        config["hf_dtype"] = precision

    # Generate experiment name if not provided
    if experiment_name is None:
        model_id = get_model_display_id(config)
        # Create a short name from the model ID
        model_short = model_id.split("/")[-1].split("-")[0] if "/" in model_id else model_id.split(".")[-1].split("-")[0]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_name = f"{model_short}_{timestamp}"

    # Load questions (CLI --questions overrides config)
    project_root = Path(__file__).parent.parent
    if questions_override:
        qpath = Path(questions_override)
        questions_path = qpath if qpath.is_absolute() else project_root / qpath
    else:
        q_raw = config.get("questions", "data/train_QA.csv")
        questions_path = project_root / q_raw.removeprefix("../").removeprefix("../")
    config["_questions_file"] = str(questions_path.name)

    # Derive datafile subfolder from questions filename (e.g. "train_QA", "test_solutions")
    datafile_stem = questions_path.stem

    # Organize output: artifacts/experiments/<env>/<datafile>/<name>/
    experiments_base = Path(__file__).parent.parent / "artifacts" / "experiments"
    if run_environment:
        output_dir = experiments_base / run_environment / datafile_stem / experiment_name
    else:
        output_dir = experiments_base / datafile_stem / experiment_name
    questions_df = pd.read_csv(questions_path)
    print(f"Loaded {len(questions_df)} questions from {questions_path}")

    # Run experiment
    runner = ExperimentRunner(config, experiment_name, output_dir)
    await runner.initialize()
    summary = await runner.run(questions_df)
    runner.save_results(summary)

    # Print summary
    print(f"\n{'='*60}")
    print("EXPERIMENT COMPLETE")
    print(f"{'='*60}")
    print(f"Provider     : {summary.llm_provider}")
    print(f"Model        : {summary.model_id}")
    print(f"Quantization : {summary.quantization}")
    if summary.run_environment:
        print(f"Environment  : {summary.run_environment}")
    print(f"Dataset      : {summary.questions_file}")
    print(f"Questions    : {summary.num_questions}")
    print(f"Correct      : {summary.questions_correct} ({100*summary.value_accuracy:.1f}%)")
    print(f"Wrong        : {summary.questions_wrong}")
    print(f"Errors       : {summary.error_count}")
    print(f"\nComponent Scores:")
    print(f"  Value match  : {summary.value_accuracy:.3f}")
    print(f"  Ref overlap  : {summary.ref_overlap:.3f}")
    print(f"  NA recall    : {summary.na_accuracy:.3f}")
    print(f"\nOVERALL SCORE  : {summary.overall_score:.3f}")
    if summary.input_tokens > 0:
        print(f"\nCost Tracking:")
        print(f"  Input tokens : {summary.input_tokens:,}")
        print(f"  Output tokens: {summary.output_tokens:,}")
        print(f"  Est. cost    : ${summary.estimated_cost_usd:.4f}")
    print(f"\nTiming:")
    print(f"  Total time   : {summary.total_time_seconds:.1f}s ({summary.total_time_seconds/60:.1f} min)")
    print(f"  Avg latency  : {summary.avg_latency_seconds:.2f}s/question")
    print(f"    Retrieval  : {summary.avg_retrieval_seconds:.2f}s/question")
    print(f"    Generation : {summary.avg_generation_seconds:.2f}s/question")
    hw = summary.hardware
    if hw:
        print(f"\nHardware Metrics:")
        if hw.get("hostname"):
            gpu_ct = hw.get("gpu_count", 0)
            gpu_note = f" ({gpu_ct} GPU{'s' if gpu_ct != 1 else ''})" if gpu_ct else ""
            print(f"  Host         : {hw['hostname']}{gpu_note}")
        if hw.get("gpu_name"):
            print(f"  GPU          : {hw['gpu_name']}")
        if hw.get("gpu_vram_allocated_gb"):
            print(f"  VRAM (peak)  : {hw['gpu_vram_allocated_gb']:.2f} GB allocated / {hw['gpu_vram_total_gb']:.1f} GB total")
        if hw.get("model_disk_size_gb"):
            print(f"  Model on disk: {hw['model_disk_size_gb']:.2f} GB")
        if hw.get("model_load_time_seconds"):
            llm_t = hw.get("llm_load_time_seconds", 0)
            embed_t = hw.get("embedder_load_time_seconds", 0)
            print(f"  Load time    : {hw['model_load_time_seconds']:.1f}s (LLM={llm_t:.1f}s, Embedder={embed_t:.1f}s)")
        if hw.get("cpu_rss_peak_gb", 0) > 0:
            print(f"  CPU RSS peak : {hw['cpu_rss_peak_gb']:.2f} GB")
        if hw.get("gpu_energy_wh", 0) > 0:
            method = hw.get("gpu_energy_method", "")
            method_note = f" ({method})" if method else ""
            print(f"  Energy       : {hw['gpu_energy_wh']:.3f} Wh{method_note}")
            print(f"  Avg power    : {hw['gpu_avg_power_watts']:.1f} W")
            print(f"  Peak power   : {hw['gpu_peak_power_watts']:.1f} W")
    print(f"\nOutput dir     : {output_dir}")


def cli():
    """Parse arguments and run."""
    parser = argparse.ArgumentParser(
        description="Run WattBot model experiments with automatic scoring (provider-agnostic)"
    )
    parser.add_argument(
        "--config", "-c",
        required=True,
        help="Path to config file (e.g., vendor/KohakuRAG/configs/hf_qwen7b.py)"
    )
    parser.add_argument(
        "--name", "-n",
        default=None,
        help="Experiment name (default: auto-generated from model + timestamp)"
    )
    parser.add_argument(
        "--env", "-e",
        default="",
        help="Run environment label for cross-machine comparison (e.g. 'GB10', 'PowerEdge')"
    )
    parser.add_argument(
        "--questions", "-q",
        default=None,
        help="Override questions CSV path (e.g. data/test_solutions.csv)"
    )
    parser.add_argument(
        "--precision", "-p",
        default="4bit",
        choices=["4bit", "bf16", "fp16", "auto"],
        help="Model precision/quantization (default: 4bit). Only applies to hf_local models."
    )

    args = parser.parse_args()
    asyncio.run(main(args.config, args.name, args.env, args.questions, args.precision))


if __name__ == "__main__":
    cli()
