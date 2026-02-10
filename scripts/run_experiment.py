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
    python scripts/run_experiment.py --config configs/hf_qwen7b.py
    python scripts/run_experiment.py --config configs/hf_qwen7b.py --name "qwen7b-test"

Output:
    - artifacts/experiments/<name>/submission.csv - Kaggle format submission
    - artifacts/experiments/<name>/results.json - Detailed per-question results
    - artifacts/experiments/<name>/summary.json - Overall metrics and timing
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

from kohakurag import RAGPipeline
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
    error: str | None = None


@dataclass
class ExperimentSummary:
    """Summary of an experiment run."""
    name: str
    config_path: str
    model_id: str
    llm_provider: str
    timestamp: str
    num_questions: int
    total_time_seconds: float
    avg_latency_seconds: float
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
# Prompts
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
            dtype=config.get("hf_dtype", "bf16"),
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

    async def initialize(self) -> None:
        """Initialize the RAG pipeline with config settings."""
        project_root = Path(__file__).parent.parent

        db_raw = self.config.get("db", "artifacts/wattbot_jinav4.db")
        db_path = project_root / db_raw.removeprefix("../").removeprefix("../")
        if not db_path.exists():
            raise FileNotFoundError(
                f"Database not found: {db_path}\n\n"
                f"The vector database must be built before running experiments.\n"
                f"To build the index, run from the vendor/KohakuRAG directory:\n\n"
                f"  cd vendor/KohakuRAG\n"
                f"  kogine run scripts/wattbot_build_index.py --config configs/jinav4/index.py\n\n"
                f"Or directly:\n\n"
                f"  python vendor/KohakuRAG/scripts/wattbot_build_index.py\n\n"
                f"This requires documents in artifacts/docs/ (or artifacts/docs_with_images/)\n"
                f"and metadata in data/metadata.csv. See vendor/KohakuRAG/docs/wattbot.md for details."
            )

        model_id = get_model_display_id(self.config)
        provider = self.config.get("llm_provider", "openrouter")

        print(f"[init] Provider: {provider}")
        print(f"[init] Model: {model_id}")

        # Reset GPU peak stats before loading (for accurate VRAM measurement)
        reset_gpu_peak_stats()

        load_start = time.time()
        self.chat_model = create_chat_model_from_config(self.config, SYSTEM_PROMPT)
        embedder = create_embedder_from_config(self.config)
        self.model_load_time = time.time() - load_start
        print(f"[init] Model loaded in {self.model_load_time:.1f}s")

        table_prefix = self.config.get("table_prefix", "wattbot_jv4")
        print(f"[init] Loading vector store from {db_path} (prefix: {table_prefix})...")
        store = KVaultNodeStore(
            db_path,
            table_prefix=table_prefix,
            dimensions=None,
            paragraph_search_mode="averaged",
        )

        print("[init] Building RAG pipeline...")
        self.pipeline = RAGPipeline(
            store=store,
            embedder=embedder,
            chat_model=self.chat_model,
            planner=None,
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
        """Process a single question and return detailed result."""
        async with self.semaphore:
            start_time = time.time()
            error_msg = None
            raw_response = ""

            try:
                additional_info = {
                    "answer_unit": row.get("answer_unit", ""),
                    "question_id": row["id"],
                }

                result = await self.pipeline.run_qa(
                    question=row["question"],
                    top_k=self.config.get("top_k", 5),
                    system_prompt=SYSTEM_PROMPT,
                    user_template=USER_TEMPLATE,
                    additional_info=additional_info,
                )

                pred_value = result.answer.answer_value
                pred_ref = result.answer.ref_id
                pred_explanation = result.answer.explanation
                raw_response = getattr(result, "raw_response", "")

                if isinstance(pred_ref, list):
                    pred_ref_str = json.dumps(pred_ref)
                elif pred_ref == "is_blank" or not pred_ref:
                    pred_ref_str = "is_blank"
                else:
                    pred_ref_str = json.dumps([pred_ref])

            except Exception as e:
                error_msg = str(e)
                pred_value = "is_blank"
                pred_ref_str = "is_blank"
                pred_explanation = f"Error: {error_msg}"

            latency = time.time() - start_time

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
            print(f"[{index}/{total}] {row['id']}: {preview} [{status}] ({latency:.1f}s)")

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
                error=error_msg,
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

        # Process all questions
        tasks = []
        for idx, row in questions_df.iterrows():
            task = self.process_question(row, len(tasks) + 1, total)
            tasks.append(task)

        self.results = await asyncio.gather(*tasks)
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
        error_count = sum(1 for r in self.results if r.error)

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
        )
        hw_dict = asdict(hw_metrics)

        return ExperimentSummary(
            name=self.experiment_name,
            config_path=str(self.config.get("_config_path", "unknown")),
            model_id=model_id,
            llm_provider=provider,
            timestamp=datetime.now().isoformat(),
            num_questions=total,
            total_time_seconds=total_time,
            avg_latency_seconds=avg_latency,
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
        )

    def save_results(self, summary: ExperimentSummary) -> None:
        """Save all experiment outputs."""
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
                "ref_url": "is_blank",
                "supporting_materials": "is_blank",
                "explanation": r.pred_explanation,
            })

        sub_df = pd.DataFrame(submission_rows)
        sub_path = self.output_dir / "submission.csv"
        sub_df.to_csv(sub_path, index=False, quoting=csv.QUOTE_ALL)
        print(f"\nSaved submission: {sub_path}")

        # Save detailed results JSON
        results_path = self.output_dir / "results.json"
        results_data = [asdict(r) for r in self.results]
        with open(results_path, "w") as f:
            json.dump(results_data, f, indent=2)
        print(f"Saved results: {results_path}")

        # Save summary JSON
        summary_path = self.output_dir / "summary.json"
        with open(summary_path, "w") as f:
            json.dump(asdict(summary), f, indent=2)
        print(f"Saved summary: {summary_path}")


# =============================================================================
# Main
# =============================================================================

async def main(config_path: str, experiment_name: str | None = None) -> None:
    """Run an experiment with the given config."""
    config = load_config(config_path)
    config["_config_path"] = config_path

    # Generate experiment name if not provided
    if experiment_name is None:
        model_id = get_model_display_id(config)
        # Create a short name from the model ID
        model_short = model_id.split("/")[-1].split("-")[0] if "/" in model_id else model_id.split(".")[-1].split("-")[0]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_name = f"{model_short}_{timestamp}"

    output_dir = Path(__file__).parent.parent / "artifacts" / "experiments" / experiment_name

    # Load questions
    project_root = Path(__file__).parent.parent
    q_raw = config.get("questions", "data/train_QA.csv")
    questions_path = project_root / q_raw.removeprefix("../").removeprefix("../")
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
    hw = summary.hardware
    if hw:
        print(f"\nHardware Metrics:")
        if hw.get("gpu_name"):
            print(f"  GPU          : {hw['gpu_name']}")
        if hw.get("gpu_vram_allocated_gb"):
            print(f"  VRAM (peak)  : {hw['gpu_vram_allocated_gb']:.2f} GB allocated / {hw['gpu_vram_total_gb']:.1f} GB total")
        if hw.get("model_disk_size_gb"):
            print(f"  Model on disk: {hw['model_disk_size_gb']:.2f} GB")
        if hw.get("model_load_time_seconds"):
            print(f"  Load time    : {hw['model_load_time_seconds']:.1f}s")
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
        help="Path to config file (e.g., configs/hf_qwen7b.py)"
    )
    parser.add_argument(
        "--name", "-n",
        default=None,
        help="Experiment name (default: auto-generated from model + timestamp)"
    )

    args = parser.parse_args()
    asyncio.run(main(args.config, args.name))


if __name__ == "__main__":
    cli()
