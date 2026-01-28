#!/usr/bin/env python3
"""
Automated Model Experiment Runner

Runs experiments with different Bedrock models and automatically scores results.
Saves detailed outputs including raw responses, per-question scores, and timing info.

Usage:
    python scripts/run_experiment.py --config configs/bedrock_sonnet.py
    python scripts/run_experiment.py --config configs/bedrock_llama4_scout.py --name "llama4-scout-test"

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
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "KohakuRAG" / "src"))
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd

from llm_bedrock import BedrockChatModel
from kohakurag import RAGPipeline
from kohakurag.datastore import KVaultNodeStore
from kohakurag.embeddings import JinaEmbeddingModel, JinaV4EmbeddingModel

from score import score as compute_wattbot_score, row_bits, is_blank


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
    # Cost tracking
    input_tokens: int = 0
    output_tokens: int = 0
    estimated_cost_usd: float = 0.0
    config_snapshot: dict = field(default_factory=dict)


# Bedrock pricing per 1M tokens (as of Jan 2025)
# https://aws.amazon.com/bedrock/pricing/
BEDROCK_PRICING = {
    "claude-3-haiku": {"input": 0.25, "output": 1.25},
    "claude-3-sonnet": {"input": 3.00, "output": 15.00},
    "claude-3-5-sonnet": {"input": 3.00, "output": 15.00},
    "claude-3-opus": {"input": 15.00, "output": 75.00},
    "llama4-scout": {"input": 0.17, "output": 0.17},  # Estimated
    "deepseek-v3": {"input": 0.50, "output": 1.00},   # Estimated
}


def estimate_cost(model_id: str, input_tokens: int, output_tokens: int) -> float:
    """Estimate cost in USD based on model and token counts."""
    # Find matching pricing tier
    model_lower = model_id.lower()
    pricing = None

    for key, prices in BEDROCK_PRICING.items():
        if key in model_lower:
            pricing = prices
            break

    if pricing is None:
        # Default to Haiku pricing as conservative estimate
        pricing = BEDROCK_PRICING["claude-3-haiku"]

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
        "db", "table_prefix", "questions", "output", "metadata",
        "llm_provider", "bedrock_profile", "bedrock_region", "bedrock_model",
        "embedding_model", "embedding_dim", "embedding_task",
        "top_k", "planner_max_queries", "deduplicate_retrieval",
        "rerank_strategy", "top_k_final", "retrieval_threshold",
        "max_retries", "max_concurrent",
    ]

    for key in config_keys:
        if hasattr(module, key):
            config[key] = getattr(module, key)

    return config


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
        self.chat_model: BedrockChatModel | None = None
        self.semaphore: asyncio.Semaphore | None = None
        self.results: list[QuestionResult] = []

    async def initialize(self) -> None:
        """Initialize the RAG pipeline with config settings."""
        # Resolve paths relative to project root
        project_root = Path(__file__).parent.parent

        db_path = project_root / self.config.get("db", "artifacts/wattbot_jinav4.db").lstrip("../")
        if not db_path.exists():
            raise FileNotFoundError(f"Database not found: {db_path}")

        model_id = self.config.get("bedrock_model", "us.anthropic.claude-3-haiku-20240307-v1:0")
        profile = self.config.get("bedrock_profile", "bedrock_nils")
        region = self.config.get("bedrock_region", "us-east-2")

        print(f"[init] Loading Bedrock model: {model_id}")
        self.chat_model = BedrockChatModel(
            model_id=model_id,
            profile_name=profile,
            region_name=region,
            system_prompt=SYSTEM_PROMPT,
            max_retries=self.config.get("max_retries", 3),
            experiment_tag=f"wattbot-{self.experiment_name}",
        )
        chat = self.chat_model

        # Load embedding model based on config
        embedding_model = self.config.get("embedding_model", "jina")
        embedding_dim = self.config.get("embedding_dim", 1024)
        embedding_task = self.config.get("embedding_task", "retrieval")

        if embedding_model == "jinav4":
            print(f"[init] Loading JinaV4 embeddings (dim={embedding_dim}, task={embedding_task})...")
            embedder = JinaV4EmbeddingModel(
                task=embedding_task,
                truncate_dim=embedding_dim,
            )
        else:
            print("[init] Loading Jina V3 embeddings...")
            embedder = JinaEmbeddingModel()

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
            chat_model=chat,
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

        print(f"\n{'='*60}")
        print(f"EXPERIMENT: {self.experiment_name}")
        print(f"Model: {self.config.get('bedrock_model', 'unknown')}")
        print(f"Questions: {total}")
        print(f"{'='*60}\n")

        # Process all questions
        tasks = []
        for idx, row in questions_df.iterrows():
            task = self.process_question(row, len(tasks) + 1, total)
            tasks.append(task)

        self.results = await asyncio.gather(*tasks)
        total_time = time.time() - start_time

        # Compute aggregate metrics
        value_acc = sum(1 for r in self.results if r.value_correct) / total
        ref_overlap = sum(r.ref_score for r in self.results) / total
        na_acc = sum(1 for r in self.results if r.na_correct) / total
        overall = 0.75 * value_acc + 0.15 * ref_overlap + 0.10 * na_acc
        avg_latency = sum(r.latency_seconds for r in self.results) / total
        error_count = sum(1 for r in self.results if r.error)

        # Get token usage from chat model
        input_tokens = 0
        output_tokens = 0
        if self.chat_model:
            input_tokens = self.chat_model.token_usage.input_tokens
            output_tokens = self.chat_model.token_usage.output_tokens

        model_id = self.config.get("bedrock_model", "unknown")
        estimated_cost = estimate_cost(model_id, input_tokens, output_tokens)

        return ExperimentSummary(
            name=self.experiment_name,
            config_path=str(self.config.get("_config_path", "unknown")),
            model_id=model_id,
            timestamp=datetime.now().isoformat(),
            num_questions=total,
            total_time_seconds=total_time,
            avg_latency_seconds=avg_latency,
            value_accuracy=value_acc,
            ref_overlap=ref_overlap,
            na_accuracy=na_acc,
            overall_score=overall,
            questions_correct=sum(1 for r in self.results if r.value_correct),
            questions_wrong=sum(1 for r in self.results if not r.value_correct),
            error_count=error_count,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            estimated_cost_usd=estimated_cost,
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
        model_id = config.get("bedrock_model", "unknown")
        model_short = model_id.split(".")[-1].split("-")[0]  # e.g., "claude" from full ID
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_name = f"{model_short}_{timestamp}"

    output_dir = Path(__file__).parent.parent / "artifacts" / "experiments" / experiment_name

    # Load questions
    project_root = Path(__file__).parent.parent
    questions_path = project_root / config.get("questions", "data/train_QA.csv").lstrip("../")
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
    print(f"Model        : {summary.model_id}")
    print(f"Questions    : {summary.num_questions}")
    print(f"Correct      : {summary.questions_correct} ({100*summary.value_accuracy:.1f}%)")
    print(f"Wrong        : {summary.questions_wrong}")
    print(f"Errors       : {summary.error_count}")
    print(f"\nComponent Scores:")
    print(f"  Value match  : {summary.value_accuracy:.3f}")
    print(f"  Ref overlap  : {summary.ref_overlap:.3f}")
    print(f"  NA agreement : {summary.na_accuracy:.3f}")
    print(f"\nOVERALL SCORE  : {summary.overall_score:.3f}")
    print(f"\nCost Tracking:")
    print(f"  Input tokens : {summary.input_tokens:,}")
    print(f"  Output tokens: {summary.output_tokens:,}")
    print(f"  Est. cost    : ${summary.estimated_cost_usd:.4f}")
    print(f"\nTiming:")
    print(f"  Total time   : {summary.total_time_seconds:.1f}s ({summary.total_time_seconds/60:.1f} min)")
    print(f"  Avg latency  : {summary.avg_latency_seconds:.2f}s/question")
    print(f"\nOutput dir     : {output_dir}")


def cli():
    """Parse arguments and run."""
    parser = argparse.ArgumentParser(
        description="Run WattBot model experiments with automatic scoring"
    )
    parser.add_argument(
        "--config", "-c",
        required=True,
        help="Path to config file (e.g., configs/bedrock_sonnet.py)"
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
