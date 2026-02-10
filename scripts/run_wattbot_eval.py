#!/usr/bin/env python3
"""
WattBot Evaluation Pipeline (Provider-Agnostic)
=================================================

Runs the RAG pipeline on train_QA.csv and computes the official WattBot
competition score. Supports local HF models, OpenRouter, OpenAI, and Bedrock.

Usage:
    # Local HF model (default):
    python scripts/run_wattbot_eval.py

    # With a specific config:
    python scripts/run_wattbot_eval.py --config configs/hf_qwen7b.py

    # Override provider via CLI:
    python scripts/run_wattbot_eval.py --provider hf_local --hf-model Qwen/Qwen2.5-7B-Instruct

    # Custom paths:
    python scripts/run_wattbot_eval.py --input data/train_QA.csv --output artifacts/submission.csv

Output:
    - artifacts/submission.csv: The generated predictions in Kaggle format
    - Console output: Detailed scoring breakdown

WattBot Scoring Rubric:
    - answer_value (75%): Exact match with +/-0.1% tolerance for numerics
    - ref_id (15%): Jaccard overlap between predicted and ground truth doc IDs
    - is_NA (10%): Correctly identifying unanswerable questions with "is_blank"
"""

import argparse
import asyncio
import csv
import importlib.util
import json
import sys
import time
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

# Optional: Bedrock support
try:
    from llm_bedrock import BedrockChatModel
    HAS_BEDROCK = True
except ImportError:
    HAS_BEDROCK = False

from score import score as compute_wattbot_score


# =============================================================================
# Default Configuration
# =============================================================================

DEFAULT_INPUT = "data/train_QA.csv"
DEFAULT_OUTPUT = "artifacts/submission.csv"
DEFAULT_DB = "artifacts/wattbot.db"
DEFAULT_PROVIDER = "hf_local"

# Default HF model settings
DEFAULT_HF_MODEL = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_HF_DTYPE = "bf16"
DEFAULT_EMBED_MODEL = "BAAI/bge-base-en-v1.5"

# Concurrency
MAX_CONCURRENT_REQUESTS = 2  # Conservative for local models
TOP_K_CHUNKS = 5


# =============================================================================
# Prompts
# =============================================================================

SYSTEM_PROMPT = """
You must answer strictly based on the provided context snippets.
Do NOT use external knowledge or assumptions.
If the context does not clearly support an answer, you must output the literal string "is_blank" for both answer_value and ref_id.
""".strip()

USER_TEMPLATE = """
You will be given a question and context snippets taken from documents.
You must follow these rules:
- Use only the provided context; do not rely on external knowledge.
- If the context does not clearly support an answer, use "is_blank" for all fields except explanation.
- For unanswerable questions, set answer to "Unable to answer with confidence based on the provided documents."

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
# Provider Factory
# =============================================================================

def create_chat_model(config: dict, system_prompt: str):
    """Create chat model from config dict."""
    provider = config.get("llm_provider", DEFAULT_PROVIDER)

    if provider == "hf_local":
        return HuggingFaceLocalChatModel(
            model=config.get("hf_model_id", DEFAULT_HF_MODEL),
            system_prompt=system_prompt,
            dtype=config.get("hf_dtype", DEFAULT_HF_DTYPE),
            max_new_tokens=config.get("hf_max_new_tokens", 512),
            temperature=config.get("hf_temperature", 0.2),
            max_concurrent=config.get("max_concurrent", MAX_CONCURRENT_REQUESTS),
        )
    elif provider == "bedrock":
        if not HAS_BEDROCK:
            raise ImportError("Bedrock provider requires llm_bedrock module.")
        return BedrockChatModel(
            model_id=config.get("bedrock_model", "us.anthropic.claude-3-haiku-20240307-v1:0"),
            profile_name=config.get("bedrock_profile", "bedrock_nils"),
            region_name=config.get("bedrock_region", "us-east-2"),
            system_prompt=system_prompt,
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
        return OpenAIChatModel(
            model=config.get("model", "gpt-4o-mini"),
            system_prompt=system_prompt,
            max_concurrent=config.get("max_concurrent", 5),
        )


def create_embedder(config: dict):
    """Create embedder from config dict."""
    model_type = config.get("embedding_model", "hf_local")

    if model_type == "hf_local":
        return LocalHFEmbeddingModel(
            model_name=config.get("embedding_model_id", DEFAULT_EMBED_MODEL)
        )
    elif model_type == "jinav4":
        return JinaV4EmbeddingModel(
            task=config.get("embedding_task", "retrieval"),
            truncate_dim=config.get("embedding_dim", 1024),
        )
    else:
        return JinaEmbeddingModel()


# =============================================================================
# Evaluator
# =============================================================================

class WattBotEvaluator:
    """Orchestrates batch evaluation of the RAG pipeline on WattBot questions."""

    def __init__(self, config: dict):
        self.config = config
        self.pipeline: RAGPipeline | None = None
        self.semaphore: asyncio.Semaphore | None = None

    async def initialize(self) -> None:
        """Load all pipeline components."""
        db_path = Path(self.config.get("db", DEFAULT_DB))
        if not db_path.exists():
            raise FileNotFoundError(
                f"Vector database not found at {db_path}.\n\n"
                f"The vector database must be built before running experiments.\n"
                f"To build the index, run from the vendor/KohakuRAG directory:\n\n"
                f"  cd vendor/KohakuRAG\n"
                f"  kogine run scripts/wattbot_build_index.py --config configs/jinav4/index.py\n\n"
                f"Or directly:\n\n"
                f"  python vendor/KohakuRAG/scripts/wattbot_build_index.py\n\n"
                f"This requires documents in artifacts/docs/ and metadata in data/metadata.csv.\n"
                f"See vendor/KohakuRAG/docs/wattbot.md for details."
            )

        provider = self.config.get("llm_provider", DEFAULT_PROVIDER)
        print(f"[init] Provider: {provider}")

        chat = create_chat_model(self.config, SYSTEM_PROMPT)
        embedder = create_embedder(self.config)

        table_prefix = self.config.get("table_prefix", "wattbot")
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

        max_concurrent = self.config.get("max_concurrent", MAX_CONCURRENT_REQUESTS)
        self.semaphore = asyncio.Semaphore(max_concurrent)
        print(f"[init] Ready! (max_concurrent={max_concurrent})")

    async def _process_single_question(
        self,
        question_id: str,
        question_text: str,
        answer_unit: str,
        index: int,
        total: int,
    ) -> dict[str, Any]:
        """Process a single question through the RAG pipeline."""
        async with self.semaphore:
            start_time = time.time()

            try:
                additional_info = {
                    "answer_unit": answer_unit,
                    "question_id": question_id,
                }

                result = await self.pipeline.run_qa(
                    question=question_text,
                    top_k=self.config.get("top_k", TOP_K_CHUNKS),
                    system_prompt=SYSTEM_PROMPT,
                    user_template=USER_TEMPLATE,
                    additional_info=additional_info,
                )

                answer_value = result.answer.answer_value
                answer_nl = getattr(result.answer, 'answer', answer_value)
                ref_id = result.answer.ref_id
                ref_url = getattr(result.answer, 'ref_url', 'is_blank')
                supporting = getattr(result.answer, 'supporting_materials', 'is_blank')
                explanation = result.answer.explanation

                if isinstance(ref_id, list):
                    ref_id_str = json.dumps(ref_id)
                elif ref_id == "is_blank" or not ref_id:
                    ref_id_str = "is_blank"
                else:
                    ref_id_str = json.dumps([ref_id])

                if isinstance(ref_url, list):
                    ref_url_str = json.dumps(ref_url)
                elif ref_url == "is_blank" or not ref_url:
                    ref_url_str = "is_blank"
                else:
                    ref_url_str = json.dumps([ref_url])

                elapsed = time.time() - start_time
                preview = str(answer_value)[:50] if answer_value else "is_blank"
                print(f"[{index}/{total}] {question_id}: {preview}... ({elapsed:.1f}s)")

                return {
                    "id": question_id,
                    "question": question_text,
                    "answer": answer_nl,
                    "answer_value": answer_value,
                    "answer_unit": answer_unit,
                    "ref_id": ref_id_str,
                    "ref_url": ref_url_str,
                    "supporting_materials": supporting,
                    "explanation": explanation,
                }

            except Exception as e:
                elapsed = time.time() - start_time
                print(f"[{index}/{total}] {question_id}: ERROR - {e} ({elapsed:.1f}s)")
                return {
                    "id": question_id,
                    "question": question_text,
                    "answer": "Unable to answer with confidence based on the provided documents.",
                    "answer_value": "is_blank",
                    "answer_unit": answer_unit,
                    "ref_id": "is_blank",
                    "ref_url": "is_blank",
                    "supporting_materials": "is_blank",
                    "explanation": f"Error during processing: {str(e)}",
                }

    async def evaluate_all(self, questions_df: pd.DataFrame) -> pd.DataFrame:
        """Evaluate all questions from the input DataFrame."""
        if self.pipeline is None:
            raise RuntimeError("Pipeline not initialized. Call initialize() first.")

        total = len(questions_df)
        print(f"\n{'='*60}")
        print(f"Starting evaluation of {total} questions")
        print(f"{'='*60}\n")

        tasks = []
        for idx, row in questions_df.iterrows():
            task = self._process_single_question(
                question_id=row["id"],
                question_text=row["question"],
                answer_unit=row.get("answer_unit", ""),
                index=len(tasks) + 1,
                total=total,
            )
            tasks.append(task)

        results = await asyncio.gather(*tasks)
        return pd.DataFrame(results)


# =============================================================================
# Config Loading
# =============================================================================

def load_config_file(config_path: str) -> dict:
    """Load configuration from a Python config file."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    spec = importlib.util.spec_from_file_location("config", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    config = {}
    for key in dir(module):
        if not key.startswith("_"):
            config[key] = getattr(module, key)
    return config


# =============================================================================
# Main
# =============================================================================

async def main(args) -> None:
    """Main evaluation workflow."""
    start_time = time.time()

    # Build config from file or CLI args
    if args.config:
        config = load_config_file(args.config)
    else:
        config = {
            "llm_provider": args.provider,
            "hf_model_id": args.hf_model,
            "hf_dtype": args.hf_dtype,
            "embedding_model": "hf_local" if args.provider == "hf_local" else "jina",
            "embedding_model_id": args.embed_model,
            "db": args.db,
            "table_prefix": "wattbot",
            "max_concurrent": 2 if args.provider == "hf_local" else 5,
            "top_k": TOP_K_CHUNKS,
        }

    # Validate input
    input_file = Path(args.input)
    if not input_file.exists():
        print(f"ERROR: Input file not found: {args.input}")
        sys.exit(1)

    print(f"[main] Loading questions from {args.input}...")
    questions_df = pd.read_csv(input_file)
    print(f"[main] Loaded {len(questions_df)} questions")

    # Initialize and run
    evaluator = WattBotEvaluator(config)
    await evaluator.initialize()
    predictions_df = await evaluator.evaluate_all(questions_df)

    # Save predictions
    output_file = Path(args.output)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    predictions_df.to_csv(output_file, index=False, quoting=csv.QUOTE_ALL)
    print(f"\n[main] Saved predictions to {args.output}")

    # Compute WattBot score
    print(f"\n{'='*60}")
    print("WATTBOT SCORE RESULTS")
    print(f"{'='*60}\n")

    try:
        solution_df = pd.read_csv(input_file)
        submission_df = pd.read_csv(output_file)
        overall_score = compute_wattbot_score(solution_df, submission_df)
        print(f"\n{'='*60}")
        print(f"FINAL WATTBOT SCORE: {overall_score:.4f}")
        print(f"{'='*60}")
    except Exception as e:
        print(f"ERROR computing score: {e}")
        print("You can manually run: python scripts/score.py data/train_QA.csv artifacts/submission.csv")

    elapsed = time.time() - start_time
    print(f"\n[main] Total evaluation time: {elapsed:.1f}s ({elapsed/60:.1f} minutes)")


def cli() -> None:
    """Parse command line arguments and run evaluation."""
    parser = argparse.ArgumentParser(
        description="Run WattBot evaluation (provider-agnostic).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Local HF model (default):
    python scripts/run_wattbot_eval.py

    # With config file:
    python scripts/run_wattbot_eval.py --config configs/hf_qwen7b.py

    # Override model via CLI:
    python scripts/run_wattbot_eval.py --provider hf_local --hf-model Qwen/Qwen2.5-7B-Instruct
        """
    )
    parser.add_argument("--input", "-i", default=DEFAULT_INPUT, help=f"Input questions CSV (default: {DEFAULT_INPUT})")
    parser.add_argument("--output", "-o", default=DEFAULT_OUTPUT, help=f"Output submission CSV (default: {DEFAULT_OUTPUT})")
    parser.add_argument("--db", default=DEFAULT_DB, help=f"Vector database path (default: {DEFAULT_DB})")
    parser.add_argument("--config", "-c", default=None, help="Python config file (overrides other CLI args)")
    parser.add_argument("--provider", default=DEFAULT_PROVIDER, choices=["hf_local", "openrouter", "openai", "bedrock"])
    parser.add_argument("--hf-model", default=DEFAULT_HF_MODEL, help="HuggingFace model ID")
    parser.add_argument("--hf-dtype", default=DEFAULT_HF_DTYPE, choices=["bf16", "fp16", "auto"])
    parser.add_argument("--embed-model", default=DEFAULT_EMBED_MODEL, help="Embedding model ID")

    args = parser.parse_args()
    asyncio.run(main(args))


if __name__ == "__main__":
    cli()
