#!/usr/bin/env python3
"""
WattBot Evaluation Pipeline
============================

Runs the Bedrock RAG pipeline on train_QA.csv and computes the official WattBot
competition score. This script is designed to be a sanity check to measure how
well our RAG system performs on the labeled training data.

Authors: Nils Matteson, Blaise (KohakuRAG_UI team)
Last Updated: 2026-01-23

Usage:
    # Make sure you're logged into AWS SSO first:
    aws sso login --profile bedrock_nils

    # Then run the evaluation:
    python scripts/run_wattbot_eval.py

    # Optional: specify custom paths
    python scripts/run_wattbot_eval.py --input data/train_QA.csv --output artifacts/submission.csv

Output:
    - artifacts/submission.csv: The generated predictions in Kaggle format
    - Console output: Detailed scoring breakdown

WattBot Scoring Rubric:
    - answer_value (75%): Exact match with ±0.1% tolerance for numerics
    - ref_id (15%): Jaccard overlap between predicted and ground truth doc IDs
    - is_NA (10%): Correctly identifying unanswerable questions with "is_blank"

Dependencies:
    - AWS SSO authentication (bedrock_nils profile)
    - Jina embeddings model
    - Pre-built wattbot.db vector database (see README.md for setup)
"""

import argparse
import asyncio
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any

# Add paths for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "KohakuRAG" / "src"))

import pandas as pd

from llm_bedrock import BedrockChatModel
from kohakurag import RAGPipeline
from kohakurag.datastore import KVaultNodeStore
from kohakurag.embeddings import JinaEmbeddingModel

# Import the official scorer
sys.path.insert(0, str(Path(__file__).parent))
from score import score as compute_wattbot_score

# =============================================================================
# Configuration
# =============================================================================

# Default paths (relative to project root)
DEFAULT_INPUT = "data/train_QA.csv"
DEFAULT_OUTPUT = "artifacts/submission.csv"
DEFAULT_DB = "artifacts/wattbot.db"

# AWS Bedrock settings
DEFAULT_PROFILE = "bedrock_nils"
DEFAULT_REGION = "us-east-2"
DEFAULT_MODEL = "us.anthropic.claude-3-haiku-20240307-v1:0"

# Concurrency settings to avoid AWS throttling
MAX_CONCURRENT_REQUESTS = 5  # Be conservative with Bedrock API limits
REQUEST_DELAY_SECONDS = 0.5  # Small delay between requests

# RAG retrieval settings
TOP_K_CHUNKS = 5  # Number of context chunks to retrieve per question


# =============================================================================
# Prompts (matching the WattBot competition format)
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
- explanation          (1–3 sentences explaining how the context supports the answer; or "is_blank")
- answer               (short natural-language response, e.g. "1438 lbs", "Water consumption", "TRUE")
- answer_value         (ONLY the numeric or categorical value, e.g. "1438", "Water consumption", "1"; or "is_blank")
- ref_id               (list of document ids from the context used as evidence; or "is_blank")
- ref_url              (list of URLs for the cited documents; or "is_blank")
- supporting_materials (verbatim quote, table reference, or figure reference from the cited document; or "is_blank")

JSON Answer:
""".strip()


# =============================================================================
# Pipeline Setup
# =============================================================================

class WattBotEvaluator:
    """
    Orchestrates batch evaluation of the RAG pipeline on WattBot questions.
    
    This class manages:
    - Loading the RAG pipeline components (embeddings, vector store, LLM)
    - Processing questions with concurrency limits
    - Generating submission CSV in the required format
    
    Example:
        evaluator = WattBotEvaluator(db_path="artifacts/wattbot.db")
        await evaluator.initialize()
        results = await evaluator.evaluate_all(questions_df)
    """
    
    def __init__(
        self,
        db_path: str = DEFAULT_DB,
        profile_name: str = DEFAULT_PROFILE,
        region_name: str = DEFAULT_REGION,
        model_id: str = DEFAULT_MODEL,
        max_concurrent: int = MAX_CONCURRENT_REQUESTS,
    ):
        """
        Initialize the evaluator with configuration.
        
        Args:
            db_path: Path to the wattbot.db SQLite vector database
            profile_name: AWS SSO profile for Bedrock authentication
            region_name: AWS region where Bedrock is available
            model_id: Bedrock model ID (Claude 3 Haiku recommended for cost)
            max_concurrent: Maximum concurrent API requests to Bedrock
        """
        self.db_path = Path(db_path)
        self.profile_name = profile_name
        self.region_name = region_name
        self.model_id = model_id
        self.max_concurrent = max_concurrent
        
        # These get initialized in initialize()
        self.pipeline: RAGPipeline | None = None
        self.semaphore: asyncio.Semaphore | None = None
        
    async def initialize(self) -> None:
        """
        Load all pipeline components. Call this before evaluate_all().
        
        Raises:
            FileNotFoundError: If wattbot.db doesn't exist
        """
        if not self.db_path.exists():
            raise FileNotFoundError(
                f"Vector database not found at {self.db_path}. "
                "Please run the indexing script first. See README.md for setup."
            )
        
        print(f"[init] Loading Bedrock model: {self.model_id}")
        chat = BedrockChatModel(
            model_id=self.model_id,
            profile_name=self.profile_name,
            region_name=self.region_name,
            system_prompt=SYSTEM_PROMPT,
        )
        
        print("[init] Loading Jina embeddings...")
        embedder = JinaEmbeddingModel()
        
        print(f"[init] Loading vector store from {self.db_path}...")
        store = KVaultNodeStore(
            self.db_path,
            table_prefix="wattbot",
            dimensions=None,
            paragraph_search_mode="averaged",
        )
        
        print("[init] Building RAG pipeline...")
        self.pipeline = RAGPipeline(
            store=store,
            embedder=embedder,
            chat_model=chat,
            planner=None,  # No query expansion for evaluation (deterministic)
        )
        
        self.semaphore = asyncio.Semaphore(self.max_concurrent)
        print("[init] Ready!")
        
    async def _process_single_question(
        self,
        question_id: str,
        question_text: str,
        answer_unit: str,
        index: int,
        total: int,
    ) -> dict[str, Any]:
        """
        Process a single question through the RAG pipeline.
        
        Args:
            question_id: Unique ID from train_QA.csv (e.g., "q003")
            question_text: The actual question to answer
            answer_unit: Expected unit from the ground truth (for context)
            index: Current question index (for progress logging)
            total: Total number of questions (for progress logging)
            
        Returns:
            Dict with all WattBot submission columns:
            id, question, answer, answer_value, answer_unit, ref_id, ref_url,
            supporting_materials, explanation
        """
        async with self.semaphore:
            start_time = time.time()
            
            try:
                # Build additional info for the prompt template
                additional_info = {
                    "answer_unit": answer_unit,
                    "question_id": question_id,
                }
                
                # Run the RAG pipeline
                result = await self.pipeline.run_qa(
                    question=question_text,
                    top_k=TOP_K_CHUNKS,
                    system_prompt=SYSTEM_PROMPT,
                    user_template=USER_TEMPLATE,
                    additional_info=additional_info,
                )
                
                # Extract fields from the structured answer
                answer_value = result.answer.answer_value
                answer_nl = getattr(result.answer, 'answer', answer_value)  # Natural language answer
                ref_id = result.answer.ref_id
                ref_url = getattr(result.answer, 'ref_url', 'is_blank')
                supporting = getattr(result.answer, 'supporting_materials', 'is_blank')
                explanation = result.answer.explanation
                
                # Format ref_id as JSON list string (required by Score.py)
                if isinstance(ref_id, list):
                    ref_id_str = json.dumps(ref_id)
                elif ref_id == "is_blank" or not ref_id:
                    ref_id_str = "is_blank"
                else:
                    ref_id_str = json.dumps([ref_id])
                
                # Format ref_url as JSON list string
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
                
                # Return blank answers on error (per WattBot spec for unanswerable)
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
            
            finally:
                # Small delay to be nice to AWS
                await asyncio.sleep(REQUEST_DELAY_SECONDS)
    
    async def evaluate_all(self, questions_df: pd.DataFrame) -> pd.DataFrame:
        """
        Evaluate all questions from the input DataFrame.
        
        Args:
            questions_df: DataFrame with columns: id, question, answer_unit
            
        Returns:
            DataFrame with predictions: id, answer_value, answer_unit, ref_id, explanation
        """
        if self.pipeline is None:
            raise RuntimeError("Pipeline not initialized. Call initialize() first.")
        
        total = len(questions_df)
        print(f"\n{'='*60}")
        print(f"Starting evaluation of {total} questions")
        print(f"{'='*60}\n")
        
        # Create tasks for all questions
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
        
        # Run all tasks with concurrency control
        results = await asyncio.gather(*tasks)
        
        return pd.DataFrame(results)


# =============================================================================
# Main Entry Point
# =============================================================================

async def main(input_path: str, output_path: str, db_path: str) -> None:
    """
    Main evaluation workflow.
    
    1. Load questions from input CSV
    2. Initialize the RAG pipeline
    3. Process all questions
    4. Save predictions to output CSV
    5. Compute and display the WattBot score
    """
    start_time = time.time()
    
    # Validate input file exists
    input_file = Path(input_path)
    if not input_file.exists():
        print(f"ERROR: Input file not found: {input_path}")
        sys.exit(1)
    
    # Load questions
    print(f"[main] Loading questions from {input_path}...")
    questions_df = pd.read_csv(input_file)
    print(f"[main] Loaded {len(questions_df)} questions")
    
    # Initialize evaluator
    evaluator = WattBotEvaluator(db_path=db_path)
    await evaluator.initialize()
    
    # Run evaluation
    predictions_df = await evaluator.evaluate_all(questions_df)
    
    # Save predictions
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    predictions_df.to_csv(output_file, index=False, quoting=csv.QUOTE_ALL)
    print(f"\n[main] Saved predictions to {output_path}")
    
    # Compute WattBot score
    print(f"\n{'='*60}")
    print("WATTBOT SCORE RESULTS")
    print(f"{'='*60}\n")
    
    try:
        # Load solution (ground truth) and submission (our predictions)
        solution_df = pd.read_csv(input_file)
        submission_df = pd.read_csv(output_file)
        
        # The score function prints detailed breakdown and returns overall score
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
        description="Run WattBot evaluation on train_QA.csv and compute competition score.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run with defaults:
    python scripts/run_wattbot_eval.py

    # Custom input/output:
    python scripts/run_wattbot_eval.py --input data/train_QA.csv --output results/my_submission.csv

    # Use a different database:
    python scripts/run_wattbot_eval.py --db artifacts/custom.db
        """
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        default=DEFAULT_INPUT,
        help=f"Path to input questions CSV (default: {DEFAULT_INPUT})"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=DEFAULT_OUTPUT,
        help=f"Path to output submission CSV (default: {DEFAULT_OUTPUT})"
    )
    parser.add_argument(
        "--db",
        type=str,
        default=DEFAULT_DB,
        help=f"Path to wattbot.db vector database (default: {DEFAULT_DB})"
    )
    
    args = parser.parse_args()
    
    # Run the async main function
    asyncio.run(main(
        input_path=args.input,
        output_path=args.output,
        db_path=args.db,
    ))


if __name__ == "__main__":
    cli()
