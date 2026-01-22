"""Demo script: Ask a single question using KohakuRAG + AWS Bedrock.

This script demonstrates end-to-end RAG with Bedrock as the LLM backend:
1. Loads the KohakuRAG pipeline from the submodule
2. Uses BedrockChatModel for LLM calls
3. Retrieves context from the vector store
4. Generates a structured answer

Usage:
    python scripts/demo_bedrock_rag.py --question "What is the carbon footprint of LLMs?"
    
Prerequisites:
    - AWS SSO login: aws sso login --profile bedrock_nils
    - Built index: Requires artifacts/wattbot.db (see KohakuRAG docs)
"""

import argparse
import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

# Add paths for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "KohakuRAG" / "src"))

from llm_bedrock import BedrockChatModel

# Import from KohakuRAG submodule
from kohakurag import RAGPipeline
from kohakurag.datastore import KVaultNodeStore
from kohakurag.embeddings import JinaEmbeddingModel


# Default configuration
DEFAULT_DB = "artifacts/wattbot.db"
DEFAULT_TABLE_PREFIX = "wattbot"
DEFAULT_PROFILE = "bedrock_nils"
DEFAULT_REGION = "us-east-2"
DEFAULT_MODEL = "us.anthropic.claude-3-haiku-20240307-v1:0"

# System prompts (simplified from wattbot_answer.py)
PLANNER_SYSTEM_PROMPT = """You are a query expansion assistant.

Given a user question, generate 2-3 alternative search queries that would help find relevant information.
Return ONLY the queries, one per line. No explanations or numbering."""

ANSWER_SYSTEM_PROMPT = """You are a helpful research assistant specializing in AI and sustainability.

Given a question and relevant context snippets, provide a clear, accurate answer.
Always cite your sources using [doc_id] notation.
If the context doesn't contain enough information, say so honestly."""


class SimpleLLMQueryPlanner:
    """Simple LLM-backed query planner for Bedrock."""

    def __init__(self, chat: BedrockChatModel, max_queries: int = 3):
        self._chat = chat
        self._max_queries = max_queries

    async def plan(self, question: str) -> list[str]:
        """Generate retrieval queries from a question."""
        queries = [question]  # Always include original

        try:
            prompt = f"Generate alternative search queries for: {question}"
            response = await self._chat.complete(prompt)

            # Parse response into queries
            for line in response.strip().split("\n"):
                line = line.strip().lstrip("0123456789.-) ")
                if line and line not in queries:
                    queries.append(line)
                if len(queries) >= self._max_queries:
                    break
        except Exception as e:
            print(f"Warning: Query expansion failed: {e}")

        return queries[:self._max_queries]


async def run_rag_query(
    question: str,
    db_path: str = DEFAULT_DB,
    table_prefix: str = DEFAULT_TABLE_PREFIX,
    profile_name: str = DEFAULT_PROFILE,
    region_name: str = DEFAULT_REGION,
    model_id: str = DEFAULT_MODEL,
    top_k: int = 5,
) -> dict:
    """Run a single RAG query using Bedrock.
    
    Args:
        question: The question to answer
        db_path: Path to the KohakuRAG SQLite database
        table_prefix: Table prefix in the database
        profile_name: AWS SSO profile name
        region_name: AWS region
        model_id: Bedrock model ID
        top_k: Number of context snippets to retrieve
    
    Returns:
        Dictionary with answer, sources, and metadata
    """
    print(f"\n{'='*60}")
    print(f"Question: {question}")
    print(f"{'='*60}\n")

    # Check if database exists
    db = Path(db_path)
    if not db.exists():
        return {
            "error": f"Database not found: {db_path}",
            "help": "Run the KohakuRAG indexing pipeline first. See KohakuRAG/README.md"
        }

    # Create Bedrock chat models
    print("Creating Bedrock chat models...")
    planner_chat = BedrockChatModel(
        model_id=model_id,
        profile_name=profile_name,
        region_name=region_name,
        system_prompt=PLANNER_SYSTEM_PROMPT,
    )
    
    answer_chat = BedrockChatModel(
        model_id=model_id,
        profile_name=profile_name,
        region_name=region_name,
        system_prompt=ANSWER_SYSTEM_PROMPT,
    )

    # Create embedder
    print("Loading Jina embeddings...")
    embedder = JinaEmbeddingModel()

    # Create datastore
    print("Loading vector store...")
    store = KVaultNodeStore(
        db,
        table_prefix=table_prefix,
        dimensions=None,
        paragraph_search_mode="averaged",
    )

    # Create query planner
    planner = SimpleLLMQueryPlanner(planner_chat, max_queries=3)

    # Create RAG pipeline
    print("Initializing RAG pipeline...")
    pipeline = RAGPipeline(
        store=store,
        embedder=embedder,
        chat_model=answer_chat,
        planner=planner,
    )

    # Run retrieval
    print(f"Retrieving top-{top_k} context snippets...")
    retrieval = await pipeline.retrieve(question, top_k=top_k)

    print(f"\nFound {len(retrieval.snippets)} snippets from {len(retrieval.matches)} matches")

    # Display context snippets
    print(f"\n{'-'*60}")
    print("Retrieved Context:")
    print(f"{'-'*60}")
    for i, snippet in enumerate(retrieval.snippets[:3], 1):
        doc_id = snippet.node.metadata.get("doc_id", "unknown")
        text_preview = snippet.text[:200] + "..." if len(snippet.text) > 200 else snippet.text
        print(f"\n[{i}] {doc_id}")
        print(f"    {text_preview}")

    # Generate answer
    print(f"\n{'-'*60}")
    print("Generating answer with Bedrock...")
    print(f"{'-'*60}")

    try:
        result = await pipeline.run_qa(question, top_k=top_k)
        
        print(f"\nAnswer: {result.answer.answer}")
        print(f"\nExplanation: {result.answer.explanation}")
        print(f"\nSources: {', '.join(result.answer.ref_id)}")

        return {
            "question": question,
            "answer": result.answer.answer,
            "answer_value": result.answer.answer_value,
            "explanation": result.answer.explanation,
            "sources": result.answer.ref_id,
            "num_snippets": len(retrieval.snippets),
        }
    except Exception as e:
        print(f"\nError generating answer: {e}")
        return {
            "question": question,
            "error": str(e),
            "num_snippets": len(retrieval.snippets),
        }


def main():
    parser = argparse.ArgumentParser(
        description="Ask a question using KohakuRAG + AWS Bedrock"
    )
    parser.add_argument(
        "--question", "-q",
        type=str,
        default="What is the carbon footprint of large language models?",
        help="Question to ask",
    )
    parser.add_argument(
        "--db",
        type=str,
        default=DEFAULT_DB,
        help="Path to KohakuRAG database",
    )
    parser.add_argument(
        "--profile",
        type=str,
        default=DEFAULT_PROFILE,
        help="AWS SSO profile name",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help="Bedrock model ID",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of context snippets to retrieve",
    )

    args = parser.parse_args()

    result = asyncio.run(run_rag_query(
        question=args.question,
        db_path=args.db,
        profile_name=args.profile,
        model_id=args.model,
        top_k=args.top_k,
    ))

    print(f"\n{'='*60}")
    print("Result Summary")
    print(f"{'='*60}")
    for key, value in result.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
