"""
Demo: End-to-End RAG with AWS Bedrock

This script proves the whole pipeline works. It:
1. Takes a question (e.g., "What is the carbon footprint of LLMs?")
2. Searches our local vector database for relevant research papers
3. Sends the context + question to Claude 3 on AWS Bedrock
4. Prints out the answer with citations

It's a great starting point for understanding how the pieces fit together.

Prerequisites:
- You need to be logged into AWS SSO: `aws sso login --profile bedrock_nils`
- You need the WattBot index built: `python KohakuRAG/scripts/wattbot_build_index.py ...`

Usage:
    python scripts/demo_bedrock_rag.py --question "How much water does ChatGPT consume?"
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
    """
    Orchestrates the full RAG process for a single question.
    
    This function:
    1. Sets up the Bedrock client (Claude 3)
    2. Loads the Jina embeddings and SQLite vector store
    3. Retrieves the top-k most relevant text chunks
    4. Sends everything to Bedrock to synthesize an answer
    """
    print(f"\n{'='*60}")
    print(f"Question: {question}")
    print(f"{'='*60}\n")

    # 1. Validation
    # We can't do anything if the vector DB doesn't exist.
    db = Path(db_path)
    if not db.exists():
        return {
            "error": f"Database missing at {db_path}",
            "help": "Please run the indexing script first! Check README.md for instructions."
        }

    # 2. Setup Bedrock models
    # We need two models: one for 'planning' (query expansion) and one for 'answering'.
    # In this demo, we use the same model (Claude 3 Haiku) for both because it's efficient.
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

    # 3. Load Retrieval Components
    # Jina embeddings turn text into vectors. KVaultNodeStore holds our indexed vectors.
    print("Loading Jina embeddings...")
    embedder = JinaEmbeddingModel()

    print("Loading vector store...")
    store = KVaultNodeStore(
        db,
        table_prefix=table_prefix,
        dimensions=None,
        paragraph_search_mode="averaged",
    )

    # 4. Build the Pipeline
    # The Query Planner breaks complex questions into simpler search queries.
    planner = SimpleLLMQueryPlanner(planner_chat, max_queries=3)

    print("Initializing RAG pipeline...")
    pipeline = RAGPipeline(
        store=store,
        embedder=embedder,
        chat_model=answer_chat,
        planner=planner,
    )

    # 5. Execute Retrieval
    # This searches the database for relevant content.
    print(f"Retrieving top-{top_k} context snippets...")
    retrieval = await pipeline.retrieve(question, top_k=top_k)

    print(f"\nFound {len(retrieval.snippets)} snippets from {len(retrieval.matches)} matches")

    # Show what we found (sanity check)
    print(f"\n{'-'*60}")
    print("Retrieved Context (Top 3):")
    print(f"{'-'*60}")
    for i, snippet in enumerate(retrieval.snippets[:3], 1):
        doc_id = snippet.metadata.get("doc_id", "unknown")
        text_preview = snippet.text[:200] + "..." if len(snippet.text) > 200 else snippet.text
        print(f"\n[{i}] {doc_id}")
        print(f"    {text_preview}")

    # Generate answer
    print(f"\n{'-'*60}")
    print("Generating answer with Bedrock...")
    print(f"{'-'*60}")

# Standard WattBot Prompts
    SYSTEM_PROMPT = """
You must answer strictly based on the provided context snippets.
Do NOT use external knowledge or assumptions.
If the context does not clearly support an answer, you must output the literal string "is_blank" for both answer_value and ref_id.
""".strip()

    USER_TEMPLATE = """
You will be given a question and context snippets taken from documents.
You must follow these rules:
- Use only the provided context; do not rely on external knowledge.
- If the context does not clearly support an answer, use "is_blank".

Additional info (JSON): {additional_info_json}

Question: {question}

Context:
{context}

Return STRICT JSON with the following keys, in this order:
- explanation          (1–3 sentences explaining how the context supports the answer; or "is_blank")
- answer               (short sentence in natural language)
- answer_value         (string with ONLY the numeric or categorical value, or "is_blank")
- ref_id               (list of document ids from the context used as evidence; or "is_blank")

JSON Answer:
""".strip()

    # Pass additional_info needed by the template
    additional_info = {
        "answer_unit": "",  # Demo doesn't use units
        "question_id": "demo-001"
    }

    try:
        result = await pipeline.run_qa(
            question=question, 
            top_k=top_k,
            system_prompt=SYSTEM_PROMPT,
            user_template=USER_TEMPLATE,
            additional_info=additional_info,
        )
        
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
