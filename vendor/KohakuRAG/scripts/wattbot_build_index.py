"""Build a KohakuVault-backed index for WattBot documents.

If the docs directory (structured JSONs from parsed PDFs) doesn't exist,
the script will automatically download PDFs using URLs from metadata.csv,
parse them into structured JSON, and then index them.

Usage:
    cd vendor/KohakuRAG
    kogine run scripts/wattbot_build_index.py --config configs/jinav4/index.py
"""

import asyncio
import csv
import json
import sys
from pathlib import Path
from typing import Iterable
from urllib.parse import urlsplit

from kohakurag import (
    DocumentIndexer,
    dict_to_payload,
    text_to_payload,
)
from kohakurag.datastore import KVaultNodeStore
from kohakurag.embeddings import JinaEmbeddingModel, JinaV4EmbeddingModel

# ============================================================================
# GLOBAL CONFIGURATION
# These defaults can be overridden by KohakuEngine config injection or CLI args
# ============================================================================

metadata = "../../data/metadata.csv"
docs_dir = "../../data/corpus"
db = "../../data/embeddings/wattbot.db"
table_prefix = "wattbot"
use_citations = False

# PDF fetching settings (used when docs_dir doesn't exist)
pdf_dir = "../../data/pdfs"  # Where to cache downloaded PDFs

# Embedding settings
embedding_model = "jina"  # Options: "jina" (v3), "jinav4"
embedding_dim = None  # For JinaV4: 128, 256, 512, 1024, 2048
embedding_task = "retrieval"  # For JinaV4: "retrieval", "text-matching", "code"

# Paragraph embedding mode
# Options:
#   - "averaged": Paragraph embedding = average of sentence embeddings (default for backward compat)
#   - "full": Paragraph embedding = direct embedding of paragraph text
#   - "both": Store both averaged (main) and full (separate table) - allows runtime toggle
paragraph_embedding_mode = "both"


# ============================================================================
# METADATA & DOCUMENT LOADING
# ============================================================================


def load_metadata(path: Path) -> dict[str, dict[str, str]]:
    records: dict[str, dict[str, str]] = {}
    with path.open(newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            records[row["id"]] = row
    return records


def iter_structured_docs(docs_dir: Path) -> Iterable[dict]:
    if not docs_dir.exists():
        return []
    for json_path in sorted(docs_dir.glob("*.json")):
        data = json.loads(json_path.read_text(encoding="utf-8"))
        yield data


def iter_documents(
    docs_dir: Path | None,
    metadata: dict[str, dict[str, str]],
    use_citations: bool,
):
    if docs_dir and docs_dir.exists():
        json_files = list(docs_dir.glob("*.json"))
        if json_files:
            for data in iter_structured_docs(docs_dir):
                yield dict_to_payload(data)
            return
    if use_citations:
        for doc_id, info in metadata.items():
            citation = info.get("citation") or info.get("title") or doc_id
            yield text_to_payload(
                document_id=doc_id,
                title=info.get("title", doc_id),
                text=citation,
                metadata={
                    "document_id": doc_id,
                    "document_title": info.get("title", doc_id),
                    "url": info.get("url"),
                    "type": info.get("type"),
                    "year": info.get("year"),
                },
            )
        return

    raise SystemExit(
        "No structured JSON docs found and --use-citations not set.\n"
        "The auto-fetch step may have failed. Check the errors above."
    )


# ============================================================================
# PDF FETCHING (auto-download when docs_dir is missing)
# ============================================================================


def _clean_url(url: str) -> str:
    return (url or "").strip()


def _is_pdf_url(url: str) -> bool:
    cleaned = _clean_url(url)
    if not cleaned:
        return False
    parts = urlsplit(cleaned)
    path_lower = parts.path.lower().rstrip("/")
    if path_lower.endswith(".pdf"):
        return True
    if parts.netloc.endswith("arxiv.org") and path_lower.startswith("/pdf/"):
        return True
    return False


async def _has_pdf_mime(url: str, client) -> bool:
    try:
        resp = await client.head(url, follow_redirects=True, timeout=30)
        ctype = resp.headers.get("Content-Type", "").lower()
        return "pdf" in ctype
    except Exception:
        return False


async def _download_pdf(url: str, dest: Path, client) -> None:
    if dest.exists():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    headers = {"User-Agent": "KohakuRAG/0.0.1"}
    resp = await client.get(url, headers=headers, timeout=120)
    resp.raise_for_status()
    dest.write_bytes(resp.content)


async def fetch_and_parse_pdfs(
    metadata_records: dict[str, dict[str, str]],
    target_docs_dir: Path,
    raw_pdf_dir: Path,
) -> int:
    """Download PDFs from metadata URLs and parse into structured JSON.

    Returns the number of documents successfully processed.
    """
    try:
        import httpx
    except ImportError:
        print(
            "\nERROR: httpx is required to download PDFs.\n"
            "  Install with: uv pip install httpx\n",
            file=sys.stderr,
        )
        return 0

    try:
        from kohakurag.parsers import payload_to_dict
        from kohakurag.pdf_utils import pdf_to_document_payload
    except ImportError:
        print(
            "\nERROR: PDF parsing requires additional dependencies.\n"
            "  Install with: uv pip install pymupdf\n",
            file=sys.stderr,
        )
        return 0

    target_docs_dir.mkdir(parents=True, exist_ok=True)
    raw_pdf_dir.mkdir(parents=True, exist_ok=True)

    processed = 0
    skipped = 0
    total = len(metadata_records)

    print(f"\nAuto-fetching PDFs for {total} documents from metadata URLs...")
    print(f"  PDF cache : {raw_pdf_dir}")
    print(f"  Output dir: {target_docs_dir}\n")

    async with httpx.AsyncClient() as client:
        for doc_id, info in metadata_records.items():
            url = _clean_url(info.get("url") or "")
            title = info.get("title") or doc_id

            if not url:
                skipped += 1
                print(f"  [skip] {doc_id}: no URL in metadata", file=sys.stderr)
                continue

            if not _is_pdf_url(url) and not await _has_pdf_mime(url, client):
                skipped += 1
                print(f"  [skip] {doc_id}: URL not a PDF ({url})", file=sys.stderr)
                continue

            pdf_path = raw_pdf_dir / f"{doc_id}.pdf"
            json_path = target_docs_dir / f"{doc_id}.json"

            # Skip if already parsed
            if json_path.exists():
                processed += 1
                continue

            try:
                await _download_pdf(url, pdf_path, client)

                payload = pdf_to_document_payload(
                    pdf_path,
                    doc_id=doc_id,
                    title=title,
                    metadata={
                        "url": url,
                        "type": info.get("type"),
                        "year": info.get("year"),
                    },
                )

                json_path.write_text(
                    json.dumps(payload_to_dict(payload), ensure_ascii=False),
                    encoding="utf-8",
                )

                processed += 1
                print(f"  [ok] {doc_id} -> {json_path}")

            except Exception as exc:
                skipped += 1
                print(f"  [error] {doc_id}: {exc}", file=sys.stderr)

    print(f"\nFetch complete: {processed} processed, {skipped} skipped")
    return processed


# ============================================================================
# EMBEDDER FACTORY
# ============================================================================


def create_embedder():
    """Create embedder based on module-level config."""
    if embedding_model == "jinav4":
        print(f"Using JinaV4 embeddings (dim={embedding_dim}, task={embedding_task})")
        return JinaV4EmbeddingModel(
            truncate_dim=embedding_dim or 1024,
            task=embedding_task,
        )
    else:
        print("Using JinaV3 embeddings (default)")
        return JinaEmbeddingModel()


# ============================================================================
# MAIN INDEXING PIPELINE
# ============================================================================


async def main() -> None:
    """Build the hierarchical index from documents."""
    metadata_path = Path(metadata)
    docs_path = Path(docs_dir)

    # Load metadata
    if not metadata_path.exists():
        raise SystemExit(
            f"Metadata file not found: {metadata_path}\n"
            f"Ensure data/metadata.csv exists at the repo root."
        )
    metadata_records = load_metadata(metadata_path)

    # Auto-fetch PDFs if docs directory doesn't exist or is empty
    if not docs_path.exists() or not list(docs_path.glob("*.json")):
        print(f"Docs directory not found or empty: {docs_path}")
        print("Attempting to download and parse PDFs from metadata URLs...")
        fetched = await fetch_and_parse_pdfs(
            metadata_records, docs_path, Path(pdf_dir)
        )
        if fetched == 0:
            print(
                "\nNo documents could be fetched. You can still build a lightweight "
                "index using citation text from metadata.csv by setting use_citations = True "
                "in the config or passing --use-citations.",
                file=sys.stderr,
            )
            if not use_citations:
                raise SystemExit("No documents available for indexing.")

    # Load documents
    documents = list(iter_documents(docs_path, metadata_records, use_citations))
    total_docs = len(documents)

    if not total_docs:
        raise SystemExit("No documents found to index.")

    # Setup indexer and datastore
    db_path = Path(db)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    # Create embedder using factory
    embedder = create_embedder()
    indexer = DocumentIndexer(
        embedding_model=embedder,
        paragraph_embedding_mode=paragraph_embedding_mode,
    )
    print(f"Paragraph embedding mode: {paragraph_embedding_mode}")

    store: KVaultNodeStore | None = None  # Lazy init after first document
    total_nodes = 0

    # Index each document and upsert nodes
    for idx, payload in enumerate(documents, start=1):
        print(f"[{idx}/{total_docs}] indexing {payload.document_id}...", flush=True)

        # Build hierarchical tree and compute embeddings
        nodes = await indexer.index(payload)
        if not nodes:
            print(f"  -> no nodes generated, skipping.", flush=True)
            continue

        # Initialize store on first document (infer dimensions)
        if store is None:
            store = KVaultNodeStore(
                db_path,
                table_prefix=table_prefix,
                dimensions=nodes[0].embedding.shape[0],
            )

        # Persist nodes to SQLite + sqlite-vec
        await store.upsert_nodes(nodes)
        total_nodes += len(nodes)
        print(
            f"  -> added {len(nodes)} nodes (running total {total_nodes})", flush=True
        )

    print(f"Indexed {len(documents)} documents with {total_nodes} nodes into {db_path}")


if __name__ == "__main__":
    asyncio.run(main())
