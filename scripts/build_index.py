#!/usr/bin/env python3
"""Build the WattBot vector index without requiring KohakuEngine (kogine).

This is a standalone wrapper around vendor/KohakuRAG/scripts/wattbot_build_index.py
that applies config file settings directly, replacing the kogine config-injection
mechanism.

Usage:
    # Build with JinaV4 embeddings (matches experiment configs)
    python scripts/build_index.py --config vendor/KohakuRAG/configs/jinav4/index.py

    # Build with defaults (Jina v3, artifacts/wattbot.db)
    python scripts/build_index.py

    # Override specific settings
    python scripts/build_index.py --db artifacts/wattbot_custom.db --embedding-model jinav4

    # Use citation-based indexing (no docs/ folder needed, uses metadata.csv titles)
    python scripts/build_index.py --use-citations

Prerequisites:
    - data/metadata.csv must exist
    - For full indexing: artifacts/docs/ (or artifacts/docs_with_images/) with
      structured JSON files from the document parsing pipeline
    - For citation-only: just metadata.csv (lighter index, lower quality)
"""

import argparse
import asyncio
import importlib.util
import sys
from pathlib import Path

# Setup paths
_repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_repo_root / "vendor" / "KohakuRAG" / "src"))

# Import the build script as a module so we can patch its globals
_build_script = _repo_root / "vendor" / "KohakuRAG" / "scripts" / "wattbot_build_index.py"


def load_config_file(config_path: str) -> dict:
    """Load a Python config file and extract its module-level variables."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    spec = importlib.util.spec_from_file_location("config", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # Extract config variables (skip dunder, callables, and imports)
    config = {}
    for key in dir(module):
        if key.startswith("_"):
            continue
        val = getattr(module, key)
        if callable(val) and not isinstance(val, (bool, int, float, str)):
            continue
        config[key] = val
    return config


def main():
    parser = argparse.ArgumentParser(
        description="Build WattBot vector index (standalone, no kogine required)"
    )
    parser.add_argument(
        "--config", "-c",
        default=None,
        help="Path to config file (e.g., vendor/KohakuRAG/configs/jinav4/index.py). "
             "Overrides default build settings."
    )
    parser.add_argument(
        "--db",
        default=None,
        help="Output database path (default: artifacts/wattbot.db)"
    )
    parser.add_argument(
        "--docs-dir",
        default=None,
        help="Directory with structured JSON docs (default: artifacts/docs)"
    )
    parser.add_argument(
        "--metadata",
        default=None,
        help="Path to metadata.csv (default: data/metadata.csv)"
    )
    parser.add_argument(
        "--embedding-model",
        choices=["jina", "jinav4"],
        default=None,
        help="Embedding model to use (default: jina)"
    )
    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=None,
        help="Embedding dimension for JinaV4 (128, 256, 512, 1024, 2048)"
    )
    parser.add_argument(
        "--table-prefix",
        default=None,
        help="SQLite table prefix (default: wattbot)"
    )
    parser.add_argument(
        "--use-citations",
        action="store_true",
        help="Build index from citation text in metadata.csv (no docs/ needed)"
    )
    args = parser.parse_args()

    # Load the build script as a module
    spec = importlib.util.spec_from_file_location("wattbot_build_index", _build_script)
    build_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(build_mod)

    # Apply config file overrides
    if args.config:
        config = load_config_file(args.config)
        print(f"[build_index] Applying config from {args.config}")
        for key, val in config.items():
            if hasattr(build_mod, key):
                old_val = getattr(build_mod, key)
                if old_val != val:
                    print(f"  {key}: {old_val!r} -> {val!r}")
                setattr(build_mod, key, val)

    # Apply CLI overrides (take precedence over config file)
    cli_overrides = {
        "db": args.db,
        "docs_dir": args.docs_dir,
        "metadata": args.metadata,
        "embedding_model": args.embedding_model,
        "embedding_dim": args.embedding_dim,
        "table_prefix": args.table_prefix,
        "use_citations": args.use_citations if args.use_citations else None,
    }
    for key, val in cli_overrides.items():
        if val is not None:
            setattr(build_mod, key, val)
            print(f"  [CLI] {key} = {val!r}")

    # Resolve paths relative to repo root
    db_path = Path(build_mod.db)
    if not db_path.is_absolute():
        db_path = _repo_root / db_path
    build_mod.db = str(db_path)

    metadata_path = Path(build_mod.metadata)
    if not metadata_path.is_absolute():
        metadata_path = _repo_root / metadata_path
    build_mod.metadata = str(metadata_path)

    docs_path = Path(build_mod.docs_dir)
    if not docs_path.is_absolute():
        docs_path = _repo_root / docs_path
    build_mod.docs_dir = str(docs_path)

    # Validate prerequisites
    if not metadata_path.exists():
        print(f"\nERROR: metadata.csv not found at {metadata_path}")
        print("This file is required for indexing. It should contain document")
        print("IDs, titles, and URLs for the WattBot reference corpus.")
        sys.exit(1)

    if not docs_path.exists() and not build_mod.use_citations:
        print(f"\nWARNING: Documents directory not found: {docs_path}")
        print("You have two options:")
        print()
        print("  1. Run the document fetch/parse pipeline first:")
        print(f"     mkdir -p {docs_path}")
        print("     (then parse PDFs into structured JSON — see vendor/KohakuRAG/docs/wattbot.md)")
        print()
        print("  2. Use citation-based indexing (lighter, uses metadata.csv only):")
        print("     python scripts/build_index.py --config vendor/KohakuRAG/configs/jinav4/index.py --use-citations")
        print()
        sys.exit(1)

    # Create output directory
    db_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print("BUILD INDEX")
    print(f"{'='*60}")
    print(f"  Database    : {build_mod.db}")
    print(f"  Table prefix: {build_mod.table_prefix}")
    print(f"  Docs dir    : {build_mod.docs_dir}")
    print(f"  Metadata    : {build_mod.metadata}")
    print(f"  Embeddings  : {build_mod.embedding_model}", end="")
    if build_mod.embedding_model == "jinav4":
        print(f" (dim={build_mod.embedding_dim}, task={build_mod.embedding_task})")
    else:
        print()
    print(f"  Citations   : {build_mod.use_citations}")
    print(f"{'='*60}\n")

    # Run the indexer
    asyncio.run(build_mod.main())

    print(f"\nDone! Database written to: {build_mod.db}")
    print(f"You can now run experiments with:")
    print(f"  python scripts/run_experiment.py --config vendor/KohakuRAG/configs/hf_qwen7b.py")


if __name__ == "__main__":
    main()
