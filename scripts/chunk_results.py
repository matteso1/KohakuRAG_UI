#!/usr/bin/env python3
"""Split existing monolithic results.json files into chunks.

Scans for results.json files under the experiments directory and converts
each into a set of ``results_chunk_NNN.json`` files (default 50 questions
per chunk).  The original ``results.json`` is removed after a successful
split so git no longer tracks the oversized file.

Usage
-----
    # Chunk all results.json under artifacts/experiments/
    python scripts/chunk_results.py

    # Chunk a single experiment directory
    python scripts/chunk_results.py artifacts/experiments/PowerEdge/train_QA/qwen7b-bench

    # Custom chunk size
    python scripts/chunk_results.py --chunk-size 25

    # Dry run (show what would happen, don't write)
    python scripts/chunk_results.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow imports from scripts/
sys.path.insert(0, str(Path(__file__).resolve().parent))

from results_io import save_results_chunked, CHUNK_SIZE, _CHUNK_GLOB


def chunk_one(results_json: Path, chunk_size: int, *, dry_run: bool = False) -> None:
    """Split a single results.json into chunk files."""
    with open(results_json) as f:
        results = json.load(f)

    n = len(results)
    n_chunks = (n + chunk_size - 1) // chunk_size

    if n_chunks <= 1:
        print(f"  SKIP {results_json}  ({n} questions fits in 1 chunk)")
        return

    size_mb = results_json.stat().st_size / (1024 * 1024)
    print(f"  SPLIT {results_json}")
    print(f"        {n} questions, {size_mb:.1f} MB -> {n_chunks} chunks of <={chunk_size}")

    if dry_run:
        return

    out_dir = results_json.parent

    # Write chunks
    written = save_results_chunked(results, out_dir, chunk_size=chunk_size)
    for p in written:
        print(f"        wrote {p.name}")

    # Verify round-trip before deleting original
    reloaded: list[dict] = []
    for p in written:
        with open(p) as f:
            reloaded.extend(json.load(f))

    if len(reloaded) != n:
        print(f"        ERROR: round-trip mismatch ({len(reloaded)} != {n}), keeping original")
        return

    results_json.unlink()
    print(f"        removed original results.json")


def main() -> None:
    parser = argparse.ArgumentParser(description="Split large results.json files into chunks")
    parser.add_argument(
        "path",
        nargs="?",
        default=None,
        help="Experiment directory or specific results.json (default: scan all under artifacts/experiments/)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=CHUNK_SIZE,
        help=f"Questions per chunk file (default: {CHUNK_SIZE})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would happen without writing files",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent

    if args.path:
        target = Path(args.path)
        if target.is_file() and target.name == "results.json":
            targets = [target]
        elif target.is_dir():
            targets = sorted(target.glob("**/results.json"))
        else:
            print(f"Error: {target} is not a results.json file or directory")
            sys.exit(1)
    else:
        experiments_dir = project_root / "artifacts" / "experiments"
        if not experiments_dir.exists():
            print(f"No experiments directory found at {experiments_dir}")
            sys.exit(1)
        targets = sorted(experiments_dir.glob("**/results.json"))

    if not targets:
        print("No results.json files found.")
        return

    print(f"Found {len(targets)} results.json file(s)")
    if args.dry_run:
        print("[DRY RUN — no files will be written or deleted]\n")
    print()

    for rj in targets:
        # Skip directories that already have chunk files
        existing_chunks = list(rj.parent.glob(_CHUNK_GLOB))
        if existing_chunks:
            print(f"  SKIP {rj}  (already has {len(existing_chunks)} chunk files)")
            continue
        chunk_one(rj, args.chunk_size, dry_run=args.dry_run)

    print("\nDone.")


if __name__ == "__main__":
    main()
