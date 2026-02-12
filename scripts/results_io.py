"""Chunked results I/O for experiment outputs.

When a results.json file exceeds ~50-100 MB it causes problems with git.
This module provides helpers to save results in fixed-size chunks
(e.g. 50 questions per file) and transparently reload them.

File layout (inside an experiment directory)::

    results_chunk_000.json   # questions 0-49
    results_chunk_001.json   # questions 50-99
    ...

A single ``results.json`` is still supported for backwards compatibility —
:func:`load_results` will fall back to it when no chunk files are found.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

CHUNK_SIZE = 50  # default questions per chunk file
_CHUNK_GLOB = "results_chunk_*.json"
_CHUNK_RE = re.compile(r"^results_chunk_(\d+)\.json$")


# ── loading ──────────────────────────────────────────────────────────────

def load_results(experiment_dir: Path) -> list[dict]:
    """Load all question results from *experiment_dir*.

    Looks for ``results_chunk_*.json`` first (sorted by chunk index).
    Falls back to a monolithic ``results.json`` if no chunks exist.
    Raises ``FileNotFoundError`` when neither format is present.
    """
    experiment_dir = Path(experiment_dir)

    chunk_files = sorted(
        (p for p in experiment_dir.glob(_CHUNK_GLOB) if _CHUNK_RE.match(p.name)),
        key=lambda p: int(_CHUNK_RE.match(p.name).group(1)),
    )

    if chunk_files:
        results: list[dict] = []
        for cf in chunk_files:
            with open(cf) as f:
                results.extend(json.load(f))
        return results

    # Fallback: monolithic results.json
    mono = experiment_dir / "results.json"
    if mono.exists():
        with open(mono) as f:
            return json.load(f)

    raise FileNotFoundError(
        f"No results found in {experiment_dir} "
        f"(looked for {_CHUNK_GLOB} and results.json)"
    )


# ── saving ───────────────────────────────────────────────────────────────

def save_results_chunked(
    results: list[dict],
    output_dir: Path,
    chunk_size: int = CHUNK_SIZE,
) -> list[Path]:
    """Save *results* as chunked JSON files under *output_dir*.

    Returns the list of chunk file paths that were written.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    for i in range(0, len(results), chunk_size):
        chunk = results[i : i + chunk_size]
        chunk_path = output_dir / f"results_chunk_{i // chunk_size:03d}.json"
        with open(chunk_path, "w") as f:
            json.dump(chunk, f, indent=2)
        written.append(chunk_path)

    return written
