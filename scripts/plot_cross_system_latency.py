#!/usr/bin/env python3
"""
Cross-System Latency Comparison

Compares per-question latency (total, retrieval, generation) for models
that have been benchmarked on multiple hardware systems (e.g. GB10,
PowerEdge, Bedrock).

Generates:
  1. Horizontal stacked-bar chart of ALL models (retrieval + generation),
     sorted by latency fastest-first, color-coded by system.
  2. Box-plot of per-question latency distributions for shared models.

Usage:
    python scripts/plot_cross_system_latency.py
    python scripts/plot_cross_system_latency.py --experiments artifacts/experiments \
        --datafile test_solutions --output artifacts/plots
"""

import argparse
import json
import sys
from pathlib import Path

try:
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    import numpy as np
except ImportError:
    print("matplotlib and numpy required: pip install matplotlib numpy")
    sys.exit(1)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from results_io import load_results  # noqa: E402

# Active parameter counts (B) for sorting by inference cost.
# For dense models this equals total params; for MoE it's the active subset.
ACTIVE_PARAMS: dict[str, float] = {
    # Bedrock / API
    "claude-3-haiku":       8,
    "claude-3-5-haiku":     8,
    "claude-3-5-sonnet":    70,
    "claude-3-7-sonnet":    70,
    "nova-pro":             40,
    "llama3-3-70b":         70,
    "llama3-1-70b":         70,
    "llama4-scout":         17,
    "llama4-maverick":      17,
    "deepseek":             70,
    # Local HF
    "qwen2.5-7b":          7,
    "qwen2.5-14b":         14,
    "qwen2.5-32b":         32,
    "qwen2.5-72b":         72,
    "qwen1.5-110b":        110,
    "qwen3-30b-a3b":       3.3,
    "qwen3-next-80b-a3b":  3,
}


def _active_params(model_name: str) -> float:
    """Return active param count for *model_name*, or inf if unknown."""
    low = model_name.lower()
    for key, params in ACTIVE_PARAMS.items():
        if key in low:
            return params
    return float("inf")


# ============================================================================
# Data Loading
# ============================================================================

def discover_systems(experiments_dir: Path) -> list[str]:
    """Return sorted list of system directory names under experiments_dir."""
    systems = []
    for child in sorted(experiments_dir.iterdir()):
        if child.is_dir() and not child.name.startswith("."):
            systems.append(child.name)
    return systems


def load_per_question_latency(experiments_dir: Path, datafile: str | None = None,
                               ) -> dict[str, dict[str, dict]]:
    """Load per-question latency data grouped by system and model.

    Returns::

        {
            "PowerEdge": {
                "qwen7b-bench": {
                    "latencies": [...],
                    "generation": [...],
                    "retrieval": [...],
                    "avg_latency": float,
                    "avg_generation": float,
                    "avg_retrieval": float,
                    "gpu_name": str,
                },
                ...
            },
            "GB10": { ... },
        }
    """
    result: dict[str, dict[str, dict]] = {}

    for system_dir in sorted(experiments_dir.iterdir()):
        if not system_dir.is_dir() or system_dir.name.startswith("."):
            continue
        system_name = system_dir.name

        # Walk datafile subdirs
        if datafile:
            search_dirs = [system_dir / datafile]
        else:
            search_dirs = [d for d in system_dir.iterdir() if d.is_dir()]

        for ds_dir in search_dirs:
            if not ds_dir.exists():
                continue
            for model_dir in sorted(ds_dir.iterdir()):
                if not model_dir.is_dir():
                    continue

                model_name = model_dir.name
                if "ensemble" in model_name.lower():
                    continue

                # Load per-question results
                try:
                    items = load_results(model_dir)
                except FileNotFoundError:
                    continue

                latencies = []
                generation_times = []
                retrieval_times = []
                gpu_name = ""

                # Try to get GPU name from summary.json
                summary_path = model_dir / "summary.json"
                if summary_path.exists():
                    try:
                        with open(summary_path) as f:
                            summary = json.load(f)
                        gpu_name = summary.get("hardware", {}).get("gpu_name", "")
                    except (json.JSONDecodeError, OSError):
                        pass

                for item in items:
                    if item.get("error"):
                        continue
                    lat = item.get("latency_seconds")
                    if lat is not None and lat > 0:
                        latencies.append(lat)
                    gen = item.get("generation_seconds")
                    if gen is not None and gen > 0:
                        generation_times.append(gen)
                    ret = item.get("retrieval_seconds")
                    if ret is not None and ret > 0:
                        retrieval_times.append(ret)

                if not latencies:
                    print(f"  Skipping {system_name}/{model_name}: no latency data")
                    continue

                if system_name not in result:
                    result[system_name] = {}

                result[system_name][model_name] = {
                    "latencies": latencies,
                    "generation": generation_times,
                    "retrieval": retrieval_times,
                    "avg_latency": float(np.mean(latencies)),
                    "avg_generation": float(np.mean(generation_times)) if generation_times else 0,
                    "avg_retrieval": float(np.mean(retrieval_times)) if retrieval_times else 0,
                    "gpu_name": gpu_name,
                    "n_questions": len(latencies),
                }

    return result


def find_shared_models(data: dict[str, dict[str, dict]]) -> list[str]:
    """Find models that appear in at least 2 systems."""
    model_systems: dict[str, set[str]] = {}
    for system, models in data.items():
        for model in models:
            model_systems.setdefault(model, set()).add(system)

    shared = [m for m, systems in model_systems.items() if len(systems) >= 2]
    return sorted(shared, key=_active_params)


def find_comparable_systems(data: dict[str, dict[str, dict]]) -> list[str]:
    """Return systems that share at least one model with another system.

    Systems with entirely disjoint model sets (e.g. Bedrock vs local GPU)
    are excluded since cross-system latency comparison is meaningless.
    """
    model_systems: dict[str, set[str]] = {}
    for system, models in data.items():
        for model in models:
            model_systems.setdefault(model, set()).add(system)

    connected: set[str] = set()
    for systems in model_systems.values():
        if len(systems) >= 2:
            connected.update(systems)

    return sorted(connected)


# ============================================================================
# Plot Helpers
# ============================================================================

SYSTEM_COLORS = {
    "PowerEdge": "#3b82f6",   # blue
    "GB10": "#f59e0b",        # amber
    "Bedrock": "#10b981",     # green
}


def get_system_color(system: str) -> str:
    return SYSTEM_COLORS.get(system, "#6b7280")


def style_axis(ax, title, xlabel, ylabel):
    ax.set_title(title, fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ============================================================================
# Plots
# ============================================================================

def plot_latency_overview(data: dict, all_systems: list[str], output_dir: Path):
    """Horizontal stacked-bar chart: ALL models, ALL systems, sorted by latency.

    Each bar shows retrieval (hatched) + generation (solid) breakdown.
    Fastest model at the top; slowest at the bottom.  Color = system.
    """
    from matplotlib.patches import Patch

    entries = []
    for system in all_systems:
        for model, info in data.get(system, {}).items():
            entries.append({
                "system": system,
                "model": model,
                "avg_latency": info["avg_latency"],
                "avg_retrieval": info["avg_retrieval"],
                "avg_generation": info["avg_generation"],
                "gpu_name": info.get("gpu_name", ""),
            })

    if not entries:
        print("  No data for latency overview")
        return

    # Sort by latency: fastest at top (reversed for barh so top = index 0)
    entries.sort(key=lambda e: e["avg_latency"])

    n = len(entries)
    fig, ax = plt.subplots(figsize=(13, max(6, n * 0.45)))

    labels = []
    retrievals = []
    generations = []
    colors = []
    for e in entries:
        label = f"{e['model']}  [{e['system']}]"
        labels.append(label)
        retrievals.append(e["avg_retrieval"])
        generations.append(e["avg_generation"])
        colors.append(get_system_color(e["system"]))

    y_pos = np.arange(n)

    # Retrieval (hatched) then generation (solid) stacked horizontally
    bars_ret = ax.barh(y_pos, retrievals, color=colors, alpha=0.5,
                       edgecolor="white", linewidth=0.8, height=0.65,
                       hatch="//", label="Retrieval")
    bars_gen = ax.barh(y_pos, generations, left=retrievals, color=colors,
                       alpha=0.9, edgecolor="white", linewidth=0.8,
                       height=0.65, label="Generation")

    # Annotate total latency to the right of each bar
    max_lat = max(e["avg_latency"] for e in entries) if entries else 1
    for i, e in enumerate(entries):
        total = e["avg_latency"]
        ax.text(total + max_lat * 0.012,
                y_pos[i],
                f"{total:.1f}s",
                va="center", fontsize=9, fontweight="bold")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=10)
    ax.invert_yaxis()  # fastest at top
    ax.set_xlabel("Average Latency per Question (seconds)", fontsize=11)
    style_axis(ax, "Per-Question Latency by Model and System", "", "")
    ax.set_ylabel("")

    # Build legend: system colors + retrieval/generation pattern
    seen_systems = dict.fromkeys(e["system"] for e in entries)
    legend_elements = [
        Patch(facecolor=get_system_color(s), label=s) for s in seen_systems
    ]
    legend_elements.append(Patch(facecolor="#999", alpha=0.5, hatch="//",
                                 label="Retrieval"))
    legend_elements.append(Patch(facecolor="#999", alpha=0.9,
                                 label="Generation"))
    ax.legend(handles=legend_elements, loc="lower right", fontsize=9)

    # Add a bit of right margin for the annotations
    ax.set_xlim(right=max_lat * 1.12)

    plt.tight_layout()
    out_path = output_dir / "cross_system_latency.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")


def plot_latency_distributions(data: dict, shared_models: list[str],
                                systems: list[str], output_dir: Path):
    """Box-plot of per-question latency distributions for shared models."""
    n_models = len(shared_models)
    if n_models == 0:
        print("  No shared models for distribution plot")
        return

    # Sort shared models by latency (mean across systems)
    def _mean_latency(model):
        lats = [data[s][model]["avg_latency"]
                for s in systems if model in data.get(s, {})]
        return sum(lats) / len(lats) if lats else 0

    shared_models = sorted(shared_models, key=_mean_latency)

    fig, axes = plt.subplots(1, n_models, figsize=(max(6, n_models * 4), 7),
                             sharey=True, squeeze=False)
    axes = axes[0]

    for idx, model in enumerate(shared_models):
        ax = axes[idx]
        box_data = []
        box_labels = []
        box_colors = []

        for system in systems:
            if model in data.get(system, {}):
                info = data[system][model]
                box_data.append(info["latencies"])
                gpu = info.get("gpu_name", system)
                label = f"{system}\n({gpu})" if gpu and gpu != system else system
                box_labels.append(label)
                box_colors.append(get_system_color(system))

        if not box_data:
            continue

        bp = ax.boxplot(box_data, labels=box_labels, patch_artist=True,
                        widths=0.6, showfliers=True,
                        flierprops={"marker": ".", "markersize": 3, "alpha": 0.5})
        for patch, color in zip(bp["boxes"], box_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)

        ax.set_title(model, fontsize=11, fontweight="bold")
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if idx == 0:
            ax.set_ylabel("Latency (seconds)", fontsize=11)

    fig.suptitle("Per-Question Latency Distribution by System",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    out_path = output_dir / "cross_system_latency_distribution.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")


def print_summary(data: dict, shared_models: list[str], systems: list[str]):
    """Print a summary table of cross-system latency comparison."""
    print(f"\n{'=' * 100}")
    print("CROSS-SYSTEM LATENCY COMPARISON")
    print(f"{'=' * 100}")

    # Header
    header = f"{'Model':<25}"
    for system in systems:
        header += f" | {system:>12} {'(gen)':>8} {'(ret)':>8}"
    print(header)
    print("-" * 100)

    all_models = set()
    for system_data in data.values():
        all_models.update(system_data.keys())

    for model in sorted(all_models):
        row = f"{model:<25}"
        for system in systems:
            if model in data.get(system, {}):
                info = data[system][model]
                row += f" | {info['avg_latency']:>10.2f}s {info['avg_generation']:>7.2f}s {info['avg_retrieval']:>7.2f}s"
            else:
                row += f" | {'—':>12} {'—':>8} {'—':>8}"
        print(row)

    # Speedup for shared models
    if shared_models and len(systems) >= 2:
        print(f"\n{'=' * 100}")
        print("SPEEDUP RATIOS (System A / System B)")
        print(f"{'=' * 100}")
        for i, sys_a in enumerate(systems):
            for sys_b in systems[i + 1:]:
                print(f"\n  {sys_a} vs {sys_b}:")
                for model in shared_models:
                    if model in data.get(sys_a, {}) and model in data.get(sys_b, {}):
                        lat_a = data[sys_a][model]["avg_latency"]
                        lat_b = data[sys_b][model]["avg_latency"]
                        if lat_a > 0 and lat_b > 0:
                            ratio = lat_a / lat_b
                            faster = sys_b if ratio > 1 else sys_a
                            speedup = max(ratio, 1 / ratio)
                            print(f"    {model}: {faster} is {speedup:.2f}x faster "
                                  f"({lat_a:.1f}s vs {lat_b:.1f}s)")

    print(f"{'=' * 100}\n")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Compare per-question latency across hardware systems")
    parser.add_argument("--experiments", "-e", default="artifacts/experiments",
                        help="Experiments directory")
    parser.add_argument("--output", "-o", default="artifacts/plots",
                        help="Output directory for plots")
    parser.add_argument("--datafile", "-d", default=None,
                        help="Filter to this datafile subfolder "
                             "(e.g. 'test_solutions'). Default: include all.")
    args = parser.parse_args()

    experiments_dir = Path(args.experiments)
    output_dir = Path(args.output)
    if args.datafile:
        output_dir = output_dir / args.datafile
    output_dir.mkdir(parents=True, exist_ok=True)

    if not experiments_dir.exists():
        print(f"Error: experiments directory not found: {experiments_dir}")
        sys.exit(1)

    print("Loading per-question latency data...")
    data = load_per_question_latency(experiments_dir, datafile=args.datafile)

    if not data:
        print("No latency data found!")
        sys.exit(1)

    all_systems = sorted(data.keys())
    print(f"Found {len(all_systems)} system(s): {all_systems}")
    for system in all_systems:
        models = list(data[system].keys())
        print(f"  {system}: {len(models)} model(s) — {models}")

    shared_models = find_shared_models(data)
    # Only include systems that share at least one model with another system
    systems = find_comparable_systems(data)
    skipped = sorted(set(all_systems) - set(systems))
    if skipped:
        print(f"\nSkipping system(s) with no overlapping models: {skipped}")
    if shared_models:
        print(f"Models on multiple systems: {shared_models}")
        print(f"Comparable systems: {systems}")
    else:
        print("\nNo models shared across systems (yet)")

    print_summary(data, shared_models, systems)

    print("Generating plots...")

    # Plot 1: Consolidated overview — all models, all systems, sorted by latency
    plot_latency_overview(data, all_systems, output_dir)

    # Plot 2: Distribution box-plots for shared models only
    if shared_models:
        plot_latency_distributions(data, shared_models, systems, output_dir)

    n_plots = 1 + (1 if shared_models else 0)
    print(f"\n{n_plots} plot(s) saved to {output_dir}/")


if __name__ == "__main__":
    main()
