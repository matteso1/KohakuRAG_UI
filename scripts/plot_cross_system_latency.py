#!/usr/bin/env python3
"""
Cross-System Latency Comparison

Compares per-question latency (total, retrieval, generation) for models
that have been benchmarked on multiple hardware systems (e.g. GB10,
PowerEdge, Bedrock).

Generates:
  1. Grouped bar chart of mean latency per model, split by system
  2. Box-plot of per-question latency distributions per model & system
  3. Stacked bar chart showing retrieval vs generation time breakdown

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
    return sorted(shared)


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

def plot_mean_latency_comparison(data: dict, shared_models: list[str],
                                  systems: list[str], output_dir: Path):
    """Plot 1: Grouped bar chart of mean total latency per model & system."""
    n_models = len(shared_models)
    n_systems = len(systems)
    if n_models == 0:
        print("  No shared models for mean latency comparison")
        return

    fig, ax = plt.subplots(figsize=(max(10, n_models * 2.5), 7))

    x = np.arange(n_models)
    width = 0.7 / n_systems

    for i, system in enumerate(systems):
        means = []
        stds = []
        for model in shared_models:
            if model in data.get(system, {}):
                info = data[system][model]
                means.append(info["avg_latency"])
                stds.append(float(np.std(info["latencies"])))
            else:
                means.append(0)
                stds.append(0)

        offset = (i - (n_systems - 1) / 2) * width
        gpu = data.get(system, {}).get(shared_models[0], {}).get("gpu_name", system)
        label = f"{system} ({gpu})" if gpu and gpu != system else system
        bars = ax.bar(x + offset, means, width, label=label,
                      color=get_system_color(system), alpha=0.85,
                      yerr=stds, capsize=3,
                      error_kw={"linewidth": 1, "color": "#333"})

        for bar, mean in zip(bars, means):
            if mean > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                        f"{mean:.1f}s", ha="center", va="bottom", fontsize=8,
                        fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(shared_models, rotation=30, ha="right", fontsize=10)
    style_axis(ax, "Mean Per-Question Latency by System",
               "Model", "Latency (seconds)")
    ax.legend(fontsize=10)

    plt.tight_layout()
    out_path = output_dir / "cross_system_latency.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")


def plot_latency_distributions(data: dict, shared_models: list[str],
                                systems: list[str], output_dir: Path):
    """Plot 2: Box-plot of per-question latency distributions."""
    n_models = len(shared_models)
    if n_models == 0:
        print("  No shared models for distribution plot")
        return

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


def plot_latency_breakdown(data: dict, shared_models: list[str],
                            systems: list[str], output_dir: Path):
    """Plot 3: Stacked bar chart showing retrieval vs generation time."""
    n_models = len(shared_models)
    n_systems = len(systems)
    if n_models == 0:
        print("  No shared models for breakdown plot")
        return

    fig, ax = plt.subplots(figsize=(max(10, n_models * 2.5), 7))

    x = np.arange(n_models)
    width = 0.7 / n_systems

    for i, system in enumerate(systems):
        retrievals = []
        generations = []
        for model in shared_models:
            if model in data.get(system, {}):
                info = data[system][model]
                retrievals.append(info["avg_retrieval"])
                generations.append(info["avg_generation"])
            else:
                retrievals.append(0)
                generations.append(0)

        offset = (i - (n_systems - 1) / 2) * width
        gpu = data.get(system, {}).get(shared_models[0], {}).get("gpu_name", system)
        label_base = f"{system} ({gpu})" if gpu and gpu != system else system

        ax.bar(x + offset, retrievals, width, label=f"{label_base} - Retrieval",
               color=get_system_color(system), alpha=0.6, hatch="//")
        ax.bar(x + offset, generations, width, bottom=retrievals,
               label=f"{label_base} - Generation",
               color=get_system_color(system), alpha=0.9)

        # Label totals
        for j, (r, g) in enumerate(zip(retrievals, generations)):
            total = r + g
            if total > 0:
                ax.text(x[j] + offset, total + 0.3, f"{total:.1f}s",
                        ha="center", va="bottom", fontsize=7, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(shared_models, rotation=30, ha="right", fontsize=10)
    style_axis(ax, "Latency Breakdown: Retrieval vs Generation by System",
               "Model", "Time (seconds)")
    ax.legend(fontsize=8, loc="upper left")

    plt.tight_layout()
    out_path = output_dir / "cross_system_latency_breakdown.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")


def plot_all_models_by_system(data: dict, systems: list[str], output_dir: Path):
    """Plot 4: Bar chart of ALL models across all systems (not just shared).

    Useful when systems have mostly different model sets.
    """
    # Collect all (system, model, avg_latency) tuples
    entries = []
    for system in systems:
        for model, info in data.get(system, {}).items():
            entries.append({
                "system": system,
                "model": model,
                "avg_latency": info["avg_latency"],
                "gpu_name": info.get("gpu_name", ""),
                "n_questions": info["n_questions"],
            })

    if not entries:
        print("  No data for all-models plot")
        return

    # Sort by system then latency
    entries.sort(key=lambda e: (e["system"], e["avg_latency"]))

    fig, ax = plt.subplots(figsize=(12, max(7, len(entries) * 0.5)))

    labels = []
    latencies = []
    colors = []
    for e in entries:
        label = f"{e['model']}  [{e['system']}]"
        labels.append(label)
        latencies.append(e["avg_latency"])
        colors.append(get_system_color(e["system"]))

    y_pos = np.arange(len(entries))
    bars = ax.barh(y_pos, latencies, color=colors, edgecolor="white",
                   linewidth=1.2, height=0.65)

    for bar, lat in zip(bars, latencies):
        ax.text(bar.get_width() + max(latencies) * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{lat:.1f}s", va="center", fontsize=9, fontweight="bold")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel("Average Latency per Question (seconds)", fontsize=11)
    style_axis(ax, "Average Per-Question Latency: All Models by System",
               "", "")
    ax.set_ylabel("")

    # Legend for system colors
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=get_system_color(s), label=s) for s in systems
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=10)

    plt.tight_layout()
    out_path = output_dir / "cross_system_latency_all_models.png"
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

    systems = sorted(data.keys())
    print(f"Found {len(systems)} system(s): {systems}")
    for system in systems:
        models = list(data[system].keys())
        print(f"  {system}: {len(models)} model(s) — {models}")

    shared_models = find_shared_models(data)
    if shared_models:
        print(f"\nModels on multiple systems: {shared_models}")
    else:
        print("\nNo models shared across systems (yet)")

    print_summary(data, shared_models, systems)

    print("Generating plots...")
    if shared_models:
        plot_mean_latency_comparison(data, shared_models, systems, output_dir)
        plot_latency_distributions(data, shared_models, systems, output_dir)
        plot_latency_breakdown(data, shared_models, systems, output_dir)

    # Always generate the all-models overview
    plot_all_models_by_system(data, systems, output_dir)

    n_plots = (3 if shared_models else 0) + 1
    print(f"\n{n_plots} plot(s) saved to {output_dir}/")


if __name__ == "__main__":
    main()
