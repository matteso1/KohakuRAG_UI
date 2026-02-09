#!/usr/bin/env python3
"""
Model Size vs. Performance/Latency/Cost Plots

Generates publication-quality plots for ML+X meeting (2/17/2026):
  1. Model Size vs. WattBot Component Scores
  2. Model Size vs. Latency
  3. Model Size vs. Total Cost
  4. Model Size vs. Overall Score (bubble chart with cost as size)

Model sizes: Open-source models have confirmed sizes. Proprietary models
(Claude, Nova) are estimates -- marked with (*) on the plots.

Usage:
    python scripts/plot_model_size.py
    python scripts/plot_model_size.py --experiments artifacts/experiments --output artifacts/plots
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


# ============================================================================
# Model Size Registry
# ============================================================================
# Sources:
#   - Open-source models: published parameter counts
#   - Proprietary models: community estimates (marked estimated=True)
#
# For MoE models (Llama 4, DeepSeek), we report ACTIVE parameters since
# that's what determines compute cost per token. Total params noted in comments.

MODEL_SIZES = {
    # model_id substring -> (display_name, active_params_B, estimated, notes)
    "claude-3-haiku": ("Claude 3 Haiku", 8, True, "Anthropic undisclosed; community est. ~8B"),
    "claude-3-5-haiku": ("Claude 3.5 Haiku", 8, True, "Anthropic undisclosed; community est. ~8B"),
    "claude-3-5-sonnet": ("Claude 3.5 Sonnet", 70, True, "Anthropic undisclosed; community est. ~70B"),
    "claude-3-7-sonnet": ("Claude 3.7 Sonnet", 70, True, "Anthropic undisclosed; community est. ~70B"),
    "nova-pro": ("Nova Pro", 40, True, "Amazon undisclosed; rough est. ~40B"),
    "llama3-3-70b": ("Llama 3.3 70B", 70, False, "Open-source, confirmed"),
    "llama3-1-70b": ("Llama 3.1 70B", 70, False, "Open-source, confirmed"),
    "llama4-scout-17b": ("Llama 4 Scout", 17, False, "17B active, 109B total MoE"),
    "llama4-maverick-17b": ("Llama 4 Maverick", 17, False, "17B active, 400B total MoE"),
    "mistral-small": ("Mistral Small", 24, False, "Open-source, confirmed 24B"),
    "deepseek": ("DeepSeek R1 (distill)", 70, False, "Llama 70B distill variant"),
}


def match_model_size(model_id: str) -> tuple[str, int, bool] | None:
    """Match a Bedrock model ID to our size registry. Returns (name, size_B, estimated)."""
    model_lower = model_id.lower()
    for key, (name, size, estimated, _notes) in MODEL_SIZES.items():
        if key in model_lower:
            return name, size, estimated
    return None


def load_experiments(experiments_dir: Path) -> list[dict]:
    """Load all experiment summaries, skipping ensembles and duplicates."""
    experiments = []
    seen_models = {}  # model_id -> best experiment (by score)

    for summary_path in sorted(experiments_dir.glob("*/summary.json")):
        try:
            with open(summary_path) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            print(f"  Skipping {summary_path.parent.name}: {e}")
            continue

        name = data.get("name", summary_path.parent.name)
        model_id = data.get("model_id", "")

        # Skip ensembles and experiments without a model
        if not model_id or "ensemble" in name.lower():
            continue

        # Skip experiments with suspiciously low scores (likely broken runs)
        if data.get("overall_score", 0) < 0.15:
            print(f"  Skipping {name}: score too low ({data.get('overall_score', 0):.3f}), likely broken run")
            continue

        # Skip experiments with high error rates (>10% of questions failed)
        n_questions = data.get("num_questions", 0)
        error_count = data.get("error_count", 0)
        if n_questions > 0 and error_count / n_questions > 0.1:
            print(f"  Skipping {name}: high error rate ({error_count}/{n_questions})")
            continue

        # Flag experiments with suspiciously high latency (retry storms)
        avg_latency = data.get("avg_latency_seconds", 0)
        latency_suspect = avg_latency > 60  # >60s avg = likely throttling/retry issues

        # Match to model size
        size_info = match_model_size(model_id)
        if size_info is None:
            print(f"  Skipping {name}: no size info for {model_id}")
            continue

        display_name, size_b, estimated = size_info

        if latency_suspect:
            print(f"  Warning: {name} has high avg latency ({avg_latency:.0f}s) -- likely retry/throttle issues")

        entry = {
            "experiment": name,
            "model_id": model_id,
            "display_name": display_name,
            "size_b": size_b,
            "size_estimated": estimated,
            "overall_score": data.get("overall_score", 0),
            "value_accuracy": data.get("value_accuracy", 0),
            "ref_overlap": data.get("ref_overlap", 0),
            "na_accuracy": data.get("na_accuracy", 0),
            "avg_latency": avg_latency,
            "latency_suspect": latency_suspect,
            "total_cost": data.get("estimated_cost_usd", 0),
            "input_tokens": data.get("input_tokens", 0),
            "output_tokens": data.get("output_tokens", 0),
        }

        # Keep the best CLEAN run per model.
        # Prefer runs with normal latency. Only use high-latency runs if
        # no clean run exists for that model.
        if model_id in seen_models:
            existing = seen_models[model_id]
            # Prefer clean latency over suspect latency
            if existing["latency_suspect"] and not entry["latency_suspect"]:
                seen_models[model_id] = entry
            elif entry["latency_suspect"] and not existing["latency_suspect"]:
                pass  # keep existing clean run
            elif entry["overall_score"] > existing["overall_score"]:
                seen_models[model_id] = entry
        else:
            seen_models[model_id] = entry

    experiments = list(seen_models.values())
    experiments.sort(key=lambda x: x["size_b"])
    return experiments


# ============================================================================
# Plot Helpers
# ============================================================================

# Color palette: distinct colors for each model family
FAMILY_COLORS = {
    "Claude": "#6366f1",      # Indigo
    "Llama": "#f59e0b",       # Amber
    "Mistral": "#10b981",     # Emerald
    "DeepSeek": "#ef4444",    # Red
    "Nova": "#ec4899",        # Pink
}


def get_color(name: str) -> str:
    for family, color in FAMILY_COLORS.items():
        if family.lower() in name.lower():
            return color
    return "#6b7280"  # Gray fallback


def annotate_points(ax, xs, ys, labels, estimated_flags):
    """Add model name labels to scatter points."""
    for x, y, label, est in zip(xs, ys, labels, estimated_flags):
        display = f"{label}*" if est else label
        ax.annotate(
            display,
            (x, y),
            xytext=(8, 6),
            textcoords="offset points",
            fontsize=8,
            ha="left",
            va="bottom",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7, edgecolor="none"),
        )


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

def plot_size_vs_scores(experiments: list[dict], output_dir: Path):
    """Plot 1: Model Size vs. WattBot Component Scores (multi-panel)."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    metrics = [
        ("overall_score", "Overall WattBot Score", axes[0, 0]),
        ("value_accuracy", "Value Accuracy (75% weight)", axes[0, 1]),
        ("ref_overlap", "Reference Overlap (15% weight)", axes[1, 0]),
        ("na_accuracy", "NA Accuracy (10% weight)", axes[1, 1]),
    ]

    sizes = [e["size_b"] for e in experiments]
    names = [e["display_name"] for e in experiments]
    est_flags = [e["size_estimated"] for e in experiments]
    colors = [get_color(e["display_name"]) for e in experiments]

    for metric_key, metric_label, ax in metrics:
        values = [e[metric_key] for e in experiments]

        ax.scatter(sizes, values, c=colors, s=120, zorder=5, edgecolors="white", linewidth=1.5)
        annotate_points(ax, sizes, values, names, est_flags)

        style_axis(ax, metric_label, "Model Size (B parameters)", "Score")
        ax.set_ylim(0, 1.05)
        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x)}B"))

    fig.suptitle(
        "Model Size vs. WattBot Component Scores\n"
        "(*) = estimated size (proprietary model)",
        fontsize=15, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    plt.savefig(output_dir / "size_vs_scores.png", dpi=300, bbox_inches="tight")
    print(f"Saved {output_dir / 'size_vs_scores.png'}")


def plot_size_vs_latency(experiments: list[dict], output_dir: Path):
    """Plot 2: Model Size vs. Latency."""
    # Exclude experiments with suspect latencies (retry storms make the data useless)
    clean = [e for e in experiments if not e.get("latency_suspect")]
    suspect = [e for e in experiments if e.get("latency_suspect")]

    if suspect:
        print(f"  Latency plot: excluding {len(suspect)} experiments with suspect latency: "
              f"{[e['display_name'] for e in suspect]}")

    if not clean:
        print("  No clean latency data to plot")
        return

    fig, ax = plt.subplots(figsize=(11, 7))

    sizes = [e["size_b"] for e in clean]
    latencies = [e["avg_latency"] for e in clean]
    names = [e["display_name"] for e in clean]
    est_flags = [e["size_estimated"] for e in clean]
    colors = [get_color(e["display_name"]) for e in clean]

    ax.scatter(sizes, latencies, c=colors, s=150, zorder=5, edgecolors="white", linewidth=1.5)
    annotate_points(ax, sizes, latencies, names, est_flags)

    style_axis(ax, "Model Size vs. Average Latency per Question", "Model Size (B parameters)", "Latency (seconds)")
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x)}B"))

    # Add a note about what latency captures
    excluded_note = ""
    if suspect:
        excluded_note = f"\nExcluded {len(suspect)} runs with >60s avg latency (retry/throttle issues)"
    ax.text(
        0.02, 0.98,
        f"Latency = end-to-end per question (retrieval + LLM inference)\n"
        f"(*) = estimated model size{excluded_note}",
        transform=ax.transAxes, fontsize=8, va="top", style="italic", color="gray",
    )

    plt.tight_layout()
    plt.savefig(output_dir / "size_vs_latency.png", dpi=300, bbox_inches="tight")
    print(f"Saved {output_dir / 'size_vs_latency.png'}")


def plot_size_vs_cost(experiments: list[dict], output_dir: Path):
    """Plot 3: Model Size vs. Total Experiment Cost."""
    # Filter to experiments with cost data
    with_cost = [e for e in experiments if e["total_cost"] > 0]

    if len(with_cost) < 2:
        print("Not enough experiments with cost data for size_vs_cost plot")
        return

    fig, ax = plt.subplots(figsize=(11, 7))

    sizes = [e["size_b"] for e in with_cost]
    costs = [e["total_cost"] for e in with_cost]
    names = [e["display_name"] for e in with_cost]
    est_flags = [e["size_estimated"] for e in with_cost]
    colors = [get_color(e["display_name"]) for e in with_cost]

    ax.scatter(sizes, costs, c=colors, s=150, zorder=5, edgecolors="white", linewidth=1.5)
    annotate_points(ax, sizes, costs, names, est_flags)

    style_axis(ax, "Model Size vs. Total Experiment Cost", "Model Size (B parameters)", "Cost (USD)")
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x)}B"))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"${x:.2f}"))

    ax.text(
        0.02, 0.98,
        "Cost = estimated from token counts + Bedrock pricing\n(*) = estimated model size",
        transform=ax.transAxes, fontsize=8, va="top", style="italic", color="gray",
    )

    plt.tight_layout()
    plt.savefig(output_dir / "size_vs_cost.png", dpi=300, bbox_inches="tight")
    print(f"Saved {output_dir / 'size_vs_cost.png'}")


def plot_bubble_chart(experiments: list[dict], output_dir: Path):
    """Plot 4: Bubble chart -- Size (x) vs Score (y), bubble size = cost, color = family."""
    with_cost = [e for e in experiments if e["total_cost"] > 0]
    if len(with_cost) < 2:
        print("Not enough cost data for bubble chart")
        return

    fig, ax = plt.subplots(figsize=(12, 8))

    sizes = [e["size_b"] for e in with_cost]
    scores = [e["overall_score"] for e in with_cost]
    costs = [e["total_cost"] for e in with_cost]
    names = [e["display_name"] for e in with_cost]
    est_flags = [e["size_estimated"] for e in with_cost]
    colors = [get_color(e["display_name"]) for e in with_cost]

    # Scale bubble sizes (min 80, max 600)
    max_cost = max(costs)
    bubble_sizes = [max(80, (c / max_cost) * 600) for c in costs]

    ax.scatter(sizes, scores, s=bubble_sizes, c=colors, alpha=0.7, edgecolors="white", linewidth=2, zorder=5)
    annotate_points(ax, sizes, scores, names, est_flags)

    style_axis(
        ax,
        "Model Size vs. Score vs. Cost\n(bubble size = experiment cost)",
        "Model Size (B parameters)",
        "Overall WattBot Score",
    )
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x)}B"))
    ax.set_ylim(0, 1.0)

    # Legend for families
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=color, markersize=10, label=family)
        for family, color in FAMILY_COLORS.items()
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=9, title="Model Family")

    plt.tight_layout()
    plt.savefig(output_dir / "size_score_cost_bubble.png", dpi=300, bbox_inches="tight")
    print(f"Saved {output_dir / 'size_score_cost_bubble.png'}")


def print_summary_table(experiments: list[dict]):
    """Print a formatted summary table to console."""
    print(f"\n{'='*100}")
    print(f"{'Model':<25s} {'Size':>6s} {'Score':>7s} {'Value':>7s} {'Ref':>7s} {'NA':>7s} {'Latency':>9s} {'Cost':>8s} {'Est?':>5s}")
    print(f"{'-'*100}")
    for e in sorted(experiments, key=lambda x: x["overall_score"], reverse=True):
        est = "*" if e["size_estimated"] else ""
        lat_flag = " !!" if e.get("latency_suspect") else ""
        print(
            f"{e['display_name']:<25s} "
            f"{e['size_b']:>5d}B "
            f"{e['overall_score']:>6.3f} "
            f"{e['value_accuracy']:>6.3f} "
            f"{e['ref_overlap']:>6.3f} "
            f"{e['na_accuracy']:>6.3f} "
            f"{e['avg_latency']:>8.2f}s{lat_flag:3s} "
            f"${e['total_cost']:>6.4f} "
            f"{est:>5s}"
        )
    print(f"{'='*100}")
    print("(*) = estimated model size (proprietary)\n")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Plot Model Size vs. Performance metrics")
    parser.add_argument(
        "--experiments", "-e",
        default="artifacts/experiments",
        help="Path to experiments directory",
    )
    parser.add_argument(
        "--output", "-o",
        default="artifacts/plots",
        help="Output directory for plots",
    )
    args = parser.parse_args()

    experiments_dir = Path(args.experiments)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not experiments_dir.exists():
        print(f"Error: experiments directory not found: {experiments_dir}")
        sys.exit(1)

    print("Loading experiment data...")
    experiments = load_experiments(experiments_dir)

    if not experiments:
        print("No valid experiments found!")
        sys.exit(1)

    print(f"Loaded {len(experiments)} experiments")
    print_summary_table(experiments)

    print("\nGenerating plots...")
    plot_size_vs_scores(experiments, output_dir)
    plot_size_vs_latency(experiments, output_dir)
    plot_size_vs_cost(experiments, output_dir)
    plot_bubble_chart(experiments, output_dir)

    print(f"\nAll plots saved to {output_dir}/")


if __name__ == "__main__":
    main()
