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
    "gpt-oss-20b": ("GPT-OSS 20B", 20, False, "OpenAI open-weight, confirmed 20B"),
    "gpt-oss-120b": ("GPT-OSS 120B", 120, False, "OpenAI open-weight, confirmed 120B"),
}


def match_model_size(model_id: str) -> tuple[str, int, bool] | None:
    """Match a Bedrock model ID to our size registry. Returns (name, size_B, estimated)."""
    model_lower = model_id.lower()
    for key, (name, size, estimated, _notes) in MODEL_SIZES.items():
        if key in model_lower:
            return name, size, estimated
    return None


def load_experiments(experiments_dir: Path, name_filter: str | None = None) -> list[dict]:
    """Load experiment summaries, skipping ensembles and duplicates.

    Args:
        experiments_dir: Path to experiments directory.
        name_filter: If set, only load experiments whose directory name contains this string.
    """
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

        # Apply name filter
        if name_filter and name_filter not in name:
            continue

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
            "num_questions": data.get("num_questions", 0),
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
    "GPT": "#22c55e",         # Green
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


def plot_overall_ranking(experiments: list[dict], output_dir: Path):
    """Plot 5: Overall performance ranking (horizontal bar chart)."""
    fig, ax = plt.subplots(figsize=(12, 7))

    sorted_exp = sorted(experiments, key=lambda x: x["overall_score"])
    names = [e["display_name"] for e in sorted_exp]
    scores = [e["overall_score"] for e in sorted_exp]
    colors = [get_color(n) for n in names]

    bars = ax.barh(range(len(names)), scores, color=colors, edgecolor="white", linewidth=1.2, height=0.65)

    # Add score labels
    for i, (bar, score) in enumerate(zip(bars, scores)):
        ax.text(bar.get_width() + 0.008, bar.get_y() + bar.get_height() / 2,
                f"{score:.3f}", va="center", fontsize=10, fontweight="bold")

    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=11)
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("WattBot Score (0.75*Val + 0.15*Ref + 0.10*NA)", fontsize=11)
    n_questions = experiments[0].get("num_questions", len(experiments))
    # Try to get actual count from first experiment's summary
    for e in experiments:
        if e.get("num_questions"):
            n_questions = e["num_questions"]
            break
    ax.set_title(f"Model Performance Ranking (n={n_questions} questions)", fontsize=14, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", linestyle="--", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "overall_ranking.png", dpi=300, bbox_inches="tight")
    print(f"Saved {output_dir / 'overall_ranking.png'}")


def plot_cost_vs_performance(experiments: list[dict], output_dir: Path):
    """Plot 6: Cost vs. Performance scatter (the key trade-off chart)."""
    with_cost = [e for e in experiments if e["total_cost"] > 0]
    if len(with_cost) < 2:
        print("Not enough cost data for cost_vs_performance plot")
        return

    fig, ax = plt.subplots(figsize=(11, 7))

    costs = [e["total_cost"] for e in with_cost]
    scores = [e["overall_score"] for e in with_cost]
    names = [e["display_name"] for e in with_cost]
    colors = [get_color(n) for n in names]

    ax.scatter(costs, scores, c=colors, s=150, zorder=5, edgecolors="white", linewidth=1.5)

    for x, y, name in zip(costs, scores, names):
        ax.annotate(
            name, (x, y),
            xytext=(8, 6), textcoords="offset points", fontsize=9, ha="left", va="bottom",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7, edgecolor="none"),
        )

    style_axis(ax, "Cost vs. Performance Trade-off", "Total Cost (USD)", "Overall Score")
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"${x:.2f}"))
    ax.set_ylim(0.3, 0.85)

    # Highlight the Pareto-optimal region
    ax.text(
        0.02, 0.02,
        "Lower-left = cheaper but worse | Upper-left = best value",
        transform=ax.transAxes, fontsize=8, va="bottom", style="italic", color="gray",
    )

    plt.tight_layout()
    plt.savefig(output_dir / "cost_vs_performance.png", dpi=300, bbox_inches="tight")
    print(f"Saved {output_dir / 'cost_vs_performance.png'}")


def plot_score_breakdown(experiments: list[dict], output_dir: Path):
    """Plot 7: Stacked/grouped bar chart showing component score breakdown."""
    fig, ax = plt.subplots(figsize=(14, 7))

    sorted_exp = sorted(experiments, key=lambda x: x["overall_score"], reverse=True)
    names = [e["display_name"] for e in sorted_exp]
    x = np.arange(len(names))
    width = 0.22

    val_scores = [e["value_accuracy"] for e in sorted_exp]
    ref_scores = [e["ref_overlap"] for e in sorted_exp]
    na_scores = [e["na_accuracy"] for e in sorted_exp]

    bars1 = ax.bar(x - width, val_scores, width, label="Value Accuracy (75%)", color="#6366f1", alpha=0.85)
    bars2 = ax.bar(x, ref_scores, width, label="Reference Overlap (15%)", color="#f59e0b", alpha=0.85)
    bars3 = ax.bar(x + width, na_scores, width, label="NA Accuracy (10%)", color="#10b981", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=35, ha="right", fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title("Score Component Breakdown by Model", fontsize=14, fontweight="bold")
    ax.legend(loc="upper right", fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "score_breakdown.png", dpi=300, bbox_inches="tight")
    print(f"Saved {output_dir / 'score_breakdown.png'}")


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
    parser.add_argument(
        "--filter", "-f",
        default=None,
        help="Only load experiments whose name contains this string (e.g. 'bench')",
    )
    args = parser.parse_args()

    experiments_dir = Path(args.experiments)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not experiments_dir.exists():
        print(f"Error: experiments directory not found: {experiments_dir}")
        sys.exit(1)

    print("Loading experiment data...")
    if args.filter:
        print(f"  Filtering to experiments containing: '{args.filter}'")
    experiments = load_experiments(experiments_dir, name_filter=args.filter)

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
    plot_overall_ranking(experiments, output_dir)
    plot_cost_vs_performance(experiments, output_dir)
    plot_score_breakdown(experiments, output_dir)

    print(f"\nAll 7 plots saved to {output_dir}/")


if __name__ == "__main__":
    main()
