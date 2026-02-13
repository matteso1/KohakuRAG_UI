#!/usr/bin/env python3
"""
Model Size vs. Performance/Latency/Cost Plots (Provider-Agnostic)

Generates publication-quality plots:
  1. Model Size vs. WattBot Component Scores
  2. Model Size vs. Latency
  3. Model Size vs. Total Cost
  4. Model Size vs. Overall Score (bubble chart with cost as size)
  5. Overall Ranking
  6. Cost vs. Performance
  7. Score Component Breakdown
  8. Total GPU Energy per Experiment

Supports both API models (Bedrock, OpenRouter) and local HF models.

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
# For MoE models, we report TOTAL parameters (matches HF model cards).
# Active-parameter counts are noted for reference.

MODEL_SIZES = {
    # --- Bedrock / API models ---
    "claude-3-haiku": ("Claude 3 Haiku", 8, True, "Anthropic undisclosed; community est. ~8B"),
    "claude-3-5-haiku": ("Claude 3.5 Haiku", 8, True, "Anthropic undisclosed; community est. ~8B"),
    "claude-3-5-sonnet": ("Claude 3.5 Sonnet", 70, True, "Anthropic undisclosed; community est. ~70B"),
    "claude-3-7-sonnet": ("Claude 3.7 Sonnet", 70, True, "Anthropic undisclosed; community est. ~70B"),
    "nova-pro": ("Nova Pro", 40, True, "Amazon undisclosed; rough est. ~40B"),
    "llama3-3-70b": ("Llama 3.3 70B", 70, False, "Open-source, confirmed"),
    "llama3-1-70b": ("Llama 3.1 70B", 70, False, "Open-source, confirmed"),
    "llama4-scout-17b": ("Llama 4 Scout", 109, False, "109B total MoE (17B active)"),
    "llama4-maverick-17b": ("Llama 4 Maverick", 400, False, "400B total MoE (17B active)"),
    "mistral-small": ("Mistral Small", 24, False, "Open-source, confirmed 24B"),
    "deepseek": ("DeepSeek R1 (distill)", 70, False, "Llama 70B distill variant"),

    # --- Local HuggingFace models ---
    "qwen2.5-7b": ("Qwen 2.5 7B", 7, False, "Open-source, confirmed 7.6B"),
    "qwen2.5-1.5b": ("Qwen 2.5 1.5B", 1.5, False, "Open-source, confirmed 1.5B"),
    "qwen2.5-3b": ("Qwen 2.5 3B", 3, False, "Open-source, confirmed 3B"),
    "qwen2.5-14b": ("Qwen 2.5 14B", 14, False, "Open-source, confirmed 14B"),
    "qwen2.5-32b": ("Qwen 2.5 32B", 32, False, "Open-source, confirmed 32B"),
    "qwen2.5-72b": ("Qwen 2.5 72B", 72, False, "Open-source, confirmed 72B"),
    "llama-3.1-8b": ("Llama 3.1 8B", 8, False, "Open-source, confirmed 8B"),
    "llama-3.2-3b": ("Llama 3.2 3B", 3, False, "Open-source, confirmed 3B"),
    "mistral-7b": ("Mistral 7B", 7, False, "Open-source, confirmed 7B"),
    "phi-3-mini": ("Phi-3 Mini", 3.8, False, "Open-source, confirmed 3.8B"),
    "phi-3.5-mini": ("Phi-3.5 Mini", 3.8, False, "Open-source, confirmed 3.8B"),
    "gemma-2-9b": ("Gemma 2 9B", 9, False, "Open-source, confirmed 9B"),
    "gemma-2-2b": ("Gemma 2 2B", 2, False, "Open-source, confirmed 2B"),
    "mixtral-8x7b": ("Mixtral 8x7B", 46.7, False, "46.7B total MoE (12.9B active)"),
    "mixtral-8x22b": ("Mixtral 8x22B", 141, False, "141B total MoE (39B active)"),
    "qwen3-30b-a3b": ("Qwen3 30B-A3B", 30.5, False, "30.5B total MoE (3.3B active)"),
    "olmoe-1b-7b": ("OLMoE 1B-7B", 7, False, "7B total MoE (1B active)"),
}


def wilson_ci_half(p: float, n: int, confidence: float = 0.95) -> float:
    """Wilson CI half-width for a proportion p with n trials.

    Returns the half-width of the 95% CI (i.e., score ± this value).
    """
    if n == 0 or p < 0 or p > 1:
        return 0.0
    z = 1.96 if confidence == 0.95 else 1.645
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    spread = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return spread


def match_model_size(model_id: str) -> tuple[str, float, bool] | None:
    """Match a model ID to our size registry. Returns (name, size_B, estimated)."""
    model_lower = model_id.lower()
    for key, (name, size, estimated, _notes) in MODEL_SIZES.items():
        if key in model_lower:
            return name, size, estimated
    return None


def load_experiments(experiments_dir: Path, name_filter: str | None = None,
                     datafile: str | None = None,
                     system: str | None = None) -> list[dict]:
    """Load experiment summaries, skipping ensembles and duplicates.

    When *datafile* is given (e.g. ``"train_QA"``), only experiments
    whose path contains that subfolder are included.

    When *system* is given (e.g. ``"PowerEdge"``, ``"GB10"``), only
    experiments under that system directory are included.
    """
    experiments = []
    seen_models = {}  # model_id -> best experiment

    for summary_path in sorted(experiments_dir.glob("**/summary.json")):
        if datafile is not None and datafile not in summary_path.parts:
            continue
        if system is not None and system not in summary_path.parts:
            continue
        try:
            with open(summary_path) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            print(f"  Skipping {summary_path.parent.name}: {e}")
            continue

        name = data.get("name", summary_path.parent.name)
        model_id = data.get("model_id", "")

        if name_filter and name_filter not in name:
            continue

        if not model_id or "ensemble" in name.lower():
            continue

        if data.get("overall_score", 0) < 0.15:
            print(f"  Skipping {name}: score too low ({data.get('overall_score', 0):.3f})")
            continue

        n_questions = data.get("num_questions", 0)
        error_count = data.get("error_count", 0)
        if n_questions > 0 and error_count / n_questions > 0.1:
            print(f"  Skipping {name}: high error rate ({error_count}/{n_questions})")
            continue

        avg_latency = data.get("avg_latency_seconds", 0)
        latency_suspect = avg_latency > 120  # Higher threshold for local models

        size_info = match_model_size(model_id)
        if size_info is None:
            print(f"  Skipping {name}: no size info for {model_id}")
            continue

        display_name, size_b, estimated = size_info

        if latency_suspect:
            print(f"  Warning: {name} has high avg latency ({avg_latency:.0f}s)")

        # Compute Wilson CI for score components
        nq = n_questions if n_questions > 0 else 1
        val_ci = wilson_ci_half(data.get("value_accuracy", 0), nq)
        ref_ci = wilson_ci_half(data.get("ref_overlap", 0), nq)
        na_ci = wilson_ci_half(data.get("na_accuracy", 0), nq)
        # Overall CI via error propagation of weighted components
        overall_ci = np.sqrt((0.75 * val_ci)**2 + (0.15 * ref_ci)**2 + (0.10 * na_ci)**2)

        entry = {
            "experiment": name,
            "model_id": model_id,
            "display_name": display_name,
            "llm_provider": data.get("llm_provider", "unknown"),
            "size_b": size_b,
            "size_estimated": estimated,
            "overall_score": data.get("overall_score", 0),
            "value_accuracy": data.get("value_accuracy", 0),
            "ref_overlap": data.get("ref_overlap", 0),
            "na_accuracy": data.get("na_accuracy", 0),
            "num_questions": n_questions,
            "overall_ci": overall_ci,
            "val_ci": val_ci,
            "ref_ci": ref_ci,
            "na_ci": na_ci,
            "avg_latency": avg_latency,
            "latency_suspect": latency_suspect,
            "total_cost": data.get("estimated_cost_usd", 0),
            "input_tokens": data.get("input_tokens", 0),
            "output_tokens": data.get("output_tokens", 0),
            "gpu_energy_wh": data.get("hardware", {}).get("gpu_energy_wh", 0),
        }

        if model_id in seen_models:
            existing = seen_models[model_id]
            if existing["latency_suspect"] and not entry["latency_suspect"]:
                seen_models[model_id] = entry
            elif entry["latency_suspect"] and not existing["latency_suspect"]:
                pass
            elif entry["overall_score"] > existing["overall_score"]:
                seen_models[model_id] = entry
        else:
            seen_models[model_id] = entry

    experiments = list(seen_models.values())
    experiments.sort(key=lambda x: x["size_b"])
    return experiments


def load_ensemble_experiments(experiments_dir: Path, datafile: str | None = None,
                              system: str | None = None) -> list[dict]:
    """Load ensemble experiment summaries for the ranking plot.

    Ensembles are identified by ``"type": "ensemble"`` in their summary or by
    having ``"ensemble"`` in the directory name.
    """
    ensembles = []

    for summary_path in sorted(experiments_dir.glob("**/summary.json")):
        if datafile is not None and datafile not in summary_path.parts:
            continue
        if system is not None and system not in summary_path.parts:
            continue
        try:
            with open(summary_path) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue

        name = data.get("name", summary_path.parent.name)
        is_ensemble = data.get("type") == "ensemble" or "ensemble" in name.lower()
        if not is_ensemble:
            continue

        score = data.get("overall_score", 0)
        if score < 0.15:
            continue

        n_questions = data.get("num_questions", 0)
        nq = n_questions if n_questions > 0 else 1
        val_ci = wilson_ci_half(data.get("value_accuracy", 0), nq)
        ref_ci = wilson_ci_half(data.get("ref_overlap", 0), nq)
        na_ci = wilson_ci_half(data.get("na_accuracy", 0), nq)
        overall_ci = np.sqrt((0.75 * val_ci)**2 + (0.15 * ref_ci)**2 + (0.10 * na_ci)**2)

        # Build a readable display name from the experiment name
        display = name.replace("-", " ").replace("_", " ").title()

        ensembles.append({
            "experiment": name,
            "model_id": "",
            "display_name": display,
            "llm_provider": "ensemble",
            "size_b": 0,
            "size_estimated": False,
            "overall_score": score,
            "value_accuracy": data.get("value_accuracy", 0),
            "ref_overlap": data.get("ref_overlap", 0),
            "na_accuracy": data.get("na_accuracy", 0),
            "num_questions": n_questions,
            "overall_ci": overall_ci,
            "val_ci": val_ci,
            "ref_ci": ref_ci,
            "na_ci": na_ci,
            "avg_latency": data.get("avg_latency_seconds", 0),
            "latency_suspect": False,
            "total_cost": data.get("estimated_cost_usd", 0),
            "input_tokens": data.get("input_tokens", 0),
            "output_tokens": data.get("output_tokens", 0),
            "gpu_energy_wh": data.get("hardware", {}).get("gpu_energy_wh", 0),
            "is_ensemble": True,
            "num_models": data.get("num_models", 0),
            "strategy": data.get("strategy", ""),
        })

    return ensembles


# ============================================================================
# Plot Helpers
# ============================================================================

FAMILY_COLORS = {
    "Claude": "#6366f1",
    "Llama": "#f59e0b",
    "Mistral": "#10b981",
    "DeepSeek": "#ef4444",
    "Nova": "#ec4899",
    "Qwen": "#3b82f6",
    "Phi": "#8b5cf6",
    "Gemma": "#14b8a6",
}


def get_color(name: str) -> str:
    for family, color in FAMILY_COLORS.items():
        if family.lower() in name.lower():
            return color
    return "#6b7280"


def get_marker(provider: str) -> str:
    """Different markers for local vs API models."""
    if provider == "hf_local":
        return "s"  # square for local
    return "o"  # circle for API


def annotate_points(ax, xs, ys, labels, estimated_flags):
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
    """Plot 1: Model Size vs. WattBot Component Scores."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Map metric key -> CI key
    ci_keys = {
        "overall_score": "overall_ci",
        "value_accuracy": "val_ci",
        "ref_overlap": "ref_ci",
        "na_accuracy": "na_ci",
    }

    metrics = [
        ("overall_score", "Overall WattBot Score", axes[0, 0]),
        ("value_accuracy", "Value Accuracy (75% weight)", axes[0, 1]),
        ("ref_overlap", "Reference Overlap (15% weight)", axes[1, 0]),
        ("na_accuracy", "NA Recall (10% weight)", axes[1, 1]),
    ]

    sizes = [e["size_b"] for e in experiments]
    names = [e["display_name"] for e in experiments]
    est_flags = [e["size_estimated"] for e in experiments]
    colors = [get_color(e["display_name"]) for e in experiments]
    markers = [get_marker(e.get("llm_provider", "")) for e in experiments]

    for metric_key, metric_label, ax in metrics:
        values = [e[metric_key] for e in experiments]
        ci_half = [e.get(ci_keys[metric_key], 0) for e in experiments]

        for x, y, ci, c, m in zip(sizes, values, ci_half, colors, markers):
            ax.errorbar(x, y, yerr=ci, fmt='none', ecolor=c, elinewidth=1.2, capsize=3, alpha=0.6, zorder=4)
            ax.scatter(x, y, c=c, s=120, zorder=5, edgecolors="white", linewidth=1.5, marker=m)
        annotate_points(ax, sizes, values, names, est_flags)

        style_axis(ax, metric_label, "Model Size (B parameters)", "Score")
        ax.set_ylim(0, 1.05)
        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x)}B" if x >= 1 else f"{x:.1f}B"))

    fig.suptitle(
        "Model Size vs. WattBot Component Scores\n"
        "(*) = estimated size | square = local HF | circle = API",
        fontsize=15, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    plt.savefig(output_dir / "size_vs_scores.png", dpi=300, bbox_inches="tight")
    print(f"Saved {output_dir / 'size_vs_scores.png'}")


def plot_size_vs_latency(experiments: list[dict], output_dir: Path):
    """Plot 2: Model Size vs. Latency."""
    clean = [e for e in experiments if not e.get("latency_suspect")]
    if not clean:
        print("  No clean latency data to plot")
        return

    fig, ax = plt.subplots(figsize=(11, 7))

    sizes = [e["size_b"] for e in clean]
    latencies = [e["avg_latency"] for e in clean]
    names = [e["display_name"] for e in clean]
    est_flags = [e["size_estimated"] for e in clean]
    colors = [get_color(e["display_name"]) for e in clean]
    markers = [get_marker(e.get("llm_provider", "")) for e in clean]

    for x, y, c, m in zip(sizes, latencies, colors, markers):
        ax.scatter(x, y, c=c, s=150, zorder=5, edgecolors="white", linewidth=1.5, marker=m)
    annotate_points(ax, sizes, latencies, names, est_flags)

    style_axis(ax, "Model Size vs. Average Latency per Question", "Model Size (B parameters)", "Latency (seconds)")
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x)}B" if x >= 1 else f"{x:.1f}B"))

    ax.text(
        0.02, 0.98,
        "square = local HF (GPU inference) | circle = API\n(*) = estimated model size",
        transform=ax.transAxes, fontsize=8, va="top", style="italic", color="gray",
    )

    plt.tight_layout()
    plt.savefig(output_dir / "size_vs_latency.png", dpi=300, bbox_inches="tight")
    print(f"Saved {output_dir / 'size_vs_latency.png'}")


def plot_size_vs_cost(experiments: list[dict], output_dir: Path):
    """Plot 3: Model Size vs. Total Experiment Cost."""
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
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x)}B" if x >= 1 else f"{x:.1f}B"))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"${x:.2f}"))

    ax.text(
        0.02, 0.98,
        "Cost = API cost only (local HF models have $0 API cost)\n(*) = estimated model size",
        transform=ax.transAxes, fontsize=8, va="top", style="italic", color="gray",
    )

    plt.tight_layout()
    plt.savefig(output_dir / "size_vs_cost.png", dpi=300, bbox_inches="tight")
    print(f"Saved {output_dir / 'size_vs_cost.png'}")


def plot_bubble_chart(experiments: list[dict], output_dir: Path):
    """Plot 4: Bubble chart -- Size (x) vs Score (y), bubble size = cost."""
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
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x)}B" if x >= 1 else f"{x:.1f}B"))
    ax.set_ylim(0, 1.0)

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=color, markersize=10, label=family)
        for family, color in FAMILY_COLORS.items()
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=9, title="Model Family")

    plt.tight_layout()
    plt.savefig(output_dir / "size_score_cost_bubble.png", dpi=300, bbox_inches="tight")
    print(f"Saved {output_dir / 'size_score_cost_bubble.png'}")


ENSEMBLE_COLOR = "#d97706"  # amber – visually distinct from model family colors


def plot_overall_ranking(experiments: list[dict], output_dir: Path,
                         ensembles: list[dict] | None = None):
    """Plot 5: Overall performance ranking (horizontal bar chart).

    When *ensembles* is provided, ensemble entries are merged in and drawn
    with hatching + amber colour so they stand out from individual models.
    """
    all_exp = list(experiments)
    if ensembles:
        all_exp.extend(ensembles)

    sorted_exp = sorted(all_exp, key=lambda x: x["overall_score"])
    n_bars = len(sorted_exp)
    fig, ax = plt.subplots(figsize=(12, max(7, n_bars * 0.6)))

    scores = [e["overall_score"] for e in sorted_exp]
    ci_widths = [e.get("overall_ci", 0) for e in sorted_exp]

    # Per-bar colour: ensemble bars get amber, models get family colour
    colors = []
    labels = []
    hatch_indices = []
    for i, e in enumerate(sorted_exp):
        if e.get("is_ensemble"):
            colors.append(ENSEMBLE_COLOR)
            labels.append(f"{e['display_name']}  [ensemble]")
            hatch_indices.append(i)
        else:
            colors.append(get_color(e["display_name"]))
            suffix = " [local]" if e.get("llm_provider") == "hf_local" else ""
            labels.append(f"{e['display_name']}{suffix}")

    bars = ax.barh(range(n_bars), scores, color=colors, edgecolor="white",
                   linewidth=1.2, height=0.65,
                   xerr=ci_widths, capsize=3,
                   error_kw={'linewidth': 1.2, 'color': '#333'})

    # Apply hatching to ensemble bars
    for idx in hatch_indices:
        bars[idx].set_hatch("//")
        bars[idx].set_edgecolor("#92400e")

    for bar, score, ci in zip(bars, scores, ci_widths):
        ax.text(bar.get_width() + ci + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{score:.3f}", va="center", fontsize=10, fontweight="bold")

    ax.set_yticks(range(n_bars))
    ax.set_yticklabels(labels, fontsize=11)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("WattBot Score (0.75*Val + 0.15*Ref + 0.10*NA)", fontsize=11)
    n_q = sorted_exp[0].get("num_questions", "?") if sorted_exp else "?"
    ax.set_title(f"Model Performance Ranking (n={n_q}, 95% CI)", fontsize=14, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", linestyle="--", alpha=0.3)

    # Legend entry for ensemble hatching
    if hatch_indices:
        from matplotlib.patches import Patch
        ax.legend(
            handles=[Patch(facecolor=ENSEMBLE_COLOR, edgecolor="#92400e",
                           hatch="//", label="Ensemble")],
            loc="lower right", fontsize=10,
        )

    plt.tight_layout()
    plt.savefig(output_dir / "overall_ranking.png", dpi=300, bbox_inches="tight")
    print(f"Saved {output_dir / 'overall_ranking.png'}")


def plot_cost_vs_performance(experiments: list[dict], output_dir: Path):
    """Plot 6: Cost vs. Performance scatter."""
    with_cost = [e for e in experiments if e["total_cost"] > 0]
    # Also include local models at $0 cost
    local_models = [e for e in experiments if e.get("llm_provider") == "hf_local"]

    plot_data = with_cost + [e for e in local_models if e not in with_cost]
    if len(plot_data) < 2:
        print("Not enough models for cost_vs_performance plot")
        return

    fig, ax = plt.subplots(figsize=(11, 7))

    costs = [e["total_cost"] for e in plot_data]
    scores = [e["overall_score"] for e in plot_data]
    names = [e["display_name"] for e in plot_data]
    colors = [get_color(n) for n in names]
    markers = [get_marker(e.get("llm_provider", "")) for e in plot_data]

    for x, y, c, m, n in zip(costs, scores, colors, markers, names):
        ax.scatter(x, y, c=c, s=150, zorder=5, edgecolors="white", linewidth=1.5, marker=m)
        ax.annotate(
            n, (x, y),
            xytext=(8, 6), textcoords="offset points", fontsize=9, ha="left", va="bottom",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7, edgecolor="none"),
        )

    style_axis(ax, "Cost vs. Performance Trade-off", "Total API Cost (USD)", "Overall Score")
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"${x:.2f}"))

    ax.text(
        0.02, 0.02,
        "square = local HF ($0 API cost) | circle = API\nLower-left = cheaper | Upper-left = best value",
        transform=ax.transAxes, fontsize=8, va="bottom", style="italic", color="gray",
    )

    plt.tight_layout()
    plt.savefig(output_dir / "cost_vs_performance.png", dpi=300, bbox_inches="tight")
    print(f"Saved {output_dir / 'cost_vs_performance.png'}")


def plot_score_breakdown(experiments: list[dict], output_dir: Path):
    """Plot 7: Grouped bar chart showing component score breakdown."""
    fig, ax = plt.subplots(figsize=(14, 7))

    sorted_exp = sorted(experiments, key=lambda x: x["overall_score"], reverse=True)
    names = [e["display_name"] for e in sorted_exp]
    x = np.arange(len(names))
    width = 0.22

    val_scores = [e["value_accuracy"] for e in sorted_exp]
    ref_scores = [e["ref_overlap"] for e in sorted_exp]
    na_scores = [e["na_accuracy"] for e in sorted_exp]
    val_ci = [e.get("val_ci", 0) for e in sorted_exp]
    ref_ci = [e.get("ref_ci", 0) for e in sorted_exp]
    na_ci = [e.get("na_ci", 0) for e in sorted_exp]

    err_kw = {'linewidth': 1, 'color': '#333'}
    ax.bar(x - width, val_scores, width, label="Value Accuracy (75%)", color="#6366f1", alpha=0.85,
           yerr=val_ci, capsize=2, error_kw=err_kw)
    ax.bar(x, ref_scores, width, label="Reference Overlap (15%)", color="#f59e0b", alpha=0.85,
           yerr=ref_ci, capsize=2, error_kw=err_kw)
    ax.bar(x + width, na_scores, width, label="NA Recall (10%)", color="#10b981", alpha=0.85,
           yerr=na_ci, capsize=2, error_kw=err_kw)

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=35, ha="right", fontsize=10)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title("Score Component Breakdown by Model (95% Wilson CI)", fontsize=14, fontweight="bold")
    ax.legend(loc="upper right", fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "score_breakdown.png", dpi=300, bbox_inches="tight")
    print(f"Saved {output_dir / 'score_breakdown.png'}")


def plot_energy(experiments: list[dict], output_dir: Path):
    """Plot 8: Total GPU energy consumed per experiment (horizontal bar chart)."""
    with_energy = [e for e in experiments if e.get("gpu_energy_wh", 0) > 0]
    if len(with_energy) < 2:
        print("Not enough experiments with energy data for energy plot")
        return

    fig, ax = plt.subplots(figsize=(12, max(7, len(with_energy) * 0.6)))

    sorted_exp = sorted(with_energy, key=lambda x: x["gpu_energy_wh"])
    labels = []
    for e in sorted_exp:
        suffix = " [local]" if e.get("llm_provider") == "hf_local" else ""
        labels.append(f"{e['display_name']}{suffix}")
    energies = [e["gpu_energy_wh"] for e in sorted_exp]
    colors = [get_color(e["display_name"]) for e in sorted_exp]

    bars = ax.barh(range(len(labels)), energies, color=colors, edgecolor="white",
                   linewidth=1.2, height=0.65)

    for bar, energy in zip(bars, energies):
        ax.text(bar.get_width() + max(energies) * 0.01, bar.get_y() + bar.get_height() / 2,
                f"{energy:.1f} Wh", va="center", fontsize=10, fontweight="bold")

    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=11)
    ax.set_xlabel("Total GPU Energy (Wh)", fontsize=11)
    n_q = sorted_exp[0].get("num_questions", "?") if sorted_exp else "?"
    ax.set_title(f"Total GPU Energy per Experiment (n={n_q} questions)", fontsize=14, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", linestyle="--", alpha=0.3)

    ax.text(
        0.98, 0.02,
        "Energy measured via NVML counter or nvidia-smi power sampling\n"
        "API models excluded (no local GPU usage)",
        transform=ax.transAxes, fontsize=8, ha="right", va="bottom",
        style="italic", color="gray",
    )

    plt.tight_layout()
    plt.savefig(output_dir / "energy_per_experiment.png", dpi=300, bbox_inches="tight")
    print(f"Saved {output_dir / 'energy_per_experiment.png'}")


def print_summary_table(experiments: list[dict]):
    """Print formatted summary table."""
    print(f"\n{'='*110}")
    print(f"{'Model':<25s} {'Provider':>10s} {'Size':>6s} {'Score':>7s} {'Value':>7s} {'Ref':>7s} {'NA':>7s} {'Latency':>9s} {'Cost':>8s} {'Est?':>5s}")
    print(f"{'-'*110}")
    for e in sorted(experiments, key=lambda x: x["overall_score"], reverse=True):
        est = "*" if e["size_estimated"] else ""
        lat_flag = " !!" if e.get("latency_suspect") else ""
        provider = e.get("llm_provider", "?")[:10]
        size_str = f"{e['size_b']}B" if e['size_b'] >= 1 else f"{e['size_b']:.1f}B"
        print(
            f"{e['display_name']:<25s} "
            f"{provider:>10s} "
            f"{size_str:>6s} "
            f"{e['overall_score']:>6.3f} "
            f"{e['value_accuracy']:>6.3f} "
            f"{e['ref_overlap']:>6.3f} "
            f"{e['na_accuracy']:>6.3f} "
            f"{e['avg_latency']:>8.2f}s{lat_flag:3s} "
            f"${e['total_cost']:>6.4f} "
            f"{est:>5s}"
        )
    print(f"{'='*110}")
    print("(*) = estimated model size (proprietary)")
    print("square markers = local HF, circle markers = API\n")


# ============================================================================
# Helpers
# ============================================================================

def discover_systems(experiments_dir: Path) -> list[str]:
    """Return sorted list of system directory names under experiments_dir."""
    return sorted(
        d.name for d in experiments_dir.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    )


def generate_plots(experiments_dir: Path, output_dir: Path,
                   name_filter: str | None, datafile: str | None,
                   system: str | None):
    """Load experiments for one system and generate all 8 plots."""
    label = f"[{system}] " if system else ""
    print(f"\n{'=' * 60}")
    print(f"{label}Loading experiment data...")
    if datafile:
        print(f"  Filtering to datafile: {datafile}")
    if system:
        print(f"  Filtering to system: {system}")
    if name_filter:
        print(f"  Filtering to experiments containing: '{name_filter}'")

    experiments = load_experiments(experiments_dir, name_filter=name_filter,
                                  datafile=datafile, system=system)
    if not experiments:
        print(f"{label}No valid experiments found — skipping")
        return

    print(f"{label}Loaded {len(experiments)} experiments")
    print_summary_table(experiments)

    ensembles = load_ensemble_experiments(experiments_dir, datafile=datafile,
                                          system=system)
    if ensembles:
        print(f"{label}Loaded {len(ensembles)} ensemble experiment(s) for ranking plot")

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"{label}Generating plots...")
    plot_size_vs_scores(experiments, output_dir)
    plot_size_vs_latency(experiments, output_dir)
    plot_size_vs_cost(experiments, output_dir)
    plot_bubble_chart(experiments, output_dir)
    plot_overall_ranking(experiments, output_dir, ensembles=ensembles)
    plot_cost_vs_performance(experiments, output_dir)
    plot_score_breakdown(experiments, output_dir)
    plot_energy(experiments, output_dir)

    print(f"{label}All 8 plots saved to {output_dir}/")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Plot Model Size vs. Performance metrics")
    parser.add_argument("--experiments", "-e", default="artifacts/experiments", help="Experiments directory")
    parser.add_argument("--output", "-o", default="artifacts/plots", help="Output directory for plots")
    parser.add_argument("--filter", "-f", default=None, help="Only load experiments containing this string")
    parser.add_argument(
        "--datafile", "-d", default=None,
        help="Filter to experiments from this datafile subfolder "
             "(e.g. 'train_QA', 'test_solutions'). Default: include all.",
    )
    parser.add_argument(
        "--system", "-S", default=None,
        help="Filter to a single system subfolder "
             "(e.g. 'PowerEdge', 'GB10', 'Bedrock'). "
             "Default: auto-discover all systems and generate plots for each.",
    )
    args = parser.parse_args()

    experiments_dir = Path(args.experiments)
    if not experiments_dir.exists():
        print(f"Error: experiments directory not found: {experiments_dir}")
        sys.exit(1)

    base_output = Path(args.output)

    if args.system:
        # Single system specified — generate one set of plots
        out = base_output / args.system
        if args.datafile:
            out = out / args.datafile
        generate_plots(experiments_dir, out, args.filter, args.datafile, args.system)
    else:
        # Auto-discover systems and generate per-system plots
        systems = discover_systems(experiments_dir)
        if not systems:
            print("No system directories found under experiments dir")
            sys.exit(1)
        print(f"Discovered {len(systems)} system(s): {systems}")
        for system in systems:
            out = base_output / system
            if args.datafile:
                out = out / args.datafile
            generate_plots(experiments_dir, out, args.filter, args.datafile, system)


if __name__ == "__main__":
    main()
