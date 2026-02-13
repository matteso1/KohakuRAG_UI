#!/usr/bin/env python3
"""
Generate Final Presentation Plots (Clean Slate)

Generates 4 specific, high-quality plots for the ML+X presentation:
1.  Model Ranking (Horizontal Bar Chart) - Best for readability
2.  Value vs. Cost Trade-off (Scatter)
3.  Model Size vs. Score (Scatter)
4.  Latency vs. Score (Scatter)

Usage:
    python scripts/generate_final_plots.py
"""

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import json
from pathlib import Path

# ============================================================================
# Configuration & Constants
# ============================================================================

PLOT_DIR = Path("artifacts/plots")
EXP_DIR = Path("artifacts/experiments")

# Target models to include (exact match or substring)
TARGET_MODELS = [
    "claude-3-haiku", "claude-3-5-haiku", 
    "claude-3-5-sonnet", "claude-3-7-sonnet",
    "llama3-1-70b", "llama3-3-70b", 
    "llama4-scout", "llama4-maverick",
    "deepseek", 
    "gpt-oss-20b", "gpt-oss-120b"
]

# Formatting constants
FONT_TITLE = 20
FONT_LABEL = 14
FONT_TICK = 12
FIG_SIZE = (12, 8)

FAMILY_COLORS = {
    "Claude": "#6366f1",      # Indigo
    "Llama": "#f59e0b",       # Amber
    "GPT": "#22c55e",         # Green
    "Mistral": "#10b981",     # Emerald
    "DeepSeek": "#ef4444",    # Red
    "Nova": "#ec4899",        # Pink
}

MODEL_DISPLAY_NAMES = {
    "claude-3-haiku": "Claude 3 Haiku",
    "claude-3-5-haiku": "Claude 3.5 Haiku",
    "claude-3-5-sonnet": "Claude 3.5 Sonnet",
    "claude-3-7-sonnet": "Claude 3.7 Sonnet",
    "llama3-1-70b": "Llama 3.1 70B",
    "llama3-3-70b": "Llama 3.3 70B",
    "llama4-scout": "Llama 4 Scout",
    "llama4-maverick": "Llama 4 Maverick",
    "deepseek": "DeepSeek R1",
    "gpt-oss-20b": "GPT-OSS 20B",
    "gpt-oss-120b": "GPT-OSS 120B",
}

# Manual offsets for Scatter Plots (dx, dy) to fix overlaps
# Adjusted based on previous failed attempts
SCATTER_OFFSETS = {
    "Claude 3 Haiku": (0, -0.015),
    "Claude 3.5 Haiku": (0, 0.015),
    "Llama 4 Scout": (-0.1, -0.01),
    "GPT-OSS 20B": (0.1, -0.01),
    "Llama 3.3 70B": (0, -0.02),
    "DeepSeek R1": (0, 0.02),
    "Claude 3.5 Sonnet": (0, -0.015),
    "Claude 3.7 Sonnet": (0, 0.015),
}

# ============================================================================
# Helpers
# ============================================================================

def get_color(name):
    for family, color in FAMILY_COLORS.items():
        if family.lower() in name.lower():
            return color
    return "#888888"

def match_model(model_id):
    """Return display name if model is in TARGET_MODELS."""
    mid = model_id.lower()
    for tm in TARGET_MODELS:
        if tm in mid:
            # check specific override first
            for key, val in MODEL_DISPLAY_NAMES.items():
                if key in mid:
                    return val
            return tm.title()
    return None

def load_data():
    data = []
    seen = set()
    
    # Iterate over summary files
    for f in sorted(EXP_DIR.glob("*/summary.json")):
        try:
            with open(f) as fp:
                d = json.load(fp)
        except:
            continue
            
        name = d.get("name", "")
        if "test" not in name: continue # Only test runs
        
        mid = d.get("model_id", "")
        disp_name = match_model(mid)
        if not disp_name: continue
        
        # Dedup (keep best score)
        if disp_name in seen: continue
        seen.add(disp_name)
        
        # Extract metrics
        entry = {
            "name": disp_name,
            "score": d.get("overall_score", 0),
            "cost": d.get("estimated_cost_usd", 0),
            "latency": d.get("avg_latency_seconds", 0),
            "size": get_size(disp_name),
            "questions": d.get("num_questions", 0)
        }
        data.append(entry)
    
    return sorted(data, key=lambda x: x["score"], reverse=True)

def get_size(name):
    # Hardcoded sizes for logic
    if "Haiku" in name: return 8
    if "Sonnet" in name: return 70
    if "70B" in name: return 70
    if "Scout" in name or "Maverick" in name: return 17
    if "20B" in name: return 20
    if "120B" in name: return 120
    if "DeepSeek" in name: return 70
    return 10 # Default

# ============================================================================
# Plotting Functions
# ============================================================================

def setup_style():
    plt.rcParams['font.size'] = FONT_LABEL
    plt.rcParams['axes.titlesize'] = FONT_TITLE
    plt.rcParams['lines.linewidth'] = 2
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.3

def plot_ranking(data):
    """Horizontal Bar Chart - Cleanest for comparison."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    names = [d["name"] for d in data]
    scores = [d["score"] for d in data]
    colors = [get_color(n) for n in names]
    
    y_pos = np.arange(len(names))
    
    bars = ax.barh(y_pos, scores, align='center', color=colors, alpha=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names)
    ax.invert_yaxis()  # labels read top-to-bottom
    ax.set_xlabel('Overall Score')
    ax.set_title(f'Model Performance Ranking (N={data[0]["questions"]})')
    ax.set_xlim(0.5, 0.8) # Zoom in on the tight range
    
    # Add values to bars
    for bar, score in zip(bars, scores):
        ax.text(score + 0.005, bar.get_y() + bar.get_height()/2, 
                f"{score:.3f}", va='center', fontsize=10, fontweight='bold')
                
    plt.tight_layout()
    save_plot("ranking_bar_chart.png")

def plot_scatter(data, x_key, y_key, xlabel, ylabel, title, filename, log_x=False):
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    
    xs = [d[x_key] for d in data]
    ys = [d[y_key] for d in data]
    names = [d["name"] for d in data]
    colors = [get_color(n) for n in names]
    
    # Jitter for Size plots to separate stacking
    if x_key == "size":
        # Add slight random jitter to identical sizes for visibility
        # But deterministic based on name hash to stay consistent
        for i, name in enumerate(names):
            if xs[i] in [8, 17, 70]: # Common buckets
                np.random.seed(sum(map(ord, name)))
                xs[i] += np.random.uniform(-1, 1)

    ax.scatter(xs, ys, c=colors, s=150, edgecolors='white', zorder=5)
    
    # Annotate with smart offsets
    for i, name in enumerate(names):
        x, y = xs[i], ys[i]
        
        # Apply manual offsets if defined
        dx, dy = 0, 0
        # Simple heuristic for common clusters if not manually set
        if name in SCATTER_OFFSETS:
            dx, dy = SCATTER_OFFSETS[name]
        
        # Adjust placement
        ha = 'center'
        if dx > 0: ha = 'left'
        if dx < 0: ha = 'right'
        
        ax.annotate(name, (x, y), xytext=(dx*100, dy*100), 
                   textcoords='offset points', ha=ha, va='center',
                   fontsize=10, bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7, ec="none"))

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    
    if log_x:
        ax.set_xscale('log')
        ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
        
    plt.tight_layout()
    save_plot(filename)

def save_plot(name):
    path = PLOT_DIR / name
    plt.savefig(path, dpi=300, bbox_inches='tight')
    print(f"Saved {path}")

# ============================================================================
# Main
# ============================================================================

def main():
    setup_style()
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    
    print("Loading data...")
    data = load_data()
    if not data:
        print("No data found!")
        return

    print(f"Generating plots for {len(data)} models...")
    
    # 1. Ranking (Essential)
    plot_ranking(data)
    
    # 2. Size vs Score (Trade-off)
    plot_scatter(data, "size", "score", 
                "Model Size (Billions of Params)", "Overall Score",
                "Score vs Model Size", "scatter_size_score.png", log_x=True)
                
    # 3. Cost vs Score (Value)
    plot_scatter(data, "cost", "score",
                "Total Benchmark Cost ($)", "Overall Score",
                "Value Map: Score vs Cost", "scatter_cost_score.png")

    # 4. Latency vs Score (Speed)
    plot_scatter(data, "latency", "score",
                "Avg Latency (s)", "Overall Score",
                "Latency vs Score", "scatter_latency_score.png")

if __name__ == "__main__":
    main()
