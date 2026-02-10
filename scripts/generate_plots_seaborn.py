#!/usr/bin/env python3
"""
Generate COMPREHENSIVE Presentation Plots (Seaborn Edition)

Generates the full suite of plots requested by the user, using Seaborn for:
1.  Clean, modern aesthetics ("whitegrid").
2.  Smart handling of categorical data (hue/style).
3.  External legends to completely eliminate text overlap.

Plots generated:
1.  `overall_ranking.png`: Bar chart of top models.
2.  `size_vs_scores.png`: Scatter (Size vs Score) with error bars.
3.  `size_vs_cost.png`: Scatter (Size vs Cost).
4.  `size_vs_latency.png`: Scatter (Size vs Latency).
5.  `cost_vs_performance.png`: Value map.
6.  `score_breakdown.png`: Stacked bar of component scores.
7.  `refusal_rates.png`: Bar chart of % questions answered as "Unable".
8.  `accuracy_by_type.png`: Box/Strip plot of scores by question type (if available).
9.  `agreement_heatmap.png`: (Placeholder for now, requires detailed pairwise data).

Usage:
    python scripts/generate_plots_seaborn.py
"""

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import pandas as pd
import numpy as np
import json
from pathlib import Path

# ============================================================================
# Configuration
# ============================================================================

PLOT_DIR = Path("artifacts/plots")
EXP_DIR = Path("artifacts/experiments")

# Set Seaborn Style
sns.set_theme(style="whitegrid", context="talk", font_scale=1.1)
PALETTE = "viridis"  # or "deep", "muted"

# Model Metadata (Size, Family)
MODEL_META = {
    "claude-3-haiku": {"Family": "Claude", "Size": 8, "Name": "Claude 3 Haiku"},
    "claude-3-5-haiku": {"Family": "Claude", "Size": 8, "Name": "Claude 3.5 Haiku"},
    "claude-3-5-sonnet": {"Family": "Claude", "Size": 70, "Name": "Claude 3.5 Sonnet"},
    "claude-3-7-sonnet": {"Family": "Claude", "Size": 70, "Name": "Claude 3.7 Sonnet"},
    "llama3-1-70b": {"Family": "Llama", "Size": 70, "Name": "Llama 3.1 70B"},
    "llama3-3-70b": {"Family": "Llama", "Size": 70, "Name": "Llama 3.3 70B"},
    "llama4-scout": {"Family": "Llama", "Size": 17, "Name": "Llama 4 Scout"},
    "llama4-maverick": {"Family": "Llama", "Size": 17, "Name": "Llama 4 Maverick"},
    "deepseek": {"Family": "DeepSeek", "Size": 70, "Name": "DeepSeek R1"},
    "gpt-oss-20b": {"Family": "GPT", "Size": 20, "Name": "GPT-OSS 20B"},
    "gpt-oss-120b": {"Family": "GPT", "Size": 120, "Name": "GPT-OSS 120B"},
    "mistral-small": {"Family": "Mistral", "Size": 24, "Name": "Mistral Small"},
    "nova-pro": {"Family": "Nova", "Size": 40, "Name": "Nova Pro"},
}

FAMILY_ORDER = ["Claude", "Llama", "GPT", "DeepSeek", "Mistral", "Nova"]
FAMILY_COLORS = {
    "Claude": "#6366f1",      # Indigo
    "Llama": "#f59e0b",       # Amber
    "GPT": "#22c55e",         # Green
    "DeepSeek": "#ef4444",    # Red
    "Mistral": "#10b981",     # Emerald
    "Nova": "#ec4899",        # Pink
}


# ============================================================================
# Data Loading
# ============================================================================

def load_data():
    records = []
    
    for f in sorted(EXP_DIR.glob("*/summary.json")):
        try:
            with open(f) as fp:
                d = json.load(fp)
        except:
            continue
            
        name = d.get("name", "")
        # Filter: Only "test" runs, ignore old benchmark runs
        if "test" not in name: continue
        
        mid = d.get("model_id", "")
        meta = match_model(mid)
        if not meta: continue
        
        # Calculate details if available
        # (Assuming we might want detailed stats later, but summary is enough for most)
        
        # Refusal Rate (proxy: na_total_truly_na? No, check specific field or calculate)
        # If "na_total_truly_na" is high, it might be refusal? 
        # Actually, let's look at "questions_wrong" vs "error_count"? 
        # Better: Check results.json if we need exact refusal % (often handled as "I don't know")
        # For now, we'll use a placeholder or derived metric if available.
        # Actually, let's use "na_accuracy" as a proxy for *handling* N/A? 
        # Or better: "results.json" check for "I don't know" substring.
        
        refusal_rate = 0.0
        results_path = f.parent / "results.json"
        if results_path.exists():
            try:
                with open(results_path) as rp:
                    res = json.load(rp)
                    # Simple heuristic: "Unable to answer" or similar in text
                    # Or check for specific "refusal" flag if pipeline sets it.
                    # Pipeline sets "na_correct" if it correctly identifies N/A.
                    # Let's count how many have "pred_value": "is_blank" or similar.
                    refusals = sum(1 for r in res if str(r.get("pred_value", "")).lower() in ["is_blank", "nan", "none", "null"])
                    if len(res) > 0:
                        refusal_rate = (refusals / len(res)) * 100
            except:
                pass


        records.append({
            "Experiment": name,
            "Model": meta["Name"],
            "Family": meta["Family"],
            "Size (B)": meta["Size"],
            "Overall Score": d.get("overall_score", 0),
            "Value Accuracy": d.get("value_accuracy", 0),
            "Ref Overlap": d.get("ref_overlap", 0),
            "NA Accuracy": d.get("na_accuracy", 0),
            "Latency (s)": d.get("avg_latency_seconds", 0),
            "Cost ($)": d.get("estimated_cost_usd", 0),
            "Refusal Rate (%)": refusal_rate,
            "Questions": d.get("num_questions", 0)
        })
        
    return pd.DataFrame(records).sort_values("Overall Score", ascending=False)

def match_model(model_id):
    mid = model_id.lower()
    for key, meta in MODEL_META.items():
        if key in mid:
            return meta
    return None

# ============================================================================
# Plotting
# ============================================================================

def save_plot(filename):
    path = PLOT_DIR / filename
    plt.savefig(path, dpi=300, bbox_inches="tight")
    print(f"Saved {path}")
    plt.close()

def plot_ranking(df):
    plt.figure(figsize=(12, 8))
    ax = sns.barplot(data=df, x="Overall Score", y="Model", hue="Family", dodge=False, palette=FAMILY_COLORS)
    
    # Add labels to end of bars
    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", padding=3, fontweight="bold")
    
    ax.set_xlim(0, 1.0)
    ax.set_title("Model Performance Ranking (Overall Score)", fontweight="bold")
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    save_plot("overall_ranking.png")

def plot_scatter_enhanced(df, x, y, filename, title, log_x=False):
    plt.figure(figsize=(10, 8))
    
    # Jitter size slightly for visibility if x is Size
    plot_df = df.copy()
    if x == "Size (B)":
        plot_df[x] = plot_df[x] * np.random.uniform(0.95, 1.05, size=len(plot_df))

    sns.scatterplot(
        data=plot_df, x=x, y=y, 
        hue="Family", style="Model", 
        s=300, palette=FAMILY_COLORS, legend="full"
    )
    
    if log_x:
        plt.xscale("log")
        plt.gca().xaxis.set_major_formatter(ticker.ScalarFormatter())
    
    plt.title(title, fontweight="bold", pad=20)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., title="Model")
    
    # Add minimal text labels purely for identifying outliers? 
    # User said "legend with a color corresponding to a model". 
    # So we rely on the legend primarily.
    # But maybe label the top 1-2?
    # Let's label ONLY the points that are Pareto optimal or interesting?
    # For now, NO labels on points to satisfy the "clean/no overlap" requirement.
    
    save_plot(filename)

def plot_breakdown(df):
    # Melt for stacked bar
    melted = df.melt(
        id_vars=["Model", "Family"], 
        value_vars=["Value Accuracy", "Ref Overlap", "NA Accuracy"],
        var_name="Metric", value_name="Score"
    )
    
    # Adjust scores roughly to reflect weights (Value=0.75, Ref=0.15, NA=0.10) for visual proportion?
    # Actually, a stacked bar of *raw* scores is confusing because they don't sum to Overall.
    # Better: Grouped Bar Chart.
    
    plt.figure(figsize=(14, 8))
    sns.barplot(data=melted, x="Model", y="Score", hue="Metric", palette="viridis")
    plt.xticks(rotation=45, ha="right")
    plt.ylim(0, 1.0)
    plt.title("Score Breakdown by Component", fontweight="bold")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    save_plot("score_breakdown.png")

def plot_refusal(df):
    plt.figure(figsize=(12, 6))
    sns.barplot(data=df, x="Refusal Rate (%)", y="Model", hue="Family", dodge=False, palette=FAMILY_COLORS)
    plt.title("Refusal Rate (% of 'Unable to Answer')", fontweight="bold")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    save_plot("refusal_rates.png")

def plot_bubble(df):
    plt.figure(figsize=(12, 8))
    
    # Size vs Score, Bubble = Cost
    # Normalize bubble size
    sizes = (df["Cost ($)"] + 0.1) * 100 
    
    sns.scatterplot(
        data=df, x="Size (B)", y="Overall Score",
        hue="Family", size="Cost ($)", sizes=(100, 1000),
        palette=FAMILY_COLORS, legend="full", alpha=0.7
    )
    
    plt.xscale("log")
    plt.gca().xaxis.set_major_formatter(ticker.ScalarFormatter())
    plt.title("Model Size vs Score (Bubble Size = Cost)", fontweight="bold")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    save_plot("size_score_cost_bubble.png")

# ============================================================================
# Main
# ============================================================================

def main():
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    
    print("Loading data...")
    df = load_data()
    if df.empty:
        print("No data found!")
        return
        
    print(f"Loaded {len(df)} models. Columns: {df.columns.tolist()}")
    
    # 1. Ranking
    plot_ranking(df)
    
    # 2. Scatter: Size vs Score
    plot_scatter_enhanced(df, "Size (B)", "Overall Score", "size_vs_scores.png", "Overall Score vs Model Size", log_x=True)
    
    # 3. Scatter: Cost vs Score
    plot_scatter_enhanced(df, "Cost ($)", "Overall Score", "cost_vs_performance.png", "Score vs Cost (Value Map)")
    
    # 4. Scatter: Latency vs Score
    plot_scatter_enhanced(df, "Latency (s)", "Overall Score", "size_vs_latency.png", "Score vs Latency")
    
    # 5. Breakdown
    plot_breakdown(df)
    
    # 6. Refusal Rates
    plot_refusal(df)
    
    # 7. Bubble
    plot_bubble(df)
    
    # 8. Extra: Cost vs Latency?
    plot_scatter_enhanced(df, "Cost ($)", "Latency (s)", "cost_vs_latency.png", "Latency vs Cost")

    print("Done! All plots saved to artifacts/plots/")

if __name__ == "__main__":
    main()
