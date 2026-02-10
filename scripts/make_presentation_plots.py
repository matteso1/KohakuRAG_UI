#!/usr/bin/env python3
"""
make_presentation_plots.py  –  Presentation-grade plots for ML+X meeting

Generates 8 publication-quality plots using Seaborn.
Scatter plots use NUMBERED MARKERS + a legend table beneath the chart
to guarantee zero text overlap regardless of data clustering.

Usage:
    python scripts/make_presentation_plots.py
"""

import json, math, sys, itertools
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import seaborn as sns

try:
    from scipy import stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

# ============================================================================
# Global Config
# ============================================================================

EXP_DIR   = Path("artifacts/experiments")
MATRIX    = Path("artifacts/results_matrix_test.csv")
PLOT_DIR  = Path("artifacts/plots")

# Models to include (test runs only)
MODEL_ORDER = [
    "claude37-sonnet-test",
    "sonnet-test",
    "deepseek-r1-test",
    "gpt-oss-120b-test",
    "llama4-maverick-test",
    "claude35-haiku-test",
    "gpt-oss-20b-test",
    "llama4-scout-test",
    "llama3-70b-test",
    "haiku-test",
]

DISPLAY = {
    "claude37-sonnet-test":  "Claude 3.7 Sonnet",
    "sonnet-test":           "Claude 3.5 Sonnet",
    "deepseek-r1-test":      "DeepSeek R1",
    "gpt-oss-120b-test":     "GPT-OSS 120B",
    "llama4-maverick-test":  "Llama 4 Maverick",
    "claude35-haiku-test":   "Claude 3.5 Haiku",
    "gpt-oss-20b-test":      "GPT-OSS 20B",
    "llama4-scout-test":     "Llama 4 Scout",
    "llama3-70b-test":       "Llama 3.3 70B",
    "haiku-test":            "Claude 3 Haiku",
}

# Short labels for scatter plots — descriptive but compact
SHORT = {
    "claude37-sonnet-test":  "C-3.7",
    "sonnet-test":           "C-3.5",
    "deepseek-r1-test":      "DS-R1",
    "gpt-oss-120b-test":     "GPT-120",
    "llama4-maverick-test":  "L-Mav",
    "claude35-haiku-test":   "C-H3.5",
    "gpt-oss-20b-test":      "GPT-20",
    "llama4-scout-test":     "L-Scout",
    "llama3-70b-test":       "L3-70B",
    "haiku-test":            "Haiku3",
}

FAMILY = {
    "claude37-sonnet-test":  "Claude",
    "sonnet-test":           "Claude",
    "deepseek-r1-test":      "DeepSeek",
    "gpt-oss-120b-test":     "GPT",
    "llama4-maverick-test":  "Llama",
    "claude35-haiku-test":   "Claude",
    "gpt-oss-20b-test":      "GPT",
    "llama4-scout-test":     "Llama",
    "llama3-70b-test":       "Llama",
    "haiku-test":            "Claude",
}

SIZE_B = {
    "claude37-sonnet-test":  70,
    "sonnet-test":           70,
    "deepseek-r1-test":      70,
    "gpt-oss-120b-test":     120,
    "llama4-maverick-test":  17,
    "claude35-haiku-test":   8,
    "gpt-oss-20b-test":      20,
    "llama4-scout-test":     17,
    "llama3-70b-test":       70,
    "haiku-test":            8,
}

FAMILY_PAL = {
    "Claude":   "#6366f1",
    "Llama":    "#f59e0b",
    "GPT":      "#22c55e",
    "DeepSeek": "#ef4444",
    "Mistral":  "#10b981",
    "Nova":     "#ec4899",
}

# ============================================================================
# Helpers
# ============================================================================

def dname(m):
    return DISPLAY.get(m, m)

def fcolor(m):
    return FAMILY_PAL.get(FAMILY.get(m, ""), "#888888")

def wilson_ci(k, n, z=1.96):
    """Wilson score interval for binomial proportion."""
    if n == 0: return 0.0, 0.0
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2*n)) / denom
    spread = z * math.sqrt(p*(1-p)/n + z**2/(4*n**2)) / denom
    return max(0, centre - spread), min(1, centre + spread)

def load_summaries():
    """Load cost / latency from summary.json files."""
    info = {}
    for d in EXP_DIR.iterdir():
        s = d / "summary.json"
        if not s.exists(): continue
        try:
            data = json.loads(s.read_text())
            info[d.name] = {
                "cost": data.get("estimated_cost_usd", 0),
                "latency": data.get("avg_latency_seconds", 0),
            }
        except: pass
    return info

def save(name):
    p = PLOT_DIR / name
    plt.savefig(p, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  ✓ {p}")

# ============================================================================
# Score Calculation
# ============================================================================

def compute_scores(df, models):
    """Compute per-model Overall, Value, Ref, NA scores + 95% CI."""
    gt_col = "GT_Value"
    truly_na = df[gt_col].fillna("").astype(str).str.strip().str.lower().isin(
        {"", "na", "n/a", "is_blank"}
    ) | df[gt_col].fillna("").astype(str).str.lower().str.startswith("unable")
    n_na = truly_na.sum()
    n = len(df)

    out = {}
    for m in models:
        vc = f"{m}_ValCorrect"
        rc = f"{m}_RefScore"
        nc = f"{m}_NACorrect"
        if vc not in df.columns: continue

        val = df[vc].mean()
        ref = df[rc].mean() if rc in df.columns else 0
        na_r = df.loc[truly_na, nc].mean() if nc in df.columns and n_na > 0 else 1.0

        overall = 0.75*val + 0.15*ref + 0.10*na_r

        # SE propagation for 95% CI
        se_v = df[vc].std()/math.sqrt(n) if n else 0
        se_r = (df[rc].std()/math.sqrt(n)) if rc in df.columns and n else 0
        se_n = math.sqrt(na_r*(1-na_r)/n_na) if n_na > 0 else 0
        se  = math.sqrt((0.75*se_v)**2 + (0.15*se_r)**2 + (0.10*se_n)**2)
        if math.isnan(se): se = 0

        out[m] = dict(
            overall=overall, val=val, ref=ref, na=na_r,
            ci=1.96*se, n=n
        )
    return out

# ============================================================================
# Plot 1: Overall Ranking (Horizontal Bar)
# ============================================================================

def plot_ranking(scores):
    models = sorted(scores, key=lambda m: scores[m]["overall"], reverse=True)
    n_q = scores[models[0]]["n"]

    fig, ax = plt.subplots(figsize=(11, 7))
    y = np.arange(len(models))
    vals = [scores[m]["overall"] for m in models]
    cis  = [scores[m]["ci"]      for m in models]
    cols = [fcolor(m) for m in models]
    names = [dname(m) for m in models]

    bars = ax.barh(y, vals, height=0.6, color=cols, xerr=cis,
                   capsize=4, error_kw=dict(lw=1.2, color="#333"))
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=13)
    ax.invert_yaxis()
    ax.set_xlim(0.5, 0.88)
    ax.set_xlabel("WattBot Score  (0.75·Val + 0.15·Ref + 0.10·NA)", fontsize=12)
    ax.set_title(f"Model Performance Ranking  (n = {n_q} questions)", fontsize=16, fontweight="bold")

    # Place score label PAST the error bar cap so they never collide
    for bar, v, ci in zip(bars, vals, cis):
        label_x = v + ci + 0.008  # start after CI cap
        ax.text(label_x, bar.get_y() + bar.get_height()/2,
                f"{v:.3f}", va="center", fontsize=11, fontweight="bold")

    ax.text(0.01, 0.99, "Error bars: 95 % CI", transform=ax.transAxes,
            fontsize=9, va="top", style="italic", color="gray")
    ax.grid(axis="x", ls="--", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    save("01_overall_ranking.png")

# ============================================================================
# Plot 2: Score Breakdown (Grouped Bar)
# ============================================================================

def plot_breakdown(scores):
    models = sorted(scores, key=lambda m: scores[m]["overall"], reverse=True)
    n_q = scores[models[0]]["n"]
    names = [dname(m) for m in models]
    x = np.arange(len(models))
    w = 0.25

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.bar(x - w, [scores[m]["val"] for m in models], w, label="Value Accuracy (75 %)", color="#6366f1", alpha=0.85)
    ax.bar(x,     [scores[m]["ref"] for m in models], w, label="Ref Overlap (15 %)",    color="#f59e0b", alpha=0.85)
    ax.bar(x + w, [scores[m]["na"]  for m in models], w, label="NA Recall (10 %)",      color="#22c55e", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score", fontsize=13)
    ax.set_title(f"Score Component Breakdown  (n = {n_q})", fontsize=16, fontweight="bold")
    ax.legend(fontsize=11, loc="upper right")
    ax.grid(axis="y", ls="--", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    save("02_score_breakdown.png")

# ============================================================================
# Plot 3: Refusal Rates (Horizontal Bar)
# ============================================================================

def plot_refusals(df, models):
    rates, ci_lo, ci_hi = [], [], []
    valid = []
    for m in models:
        vc = f"{m}_Value"
        if vc not in df.columns: continue
        vals = df[vc].fillna("is_blank").astype(str).str.lower()
        ref_mask = vals.apply(lambda x: "is_blank" in x or "unable" in x)
        k = int(ref_mask.sum())
        n = len(df)
        rate = k / n * 100
        lo, hi = wilson_ci(k, n)
        rates.append(rate)
        ci_lo.append(rate - lo*100)
        ci_hi.append(hi*100 - rate)
        valid.append(m)

    # Sort by rate
    order = np.argsort(rates)
    rates = [rates[i] for i in order]
    ci_lo = [ci_lo[i] for i in order]
    ci_hi = [ci_hi[i] for i in order]
    valid = [valid[i] for i in order]
    names = [dname(m) for m in valid]

    fig, ax = plt.subplots(figsize=(11, 7))
    y = np.arange(len(valid))
    bars = ax.barh(y, rates, height=0.6, color=[fcolor(m) for m in valid],
                   xerr=[ci_lo, ci_hi], capsize=4, error_kw=dict(lw=1.2, color="#333"))
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=12)
    ax.set_xlabel("Refusal Rate (%)", fontsize=12)
    ax.set_title("Refusal Rate  (% answered 'Unable')", fontsize=16, fontweight="bold")
    # Place label PAST the error bar cap so they never collide
    for bar, r, ci_h in zip(bars, rates, ci_hi):
        label_x = r + ci_h + 0.8
        ax.text(label_x, bar.get_y() + bar.get_height()/2,
                f"{r:.1f}%", va="center", fontsize=10, fontweight="bold")
    ax.set_xlim(0, max(r + c for r, c in zip(rates, ci_hi)) + 8)
    ax.text(0.01, 0.99, "Error bars: 95 % Wilson CI", transform=ax.transAxes,
            fontsize=9, va="top", style="italic", color="gray")
    ax.grid(axis="x", ls="--", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    save("03_refusal_rates.png")

# ============================================================================
# Plot 4: Unique Wins (Horizontal Bar)
# ============================================================================

def plot_unique_wins(df, models):
    wins = {}
    for mt in models:
        tc = f"{mt}_ValCorrect"
        if tc not in df.columns: continue
        others = [f"{m}_ValCorrect" for m in models if m != mt and f"{m}_ValCorrect" in df.columns]
        if not others: continue
        mask = (df[tc] == True) & (df[others] == False).all(axis=1)
        wins[mt] = int(mask.sum())

    order = sorted(wins, key=lambda m: wins[m], reverse=True)
    names = [dname(m) for m in order]
    vals  = [wins[m] for m in order]

    fig, ax = plt.subplots(figsize=(11, 7))
    y = np.arange(len(order))
    bars = ax.barh(y, vals, height=0.6, color=[fcolor(m) for m in order])
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=12)
    ax.invert_yaxis()
    ax.set_xlabel("# Questions", fontsize=12)
    ax.set_title("Unique Wins  (questions only THIS model got right)", fontsize=16, fontweight="bold")
    for bar, v in zip(bars, vals):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                str(v), va="center", fontsize=11, fontweight="bold")
    ax.grid(axis="x", ls="--", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    save("04_unique_wins.png")

# ============================================================================
# Pareto Frontier Scatter — for Cost and Latency vs Score
# ============================================================================

def pareto_front(xs, ys):
    """Return indices of Pareto-optimal points (minimise x, maximise y)."""
    pareto = []
    for i in range(len(xs)):
        dominated = False
        for j in range(len(xs)):
            if i == j: continue
            # j dominates i if j is at least as good on BOTH axes and strictly better on one
            if xs[j] <= xs[i] and ys[j] >= ys[i] and (xs[j] < xs[i] or ys[j] > ys[i]):
                dominated = True
                break
        if not dominated:
            pareto.append(i)
    return sorted(pareto, key=lambda i: xs[i])

# Hand-tuned label offsets so no two labels overlap.
# Key = short label, value = (dx, dy) in points from the data point.
# Positive dx = right, positive dy = up.
COST_OFFSETS = {
    "C-3.7": (15, 12),
    "C-3.5": (15, -18),
    "DS-R1":   (-55, 12),
    "GPT-120": (15, 10),
    "L-Mav":  (15, -18),
    "C-H3.5": (15, 8),
    "GPT-20":  (15, 12),
    "L-Scout":  (-60, -5),
    "L3-70B":  (15, -18),
    "Haiku3":   (15, 8),
}

LATENCY_OFFSETS = {
    "C-3.7": (15, 10),
    "C-3.5": (15, -18),
    "DS-R1":   (15, -18),
    "GPT-120": (15, 8),
    "L-Mav":  (15, -18),
    "C-H3.5": (15, 12),
    "GPT-20":  (-55, 8),
    "L-Scout":  (-60, -5),
    "L3-70B":  (-55, 8),
    "Haiku3":   (15, -18),
}

def plot_pareto_scatter(models, xs, ys, xlabel, ylabel, title, filename,
                        x_fmt_str="{:.2f}", offsets=None):
    """
    Scatter plot with Pareto frontier line and leader-line labels.
    Each label has a thin arrow pointing to its dot.
    """
    from matplotlib.lines import Line2D

    fig, ax = plt.subplots(figsize=(11, 8))

    FAMILY_MARKERS = {"Claude": "o", "Llama": "s", "GPT": "D", "DeepSeek": "^",
                      "Mistral": "v", "Nova": "p"}

    # Draw all points
    for i, m in enumerate(models):
        marker = FAMILY_MARKERS.get(FAMILY.get(m, ""), "o")
        ax.scatter(xs[i], ys[i], color=fcolor(m), s=180, marker=marker,
                   zorder=5, edgecolors="white", linewidth=1.5)

    # Pareto frontier line
    pidx = pareto_front(xs, ys)
    if len(pidx) >= 2:
        px = [xs[i] for i in pidx]
        py = [ys[i] for i in pidx]
        ax.plot(px, py, color="#888", ls="--", lw=1.5, alpha=0.7, zorder=3)
        ax.fill_between(px, py, max(ys)*1.05, color="#d4edda", alpha=0.15, zorder=1)

    # Labels with LEADER LINES (thin arrows connecting label → dot)
    for i, m in enumerate(models):
        lbl = SHORT.get(m, m[:3])
        # Use hand-tuned offsets if provided, otherwise default
        if offsets and lbl in offsets:
            ox, oy = offsets[lbl]
        else:
            ox, oy = 15, 8
        ax.annotate(lbl, (xs[i], ys[i]), xytext=(ox, oy),
                    textcoords="offset points", fontsize=9, fontweight="bold",
                    color="#333",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#ccc", alpha=0.9),
                    arrowprops=dict(arrowstyle="-", color="#999", lw=0.8,
                                    connectionstyle="arc3,rad=0.1"),
                    zorder=7)

    ax.set_xlabel(xlabel, fontsize=13)
    ax.set_ylabel(ylabel, fontsize=13)
    ax.set_title(title, fontsize=16, fontweight="bold", pad=15)
    ax.grid(True, ls="--", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Padding
    xr = max(xs) - min(xs) if max(xs) != min(xs) else 1
    yr = max(ys) - min(ys) if max(ys) != min(ys) else 0.1
    ax.set_xlim(min(xs) - xr*0.1, max(xs) + xr*0.15)
    ax.set_ylim(min(ys) - yr*0.15, max(ys) + yr*0.15)

    # Family legend
    seen = {}
    for m in models:
        f = FAMILY.get(m, "")
        if f not in seen:
            seen[f] = (fcolor(m), FAMILY_MARKERS.get(f, "o"))
    handles = [Line2D([0],[0], marker=mk, color="w", markerfacecolor=c,
                      markersize=10, label=f) for f, (c, mk) in seen.items()]
    ax.legend(handles=handles, title="Family", loc="lower right", fontsize=10,
             framealpha=0.9)

    # Pareto label
    if len(pidx) >= 2:
        ax.annotate("Pareto frontier", xy=(0.98, 0.02), xycoords="axes fraction",
                    ha="right", fontsize=9, color="#888", style="italic")

    save(filename)

# ============================================================================
# Category Strip Plot — for Size vs Score
# ============================================================================

def plot_size_strip(models, scores):
    """
    X-axis = size bucket (categorical: 8B, 17B, 20B, 70B, 120B).
    Y-axis = overall score.
    Each dot = one model, with short label.
    Directly shows that size does NOT determine quality.
    """
    from matplotlib.lines import Line2D

    # Build data
    rows = []
    for m in models:
        rows.append({
            "model": m,
            "size_label": f"{SIZE_B[m]}B",
            "size": SIZE_B[m],
            "score": scores[m]["overall"],
            "family": FAMILY.get(m, ""),
        })
    pdf = pd.DataFrame(rows)
    pdf = pdf.sort_values("size")

    # Order categories by size
    size_order = sorted(pdf["size_label"].unique(), key=lambda s: int(s.replace("B","")))

    fig, ax = plt.subplots(figsize=(11, 8))

    # Use seaborn stripplot for automatic jitter within each category
    FAMILY_MARKERS = {"Claude": "o", "Llama": "s", "GPT": "D", "DeepSeek": "^", "Mistral": "v", "Nova": "p"}

    # Plot manually for family-specific colors and markers
    for _, row in pdf.iterrows():
        x_pos = size_order.index(row["size_label"])
        marker = FAMILY_MARKERS.get(row["family"], "o")
        ax.scatter(x_pos, row["score"], color=fcolor(row["model"]),
                   s=200, marker=marker, zorder=5, edgecolors="white", linewidth=1.5)
        ax.annotate(SHORT.get(row["model"], row["model"][:3]),
                    (x_pos, row["score"]),
                    xytext=(10, 0), textcoords="offset points",
                    fontsize=9, fontweight="bold", color="#333",
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="#ccc", alpha=0.8))

    ax.set_xticks(range(len(size_order)))
    ax.set_xticklabels(size_order, fontsize=13)
    ax.set_xlabel("Model Size (Parameters)", fontsize=13)
    ax.set_ylabel("Overall WattBot Score", fontsize=13)
    ax.set_title("Model Size vs. Performance", fontsize=16, fontweight="bold", pad=15)
    ax.grid(axis="y", ls="--", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Show spread per category with a thin vertical range line
    for xi, sz in enumerate(size_order):
        subset = pdf[pdf["size_label"] == sz]
        if len(subset) > 1:
            ax.vlines(xi, subset["score"].min(), subset["score"].max(),
                      color="#aaa", lw=1, ls=":", zorder=2)

    yr = pdf["score"].max() - pdf["score"].min()
    ax.set_ylim(pdf["score"].min() - yr*0.1, pdf["score"].max() + yr*0.1)

    # Family legend
    seen = {}
    for m in models:
        f = FAMILY.get(m, "")
        if f not in seen:
            seen[f] = (fcolor(m), FAMILY_MARKERS.get(f, "o"))
    handles = [Line2D([0],[0], marker=mk, color="w", markerfacecolor=c,
                      markersize=10, label=f) for f, (c, mk) in seen.items()]
    ax.legend(handles=handles, title="Family", loc="lower right", fontsize=10,
             framealpha=0.9)

    # Annotation: highlight that smaller model beats bigger ones
    ax.annotate("Size ≠ Quality", xy=(0.02, 0.98), xycoords="axes fraction",
                fontsize=10, va="top", style="italic", color="#888")

    save("06_size_vs_score.png")

# ============================================================================
# Plot 5-7 Suite
# ============================================================================

def plot_metric_suite(scores, summaries, models):
    mwd = [m for m in models if m in summaries]
    ys = [scores[m]["overall"] for m in mwd]

    # 5. Cost vs Score (Pareto frontier)
    xs_cost = [summaries[m]["cost"] for m in mwd]
    plot_pareto_scatter(mwd, xs_cost, ys,
                        "Total Benchmark Cost ($)", "Overall WattBot Score",
                        "Cost vs. Performance  (Pareto Frontier)",
                        "05_cost_vs_score.png", offsets=COST_OFFSETS)

    # 6. Size vs Score (strip plot)
    plot_size_strip(mwd, scores)

    # 7. Latency vs Score (Pareto frontier)
    xs_lat = [summaries[m]["latency"] for m in mwd]
    plot_pareto_scatter(mwd, xs_lat, ys,
                        "Average Latency (seconds)", "Overall WattBot Score",
                        "Latency vs. Performance  (Pareto Frontier)",
                        "07_latency_vs_score.png", offsets=LATENCY_OFFSETS)

# ============================================================================
# Plot 8: Agreement Heatmap
# ============================================================================

def plot_heatmap(df, models):
    n = len(models)
    mat = np.zeros((n, n))
    for i, m1 in enumerate(models):
        for j, m2 in enumerate(models):
            c1 = f"{m1}_ValCorrect"
            c2 = f"{m2}_ValCorrect"
            if c1 in df.columns and c2 in df.columns:
                mat[i,j] = (df[c1] == df[c2]).mean()

    names = [dname(m) for m in models]
    fig, ax = plt.subplots(figsize=(10, 9))
    sns.heatmap(mat, annot=True, fmt=".2f", cmap="YlGnBu",
                xticklabels=names, yticklabels=names,
                vmin=0.5, vmax=1.0, linewidths=0.5, ax=ax,
                annot_kws={"fontsize": 10})
    ax.set_title("Model Agreement (Correctness Correlation)", fontsize=16, fontweight="bold", pad=15)
    plt.xticks(rotation=40, ha="right", fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    save("08_agreement_heatmap.png")

# ============================================================================
# Accuracy by Question Type
# ============================================================================

def plot_accuracy_by_type(df, models):
    """Grouped bar of value accuracy by question type."""
    def get_types(row):
        expl = str(row.get("GT_Explanation", "")).lower()
        types = []
        if "table" in expl: types.append("Table")
        if "figure" in expl or "fig" in expl: types.append("Figure")
        if "quote" in expl: types.append("Quote")
        if "math" in expl or "calculation" in expl: types.append("Math")
        gt_val = str(row.get("GT_Value", "")).lower()
        if "unable" in gt_val or "is_blank" in gt_val:
            types.append("N/A")
        return types

    df["_types"] = df.apply(get_types, axis=1)
    all_types = ["Table", "Figure", "Quote", "Math", "N/A"]

    valid_models = [m for m in models if f"{m}_ValCorrect" in df.columns]
    n_m = len(valid_models)
    pal = sns.color_palette("husl", n_m)

    fig, ax = plt.subplots(figsize=(14, 7))
    x = np.arange(len(all_types))
    w = 0.8 / n_m

    for i, m in enumerate(valid_models):
        accs, ci_lo, ci_hi = [], [], []
        for t in all_types:
            mask = df["_types"].apply(lambda ts: t in ts)
            sub = df.loc[mask]
            vc = f"{m}_ValCorrect"
            if len(sub) == 0 or vc not in sub.columns:
                accs.append(0); ci_lo.append(0); ci_hi.append(0)
                continue
            k = int(sub[vc].sum())
            n = len(sub)
            acc = k / n
            lo, hi = wilson_ci(k, n)
            accs.append(acc)
            ci_lo.append(acc - lo)
            ci_hi.append(hi - acc)

        ax.bar(x + i*w, accs, w, label=dname(m), color=pal[i],
               yerr=[ci_lo, ci_hi], capsize=2, error_kw=dict(lw=0.8))

    type_counts = {t: int(df["_types"].apply(lambda ts: t in ts).sum()) for t in all_types}
    labels = [f"{t}\n(n={type_counts[t]})" for t in all_types]
    ax.set_xticks(x + w*(n_m-1)/2)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Value Accuracy", fontsize=13)
    ax.set_title("Accuracy by Question Type  (95 % CI)", fontsize=16, fontweight="bold")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)
    ax.grid(axis="y", ls="--", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    save("09_accuracy_by_type.png")

# ============================================================================
# Main
# ============================================================================

def main():
    sns.set_theme(style="whitegrid", context="talk", font_scale=0.95)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    df = pd.read_csv(MATRIX)
    print(f"  Matrix: {df.shape[0]} questions, {df.shape[1]} columns")

    # Filter to models present in the matrix
    models = [m for m in MODEL_ORDER if f"{m}_ValCorrect" in df.columns]
    print(f"  Models: {len(models)}")

    summaries = load_summaries()
    scores = compute_scores(df, models)

    print(f"\nGenerating plots...")

    plot_ranking(scores)
    plot_breakdown(scores)
    plot_refusals(df, models)
    plot_unique_wins(df, models)
    plot_metric_suite(scores, summaries, models)
    plot_heatmap(df, models)
    plot_accuracy_by_type(df, models)

    print(f"\nDone! {len(list(PLOT_DIR.glob('*.png')))} plots in {PLOT_DIR}/")

if __name__ == "__main__":
    main()
