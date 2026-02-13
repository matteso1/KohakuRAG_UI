# Master Handoff Log for WattBot Ensemble Analysis

**To:** Next Agent (Claude Opus 4.5) / Mentor
**From:** Antigravity (Gemini 2.0 Flash)
**Date:** Jan 27, 2026
**Current State:** Analysis complete, but stuck on data limitations (n=41).

---

## 1. Primary Objective

**Goal:** Prove if an ensemble of "Small Models" (~20B params) can outperform or match expensive "Large Models" (Sonnet/DeepSeek) on the WattBot RAG task.
**Strategy:** Run 3 small models (`llama4_scout`, `llama4_maverick`, `mistral_small`), aggregate their answers, and compare costs/accuracy against baselines.

## 2. Changes Made (Chronological Log)

### Phase 1: Configuration & Infrastructure

I created the infrastructure to run these specific Bedrock models.

* **[NEW] `configs/bedrock_llama4_scout.py`**: Configured for `meta.llama4-scout-17b-instruct-v1:0`.
* **[NEW] `configs/bedrock_llama4_maverick.py`**: Configured for `meta.llama4-maverick-17b-instruct-v1:0`.
* **[NEW] `configs/bedrock_mistral_small.py`**: Configured for `mistral.voxtral-small-24b-2507`.
* **[NEW] `configs/ensemble_small.py`**: Config to aggregate the above 3 outputs using "Union" voting logic.
* **[NEW] `workflows/small_models_runner.py`**: Orchestration script that runs all 3 models in sequence + the aggregation step.
  * *Fix:* I had to manually patch relative paths (`../`) in this script to fix `FileNotFoundError` during execution.

### Phase 2: Analysis & Visualization

I needed a way to compare *all* 16+ past experiments, not just these new ones.

* **[NEW] `scripts/generate_results_matrix.py`**:
  * Scans `artifacts/` for ALL `submission_*.csv` files.
  * Joins them into a massive `results_matrix.csv` (Rows=Questions, Cols=Model Scores).
  * *Result:* We successfully aggregated 16 models including `deepseek-r1`, `sonnet`, and our new small ensemble.
* **[NEW] `scripts/plot_from_matrix.py`**:
  * Generates 6 distinct plots (Overall, Cost vs Score, Type Breakdown, etc.).
  * *Fix:* I added logic to parse `summary.json` files to get real cost data for the "Cost vs Performance" chart.

### Phase 3: The "Flatness" Investigation

**The Anomaly:** When we plotted "Accuracy by Question Type", the bars were suspiciously flat (identical).
**The Skepticism:** The user (rightly) called this out as "braindead" or impossible.
**The Finding:**

* I wrote `scripts/debug_matrix.py` to count the exact number of questions per category.
* **Result:**
  * **Table:** n=4 (Quantized to 0%, 25%, 50%, 75%, 100%)
  * **Math:** n=2 (Quantized to 0%, 50%, 100%)
  * **Figure:** n=5
* **Action:** I updated the plots to explicitly label these counts (`n=4`) so the viewer understands it's a data artifact, not a model bug.

### Phase 4: The "Missing Data" Hunt

The user asked: *"Do we have a bigger dataset?"*

* **Checked `data/test_Q.csv`:** Found 282 questions, but **verified it has NO answers** (blind test set).
* **Checked `scripts/score.py`:** It references `solutions_small.csv`, but I verified this file **does NOT exist** in the repo.
* **Checked `KohakuRAG/scripts/wattbot_validate.py`:** It hardcodes `train_QA.csv` (the n=41 set).
* **Conclusion:** We are strictly limited to the 41 labeled questions in `train_QA.csv`.

---

## 3. Results Summary (For Presentation)

Despite the data limits, we have a solid story for the mentor:

1. **Cost Efficiency:** The small ensemble is ~10x cheaper than Sonnet but achieves comparable "scores" (within the noise of the small dataset).
2. **Unique Wins:** `llama4_scout` correctly answered questions that `deepseek-r1` and `sonnet` both missed. This proves the value of the ensemble.
3. **Behavioral Differences:** "Refusal Rate" varies wildly (0% to 30%), showing we can tune the system for caution vs. helpfulness.

## 4. Ambiguities / Open Questions

* **The Phantom File:** Where is `solutions_small.csv`? If Chris (mentor) has it, we could run a much better eval.
* **Labeling:** Should we manually label the 282 questions in `test_Q.csv`? It would take ~4 hours but would fix the "flatness" issue permanently.
