# Benchmarking Guide

How to run experiments, compare models, use multiple GPUs, and add new models
to the WattBot RAG evaluation pipeline.

All commands assume you are at the **repo root** with your venv active.

---

## 1) Running a single experiment

Every experiment is driven by a **config file** in `configs/`.
Each config specifies the LLM provider, model, embedding settings, and retrieval
parameters.

```bash
# Run with Qwen 2.5 7B (local HF)
python scripts/run_experiment.py --config configs/hf_qwen7b.py

# Give it a custom name
python scripts/run_experiment.py --config configs/hf_qwen7b.py --name qwen7b-v1
```

Output lands in `artifacts/experiments/<name>/`:

```
artifacts/experiments/qwen7b-v1/
├── submission.csv   # Kaggle-format predictions
├── results.json     # Per-question details (latency, scores, raw LLM output)
└── summary.json     # Aggregate metrics (overall score, timing, cost)
```

### Quick eval (alternative)

`run_wattbot_eval.py` is a lighter-weight script that runs the pipeline and
prints the WattBot score directly:

```bash
# Default: hf_local with Qwen 7B
python scripts/run_wattbot_eval.py

# With a config file
python scripts/run_wattbot_eval.py --config configs/hf_qwen7b.py

# Override provider/model via CLI (no config file needed)
python scripts/run_wattbot_eval.py --provider hf_local --hf-model Qwen/Qwen2.5-1.5B-Instruct
```

---

## 2) Available configs

| Config file              | Model                   | VRAM needed | Provider  |
|--------------------------|-------------------------|-------------|-----------|
| `hf_qwen7b.py`          | Qwen 2.5 7B Instruct   | ~16 GB      | hf_local  |
| `hf_qwen1_5b.py`        | Qwen 2.5 1.5B Instruct | ~4 GB       | hf_local  |
| `hf_llama3_8b.py`       | Llama 3.1 8B Instruct   | ~18 GB      | hf_local  |
| `hf_mistral7b.py`       | Mistral 7B Instruct v0.3| ~16 GB      | hf_local  |
| `hf_phi3_mini.py`       | Phi-3.5 Mini (3.8B)    | ~8 GB       | hf_local  |

Bedrock configs (from the `bedrock` branch) also work if you have
`llm_bedrock.py` and AWS credentials set up.

---

## 3) Running all models (full benchmark)

```bash
# Smoke test first (1 question per model, catches config/loading errors fast)
python scripts/run_full_benchmark.py --smoke-test --provider hf_local

# Full benchmark — all local HF models
python scripts/run_full_benchmark.py --provider hf_local

# Single model only
python scripts/run_full_benchmark.py --model qwen7b

# All providers (local + bedrock, if bedrock configs exist)
python scripts/run_full_benchmark.py
```

The benchmark runner:
- Skips models whose config files don't exist
- Runs each model as a subprocess with a 30-minute timeout
- Prints a pass/fail summary at the end

---

## 4) Comparing results across runs

### Score a submission against ground truth

```bash
python scripts/score.py data/train_QA.csv artifacts/experiments/qwen7b-v1/submission.csv
```

### Generate a side-by-side comparison matrix

```bash
# Auto-discovers all experiments
python scripts/generate_results_matrix.py

# Or specify files manually
python scripts/generate_results_matrix.py \
    --submissions artifacts/experiments/*/submission.csv \
    --output artifacts/results_matrix.csv
```

This produces a CSV where each row is a question and columns show each model's
prediction + correctness, making it easy to spot which questions each model
gets right or wrong.

### Ensemble voting (combine multiple models)

```bash
python scripts/run_ensemble.py \
    --experiments qwen7b-v1 llama3-8b-v1 mistral7b-v1 \
    --name ensemble-3way
```

Aggregation strategies: `majority` (default), `first_non_blank`, `confidence`.

### Audit experiment quality

```bash
python scripts/audit_experiments.py
```

Checks for: missing token counts, high latency, high error rates, score
inconsistencies, duplicate model runs.

---

## 5) Generating plots

### Model size vs. performance (7 plots)

```bash
python scripts/plot_model_size.py
# Output: artifacts/plots/size_vs_scores.png, overall_ranking.png, etc.
```

Plots generated:
1. **Size vs. Scores** — 4-panel (overall, value accuracy, ref overlap, NA)
2. **Size vs. Latency** — per-question average
3. **Size vs. Cost** — API cost (local models show $0)
4. **Bubble chart** — size × score × cost
5. **Overall ranking** — horizontal bar chart
6. **Cost vs. Performance** — trade-off scatter
7. **Score breakdown** — grouped bar (value/ref/NA per model)

Local HF models show as **squares**, API models as **circles**.

### Results matrix plots (6 plots)

```bash
# First generate the matrix
python scripts/generate_results_matrix.py

# Then plot
python scripts/plot_from_matrix.py
```

### Score component breakdown

```bash
python scripts/plot_score_breakdown.py
```

---

## 6) Multi-GPU parallel experiments

If your machine has multiple GPUs (e.g., PowerEdge with 2× A6000), you can
run experiments on different GPUs simultaneously.

### Option A: Manual GPU assignment

Open two terminals and assign each experiment to a different GPU:

```bash
# Terminal 1 — GPU 0
CUDA_VISIBLE_DEVICES=0 python scripts/run_experiment.py --config configs/hf_qwen7b.py --name qwen7b-gpu0

# Terminal 2 — GPU 1
CUDA_VISIBLE_DEVICES=1 python scripts/run_experiment.py --config configs/hf_llama3_8b.py --name llama3-8b-gpu1
```

### Option B: Script it

```bash
#!/bin/bash
# run_parallel.sh — run two experiments on separate GPUs

CUDA_VISIBLE_DEVICES=0 python scripts/run_experiment.py \
    --config configs/hf_qwen7b.py --name qwen7b-bench &
PID0=$!

CUDA_VISIBLE_DEVICES=1 python scripts/run_experiment.py \
    --config configs/hf_llama3_8b.py --name llama3-8b-bench &
PID1=$!

wait $PID0 $PID1
echo "Both experiments complete."
```

### Option C: Ensemble with parallel GPUs

The test notebook (`notebooks/test_local_hf_pipeline.ipynb`, Step 8b) includes
an example of parallel ensemble voting using `ProcessPoolExecutor` with
per-process `CUDA_VISIBLE_DEVICES` assignment.

### Important notes for multi-GPU

- Each process loads its own copy of the model — so you need enough VRAM on
  each GPU for the model you assign to it.
- Don't run two experiments on the same GPU unless the models are small enough
  to share VRAM.
- Check GPU usage: `nvidia-smi` or `watch -n 1 nvidia-smi`.
- If a model doesn't fit, try a smaller one (e.g., `hf_qwen1_5b.py` or
  `hf_phi3_mini.py`) or use `hf_dtype = "fp16"` / quantization.

---

## 7) Adding a new model

### Step 1: Create a config file

Copy an existing config and modify:

```bash
cp configs/hf_qwen7b.py configs/hf_newmodel.py
```

Edit `configs/hf_newmodel.py`:

```python
# LLM settings
llm_provider = "hf_local"
hf_model_id = "organization/Model-Name-Instruct"  # HuggingFace model ID
hf_dtype = "bf16"        # or "fp16", "auto"
hf_max_new_tokens = 512
hf_temperature = 0.2

# Adjust concurrency based on model size
max_concurrent = 2  # lower for bigger models
```

### Step 2: Register in the benchmark runner (optional)

If you want `run_full_benchmark.py` to include it, add to
`scripts/run_full_benchmark.py`:

```python
HF_LOCAL_MODELS = {
    ...
    "hf_newmodel": "newmodel-bench",  # key = config filename without .py
}
```

### Step 3: Register model size for plots (optional)

If you want `plot_model_size.py` to include the model in size-based plots,
add to the `MODEL_SIZES` dict in `scripts/plot_model_size.py`:

```python
MODEL_SIZES = {
    ...
    "newmodel": ("New Model 13B", 13, False, "Open-source, confirmed 13B"),
    #            display name    size_B  estimated?   notes
}
```

The key should be a substring that matches the model ID.

### Step 4: Test it

```bash
# Quick smoke test
python scripts/run_experiment.py --config configs/hf_newmodel.py --name newmodel-smoke

# Check the output
cat artifacts/experiments/newmodel-smoke/summary.json | python -m json.tool
```

### Adding a non-HF model (API-based)

For OpenRouter, OpenAI, or Bedrock models, set `llm_provider` accordingly
in the config:

```python
# OpenRouter example
llm_provider = "openrouter"
model = "meta-llama/llama-4-scout"
max_concurrent = 10

# Embedding: can still use local HF embeddings with an API LLM
embedding_model = "hf_local"
embedding_model_id = "BAAI/bge-base-en-v1.5"
```

---

## 8) WattBot scoring explained

The WattBot competition score is a weighted combination:

| Component        | Weight | What it measures                                    |
|------------------|--------|-----------------------------------------------------|
| `value_accuracy` | 75%    | Exact match on `answer_value` (±0.1% for numerics)  |
| `ref_overlap`    | 15%    | Jaccard overlap between predicted and GT `ref_id`   |
| `na_accuracy`    | 10%    | Correctly outputting `is_blank` for unanswerable Qs |

**Overall = 0.75 × value_accuracy + 0.15 × ref_overlap + 0.10 × na_accuracy**

The scoring logic lives in `scripts/score.py` and is provider-agnostic — it
only looks at the CSV columns, not how they were generated.

---

## 9) Directory structure reference

```
KohakuRAG_UI/
├── configs/                  # Experiment configs (one per model)
│   ├── hf_qwen7b.py
│   ├── hf_qwen1_5b.py
│   └── ...
├── scripts/                  # Benchmarking & analysis tools
│   ├── run_experiment.py     # Run one experiment
│   ├── run_full_benchmark.py # Run all models
│   ├── run_wattbot_eval.py   # Quick eval + score
│   ├── run_ensemble.py       # Combine multiple runs
│   ├── score.py              # WattBot scoring
│   ├── generate_results_matrix.py
│   ├── audit_experiments.py
│   ├── plot_model_size.py
│   ├── plot_from_matrix.py
│   └── plot_score_breakdown.py
├── artifacts/                # Output (gitignored)
│   ├── experiments/          # Per-experiment results
│   │   ├── qwen7b-v1/
│   │   │   ├── submission.csv
│   │   │   ├── results.json
│   │   │   └── summary.json
│   │   └── ...
│   ├── plots/                # Generated charts
│   └── results_matrix.csv
├── data/
│   └── train_QA.csv          # Ground truth questions
├── notebooks/
│   └── test_local_hf_pipeline.ipynb
└── vendor/KohakuRAG/         # Core RAG library
    └── src/kohakurag/
```
