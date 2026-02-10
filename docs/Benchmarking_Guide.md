# Benchmarking Guide

How to run experiments, compare models, use multiple GPUs, and add new models
to the WattBot RAG evaluation pipeline.

All commands assume you are at the **repo root** with your venv active.

---

## 0) Prerequisites — Build the vector index

Before running any experiments, you need a vector database. The config files
reference `data/embeddings/wattbot_jinav4.db`, which is built locally by the
indexing pipeline. The `.db` files are **gitignored** (>100MB) so each machine
must build its own index from the tracked source data.

### Data prerequisites

You need two files in `data/`:
- `data/train_QA.csv` — ground-truth question set (should already be in the repo)
- `data/metadata.csv` — document bibliography with `id`, `title`, `url` columns
  (see `vendor/KohakuRAG/docs/wattbot.md` for details)

### Install dependencies

```bash
uv pip install -r local_requirements.txt  # includes kohaku-engine, httpx, etc.
```

### Build the index

```bash
cd vendor/KohakuRAG
kogine run scripts/wattbot_build_index.py --config configs/jinav4/index.py
cd ../..

# Verify the database was created
ls -lh data/embeddings/wattbot_jinav4.db
```

The build script will:
1. Check for structured JSON docs in `data/corpus/`
2. If missing, **automatically download PDFs** using URLs from `data/metadata.csv`
3. Parse PDFs into structured JSON (saved to `data/corpus/`)
4. Build the vector index from those documents

Downloaded PDFs are cached in `data/pdfs/` so subsequent runs skip
already-fetched files.

---

## 1) Running a single experiment

Every experiment is driven by a **config file** in `vendor/KohakuRAG/configs/`.
Each config specifies the LLM provider, model, embedding settings, and retrieval
parameters.

**Always pass `--env` to tag which machine you're running on.** This is saved in
`summary.json` and comparison CSVs so you can distinguish results across machines.

```bash
# Run with Qwen 2.5 7B on the GB10. Provide custom name for experimnet logging
python scripts/run_experiment.py \
    --config vendor/KohakuRAG/configs/hf_qwen7b.py \
    --env GB10
    --name qwen7b-v1

# Same model on the PowerEdge
python scripts/run_experiment.py \
    --config vendor/KohakuRAG/configs/hf_qwen7b.py \
    --env PowerEdge \
    --name qwen7b-v1
```

**Precision flag:** All local HF models default to `--precision 4bit` (NF4
quantization via `bitsandbytes`). Override with `--precision bf16`, `fp16`, or
`auto`. Precision is saved to `summary.json` as the `quantization` field.

### Using the test dataset

The competition test set (`test_solutions.csv`) is **not** stored in the repo.
Upload it to `data/test_solutions.csv` on the machine, then use `--questions`:

```bash
# Run on test set with any config
python scripts/run_experiment.py \
    --config vendor/KohakuRAG/configs/hf_qwen7b.py \
    --questions data/test_solutions.csv \
    --name qwen7b-test --env PowerEdge
```

The CSV must have the same columns as `train_QA.csv` (`id`, `question`,
`answer_value`, `answer_unit`, `ref_id`, etc.). The `--questions` flag
overrides the `questions` path in the config file.

`data/test_solutions.csv` is gitignored so it will never be committed.

When `--env` is set, output is organized by environment:

```
artifacts/experiments/PowerEdge/qwen7b-v1/
├── submission.csv   # Kaggle-format predictions
├── results.json     # Per-question details (latency, scores, raw LLM output)
└── summary.json     # Aggregate metrics (overall score, timing, dataset info)
```

Without `--env`, results go directly under `artifacts/experiments/<name>/`.

### Quick eval (alternative)

`run_wattbot_eval.py` is a lighter-weight script that runs the pipeline and
prints the WattBot score directly:

```bash
# Default: hf_local with Qwen 7B
python scripts/run_wattbot_eval.py

# With a config file
python scripts/run_wattbot_eval.py --config vendor/KohakuRAG/configs/hf_qwen7b.py

# Override provider/model via CLI (no config file needed)
python scripts/run_wattbot_eval.py --provider hf_local --hf-model Qwen/Qwen2.5-1.5B-Instruct
```

---

## 2) Available configs

All local models default to **4-bit NF4 quantization** (`--precision 4bit`).
VRAM estimates below reflect the 4-bit default. Pass `--precision bf16` for
full precision (roughly 4× more VRAM).

| Config file              | Model                        | Params | VRAM (4-bit) | Provider  |
|--------------------------|------------------------------|--------|--------------|-----------|
| `hf_qwen1_5b.py`        | Qwen 2.5 1.5B Instruct      | 1.5B   | ~2 GB        | hf_local  |
| `hf_qwen3b.py`          | Qwen 2.5 3B Instruct        | 3B     | ~3 GB        | hf_local  |
| `hf_qwen7b.py`          | Qwen 2.5 7B Instruct        | 7B     | ~6 GB        | hf_local  |
| `hf_qwen14b.py`         | Qwen 2.5 14B Instruct       | 14B    | ~10 GB       | hf_local  |
| `hf_qwen32b.py`         | Qwen 2.5 32B Instruct       | 32B    | ~20 GB       | hf_local  |
| `hf_qwen72b.py`         | Qwen 2.5 72B Instruct       | 72B    | ~40 GB       | hf_local  |
| `hf_llama3_8b.py`       | Llama 3.1 8B Instruct       | 8B     | ~6 GB        | hf_local  |
| `hf_gemma2_9b.py`       | Gemma 2 9B Instruct         | 9B     | ~7 GB        | hf_local  |
| `hf_gemma2_27b.py`      | Gemma 2 27B Instruct        | 27B    | ~17 GB       | hf_local  |
| `hf_mixtral_8x7b.py`    | Mixtral 8x7B Instruct (MoE) | 46.7B  | ~26 GB       | hf_local  |
| `hf_mistral7b.py`       | Mistral 7B Instruct v0.3    | 7B     | ~6 GB        | hf_local  |
| `hf_phi3_mini.py`       | Phi-3.5 Mini (3.8B)         | 3.8B   | ~3 GB        | hf_local  |

Bedrock configs (from the `bedrock` branch) also work if you have
`llm_bedrock.py` and AWS credentials set up.

---

## 3) Running all models (full benchmark)

**Always pass `--env`** so every sub-experiment is tagged with the machine name.

```bash
# Smoke test first (1 question per model, catches config/loading errors fast)
python scripts/run_full_benchmark.py --smoke-test --provider hf_local --env GB10

# Full benchmark — all local HF models on the PowerEdge
python scripts/run_full_benchmark.py --provider hf_local --env PowerEdge

# Full benchmark with test dataset
python scripts/run_full_benchmark.py --provider hf_local --env PowerEdge \
    --questions data/test_solutions.csv

# Single model only
python scripts/run_full_benchmark.py --model qwen7b --env GB10

# All providers (local + bedrock, if bedrock configs exist)
python scripts/run_full_benchmark.py --env PowerEdge

# Run all models in bf16 instead of default 4-bit
python scripts/run_full_benchmark.py --provider hf_local --precision bf16 --env PowerEdge
```

The benchmark runner:
- **Skips models that already have results** (`summary.json` exists for that
  experiment name + env). Use `--force` to re-run everything.
- Skips models whose config files don't exist
- Runs each model as a subprocess with a 30-minute timeout
- Prints a pass/fail/skip summary at the end

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
CUDA_VISIBLE_DEVICES=0 python scripts/run_experiment.py \
    --config vendor/KohakuRAG/configs/hf_qwen7b.py --name qwen7b-gpu0 --env PowerEdge

# Terminal 2 — GPU 1
CUDA_VISIBLE_DEVICES=1 python scripts/run_experiment.py \
    --config vendor/KohakuRAG/configs/hf_llama3_8b.py --name llama3-8b-gpu1 --env PowerEdge
```

### Option B: Script it

```bash
#!/bin/bash
# run_parallel.sh — run two experiments on separate GPUs
ENV="PowerEdge"

CUDA_VISIBLE_DEVICES=0 python scripts/run_experiment.py \
    --config vendor/KohakuRAG/configs/hf_qwen7b.py --name qwen7b-bench --env $ENV &
PID0=$!

CUDA_VISIBLE_DEVICES=1 python scripts/run_experiment.py \
    --config vendor/KohakuRAG/configs/hf_llama3_8b.py --name llama3-8b-bench --env $ENV &
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
  `hf_phi3_mini.py`). Models already default to 4-bit quantization;
  use `--precision bf16` only if you have enough VRAM.

---

## 7) Adding a new model

Follow these steps whenever you want to add a new model (local HF or API).

### Step 1: Create a config file

Copy the closest existing config and modify it. All configs live in
`vendor/KohakuRAG/configs/`.

```bash
# Example: adding a new 13B model
cp vendor/KohakuRAG/configs/hf_qwen7b.py vendor/KohakuRAG/configs/hf_newmodel13b.py
```

Edit `vendor/KohakuRAG/configs/hf_newmodel13b.py`:

```python
"""
WattBot Evaluation Config - New Model 13B Instruct (Local HF)

Brief description: what the model is, VRAM requirements, any special notes.
"""

# Database settings (keep these the same)
db = "../../data/embeddings/wattbot_jinav4.db"
table_prefix = "wattbot_jv4"

# Input/output
questions = "../../data/train_QA.csv"
output = "../../artifacts/submission_newmodel13b.csv"  # unique output path
metadata = "../../data/metadata.csv"

# LLM settings
llm_provider = "hf_local"
hf_model_id = "organization/Model-13B-Instruct"  # HuggingFace model ID
hf_max_new_tokens = 512
hf_temperature = 0.2

# Embedding settings (Jina v4 — must match index, keep these the same)
embedding_model = "jinav4"
embedding_dim = 1024
embedding_task = "retrieval"

# Retrieval settings (keep these the same for fair comparison)
top_k = 8
planner_max_queries = 3
deduplicate_retrieval = True
rerank_strategy = "combined"
top_k_final = 10

# Unanswerable detection
retrieval_threshold = 0.25

# Other settings
max_retries = 2
max_concurrent = 2  # lower (1) for large models, higher (3-4) for small ones
```

**Key things to change:**
- `output` — give it a unique filename so submissions don't overwrite each other
- `hf_model_id` — the HuggingFace model identifier
- `max_concurrent` — lower for larger models (GPU memory), higher for smaller ones

**Note:** Precision is controlled at runtime via `--precision` (default: `4bit`),
not in the config file. Config files define model identity only.

**Important:** Keep the `../../` path prefix on `db`, `questions`, `output`, and
`metadata`. These paths are relative to `vendor/KohakuRAG/` (where `kogine` runs
from), and the experiment scripts strip the prefix automatically.

### Step 2: Test with a smoke run

```bash
# Single-question smoke test
python scripts/run_experiment.py \
    --config vendor/KohakuRAG/configs/hf_newmodel13b.py \
    --name newmodel13b-smoke --env GB10

# Check the output
python -m json.tool artifacts/experiments/newmodel13b-smoke/summary.json
```

If this passes, the model loads correctly and can answer questions.

### Step 3: Register in the benchmark runner

To include the model in `run_full_benchmark.py`, add an entry to the
`HF_LOCAL_MODELS` dict in `scripts/run_full_benchmark.py`:

```python
HF_LOCAL_MODELS = {
    ...
    "hf_newmodel13b": "newmodel13b-bench",  # key = config filename (no .py)
}
```

Then verify:
```bash
# Check it shows up
python scripts/run_full_benchmark.py --model newmodel13b --smoke-test
```

### Step 4: Register in the scaling experiment (Qwen family only)

If adding a new Qwen model size, also add to the `QWEN_MODELS` dict in
`scripts/run_qwen_scaling.py`:

```python
QWEN_MODELS = {
    ...
    "13": {
        "config": "vendor/KohakuRAG/configs/hf_qwen13b.py",
        "name": "qwen13b",
        "model_id": "Qwen/Qwen2.5-13B-Instruct",
        "params_b": 13,
        "approx_vram_gb": 10,  # approximate 4-bit VRAM
    },
}
```

### Step 5: Register model size for plots (optional)

To include the model in size-based plots, add to the `MODEL_SIZES` dict in
`scripts/plot_model_size.py`:

```python
MODEL_SIZES = {
    ...
    "newmodel13b": ("New Model 13B", 13, False, "Open-source, confirmed 13B"),
    #               display name    size_B  estimated?   notes
}
```

The key should be a substring that matches the model ID in experiment results.

### Step 6: Update this guide

Add the new config to the table in Section 2 so others know it's available.

### Adding a non-HF model (API-based)

For OpenRouter, OpenAI, or Bedrock models, set `llm_provider` accordingly:

```python
# OpenRouter example
llm_provider = "openrouter"
model = "meta-llama/llama-4-scout"
max_concurrent = 10

# Embedding: must match the index (jinav4 with 1024 dims)
embedding_model = "jinav4"
embedding_dim = 1024
embedding_task = "retrieval"
```

For Bedrock, ensure `llm_bedrock.py` is on the Python path and AWS credentials
are configured (see the `bedrock` branch for examples).

---

## 8) WattBot scoring explained

The WattBot competition score is a weighted combination:

| Component        | Weight | What it measures                                    |
|------------------|--------|-----------------------------------------------------|
| `value_accuracy` | 75%    | Exact match on `answer_value` (±0.1% for numerics)  |
| `ref_overlap`    | 15%    | Jaccard overlap between predicted and GT `ref_id`   |
| `na_recall`      | 10%    | **Recall** over truly-NA questions: of all ground-truth unanswerable questions, what fraction did the model correctly mark as `is_blank`? |

**Overall = 0.75 × value_accuracy + 0.15 × ref_overlap + 0.10 × na_recall**

Why recall instead of accuracy? Most questions are answerable, so a naive
accuracy metric (correct NAs + all non-NA questions) / total is dominated by
the non-NA majority — a model that never abstains still scores ~0.96. Recall
isolates the signal. False NAs (answering `is_blank` on answerable questions)
are already penalized by the value_accuracy component.

The scoring logic lives in `scripts/score.py` and is provider-agnostic — it
only looks at the CSV columns, not how they were generated.

---

## 9) Hardware metrics & run environment

Every experiment collects hardware metrics and machine identification
automatically. Use `--env` to add an explicit label for cross-machine comparison.

### Machine identification

| Field                 | How it's collected                                     | Example                  |
|-----------------------|--------------------------------------------------------|--------------------------|
| **`run_environment`** | `--env` flag (user-specified)                          | `"GB10"`, `"PowerEdge"`  |
| **`hostname`**        | `socket.gethostname()` (auto)                          | `"endemann-gb10"`        |
| **`cpu_model`**       | `/proc/cpuinfo` (auto)                                 | `"ARMv8 Processor"`      |
| **`gpu_count`**       | `torch.cuda.device_count()` (auto)                     | `2`                      |
| **`gpu_name`**        | `torch.cuda.get_device_properties()` (auto)            | `"NVIDIA RTX A6000"`     |
| **`os_platform`**     | `platform.system()-release-machine` (auto)             | `"Linux-5.15.0-x86_64"`  |

Hostname, CPU, and GPU count are auto-detected — but **`--env` is the primary
way to label runs** for later comparison (e.g., filtering a CSV by `run_environment`).

### Hardware performance metrics

| Metric                | How it's measured                                      |
|-----------------------|--------------------------------------------------------|
| **VRAM (peak)**       | `torch.cuda.max_memory_allocated()` after experiment   |
| **Model disk size**   | Total size of HuggingFace cache dir for the model      |
| **Model load time**   | Wall-clock time to load LLM + embedder (split)         |
| **Energy (Wh)**       | NVML hardware counter (preferred) or trapezoidal integration of `nvidia-smi` power readings |
| **Avg/Peak power (W)**| From 1-second interval `nvidia-smi` polling            |
| **CPU RSS peak**      | Peak resident set size via `psutil` background polling |

These are saved in `summary.json`:

```json
{
  "run_environment": "PowerEdge",
  "hardware": {
    "hostname": "endemann-poweredge",
    "cpu_model": "Intel Xeon w5-2465X",
    "gpu_count": 2,
    "gpu_name": "NVIDIA RTX A6000",
    "os_platform": "Linux-5.15.0-x86_64",
    "gpu_vram_allocated_gb": 14.23,
    "gpu_vram_total_gb": 48.0,
    "model_disk_size_gb": 13.47,
    "gpu_energy_wh": 2.541,
    "gpu_energy_method": "nvml",
    "gpu_avg_power_watts": 185.3,
    "llm_load_time_seconds": 10.2,
    "embedder_load_time_seconds": 2.2,
    "cpu_rss_peak_gb": 8.42
  }
}
```

For API models (Bedrock/OpenRouter), VRAM and power metrics will be zero but
API cost tracking (token counts + estimated USD) is still recorded. You should
still pass `--env` for Bedrock runs (e.g., `--env Bedrock-us-east-1`).

---

## 10) Qwen model-size scaling experiment

To measure how WattBot performance scales with model size (keeping
everything else constant), there's a dedicated script:

```bash
# Run all Qwen sizes that fit in your GPU (always tag the machine!)
python scripts/run_qwen_scaling.py --env GB10

# Run specific sizes only
python scripts/run_qwen_scaling.py --sizes 1.5 3 7 --env PowerEdge

# Skip sizes already run (resume after crash)
python scripts/run_qwen_scaling.py --skip-existing --env GB10

# See what would run without running
python scripts/run_qwen_scaling.py --dry-run

# Run in bf16 (full precision) instead of default 4-bit
python scripts/run_qwen_scaling.py --precision bf16 --env PowerEdge
```

Available Qwen sizes (4-bit VRAM): 1.5B (~2GB), 3B (~3GB), 7B (~6GB), 14B (~10GB),
32B (~20GB), 72B (~40GB). Pass `--precision bf16` for full precision (roughly 4× more).

The script:
- Runs models **sequentially** (frees GPU between runs for clean measurements)
- Auto-skips models that won't fit in available VRAM
- Produces per-model results + a combined comparison table

Output:
```
artifacts/experiments/qwen-scaling/
├── qwen1.5b/submission.csv, results.json, summary.json
├── qwen3b/...
├── qwen7b/...
├── scaling_comparison.csv    ← side-by-side comparison
└── scaling_comparison.json   ← same data, for notebooks
```

The `scaling_comparison.csv` has columns for scores, latency, VRAM, disk
size, energy (Wh), and power — ready for plotting.

---

## 11) Results output for post-processing

Every experiment produces three files for iteration:

| File               | Format | What's in it                                      |
|--------------------|--------|---------------------------------------------------|
| `submission.csv`   | CSV    | Kaggle-format predictions (id, answer_value, ...) |
| `results.json`     | JSON   | Per-question: GT, prediction, scores, latency     |
| `summary.json`     | JSON   | Aggregate: overall score, hardware metrics, config |

To iterate on post-processing:

```python
import json, pandas as pd

# Load per-question results
with open("artifacts/experiments/qwen7b-v1/results.json") as f:
    results = json.load(f)

# Convert to DataFrame for analysis
df = pd.DataFrame(results)
print(df[["id", "pred_value", "gt_value", "value_correct", "latency_seconds"]])

# Load summary (includes hardware)
with open("artifacts/experiments/qwen7b-v1/summary.json") as f:
    summary = json.load(f)
print(f"VRAM: {summary['hardware']['gpu_vram_allocated_gb']} GB")
print(f"Energy: {summary['hardware']['gpu_energy_wh']} Wh")
```

For the scaling experiment, use the combined file:

```python
scaling = pd.read_csv("artifacts/experiments/qwen-scaling/scaling_comparison.csv")
print(scaling[["model", "params_b", "overall_score", "vram_allocated_gb", "energy_wh"]])
```

---

## 12) Directory structure reference

```
KohakuRAG_UI/
├── data/                     # Tracked in git (source data shared across machines)
│   ├── train_QA.csv          # Ground truth questions (training set)
│   ├── test_solutions.csv    # Competition test set (gitignored, upload manually)
│   ├── metadata.csv          # Document bibliography
│   ├── embeddings/           # Vector databases — gitignored, built locally (>100MB)
│   │   └── wattbot_jinav4.db # (built by kogine, not in git)
│   ├── pdfs/                 # Downloaded source PDFs
│   └── corpus/               # Parsed JSON documents
├── scripts/                  # Benchmarking & analysis tools
│   ├── hardware_metrics.py   # VRAM, disk, energy, CPU RSS, machine ID
│   ├── run_experiment.py     # Run one experiment (--env for machine label)
│   ├── run_qwen_scaling.py   # Qwen size scaling experiment
│   ├── run_full_benchmark.py # Run all models
│   ├── run_wattbot_eval.py   # Quick eval + score
│   ├── run_ensemble.py       # Combine multiple runs
│   ├── score.py              # WattBot scoring
│   ├── generate_results_matrix.py
│   ├── audit_experiments.py
│   ├── plot_model_size.py
│   ├── plot_from_matrix.py
│   └── plot_score_breakdown.py
├── artifacts/                # Experiment outputs (gitignored, machine-specific)
│   ├── experiments/          # Per-experiment results, organized by --env
│   │   ├── PowerEdge/       # Results from PowerEdge runs
│   │   │   ├── qwen7b-v1/
│   │   │   │   ├── submission.csv
│   │   │   │   ├── results.json
│   │   │   │   └── summary.json
│   │   │   └── ...
│   │   └── GB10/            # Results from GB10 runs
│   ├── plots/                # Generated charts
│   └── results_matrix.csv
├── notebooks/
│   └── test_local_hf_pipeline.ipynb
└── vendor/KohakuRAG/         # Core RAG library
    ├── configs/              # All configs (experiment + indexing)
    │   ├── hf_qwen7b.py     # Experiment configs (hf_*.py)
    │   ├── hf_qwen1_5b.py
    │   ├── hf_qwen3b.py
    │   ├── hf_qwen14b.py
    │   ├── hf_qwen32b.py
    │   ├── hf_qwen72b.py
    │   ├── hf_llama3_8b.py
    │   ├── hf_gemma2_9b.py
    │   ├── hf_gemma2_27b.py
    │   ├── hf_mixtral_8x7b.py
    │   ├── hf_mistral7b.py
    │   ├── hf_phi3_mini.py
    │   ├── jinav4/index.py   # Indexing configs
    │   ├── text_only/
    │   └── with_images/
    ├── scripts/              # Indexing scripts
    └── src/kohakurag/        # Core library
```
