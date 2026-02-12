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

### Build the index (in not completed yet)

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

**Retry & prompt ordering:** Pipeline config keys that control robustness
(matching KohakuRAG competition settings):

| Config key              | Default | Effect |
|-------------------------|---------|--------|
| `max_retries`           | `2`     | If the LLM answer is blank, re-run retrieval with increasing `top_k` (iterative deepening: 2x, 3x, ...) |
| `use_reordered_prompt`  | `True`  | Context is placed **before** the question (C→Q ordering) to combat the "lost in the middle" effect (~80% relative improvement) |
| `planner_max_queries`   | `4`     | Number of diverse retrieval queries generated per question by the LLM query planner |

All are set in every config file (`hf_*.py` and `bedrock_*.py`).

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

Results are organized by environment and datafile:

```
artifacts/experiments/PowerEdge/train_QA/qwen7b-v1/
├── results.json     # Per-question details (raw LLM output, latency, scores)
├── summary.json     # Aggregate metrics (overall score, timing, dataset info)
└── submission.csv   # Normalised Kaggle-format predictions (created by posthoc.py)
```

**Important:** `results.json` stores **raw model output** (un-normalised).
To produce a normalised `submission.csv` for Kaggle and get the final score,
run the post-hoc processing step:

```bash
python scripts/posthoc.py artifacts/experiments/PowerEdge/train_QA/qwen7b-v1/
```

This applies answer normalisation (comma stripping, range formatting,
abbreviation expansion, etc.) and re-scores against ground truth.
See `scripts/posthoc.py` for details.

The `<datafile>` subfolder is derived from the questions CSV filename
(e.g. `train_QA` from `data/train_QA.csv`, `test_solutions` from
`data/test_solutions.csv`). Without `--env`, results go under
`artifacts/experiments/<datafile>/<name>/`.

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

### AWS Bedrock models (API — no GPU required)

Requires `boto3` and AWS credentials. See [Setup_Bedrock.md](Setup_Bedrock.md) for
full setup instructions.

| Config file                 | Model                   | Cost (per 1M tok) | Provider |
|-----------------------------|-------------------------|--------------------|----------|
| `bedrock_claude_haiku.py`   | Claude 3 Haiku          | $0.25 in / $1.25 out | bedrock |
| `bedrock_claude_sonnet.py`  | Claude 3.5 Sonnet v2    | $3.00 in / $15.00 out | bedrock |
| `bedrock_nova_pro.py`       | Amazon Nova Pro         | $0.80 in / $3.20 out | bedrock |
| `bedrock_llama4_scout.py`   | Meta Llama 4 Scout 17B  | $0.17 in / $0.17 out | bedrock |

All configs (local and bedrock) use **identical retrieval settings** (top_k, planner_max_queries,
rerank_strategy, etc.) so results are directly comparable.

## 3) Running all models (full benchmark)

**Always pass `--env`** so every sub-experiment is tagged with the machine name.

```bash
# --- Local HF models ---
python scripts/run_full_benchmark.py --provider hf_local --env PowerEdge \
    --questions data/train_QA.csv

# --- AWS Bedrock models ---
python scripts/run_full_benchmark.py --provider bedrock --env Bedrock \
    --questions data/train_QA.csv

# --- Both providers together ---
python scripts/run_full_benchmark.py --env PowerEdge

# Full benchmark with test dataset (local HF models on the PowerEdge)
python scripts/run_full_benchmark.py --provider hf_local --env PowerEdge \
    --questions data/test_solutions.csv

# Full benchmark with train dataset (local HF models on the PowerEdge)
python scripts/run_full_benchmark.py --provider hf_local --env PowerEdge \
    --questions data/train_QA.csv

# Single model only
python scripts/run_full_benchmark.py --model qwen7b --env GB10
python scripts/run_full_benchmark.py --model claude_haiku --env Bedrock

# Run local models in bf16 instead of default 4-bit
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

### Post-hoc normalisation and scoring

After an experiment completes, run post-hoc processing to normalise raw
model output and produce a Kaggle-ready `submission.csv`:

```bash
# Process a single experiment (auto-finds results.json)
python scripts/posthoc.py artifacts/experiments/PowerEdge/train_QA/qwen7b-v1/

# Dry-run: see the score without writing files
python scripts/posthoc.py artifacts/experiments/PowerEdge/train_QA/qwen7b-v1/ --dry-run
```

### Score a submission against ground truth

```bash
python scripts/score.py data/train_QA.csv artifacts/experiments/train_QA/qwen7b-v1/submission.csv
```

### Generate a side-by-side comparison matrix

```bash
# Auto-discovers all experiments (both train_QA and test_solutions)
python scripts/generate_results_matrix.py 

# Only train_QA experiments
python scripts/generate_results_matrix.py --datafile train_QA

# Only test_solutions experiments
python scripts/generate_results_matrix.py --datafile test_solutions

# Or specify files manually
python scripts/generate_results_matrix.py \
    --submissions artifacts/experiments/*/submission.csv \
    --output artifacts/results_matrix.csv
```

This produces a CSV where each row is a question and columns show each model's
prediction + correctness, making it easy to spot which questions each model
gets right or wrong.

### Ensemble voting

The ensemble runner supports two modes:

**Mode 1 — Same-model ensemble** (KohakuRAG competition strategy): Run *m*
independent inference passes of the **same model** and aggregate via voting.
This was the only strategy that remained #1 on both public and private
leaderboard partitions.

```bash
# 5 independent runs of qwen7b, answer_priority voting (default)
python scripts/run_ensemble.py \
    --config vendor/KohakuRAG/configs/hf_qwen7b.py \
    --num-runs 5 --name qwen7b-ens5 --env GB10

# 9 runs (closer to competition setup), majority voting
python scripts/run_ensemble.py \
    --config vendor/KohakuRAG/configs/hf_qwen7b.py \
    --num-runs 9 --strategy majority --name qwen7b-ens9 --env PowerEdge
```

Each run is executed as a separate subprocess for completely independent model
state.  LLM sampling temperature introduces per-run diversity.

**Mode 2 — Cross-model ensemble**: Aggregate results from previously completed
experiments (different models).

```bash
python scripts/run_ensemble.py \
    --experiments qwen7b-v1 llama3-8b-v1 mistral7b-v1 \
    --name ensemble-3way --env PowerEdge

# Specify datafile explicitly (auto-detected from source experiments by default)
python scripts/run_ensemble.py \
    --experiments qwen7b-v1 llama3-8b-v1 \
    --name ensemble-test --env PowerEdge --datafile test_solutions
```

**Aggregation strategies:**

| Strategy          | Description |
|-------------------|-------------|
| `answer_priority` | (Default) Vote on answer first, then collect refs only from matching runs — ensures citation consistency |
| `majority`        | Most common answer wins; refs scoped to winning-answer runs only (functionally equivalent to `answer_priority`) |
| `first_non_blank` | First non-blank answer wins; union of all refs (including from disagreeing runs) |

**Abstention-aware voting** (`--ignore-blank`): If any run produces a non-blank
answer, blank ("is_blank") runs are filtered out before voting.  Enabled by
default for same-model ensembles.

### Recommended ensemble: Top-3 majority vote

Based on test_solutions benchmarking (n=282), the recommended production
ensemble is **Qwen 2.5 72B + Qwen 2.5 32B + Qwen 2.5 14B** using majority
voting. This combination was selected for complementary strengths:

| Model          | WattBot Score | NA Recall | Unique Wins | Latency | VRAM (4-bit) |
|----------------|:---:|:---:|:---:|:---:|:---:|
| Qwen 2.5 72B  | 0.752 | 0.938 | 0 | 15.7s | ~33 GB |
| Qwen 2.5 32B  | 0.710 | **1.000** | 2 | **8.4s** | ~22 GB |
| Qwen 2.5 14B  | 0.660 | 0.875 | **4** | 16.0s | ~15 GB |

**Why these three:**

- **72B** is the top scorer overall (highest value accuracy and ref overlap).
- **32B** has perfect NA recall (1.0) — it never misclassifies an
  unanswerable question, acting as a safety anchor in the vote.
- **14B** has the most unique wins (4 questions only it gets right) and
  the lowest agreement with the other two (~0.82), providing the diversity
  that makes ensembling worthwhile.
- All three are Qwen 2.5 family at 4-bit, so inference infrastructure is
  uniform. Combined VRAM is ~70 GB (fits sequentially on a single 96 GB GPU).

**Why not Qwen3 30B-A3B?** Despite ranking #2 individually (0.724), it is
9x slower than 32B (76s vs 8.4s per question), uses 2x more energy
(297 Wh vs 131 Wh), has worse NA recall (0.81), and only 1 unique win.
Its agreement with 72B (0.91) is the same as 32B's, so it adds no extra
diversity.

```bash
# Run the recommended ensemble on test_solutions
python scripts/run_ensemble.py \
    --experiments qwen72b-bench qwen32b-bench qwen14b-bench \
    --name ensemble-top3-majority \
    --strategy majority --ignore-blank \
    --env PowerEdge \
    --datafile test_solutions

# Same ensemble on train_QA
python scripts/run_ensemble.py \
    --experiments qwen72b-bench qwen32b-bench qwen14b-bench \
    --name ensemble-top3-majority \
    --strategy majority --ignore-blank \
    --env PowerEdge \
    --datafile train_QA
```

**Always use `--ignore-blank`** with this ensemble. Without it, refusal
answers from 14B (27% refusal rate) and 32B (21%) can outvote 72B's correct
answers via majority rule. The flag filters out `is_blank` votes before
counting, so a real answer always beats a refusal. Refs are also scoped to
only the models that voted for the winning answer, preventing spurious
references from inflating the predicted set.

The individual experiments (`qwen72b-bench`, `qwen32b-bench`, `qwen14b-bench`)
must already exist under `artifacts/experiments/<env>/<datafile>/`. Run them
first with `run_full_benchmark.py` or `run_experiment.py` if they don't.

### Recommended ensemble: Top-5 majority vote

For environments with more VRAM headroom or when maximum accuracy matters more
than latency, extending the ensemble to **5 models** adds two high-performing
architectures while keeping majority-vote semantics (3-of-5 agreement).

| Model              | WattBot Score | NA Recall | Unique Wins | Latency | VRAM (4-bit) |
|--------------------|:---:|:---:|:---:|:---:|:---:|
| Qwen 2.5 72B      | 0.752 | 0.938 | 0 | 15.7s | ~33 GB |
| Qwen3 30B-A3B     | 0.724 | 0.813 | 1 | 76.3s | ~18 GB |
| Qwen 2.5 32B      | 0.710 | **1.000** | 2 | **8.4s** | ~22 GB |
| Qwen 2.5 14B      | 0.660 | 0.875 | **4** | 16.0s | ~15 GB |
| Mixtral 8x22B      | 0.643 | 0.875 | 0 | 17.3s | ~80 GB |

**Why these five:**

- The **top-3 core** (72B + 32B + 14B) is unchanged; see rationale above.
- **Qwen3 30B-A3B** is the #2 individual scorer (0.724) and a MoE
  architecture, so its error profile differs from the dense Qwen 2.5 models.
  The latency penalty is acceptable here because each model only runs once in
  a cross-model ensemble.
- **Mixtral 8x22B** (0.643) adds a completely different model family.
  Architectural diversity is the primary benefit — it breaks Qwen-family
  correlated errors. Its agreement with 72B is lower than any other Qwen
  model, maximising the value of a fifth vote.
- Combined VRAM is ~168 GB (sequential loading) or can be distributed across
  multiple GPUs. The 3-of-5 majority threshold means any two models can be
  wrong on a question and the ensemble still answers correctly.

```bash
# Run the top-5 ensemble on test_solutions
python scripts/run_ensemble.py \
    --experiments qwen72b-bench qwen3-30b-a3b-bench qwen32b-bench qwen14b-bench mixtral-8x22b-bench \
    --name ensemble-top5-majority \
    --strategy majority --ignore-blank \
    --env PowerEdge \
    --datafile test_solutions

python scripts/run_ensemble.py \
    --experiments qwen72b-bench qwen32b-bench qwen14b-bench qwen7b-bench \
    --name ensemble-top5-majority \
    --strategy majority --ignore-blank \
    --env PowerEdge \
    --datafile test_solutions


# Same ensemble on train_QA
python scripts/run_ensemble.py \
    --experiments qwen72b-bench qwen3-30b-a3b-bench qwen32b-bench qwen14b-bench mixtral-8x22b-bench \
    --name ensemble-top5-majority \
    --strategy majority --ignore-blank \
    --env PowerEdge \
    --datafile train_QA
```

The same `--ignore-blank` rationale from the top-3 ensemble applies: blank
votes from lower-scoring models must not outvote correct answers.

### Kitchen-sink ensemble: All 11 models

For a full-diversity baseline, combine every benchmarked local model. With 11
voters, majority requires 6-of-11 agreement — a high bar that rewards only
answers with broad consensus.

| Model              | WattBot Score | NA Recall | Latency | VRAM (4-bit) |
|--------------------|:---:|:---:|:---:|:---:|
| Qwen 2.5 72B      | 0.752 | 0.938 | 15.7s | ~33 GB |
| Qwen3 30B-A3B     | 0.724 | 0.813 | 76.3s | ~18 GB |
| Qwen 2.5 32B      | 0.710 | 1.000 | 8.4s  | ~22 GB |
| Qwen 2.5 14B      | 0.660 | 0.875 | 16.0s | ~15 GB |
| Mixtral 8x22B      | 0.643 | 0.875 | 17.3s | ~80 GB |
| Qwen 2.5 7B       | 0.570 | 0.563 | 11.8s | ~6 GB  |
| Qwen 2.5 3B       | 0.526 | 0.375 | 21.7s | ~3 GB  |
| Mistral 7B         | 0.501 | 0.938 | 14.7s | ~6 GB  |
| Mixtral 8x7B       | 0.416 | 0.938 | 19.5s | ~25 GB |
| Phi-3 Mini         | 0.398 | 0.750 | 39.1s | ~3 GB  |
| OLMoE 1B-7B        | 0.143 | 1.000 | 213.5s | ~5 GB |

**Trade-offs vs. the top-3 / top-5 ensembles:**

- **Pros:** Maximum architectural diversity (5 model families); weak models
  can't hurt if they're in the minority; useful as an upper/lower bound
  comparison for ablation studies.
- **Cons:** The 6 weaker models (score < 0.64) add noise more often than
  signal. If the top-3 already agree, the extra votes are redundant. Total
  sequential VRAM is ~216 GB. OLMoE's 213s latency dominates wall-clock time.

```bash
# All-model ensemble on test_solutions
python scripts/run_ensemble.py \
    --experiments \
        qwen72b-bench qwen3-30b-a3b-bench qwen32b-bench qwen14b-bench \
        mixtral-8x22b-bench qwen7b-bench qwen3b-bench mistral7b-bench \
        mixtral-8x7b-bench phi3-mini-bench olmoe-1b7b-bench \
    --name ensemble-all11-majority \
    --strategy majority --ignore-blank \
    --env PowerEdge \
    --datafile test_solutions

# All-model ensemble on train_QA
python scripts/run_ensemble.py \
    --experiments \
        qwen72b-bench qwen3-30b-a3b-bench qwen32b-bench qwen14b-bench \
        mixtral-8x22b-bench qwen7b-bench qwen3b-bench mistral7b-bench \
        mixtral-8x7b-bench phi3-mini-bench olmoe-1b7b-bench \
    --name ensemble-all11-majority \
    --strategy majority --ignore-blank \
    --env PowerEdge \
    --datafile train_QA
```

### Audit experiment quality

```bash
# Audit all experiments
python scripts/audit_experiments.py

# Audit only train_QA experiments
python scripts/audit_experiments.py --datafile train_QA
```

Checks for: missing token counts, high latency, high error rates, score
inconsistencies, duplicate model runs.

---

## 4b) Bedrock vs Local comparison workflow

The pipeline is designed so that both providers use identical retrieval settings,
prompts, and scoring — only the LLM call differs. This makes results directly
comparable for evaluating model quality, latency, and cost trade-offs.

### Step 1: Run both providers

```bash
# Local HF models on your GPU machine
python scripts/run_full_benchmark.py --provider hf_local --env PowerEdge

# AWS Bedrock models (from any machine with credentials)
python scripts/run_full_benchmark.py --provider bedrock --env Bedrock
```

Or run specific models from each provider:

```bash
# Bedrock: Claude 3 Haiku
python scripts/run_experiment.py \
    --config vendor/KohakuRAG/configs/bedrock_claude_haiku.py \
    --name claude-haiku-bench --env Bedrock

# Local: Qwen 7B (comparable size/speed tier)
python scripts/run_experiment.py \
    --config vendor/KohakuRAG/configs/hf_qwen7b.py \
    --name qwen7b-bench --env PowerEdge
```

### Step 2: Normalise and score both

```bash
python scripts/posthoc.py artifacts/experiments/Bedrock/train_QA/claude-haiku-bench/
python scripts/posthoc.py artifacts/experiments/PowerEdge/train_QA/qwen7b-bench/
```

### Step 3: Compare

```bash
# Generate cross-provider comparison matrix
python scripts/generate_results_matrix.py

# Or compare specific submissions
python scripts/score.py data/train_QA.csv \
    artifacts/experiments/Bedrock/train_QA/claude-haiku-bench/submission.csv
python scripts/score.py data/train_QA.csv \
    artifacts/experiments/PowerEdge/train_QA/qwen7b-bench/submission.csv
```

### Step 4: Cross-provider ensemble (optional)

You can also mix bedrock and local models in a cross-model ensemble:

```bash
python scripts/run_ensemble.py \
    --experiments claude-sonnet-bench qwen72b-bench qwen32b-bench \
    --name ensemble-hybrid \
    --strategy majority --ignore-blank \
    --env Hybrid \
    --datafile train_QA
```

### What to compare

| Metric | Where to find it | Notes |
|--------|-----------------|-------|
| Accuracy (WattBot score) | `summary.json → overall_score` | Primary quality metric |
| Latency per question | `summary.json → avg_latency_seconds` | Bedrock ~2-5s; local varies by model |
| Cost per run | `summary.json → estimated_cost_usd` | $0 for local; tracked for API providers |
| NA recall | `summary.json → na_accuracy` | How well the model detects unanswerable Qs |
| Ref overlap | `summary.json → ref_overlap` | Citation accuracy |
| Energy consumption | `summary.json → hardware` | GPU wattage for local (not applicable to bedrock) |

---

## 5) Generating plots

### Generate all plots

All plotting scripts accept `--datafile <name>` to restrict to a specific
question set (e.g. `train_QA` or `test_solutions`). Without `--datafile`,
all experiments are included regardless of which question set was used.

```bash
# 1. Build the results matrix (required by plot_from_matrix.py)
python scripts/generate_results_matrix.py --datafile train_QA

# 2. Generate all plots (for train_QA only)
python scripts/plot_model_size.py      --datafile train_QA
python scripts/plot_from_matrix.py     --datafile train_QA
python scripts/plot_score_breakdown.py --datafile train_QA

# Or for test_solutions
python scripts/generate_results_matrix.py --datafile test_solutions
python scripts/plot_model_size.py      --datafile test_solutions
python scripts/plot_from_matrix.py     --datafile test_solutions
python scripts/plot_score_breakdown.py --datafile test_solutions
```

When `--datafile` is provided, plots are saved to a matching subdirectory
(e.g. `artifacts/plots/train_QA/`, `artifacts/plots/test_solutions/`).
Without `--datafile`, plots go to `artifacts/plots/` directly.

Ground truth is auto-detected: `data/test_solutions.csv` if present, otherwise
`data/train_QA.csv`. Override with `--ground-truth <path>`.

### plot_model_size.py — Size & scaling analysis (8 plots)

1. **Size vs. Scores** — 4-panel (overall, value accuracy, ref overlap, NA)
2. **Size vs. Latency** — per-question average
3. **Size vs. Cost** — API cost (local models show $0)
4. **Bubble chart** — size × score × cost
5. **Overall ranking** — horizontal bar chart
6. **Cost vs. Performance** — trade-off scatter
7. **Score breakdown** — grouped bar (value/ref/NA per model)
8. **Energy per experiment** — total GPU energy (Wh) per model (local HF only)

Local HF models show as **squares**, API models as **circles**.

### plot_from_matrix.py — Matrix-based comparisons (6 plots)

1. **Overall scores** — bar chart with 95% CI
2. **Accuracy by type** — Table / Figure / Quote / Math / NA breakdown
3. **Agreement heatmap** — pairwise model agreement
4. **Unique wins** — questions only one model got right
5. **Refusal rates** — % of "unable to answer" responses
6. **Cost vs. score** — scatter (requires summary.json cost data)

### plot_score_breakdown.py — Component breakdown (1 plot)

Grouped bars showing Value Accuracy, Ref Overlap, and NA Recall per model
with 95% Wilson CI error bars.

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
planner_max_queries = 4
deduplicate_retrieval = True
rerank_strategy = "combined"
top_k_final = 10
use_reordered_prompt = True   # C→Q prompt ordering (context before question)

# Unanswerable detection
retrieval_threshold = 0.25

# Robustness settings
max_retries = 2               # iterative deepening retries on blank answers
max_concurrent = 2            # lower (1) for large models, higher (3-4) for small ones
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
python -m json.tool artifacts/experiments/train_QA/newmodel13b-smoke/summary.json
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

## 10) Running on the test set

By default, benchmarks run against `data/train_QA.csv`. To run on a
different question set (e.g. the competition test set), use `--questions`:

```bash
# Full batch on test set — results go to <env>/test_solutions/ subfolder
python scripts/run_full_benchmark.py \
    --env PowerEdge \
    --questions data/test_solutions.csv

# Single model on test set
python scripts/run_experiment.py \
    --config vendor/KohakuRAG/configs/hf_qwen7b.py \
    --name qwen7b-bench \
    --env PowerEdge \
    --questions data/test_solutions.csv
```

Results are automatically separated by datafile subfolder, so train and
test results never collide:

```
artifacts/experiments/PowerEdge/
├── train_QA/              ← train results
│   ├── qwen7b-bench/
│   └── ...
├── test_solutions/        ← test results
│   ├── qwen7b-bench/
│   └── ...
```

The `--split` flag is still supported and appends a suffix to experiment
names within the datafile subfolder. The skip-existing check uses the
datafile subfolder, so you can re-run with `--force` if needed.

---

## 11) Qwen model-size scaling experiment

To measure how WattBot performance scales with model size (keeping
everything else constant), there's a dedicated script:

```bash
# Collect existing Qwen results into a comparison table
python scripts/run_qwen_scaling.py --env PowerEdge

# Also run any missing sizes that fit in VRAM
python scripts/run_qwen_scaling.py --env PowerEdge --run-missing

# Specific sizes only
python scripts/run_qwen_scaling.py --sizes 1.5 3 7 --env PowerEdge

# Dry run (show what would run)
python scripts/run_qwen_scaling.py --run-missing --dry-run --env PowerEdge

# Run in bf16 instead of default 4-bit
python scripts/run_qwen_scaling.py --run-missing --precision bf16 --env PowerEdge
```

Available Qwen sizes (4-bit VRAM): 1.5B (~2GB), 3B (~3GB), 7B (~6GB), 14B (~10GB),
32B (~20GB), 72B (~40GB). Pass `--precision bf16` for full precision (roughly 4× more).

The script:
- **Reuses existing results** from `run_full_benchmark.py` — no duplicate runs
- Optionally runs missing sizes with `--run-missing` (sequential, frees GPU between runs)
- Auto-skips models that won't fit in available VRAM
- Produces a combined comparison table (CSV + JSON)

Output:
```
artifacts/experiments/<env>/train_QA/
├── qwen1.5b-bench/...                  ← standard experiment results
├── qwen3b-bench/...
├── qwen7b-bench/...
├── qwen_scaling_comparison.csv         ← side-by-side comparison
└── qwen_scaling_comparison.json        ← same data, for notebooks
```

The `scaling_comparison.csv` has columns for scores, latency, VRAM, disk
size, energy (Wh), and power — ready for plotting.

---

## 12) Results output for post-processing

Every experiment produces two files immediately, and a third after post-hoc
processing:

| File               | Format | What's in it                                                  |
|--------------------|--------|---------------------------------------------------------------|
| `results.json`     | JSON   | Per-question: **raw** model output, GT, latency, retrieval    |
| `summary.json`     | JSON   | Aggregate: overall score, hardware metrics, config snapshot    |
| `submission.csv`   | CSV    | Normalised Kaggle-format predictions (created by `posthoc.py`)|

`results.json` intentionally stores **raw (un-normalised)** model output so
you can always re-run normalisation with improved rules without re-running
expensive LLM inference:

```bash
# Normalise and re-score (writes submission.csv alongside results.json)
python scripts/posthoc.py artifacts/experiments/train_QA/qwen7b-v1/

# Dry-run: see the score without writing files
python scripts/posthoc.py artifacts/experiments/train_QA/qwen7b-v1/ --dry-run
```

All answer normalisation logic (comma stripping, abbreviation expansion,
range formatting, hedging prefix removal, etc.) lives in **one place**:
`scripts/posthoc.py`.

To iterate on post-processing in Python:

```python
import json, pandas as pd

# Load per-question results (raw model output)
with open("artifacts/experiments/train_QA/qwen7b-v1/results.json") as f:
    results = json.load(f)

# Convert to DataFrame for analysis
df = pd.DataFrame(results)
print(df[["id", "pred_value", "gt_value", "value_correct", "latency_seconds"]])

# Load summary (includes hardware)
with open("artifacts/experiments/train_QA/qwen7b-v1/summary.json") as f:
    summary = json.load(f)
print(f"VRAM: {summary['hardware']['gpu_vram_allocated_gb']} GB")
print(f"Energy: {summary['hardware']['gpu_energy_wh']} Wh")
```

For the scaling experiment, use the combined file:

```python
scaling = pd.read_csv("artifacts/experiments/PowerEdge/train_QA/qwen_scaling_comparison.csv")
print(scaling[["model", "params_b", "overall_score", "vram_allocated_gb", "energy_wh"]])
```

---

## 13) Directory structure reference

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
│   ├── posthoc.py            # Post-hoc normalisation & scoring (single source of truth)
│   ├── score.py              # WattBot scoring metric (used by Kaggle + posthoc)
│   ├── run_qwen_scaling.py   # Qwen size scaling experiment
│   ├── run_full_benchmark.py # Run all models
│   ├── run_wattbot_eval.py   # Quick eval + score
│   ├── run_ensemble.py       # Combine multiple runs
│   ├── generate_results_matrix.py
│   ├── audit_experiments.py
│   ├── plot_model_size.py
│   ├── plot_from_matrix.py
│   └── plot_score_breakdown.py
├── artifacts/                # Experiment outputs (gitignored, machine-specific)
│   ├── experiments/          # Per-experiment results, organized by --env and datafile
│   │   ├── PowerEdge/       # Results from PowerEdge runs
│   │   │   ├── train_QA/   # Results from train_QA.csv questions
│   │   │   │   ├── qwen7b-v1/
│   │   │   │   │   ├── submission.csv
│   │   │   │   │   ├── results.json
│   │   │   │   │   └── summary.json
│   │   │   │   └── ...
│   │   │   └── test_solutions/  # Results from test_solutions.csv
│   │   │       └── ...
│   │   └── GB10/            # Results from GB10 runs
│   │       └── train_QA/
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
