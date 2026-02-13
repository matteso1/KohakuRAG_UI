# AWS Bedrock Setup Guide

End-to-end instructions for running the WattBot RAG pipeline with
**AWS Bedrock** as the LLM backend.  Works on **Windows**, **macOS**, and
**Linux**.

Repo: https://github.com/qualiaMachine/KohakuRAG_UI

---

## Prerequisites

- An AWS account with Bedrock access (provided by Chris / UW-Madison)
- Python 3.10+ ([python.org](https://www.python.org/downloads/) or your
  system package manager)
- Git ([git-scm.com](https://git-scm.com/downloads))
- The KohakuRAG_UI repository cloned locally

---

## Phase 1 — Environment setup

### 1) Clone the repo

Open a terminal (**Git Bash** on Windows, **Terminal** on macOS/Linux).

```bash
# Navigate to where you keep git repos (adjust to your preference)
# Windows example:
cd ~/GitHub
# macOS / Linux example:
cd ~/GitHub

git clone -b main https://github.com/qualiaMachine/KohakuRAG_UI.git
cd KohakuRAG_UI
```

### 2) Install uv (fast Python package manager)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Close and reopen your terminal, then verify:

```bash
uv --version
```

### 3) Create and activate the virtual environment

From the repo root (`KohakuRAG_UI/`):

```bash
uv venv --python 3.11
```

Activate the venv:

```bash
# Windows (Git Bash)
source .venv/Scripts/activate

# macOS / Linux
source .venv/bin/activate
```

Verify:

```bash
python --version   # should show 3.11.x
```

> **Tip:** Always make sure the venv is active (you should see `(.venv)` in
> your prompt) before running any `pip install` or `python` commands.

### 4) Install vendored dependencies

The core RAG engine and vector store are vendored in the repo. Install them
editable so imports resolve locally:

```bash
uv pip install -e vendor/KohakuVault
uv pip install -e vendor/KohakuRAG
```

### 5) Install Bedrock and app dependencies

```bash
uv pip install boto3 streamlit python-dotenv
```

> `boto3` is the only extra dependency needed for Bedrock. The rest of the
> pipeline (embeddings, vector store, scoring) uses the same packages as the
> local pipeline.

### 6) Smoke test

```bash
python -c "import kohakuvault, kohakurag; print('Imports OK')"
python -c "import boto3; print(f'boto3 {boto3.__version__} OK')"
```

Both should print without errors.

---

## Phase 2 — AWS credentials

### 7) Install the AWS CLI

Follow the official guide for your OS:
https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html

- **Windows**: Download the MSI installer from https://awscli.amazonaws.com/AWSCLIV2.msi
  (or `winget install Amazon.AWSCLI` from cmd/PowerShell). Reopen Git Bash after installing.
- **macOS**: `brew install awscli`
- **Linux**: `curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip" && unzip awscliv2.zip && sudo ./aws/install`

Verify:

```bash
aws --version
```

### 8) Configure AWS SSO (recommended for team members)

```bash
aws configure sso
```

You will be prompted for:

| Prompt | Value |
|--------|-------|
| SSO session name | `bedrock_yourname` (e.g. `bedrock_endemann`) |
| SSO start URL | `https://uw-madison-dlt3.awsapps.com/start/` |
| SSO region | `us-east-2` |
| SSO registration scopes | *(press Enter for default)* |

A browser window will open for UW login. You may be asked to
**"Allow botocore-client to access your data"** — click Allow.

After authenticating, you'll see available accounts. Select the
**`ml-marathon-2024`** account, then choose the **`ml-bedrock...`** role.

Finish the CLI prompts:

| Prompt | Value |
|--------|-------|
| CLI default client Region | `us-east-2` |
| CLI default output format | *(press Enter for json)* |
| CLI profile name | `bedrock_yourname` (match the session name) |

### 9) Verify SSO works

```bash
aws sso login --profile bedrock_yourname
aws sts get-caller-identity --profile bedrock_yourname
```

You should see your email and the account ID (`183295408236`) in the output.

### 10) Create your `.env` file

Copy the template and fill in your profile:

```bash
cp .env.example .env
```

Edit `.env` and set (use your actual profile name):

```bash
AWS_PROFILE=bedrock_yourname
AWS_REGION=us-east-2
```

> **Never commit `.env`** — it is already in `.gitignore`.

### Alternative credential methods

**Option B — Direct credentials (CI/CD or non-SSO):**

```bash
# In .env (DO NOT commit this file)
AWS_ACCESS_KEY_ID=AKIA...
AWS_SECRET_ACCESS_KEY=...
AWS_REGION=us-east-2
```

**Option C — IAM instance role (EC2/Lambda):**

If running on an AWS EC2 instance or Lambda with an attached IAM role that
has `bedrock:InvokeModel` permissions, no credentials need to be configured.
The SDK will use the instance role automatically.

---

## Phase 3 — Verify Bedrock access

### 11) Refresh SSO session

SSO tokens expire after ~8–12 hours. Refresh before running experiments:

```bash
aws sso login --profile bedrock_yourname
```

### 12) Quick test script

Save this as `test_bedrock.py` (or just paste into a Python REPL):

```python
import boto3

session = boto3.Session(profile_name="bedrock_yourname")  # omit for env vars / instance role
client = session.client("bedrock-runtime", region_name="us-east-2")

response = client.converse(
    modelId="us.anthropic.claude-3-haiku-20240307-v1:0",
    messages=[{"role": "user", "content": [{"text": "Say hello in one word."}]}],
    system=[{"text": "You are a helpful assistant."}],
    inferenceConfig={"maxTokens": 50},
)

text = response["output"]["message"]["content"][0]["text"]
print(f"Bedrock says: {text}")
print(f"Tokens: {response['usage']}")
```

Run it:

```bash
python test_bedrock.py
```

Expected output:

```
Bedrock says: Hello!
Tokens: {'inputTokens': 18, 'outputTokens': 4, 'totalTokens': 22}
```

### Troubleshooting

| Error | Cause | Fix |
|-------|-------|-----|
| `InvalidRequestException` on `RegisterClient` | Wrong SSO start URL or region | Verify the start URL (`https://uw-madison-dlt3.awsapps.com/start/`) and SSO region (`us-east-2`). Re-run `aws configure sso` |
| `ExpiredTokenException` | SSO session expired | `aws sso login --profile bedrock_yourname` |
| `AccessDeniedException` | Model not enabled | See "Enable Models" below |
| `ValidationException: model not found` | Wrong model ID or region | Check model ID and try `us-east-1` |
| `UnrecognizedClientException` | Bad credentials | Re-run `aws configure sso` |
| `ThrottlingException` | Too many concurrent requests | Reduce `max_concurrent` in config; retry logic handles transient throttles |
| Slow first query | Embedding model loading (~2 GB) | Normal — subsequent queries are fast |
| `ResourceNotFoundException` | Model not available in region | Try `us-east-1` or `us-west-2` |

### Enable models in the Bedrock Console

If you get `AccessDeniedException` for a specific model:

1. Go to the [AWS Bedrock Console](https://console.aws.amazon.com/bedrock/)
2. Select region **us-east-2** in the top-right
3. Navigate to **Model access** in the left sidebar
4. Click **Manage model access**
5. Check the models you need (Claude, Llama, Nova, etc.)
6. Click **Save changes**

Most models are available immediately. Some (like Claude 3 Opus) may require
a brief approval process.

---

## Phase 4 — Build the index and run experiments

Set these variables once per session. All commands below use them.

```bash
ENV=Bedrock
DS=test_solutions                # or train_QA
```

### 13) Get or build the document index

**Option A — Download pre-built index from S3** (fastest):

```bash
aws s3 cp s3://wattbot-nils-kohakurag/indexes/wattbot_jinav4.db \
    data/embeddings/wattbot_jinav4.db --profile bedrock_yourname
```

**Option B — Build from scratch** (only needed if the corpus changes):

```bash
cd vendor/KohakuRAG
kogine run scripts/wattbot_build_index.py --config configs/jinav4/index.py
cd ../..
```

Verify the database was created:

```bash
ls -lh data/embeddings/wattbot_jinav4.db
```

### 14) Quick RAG smoke test

```bash
python scripts/demo_bedrock_rag.py --question "What is the carbon footprint of GPT-3?"
```

If you get an answer with citations, Bedrock + RAG is working end-to-end.

### 15) Run a single experiment

```bash
# Claude 3 Haiku (fast, cheap — good first test)
python scripts/run_experiment.py \
  --config vendor/KohakuRAG/configs/bedrock_haiku.py \
  --name haiku-bench \
  --env $ENV --questions data/$DS.csv

# Claude 3.5 Sonnet (higher quality)
python scripts/run_experiment.py \
  --config vendor/KohakuRAG/configs/bedrock_sonnet.py \
  --name sonnet-bench \
  --env $ENV --questions data/$DS.csv
```

Results land in `artifacts/experiments/$ENV/$DS/<name>/`. The dataset
filename stem (`train_QA` or `test_solutions`) becomes a subfolder
automatically, so train and test results never collide:

```
artifacts/experiments/Bedrock/
├── train_QA/
│   └── haiku-bench/
└── test_solutions/
    └── haiku-bench/
```

### 16) Full Bedrock benchmark

```bash
python scripts/run_full_benchmark.py --provider bedrock --env $ENV \
    --questions data/$DS.csv
```

### Choosing which models run

The **model registry** lives at the top of `scripts/run_full_benchmark.py`
in two dictionaries:

| Dictionary | Purpose |
|---|---|
| `HF_LOCAL_MODELS` | Local HuggingFace models (Qwen, Mistral, Phi, OLMoE, etc.) |
| `BEDROCK_MODELS` | AWS Bedrock API models (Claude, Llama, Nova, DeepSeek, etc.) |

Each entry maps a **config filename** (without `.py`) to an
**experiment name**:

```python
BEDROCK_MODELS = {
    "bedrock_haiku": "haiku-bench",
    "bedrock_claude35_haiku": "claude35-haiku-bench",
    "bedrock_sonnet": "sonnet-bench",
    "bedrock_claude37_sonnet": "claude37-sonnet-bench",
    "bedrock_llama3_70b": "llama3-70b-bench",
    "bedrock_llama4_scout": "llama4-scout-bench",
    "bedrock_llama4_maverick": "llama4-maverick-bench",
    "bedrock_nova_pro": "nova-pro-bench",
    "bedrock_deepseek_v3": "deepseek-r1-bench",
}
```

To **add or remove a model**, edit these dictionaries directly —
comment out a line to skip it, add a new line to include it.
The config file must exist in `vendor/KohakuRAG/configs/`; entries
without a matching config are silently skipped.

**Filtering at the command line** (no code changes needed):

```bash
# Only bedrock models
python scripts/run_full_benchmark.py --provider bedrock --env $ENV \
    --questions data/$DS.csv

# Only local models
python scripts/run_full_benchmark.py --provider hf_local --env PowerEdge \
    --questions data/$DS.csv

# A single model (substring match on config name or experiment name)
python scripts/run_full_benchmark.py --model haiku --env $ENV \
    --questions data/$DS.csv
```

### 17) Generate matrices & plots

```bash
# Results matrix (auto-discovers all systems)
python scripts/generate_results_matrix.py --datafile $DS

# Per-system plots (auto-discovers all systems)
python scripts/plot_model_size.py      --datafile $DS
python scripts/plot_score_breakdown.py --datafile $DS

# Matrix-based plots (one call per system — needs corresponding matrix)
python scripts/plot_from_matrix.py \
    --matrix artifacts/results_matrix_${ENV}.csv \
    --datafile $DS --system $ENV

# Cross-system latency comparison (all systems)
python scripts/plot_cross_system_latency.py --datafile $DS
```

Results are organized by environment and dataset:

```
artifacts/experiments/
├── Bedrock/
│   └── test_solutions/
│       ├── haiku-bench/
│       │   ├── summary.json
│       │   └── submission.csv
│       └── sonnet-bench/
└── PowerEdge/
    └── test_solutions/
        └── qwen7b-bench/
```

---

## Phase 5 — Streamlit app

### 19) Launch the app

```bash
streamlit run app.py
```

In the sidebar you'll see both `bedrock_*` and `hf_*` model configs.
Bedrock models don't require a GPU — when only bedrock models are selected,
the app skips VRAM estimation and shows "API mode".

You can create ensembles mixing bedrock and local models (e.g.,
`bedrock_claude_sonnet` + `hf_qwen7b`).

---

## Using Bedrock in Python / Streamlit

### BedrockChatModel

```python
from llm_bedrock import BedrockChatModel

chat = BedrockChatModel(
    profile_name="bedrock_yourname",   # AWS SSO profile
    region_name="us-east-2",           # Bedrock region
    model_id="us.anthropic.claude-3-haiku-20240307-v1:0",
    max_concurrent=3,                  # Parallel requests (keep low)
    max_retries=5,                     # Retry on throttle
    base_retry_delay=3.0               # Seconds between retries
)

# Simple completion
answer = await chat.complete(prompt)

# With system message
answer = await chat.complete_with_system(system_prompt, user_prompt)
```

### Full RAG query example

```python
from llm_bedrock import BedrockChatModel
from kohakurag.datastore import KVaultNodeStore
from kohakurag.embeddings import JinaEmbeddingModel

# Init (once at app startup)
chat = BedrockChatModel(profile_name="bedrock_yourname", region_name="us-east-2",
                        model_id="us.anthropic.claude-3-haiku-20240307-v1:0")
store = KVaultNodeStore(path="data/embeddings/wattbot_jinav4.db",
                        table_prefix="wattbot_jv4", dimensions=1024)
embedder = JinaEmbeddingModel()

async def answer_question(question: str) -> dict:
    query_embedding = await embedder.embed([question])
    results = store.search_sync(query_embedding[0], k=5)
    context = "\n\n".join([node.content for node, score in results])
    sources = [node.metadata.get("doc_id", "unknown") for node, score in results]
    prompt = f"Answer based on this context.\n\nContext:\n{context}\n\nQuestion: {question}"
    answer = await chat.complete(prompt)
    return {"answer": answer, "sources": list(set(sources))}
```

### Key files

| File | Purpose |
|------|---------|
| `src/llm_bedrock.py` | Bedrock integration (BedrockChatModel) |
| `scripts/demo_bedrock_rag.py` | Full working RAG example |
| `data/embeddings/wattbot_jinav4.db` | JinaV4 vector index |

---

## Available Bedrock Models

Pre-configured models in `vendor/KohakuRAG/configs/`:

| Config file | Model | Cost (per 1M tokens) | Notes |
|-------------|-------|----------------------|-------|
| `bedrock_claude_haiku.py` | Claude 3 Haiku | $0.25 in / $1.25 out | Fast, cheap — good for prototyping |
| `bedrock_claude_sonnet.py` | Claude 3.5 Sonnet v2 | $3.00 in / $15.00 out | Best quality for technical QA |
| (see `configs/bedrock_claude37_sonnet.py`) | Claude 3.7 Sonnet | $3.00 in / $15.00 out | Latest Sonnet |
| `bedrock_nova_pro.py` | Amazon Nova Pro | $0.80 in / $3.20 out | Amazon's native model |
| `bedrock_llama4_scout.py` | Llama 4 Scout 17B | $0.17 in / $0.17 out | Cheapest option |

### Cost estimation per experiment

A typical WattBot experiment with 40 questions uses roughly:
- **~2,000 input tokens per question** (context + prompt)
- **~200 output tokens per question** (structured JSON answer)
- **Total per run: ~80K input + ~8K output tokens**

| Model | Est. cost per 40-question run |
|-------|-------------------------------|
| Claude 3 Haiku | ~$0.03 |
| Claude 3.5 Sonnet | ~$0.36 |
| Amazon Nova Pro | ~$0.09 |
| Llama 4 Scout | ~$0.02 |

**Cost monitoring**: Check [AWS Cost Explorer](https://console.aws.amazon.com/cost-management/home)
after 24 hours (there can be a lag before costs appear).

---

## Adding New Bedrock Models

Create a new config file in `vendor/KohakuRAG/configs/`:

```python
"""
WattBot Evaluation Config - <Model Name> via AWS Bedrock

Usage:
    python scripts/run_experiment.py --config vendor/KohakuRAG/configs/bedrock_<name>.py
"""

# Database settings (shared — must match the index)
db = "../../data/embeddings/wattbot_jinav4.db"
table_prefix = "wattbot_jv4"

# Input/output
questions = "../../data/train_QA.csv"
output = "../../artifacts/submission_bedrock_<name>.csv"
metadata = "../../data/metadata.csv"

# LLM settings
llm_provider = "bedrock"
bedrock_model = "<model-id>"            # e.g. "us.anthropic.claude-3-haiku-20240307-v1:0"
bedrock_region = "us-east-2"
# bedrock_profile = "your-sso-profile"  # Uncomment if using SSO

# Embedding settings (must match the index — do not change)
embedding_model = "jinav4"
embedding_dim = 1024
embedding_task = "retrieval"

# Retrieval settings (keep identical across all configs for fair comparison)
top_k = 8
planner_max_queries = 4
deduplicate_retrieval = True
rerank_strategy = "combined"
top_k_final = 10
use_reordered_prompt = True

# Unanswerable detection
retrieval_threshold = 0.25

# Other settings
max_retries = 3
max_concurrent = 5
```

### Finding Bedrock model IDs

```bash
aws bedrock list-foundation-models --region us-east-2 \
    --query "modelSummaries[].modelId" --output table
```

Or check the [Bedrock model catalog](https://docs.aws.amazon.com/bedrock/latest/userguide/models-supported.html).

Common model IDs:

```
us.anthropic.claude-3-haiku-20240307-v1:0
us.anthropic.claude-3-5-haiku-20241022-v1:0
us.anthropic.claude-3-sonnet-20240229-v1:0
us.anthropic.claude-3-5-sonnet-20241022-v2:0
us.anthropic.claude-3-7-sonnet-20250219-v1:0
us.meta.llama3-70b-instruct-v1:0
us.meta.llama4-scout-17b-instruct-v1:0
us.meta.llama4-maverick-17b-instruct-v1:0
amazon.nova-pro-v1:0
mistral.mistral-small-2402-v1:0
```

---

## Architecture: What's Shared vs. Provider-Specific

Both bedrock and local pipelines use the **exact same**:
- Jina V4 embeddings (1024-dim)
- SQLite + KohakuVault vector store
- 4-level hierarchical document index
- LLM query planning (3-4 diverse queries)
- Consensus reranking (0.4 freq + 0.6 score)
- C→Q prompt ordering
- WattBot scoring (inline during experiment run)

The **only** difference is the LLM call:
- `BedrockChatModel` → AWS Bedrock Converse API (boto3)
- `HuggingFaceLocalChatModel` → Local GPU inference (transformers)

This means results are directly comparable for evaluating model quality,
latency, and cost.
