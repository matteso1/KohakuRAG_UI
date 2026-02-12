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

Open a terminal (**PowerShell** on Windows, **Terminal** on macOS/Linux).

```bash
# Navigate to where you keep git repos (adjust to your preference)
# Windows example:
cd ~/GitHub
# macOS / Linux example:
cd ~/GitHub

git clone https://github.com/qualiaMachine/KohakuRAG_UI.git
cd KohakuRAG_UI
```

### 2) Install uv (fast Python package manager)

<details>
<summary><b>Windows (PowerShell)</b></summary>

```powershell
irm https://astral.sh/uv/install.ps1 | iex
```

Close and reopen PowerShell, then verify:

```powershell
uv --version
```

</details>

<details>
<summary><b>macOS / Linux</b></summary>

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc   # or ~/.zshrc on macOS
uv --version
```

</details>

### 3) Create and activate the virtual environment

From the repo root (`KohakuRAG_UI/`):

```bash
uv venv --python 3.11
```

Activate the venv:

<details>
<summary><b>Windows (PowerShell)</b></summary>

```powershell
.\.venv\Scripts\Activate.ps1
```

If you get an "execution policy" error:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
.\.venv\Scripts\Activate.ps1
```

</details>

<details>
<summary><b>Windows (cmd.exe)</b></summary>

```cmd
.\.venv\Scripts\activate.bat
```

</details>

<details>
<summary><b>macOS / Linux</b></summary>

```bash
source .venv/bin/activate
```

</details>

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

<details>
<summary><b>Windows</b></summary>

Download and run the official MSI installer:

https://awscli.amazonaws.com/AWSCLIV2.msi

Or, if you have `winget`:

```powershell
winget install Amazon.AWSCLI
```

Close and reopen your terminal after installation.

</details>

<details>
<summary><b>macOS</b></summary>

```bash
brew install awscli
```

Or use the official pkg installer:
https://awscli.amazonaws.com/AWSCLIV2.pkg

</details>

<details>
<summary><b>Linux</b></summary>

```bash
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip && sudo ./aws/install
```

</details>

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
| SSO session name | `uw-madison` (or any name you prefer) |
| SSO start URL | *(provided by Chris — check team Slack/email)* |
| SSO region | `us-east-2` |
| SSO registration scopes | `sso:account:access` |

A browser window will open for login. After authenticating, select the
account and role, then finish the CLI prompts:

| Prompt | Value |
|--------|-------|
| CLI default client Region | `us-east-2` |
| CLI default output format | `json` |
| CLI profile name | `bedrock` (or whatever you prefer) |

### 9) Create your `.env` file

Copy the template and fill in your profile:

<details>
<summary><b>Windows (PowerShell)</b></summary>

```powershell
Copy-Item .env.example .env
```

</details>

<details>
<summary><b>macOS / Linux</b></summary>

```bash
cp .env.example .env
```

</details>

Edit `.env` and set:

```bash
AWS_PROFILE=bedrock
AWS_REGION=us-east-2
```

> **Never commit `.env`** — it is already in `.gitignore`.

### Alternative credential methods

<details>
<summary><b>Option B: Direct credentials (CI/CD or non-SSO)</b></summary>

```bash
# In .env (DO NOT commit this file)
AWS_ACCESS_KEY_ID=AKIA...
AWS_SECRET_ACCESS_KEY=...
AWS_REGION=us-east-2
```

</details>

<details>
<summary><b>Option C: IAM instance role (EC2/Lambda)</b></summary>

If running on an AWS EC2 instance or Lambda with an attached IAM role that
has `bedrock:InvokeModel` permissions, no credentials need to be configured.
The SDK will use the instance role automatically.

</details>

---

## Phase 3 — Verify Bedrock access

### 10) Refresh SSO session

SSO tokens expire after ~8–12 hours. Refresh before running experiments:

```bash
aws sso login --profile bedrock
```

### 11) Quick test script

Save this as `test_bedrock.py` (or just paste into a Python REPL):

```python
import boto3

session = boto3.Session(profile_name="bedrock")  # omit for env vars / instance role
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
| `ExpiredTokenException` | SSO session expired | `aws sso login --profile bedrock` |
| `AccessDeniedException` | Model not enabled | See "Enable Models" below |
| `ValidationException: model not found` | Wrong model ID or region | Check model ID and try `us-east-1` |
| `UnrecognizedClientException` | Bad credentials | Re-run `aws configure sso` |
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

### 12) Build the document index

The index only needs to be built once (or when the corpus changes):

```bash
cd vendor/KohakuRAG
kogine run scripts/wattbot_build_index.py --config configs/jinav4/index.py
cd ../..
```

Verify the database was created:

<details>
<summary><b>Windows (PowerShell)</b></summary>

```powershell
Get-Item data\embeddings\wattbot_jinav4.db | Select-Object Length
```

</details>

<details>
<summary><b>macOS / Linux</b></summary>

```bash
ls -lh data/embeddings/wattbot_jinav4.db
```

</details>

### 13) Run a single experiment

```bash
# Claude 3 Haiku (fast, cheap — good first test)
python scripts/run_experiment.py ^
  --config vendor/KohakuRAG/configs/bedrock_claude_haiku.py ^
  --name claude-haiku-bench ^
  --env Bedrock
```

> **Note:** On Windows cmd.exe, use `^` for line continuation. In
> PowerShell, use `` ` `` (backtick). On macOS/Linux, use `\`.

```bash
# Claude 3.5 Sonnet (higher quality)
python scripts/run_experiment.py \
  --config vendor/KohakuRAG/configs/bedrock_claude_sonnet.py \
  --name claude-sonnet-bench \
  --env Bedrock
```

### 14) Full Bedrock benchmark

```bash
python scripts/run_full_benchmark.py --provider bedrock --env Bedrock
```

### 15) Post-hoc normalization and scoring

```bash
python scripts/posthoc.py artifacts/experiments/Bedrock/train_QA/claude-haiku-bench/results.json
```

### 16) Compare Bedrock vs local results

After running both providers:

```bash
# Run bedrock models
python scripts/run_full_benchmark.py --provider bedrock --env Bedrock

# Run local models (on a GPU machine)
python scripts/run_full_benchmark.py --provider hf_local --env PowerEdge

# Generate comparison plots
python scripts/plot_from_matrix.py
```

Results are organized by environment:

```
artifacts/experiments/
  Bedrock/
    train_QA/
      claude-haiku-bench/
        results.json
        summary.json
      claude-sonnet-bench/
        ...
  PowerEdge/
    train_QA/
      qwen7b-bench/
        ...
```

---

## Phase 5 — Streamlit app

### 17) Launch the app

```bash
streamlit run app.py
```

In the sidebar you'll see both `bedrock_*` and `hf_*` model configs.
Bedrock models don't require a GPU — when only bedrock models are selected,
the app skips VRAM estimation and shows "API mode".

You can create ensembles mixing bedrock and local models (e.g.,
`bedrock_claude_sonnet` + `hf_qwen7b`).

---

## Available Bedrock Models

Pre-configured models in `vendor/KohakuRAG/configs/`:

| Config file | Model | Cost (per 1M tokens) | Notes |
|-------------|-------|----------------------|-------|
| `bedrock_claude_haiku.py` | Claude 3 Haiku | $0.25 in / $1.25 out | Fast, cheap — good for prototyping |
| `bedrock_claude_sonnet.py` | Claude 3.5 Sonnet v2 | $3.00 in / $15.00 out | Best quality for technical QA |
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
- Post-hoc answer normalization
- WattBot scoring

The **only** difference is the LLM call:
- `BedrockChatModel` → AWS Bedrock Converse API (boto3)
- `HuggingFaceLocalChatModel` → Local GPU inference (transformers)

This means results are directly comparable for evaluating model quality,
latency, and cost.
