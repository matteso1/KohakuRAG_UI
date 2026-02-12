# AWS Bedrock Setup Guide

End-to-end instructions for running the WattBot RAG pipeline with AWS Bedrock as the LLM backend.

---

## Prerequisites

- An AWS account with Bedrock access (provided by Chris / UW-Madison)
- Python 3.10+
- The KohakuRAG_UI repository cloned locally

---

## 1. Install boto3

```bash
pip install boto3
```

This is the only additional dependency needed for Bedrock. The rest of the pipeline (embeddings, vector store, scoring) uses the same packages as the local pipeline.

---

## 2. Configure AWS Credentials

### Option A: AWS SSO (Recommended for team members)

This is the recommended approach for the UW-Madison team.

```bash
# Install AWS CLI if not already installed
# macOS:
brew install awscli
# Linux:
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip && sudo ./aws/install

# Configure SSO
aws configure sso
```

You will be prompted for:

| Prompt | Value |
|--------|-------|
| SSO session name | `uw-madison` (or any name you prefer) |
| SSO start URL | *(provided by Chris — check team Slack/email)* |
| SSO region | `us-east-2` |
| SSO registration scopes | `sso:account:access` |

After browser-based login, select the account and role, then:

| Prompt | Value |
|--------|-------|
| CLI default client Region | `us-east-2` |
| CLI default output format | `json` |
| CLI profile name | `bedrock` (or whatever you prefer) |

**Set the profile in your `.env`:**

```bash
# In .env
AWS_PROFILE=bedrock
AWS_REGION=us-east-2
```

**Refresh SSO session** (tokens expire after ~8-12 hours):

```bash
aws sso login --profile bedrock
```

### Option B: Direct Credentials (for CI/CD or non-SSO environments)

```bash
# In .env (DO NOT commit this file)
AWS_ACCESS_KEY_ID=AKIA...
AWS_SECRET_ACCESS_KEY=...
AWS_REGION=us-east-2
```

### Option C: IAM Instance Role (EC2/Lambda)

If running on an AWS EC2 instance or Lambda with an attached IAM role that has `bedrock:InvokeModel` permissions, no credentials need to be configured. The SDK will use the instance role automatically.

---

## 3. Verify Bedrock Access

### Quick test script

```python
import boto3

session = boto3.Session(profile_name="bedrock")  # or omit for env vars/instance role
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

Save as `test_bedrock.py` and run:

```bash
python test_bedrock.py
```

**Expected output:**

```
Bedrock says: Hello!
Tokens: {'inputTokens': 18, 'outputTokens': 4, 'totalTokens': 22}
```

### Troubleshooting access errors

| Error | Cause | Fix |
|-------|-------|-----|
| `ExpiredTokenException` | SSO session expired | Run `aws sso login --profile bedrock` |
| `AccessDeniedException` | Model not enabled | See "Enable Models" below |
| `ValidationException: model not found` | Wrong model ID or region | Check model ID spelling and try `us-east-1` |
| `UnrecognizedClientException` | Bad credentials | Re-run `aws configure sso` |
| `ResourceNotFoundException` | Model not available in region | Try `us-east-1` or `us-west-2` |

### Enable models in Bedrock Console

If you get `AccessDeniedException` for a specific model:

1. Go to [AWS Bedrock Console](https://console.aws.amazon.com/bedrock/)
2. Select region `us-east-2` in the top-right
3. Navigate to **Model access** in the left sidebar
4. Click **Manage model access**
5. Check the models you need (Claude, Llama, Nova, etc.)
6. Click **Save changes**

Most models are available immediately. Some (like Claude 3 Opus) may require a brief approval process.

---

## 4. Available Bedrock Models

These are the pre-configured models in `vendor/KohakuRAG/configs/`:

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

**Cost monitoring**: Check [AWS Cost Explorer](https://console.aws.amazon.com/cost-management/home) after 24 hours (there can be a lag before costs appear).

---

## 5. Run Experiments

### Single model

```bash
# Claude 3 Haiku (fast, cheap — good first test)
python scripts/run_experiment.py \
  --config vendor/KohakuRAG/configs/bedrock_claude_haiku.py \
  --name claude-haiku-bench \
  --env Bedrock

# Claude 3.5 Sonnet (higher quality)
python scripts/run_experiment.py \
  --config vendor/KohakuRAG/configs/bedrock_claude_sonnet.py \
  --name claude-sonnet-bench \
  --env Bedrock
```

### Full bedrock benchmark

```bash
python scripts/run_full_benchmark.py --provider bedrock --env Bedrock
```

### Post-hoc normalization and scoring

```bash
python scripts/posthoc.py artifacts/experiments/Bedrock/train_QA/claude-haiku-bench/results.json
```

### Compare bedrock vs local results

After running both:

```bash
# Run local models
python scripts/run_full_benchmark.py --provider hf_local --env PowerEdge

# Run bedrock models
python scripts/run_full_benchmark.py --provider bedrock --env Bedrock

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

## 6. Streamlit App with Bedrock

The Streamlit app automatically discovers `bedrock_*.py` configs alongside `hf_*.py` configs.

```bash
streamlit run app.py
```

In the sidebar, you'll see both bedrock and local model configs. Bedrock models don't require a GPU — when only bedrock models are selected, the app skips VRAM estimation and shows "API mode".

You can also create ensembles mixing bedrock and local models (e.g., `bedrock_claude_sonnet` + `hf_qwen7b`).

---

## 7. Adding New Bedrock Models

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
aws bedrock list-foundation-models --region us-east-2 --query "modelSummaries[].modelId" --output table
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

## 8. Architecture: What's Shared vs. Provider-Specific

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

This means results are directly comparable for evaluating model quality, latency, and cost.
