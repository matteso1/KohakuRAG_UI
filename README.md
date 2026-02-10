# WattBot RAG User Interface

Streamlit interface for the KohakuRAG pipeline. Answers questions about AI sustainability research using AWS Bedrock.

## Project Overview

This repo wraps [KohakuRAG](https://github.com/KohakuBlueleaf/KohakuRAG) (the #1 solution from WattBot 2025) with a web interface and deploys it on AWS. The goal is a pay-per-use chatbot that answers questions about the environmental impacts of AI, with proper citations.

**Current Status:**

- Bedrock integration: **Complete & Verified** (Pricing audit done Feb 2026)
- Streamlit UI: **Complete** (Dark theme, Cost Efficiency plots)
- Deployment: planned

## Architecture

```mermaid
flowchart LR
    User --> Streamlit
    Streamlit --> Pipeline[RAG Pipeline]
    Pipeline --> Embeddings[Jina Embeddings]
    Pipeline --> VectorDB[(SQLite Index)]
    Pipeline --> Bedrock[AWS Bedrock]
    Bedrock --> Claude[Claude 3]
```

The vector index is a single SQLite file (~80MB) containing embedded research papers. Queries hit Bedrock for answer generation. No always-on infrastructure.

## Setup

### Prerequisites

- Python 3.10+
- AWS CLI v2 with SSO configured
- NVIDIA GPU (for embedding, optional if using pre-built index)

### Installation

```bash
git clone --recurse-submodules https://github.com/matteso1/KohakuRAG_UI.git
cd KohakuRAG_UI

pip install -r requirements.txt
pip install -e KohakuRAG/

# AWS auth
aws sso login --profile bedrock_nils
```

### Get the Vector Index

Pre-built index is on S3:

```bash
aws s3 cp s3://wattbot-nils-kohakurag/indexes/wattbot_jinav4.db artifacts/wattbot_jinav4.db --profile bedrock_nils
```

### Run Demo

```bash
python scripts/demo_bedrock_rag.py --question "What is the carbon footprint of training GPT-3?"
```

## Benchmark Scores

| Config | Score | Cost/1M (In/Out) | Notes |
|--------|-------|------------------|-------|
| JinaV3 + Haiku | 0.665 | $0.80 / $4.00 | Previous baseline |
| JinaV4 + Sonnet 3.5 | 0.747 | $3.00 / $15.00 | High accuracy leader |
| GPT-OSS 120B | 0.725 | $0.15 / $0.60 | Best cost efficiency |
| DeepSeek R1 | 0.735 | $1.35 / $5.40 | Strong reasoning, moderate cost |

Gap is mostly model size and ensemble voting. DeepSeek R1 shows strong reasoning capabilities at a fraction of Sonnet's cost.

## Cost

- **Claude 3 Haiku**: ~$2.86 per full benchmark run
- **Claude 3.5 Sonnet**: ~$10.80 per run
- **GPT-OSS 120B**: ~$0.51 per run (Highly efficient)
- **DeepSeek R1**: ~$1.87 per run
- Idle: $0 (Bedrock is pay-per-use)

## Repo Structure

```
KohakuRAG_UI/
├── src/llm_bedrock.py         # Bedrock integration
├── scripts/
│   ├── demo_bedrock_rag.py    # E2E demo
│   └── run_wattbot_eval.py    # Batch evaluation
├── configs/                    # Index configs
├── docs/                       # Documentation
├── KohakuRAG/                  # Submodule
└── artifacts/                  # Vector indexes (gitignored)
```

## Branches

| Branch | Owner | Status |
|--------|-------|--------|
| main | Team | Stable |
| bedrock | Nils | Complete |
| local | Blaise | In progress |

## For Blaise

Download the index:

```bash
aws s3 cp s3://wattbot-nils-kohakurag/indexes/wattbot_jinav4.db artifacts/
```

Basic usage:

```python
from llm_bedrock import BedrockChatModel

chat = BedrockChatModel(
    profile_name="bedrock_nils",
    region_name="us-east-2",
    model_id="us.anthropic.claude-3-haiku-20240307-v1:0"
)

answer = await chat.complete(prompt)
```

See `scripts/demo_bedrock_rag.py` for full pipeline example.

## Docs

- [Progress Report](docs/tuesday-progress-report.md) - Week 1 accomplishments
- [Bedrock Proposal](docs/bedrock-integration-proposal.md) - Technical design
- [Bedrock Integration Details](docs/BEDROCK_INTEGRATION.md) - **New**: Pricing audit, DeepSeek token tracking, Cost Analysis

## Team

- Chris Endemann - Supervisor
- Nils Matteson - Bedrock integration
- Blaise Enuh - Streamlit UI
