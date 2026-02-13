# WattBot RAG User Interface

A unified RAG pipeline and Streamlit chat interface for the WattBot research corpus on sustainable AI. Supports **AWS Bedrock** (managed API) and **local HuggingFace** (fully offline) inference through a single codebase.

## Project Context

This repository supports the Research Cyberinfrastructure Exploration initiative at UW-Madison. The goal is to build a chatbot that answers questions about the environmental impacts of AI using a curated corpus of energy and sustainability research papers.

The project uses [KohakuRAG](https://github.com/KohakuBlueleaf/KohakuRAG), the top-ranked solution from the 2025 WattBot Challenge, as the core retrieval engine. This repository provides:

1. A Streamlit chat UI with single-model and ensemble modes
2. A provider-agnostic experiment runner for benchmarking models
3. Support for AWS Bedrock, local HuggingFace, OpenRouter, and OpenAI backends

## Architecture

Both providers share the **exact same RAG pipeline** — only the LLM call differs:

```
User Question
    │
    ▼
┌──────────────────────────────────────────────────┐
│  KohakuRAG Pipeline (shared across all providers)│
│                                                  │
│  1. Query Planning   ─── LLM call ──┐           │
│  2. Hierarchical Vector Search       │           │
│  3. Dedup + Consensus Reranking      │           │
│  4. Context Expansion (C→Q)          │           │
│  5. Answer Generation ─── LLM call ──┘           │
└──────────────────────────────────────────────────┘
                    │
    ┌───────────────┼───────────────┐
    ▼               ▼               ▼
 Bedrock        HF Local       OpenRouter
 (API)         (GPU/CPU)         (API)
```

### Key Pipeline Features (from KohakuRAG paper)

- **4-level hierarchical indexing**: Document → Section → Paragraph → Sentence
- **Bottom-up length-weighted embeddings**: Parent vectors are weighted averages of children
- **LLM query planning**: Single question expanded into 3-4 diverse retrieval queries
- **C→Q prompt ordering**: Context placed before question (combats "lost in the middle")
- **Consensus reranking**: `0.4 × frequency + 0.6 × score` deduplication
- **Post-hoc normalization**: Raw model output saved; normalization applied separately

### Provider Comparison

| Feature | Bedrock | Local HF | OpenRouter |
|---------|---------|----------|------------|
| Network | Required | None | Required |
| Cost | Per-token API | $0 (local compute) | Per-token API |
| Setup | AWS SSO + boto3 | GPU + torch | API key |
| Models | Claude, Llama, Nova | Qwen, Mistral, Llama, etc. | 300+ models |
| Concurrency | High (API) | Limited by VRAM | High (API) |

## Quick Start

### 1. Clone and install

```bash
git clone https://github.com/qualiaMachine/KohakuRAG_UI.git
cd KohakuRAG_UI
cp .env.example .env
# Edit .env with your credentials
```

### 2a. Bedrock setup (AWS API)

```bash
pip install boto3
aws configure sso  # Set up your SSO profile
# Edit .env: AWS_PROFILE=your-sso-profile-name
```

### 2b. Local HF setup (GPU inference)

```bash
pip install -r local_requirements.txt
# Install vendor packages:
cd vendor/KohakuRAG && pip install -e ".[local]" && cd ../..
cd vendor/KohakuVault && pip install -e . && cd ../..
```

### 3. Build the vector index (first time only)

```bash
cd vendor/KohakuRAG
kogine run scripts/wattbot_build_index.py --config configs/jinav4/index.py
```

### 4. Run the Streamlit app

```bash
streamlit run app.py
```

### 5. Run batch experiments

```bash
# Single model
python scripts/run_experiment.py --config vendor/KohakuRAG/configs/bedrock_claude_haiku.py
python scripts/run_experiment.py --config vendor/KohakuRAG/configs/hf_qwen7b.py

# Full benchmark (all configured models)
python scripts/run_full_benchmark.py --provider bedrock --env Bedrock
python scripts/run_full_benchmark.py --provider hf_local --env PowerEdge

# Post-hoc normalization + scoring
python scripts/posthoc.py artifacts/experiments/<env>/<data>/<model>/results.json
```

## Repository Structure

```
.
├── app.py                              # Streamlit chat UI (all providers)
├── .env.example                        # Environment template
├── local_requirements.txt              # Dependencies for local HF inference
├── data/
│   ├── train_QA.csv                    # Training questions (ground truth)
│   ├── metadata.csv                    # Document bibliography
│   ├── corpus/                         # Parsed document JSONs
│   ├── pdfs/                           # Cached PDF downloads
│   └── embeddings/                     # Vector DB (gitignored)
├── docs/
│   ├── Pipeline_Architecture.md        # KohakuRAG pipeline details
│   ├── Benchmarking_Guide.md           # Running and comparing experiments
│   ├── Streamlit_App_Guide.md          # App deployment guide
│   ├── Setup_GB10.md                   # Dell GB10 ARM setup
│   ├── Setup_PowerEdge.md              # PowerEdge cluster setup
│   ├── bedrock-integration-proposal.md # AWS Bedrock design document
│   └── meeting-notes.md               # Team decisions
├── scripts/
│   ├── run_experiment.py               # Single-model experiment runner
│   ├── run_ensemble.py                 # Same-model & cross-model ensembles
│   ├── run_full_benchmark.py           # Orchestrate all model benchmarks
│   ├── posthoc.py                      # Answer normalization (single source of truth)
│   ├── score.py                        # WattBot ground-truth scoring
│   └── hardware_metrics.py             # GPU/CPU power monitoring
├── vendor/
│   ├── KohakuRAG/                      # Core RAG engine (vendored)
│   │   ├── src/kohakurag/
│   │   │   ├── llm.py                  # All LLM providers (Bedrock, HF, OpenRouter, OpenAI)
│   │   │   ├── pipeline.py             # RAG orchestration
│   │   │   ├── embeddings.py           # Embedding models
│   │   │   ├── datastore.py            # Vector store
│   │   │   ├── indexer.py              # Document indexing
│   │   │   └── types.py                # Data structures
│   │   ├── configs/
│   │   │   ├── bedrock_*.py            # Bedrock model configs
│   │   │   └── hf_*.py                 # Local HF model configs
│   │   └── scripts/                    # Indexing and QA scripts
│   └── KohakuVault/                    # Vector store backend (vendored)
└── artifacts/experiments/              # Benchmark results
```

## Config Files

Each model gets a config file in `vendor/KohakuRAG/configs/`. All configs share the same retrieval settings for fair comparison — only the LLM provider differs.

**Bedrock configs** (`bedrock_*.py`):
- `bedrock_claude_haiku` — Claude 3 Haiku (fast, cheap)
- `bedrock_claude_sonnet` — Claude 3.5 Sonnet (balanced)
- `bedrock_nova_pro` — Amazon Nova Pro
- `bedrock_llama4_scout` — Meta Llama 4 Scout

**Local HF configs** (`hf_*.py`):
- `hf_qwen7b`, `hf_qwen14b`, `hf_qwen32b`, `hf_qwen72b` — Qwen 2.5 family
- `hf_mistral7b`, `hf_llama3_8b` — Other open models
- `hf_phi3_mini`, `hf_mixtral_8x7b` — Small/MoE variants

## Documentation

- [Pipeline Architecture](docs/Pipeline_Architecture.md) — How the 4-level hierarchical RAG works
- [Benchmarking Guide](docs/Benchmarking_Guide.md) — Running experiments and comparing models
- [Streamlit App Guide](docs/Streamlit_App_Guide.md) — App deployment and PowerEdge integration
- [Setup: Bedrock](docs/Setup_Bedrock.md) — AWS credentials, model access, and testing
- [Setup: GB10](docs/Setup_GB10.md) — Dell GB10 ARM workstation setup
- [Setup: PowerEdge](docs/Setup_PowerEdge.md) — PowerEdge cluster setup
- [Bedrock Integration Proposal](docs/bedrock-integration-proposal.md) — Original AWS design document
- [Dependency Fixes: GB10](docs/Dep_fixes_GB10.md) — ARM-specific patches

## Team

| Name | Role | GitHub |
|------|------|--------|
| Chris Endemann | Research Supervisor | [@qualiaMachine](https://github.com/qualiaMachine) |
| Blaise Enuh | Local deployment | [@EnuhBlaise](https://github.com/EnuhBlaise) |
| Nils Matteson | AWS Bedrock integration | [@matteso1](https://github.com/matteso1) |

## Related Resources

- [KohakuRAG](https://github.com/KohakuBlueleaf/KohakuRAG) — Core RAG engine
- [WattBot 2025 Competition](https://www.kaggle.com/competitions/WattBot2025/overview) — Original challenge
- [AWS Bedrock Documentation](https://docs.aws.amazon.com/bedrock/) — Managed LLM service

## License

Research project under UW-Madison Research Cyberinfrastructure.
