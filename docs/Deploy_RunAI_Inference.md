# Deploying WattBot RAG on RunAI (PowerEdge) — Split Inference Architecture

This guide walks through deploying the WattBot RAG system as **three RunAI
Inference workloads** on a PowerEdge server with shared GPU resources.

## Architecture Overview

```
┌──────────────────────────────────────────────────────────────┐
│                    RunAI Cluster (PowerEdge)                  │
│                                                              │
│  ┌─────────────────────┐   ┌────────────────────────┐       │
│  │  Job 1: vLLM        │   │  Job 2: Embedding      │       │
│  │  (Qwen 7B, 0.75 GPU)│   │  (Jina V4, 0.25 GPU)  │       │
│  │  :8000/v1            │   │  :8080                 │       │
│  └─────────┬───────────┘   └────────────┬───────────┘       │
│            │                             │                    │
│            │     HTTP (OpenAI API)       │   HTTP (REST)      │
│            │                             │                    │
│  ┌─────────┴─────────────────────────────┴───────────┐       │
│  │            Job 3: Streamlit App (CPU only)         │       │
│  │            :8501                                   │       │
│  │  ┌──────────────────────────────────────────────┐ │       │
│  │  │  Vector Store (SQLite on PVC) — read only    │ │       │
│  │  └──────────────────────────────────────────────┘ │       │
│  └───────────────────────────────────────────────────┘       │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐    │
│  │           PVC: /workspace/kohakurag                   │    │
│  │  ├── repo/           (cloned code)                    │    │
│  │  ├── data/           (vector DB, metadata)            │    │
│  │  └── hf_cache/       (HuggingFace model weights)     │    │
│  └──────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────┘
```

**Key idea:** The Streamlit app has **no GPU** — it makes HTTP calls to vLLM
(for LLM inference) and to the embedding server (for query embedding). The
vector store (SQLite DB) is read directly from the shared PVC.

## Prerequisites

- RunAI cluster access on the PowerEdge server
- A PVC (Persistent Volume Claim) provisioned and accessible to all jobs
- HuggingFace model weights cached on the PVC (or internet access to download)

## Step 0: Build the Vector Index (One-Time)

Before deploying the always-on inference services, you need to build the
vector index. This is a **batch operation** — run it once as a RunAI
**Training** or **Workspace** job, not as an Inference job.

### What happens at index time

1. Load all source documents (PDFs, CSVs, etc.)
2. Chunk them into passages
3. Embed every passage using Jina V4 (GPU-accelerated)
4. Store embeddings + metadata in a SQLite vector database

### How to run the index build

```bash
# SSH into a RunAI Workspace job with GPU, or submit a Training job:
runai submit index-build \
    --pvc kohakurag-pvc:/workspace \
    --gpu 1 \
    --image nvidia/cuda:12.4.1-devel-ubuntu22.04 \
    --command -- bash -c "
        cd /workspace/kohakurag/repo &&
        pip install -e vendor/KohakuRAG &&
        pip install -e vendor/KohakuVault &&
        python -m kohakurag.cli build-index \
            --config vendor/KohakuRAG/configs/hf_qwen7b.py
    "
```

The resulting SQLite database (e.g., `data/embeddings/wattbot_jinav4.db`)
lives on the PVC and is read by the Streamlit app at query time.

**Important:** The embedding model used at index time must match the one
served at query time. Both default to `jinaai/jina-embeddings-v4` with
`truncate_dim=1024`.

## Step 1: Deploy the vLLM Server (GPU)

vLLM serves the LLM via an OpenAI-compatible API.

```bash
runai submit-inference wattbot-vllm \
    --pvc kohakurag-pvc:/workspace \
    --gpu 0.75 \
    --image vllm/vllm-openai:latest \
    --port 8000 \
    --env HF_HOME=/workspace/kohakurag/hf_cache \
    --command -- python -m vllm.entrypoints.openai.api_server \
        --model Qwen/Qwen2.5-7B-Instruct \
        --dtype auto \
        --max-model-len 4096 \
        --port 8000
```

**Verify:**
```bash
curl http://wattbot-vllm:8000/v1/models
```

## Step 2: Deploy the Embedding Server (GPU)

The FastAPI embedding server wraps Jina V4 for query-time embedding.

```bash
runai submit-inference wattbot-embedding \
    --pvc kohakurag-pvc:/workspace \
    --gpu 0.25 \
    --image nvidia/cuda:12.4.1-runtime-ubuntu22.04 \
    --port 8080 \
    --env HF_HOME=/workspace/kohakurag/hf_cache \
    --env EMBEDDING_MODEL=jinaai/jina-embeddings-v4 \
    --env EMBEDDING_DIM=1024 \
    --env EMBEDDING_TASK=retrieval \
    --command -- bash -c "
        cd /workspace/kohakurag/repo &&
        pip install -e vendor/KohakuRAG &&
        pip install fastapi uvicorn &&
        python scripts/embedding_server.py
    "
```

**Verify:**
```bash
curl http://wattbot-embedding:8080/health
curl http://wattbot-embedding:8080/info
```

## Step 3: Deploy the Streamlit App (CPU Only)

The Streamlit UI connects to the two GPU services over HTTP.

```bash
runai submit-inference wattbot-ui \
    --pvc kohakurag-pvc:/workspace \
    --gpu 0 \
    --cpu 2 --memory 4Gi \
    --image python:3.11-slim \
    --port 8501 \
    --env RAG_MODE=remote \
    --env VLLM_BASE_URL=http://wattbot-vllm:8000/v1 \
    --env VLLM_MODEL=Qwen/Qwen2.5-7B-Instruct \
    --env EMBEDDING_SERVICE_URL=http://wattbot-embedding:8080 \
    --command -- bash -c "
        cd /workspace/kohakurag/repo &&
        pip install -r remote_requirements.txt &&
        pip install -e vendor/KohakuVault &&
        pip install -e vendor/KohakuRAG &&
        streamlit run app.py \
            --server.port=8501 \
            --server.address=0.0.0.0 \
            --server.headless=true
    "
```

**Access:** Open `http://<runai-cluster>:8501` in your browser.

## PVC Layout

```
/workspace/kohakurag/
├── repo/                          # git clone of KohakuRAG_UI
│   ├── app.py
│   ├── vendor/KohakuRAG/
│   ├── vendor/KohakuVault/
│   ├── scripts/embedding_server.py
│   ├── data/
│   │   ├── embeddings/wattbot_jinav4.db   ← vector index
│   │   └── metadata.csv
│   └── remote_requirements.txt
└── hf_cache/                      # shared HuggingFace model cache
    └── hub/
        ├── models--Qwen--Qwen2.5-7B-Instruct/
        └── models--jinaai--jina-embeddings-v4/
```

## Alternative: PVC-Based (No Docker Build)

All three commands above use the PVC-based approach — no custom Docker images
needed. The code and model weights live on the PVC, and `pip install` runs
at container startup. This is simpler for development but adds startup time.

For production, build custom Docker images using the Dockerfiles in `deploy/`:
- `deploy/Dockerfile.embedding` — embedding server with Jina V4 baked in
- `deploy/Dockerfile.streamlit` — Streamlit app with lightweight deps

## Environment Variables Reference

| Variable | Used By | Default | Description |
|---|---|---|---|
| `RAG_MODE` | Streamlit | `bedrock` | Set to `remote` for vLLM mode |
| `VLLM_BASE_URL` | Streamlit | `http://localhost:8000/v1` | vLLM OpenAI-compatible endpoint |
| `VLLM_MODEL` | Streamlit | `default` | Model ID served by vLLM |
| `EMBEDDING_SERVICE_URL` | Streamlit | `http://localhost:8080` | Embedding server base URL |
| `EMBEDDING_MODEL` | Embedding server | `jinaai/jina-embeddings-v4` | HF model ID |
| `EMBEDDING_DIM` | Embedding server | `1024` | Matryoshka truncation dimension |
| `EMBEDDING_TASK` | Embedding server | `retrieval` | Jina task mode |
| `HF_HOME` | vLLM, Embedding | system default | HuggingFace cache directory |

## Query Flow

When a user asks a question in the Streamlit UI:

1. **Streamlit** sends the query text to the **Embedding Server** (`POST /embed`)
2. **Embedding Server** returns a vector using Jina V4
3. **Streamlit** searches the local SQLite vector DB (on PVC) for top-k similar passages
4. **Streamlit** builds a prompt with the retrieved context and sends it to **vLLM** (`POST /v1/chat/completions`)
5. **vLLM** returns the generated answer
6. **Streamlit** displays the answer with citations

## Troubleshooting

### vLLM won't start / OOM
- Check GPU memory allocation: `runai describe job wattbot-vllm`
- Reduce `--max-model-len` or switch to a smaller model
- Ensure `--gpu 0.75` leaves enough for the embedding server

### Embedding server returns 503
- Model is still loading (Jina V4 takes ~30s on first request)
- Check logs: `runai logs wattbot-embedding`

### Streamlit shows "unreachable" for services
- Verify service DNS names match your RunAI job names
- Check that ports 8000 and 8080 are exposed
- Test connectivity: `curl http://wattbot-vllm:8000/health` from the Streamlit pod

### Vector DB not found
- Ensure you ran Step 0 (index build) first
- Confirm the DB path in your config matches the PVC location
- Check PVC mount: `ls /workspace/kohakurag/repo/data/embeddings/`

### Index vs Query embedding mismatch
- The **same model** (`jinaai/jina-embeddings-v4`) and **same dimension**
  (`1024`) must be used at both index-build time and query time
- If you re-index with different settings, restart the embedding server to match
