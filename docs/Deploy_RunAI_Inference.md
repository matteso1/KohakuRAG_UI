# Deploying WattBot RAG on RunAI (Inference Jobs)

Production deployment using 3 RunAI **Inference** workloads on a single
GPU, with vLLM for high-throughput LLM serving.

---

## Architecture Overview

The system has two distinct phases that use embeddings differently:

### Phase 1: Index Build (one-time, batch)

Embeds all documents in the corpus into a vector database. This runs once
(or when the corpus changes) and produces `wattbot_jinav4.db`.

```
┌──────────────────────────────────────────┐
│  Index Build Job (Training or Workspace) │
│                                          │
│  Jina V4 model (loaded locally on GPU)   │
│         │                                │
│         ▼                                │
│  data/corpus/*.json ──► wattbot_jinav4.db│
│                         (written to PVC) │
└──────────────────────────────────────────┘
```

This is a **batch job** — it loads the full Jina V4 model, processes every
document, writes the SQLite vector DB, and exits. Use a RunAI **Training**
or **Workspace** workload for this.

### Phase 2: Query Serving (always-on, 3 Inference jobs)

Handles live user queries. The embedding server here only encodes the
user's question (a few sentences) for vector search — it does NOT
re-embed the corpus.

```
  Users (browser)
       │
       ▼
┌─────────────────────┐
│   Streamlit App     │  Inference job — CPU only, no GPU
│   Port 8501         │  Reads wattbot_jinav4.db from PVC
└────────┬────────────┘
         │ HTTP (internal cluster DNS)
   ┌─────┴──────┐
   ▼            ▼
┌──────────┐  ┌──────────────────────────────┐
│  vLLM    │  │  Embedding Server            │
│  Server  │  │  (query-time only)           │
│  Port    │  │  Encodes user questions into │
│  8000    │  │  vectors for DB lookup       │
│  GPU     │  │  Port 8080, GPU ~0.25        │
│  ~0.75   │  └──────────────────────────────┘
└──────────┘
```

**Query flow:**
1. User types a question in Streamlit
2. Streamlit sends the question text to the **Embedding Server**
3. Embedding Server returns a 1024-dim vector
4. Streamlit searches the pre-built **vector DB** (local SQLite on PVC)
5. Streamlit sends the question + retrieved context to **vLLM**
6. vLLM returns the generated answer
7. Streamlit displays the answer with citations

**Why split it this way?**
- **vLLM** gives continuous batching — 2-4x throughput vs raw HF
- **Embedding server** encodes queries only (tiny workload), shares GPU
- **Streamlit app** is CPU-only — just UI, HTTP calls, SQLite reads
- The vector DB is **read-only** at query time

---

## Prerequisites

1. **PVC** mounted at `/workspace` with the cloned repo and enough disk for model weights
2. **HuggingFace token** for gated models: `export HF_TOKEN="hf_..."`
3. **Internal DNS** for service discovery between RunAI jobs

---

## Step 0: Build the Vector Index (one-time)

Before deploying inference jobs, build the vector database.

**When to run:** First-time setup, or when corpus changes.
**NOT needed when:** Changing LLM model, retrieval settings, or restarting jobs.

### From a Workspace (interactive)

```bash
cd /workspace/KohakuRAG_UI
pip install -e vendor/KohakuVault -e vendor/KohakuRAG -r local_requirements.txt
cd vendor/KohakuRAG
kogine run scripts/wattbot_build_index.py --config configs/jinav4/index.py
```

### As a Training job

| Field | Value |
|-------|-------|
| Image | `pytorch/pytorch:2.4.1-cuda12.4-cudnn9-runtime` |
| GPU | `0.25` |
| PVC | `<your-pvc>:/workspace` |
| Env | `HF_HOME=/workspace/.cache/huggingface` |

---

## Step 1: Deploy the vLLM Server

| Field | Value |
|-------|-------|
| Name | `wattbot-vllm` |
| Image | `vllm/vllm-openai:latest` |
| GPU | `0.75` |
| Port | `8000` |
| PVC | `<your-pvc>:/workspace` |

**Command:**
```bash
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-7B-Instruct \
  --host 0.0.0.0 --port 8000 --max-model-len 8192 --dtype auto
```

**Env:** `HF_HOME=/workspace/.cache/huggingface`

**Verify:** `curl http://wattbot-vllm:8000/v1/models`

---

## Step 2: Deploy the Embedding Server

| Field | Value |
|-------|-------|
| Name | `wattbot-embedding` |
| Image | `pytorch/pytorch:2.4.1-cuda12.4-cudnn9-runtime` |
| GPU | `0.25` |
| Port | `8080` |
| PVC | `<your-pvc>:/workspace` |

**Command:**
```bash
pip install fastapi uvicorn httpx sentence-transformers "transformers>=4.42,<5" accelerate && \
cd /workspace/KohakuRAG_UI && \
pip install -e vendor/KohakuVault -e vendor/KohakuRAG && \
python scripts/embedding_server.py
```

**Env:**
```
HF_HOME=/workspace/.cache/huggingface
EMBEDDING_MODEL=jinaai/jina-embeddings-v4
EMBEDDING_DIM=1024
EMBEDDING_TASK=retrieval
```

**Verify:** `curl http://wattbot-embedding:8080/health`

---

## Step 3: Deploy the Streamlit App

| Field | Value |
|-------|-------|
| Name | `wattbot-app` |
| Image | `python:3.11-slim` |
| GPU | none |
| Port | `8501` |
| PVC | `<your-pvc>:/workspace` |

**Command:**
```bash
pip install streamlit openai httpx numpy python-dotenv && \
cd /workspace/KohakuRAG_UI && \
pip install -e vendor/KohakuVault -e vendor/KohakuRAG && \
streamlit run app.py --server.port=8501 --server.address=0.0.0.0 --server.headless=true
```

**Env:**
```
RAG_MODE=remote
VLLM_BASE_URL=http://wattbot-vllm:8000/v1
VLLM_MODEL=Qwen/Qwen2.5-7B-Instruct
EMBEDDING_SERVICE_URL=http://wattbot-embedding:8080
```

---

## PVC Layout

```
/workspace/
├── KohakuRAG_UI/
│   ├── app.py
│   ├── data/embeddings/wattbot_jinav4.db  ← built in Step 0
│   ├── vendor/KohakuRAG/
│   └── scripts/embedding_server.py
└── .cache/huggingface/hub/
    ├── models--Qwen--Qwen2.5-7B-Instruct/
    └── models--jinaai--jina-embeddings-v4/
```

---

## Changing the LLM Model

1. Update vLLM job's `--model` argument
2. Update Streamlit job's `VLLM_MODEL` env var
3. Restart both jobs — no code changes needed

---

## Troubleshooting

- **vLLM OOM:** Reduce `--max-model-len` or use `--quantization awq`
- **Embedding server 503:** Model still loading (~30s on first request)
- **Streamlit can't connect:** Check service DNS names and ports
- **Vector DB not found:** Run Step 0 first
- **Mismatch errors:** Ensure same `EMBEDDING_DIM=1024` at index and query time

---

## Local Development

```bash
# Default — local GPU models
streamlit run app.py

# Test remote mode locally
# Terminal 1: python scripts/embedding_server.py
# Terminal 2: python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen2.5-7B-Instruct
# Terminal 3: RAG_MODE=remote streamlit run app.py
```
