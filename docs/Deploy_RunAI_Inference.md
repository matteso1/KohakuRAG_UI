# Deploying WattBot RAG on RunAI (Inference Jobs)

Production deployment using 3 RunAI **Inference** workloads on a single
GPU, with vLLM for high-throughput LLM serving.

The three services:

| Job | What it does | GPU | Port |
|-----|-------------|-----|------|
| **`wattbot-vllm`** | Serves the LLM (Qwen 7B) via vLLM's OpenAI-compatible API | 0.75 | 8000 |
| **`wattbot-embedding`** | Encodes user questions into vectors (Jina V4) for DB lookup | 0.25 | 8080 |
| **`wattbot-app`** | Streamlit UI — connects to the other two via HTTP | 0 | 8501 |

All three share a PVC (persistent network disk) and run on a single
physical GPU using RunAI's fractional GPU allocation.

All steps below use the **RunAI web UI only** — no CLI tools required.

---

## Why This Architecture?

RunAI offers three workload types: **Workspace** (interactive dev),
**Training** (batch jobs), and **Inference** (always-on serving). We use
Inference because we want WattBot available as a persistent service — not
something that has to be manually launched each time.

### Why vLLM?

Standard HuggingFace `model.generate()` processes one request at a time —
if two users send a question simultaneously, one blocks until the other
finishes. This is fine for a single developer but breaks down for any
multi-user deployment like a RAG system.

**vLLM** solves this with two key innovations:

- **Continuous batching** — instead of waiting for one request to finish
  before starting the next, vLLM dynamically groups incoming requests
  into GPU batches. Multiple users get served concurrently on a single
  GPU, typically 2-4x more throughput than naive generation.
- **PagedAttention** — LLM inference is bottlenecked by the KV cache
  (key-value memory that grows with sequence length). Standard frameworks
  pre-allocate worst-case memory per request, wasting 60-80% of GPU RAM.
  PagedAttention manages KV cache like virtual memory pages — allocating
  only what's actually needed and sharing common prefixes across requests.
  This means vLLM can serve **3-5x more concurrent requests** in the same
  GPU memory compared to naive HuggingFace serving.

For a RAG system where multiple users may query at once, each with
different context lengths, this memory efficiency is critical.

**What vLLM replaces (and what it doesn't):** vLLM only handles the
"run the LLM on the GPU" part. It exposes an OpenAI-compatible API
(`/v1/chat/completions`) that our code calls over HTTP. Everything
else — the RAG pipeline, retrieval, context assembly, prompt
construction, embedding search — is still our custom KohakuRAG code.
We wrote `VLLMChatModel` (in `kohakurag/remote.py`) as a thin client
that sends our assembled prompts to vLLM and gets completions back.
Think of vLLM as replacing `model.generate()`, not replacing our RAG
logic.

### Why not a single monolithic Inference job?

You *could* bundle vLLM + Jina V4 + Streamlit into one container — and
it would technically work. But splitting them out has practical benefits:

- **Wasted GPU on the UI.** Streamlit is pure Python/CPU. In a monolith,
  RunAI allocates GPU to the whole container even though the UI never
  touches it. Splitting lets the Streamlit job request 0 GPU.
- **Rigid scaling.** With a monolith you can't independently restart the
  LLM (e.g. to swap from Qwen 7B to a larger model) without also
  killing the UI and losing user sessions. Separate jobs let you restart
  one without affecting the others.
- **Simpler containers.** The Streamlit app only needs `pip install
  streamlit openai httpx` — a tiny image. A monolith needs PyTorch,
  vLLM, and Jina V4 all in one image, which is harder to build and
  debug.

### Why three jobs instead of two?

A natural simplification is two jobs: **Job 1** runs vLLM (LLM only),
and **Job 2** bundles the Streamlit UI with Jina V4 embeddings together.
Fewer moving parts, but now Job 2 needs a GPU for Jina V4 (~3 GB VRAM),
so you can't use a lightweight CPU-only image — you'd need a full
PyTorch + CUDA container just for the UI pod.

By splitting into three — vLLM, embedding server, Streamlit — each job
gets exactly the resources it needs. The embedding model (Jina V4,
~3 GB VRAM) and the LLM (Qwen 7B, ~6-14 GB) have very different
resource profiles, so RunAI can allocate fractional GPU to each (`0.75`
for vLLM, `0.25` for embeddings) sharing one physical GPU efficiently.
The Streamlit app gets `0` GPU — just CPU and RAM.

### Alternatives considered

| Option | What's in each job | Pros | Cons |
|--------|-------------------|------|------|
| Single Workspace job | Everything in one process (LLM + embeddings + UI) | Simple | Not persistent, no batching, wastes GPU on UI |
| Two jobs | **Job 1:** vLLM (LLM only) — **Job 2:** Streamlit + Jina V4 embeddings bundled together | Fewer moving parts | Job 2 needs GPU for Jina V4, can't use lightweight `python:3.11-slim` image |
| **Three jobs (chosen)** | **Job 1:** vLLM (LLM) — **Job 2:** Jina V4 (embeddings) — **Job 3:** Streamlit (UI, CPU-only) | Best resource efficiency, independent scaling | More services to configure |

---

## How Data Sharing Works (PVC)

All workloads share data through a **PVC** (Persistent Volume Claim) — a
network disk that any job can mount. Think of it as a shared drive.

```
PVC: "wattbot-pvc"  (a network disk that persists across jobs)
     │
     ├── Workspace mounts at /workspace  → you clone repo + build index here
     ├── vLLM job mounts at /workspace   → reads model weights from cache
     ├── Embedding job mounts /workspace  → reads model weights from cache
     └── Streamlit job mounts /workspace  → reads vector DB + code
```

**Key points:**
- You set up the PVC **once** in the RunAI UI (Data Sources section)
- Every workload you create can attach that same PVC
- Files written by one job are immediately visible to all others
- Data persists even when jobs are stopped or deleted
- The Workspace does NOT need to be running for Inference jobs to read its files

### Data Sources vs Data Volumes

The RunAI UI has two sections under **Data & Storage**: **Data Sources** and
**Data Volumes**. Use **Data Sources** — it's the general-purpose option that
lets you create a new PVC directly in the UI. Data Volumes are a higher-level
wrapper for cross-project sharing with dedicated admin permissions; we don't
need that since all our jobs live in the same project.

### Creating the Data Source (one-time)

1. Go to **Assets** > **Data Sources** > **+ New Data Source**
2. Set:
   - **Scope:** your project (e.g. `runai/doit-ai-cluster/default/<your-project>`)
   - **Name:** `wattbot-pvc`
   - **Type:** PVC
   - **PVC:** select **New PVC**
   - **Storage class:** `local-path` (or whatever your cluster provides)
   - **Access mode:** ReadWriteMany (so multiple jobs can mount it)
   - **Claim size:** `50` GB (Qwen 7B ~14 GB + Jina V4 ~3 GB + index ~30 MB + headroom)
   - **Volume mode:** Filesystem
   - **Container path:** `/workspace`
3. Click **Create**

Once created, this data source appears in the **Data Sources** dropdown
when creating any workload. Every workload (Workspace, vLLM, embeddings,
Streamlit) should attach **the same** `wattbot-pvc` data source mounted
at `/workspace`.

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

---

## Step 0: Prepare the PVC (one-time setup)

Before deploying anything, you need code and data on the shared PVC.
Since RunAI has no file upload UI, the easiest approach is a **Workspace**.

### 0a. Create a Workspace

In the RunAI UI:

1. Go to **Workloads** > **New Workload** > **Workspace**
2. Set:
   - **Name:** `wattbot-setup`
   - **Image:** `pytorch/pytorch:2.4.1-cuda12.4-cudnn9-runtime`
   - **GPU:** `0.25` (needed for index build)
   - **Data Sources:** select the `wattbot-pvc` data source you created above
   - **Environment:** `HF_HOME=/workspace/.cache/huggingface`
3. Create the Workspace and wait for it to start
4. Click **Connect** > open the **terminal** (JupyterLab or shell)

### 0b. Clone the repo and build the index

In the Workspace terminal:

```bash
# Clone the repo onto the PVC
cd /workspace
git clone https://github.com/qualiaMachine/KohakuRAG_UI.git
cd KohakuRAG_UI

# Install dependencies
pip install -e vendor/KohakuVault -e vendor/KohakuRAG -r local_requirements.txt

# Build the vector index (embeds all documents — takes a few minutes)
cd vendor/KohakuRAG
kogine run scripts/wattbot_build_index.py --config configs/jinav4/index.py
cd ../..

# Verify the index was created
ls -lh data/embeddings/wattbot_jinav4.db
# Should be ~30+ MB
```

### 0c. Stop the Workspace

Once the index is built, you can **stop the Workspace** from the RunAI UI
to free its GPU. The files persist on the PVC — the Inference jobs will
read them.

**When to re-run this step:**
- When you add/remove/update documents in `data/corpus/`
- When you change embedding settings (dimension, model)

**NOT needed when:**
- Changing the LLM model (Qwen → Llama, etc.)
- Changing retrieval settings (top_k)
- Restarting Inference jobs

---

## Step 1: Deploy the vLLM Server

In the RunAI UI: **Workloads** > **New Workload** > **Inference**

| Field | Value |
|-------|-------|
| Name | `wattbot-vllm` |
| Image | `vllm/vllm-openai:latest` |
| GPU | `0.75` |
| CPU | `4` |
| Memory | `16Gi` |
| Port | `8000` |
| Data Sources | `wattbot-pvc` (mount at `/workspace`) |

**Command:**
```bash
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-7B-Instruct \
  --host 0.0.0.0 --port 8000 --max-model-len 8192 --dtype auto
```

**Environment variables:**
| Key | Value |
|-----|-------|
| `HF_HOME` | `/workspace/.cache/huggingface` |

First startup downloads model weights (~14 GB) to the PVC cache.
Subsequent restarts use the cache and start in seconds.

**Verify (from any other pod's terminal):**
```bash
curl http://wattbot-vllm:8000/v1/models
```

---

## Step 2: Deploy the Embedding Server

In the RunAI UI: **Workloads** > **New Workload** > **Inference**

| Field | Value |
|-------|-------|
| Name | `wattbot-embedding` |
| Image | `pytorch/pytorch:2.4.1-cuda12.4-cudnn9-runtime` |
| GPU | `0.25` |
| CPU | `2` |
| Memory | `8Gi` |
| Port | `8080` |
| Data Sources | `wattbot-pvc` (mount at `/workspace`) |

**Command:**
```bash
pip install fastapi uvicorn httpx sentence-transformers "transformers>=4.42,<5" accelerate && \
cd /workspace/KohakuRAG_UI && \
pip install -e vendor/KohakuVault -e vendor/KohakuRAG && \
python scripts/embedding_server.py
```

**Environment variables:**
| Key | Value |
|-----|-------|
| `HF_HOME` | `/workspace/.cache/huggingface` |
| `EMBEDDING_MODEL` | `jinaai/jina-embeddings-v4` |
| `EMBEDDING_DIM` | `1024` |
| `EMBEDDING_TASK` | `retrieval` |

First startup downloads Jina V4 (~3 GB) to the PVC cache. The model
takes ~30 seconds to load before it starts serving requests.

**Verify:**
```bash
curl http://wattbot-embedding:8080/health
# {"status": "ok"}
```

---

## Step 3: Deploy the Streamlit App

In the RunAI UI: **Workloads** > **New Workload** > **Inference**

| Field | Value |
|-------|-------|
| Name | `wattbot-app` |
| Image | `python:3.11-slim` |
| GPU | `0` (none) |
| CPU | `1` |
| Memory | `2Gi` |
| Port | `8501` |
| Data Sources | `wattbot-pvc` (mount at `/workspace`) |

**Command:**
```bash
pip install streamlit openai httpx numpy python-dotenv && \
cd /workspace/KohakuRAG_UI && \
pip install -e vendor/KohakuVault -e vendor/KohakuRAG && \
streamlit run app.py --server.port=8501 --server.address=0.0.0.0 --server.headless=true
```

**Environment variables:**
| Key | Value |
|-----|-------|
| `RAG_MODE` | `remote` |
| `VLLM_BASE_URL` | `http://wattbot-vllm:8000/v1` |
| `VLLM_MODEL` | `Qwen/Qwen2.5-7B-Instruct` |
| `EMBEDDING_SERVICE_URL` | `http://wattbot-embedding:8080` |

**Access the app:** Check the RunAI UI for the ingress URL. Typically:
```
https://<cluster-url>/<project>/wattbot-app/proxy/8501/
```

---

## Deployment Order

1. **Step 0** — Workspace: clone repo + build index on PVC, then stop
2. **Step 1** — vLLM: takes longest to start (downloads + loads model)
3. **Step 2** — Embedding server: also downloads Jina V4 on first run
4. **Step 3** — Streamlit app: last, needs both services running

After first run, all model weights are cached on the PVC. Restarts are fast.

---

## PVC Layout

After setup, the shared PVC looks like:

```
/workspace/
├── KohakuRAG_UI/                          # git repo (cloned in Step 0)
│   ├── app.py                             # Streamlit app
│   ├── data/
│   │   ├── corpus/                        # source documents
│   │   ├── metadata.csv                   # document URLs
│   │   └── embeddings/
│   │       └── wattbot_jinav4.db          # vector index (built in Step 0)
│   ├── vendor/KohakuRAG/                  # RAG library
│   └── scripts/
│       └── embedding_server.py            # query-time embedding server
└── .cache/
    └── huggingface/hub/                   # auto-downloaded model weights
        ├── models--Qwen--Qwen2.5-7B-Instruct/
        └── models--jinaai--jina-embeddings-v4/
```

---

## Changing the LLM Model

To swap models (e.g., Qwen 7B to Llama 3 8B):

1. In the RunAI UI, edit the `wattbot-vllm` job's command: change `--model`
2. Edit the `wattbot-app` job's env var: change `VLLM_MODEL` to match
3. Restart both jobs

No code changes needed. The embedding model and vector DB are unchanged.

---

## Troubleshooting

- **vLLM OOM:** Reduce `--max-model-len` (e.g., 4096) or use `--quantization awq`
- **Embedding server 503:** Model still loading (~30s on first request). Check logs.
- **Streamlit can't connect:** Verify service DNS names match your job names in the RunAI UI
- **Vector DB not found:** Run Step 0 first — the file `data/embeddings/wattbot_jinav4.db` must exist on the PVC
- **Mismatch errors:** Ensure `EMBEDDING_DIM=1024` matches what was used during index build
- **Job keeps crashing:** Check logs in RunAI UI (click job > Logs tab). Common causes: OOM, missing files, image pull failure

---

## Local Development

You can still run everything locally for development:

```bash
# Default — local GPU models
streamlit run app.py

# Test remote mode against local services:
# Terminal 1: python scripts/embedding_server.py
# Terminal 2: python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen2.5-7B-Instruct
# Terminal 3: RAG_MODE=remote streamlit run app.py
```
