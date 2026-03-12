# Deploying WattBot RAG on RunAI (Inference Jobs)

Production deployment using 3 RunAI **Inference** workloads across
1.5 GPUs (~90 GB), with vLLM for high-throughput LLM serving.

| Job | What it does | GPU | Port |
|-----|-------------|-----|------|
| **`wattbot-vllm`** | Serves the LLM (Qwen 7B) via vLLM's OpenAI-compatible API | 1.0 | 8000 |
| **`wattbot-embedding`** | Encodes user questions into vectors (Jina V4) for DB lookup | 0.5 | 8080 |
| **`wattbot-app`** | Streamlit UI — connects to the other two via HTTP | 0 | 8501 |

All three mount the shared model repository at `/models/` (read-only)
and share one physical GPU via RunAI's fractional allocation. A one-time
setup Workspace (Step 0) uses your personal workspace at
`/home/jovyan/work/` (writable) to clone the repo, install
dependencies, and build the vector index. Model weights (Qwen, Jina V4)
are already pre-cached on the shared PVC — no downloads needed.

```
  Users (browser)
       │
       ▼
┌─────────────────────┐
│   Streamlit App     │  CPU only, no GPU
│   Port 8501         │
└────────┬────────────┘
         │ HTTP (internal cluster DNS)
   ┌─────┴──────┐
   ▼            ▼
┌──────────┐  ┌──────────────────────────────┐
│  vLLM    │  │  Embedding Server            │
│  Server  │  │  Encodes user questions into │
│  Port    │  │  vectors for DB lookup       │
│  8000    │  │  Port 8080, GPU ~0.5         │
│  GPU     │  └──────────────────────────────┘
│  ~1.0    │
└──────────┘
```

**Query flow:** User asks a question → Streamlit [`wattbot-app`] sends
it to the Embedding Server [`wattbot-embedding`] → gets a vector back →
searches the pre-built vector DB → sends question + retrieved context to
vLLM [`wattbot-vllm`] → Streamlit [`wattbot-app`] displays the answer
with citations.

All steps below use the **RunAI web UI only** — no CLI tools required.

---

## Prerequisites: Create a Project PVC (PPVC) for shared data

The vector index (`wattbot_jinav4.db`, ~130 MB) is built once and read by
every inference job. Instead of copying it into each workspace, create a
**Project PVC (PPVC)** — Run:ai's mechanism for sharing storage across
workloads in the same project.

### What goes on the PPVC

| Directory | Contents | Written by | Read by |
|-----------|----------|------------|---------|
| `embeddings/` | `wattbot_jinav4.db` (~130 MB) | wattbot-setup (Step 0) | wattbot-app, notebooks, benchmarks |
| `corpus/` | Parsed JSON documents | wattbot-setup (Step 0) | Rebuild only |
| `pdfs/` | Downloaded source PDFs | wattbot-setup (Step 0) | Rebuild only |

### Create the Data Volume

In the RunAI UI (v2.23+):

1. Go to **Data & Storage** > **Data Volumes** > **New Data Volume**
2. Configure the data origin:
   - **Scope:** Select your project scope
     (e.g. `runai/doit-ai-cluster/default/<your-project>`)
   - **PVC name:** `wattbot-data` *(enter a new name — this creates
     a new PVC for the data volume)*
3. Set the data volume identity:
   - **Data volume name:** `wattbot-data`
   - **Description:** "Shared vector index, corpus, and PDFs for WattBot RAG"
4. Set scopes:
   - Share with the project so all workloads in the project can mount it
5. Create the Data Volume

> **Note:** The Data Volume wraps an underlying PVC. You don't need to
> create the PVC separately — the Data Volume wizard creates it for you.
> If your cluster requires a specific storage class or access mode,
> check with your admin.

### Mount path convention

All workloads mount this PPVC at **`/wattbot-data`**:

```
/wattbot-data/                    ← PPVC mount point
├── embeddings/
│   └── wattbot_jinav4.db            # vector index
├── corpus/                           # parsed JSON docs
└── pdfs/                             # cached source PDFs
```

When attaching the Data Volume to a workload in the RunAI UI, set:
- **Data volume:** `wattbot-data`
- **Mount path:** `/wattbot-data`
- **Access:** Read-write for `wattbot-setup`, read-only for inference jobs

---

## Step 0: Prepare the Workspace (one-time setup)

Before deploying the Inference jobs, you need the repo cloned,
dependencies installed, and the vector index built. Model weights
(Qwen 7B, Jina V4, and others) are **already pre-cached** on the
shared PVC at `/models/.cache/huggingface/` — no downloads needed.

### Cluster storage layout

| Path | Type | Access | Size | Purpose |
|------|------|--------|------|---------|
| `/models/` | Shared Data Volume | **Read-only** | ~744 GB | Pre-cached model weights (Qwen, Jina V4, etc.) |
| `/wattbot-data/` | **Project PVC** | **RW** (setup) / **RO** (inference) | 1 GB | Vector index, corpus, PDFs — shared across all jobs |
| `/home/jovyan/work/` | Personal workspace | Read-write | 30 GB | Git repo, Python deps, cache |

### 0a. Create a Workspace

In the RunAI UI:

1. Go to **Workloads** > **New Workload** > **Workspace**
2. Set:
   - **Name:** `wattbot-setup`
   - **Image:** `nvcr.io/nvidia/pytorch:25.02-py3`
   - **GPU:** `1.0` (PyTorch + JinaV4 model need most of a GPU's memory)
   - **Data Volumes:**
     - `shared-model-repository` → mount at `/models` (read-only)
     - `wattbot-data` → mount at `/wattbot-data` (**read-write**)
   - **Environment variables:**

     | Key | Value |
     |-----|-------|
     | `HF_HOME` | `/home/jovyan/work/.cache/huggingface` |
     | `HF_HUB_CACHE` | `/home/jovyan/work/.cache/huggingface/hub` |
     | `TRANSFORMERS_CACHE` | `/home/jovyan/work/.cache/huggingface/hub` |

3. Create the Workspace and wait for it to start
4. Click **Connect** > open the **terminal** (JupyterLab or shell)

### 0b. Verify GPU and check shared models

```bash
# Verify GPU is available
nvidia-smi --query-gpu=index,name,memory.total,memory.free \
           --format=csv,noheader

# Confirm model weights are on the shared PVC
ls /models/.cache/huggingface/ | grep models--
# Should list: models--jinaai--jina-embeddings-v4,
#              models--Qwen--Qwen2.5-7B-Instruct, etc.
```

### 0c. Set up cache directories

The shared models PVC is **read-only**, so any new downloads or cache
writes must go to your personal workspace:

```bash
# Create cache directories on writable storage
mkdir -p /home/jovyan/work/.cache/huggingface/hub
mkdir -p /home/jovyan/work/.cache/pip
mkdir -p /home/jovyan/work/.cache/uv
mkdir -p /home/jovyan/work/tmp

# Set environment variables (add to ~/.bashrc for persistence)
export TMPDIR=/home/jovyan/work/tmp
export UV_CACHE_DIR=/home/jovyan/work/.cache/uv
export PIP_CACHE_DIR=/home/jovyan/work/.cache/pip
export HF_HOME=/home/jovyan/work/.cache/huggingface
export HF_HUB_CACHE=/home/jovyan/work/.cache/huggingface/hub
export TRANSFORMERS_CACHE=/home/jovyan/work/.cache/huggingface/hub
```

### 0d. Clone the repo and install dependencies

```bash
cd /home/jovyan/work
git clone https://github.com/qualiaMachine/KohakuRAG_UI.git
cd KohakuRAG_UI

# Switch to the PowerEdge setup branch (has RunAI-specific fixes)
git checkout claude/rag-poweredge-setup-wM2Fz

# Install uv (fast Python package installer, ~10-100x faster than pip)
pip install uv

# Create and activate a virtual environment
# (uses whatever Python the NGC image ships — currently 3.12)
uv venv
source .venv/bin/activate
python --version   # verify venv is active

# Install vendored packages (order matters: KohakuVault before KohakuRAG)
uv pip install -e vendor/KohakuVault
uv pip install -e vendor/KohakuRAG

# Install remaining dependencies
uv pip install -r local_requirements.txt

# Smoke test — verify imports work
python -c "import kohakuvault, kohakurag; print('Imports OK')"

# Register a named Jupyter kernel so you can select this venv in notebooks
python -m ipykernel install --user \
  --name wattbot \
  --display-name "wattbot"
```

> **Note:** Always `source .venv/bin/activate` before running any
> subsequent steps (index build, pipeline test, etc.). This keeps
> dependencies isolated from the container's system Python. In
> JupyterLab, select the **"wattbot"** kernel to use this environment
> in notebooks.

### 0e. Build the vector index (writes to PPVC)

The index build writes directly to the PPVC so all workloads can access
it without copies. We symlink `data/embeddings/` into the PPVC mount so
the build scripts' relative paths still work.

```bash
cd /home/jovyan/work/KohakuRAG_UI

# Create directories on the PPVC
mkdir -p /wattbot-data/embeddings
mkdir -p /wattbot-data/corpus
mkdir -p /wattbot-data/pdfs

# Symlink repo data dirs to the PPVC (so kogine writes to shared storage)
rm -rf data/embeddings data/corpus data/pdfs
ln -s /wattbot-data/embeddings data/embeddings
ln -s /wattbot-data/corpus     data/corpus
ln -s /wattbot-data/pdfs       data/pdfs

# Check if index already exists on the PPVC
if [ -f /wattbot-data/embeddings/wattbot_jinav4.db ]; then
    echo "Index already exists: $(du -h /wattbot-data/embeddings/wattbot_jinav4.db | cut -f1)"
else
    echo "Building vector index (takes a few minutes)..."
    cd vendor/KohakuRAG
    kogine run scripts/wattbot_build_index.py --config configs/jinav4/index.py
    cd ../..
fi

# Verify the index was created
ls -lh /wattbot-data/embeddings/wattbot_jinav4.db
# Should be ~100-130 MB
```

### 0f. Test the full pipeline in a notebook

Before splitting into 3 separate jobs, verify the entire RAG pipeline
works end-to-end in the workspace. The embedding model is already loaded
from the index build, so this is a quick check. Open a **JupyterLab
notebook** and select the **wattbot** kernel we registered in Step 0d
(or run as a Python script with the venv activated) and test:

```python
import os, sys

REPO = "/home/jovyan/work/KohakuRAG_UI"
os.chdir(REPO)
sys.path.insert(0, f"{REPO}/vendor/KohakuRAG/src")
os.environ["HF_HOME"] = "/models/.cache/huggingface"

from kohakurag import RAGPipeline
from kohakurag.datastore import KVaultNodeStore
from kohakurag.embeddings import JinaV4EmbeddingModel
from kohakurag.llm import HuggingFaceLocalChatModel

# 1. Load embedding model (already cached from index build)
embedder = JinaV4EmbeddingModel()
print("Embedding model loaded")

# 2. Load vector index from the PPVC
# NOTE: do NOT pass dimensions= here. If the path is wrong, we want a loud
# error instead of silently creating a new empty DB.
DB = "/wattbot-data/embeddings/wattbot_jinav4.db"
store = KVaultNodeStore(DB, table_prefix="wattbot_jv4")
print(f"Vector index loaded: {len(store._vectors)} chunks")

# 3. Load LLM from shared cache (7B, not 72B!)
chat = HuggingFaceLocalChatModel(
    model="Qwen/Qwen2.5-7B-Instruct",
)
print("LLM loaded")

# 4. Run full pipeline
pipeline = RAGPipeline(embedder=embedder, store=store, chat_model=chat)
answer = await pipeline.answer("How much energy to train an LLM (ballpark)?")
print(f"\nAnswer: {answer['response']}")
print(f"\nTop snippets:")
for s in answer["snippets"][:3]:
    print(f"  - {s.document_title} ({s.node_id})")
```

If this works, you know the models, index, and code are all wired up
correctly. Any issues here are much easier to debug than across 3
separate inference jobs.

### 0g. Check HuggingFace token (for gated models)

If you plan to use gated models (Llama 3, Gemma 2, etc.), you need an
HF token. This is **not needed** for Qwen or Jina V4:

```bash
# Check if token is set
if [ -n "${HF_TOKEN:-}" ]; then
    echo "HF_TOKEN is set"
elif [ -f ~/.cache/huggingface/token ]; then
    echo "Found cached token"
else
    echo "WARNING: No HF token found."
    echo "To fix: export HF_TOKEN='hf_your_token_here'"
fi
```

### 0h. Stop the Workspace

Once the pipeline test passes, you can **stop the Workspace** from the RunAI UI
to free its GPU. The vector index persists on the PPVC (`wattbot-data`) —
Inference jobs mount it directly and don't depend on the Workspace running.

**When to re-run this step:**
- When you add/remove/update documents in `data/corpus/`
- When you change embedding settings (dimension, model)

**NOT needed when:**
- Changing the LLM model (Qwen → Llama, etc.)
- Changing retrieval settings (top_k)
- Restarting Inference jobs

---

## Step 1: Deploy the vLLM Server

Uses the official `vllm/vllm-openai` image — no pip-installing vLLM at
runtime.

> **Why not "Model: from Hugging Face"?** That inference type is a black
> box — crashes produce no logs and it's unclear how arguments are
> passed. It also has a **Model store** field (separate from Data &
> Storage) that expects a specially registered RunAI data source — the
> `shared-models` PVC doesn't qualify and can't be selected. While you
> can still attach `shared-models` as a regular data volume via Advanced
> setup's Data & Storage section, the empty model store likely causes
> the HF type to download the model to ephemeral storage, leading to
> crashes or silent failures. The **Custom** inference type avoids all
> of this: full logs, explicit command/arguments, and straightforward
> PVC mounting — no "model store" abstraction needed.

In the RunAI UI: **Workloads** > **New Workload** > **Inference**

### 1a. Basic settings

| Field | Value |
|-------|-------|
| **Cluster** | `doit-ai-cluster` |
| **Project** | Your project (e.g. `jupyter-endemann01`) |
| **Inference type** | **Custom** |
| **Inference name** | `wattbot-vllm` |

### 1b. Environment image

| Field | Value |
|-------|-------|
| **Image** | Custom image |
| **Image URL** | `vllm/vllm-openai:latest` |
| **Image pull** | Pull the image only if it's not already present on the host (recommended) |
| **Image pull secret** | *(leave empty — public Docker Hub image)* |

### 1c. Serving endpoint

| Field | Value |
|-------|-------|
| **Protocol** | HTTP |
| **Container port** | `8000` |

### 1d. Runtime settings

The `vllm/vllm-openai` image has a built-in entrypoint that launches the
API server — you only need to pass `--model` as an argument. No command
is required.

| Field | Value |
|-------|-------|
| **Command** | *(leave empty — image default launches the API server)* |
| **Arguments** | `--model Qwen/Qwen2.5-7B-Instruct` |
| **Environment variable** | Name: `HF_HOME`, Value: `/models/.cache/huggingface` |
| **Working directory** | *(leave empty)* |

> **Note:** The image defaults to `--host 0.0.0.0` and uses the
> container port from the serving endpoint config. If you need to
> tune memory usage, add `--max-model-len 8192` or `--dtype float16`
> to the Arguments field. For a first deploy, just `--model` is enough.

### 1e. Compute resources

| Field | Value |
|-------|-------|
| **GPU devices** | `1` (full GPU) |
| **GPU fractioning** | *(leave disabled — using full device)* |
| **CPU request** | `4` cores |
| **CPU memory request** | `16 GB` |
| **Replica autoscaling** | Min `1`, Max `1` (no autoscaling) |

### 1f. Data & storage

Under **Data & storage**, select the `shared-models` data volume and
set the container path. (In Custom inference type, data volumes appear
directly in the initial setup form — no need for Advanced setup.)

| Data volume name | Container path |
|------------------|----------------|
| `shared-models` | `/models` |

### 1g. General

| Field | Value |
|-------|-------|
| **Priority** | `very-high` (or as appropriate) |

### Expected startup time

First deploy takes **5-10 minutes**:
- **Image pull** (~2-5 min): The `vllm/vllm-openai` image is ~15 GB.
  Subsequent deploys skip this if the image is cached on the node.
- **Model loading** (~1-2 min): vLLM loads Qwen 7B weights (~14 GB)
  from the shared PVC into GPU memory.
- **Engine warmup** (~30s): vLLM compiles CUDA kernels and initializes
  the KV cache.

You'll see `Initializing` in the RunAI UI during this time — this is
normal. The job transitions to `Running` once the HTTP health check
passes. Subsequent restarts (same node, cached image) take ~2-3 minutes.

### How it works

Qwen 7B weights are already pre-cached on the shared PVC at
`/models/.cache/huggingface/` — vLLM loads them directly on startup
(no download needed). The Data Volume is read-only, so vLLM can't
accidentally modify or delete weights.

> **Note:** vLLM exposes an **OpenAI-compatible** API (`/v1/chat/completions`),
> but it runs **entirely on your local GPU** — no OpenAI account or API
> charges. The `openai` Python package is just used as a client library
> to talk to your local vLLM server.

**Verify (from any other pod's terminal):**
```bash
curl http://wattbot-vllm:8000/v1/models
```

---

## Step 2: Deploy the Embedding Server

The embedding server is a custom FastAPI service that wraps Jina V4.
We use the same `vllm/vllm-openai` image as the vLLM server — it
already has PyTorch, CUDA, and `curl` pre-installed, so only a handful
of lightweight Python packages need to be added at startup.

> **Why the vLLM image?** Using one image for both services means
> fewer image pulls and less storage on each node. The vLLM image
> (~15 GB) ships with PyTorch + CUDA, which is everything the
> embedding server needs. The NGC PyTorch image
> (`nvcr.io/nvidia/pytorch:25.02-py3`, ~20 GB) also works but is
> larger and adds no benefit here.

In the RunAI UI: **Workloads** > **New Workload** > **Inference**

### 2a. Basic settings

| Field | Value |
|-------|-------|
| **Cluster** | `doit-ai-cluster` |
| **Project** | Your project (e.g. `jupyter-endemann01`) |
| **Inference type** | **Custom** (not "Model: from Hugging Face") |
| **Inference name** | `wattbot-embedding` |

### 2b. Environment image

| Field | Value |
|-------|-------|
| **Image** | Custom image |
| **Image URL** | `vllm/vllm-openai:latest` |
| **Image pull** | Pull the image only if it's not already present on the host (recommended) |
| **Image pull secret** | *(leave empty — public Docker Hub image)* |

### 2c. Serving endpoint

| Field | Value |
|-------|-------|
| **Protocol** | HTTP |
| **Container port** | `8080` |

### 2d. Runtime settings

Inference jobs don't have access to the personal workspace
(`/home/jovyan/work/`), so the command downloads the repo as a
tarball and installs dependencies at startup.

| Field | Value |
|-------|-------|
| **Command** | `bash` |
| **Arguments** | `-c "pip install uv && curl -sL https://github.com/qualiaMachine/KohakuRAG_UI/archive/refs/heads/rag-poweredge.tar.gz | tar xz -C /tmp && mv /tmp/KohakuRAG_UI-rag-poweredge /tmp/KohakuRAG_UI && cd /tmp/KohakuRAG_UI && uv pip install --system fastapi uvicorn httpx sentence-transformers 'transformers>=4.42,<5' accelerate && python3 scripts/embedding_server.py"` |
| **Working directory** | *(leave empty)* |

> **Why `curl` tarball instead of `git clone`?** The vLLM image
> doesn't include `git`. Downloading a tarball via `curl` (which is
> pre-installed) avoids needing to install git at runtime.
>
> **Why `python3` not `python`?** The vLLM image provides `python3`
> but does not alias it to `python`.
>
> **Why uv?** `uv` is a drop-in replacement for `pip` that's 10-100x
> faster. Installs that take 1-3 minutes with pip finish in seconds.

**Environment variables:**

| Name | Value |
|------|-------|
| `HF_HOME` | `/models/.cache/huggingface` |
| `EMBEDDING_MODEL` | `jinaai/jina-embeddings-v4` |
| `EMBEDDING_DIM` | `1024` |
| `EMBEDDING_TASK` | `retrieval` |

### 2e. Compute resources

| Field | Value |
|-------|-------|
| **GPU devices** | `1` |
| **GPU fractioning** | Enabled — set to `50%` of device (Jina V4 needs ~3 GB VRAM). The UI will show "0.33–0.66 GPUs" as the allocated range. |
| **CPU request** | `2` cores |
| **CPU memory request** | `8 GB` |
| **Replica autoscaling** | Min `1`, Max `1` (no autoscaling) |

### 2f. Data & storage

Under **Data & storage**, add the data volumes and set container paths:

| Data volume name | Container path |
|------------------|----------------|
| `shared-models` | `/models` |
| `wattbot-data` | `/wattbot-data` |

### 2g. General

| Field | Value |
|-------|-------|
| **Priority** | `very-high` (or as appropriate) |

### Expected startup time

First deploy takes **3-5 minutes**:
- **Image pull** (~2-3 min): If the vLLM image is already cached on
  the node (from Step 1), this is instant. Otherwise ~15 GB download.
- **Dependency install** (~30-60s): `uv` installs FastAPI,
  sentence-transformers, etc. (only a few lightweight packages —
  PyTorch is already in the image).
- **Model loading** (~30s): Jina V4 weights (~3 GB) load from the
  shared PVC into GPU memory.

### How it works

Jina V4 weights are already pre-cached on the shared PVC at
`/models/.cache/huggingface/`. The server loads the model into GPU
memory and exposes a FastAPI endpoint for encoding queries into vectors.

**Verify (from any other pod's terminal):**
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
| Data Volume | `shared-models` (read-only, mount at `/models`) |
| PPVC | `wattbot-data` (read-only, mount at `/wattbot-data`) |

| Field | Value |
|-------|-------|
| **Command** | `bash` |
| **Arguments** | `-c "pip install uv && git clone -b rag-poweredge https://github.com/qualiaMachine/KohakuRAG_UI.git /tmp/KohakuRAG_UI && cd /tmp/KohakuRAG_UI && ln -sf /wattbot-data/embeddings data/embeddings && ln -sf /wattbot-data/corpus data/corpus && uv pip install --system streamlit openai httpx numpy python-dotenv && uv pip install --system vendor/KohakuVault vendor/KohakuRAG && streamlit run app.py --server.port=8501 --server.address=0.0.0.0 --server.headless=true"` |

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

1. **Step 0** — Workspace: clone repo, install deps, build index, then stop
2. **Step 1** — vLLM: loads Qwen from shared cache (~30s)
3. **Step 2** — Embedding server: loads Jina V4 from shared cache (~30s)
4. **Step 3** — Streamlit app: last, needs both services running

GPU budget: **1.5 GPUs** (~90 GB) total — 1.0 for vLLM, 0.5 for
embeddings, 0 for Streamlit. All model weights are pre-cached on
`/models/`. Restarts are fast.

---

## Troubleshooting

- **vLLM OOM:** Reduce `--max-model-len` (e.g., 4096) or use `--quantization awq`
- **Embedding server 503:** Model still loading (~30s on first request). Check logs.
- **Streamlit can't connect:** Verify service DNS names match your job names in the RunAI UI
- **Vector DB not found:** Run Step 0 first — `wattbot_jinav4.db` must exist on the PPVC at `/wattbot-data/embeddings/`. Also verify the PPVC is mounted in your workload.
- **Mismatch errors:** Ensure `EMBEDDING_DIM=1024` matches what was used during index build
- **Job keeps crashing:** Check logs in RunAI UI (click job > Logs tab). Common causes: OOM, missing files, image pull failure

### PVC won't bind / "OriginalPvcNotBound" error

If you create a Data Volume in RunAI and see `OriginalPvcNotBound`, the
underlying PVC hasn't been claimed by any pod yet. Most clusters use
`WaitForFirstConsumer` binding mode, meaning the PVC stays `Pending`
until a workload actually mounts it.

**The fix — create the PVC with your first job:**

1. **Job 1 (e.g., `wattbot-test`):** Create a new workload and
   configure the PVC as part of that job (under **Data & Storage** >
   **New PVC**). When the job starts, the pod claims the PVC and it
   binds automatically.
2. **Next job:** Now go to **Data & Storage** > **Data Volumes** and
   create a Data Volume referencing the already-bound PVC. Attach
   that Data Volume to your next workload — it will mount successfully
   because the PVC is already bound.

**Why this happens:** RunAI's Data Volume wizard creates the PVC
object, but with `WaitForFirstConsumer`, Kubernetes won't actually
bind it to a storage backend until a pod schedules that references
it. Creating the Data Volume *before* any pod uses it leaves the PVC
in a `Pending` state, which RunAI reports as `OriginalPvcNotBound`.
The workaround is to let a job create and claim the PVC first, then
wrap it in a Data Volume afterward.

---
---

# Reference

## How Data Sharing Works (PVCs)

The cluster has three storage areas:

| Path | Type | Access | Size | Purpose |
|------|------|--------|------|---------|
| `/models/` | **Shared Data Volume** | **Read-only** | ~744 GB | Pre-cached model weights (Qwen, Jina V4, etc.) |
| `/wattbot-data/` | **Project PVC (PPVC)** | RW (setup) / RO (inference) | 1 GB | Vector index, corpus, PDFs — shared across all jobs |
| `/home/jovyan/work/` | **Personal workspace** | Read-write | 30 GB | Git repo, Python deps, cache |

```
/models/                                    ← shared Data Volume, read-only
└── .cache/huggingface/
    ├── models--Qwen--Qwen2.5-7B-Instruct/
    ├── models--Qwen--Qwen2.5-14B-Instruct/
    ├── models--Qwen--Qwen2.5-72B-Instruct/
    ├── models--Qwen--Qwen3.5-35B-A3B/
    ├── models--jinaai--jina-embeddings-v4/
    └── ...  (~744 GB total)

/wattbot-data/                          ← PPVC, shared across jobs
├── embeddings/
│   └── wattbot_jinav4.db                   # vector index (~130 MB)
├── corpus/                                 # parsed JSON documents
└── pdfs/                                   # cached source PDFs

/home/jovyan/work/                          ← personal, read-write
├── KohakuRAG_UI/                           # git repo (cloned in Step 0)
│   ├── app.py                              # Streamlit app
│   ├── data/
│   │   ├── metadata.csv                    # document URLs
│   │   ├── embeddings -> /wattbot-data/embeddings  # symlink
│   │   ├── corpus -> /wattbot-data/corpus          # symlink
│   │   └── pdfs -> /wattbot-data/pdfs              # symlink
│   ├── vendor/KohakuRAG/                   # RAG library
│   └── scripts/
│       └── embedding_server.py             # query-time embedding server
└── .cache/
    └── huggingface/                        # for any new model downloads
```

In the RunAI UI, these are exposed as:

| RunAI Name | Mount Point | Access | Used by |
|------------|-------------|--------|---------|
| `shared-model-repository` (Data Source) | `/models` | Read-only | All workloads — pre-cached model weights |
| `wattbot-data` (Project PVC) | `/wattbot-data` | RW for setup, RO for inference | Vector index, corpus, PDFs |
| Personal workspace | `/home/jovyan/work` | Read-write | Step 0 Workspace (clone, build) |

**Key points:**
- Model weights are **already pre-cached** — no need to download Qwen
  or Jina V4
- The shared Data Volume is **read-only** — inference jobs can't
  accidentally modify or delete weights
- The **PPVC** (`wattbot-data`) holds the vector index — all jobs
  mount it, so there's a single copy of the DB (no duplication)
- Your personal workspace persists across workspace restarts
- The Workspace does NOT need to be running for Inference jobs to read
  from the PPVC or shared Data Volume

### What lives where

| Item | Location | Size | Written by |
|------|----------|------|------------|
| Qwen 7B model weights | `/models/.cache/huggingface/` | ~14 GB | Pre-cached on shared Data Volume |
| Qwen 14B, 72B, 3.5-35B, etc. | `/models/.cache/huggingface/` | ~744 GB total | Pre-cached on shared Data Volume |
| Jina V4 model weights | `/models/.cache/huggingface/` | ~3 GB | Pre-cached on shared Data Volume |
| Vector index (`wattbot_jinav4.db`) | `/wattbot-data/embeddings/` | ~130 MB | Step 0 Workspace (on PPVC) |
| Parsed corpus (JSON) | `/wattbot-data/corpus/` | ~50 MB | Step 0 Workspace (on PPVC) |
| Cached PDFs | `/wattbot-data/pdfs/` | ~200 MB | Step 0 Workspace (on PPVC) |
| Git repo clone | `/home/jovyan/work/KohakuRAG_UI/` | ~50 MB | Step 0 Workspace |
| Python packages + cache | `/home/jovyan/work/.cache/` | Varies | Step 0 Workspace |

---

## Data Sources vs Data Volumes

The RunAI UI has two sections under **Data & Storage**: **Data Sources**
and **Data Volumes**. The shared model repository appears as both:

- **Data Sources** shows the underlying PVC (`shared-model-repository`).
  When you attach it to a workload, it mounts at `/models/` as
  **read-only** — all workloads see the same pre-cached models.
- **Data Volumes** shows `shared-models`, a shareable wrapper built on
  top of that same PVC.

Your personal workspace at `/home/jovyan/work/` is separate writable
storage for code, indexes, and caches.

---

## Access Control

### Who can modify the shared PVC?

| Action | Who can do it | How to configure |
|--------|--------------|-----------------|
| Create / delete Data Volumes | **Data Volumes Administrator** role only | **Access** > **Access Rules** > assign the `Data Volumes Administrator` role to specific users |
| Write to the PVC (download models, clone repos) | Anyone who mounts the **Data Source** in a workload | Control by limiting who has access to the project that owns the PVC (`runai/doit-ai-cluster/default/shared-models`) |
| Read shared data (via Data Volume) | Anyone in a project the Data Volume is shared with | Data Volume admin sets sharing scopes |

To restrict who can modify models on the PVC:

1. **Limit project access.** Only users assigned to the
   `shared-models` project can create workloads that mount the
   read-write Data Source. Go to **Access** > **Access Rules** and
   ensure only trusted users have roles in that project.
2. **Use the Data Volume for consumers.** Other projects should mount
   `shared-models` as a **Data Volume** (read-only), not the raw Data
   Source. This prevents accidental writes.
3. **Assign a Data Volumes Administrator.** Only users with the
   `Data Volumes Administrator` role can create, share, or delete Data
   Volumes. Keep this to a small set of admins.

### Preventing accidents on the shared PVC

The shared PVC holds ~744 GB of model weights that are expensive to
re-download. Key safeguard: the shared PVC is mounted **read-only** at
`/models/` for all workloads. Only cluster admins with direct PVC access
can modify model weights.

- **Use a naming convention.** Models live under
  `/models/.cache/huggingface/models--<org>--<name>/`. Don't put
  arbitrary files at the PVC root — keep it organized.
- **Document what's on the PVC.** After adding or removing a model, run
  `du -sh /models/.cache/huggingface/models--*/` and note the change
  so the team knows what's available.

---

## Adding and Changing Models

Model weights are pre-cached on the shared PVC at
`/models/.cache/huggingface/`. The PVC is scoped to the whole cluster
(`runai/doit-ai-cluster`), so any project can access the cached models.

### vLLM compatibility

Not every HuggingFace model works with vLLM. Before choosing a new LLM,
check the [vLLM supported models list](https://docs.vllm.ai/en/latest/models/supported_models/).
Well-supported families include:

- **Qwen** (Qwen2, Qwen2.5, Qwen3, Qwen3.5) — first-class support,
  [official deployment guide](https://qwen.readthedocs.io/en/latest/deployment/vllm.html)
- **Llama** (Llama 2, Llama 3, Llama 3.1, Llama 4)
- **Mistral / Mixtral**
- **Gemma** (Gemma 2)
- **Phi** (Phi-3, Phi-4)

Models that use non-standard architectures or custom generation code
(e.g., some multimodal or retrieval-augmented models) may not be
supported. When in doubt, search the
[vLLM GitHub issues](https://github.com/vllm-project/vllm/issues) for
the model name.

**Quantization:** vLLM supports AWQ and GPTQ quantized models out of
the box (pass `--quantization awq` or `--quantization gptq`). FP8
quantized models (e.g., `Qwen3-8B-FP8`) work on Ada Lovelace / Hopper
GPUs natively, and on Ampere GPUs via FP8 Marlin (vLLM v0.9.0+).

### Adding a new model to the shared PVC

1. Contact your cluster admin — the shared PVC at `/models/` is
   **read-only** for regular workloads. New models must be added by
   someone with write access to the underlying PVC.
2. Once added, verify it's cached from any workspace:
   ```bash
   ls /models/.cache/huggingface/models--<org>--<model-name>/
   ```

### Swapping the LLM (e.g., Qwen 7B → Llama 3 8B)

1. Make sure the new model is on the PVC (see above)
2. In the RunAI UI, edit the `wattbot-vllm` job's command: change `--model`
3. Edit the `wattbot-app` job's env var: change `VLLM_MODEL` to match
4. Restart both jobs

No code changes needed. The embedding model and vector DB are unchanged.

### Swapping the embedding model

Changing the embedding model requires rebuilding the vector index:

1. Download the new model to the PVC (see above)
2. Update the index build config and re-run Step 0e
3. Update `wattbot-embedding` env vars (`EMBEDDING_MODEL`, `EMBEDDING_DIM`)
4. Restart the embedding server

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
resource profiles, so RunAI can allocate fractional GPU to each (`1.0`
for vLLM, `0.5` for embeddings) across the available 1.5 GPUs (~90 GB).
The Streamlit app gets `0` GPU — just CPU and RAM.

### Alternatives considered

| Option | What's in each job | Pros | Cons |
|--------|-------------------|------|------|
| Single Workspace job | Everything in one process (LLM + embeddings + UI) | Simple | Not persistent, no batching, wastes GPU on UI |
| Two jobs | **Job 1:** vLLM (LLM only) — **Job 2:** Streamlit + Jina V4 embeddings bundled together | Fewer moving parts | Job 2 needs GPU for Jina V4, can't use lightweight `python:3.11-slim` image |
| **Three jobs (chosen)** | **Job 1:** vLLM (LLM, 1.0 GPU) — **Job 2:** Jina V4 (embeddings, 0.5 GPU) — **Job 3:** Streamlit (UI, CPU-only) | Best resource efficiency, independent scaling | More services to configure |

### Index Build vs Query Serving

The system has two distinct phases that use embeddings differently:

**Phase 1: Index Build (one-time, batch)** — Embeds all documents in
the corpus into a vector database. This runs once (or when the corpus
changes) and produces `wattbot_jinav4.db`. Use a RunAI **Workspace**
for this (Step 0).

**Phase 2: Query Serving (always-on, 3 Inference jobs)** — Handles
live user queries. The embedding server only encodes the user's
question (a few sentences) for vector search — it does NOT re-embed
the corpus.

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
