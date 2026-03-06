# Deploying WattBot RAG on RunAI (Inference Jobs)

Production deployment using 3 RunAI **Inference** workloads on a single
GPU, with vLLM for high-throughput LLM serving.

The three services:

| Job | What it does | GPU | Port |
|-----|-------------|-----|------|
| **`wattbot-vllm`** | Serves the LLM (Qwen 7B) via vLLM's OpenAI-compatible API | 0.75 | 8000 |
| **`wattbot-embedding`** | Encodes user questions into vectors (Jina V4) for DB lookup | 0.25 | 8080 |
| **`wattbot-app`** | Streamlit UI — connects to the other two via HTTP | 0 | 8501 |

All three mount the existing `shared-model-repository` data source (a
PVC that already contains Qwen model weights) and run on a single
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

All workloads share data through the **`shared-model-repository`** data
source — a PVC (Persistent Volume Claim) that already exists on your
cluster. Think of it as a shared network drive that any job can mount.

```
shared-model-repository  (PVC — already exists on your cluster)
     │
     ├── Workspace mounts at /workspace  → you clone repo + build index here
     ├── vLLM job mounts at /workspace   → reads model weights from cache
     ├── Embedding job mounts /workspace  → reads Jina V4 weights from cache
     └── Streamlit job mounts /workspace  → reads vector DB + code
```

**Key points:**
- `shared-model-repository` already exists (visible under **Assets** >
  **Data Sources** in the RunAI UI) — no need to create a new PVC
- Every workload you create can attach this same data source
- Files written by one job are immediately visible to all others
- Data persists even when jobs are stopped or deleted
- The Workspace does NOT need to be running for Inference jobs to read
  its files

### What lives on the PVC

| Item | Size | Written by |
|------|------|------------|
| Qwen 7B model weights | ~14 GB | Already on PVC |
| Jina V4 model weights | ~3 GB | Downloaded in Step 0b |
| Git repo clone | ~50 MB | Step 0 Workspace |
| Vector index (`wattbot_jinav4.db`) | ~30 MB | Step 0 Workspace |

### Data Sources vs Data Volumes

The RunAI UI has two sections under **Data & Storage**: **Data Sources**
and **Data Volumes**. You'll notice `shared-model-repository` appears in
**both** — this is expected:

- **Data Sources** shows the underlying PVC (`shared-model-repository`).
  This is the raw storage. When you attach it to a workload, you get
  **read-write** access.
- **Data Volumes** shows `shared-models`, a shareable wrapper built on
  top of that same PVC. When other projects mount the Data Volume, they
  get **read-only** access.

Same PVC, two views. We attach the **Data Source** version (read-write)
so our workloads can write model weights, clone repos, and build indexes.

### Access control: who can modify the shared PVC?

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

The shared PVC holds model weights that are expensive to re-download.
A few safeguards:

- **Use a naming convention.** Models live under
  `.cache/huggingface/hub/models--<org>--<name>/`. Don't put arbitrary
  files at the PVC root — keep it organized.
- **Don't run `rm -rf` on `/workspace/.cache`.** If a model needs to be
  removed, delete only the specific model directory:
  ```bash
  # Safe: remove one specific model
  rm -rf /workspace/.cache/huggingface/hub/models--<org>--<model-name>/

  # DANGEROUS: never do this
  rm -rf /workspace/.cache/
  ```
- **Read-only for consumers.** Projects that only need to *use* models
  should mount the **Data Volume** (read-only), not the Data Source.
  Only the setup/admin project needs write access.
- **Document what's on the PVC.** After adding or removing a model, run
  `du -sh /workspace/.cache/huggingface/hub/models--*/` and note the
  change so the team knows what's available.

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

Before deploying anything, you need the repo, vector index, and Jina V4
model weights on the shared PVC. Since RunAI has no file upload UI, the
easiest approach is a **Workspace**.

### 0a. Create a Workspace

In the RunAI UI:

1. Go to **Workloads** > **New Workload** > **Workspace**
2. Set:
   - **Name:** `wattbot-setup`
   - **Image:** `pytorch/pytorch:2.4.1-cuda12.4-cudnn9-runtime`
   - **GPU:** `0.25` (needed for index build)
   - **Data Sources:** select `shared-model-repository`, mount at `/workspace`
   - **Environment:** `HF_HOME=/workspace/.cache/huggingface`
3. Create the Workspace and wait for it to start
4. Click **Connect** > open the **terminal** (JupyterLab or shell)

### 0b. Download Jina V4 model weights

Qwen 7B is already on the PVC. Jina V4 needs to be downloaded once:

```bash
# Download Jina V4 to the shared HF cache
pip install sentence-transformers "transformers>=4.42,<5"
python -c "
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('jinaai/jina-embeddings-v4', trust_remote_code=True)
print('Jina V4 downloaded successfully')
"

# Verify (~3 GB in the HF cache)
du -sh /workspace/.cache/huggingface/hub/models--jinaai--jina-embeddings-v4/
```

### 0c. Clone the repo and build the index

```bash
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

### 0d. Stop the Workspace

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
| Data Sources | `shared-model-repository` (mount at `/workspace`) |

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

Qwen 7B weights are already on `shared-model-repository` — vLLM loads
them from the HF cache on startup (no download needed).

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
| Data Sources | `shared-model-repository` (mount at `/workspace`) |

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

Jina V4 weights are already on `shared-model-repository` (downloaded in
Step 0b). The model takes ~30 seconds to load before it starts serving.

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
| Data Sources | `shared-model-repository` (mount at `/workspace`) |

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

1. **Step 0** — Workspace: download Jina V4, clone repo, build index, then stop
2. **Step 1** — vLLM: loads Qwen from cache (~30s)
3. **Step 2** — Embedding server: loads Jina V4 from cache (~30s)
4. **Step 3** — Streamlit app: last, needs both services running

All model weights are already cached on `shared-model-repository`.
Restarts are fast.

---

## PVC Layout

After setup, `shared-model-repository` looks like:

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

## Adding and Changing Models

Model weights live on `shared-model-repository` under
`/workspace/.cache/huggingface/hub/`. The `shared-model-repository` PVC
is backed by a **Data Volume** (`shared-models`) scoped to the whole
cluster (`runai/doit-ai-cluster`), so any project can access the cached
models.

### Adding a new model to the shared PVC

1. Start any Workspace that mounts `shared-model-repository` at `/workspace`
2. In the terminal, download the model:
   ```bash
   export HF_HOME=/workspace/.cache/huggingface
   # For a HuggingFace model (vLLM-compatible):
   pip install huggingface_hub
   huggingface-cli download <org>/<model-name>
   # Example: huggingface-cli download meta-llama/Llama-3-8B-Instruct
   ```
3. Verify it's cached:
   ```bash
   ls /workspace/.cache/huggingface/hub/models--<org>--<model-name>/
   ```
4. Stop the Workspace — the weights persist on the PVC

### Swapping the LLM (e.g., Qwen 7B → Llama 3 8B)

1. Make sure the new model is on the PVC (see above)
2. In the RunAI UI, edit the `wattbot-vllm` job's command: change `--model`
3. Edit the `wattbot-app` job's env var: change `VLLM_MODEL` to match
4. Restart both jobs

No code changes needed. The embedding model and vector DB are unchanged.

### Swapping the embedding model

Changing the embedding model requires rebuilding the vector index:

1. Download the new model to the PVC (see above)
2. Update the index build config and re-run Step 0c
3. Update `wattbot-embedding` env vars (`EMBEDDING_MODEL`, `EMBEDDING_DIM`)
4. Restart the embedding server

---

## Troubleshooting

- **vLLM OOM:** Reduce `--max-model-len` (e.g., 4096) or use `--quantization awq`
- **Embedding server 503:** Model still loading (~30s on first request). Check logs.
- **Streamlit can't connect:** Verify service DNS names match your job names in the RunAI UI
- **Vector DB not found:** Run Step 0 first — the file `data/embeddings/wattbot_jinav4.db` must exist on `shared-model-repository`
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
