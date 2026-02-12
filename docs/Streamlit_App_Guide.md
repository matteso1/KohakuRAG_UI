# Streamlit App Guide

How to run the WattBot RAG interactive app, what it does under the hood,
and how to deploy it on the PowerEdge via Run:ai.

---

## 1) Quick start

```bash
# From the repo root, with venv active
streamlit run app.py
```

The app opens at `http://localhost:8501` (or the next free port).
On a remote server, forward the port:

```bash
# From your laptop
ssh -L 8501:localhost:8501 user@poweredge
```

### Prerequisites

Everything in `local_requirements.txt` must be installed — the same deps
used for benchmarking. The key additions for the app are `streamlit` and
`python-dotenv` (both already listed).

```bash
uv pip install -r local_requirements.txt
```

You also need the vector database built (same as for experiments):

```bash
cd vendor/KohakuRAG
kogine run scripts/wattbot_build_index.py --config configs/jinav4/index.py
cd ../..
ls -lh data/embeddings/wattbot_jinav4.db   # verify
```

---

## 2) App overview

The app provides a chat interface backed by the full RAG pipeline
(retrieval + LLM generation). It supports two modes:

### Single model

Pick one model config from the sidebar, ask questions, get answers.
The model, embedder, and vector store are loaded once and cached across
Streamlit reruns via `st.cache_resource`.

### Ensemble

Pick 2+ models. The app queries all of them and aggregates answers using
a voting strategy. Two execution modes are selected automatically based
on available GPU memory:

| Execution mode | When chosen | How it works |
|----------------|-------------|--------------|
| **Parallel**   | All models fit in total free VRAM | All LLMs loaded into memory at once; each query runs through all models |
| **Sequential** | Not enough VRAM for all simultaneously | Loads one model at a time, queries it, unloads (gc + `cuda.empty_cache()`), loads next |

The sidebar shows detected GPU info (count, free VRAM) and which
execution mode was selected.

**Aggregation strategies:**

| Strategy           | Behavior |
|--------------------|----------|
| `majority`         | Most common answer wins (ties go to first model in list) |
| `first_non_blank`  | First model that returns a non-`is_blank` answer wins |

References are aggregated as the union across all models.

---

## 3) Architecture and key files

```
app.py                                   # Streamlit app entry point
vendor/KohakuRAG/src/kohakurag/
├── pipeline.py                          # RAGPipeline — retrieval + structured QA
├── llm.py                               # HuggingFaceLocalChatModel (4-bit, bf16, etc.)
├── embeddings.py                        # JinaV4EmbeddingModel (1024-dim)
├── datastore.py                         # KVaultNodeStore (SQLite vector DB)
└── types.py                             # ContextSnippet, RetrievalMatch, etc.
vendor/KohakuRAG/configs/hf_*.py         # Model configs (model ID, retrieval params)
data/embeddings/wattbot_jinav4.db        # Vector index (built locally, gitignored)
local_requirements.txt                   # Python dependencies
```

### Data flow per question

```
User question
  → JinaV4EmbeddingModel.embed()         # embed the query
  → KVaultNodeStore.search()              # vector search in jinav4.db
  → RAGPipeline.run_qa()                  # build prompt with retrieved context
  → HuggingFaceLocalChatModel.complete()  # local LLM inference
  → StructuredAnswer                      # parsed JSON answer
```

### What gets cached (stays in memory across questions)

| Resource | Cache key | Approx VRAM |
|----------|-----------|-------------|
| LLM (single) | `(config_name, precision)` | 2–40 GB depending on model |
| Embedder (Jina V4) | shared | ~3 GB |
| Vector store | shared | ~0 (mmap'd SQLite) |
| LLM (ensemble parallel) | `(tuple(config_names), precision)` | sum of all models |

Changing the model selection or precision in the sidebar triggers a full
reload. Changing only `top_k` or asking a new question does **not** reload.

### VRAM estimates (4-bit NF4)

These are hardcoded in `app.py:VRAM_4BIT_GB` for planning purposes:

| Config | VRAM |
|--------|------|
| hf_qwen1_5b | ~2 GB |
| hf_qwen3b | ~3 GB |
| hf_qwen7b | ~6 GB |
| hf_qwen14b | ~10 GB |
| hf_qwen32b | ~20 GB |
| hf_qwen72b | ~40 GB |
| hf_llama3_8b | ~6 GB |
| hf_gemma2_9b | ~7 GB |
| hf_gemma2_27b | ~17 GB |
| hf_mixtral_8x7b | ~26 GB |
| hf_mistral7b | ~6 GB |
| hf_phi3_mini | ~3 GB |

For bf16, multiply by ~4x. The app accounts for this when planning
parallel vs sequential ensemble execution.

---

## 4) Configuration

The app reads the same `hf_*.py` config files used by the benchmark
scripts. Config files define model identity only — precision is selected
in the sidebar at runtime.

Key config fields consumed by the app:

| Field | Purpose |
|-------|---------|
| `hf_model_id` | HuggingFace model identifier |
| `hf_max_new_tokens` | Max generation length (default 512) |
| `hf_temperature` | Sampling temperature (default 0.2) |
| `max_concurrent` | Max concurrent inference requests |
| `db` | Path to vector database |
| `table_prefix` | Vector table prefix |
| `embedding_dim` | Embedding dimension (1024 for jinav4) |
| `embedding_task` | Embedding task type ("retrieval") |

To add a new model to the app, just create a new `hf_*.py` config — the
app auto-discovers all `hf_*.py` files in `vendor/KohakuRAG/configs/`.

---

## 5) Gated models

Some models (Llama 3.1, Gemma 2) are gated on HuggingFace and require:

1. Accept the model license on the model's HuggingFace page
2. Set your HuggingFace token:

```bash
export HF_TOKEN="hf_your_token_here"
streamlit run app.py
```

If you see a 401 error when selecting a gated model, this is why.

---

## 6) Running on PowerEdge via Run:ai + jupyter-server-proxy

The app runs inside a Run:ai interactive workspace with JupyterLab.
Traffic from the browser reaches Streamlit through the chain:

```
Browser  →  nginx (deepthought)  →  JupyterLab  →  jupyter-server-proxy  →  Streamlit (:8501)
```

`jupyter-server-proxy` is a JupyterLab extension that forwards
`/proxy/<port>/` requests to `localhost:<port>` inside the pod. This
is what makes Streamlit accessible without a separate ingress or
nodeport.

### 6.1) Workspace startup args

When creating (or editing) the Run:ai workspace, set the container
command to install both `jupyterlab` and `jupyter-server-proxy` in the
**system** Python before starting Jupyter:

```
-lc "python -m pip install -U jupyterlab jupyter-server-proxy && jupyter lab \
  --ip=0.0.0.0 --port=8888 --no-browser \
  --ServerApp.root_dir=/workspace2 \
  --ServerApp.allow_remote_access=True \
  --ServerApp.trust_xheaders=True \
  --ServerApp.default_url=/lab \
  --ServerApp.base_url=/doit-ai-eval/<WORKSPACE-NAME>/"
```

Replace `<WORKSPACE-NAME>` with the actual Run:ai workspace name
(e.g. `endemann-pytorch22`). This must match exactly — a mismatch
(e.g. using `pytorch21` when the workspace is `pytorch22`) will
produce a **503** from nginx.

### 6.2) Launch Streamlit

After completing the environment setup (see `docs/Setup_PowerEdge.md`),
start Streamlit **without** `--server.baseUrlPath`:

```bash
streamlit run app.py --server.port 8501 --server.address 0.0.0.0 \
  --server.enableCORS=false --server.enableXsrfProtection=false
```

Do **not** pass `--server.baseUrlPath`. The proxy handles path
rewriting transparently — Streamlit should think it is serving from `/`.
Adding a base URL path causes a **404** because the proxy and Streamlit
both try to prepend the path.

CORS and XSRF protection are disabled because traffic passes through
multiple reverse proxies (nginx + jupyter-server-proxy) which rewrite
the `Origin` and `Referer` headers.

### 6.3) Access the app

Open:

```
https://deepthought.doit.wisc.edu/doit-ai-eval/<WORKSPACE-NAME>/proxy/8501/
```

The trailing slash matters.

### 6.4) Gotchas and troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| **503 Service Temporarily Unavailable** | Workspace name in the URL doesn't match the actual workspace | Fix the URL to use the correct workspace name |
| **404 Not Found** | `jupyter-server-proxy` not loaded in JupyterLab | Install it in the **system** Python (`deactivate` first, then `python -m pip install jupyter-server-proxy`), then restart JupyterLab |
| **404 Not Found** | `--server.baseUrlPath` is set on Streamlit | Remove that flag — the proxy handles the path |
| App loads but WebSocket errors or blank page | CORS/XSRF blocking | Add `--server.enableCORS=false --server.enableXsrfProtection=false` |
| `jupyter-server-proxy` not in `jupyter server extension list` | Installed in venv but Jupyter runs from system Python | `deactivate && python -m pip install jupyter-server-proxy` and restart Jupyter |
| `ValueError: Embedding dimension required` | Fresh workspace, no existing DB file | Already fixed — `app.py` now passes `embedding_dim` from config instead of `None` |

### 6.5) Why not other approaches?

| Approach | Status | Notes |
|----------|--------|-------|
| SSH tunnel (`ssh -L 8501:...`) | Works but fragile | Requires SSH access and an open terminal; breaks on disconnect |
| kubectl port-forward | Works | Same fragility as SSH; fine for quick debugging |
| Run:ai nodeport / ingress | Requires admin | Needs cluster-level config; not self-service |
| Docker container + Run:ai job | Possible future option | Reproducible but requires maintaining an image and registry |
| **jupyter-server-proxy** | **Recommended** | Self-service, works through existing JupyterLab ingress, no admin needed |
