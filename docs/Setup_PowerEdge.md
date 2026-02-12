# GB10 + KohakuRAG_UI (local branch) setup guide — v2

This guide is a **clean, ordered, end‑to‑end setup** for working on the
`local` branch of **KohakuRAG_UI** on the Dell GB10.

It preserves *all* important steps from the original remote‑access notes
(including Jupyter + kernel naming), but moves **optional tools** to the
correct place in the workflow.

Repo (local branch):  
https://github.com/matteso1/KohakuRAG_UI/tree/local

Primary development goal:
> **Replace OpenRouter-backed LLM calls with Hugging Face (HF) calls for local/on‑prem inference**

---


### 1) Clone the repo (local branch) a

In terminal from Jupyter lab...

```bash
git clone -b local https://github.com/matteso1/KohakuRAG_UI.git
cd KohakuRAG_UI
git branch --show-current   # should print: local
```

### 2) Create cache and temp directories on /workspace2 (large disk)
```bash
# create cache and temp directories on /workspace2 (large disk)
mkdir -p \
  /workspace2/.cache/uv \
  /workspace2/.cache/pip \
  /workspace2/.cache/huggingface \
  /workspace2/.cache/torch \
  /workspace2/.tmp \
  /workspace2/.tmp/uv \
  /workspace2/.tmp/hf

mkdir -p /workspace2/bin

# -------------------------
# System-wide temp (many libs respect these)
# -------------------------
export TMPDIR=/workspace2/.tmp
export TEMP=/workspace2/.tmp
export TMP=/workspace2/.tmp

# -------------------------
# Python / package managers
# -------------------------
export UV_CACHE_DIR=/workspace2/.cache/uv
export PIP_CACHE_DIR=/workspace2/.cache/pip

# (optional) uv can also use its own temp dir if you want separation
# export TMPDIR=/workspace2/.tmp/uv

# -------------------------
# Hugging Face / PyTorch
# -------------------------
export HF_HOME=/workspace2/.cache/huggingface
export TRANSFORMERS_CACHE=/workspace2/.cache/huggingface
export HF_DATASETS_CACHE=/workspace2/.cache/huggingface
export TORCH_HOME=/workspace2/.cache/torch
```

```bash


```
### 3) Install `uv` (one‑time per user on GB10)

Install uv and put /workspace2/bin on PATH (for this shell)

```bash
# optional: clean the old one
# rm -f /home/runai-home/.local/bin/uv /home/runai-home/.local/bin/uvx 2>/dev/null || true

curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="/workspace2/bin" sh
export PATH="/workspace2/bin:$PATH"
uv --version
```

### 4) Create and activate the Python virtual environment

From the repo root:

```bash
uv venv --python 3.11
source .venv/bin/activate
python --version
```

Notes:
- Always verify the venv is active before installing or running anything.


### 5) Install **vendored** Vault + RAG (critical on GB10/ARM)

The `local` branch vendors its core dependencies to avoid ARM build issues.

Install them **editable** so imports resolve locally:

```bash
uv pip install -e vendor/KohakuVault # Must be run before next line. May take a minute
uv pip install -e vendor/KohakuRAG #
```

### 6) Install **local-only development dependencies**

This branch should include a `local_requirements.txt` for dependencies
needed *after* the base install (HF backends, local inference helpers, etc.).

```bash
uv pip install -r local_requirements.txt
```

Notes:
- This file is intentionally **separate** from core requirements.
- It is expected to evolve as HF / local inference work progresses.



### 7) Final required smoke test (before tooling)

Confirm imports:

```bash
python -c "import kohakuvault, kohakurag; print('Imports OK')"
```


### 8) Build the index

```bash
cd vendor/KohakuRAG
kogine run scripts/wattbot_build_index.py --config configs/jinav4/index.py
cd ../..

# Verify the database was created
ls -lh data/embeddings/wattbot_jinav4.db
```

### 8) Jupyter Lab (headless, remote)

Jupyter is used for:
- inspecting embeddings
- debugging vector stores
- interactive HF inference tests
- demos


#### 8.1 Register a **named kernel** in case you want to work with notebooks.

```bash
python -m ipykernel install   --user   --name kohaku-poweredge   --display-name "kohaku-poweredge"
```


Within a couple of minutes, you can open a new notebook with this environment/kernel selected.


### 9) Run the Streamlit app

The app is accessed through JupyterLab's built-in proxy
(`jupyter-server-proxy`). See `docs/Streamlit_App_Guide.md` section 6
for full details on the proxy architecture.

**Prerequisites:** `jupyter-server-proxy` must be installed in the
**system** Python (the one running JupyterLab), not just your venv.
This should already be done in the runAI workspace configuration.


**Launch** (from inside the venv, in the repo root):

```bash
streamlit run app.py --server.port 8501 --server.address 0.0.0.0 \
  --server.enableCORS=false --server.enableXsrfProtection=false
```

**Access** at:

```
https://deepthought.doit.wisc.edu/doit-ai-eval/<WORKSPACE-NAME>/proxy/8501/
```

Replace `<WORKSPACE-NAME>` with your actual Run:ai workspace name
(e.g. `endemann-pytorch22`). Trailing slash matters.

If you get a 503, the workspace name in the URL is wrong.
If you get a 404, `jupyter-server-proxy` isn't loaded — see troubleshooting
in `docs/Streamlit_App_Guide.md` section 6.4.

---

## Phase 2 — Test the local HF pipeline
See KohakuRAG_UI/docs/Benchmarking_Guide.md.