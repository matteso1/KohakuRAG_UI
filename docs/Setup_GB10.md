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

## Phase 1 — Required setup (must work before anything else)

### 1) SSH into the GB10

From your **laptop** terminal (Git Bash is recommended on Windows):

```bash
ssh mlx@128.104.18.206   # ethernet
# if that fails:
ssh mlx@10.141.72.249    # wifi
```

Notes:
- Use UW GlobalProtect VPN if off campus.
- Accept the host key if prompted.
- GB10 is a headless Linux workstation — SSH is the primary interface.

---

### 2) Clone the repo (local branch) and create a working branch

On **GB10**:

```bash
ls # get oriented
cd ~/GitHub
```

Create a folder for your git repos to live (use your name)

```bash
mkdir -p ~/GitHub/your-name # adjust to your name 
cd ~/GitHub/your-name
```

Close the KohakuRAG_UI repo (local branch).
```bash
git clone -b local https://github.com/matteso1/KohakuRAG_UI.git
cd KohakuRAG_UI
git branch --show-current   # should print: local
```


### 3) Install `uv` (one‑time per user on GB10)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc
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
uv pip install -e vendor/KohakuVault # must be run before next line. May take a minute
uv pip install -e vendor/KohakuRAG #
```

Verify imports point into `vendor/`:

```bash
python -c "import kohakuvault; print(kohakuvault.__file__)"
python -c "import kohakurag; print(kohakurag.__file__)"
```

Expected:
```
.../KohakuRAG_UI/vendor/...
```

If imports point into `.venv/site-packages`, stop — you are not using the vendored versions.


### 6) Install **local-only development dependencies**

This branch should include a `local_requirements.txt` for dependencies
needed *after* the base install (HF backends, local inference helpers, etc.).

```bash
uv pip install -r local_requirements.txt
```

Notes:
- This file is intentionally **separate** from core requirements.
- It is expected to evolve as HF / local inference work progresses.



### 9) Final required smoke test (before tooling)

Confirm imports:

```bash
python -c "import kohakuvault, kohakurag; print('Imports OK')"
```


### 10) Jupyter Lab (headless, remote)

Jupyter is used for:
- inspecting embeddings
- debugging vector stores
- interactive HF inference tests
- demos


#### 10.1 Register a **named kernel** (important):

```bash
python -m ipykernel install   --user   --name kohaku-gb10   --display-name "kohaku-gb10"
```


#### 10.2 Start Jupyter on GB10

```bash
jupyter lab --no-browser --port=8888
```

Leave this running.


#### 10.3 Port‑forward Jupyter to your laptop

On your **laptop** (new terminal):

```bash
ssh -N -L 8888:localhost:8888 mlx@128.104.18.206
```

In a browser on your laptop, visit the following URL:
```
http://localhost:8888
```

When starting a new notebook, select: **kohaku-gb10** as the kernel.


