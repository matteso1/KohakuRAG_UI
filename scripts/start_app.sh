#!/usr/bin/env bash
# Bootstrap script for the Streamlit app on RunAI.
# Keeps the workspace command short to avoid UI field truncation.
#
# Usage (RunAI Workspace):
#   Command:   bash
#   Arguments: -c "bash /home/jovyan/work/KohakuRAG_UI/scripts/start_app.sh"
#
# The script uses the repo already cloned on the personal PVC from Step 0.
# No GitHub download needed — the code is on the shared filesystem.
set -euo pipefail

# Where Step 0 cloned the repo (personal workspace PVC)
REPO_DIR="${APP_REPO_DIR:-/home/jovyan/work/KohakuRAG_UI}"

if [[ ! -d "$REPO_DIR" ]]; then
    echo "[start_app] ERROR: Repo not found at ${REPO_DIR}"
    echo "[start_app] Run Step 0 first to clone the repo."
    exit 1
fi

echo "[start_app] Using repo at ${REPO_DIR}"
cd "$REPO_DIR"

# Ensure uv is available
command -v uv >/dev/null 2>&1 || {
    echo "[start_app] Installing uv..."
    pip install uv 2>&1 | tail -1
}

# Data symlinks: point data dirs at the wattbot-data PPVC so the app
# reads the pre-built vector index and corpus from shared storage.
# The git repo ships placeholder dirs, so we must replace them with
# symlinks even when the paths already exist.
if [[ -d /wattbot-data ]]; then
    echo "[start_app] Creating data symlinks to /wattbot-data PPVC..."
    mkdir -p data
    # Remove placeholder dirs (or stale symlinks) before linking
    rm -rf data/embeddings data/corpus data/pdfs
    ln -sf /wattbot-data/embeddings data/embeddings
    ln -sf /wattbot-data/corpus     data/corpus
    ln -sf /wattbot-data/pdfs       data/pdfs
else
    echo "[start_app] WARNING: /wattbot-data PPVC not mounted — using local data dirs."
fi

echo "[start_app] Installing dependencies..."
uv pip install --system streamlit openai httpx numpy python-dotenv
uv pip install --system vendor/KohakuVault vendor/KohakuRAG

echo "[start_app] Starting Streamlit on port 8501..."
exec python3 -m streamlit run app.py \
  --server.port=8501 \
  --server.address=0.0.0.0 \
  --server.headless=true \
  --server.enableCORS=false \
  --server.enableXsrfProtection=false
