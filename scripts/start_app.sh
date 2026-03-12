#!/usr/bin/env bash
# Bootstrap script for the Streamlit app on RunAI.
# Keeps the workspace command short to avoid UI field truncation.
#
# Usage (as RunAI Workspace command):
#   bash -c "curl -sL https://raw.githubusercontent.com/qualiaMachine/KohakuRAG_UI/rag-poweredge/scripts/start_app.sh | bash"
set -euo pipefail

BRANCH="${APP_BRANCH:-rag-poweredge}"
# GitHub converts slashes to dashes in tarball directory names
DIR_NAME="KohakuRAG_UI-${BRANCH//\//-}"

echo "[start_app] Installing uv..."
pip install uv 2>&1 | tail -1

echo "[start_app] Downloading repo (branch: ${BRANCH})..."
curl -sL "https://github.com/qualiaMachine/KohakuRAG_UI/archive/refs/heads/${BRANCH}.tar.gz" \
  | tar xz -C /tmp
mv "/tmp/${DIR_NAME}" /tmp/KohakuRAG_UI
cd /tmp/KohakuRAG_UI

echo "[start_app] Symlinking data from PPVC..."
ln -sf /wattbot-data/embeddings data/embeddings
ln -sf /wattbot-data/corpus data/corpus

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
