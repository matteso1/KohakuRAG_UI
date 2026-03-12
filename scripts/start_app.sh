#!/usr/bin/env bash
# Bootstrap script for the Streamlit app on RunAI.
# Keeps the workspace command short to avoid UI field truncation.
#
# Usage (RunAI Workspace — private repo, tarball pre-downloaded):
#   Command:   bash
#   Arguments: -c "pip install uv && curl -sL https://github.com/qualiaMachine/KohakuRAG_UI/archive/refs/heads/rag-poweredge.tar.gz | tar xz -C /tmp && bash /tmp/KohakuRAG_UI-rag-poweredge/scripts/start_app.sh"
#
# The script detects if it's running from an already-extracted repo
# and skips the download step.
set -euo pipefail

REPO_DIR="/tmp/KohakuRAG_UI"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# If running from an extracted tarball (e.g. /tmp/KohakuRAG_UI-rag-poweredge/scripts/),
# use that directory directly instead of downloading again.
if [[ "$SCRIPT_DIR" == /tmp/KohakuRAG_UI-*/scripts ]]; then
    EXTRACTED_DIR="${SCRIPT_DIR%/scripts}"
    echo "[start_app] Using pre-extracted repo at ${EXTRACTED_DIR}"
    mv "$EXTRACTED_DIR" "$REPO_DIR" 2>/dev/null || true
else
    # Standalone mode: download the repo
    BRANCH="${APP_BRANCH:-rag-poweredge}"
    DIR_NAME="KohakuRAG_UI-${BRANCH//\//-}"

    echo "[start_app] Installing uv..."
    pip install uv 2>&1 | tail -1

    echo "[start_app] Downloading repo (branch: ${BRANCH})..."
    curl -sL "https://github.com/qualiaMachine/KohakuRAG_UI/archive/refs/heads/${BRANCH}.tar.gz" \
      | tar xz -C /tmp
    mv "/tmp/${DIR_NAME}" "$REPO_DIR"
fi

# Ensure uv is available
command -v uv >/dev/null 2>&1 || { pip install uv 2>&1 | tail -1; }

cd "$REPO_DIR"

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
