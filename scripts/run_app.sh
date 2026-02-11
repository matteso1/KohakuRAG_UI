#!/usr/bin/env bash
# ===========================================================================
# run_app.sh — Launch the Streamlit app inside a Run:ai pod.
#
# The app binds to 0.0.0.0:8501 so it is reachable from outside the pod
# (via Run:ai nodeport, port-forward, or ingress).
#
# Usage:
#   bash scripts/run_app.sh           # default port 8501
#   PORT=8080 bash scripts/run_app.sh # custom port
# ===========================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

PORT="${PORT:-8501}"

# Quick sanity checks
if ! python -c "import streamlit" 2>/dev/null; then
    echo "ERROR: streamlit not installed. Run setup first:"
    echo "  bash scripts/setup_poweredge_pod.sh"
    exit 1
fi

DB_PATH="$REPO_ROOT/data/embeddings/wattbot_jinav4.db"
if [ ! -f "$DB_PATH" ]; then
    echo "ERROR: Vector database not found at $DB_PATH"
    echo "Run setup first:  bash scripts/setup_poweredge_pod.sh"
    exit 1
fi

echo "Starting WattBot RAG on 0.0.0.0:$PORT"
echo ""
echo "Access the app:"
echo "  - Inside the pod:   http://localhost:$PORT"
echo "  - From your laptop: use one of these methods —"
echo ""
echo "    Method 1 — kubectl port-forward (easiest):"
echo "      kubectl port-forward <pod-name> $PORT:$PORT"
echo "      then open http://localhost:$PORT in your browser"
echo ""
echo "    Method 2 — Run:ai with nodeport:"
echo "      (if submitted with --service-type nodeport --port $PORT:$PORT)"
echo "      open http://<poweredge-ip>:<node-port> in your browser"
echo ""
echo "    Method 3 — SSH tunnel to the K8s node:"
echo "      ssh -L $PORT:localhost:$PORT <your-user>@<poweredge-ip>"
echo "      then open http://localhost:$PORT in your browser"
echo ""

exec streamlit run app.py \
    --server.port="$PORT" \
    --server.address="0.0.0.0" \
    --server.headless=true \
    --browser.gatherUsageStats=false
