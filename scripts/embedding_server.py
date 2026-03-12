#!/usr/bin/env python3
"""FastAPI embedding server wrapping JinaV4EmbeddingModel.

Serves embeddings over HTTP so the Streamlit app (running in a separate
RunAI inference job) can call it without loading the model locally.

Launch:
    python scripts/embedding_server.py
    # or with uvicorn directly:
    uvicorn scripts.embedding_server:app --host 0.0.0.0 --port 8080

Environment variables:
    EMBEDDING_MODEL    - HuggingFace model ID (default: jinaai/jina-embeddings-v4)
    EMBEDDING_TASK     - Task mode: retrieval, text-matching, code (default: retrieval)
    EMBEDDING_DIM      - Matryoshka dimension (default: 1024)
    EMBEDDING_PORT     - Server port (default: 8080)
    EMBEDDING_HOST     - Server host (default: 0.0.0.0)
"""

import os
import sys
import time
from pathlib import Path
from typing import Sequence

# Import embeddings module directly to avoid kohakurag.__init__ pulling in
# kohakuvault (a Rust extension that isn't needed for the embedding server).
import importlib.util as _ilu

_repo_root = Path(__file__).resolve().parent.parent
_emb_path = _repo_root / "vendor" / "KohakuRAG" / "src" / "kohakurag" / "embeddings.py"
_spec = _ilu.spec_from_file_location("kohakurag.embeddings", _emb_path)
_mod = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
JinaV4EmbeddingModel = _mod.JinaV4EmbeddingModel

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Configuration from environment
# ---------------------------------------------------------------------------
MODEL_NAME = os.environ.get("EMBEDDING_MODEL", "jinaai/jina-embeddings-v4")
TASK = os.environ.get("EMBEDDING_TASK", "retrieval")
DIM = int(os.environ.get("EMBEDDING_DIM", "1024"))
HOST = os.environ.get("EMBEDDING_HOST", "0.0.0.0")
PORT = int(os.environ.get("EMBEDDING_PORT", "8080"))

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(title="KohakuRAG Embedding Server", version="1.0.0")

# Global embedder — initialized on startup
_embedder: JinaV4EmbeddingModel | None = None


class EmbedRequest(BaseModel):
    texts: list[str]


class EmbedResponse(BaseModel):
    embeddings: list[list[float]]
    dimension: int
    count: int
    elapsed_ms: float


class InfoResponse(BaseModel):
    model: str
    task: str
    dimension: int
    status: str


@app.on_event("startup")
async def startup():
    global _embedder
    print(f"[embedding_server] Loading {MODEL_NAME} (task={TASK}, dim={DIM})...", flush=True)
    t0 = time.time()
    _embedder = JinaV4EmbeddingModel(
        model_name=MODEL_NAME,
        task=TASK,
        truncate_dim=DIM,
    )
    # Force model load (it's lazy by default)
    _ = _embedder.dimension
    elapsed = time.time() - t0
    print(f"[embedding_server] Model loaded in {elapsed:.1f}s. Serving on {HOST}:{PORT}", flush=True)


@app.get("/health")
async def health():
    """Always return 200 so Knative's readiness probe doesn't kill the container.

    Callers that need to know if the model is actually ready should check
    /info or POST /embed (which returns 503 while loading).
    """
    if _embedder is None:
        return {"status": "loading"}
    return {"status": "ok"}


@app.get("/info", response_model=InfoResponse)
async def info():
    if _embedder is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")
    return InfoResponse(
        model=MODEL_NAME,
        task=TASK,
        dimension=_embedder.dimension,
        status="ready",
    )


@app.post("/embed", response_model=EmbedResponse)
async def embed(request: EmbedRequest):
    if _embedder is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    if not request.texts:
        return EmbedResponse(
            embeddings=[], dimension=_embedder.dimension, count=0, elapsed_ms=0.0
        )

    t0 = time.time()
    vectors = await _embedder.embed(request.texts)
    elapsed_ms = (time.time() - t0) * 1000

    return EmbedResponse(
        embeddings=vectors.tolist(),
        dimension=_embedder.dimension,
        count=len(request.texts),
        elapsed_ms=round(elapsed_ms, 2),
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=HOST, port=PORT)
