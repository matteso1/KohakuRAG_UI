"""Remote model clients for RunAI inference workloads.

Provides ChatModel and EmbeddingModel implementations that call remote
HTTP services (vLLM for LLM, FastAPI for embeddings) instead of loading
models locally. This enables splitting the RAG system across multiple
RunAI inference jobs.
"""

import asyncio
from typing import Any, Sequence

import numpy as np

try:
    import httpx

    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False

try:
    from openai import AsyncOpenAI

    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

from .pipeline import ChatModel


class VLLMChatModel(ChatModel):
    """Chat backend powered by a remote vLLM server (OpenAI-compatible API).

    vLLM exposes an OpenAI-compatible ``/v1/chat/completions`` endpoint,
    so this is a thin wrapper around AsyncOpenAI pointed at the vLLM URL.

    Example:
        model = VLLMChatModel(
            base_url="http://my-vllm-service:8000/v1",
            model="Qwen/Qwen2.5-7B-Instruct",
        )
    """

    def __init__(
        self,
        *,
        base_url: str = "http://localhost:8000/v1",
        model: str = "default",
        system_prompt: str | None = None,
        max_tokens: int = 512,
        temperature: float = 0.2,
        api_key: str = "none",
        max_concurrent: int = 10,
    ) -> None:
        if not OPENAI_AVAILABLE:
            raise ImportError(
                "openai package is required for VLLMChatModel. "
                "Install with: pip install openai"
            )

        self._system_prompt = system_prompt or "You are a helpful assistant."
        self._model = model
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._client = AsyncOpenAI(base_url=base_url, api_key=api_key)
        self._semaphore = (
            asyncio.Semaphore(max_concurrent) if max_concurrent > 0 else None
        )

    async def complete(self, prompt: str, *, system_prompt: str | None = None) -> str:
        system = system_prompt or self._system_prompt
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ]

        async def _call() -> str:
            response = await self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                max_tokens=self._max_tokens,
                temperature=self._temperature,
            )
            return response.choices[0].message.content or ""

        if self._semaphore is not None:
            async with self._semaphore:
                return await _call()
        return await _call()


class RemoteEmbeddingModel:
    """Embedding client that calls a remote FastAPI embedding server.

    The server exposes POST /embed accepting {"texts": [...]} and
    returning {"embeddings": [[...], ...], "dimension": N}.

    Example:
        embedder = RemoteEmbeddingModel(
            base_url="http://my-embedding-service:8080",
        )
    """

    def __init__(
        self,
        *,
        base_url: str = "http://localhost:8080",
        timeout: float = 120.0,
    ) -> None:
        if not HTTPX_AVAILABLE:
            raise ImportError(
                "httpx is required for RemoteEmbeddingModel. "
                "Install with: pip install httpx"
            )

        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._dimension: int | None = None

    @property
    def dimension(self) -> int:
        if self._dimension is not None:
            return self._dimension
        # Synchronous probe to get dimension
        import httpx as _httpx

        resp = _httpx.get(
            f"{self._base_url}/info", timeout=self._timeout
        )
        resp.raise_for_status()
        self._dimension = resp.json()["dimension"]
        return self._dimension

    async def embed(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            dim = self.dimension
            return np.zeros((0, dim), dtype=np.float32)

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.post(
                f"{self._base_url}/embed",
                json={"texts": list(texts)},
            )
            resp.raise_for_status()
            data = resp.json()

        self._dimension = data["dimension"]
        return np.array(data["embeddings"], dtype=np.float32)

    async def health(self) -> bool:
        """Check if the remote embedding service is reachable."""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"{self._base_url}/health")
                return resp.status_code == 200
        except Exception:
            return False
