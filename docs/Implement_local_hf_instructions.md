# Fully local KohakuRAG_UI (local branch): replace OpenRouter + hosted embeddings with local Hugging Face

Repo (branch): https://github.com/matteso1/KohakuRAG_UI/tree/local

Goal:
- **No network LLM calls** (remove OpenRouter usage at runtime)
- **Embeddings are local too** (no Jina/OpenAI/hosted embedding services)
- Everything runs on **GB10** inside one venv

This document is written as **actionable instructions** + an implementation checklist.


## 0) What your `git grep` results mean

Even though there are many `openrouter` mentions, most are:
- docs (`vendor/KohakuRAG/docs/*`, `README.md`)
- configs (`vendor/KohakuRAG/configs/*`)
- workflows/sweeps (research scripts)

The real runtime dependency is centralized in:
- `vendor/KohakuRAG/src/kohakurag/llm.py` (chat model provider)
- `vendor/KohakuRAG/src/kohakurag/vision.py` (vision provider, if used)
and the dependency itself is explicitly listed in:
- `vendor/KohakuRAG/pyproject.toml` → includes `"openrouter"`

So you do **not** need to edit 100 files. You need:
1) a **new local provider** in `llm.py` (and possibly `vision.py`)
2) a **new local embedding provider**
3) config defaults in the `local` branch updated to select those providers
4) deps updated so installs pull HF libs instead of OpenRouter



## 3) Implement a **local LLM provider** in `vendor/KohakuRAG/src/kohakurag/llm.py`

### 3.1 Add a provider name: `hf_local`

Your configs currently set:
```py
llm_provider = "openrouter"
```

You want to support:
- `llm_provider = "hf_local"`

### 3.2 Implement a simple local HF chat model wrapper

Add a class that matches the same interface the pipeline expects (returning a string, plus any metadata if your code uses it).

Implementation sketch inspired from llm.py file (mirroring OpenAIChatModel):

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

class HuggingFaceLocalChatModel(ChatModel):
    """Chat backend powered by a local Hugging Face Transformers model.

    Intended for fully local inference (no network). Uses device_map="auto".
    """

    def __init__(
        self,
        *,
        model: str,
        system_prompt: str | None = None,
        dtype: str = "bf16",  # "bf16", "fp16", or "auto"
        max_concurrent: int = 2,
        max_new_tokens: int = 512,
        temperature: float = 0.2,
    ) -> None:
        if not HF_AVAILABLE:
            raise ImportError(
                "transformers/torch are required for HuggingFaceLocalChatModel. "
                "Install with: pip install torch transformers accelerate"
            )

        self._model_id = model
        self._system_prompt = system_prompt or "You are a helpful assistant."
        self._max_new_tokens = max_new_tokens
        self._temperature = temperature

        # Concurrency limiter (local inference is heavy)
        self._semaphore: asyncio.Semaphore | None = (
            asyncio.Semaphore(max_concurrent) if max_concurrent > 0 else None
        )

        # Load tokenizer/model once
        self._tokenizer = AutoTokenizer.from_pretrained(self._model_id, use_fast=True)

        torch_dtype = None
        if dtype == "bf16":
            torch_dtype = torch.bfloat16
        elif dtype == "fp16":
            torch_dtype = torch.float16

        self._model = AutoModelForCausalLM.from_pretrained(
            self._model_id,
            torch_dtype=torch_dtype,
            device_map="auto",
        )

    async def complete(self, prompt: str, *, system_prompt: str | None = None) -> str:
        system = system_prompt or self._system_prompt

        # Build messages in the same style as OpenAI/OpenRouter
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ]

        async def _run_generate() -> str:
            return await asyncio.to_thread(self._generate_sync, messages)

        if self._semaphore is not None:
            async with self._semaphore:
                return await _run_generate()
        return await _run_generate()

    def _generate_sync(self, messages: list[dict]) -> str:
        # Use chat template if available for better formatting
        if hasattr(self._tokenizer, "apply_chat_template"):
            text = self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            # Fallback: simple role formatting
            lines = []
            for m in messages:
                lines.append(f"{m['role'].upper()}: {m['content']}")
            lines.append("ASSISTANT:")
            text = "\n".join(lines)

        inputs = self._tokenizer(text, return_tensors="pt")
        inputs = {k: v.to(self._model.device) for k, v in inputs.items()}

        do_sample = self._temperature > 0
        out = self._model.generate(
            **inputs,
            max_new_tokens=self._max_new_tokens,
            do_sample=do_sample,
            temperature=self._temperature if do_sample else None,
        )

        return self._tokenizer.decode(out[0], skip_special_tokens=True).strip()

```

### 3.3 Wire provider selection 

Wherever your code currently does something like:

```python
if provider == "openrouter":
    model = OpenRouterChatModel(...)

# hint: use grep to locate:provider == "openrouter"
```

Add:

```python
elif provider == "hf_local":
    model = HuggingFaceLocalChatModel(
        model_id=getattr(config, "hf_model_id", "Qwen/Qwen2.5-7B-Instruct"),
        dtype=getattr(config, "hf_dtype", "bf16"),
    )
```

Example: scripts/wattbot_answer.py.

Do not change prompt construction / citation logic. Only change the model backend call.

## 4) Implement **local embeddings** (no Jina/OpenAI)

This is the other half of “everything local”.

### 4.1 Find embedding usage and add an embedding provider string

Run:

```bash
git grep -n "embedding"
git grep -n "embed"
git grep -n "jina"
git grep -n "OPENAI_EMBED"
```

You’re looking for a function/class that:
- converts text chunks → vectors
- or sets an `embedding_provider` / `embedding_model` in config
- or calls a hosted embedding API

### 4.2 Recommended local embedding approach (simple + reliable)

Use `sentence-transformers` locally.

Add a local embedding class (wherever embeddings are defined, often something like `embeddings.py`, `retrieval.py`, or `indexing/*`):

```python
from sentence_transformers import SentenceTransformer
import numpy as np

class LocalHFEmbeddingModel:
    def __init__(self, model_id: str):
        self.model = SentenceTransformer(model_id)

    def embed_texts(self, texts):
        # returns list/np.ndarray of vectors
        vecs = self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(vecs, dtype=np.float32)
```

Recommended default model IDs (good quality, common):
- `intfloat/e5-base-v2`
- `BAAI/bge-base-en-v1.5`

Pick one and standardize it via config.

### 4.3 Wire embedding provider selection

Add to your config(s):

```python
embedding_provider = "hf_local"
embedding_model_id = "intfloat/e5-base-v2"
```

Then in your embedding selector:

```python
if embedding_provider == "hf_local":
    emb = LocalHFEmbeddingModel(model_id=embedding_model_id)
```

### 4.4 Storage / vector DB

KohakuRAG uses KohakuVault (SQLite-based) in many setups.
Your goal is:
- produce vectors locally
- store them using the existing vault / index path logic

Do **not** rewrite the storage layer unless the code currently forces a hosted vector DB.
Just replace the embedding computation step.

---

## 5) Update local branch configs (minimal diffs)

You already found defaults such as:
- `vendor/KohakuRAG/configs/text_only/answer.py`
- `vendor/KohakuRAG/configs/jinav4/answer.py`
- etc.

For the `local` branch, change:

```python
llm_provider = "openrouter"
```

to:

```python
llm_provider = "hf_local"
hf_model_id = "Qwen/Qwen2.5-7B-Instruct"   # example
hf_dtype = "bf16"
```

And add embedding defaults:

```python
embedding_provider = "hf_local"
embedding_model_id = "intfloat/e5-base-v2"
```

If there’s a single “central” config your UI loads, update that first; don’t chase every workflow script.

---

## 6) UI `.env` guidance (make it non-confusing)

If you keep `.env.example`, it should NOT force OpenRouter.

Recommended stance for `local` branch:
- `.env` optional
- only used for convenience overrides

Example `.env.example` for local:

```bash
# Optional overrides for local branch
HF_MODEL_ID=Qwen/Qwen2.5-7B-Instruct
HF_DTYPE=bf16
EMBED_MODEL_ID=intfloat/e5-base-v2

# Cache locations (optional)
HF_HOME=/path/to/cache
TRANSFORMERS_CACHE=/path/to/cache
```

Remove or comment out OpenRouter variables so beginners don’t think they’re required.

---

## 7) Install + run checklist (GB10)

From `KohakuRAG_UI` root:

```bash
source .venv/bin/activate

# base install (whatever your repo uses)
uv pip install -e vendor/vault
uv pip install -e vendor/rag
uv pip install -e .

# local-only deps (LLM + embeddings)
uv pip install -r local_requirements.txt
```

Sanity check:

```bash
python -c "import kohakuvault, kohakurag; print('Imports OK')"
python -c "import transformers, sentence_transformers; print('HF deps OK')"
```

Then run Streamlit (use the actual entrypoint you have):

```bash
find . -maxdepth 3 -type f -name "*.py" | grep -i -E "app|streamlit|ui|main"
streamlit run app.py   # adjust path to match repo
```

---

## 8) Final end-to-end validation (what “local” means)

To validate you are truly local:

1) Unset OpenRouter vars (or ensure they’re not used):
```bash
unset OPENROUTER_API_KEY
unset OPENAI_BASE_URL
unset OPENAI_API_KEY
```

2) Run the UI and ask a question that triggers RAG.

3) Confirm:
- retrieval executes
- embeddings are computed locally (watch logs; optionally add a log line)
- generation occurs via HF local provider
- no network call failures (no OpenRouter error messages)

---

## 9) Suggested commit plan (small, reviewable)

1) Add `local_requirements.txt`
2) Add local embedding provider
3) Add local LLM provider
4) Update one config used by the UI to set providers to local
5) Remove OpenRouter dependency from `vendor/KohakuRAG/pyproject.toml` (only after working)
6) Remove/skip OpenRouter tests for local branch

---

## Fast follow-up: what to paste for an exact patch

If you paste these two snippets, I can give you an exact file-by-file patch:
- the provider selection block in `vendor/KohakuRAG/src/kohakurag/llm.py`
- where embeddings are computed / selected (the block that calls Jina/OpenAI/etc.)

Even just 30–80 lines around each selector is enough.
