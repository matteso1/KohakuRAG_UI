"""
WattBot RAG — Streamlit App

Interactive UI for querying the WattBot RAG pipeline with local HF models.
Supports single-model and live multi-model ensemble modes.

Launch:
    streamlit run app.py
"""

import asyncio
import csv
import gc
import importlib.util
import json
import logging
import re
import sys
import time
import traceback
from collections import Counter
from pathlib import Path

import streamlit as st

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_repo_root = Path(__file__).resolve().parent
sys.path.insert(0, str(_repo_root / "vendor" / "KohakuRAG" / "src"))
sys.path.insert(0, str(_repo_root / "scripts"))

from kohakurag import RAGPipeline
from kohakurag.datastore import KVaultNodeStore
from kohakurag.embeddings import JinaV4EmbeddingModel
from kohakurag.llm import HuggingFaceLocalChatModel

# ---------------------------------------------------------------------------
# Prompts (shared with run_experiment.py)
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """
You must answer strictly based on the provided context snippets.
Do NOT use external knowledge or assumptions.
If the context does not clearly support an answer, you must output the literal string "is_blank" for both answer_value and ref_id.
For True/False questions, you MUST output "1" for True and "0" for False in answer_value. Do NOT output the words "True" or "False".
""".strip()

SYSTEM_PROMPT_BEST_GUESS = """
You must answer based on the provided context snippets.
If the context strongly supports an answer, answer normally.
If the context only partially or weakly supports an answer, still provide your best guess but set confidence to "low".
Set confidence to "high" when the context clearly and directly answers the question.
For True/False questions, you MUST output "1" for True and "0" for False in answer_value. Do NOT output the words "True" or "False".
""".strip()

USER_TEMPLATE = """
You will be given a question and context snippets taken from documents.
You must follow these rules:
- Use only the provided context; do not rely on external knowledge.
- If the context does not clearly support an answer, use "is_blank" for all fields except explanation.
- For unanswerable questions, set answer to "Unable to answer with confidence based on the provided documents."
- For True/False questions: answer_value must be "1" for True or "0" for False (not the words "True" or "False").

Question: {question}

Context:
{context}

Return STRICT JSON with the following keys, in this order:
- explanation          (1-3 sentences that directly answer the question. Cite sources by ref_id, e.g. "According to [wu2021a], ...". Do NOT use vague phrases like "the context states" or "the passage mentions".)
- answer               (short natural-language response, e.g. "1438 lbs", "Water consumption", "TRUE")
- answer_value         (ONLY the numeric or categorical value, e.g. "1438", "Water consumption", "1"; or "is_blank")
- ref_id               (list of document ids from the context used as evidence; or "is_blank")
- ref_url              (list of URLs for the cited documents; or "is_blank")
- supporting_materials (verbatim quote, table reference, or figure reference from the cited document; or "is_blank")

JSON Answer:
""".strip()

USER_TEMPLATE_BEST_GUESS = """
You will be given a question and context snippets taken from documents.
You must follow these rules:
- Use the provided context as your primary source.
- If the context clearly answers the question, answer normally with confidence "high".
- If the context only partially relates, provide your best-effort answer with confidence "low".
- For True/False questions: answer_value must be "1" for True or "0" for False (not the words "True" or "False").

Question: {question}

Context:
{context}

Return STRICT JSON with the following keys, in this order:
- explanation          (1-3 sentences that directly answer the question. Cite sources by ref_id, e.g. "According to [wu2021a], ...". Do NOT use vague phrases like "the context states" or "the passage mentions".)
- answer               (short natural-language response, e.g. "1438 lbs", "Water consumption", "TRUE")
- answer_value         (ONLY the numeric or categorical value, e.g. "1438", "Water consumption", "1"; or "is_blank")
- confidence           ("high" if the context clearly supports the answer, "low" if this is a best guess)
- ref_id               (list of document ids from the context used as evidence; or "is_blank")
- ref_url              (list of URLs for the cited documents; or "is_blank")
- supporting_materials (verbatim quote, table reference, or figure reference from the cited document; or "is_blank")

JSON Answer:
""".strip()


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CONFIGS_DIR = _repo_root / "vendor" / "KohakuRAG" / "configs"

# Approximate 4-bit NF4 VRAM (GB) per config. Used for planning only.
VRAM_4BIT_GB = {
    "hf_qwen1_5b": 2, "hf_qwen3b": 3, "hf_qwen7b": 6, "hf_qwen14b": 10,
    "hf_qwen32b": 20, "hf_qwen72b": 40, "hf_llama3_8b": 6, "hf_gemma2_9b": 7,
    "hf_gemma2_27b": 17, "hf_mixtral_8x7b": 26, "hf_mixtral_8x22b": 80,
    "hf_mistral7b": 6, "hf_phi3_mini": 3, "hf_qwen3_30b_a3b": 18,
    "hf_qwen3_next_80b_a3b": 40, "hf_olmoe_1b7b": 4,
}
EMBEDDER_OVERHEAD_GB = 3  # Jina V4 embedder + store + misc
PRECISION_MULTIPLIER = {"4bit": 1.0, "bf16": 4.0, "fp16": 4.0, "auto": 4.0}

# ---------------------------------------------------------------------------
# Metadata URL lookup  (ref_id → URL from metadata.csv)
# ---------------------------------------------------------------------------
_METADATA_CSV = _repo_root / "data" / "metadata.csv"

def _load_metadata_urls() -> dict[str, str]:
    """Build a ref_id → url mapping from metadata.csv."""
    mapping: dict[str, str] = {}
    if not _METADATA_CSV.exists():
        return mapping
    with open(_METADATA_CSV, newline="", encoding="utf-8", errors="replace") as f:
        for row in csv.DictReader(f):
            doc_id = row.get("id", "").strip()
            url = row.get("url", "").strip()
            if doc_id and url:
                mapping[doc_id] = url
    return mapping

METADATA_URLS: dict[str, str] = _load_metadata_urls()


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------
def _debug(msg: str) -> None:
    """Print debug info to terminal and, if debug mode is on, to the Streamlit UI."""
    logger.info(msg)
    print(f"[DEBUG] {msg}", flush=True)


def discover_configs() -> dict[str, Path]:
    """Find all hf_*.py config files and return {display_name: path}."""
    return {p.stem: p for p in sorted(CONFIGS_DIR.glob("hf_*.py"))}


def load_config(config_path: Path) -> dict:
    """Load a Python config file into a dict."""
    spec = importlib.util.spec_from_file_location("config", config_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config = {}
    for key in [
        "db", "table_prefix", "questions", "output", "metadata",
        "llm_provider", "top_k", "planner_max_queries", "deduplicate_retrieval",
        "rerank_strategy", "top_k_final", "retrieval_threshold",
        "max_retries", "max_concurrent",
        "embedding_model", "embedding_dim", "embedding_task", "embedding_model_id",
        "hf_model_id", "hf_dtype", "hf_max_new_tokens", "hf_temperature",
    ]:
        if hasattr(module, key):
            config[key] = getattr(module, key)
    return config


def estimate_vram(config_name: str, precision: str) -> float:
    """Estimate VRAM (GB) for a model at given precision."""
    base = VRAM_4BIT_GB.get(config_name, 8)
    return base * PRECISION_MULTIPLIER.get(precision, 1.0)


# ---------------------------------------------------------------------------
# GPU detection
# ---------------------------------------------------------------------------
def get_gpu_info() -> dict:
    """Detect GPU count, names, and free VRAM per GPU."""
    try:
        import torch
        if not torch.cuda.is_available():
            return {"gpu_count": 0, "gpus": [], "total_free_gb": 0}
    except ImportError:
        return {"gpu_count": 0, "gpus": [], "total_free_gb": 0}

    gpus = []
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        free, total = torch.cuda.mem_get_info(i)
        gpus.append({
            "index": i,
            "name": props.name,
            "total_gb": total / (1024**3),
            "free_gb": free / (1024**3),
        })
    total_free = sum(g["free_gb"] for g in gpus)
    return {"gpu_count": len(gpus), "gpus": gpus, "total_free_gb": total_free}


def plan_ensemble(config_names: list[str], precision: str, gpu_info: dict) -> dict:
    """Decide parallel vs sequential execution based on available VRAM.

    Returns:
        {"mode": "parallel"|"sequential"|"error", "model_vrams": [...], ...}
    """
    model_vrams = [estimate_vram(n, precision) for n in config_names]
    total_needed = sum(model_vrams) + EMBEDDER_OVERHEAD_GB
    total_free = gpu_info["total_free_gb"]

    if total_free == 0:
        return {"mode": "error", "model_vrams": model_vrams,
                "reason": "No GPU detected"}

    max_single_gpu = max(g["free_gb"] for g in gpu_info["gpus"])
    largest_model = max(model_vrams)

    if largest_model + EMBEDDER_OVERHEAD_GB > max_single_gpu:
        return {"mode": "error", "model_vrams": model_vrams,
                "reason": (f"Largest model needs ~{largest_model + EMBEDDER_OVERHEAD_GB:.0f} GB "
                           f"but largest GPU only has {max_single_gpu:.0f} GB free")}

    if total_needed <= total_free:
        return {"mode": "parallel", "model_vrams": model_vrams}
    return {"mode": "sequential", "model_vrams": model_vrams}


# ---------------------------------------------------------------------------
# Pipeline init
# ---------------------------------------------------------------------------
def _load_shared_resources(config: dict) -> tuple[JinaV4EmbeddingModel, KVaultNodeStore]:
    """Load embedder and vector store from config."""
    embedding_dim = config.get("embedding_dim", 1024)
    embedding_task = config.get("embedding_task", "retrieval")
    db_raw = config.get("db", "data/embeddings/wattbot_jinav4.db")
    db_path = _repo_root / db_raw.removeprefix("../").removeprefix("../")
    table_prefix = config.get("table_prefix", "wattbot_jv4")

    _debug(
        f"Loading shared resources:\n"
        f"  db_path       = {db_path} (exists={db_path.exists()})\n"
        f"  table_prefix  = {table_prefix}\n"
        f"  embedding_dim = {embedding_dim}\n"
        f"  embedding_task= {embedding_task}"
    )

    embedder = JinaV4EmbeddingModel(
        task=embedding_task,
        truncate_dim=embedding_dim,
    )
    _debug(f"Embedder loaded: dimension={embedder.dimension}")

    store = KVaultNodeStore(
        db_path,
        table_prefix=table_prefix,
        dimensions=embedding_dim,
        paragraph_search_mode="averaged",
    )
    _debug(
        f"Store opened: dimensions={store._dimensions}, "
        f"vec_count={store._vectors.info().get('count', '?')}"
    )
    return embedder, store


def _load_chat_model(config: dict, precision: str) -> HuggingFaceLocalChatModel:
    """Create a HuggingFaceLocalChatModel from config."""
    return HuggingFaceLocalChatModel(
        model=config.get("hf_model_id", "Qwen/Qwen2.5-7B-Instruct"),
        system_prompt=SYSTEM_PROMPT,
        dtype=precision,
        max_new_tokens=config.get("hf_max_new_tokens", 512),
        temperature=config.get("hf_temperature", 0.2),
        max_concurrent=config.get("max_concurrent", 2),
    )


def _unload_chat_model(chat_model: HuggingFaceLocalChatModel) -> None:
    """Free GPU memory from a loaded model."""
    import torch
    if hasattr(chat_model, "_model"):
        del chat_model._model
    if hasattr(chat_model, "_tokenizer"):
        del chat_model._tokenizer
    del chat_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@st.cache_resource(show_spinner="Loading model and vector store...")
def init_single_pipeline(config_name: str, precision: str) -> RAGPipeline:
    """Load a single-model pipeline. Cached across reruns."""
    config = load_config(CONFIGS_DIR / f"{config_name}.py")
    embedder, store = _load_shared_resources(config)
    chat_model = _load_chat_model(config, precision)
    return RAGPipeline(store=store, embedder=embedder, chat_model=chat_model, planner=None)


@st.cache_resource(show_spinner="Loading ensemble models...")
def init_ensemble_parallel(config_names: tuple[str, ...], precision: str) -> dict[str, RAGPipeline]:
    """Load all ensemble models into memory (parallel mode). Cached."""
    # Use first config for shared resources (db/embedder are the same across configs)
    ref_config = load_config(CONFIGS_DIR / f"{config_names[0]}.py")
    embedder, store = _load_shared_resources(ref_config)

    pipelines = {}
    for name in config_names:
        config = load_config(CONFIGS_DIR / f"{name}.py")
        chat_model = _load_chat_model(config, precision)
        pipelines[name] = RAGPipeline(
            store=store, embedder=embedder, chat_model=chat_model, planner=None,
        )
    return pipelines


@st.cache_resource(show_spinner="Loading embedder and vector store...")
def init_shared_only() -> tuple[JinaV4EmbeddingModel, KVaultNodeStore]:
    """Load only the embedder + store (for sequential ensemble). Cached."""
    ref_config = load_config(next(CONFIGS_DIR.glob("hf_*.py")))
    return _load_shared_resources(ref_config)


# ---------------------------------------------------------------------------
# Query helpers
# ---------------------------------------------------------------------------
def _run_qa_sync(pipeline: RAGPipeline, question: str, top_k: int, best_guess: bool = False):
    """Run pipeline.run_qa synchronously."""
    sys_prompt = SYSTEM_PROMPT_BEST_GUESS if best_guess else SYSTEM_PROMPT
    usr_template = USER_TEMPLATE_BEST_GUESS if best_guess else USER_TEMPLATE
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(
            pipeline.run_qa(
                question,
                system_prompt=sys_prompt,
                user_template=usr_template,
                top_k=top_k,
            )
        )
    finally:
        loop.close()


def run_single_query(pipeline: RAGPipeline, question: str, top_k: int, best_guess: bool = False):
    """Run a single model query."""
    return _run_qa_sync(pipeline, question, top_k, best_guess=best_guess)


def run_ensemble_parallel_query(
    pipelines: dict[str, RAGPipeline], question: str, top_k: int,
    best_guess: bool = False,
) -> dict[str, object]:
    """Query all pre-loaded models concurrently."""
    results = {}
    for name, pipeline in pipelines.items():
        t0 = time.time()
        result = _run_qa_sync(pipeline, question, top_k, best_guess=best_guess)
        results[name] = {"result": result, "time": time.time() - t0}
    return results


def run_ensemble_sequential_query(
    config_names: list[str],
    precision: str,
    question: str,
    top_k: int,
    progress_callback=None,
    best_guess: bool = False,
) -> dict[str, object]:
    """Load each model one at a time, query, unload. Saves VRAM."""
    embedder, store = init_shared_only()
    results = {}

    for i, name in enumerate(config_names):
        if progress_callback:
            progress_callback(i, len(config_names), name)

        config = load_config(CONFIGS_DIR / f"{name}.py")
        chat_model = _load_chat_model(config, precision)
        pipeline = RAGPipeline(
            store=store, embedder=embedder, chat_model=chat_model, planner=None,
        )

        t0 = time.time()
        result = _run_qa_sync(pipeline, question, top_k, best_guess=best_guess)
        elapsed = time.time() - t0
        results[name] = {"result": result, "time": elapsed}

        # Free model memory before loading next
        _unload_chat_model(chat_model)
        del pipeline

    return results


# ---------------------------------------------------------------------------
# Ensemble aggregation
# ---------------------------------------------------------------------------
def aggregate_majority(answers: list[str]) -> str:
    """Most common answer. Ties go to first occurrence."""
    valid = [a for a in answers if a and a.strip() and a != "is_blank"]
    if not valid:
        return "is_blank"
    return Counter(valid).most_common(1)[0][0]


def aggregate_first_non_blank(answers: list[str]) -> str:
    """First non-blank answer in model order."""
    for a in answers:
        if a and a.strip() and a != "is_blank":
            return a
    return "is_blank"


def aggregate_refs(ref_lists: list) -> list[str]:
    """Union of all reference IDs across models."""
    all_refs = set()
    for refs in ref_lists:
        if isinstance(refs, list):
            all_refs.update(r for r in refs if r and r != "is_blank")
        elif isinstance(refs, str) and refs != "is_blank":
            try:
                parsed = json.loads(refs.replace("'", '"'))
                all_refs.update(parsed)
            except (json.JSONDecodeError, TypeError):
                all_refs.add(refs)
    return sorted(all_refs) if all_refs else []


def build_ensemble_answer(
    model_results: dict[str, object], strategy: str,
) -> dict:
    """Aggregate individual model results into an ensemble answer."""
    answers = []
    values = []
    explanations = []
    ref_lists = []
    ref_url_lists = []

    for name, entry in model_results.items():
        ans = entry["result"].answer
        answers.append(ans.answer)
        values.append(ans.answer_value)
        explanations.append(ans.explanation)
        ref_lists.append(ans.ref_id)
        ref_url_lists.append(ans.ref_url)

    agg_fn = aggregate_majority if strategy == "majority" else aggregate_first_non_blank

    return {
        "answer": agg_fn(answers),
        "answer_value": agg_fn(values),
        "explanation": agg_fn(explanations),
        "ref_id": aggregate_refs(ref_lists),
        "ref_url": aggregate_refs(ref_url_lists),
        "individual": {
            name: {
                "answer": entry["result"].answer.answer,
                "answer_value": entry["result"].answer.answer_value,
                "explanation": entry["result"].answer.explanation,
                "ref_id": entry["result"].answer.ref_id,
                "time": entry["time"],
                "raw_response": entry["result"].raw_response,
            }
            for name, entry in model_results.items()
        },
    }


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------
def main():
    st.set_page_config(page_title="WattBot RAG", page_icon="lightning", layout="wide")
    st.title("WattBot RAG Pipeline")

    configs = discover_configs()
    if not configs:
        st.error("No HF config files found in vendor/KohakuRAG/configs/")
        return

    # ---- Sidebar ----
    with st.sidebar:
        st.header("Settings")
        mode = st.radio("Mode", ["Single model", "Ensemble"], horizontal=True)
        precision = st.selectbox("Precision", ["4bit", "bf16", "fp16", "auto"], index=0)
        top_k = st.slider("Retrieved chunks (top_k)", min_value=1, max_value=20, value=8)
        best_guess = st.toggle("Allow best-guess answers", value=False,
                               help="When enabled, out-of-scope questions get a best-effort answer labelled as a guess.")

        st.divider()
        config_list = list(configs.keys())

        if mode == "Single model":
            default_idx = config_list.index("hf_qwen7b") if "hf_qwen7b" in config_list else 0
            selected_config = st.selectbox("Model config", config_list, index=default_idx)
            selected_configs = [selected_config]
            ensemble_strategy = None
        else:
            selected_configs = st.multiselect(
                "Ensemble models (pick 2+)", config_list,
                default=["hf_qwen7b", "hf_llama3_8b"] if all(
                    c in config_list for c in ["hf_qwen7b", "hf_llama3_8b"]
                ) else config_list[:2],
            )
            ensemble_strategy = st.selectbox(
                "Aggregation", ["majority", "first_non_blank"],
            )

        # GPU info
        st.divider()
        gpu_info = get_gpu_info()
        if gpu_info["gpu_count"] > 0:
            st.caption(f"**{gpu_info['gpu_count']} GPU(s)** detected")
            for g in gpu_info["gpus"]:
                st.caption(f"  GPU {g['index']}: {g['name']} — "
                           f"{g['free_gb']:.1f} / {g['total_gb']:.1f} GB free")
        else:
            st.caption("No GPU detected")

        # Ensemble VRAM plan
        if mode == "Ensemble" and len(selected_configs) >= 2:
            plan = plan_ensemble(selected_configs, precision, gpu_info)
            vram_list = [f"{n}: ~{v:.0f}GB" for n, v in
                         zip(selected_configs, plan["model_vrams"])]
            st.caption(f"VRAM: {', '.join(vram_list)}")
            if plan["mode"] == "parallel":
                st.caption("Strategy: **parallel** (all models in memory)")
            elif plan["mode"] == "sequential":
                st.caption("Strategy: **sequential** (load one at a time)")
            else:
                st.warning(plan["reason"])

    # ---- Validate ensemble selection ----
    if mode == "Ensemble" and len(selected_configs) < 2:
        st.info("Select at least 2 models for ensemble mode.")
        return

    # ---- Load pipelines ----
    try:
        if mode == "Single model":
            pipeline = init_single_pipeline(selected_configs[0], precision)
        elif mode == "Ensemble":
            plan = plan_ensemble(selected_configs, precision, gpu_info)
            if plan["mode"] == "error":
                st.error(f"Cannot run ensemble: {plan['reason']}")
                return
            if plan["mode"] == "parallel":
                ensemble_pipelines = init_ensemble_parallel(
                    tuple(selected_configs), precision,
                )
            # sequential doesn't pre-load models
    except Exception as e:
        st.error(f"Failed to load pipeline: {e}")
        tb = traceback.format_exc()
        _debug(f"Load error:\n{tb}")
        with st.expander("Full traceback"):
            st.code(tb, language="python")
        return

    # ---- Chat interface ----
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Render history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and "details" in msg:
                _render_details(msg["details"])

    # User input
    if question := st.chat_input("Ask a question about the WattBot documents..."):
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            t0 = time.time()

            if mode == "Single model":
                with st.spinner("Retrieving and generating..."):
                    try:
                        result = run_single_query(pipeline, question, top_k, best_guess=best_guess)
                    except Exception as e:
                        st.error(f"Pipeline error: {e}")
                        tb = traceback.format_exc()
                        _debug(f"Pipeline error:\n{tb}")
                        with st.expander("Full traceback"):
                            st.code(tb, language="python")
                        return
                elapsed = time.time() - t0
                _display_single_result(result, elapsed)

            else:  # Ensemble
                try:
                    if plan["mode"] == "parallel":
                        with st.spinner(
                            f"Querying {len(selected_configs)} models in parallel..."
                        ):
                            model_results = run_ensemble_parallel_query(
                                ensemble_pipelines, question, top_k,
                                best_guess=best_guess,
                            )
                    else:
                        status = st.status(
                            f"Querying {len(selected_configs)} models sequentially...",
                            expanded=True,
                        )
                        def _progress(i, total, name):
                            status.update(label=f"[{i+1}/{total}] Loading {name}...")
                        model_results = run_ensemble_sequential_query(
                            selected_configs, precision, question, top_k,
                            progress_callback=_progress,
                            best_guess=best_guess,
                        )
                        status.update(label="Aggregating results...", state="complete")
                except Exception as e:
                    st.error(f"Ensemble error: {e}")
                    tb = traceback.format_exc()
                    _debug(f"Ensemble error:\n{tb}")
                    with st.expander("Full traceback"):
                        st.code(tb, language="python")
                    return

                elapsed = time.time() - t0
                agg = build_ensemble_answer(model_results, ensemble_strategy)
                _display_ensemble_result(agg, model_results, elapsed, ensemble_strategy)


def _extract_confidence(raw_response: str) -> str:
    """Extract confidence field from raw JSON or bullet-list response."""
    # Try JSON first
    try:
        start = raw_response.index("{")
        end = raw_response.rindex("}") + 1
        data = json.loads(raw_response[start:end])
        return str(data.get("confidence", "")).strip().lower()
    except Exception:
        pass
    # Fallback: bullet-list format (- confidence   high/low)
    m = re.search(r"-\s*confidence\s{2,}(\S+)", raw_response)
    if m:
        return m.group(1).strip().strip('"').lower()
    return ""


def _display_single_result(result, elapsed: float):
    """Display a single-model answer."""
    answer = result.answer
    timing = result.timing
    confidence = _extract_confidence(result.raw_response)

    if answer.explanation and answer.explanation != "is_blank":
        if confidence == "low":
            st.markdown(f"**{answer.explanation}**")
            st.warning("Best guess — the retrieved context only partially supports this answer.")
        else:
            st.markdown(f"**{answer.explanation}**")
    elif answer.answer and answer.answer != "is_blank":
        st.markdown(f"**{answer.answer}**")
    else:
        st.markdown("**Out-of-scope** — the provided documents do not contain enough information to answer this question.")
    if answer.answer_value and answer.answer_value != "is_blank":
        st.markdown(f"Value: `{answer.answer_value}`")

    # Clickable reference links (shown directly, not inside an expander)
    ref_ids = answer.ref_id
    ref_urls = answer.ref_url
    if ref_ids and ref_ids != "is_blank":
        links = []
        for i, rid in enumerate(ref_ids if isinstance(ref_ids, list) else [ref_ids]):
            url = METADATA_URLS.get(rid)
            if not url:
                url = ref_urls[i] if isinstance(ref_urls, list) and i < len(ref_urls) else None
            if url and url != "is_blank":
                links.append(f"[{rid}]({url})")
            else:
                links.append(rid)
        st.markdown("Sources: " + " · ".join(links))

    details = {
        "timing": timing,
        "elapsed": elapsed,
        "ref_id": answer.ref_id,
        "ref_url": answer.ref_url,
        "supporting_materials": answer.supporting_materials,
        "snippets": [
            {"rank": s.rank, "score": s.score, "title": s.document_title, "text": s.text}
            for s in result.retrieval.snippets
        ],
        "raw_response": result.raw_response,
    }
    _render_details(details)

    if answer.explanation and answer.explanation != "is_blank":
        display_answer = answer.explanation
    elif answer.answer and answer.answer != "is_blank":
        display_answer = answer.answer
    else:
        display_answer = "Out-of-scope"
    st.session_state.messages.append({
        "role": "assistant", "content": display_answer, "details": details,
    })


def _display_ensemble_result(
    agg: dict, model_results: dict, elapsed: float, strategy: str,
):
    """Display aggregated ensemble answer + per-model breakdown."""
    if agg["explanation"] and agg["explanation"] != "is_blank":
        st.markdown(f"**{agg['explanation']}**")
    elif agg["answer"] and agg["answer"] != "is_blank":
        st.markdown(f"**{agg['answer']}**")
    else:
        st.markdown("**Out-of-scope** — the provided documents do not contain enough information to answer this question.")
    if agg["answer_value"] and agg["answer_value"] != "is_blank":
        st.markdown(f"Value: `{agg['answer_value']}`")

    n_models = len(model_results)
    model_times = [e["time"] for e in model_results.values()]
    total_gen = sum(model_times)

    cols = st.columns(3)
    cols[0].metric("Models", n_models)
    cols[1].metric("Aggregation", strategy)
    cols[2].metric("Total", f"{elapsed:.1f}s")

    # Per-model answers
    with st.expander(f"Individual model answers ({n_models} models)"):
        for name, info in agg["individual"].items():
            agreed = info["answer_value"] == agg["answer_value"]
            marker = "+" if agreed else "-"
            val = info["answer_value"] if info["answer_value"] and info["answer_value"] != "is_blank" else "Out-of-scope"
            ans = info["answer"] if info["answer"] and info["answer"] != "is_blank" else "Out-of-scope"
            st.markdown(
                f"**{name}** ({info['time']:.1f}s) [{marker}]  \n"
                f"Answer: `{val}` — {ans}"
            )
            if info["explanation"] and info["explanation"] != "is_blank":
                st.caption(info["explanation"])
            st.divider()

    # Clickable reference links
    if agg["ref_id"]:
        links = []
        for rid in agg["ref_id"]:
            url = METADATA_URLS.get(rid)
            if url:
                links.append(f"[{rid}]({url})")
            else:
                links.append(rid)
        st.markdown("Sources: " + " · ".join(links))

    # First model's retrieval context (shared across models since same embedder+store)
    first_result = next(iter(model_results.values()))["result"]
    snippets = first_result.retrieval.snippets
    if snippets:
        display_snippets = snippets[:10]
        label = f"Retrieved context ({len(display_snippets)} of {len(snippets)} chunks)"
        with st.expander(label):
            for s in display_snippets:
                st.markdown(f"**#{s.rank}** _{s.document_title}_ (score: {s.score:.3f})")
                st.text(s.text[:500] + ("..." if len(s.text) > 500 else ""))
                st.divider()

    # Raw responses per model
    with st.expander("Raw LLM responses"):
        for name, info in agg["individual"].items():
            st.markdown(f"**{name}**")
            st.code(info["raw_response"], language="json")

    details = {
        "elapsed": elapsed,
        "ensemble": True,
        "strategy": strategy,
        "models": list(model_results.keys()),
        "answer": agg["answer"],
        "answer_value": agg["answer_value"],
    }
    if agg["explanation"] and agg["explanation"] != "is_blank":
        display_answer = agg["explanation"]
    elif agg["answer"] and agg["answer"] != "is_blank":
        display_answer = agg["answer"]
    else:
        display_answer = "Out-of-scope"
    st.session_state.messages.append({
        "role": "assistant", "content": display_answer, "details": details,
    })


def _render_details(details: dict):
    """Render expandable sections for a stored message (history replay)."""
    if details.get("ensemble"):
        # Minimal replay for ensemble messages
        cols = st.columns(3)
        cols[0].metric("Models", len(details.get("models", [])))
        cols[1].metric("Aggregation", details.get("strategy", ""))
        cols[2].metric("Total", f"{details.get('elapsed', 0):.1f}s")
        return

    timing = details.get("timing", {})
    elapsed = details.get("elapsed", 0)

    cols = st.columns(3)
    cols[0].metric("Retrieval", f"{timing.get('retrieval_s', 0):.1f}s")
    cols[1].metric("Generation", f"{timing.get('generation_s', 0):.1f}s")
    cols[2].metric("Total", f"{elapsed:.1f}s")

    ref_ids = details.get("ref_id", [])
    ref_urls = details.get("ref_url", [])
    if ref_ids and ref_ids != "is_blank":
        links = []
        for i, rid in enumerate(ref_ids if isinstance(ref_ids, list) else [ref_ids]):
            url = METADATA_URLS.get(rid)
            if not url:
                url = ref_urls[i] if isinstance(ref_urls, list) and i < len(ref_urls) else None
            if url and url != "is_blank":
                links.append(f"[{rid}]({url})")
            else:
                links.append(rid)
        st.markdown("Sources: " + " · ".join(links))
        sm = details.get("supporting_materials", "")
        if sm and sm != "is_blank":
            st.caption(f"Supporting: {sm}")

    snippets = details.get("snippets", [])
    if snippets:
        display_snippets = snippets[:10]
        label = f"Retrieved context ({len(display_snippets)} of {len(snippets)} chunks)"
        with st.expander(label):
            for s in display_snippets:
                st.markdown(f"**#{s['rank']}** _{s['title']}_ (score: {s['score']:.3f})")
                st.text(s["text"][:500] + ("..." if len(s["text"]) > 500 else ""))
                st.divider()

    raw = details.get("raw_response", "")
    if raw:
        with st.expander("Raw LLM response"):
            st.code(raw, language="json")


if __name__ == "__main__":
    main()
