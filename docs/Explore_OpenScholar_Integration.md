# OpenScholar Integration Exploration

**Date:** 2026-03-12
**Status:** Exploration / RFC
**Paper:** Asai et al. (2024) "OpenScholar: Synthesizing Scientific Literature with Retrieval-Augmented LMs" (Nature, 2024)
**Code:** https://github.com/AkariAsai/OpenScholar
**Models:** https://huggingface.co/OpenSciLM

---

## What Is OpenScholar?

OpenScholar is a retrieval-augmented LM specifically designed to answer scientific queries by retrieving relevant passages from 45M open-access papers and synthesizing citation-backed responses. Published in Nature, it outperforms GPT-4o by 5% and PaperQA2 by 7% in correctness on their ScholarQABench benchmark. It is fully open-source (Apache 2.0).

### Architecture (4 components)

| Component | Details | Size |
|-----------|---------|------|
| **Datastore (OSDS)** | 45M open-access papers from peS2o/Semantic Scholar, split into 250-word passages | 237M passage embeddings |
| **Retriever** | Contriever (110M params) continually pre-trained on peS2o | ~440MB |
| **Reranker** | BGE-reranker-large (340M params) fine-tuned on synthetic scientific data | ~0.6B params |
| **Generator LM** | Llama 3.1 8B fine-tuned on 130K instances (50% scientific, 50% general) | 8B params |

### Inference Pipeline

1. **Multi-source retrieval**: queries peS2o datastore (bi-encoder), Semantic Scholar API (keyword search), and web search (You.com API)
2. **Reranking**: cross-encoder scores all candidates, applies meta-filtering (max 3 passages/paper, citation count normalization)
3. **Initial generation**: LM generates response y0 with inline citations
4. **Self-feedback loop** (up to 3 iterations): LM generates feedback on its own output, retrieves additional papers if needed, refines response
5. **Citation verification**: post-hoc check that all citation-worthy statements have supporting references

---

## How Our Pipeline Compares

| Aspect | KohakuRAG | OpenScholar |
|--------|-----------|-------------|
| **Corpus** | Curated domain-specific (e.g., WattBot energy papers) | 45M open-access papers (all fields) |
| **Embedding** | Jina V3/V4 (multimodal) or Bedrock Titan V2 | Contriever (text-only, science-tuned) |
| **Vector Store** | KohakuVault (SQLite + sqlite-vec, single-file) | FAISS index (~237M embeddings, massive memory) |
| **Retrieval** | Query planner → multi-query → dedup → rerank (frequency/score/combined) | Bi-encoder + S2 API + web search → cross-encoder reranker |
| **Generation** | Any LLM (Bedrock, HF local, OpenRouter, vLLM) | OpenScholar-8B or off-the-shelf LM with self-feedback loop |
| **Post-processing** | Structured JSON output with ref_ids | Free-form text with inline citations + citation verification |
| **Self-feedback** | None (single-pass generation) | Iterative refinement with additional retrieval per feedback round |

---

## Integration Options (Ranked by Effort)

### Option 1: Add OpenScholar-8B as a Generator Model (LOW EFFORT)

**What:** Add `OpenSciLM/Llama-3.1_OpenScholar-8B` as another HF local model config, just like our existing Llama 3.1 8B config.

**How it connects:**
- Drops into existing `HFLocalChatModel` — it's a standard Llama 3.1 8B fine-tune
- Uses our existing Jina V4 embeddings + KohakuVault retrieval
- No changes to database, retrieval, or UI

**What we'd get:**
- A model specifically trained for scientific literature synthesis with better citation behavior
- Better at grounding responses in retrieved context (trained on RAG-specific data)
- 100% compatible with our existing 4-bit quantization support (~6GB VRAM)

**What we'd miss:**
- No self-feedback loop (single-pass only)
- No science-tuned retriever/reranker
- Model was trained for open-domain scientific QA — unclear how well it generalizes to our domain-specific structured output format (JSON with answer_value, ref_id, etc.)

**Effort:** ~1 hour (new config file + test)

**New config file:** `vendor/KohakuRAG/configs/hf_openscholar_8b.py`

### Option 2: Add Semantic Scholar API as Secondary Retrieval Source (MEDIUM EFFORT)

**What:** When a query looks like it needs broader scientific literature context, supplement our KohakuVault retrieval with results from the Semantic Scholar API.

**How it connects:**
- Add a new retrieval source in `pipeline.py` alongside the existing vector search
- Generate search keywords from the query (LLM-based, like OpenScholar does)
- Fetch paper abstracts via S2 API (free, rate-limited, requires API key)
- Merge S2 results with our local retrieval results before generation

**What we'd get:**
- Access to 200M+ papers without hosting any datastore
- Broader coverage for questions that go beyond our curated corpus
- Real paper metadata (titles, authors, DOIs, citation counts)

**What we'd miss:**
- Only abstracts (not full text) for most papers
- Rate-limited (100 requests/sec with API key)
- Adds network dependency and latency

**Effort:** ~1-2 days

**Key question:** Do we want this always-on, or as a toggleable "deep search" mode in the UI?

### Option 3: Add OpenScholar Reranker (MEDIUM EFFORT)

**What:** Use `OpenSciLM/OpenScholar_Reranker` (fine-tuned BGE-reranker, 0.6B params) as a cross-encoder reranking stage after our vector retrieval.

**How it connects:**
- Currently we rerank by frequency/score/combined heuristics in `pipeline.py`
- This would add a learned cross-encoder scoring step between retrieval and generation
- Runs on GPU, ~2GB VRAM for the 0.6B model

**What we'd get:**
- More accurate relevance scoring than our heuristic reranking
- Better at filtering out irrelevant passages (cross-encoder sees query + passage jointly)
- The paper shows reranker removal causes the largest performance drop in ablations

**What we'd miss:**
- Adds GPU memory overhead
- Trained on scientific data — may not transfer perfectly to all domains
- Adds inference latency (~50-200ms per reranking batch)

**Effort:** ~2-3 days (new reranker module + integration into pipeline)

### Option 4: Self-Feedback Inference Loop (HIGH EFFORT)

**What:** Implement OpenScholar's iterative self-feedback generation: generate → get LM feedback → retrieve more → refine → verify citations.

**How it connects:**
- Wraps the generation step in `pipeline.py` with a feedback loop
- Each iteration: LM critiques its own output, optionally generates a new retrieval query, then rewrites
- Adds citation verification as a post-processing step

**What we'd get:**
- Higher quality responses with better citation coverage
- Adaptive — the model decides when it needs more information
- The paper shows this improves correctness by ~3-5% and citation F1 significantly

**What we'd miss:**
- 2-4x slower inference (multiple LLM calls per query)
- Significantly more complex pipeline
- Requires prompt engineering for feedback generation and incorporation
- Works best with models trained for it (OpenScholar-8B)

**Effort:** ~1-2 weeks

### Option 5: Host the Full OpenScholar DataStore (HIGH EFFORT, LIKELY OVERKILL)

**What:** Download and host the peS2o v3 datastore (45M papers, 237M embeddings).

**Why probably not:**
- The datastore is ~150M+ rows on HuggingFace — massive download and storage
- Requires significant CPU memory for the FAISS index (the authors note this themselves)
- Would need to either convert to KohakuVault format or run a separate retrieval service
- Only valuable if our use case truly requires searching across all of scientific literature

**When it makes sense:**
- If we're building a general-purpose scientific literature assistant (not domain-specific)
- If the Semantic Scholar API (Option 2) proves too limited

---

## Do We Need All of arXiv?

**Short answer: No.**

OpenScholar's datastore is peS2o (from Semantic Scholar's S2ORC), which is broader than arXiv — it includes 45M open-access papers from many publishers. But we don't need to replicate this because:

1. **Our current use case is domain-specific** (energy/sustainability). A curated corpus outperforms a massive general one for focused domains.
2. **Semantic Scholar API** gives us on-demand access to the broader literature without hosting anything.
3. **The paper shows** that even their retriever (trained on peS2o) works well when pointed at a different datastore version, suggesting the retrieval models generalize.

The value of OpenScholar isn't the datastore size — it's the science-tuned model, reranker, and self-feedback loop.

---

## Recommended Integration Path

**Phase 1 (Quick Win):** Options 1 + 2
- Add OpenScholar-8B as a generator model config
- Add Semantic Scholar API as an optional secondary retrieval source (toggle in sidebar)
- Gives us the best model + broader search capability with minimal effort

**Phase 2 (If Phase 1 shows promise):** Option 3
- Add the OpenScholar Reranker for improved passage selection
- This is the component that showed the largest impact in ablation studies

**Phase 3 (If we want to invest):** Option 4
- Implement the self-feedback loop for highest-quality responses
- Consider this only if the use case justifies the latency tradeoff

---

## Hardware Requirements

| Component | VRAM | CPU RAM | Disk |
|-----------|------|---------|------|
| OpenScholar-8B (4-bit) | ~6 GB | 2 GB | ~5 GB |
| OpenScholar-8B (bf16) | ~16 GB | 4 GB | ~16 GB |
| OpenScholar Reranker | ~2 GB | 1 GB | ~1.2 GB |
| S2 API | None | None | None |
| Full DataStore V3 | None | 100+ GB | ~500 GB+ |

The 8B model + reranker can comfortably fit on a single GPU alongside Jina V4 embeddings.

---

## Open Questions

1. **Structured output compatibility:** OpenScholar-8B is trained for free-form scientific synthesis. Will it reliably produce our structured JSON format (answer_value, ref_id, etc.), or do we need a separate prompt/fine-tune?
2. **Domain transfer:** The model is trained on general scientific literature. How well does it handle domain-specific questions (e.g., energy infrastructure) vs. broad scientific review questions?
3. **Prompt format:** OpenScholar uses specific prompts for retrieval-augmented generation. Do we need to adapt our prompt templates, or can we use our existing ones?
4. **S2 API key limits:** The free tier allows 100 req/sec. Is this sufficient for our expected query volume?
5. **Citation format:** OpenScholar generates inline citations like [1], [2]. Our pipeline expects structured ref_id lists. Need to decide how to bridge this.

---

## References

- Paper: Asai et al. (2024) "OpenScholar: Synthesizing Scientific Literature with Retrieval-Augmented LMs" — Nature
- Code: https://github.com/AkariAsai/OpenScholar
- Models: https://huggingface.co/OpenSciLM
- Demo: https://openscholar.allen.ai
- Datastore V3: https://huggingface.co/datasets/OpenSciLM/OpenScholar-DataStore-V3
- Semantic Scholar API: https://api.semanticscholar.org/
