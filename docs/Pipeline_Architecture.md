# KohakuRAG Pipeline Architecture

How the local HF pipeline replicates the four key ingredients of the original
KohakuRAG competition system.  All components are implemented in
`vendor/KohakuRAG/src/kohakurag/` and wired through `scripts/run_experiment.py`
and `scripts/run_ensemble.py`.

---

## 1) Structural Representation & Bottom-Up Embedding

### Four-level document tree

Every ingested document is parsed into a strict hierarchy:

```
Document  (doc_id)
  Section   (doc_id:sec0, doc_id:sec1, ...)
    Paragraph (doc_id:sec0:p0, ...)
      Sentence  (doc_id:sec0:p0:s0, ...)
```

**Key files:** `types.py` (NodeKind enum, payload dataclasses), `indexer.py`
(`_build_tree`, `_propagate_embeddings`).

### Length-weighted bottom-up embedding

Leaf nodes (sentences) are embedded first via the configured encoder (Jina v3,
Jina v4, or a local HF sentence-transformer).  Parent embeddings are then
computed as a **length-weighted average** of their children, so longer, more
informative segments contribute proportionally more:

```python
child_weights = [len(child.text) for child in node.children]
parent_embedding = average_embeddings(child_vectors, weights=child_weights)
```

**Key files:** `embeddings.py` (`average_embeddings`), `indexer.py`
(`_propagate_embeddings`).

### Multimodal integration (Jina v4)

Figures and tables are treated as paragraph-level nodes.  `JinaV4EmbeddingModel`
(`embeddings.py`) supports both text and native image embedding in a unified
vector space, enabling cross-modal retrieval.

---

## 2) Strategic Retrieval & Context Expansion

### Query planning

An LLM planner (`LLMQueryPlanner` in `pipeline.py`) expands a single user
question into *n* diverse retrieval queries (default 3) covering different
technical terminologies and sub-questions.

### Search target restriction

Dense vector search is restricted to **sentence** and **paragraph** nodes only
(`pipeline.py`, `kinds={NodeKind.SENTENCE, NodeKind.PARAGRAPH}`).  Section and
document nodes are too coarse for precise matching and are filtered out as
direct search targets.

### Hierarchical context expansion

Once a node is retrieved, the tree structure is used to pull in surrounding
context (`matches_to_snippets` in `pipeline.py`):

- Sentence match -> adds parent paragraph
- Paragraph match -> adds parent section

This ensures the LLM sees broad context while the initial search remains
hyper-focused.  An optional `no_overlap` flag removes redundant child nodes
whose parents are already in the result set.

### Consensus reranking

Results from all *n* queries are merged and reranked.  The default **combined**
strategy weighs both retrieval frequency (how many queries returned a node) and
cumulative similarity score:

```
combined = 0.4 * (freq / max_freq) + 0.6 * (score / max_score)
```

Other strategies: `frequency`, `score`.

---

## 3) Robust Answering & Ensemble Pipeline

### Prompt reordering (C -> Q)

To combat the "lost in the middle" effect, context can be placed **before** the
question in the user prompt.  Controlled by the config key
`use_reordered_prompt` (default `False`).  When enabled, the
`SYSTEM_PROMPT_REORDERED` and `USER_TEMPLATE_REORDERED` templates in
`scripts/run_experiment.py` are used instead of the default Q->C ordering.

### Retry mechanism (iterative deepening)

If the initial LLM response indicates insufficient evidence (`answer_value ==
"is_blank"`), the system automatically increases retrieval depth and re-runs:

1. First attempt: `top_k`
2. Retry 1: `top_k * 2`
3. Retry 2: `top_k * 3`
4. ...up to `max_retries` (default 3)

If a context-length overflow is detected, the system reduces `top_k` by 2 and
retries once.  Controlled by the config key `max_retries`.

### Ensemble inference

The ensemble runner (`scripts/run_ensemble.py`) supports two modes:

**Same-model ensemble** (`--config` + `--num-runs`): Runs *m* independent
inference passes of the **same model** as separate subprocesses (ensuring
completely independent random state), then aggregates via voting.  This is the
core KohakuRAG competition strategy.

```bash
python scripts/run_ensemble.py \
    --config vendor/KohakuRAG/configs/hf_qwen7b.py \
    --num-runs 5 --name qwen7b-ens5 --env GB10
```

**Cross-model ensemble** (`--experiments`): Aggregates results from different
pre-existing experiments.

### Voting strategies

| Strategy          | Algorithm |
|-------------------|-----------|
| `answer_priority` | Vote on `answer_value` first, then collect refs **only from matching runs** — ensures citation consistency (default) |
| `majority`        | Most common answer wins; refs are unioned across all runs |
| `first_non_blank` | First non-blank answer wins |

### Abstention-aware voting

If any run produces a non-blank answer, all "is_blank" runs are filtered out
before voting.  Enabled by default for same-model ensembles
(`--ignore-blank`).

---

## 4) Provenance & Citation Precision

### Hierarchical node IDs

Every node has a unique hierarchical ID: `doc_id:sec{n}:p{n}:s{n}`.  These IDs
are included in the LLM context as `[ref_id=doc_id]` markers and returned in
the structured answer for traceability.

### Citation flow

1. Context formatted with `[ref_id=...]` markers (`pipeline.py:format_snippets`)
2. LLM returns `ref_id` list in structured JSON
3. Post-processing resolves IDs to URLs/titles from `metadata.csv`
4. `answer_priority` voting ensures refs come only from runs that agree on the
   winning answer

---

## Config reference (pipeline keys)

| Key                      | Default     | Description |
|--------------------------|-------------|-------------|
| `top_k`                  | `8`         | Number of retrieval results per query |
| `planner_max_queries`    | `3`         | Number of diverse queries to generate |
| `deduplicate_retrieval`  | `True`      | Remove duplicate nodes across queries |
| `rerank_strategy`        | `"combined"`| Reranking: `combined`, `frequency`, or `score` |
| `top_k_final`            | `10`        | Max results after rerank (None = no limit) |
| `max_retries`            | `3`         | Iterative deepening retries on blank answers |
| `use_reordered_prompt`   | `False`     | C->Q prompt ordering (context before question) |
| `max_concurrent`         | `5`         | Max concurrent LLM calls |
| `retrieval_threshold`    | `0.25`      | Min similarity score to consider a match |
| `embedding_model`        | `"jinav4"`  | Embedding backend: `jina`, `jinav4`, `hf_local` |
