# Bedrock Integration Progress Report
**Author**: Nils Matteson
**Date**: January 21-25, 2026
**Branch**: `bedrock`

---

## Executive Summary

Completed full AWS Bedrock integration for the KohakuRAG pipeline. The system is now **end-to-end functional** with both JinaV3 and JinaV4 embedding support. Ready for UI integration with Blaise's Streamlit frontend.

---

## Accomplishments This Week

### 1. AWS Bedrock Integration (Complete)

Built a production-ready `BedrockChatModel` class that:
- Uses **AWS SSO authentication** (no hardcoded secrets)
- Implements **exponential backoff with jitter** for rate limiting
- Controls concurrency with async semaphores
- Is a **drop-in replacement** for OpenRouter/OpenAI in the pipeline

```python
# Usage is simple
model = BedrockChatModel(
    profile_name="bedrock_nils",
    model_id="us.anthropic.claude-3-haiku-20240307-v1:0"
)
response = await model.complete("What is the carbon footprint of LLMs?")
```

### 2. JinaV4 Index Built

Built the JinaV4 multimodal embedding index to match the winning solution:
- `artifacts/wattbot_jinav4.db` (82MB)
- 512-dimensional embeddings
- Supports both averaged and full paragraph embeddings

### 3. Full Pipeline Integration

Modified `wattbot_answer.py` to support:
- `--llm_provider bedrock` flag
- Configurable retrieval settings (top_k, reranking, deduplication)
- JinaV4 embedding model selection
- Windows encoding fixes

---

## Architecture

```
User Query (from Streamlit UI)
        |
        v
+------------------+
|   RAG Pipeline   |
|   (KohakuRAG)    |
+------------------+
        |
        +---> JinaV4 Embeddings (local GPU)
        |
        +---> SQLite Vector Store (wattbot_jinav4.db)
        |
        +---> AWS Bedrock
                  |
                  +---> Claude 3 Haiku (testing)
                  +---> Claude 3.5 Sonnet (production)
                  +---> Claude Opus 4.5 (available)
        |
        v
Structured Answer + Citations
```

---

## Benchmark Results

| Configuration | Score | Notes |
|--------------|-------|-------|
| JinaV3 + Bedrock Haiku | **0.665** | Current baseline |
| JinaV4 + Bedrock Haiku | 0.559 | Context limit issues |
| Winning Solution | 0.861 | GPT-OSS-120B + 9x ensemble |

### Why the Gap?

The winning solution used:
1. **Larger model** (GPT-OSS-120B vs Haiku)
2. **9x ensemble voting** (we run 1x)
3. **Higher context** (top_k=16 vs our top_k=8 due to Haiku limits)

### Path to 0.80+

- Use **Claude 3.5 Sonnet** (larger context window, smarter)
- Increase `top_k` to 16 with Sonnet's context
- Consider ensemble voting for critical queries

---

## Available Bedrock Models

| Model | Use Case | Cost |
|-------|----------|------|
| Claude 3 Haiku | Testing, low-cost | $ |
| Claude 3.5 Sonnet | Production recommended | $$ |
| Claude Opus 4.5 | Maximum quality | $$$ |

All models verified accessible on account `183295408236`.

---

## Files Delivered

| File | Description |
|------|-------------|
| `src/llm_bedrock.py` | Bedrock integration (245 lines) |
| `scripts/demo_bedrock_rag.py` | E2E demo script |
| `scripts/run_wattbot_eval.py` | Batch evaluation |
| `configs/jinav4_index.py` | JinaV4 index config |
| `artifacts/wattbot_jinav4.db` | JinaV4 vector index |

---

## Ready for Blaise's UI

The backend is ready to integrate:

```python
# Blaise can call this from Streamlit
from llm_bedrock import BedrockChatModel
from kohakurag import RAGPipeline

# Initialize once
pipeline = RAGPipeline(
    store=store,           # Load wattbot_jinav4.db
    embedder=embedder,     # JinaV4
    chat_model=BedrockChatModel(...)
)

# Per user query
result = await pipeline.run_qa(
    question=user_input,
    top_k=8,
    ...
)
# result.answer contains the response + citations
```

---

## Next Steps

1. **Blaise**: Connect Streamlit UI to Bedrock backend
2. **Nils**: Test with Claude 3.5 Sonnet for better scores
3. **Team**: Demo at Tuesday meeting
4. **Future**: Implement ensemble voting for production

---

## Demo Commands

```bash
# Re-authenticate AWS (if needed)
aws sso login --profile bedrock_nils

# Run single question demo
python scripts/demo_bedrock_rag.py

# Run full evaluation
python KohakuRAG/scripts/wattbot_answer.py \
  --db artifacts/wattbot_jinav4.db \
  --questions data/train_QA.csv \
  --output artifacts/submission.csv \
  --llm_provider bedrock
```
