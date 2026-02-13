# Bedrock Backend Usage Guide

For Blaise - how to use the Bedrock integration in Streamlit.

## First: Set Up Your AWS Profile

Create your own SSO profile so your usage is tracked separately.

### 1. Configure AWS CLI

Run:
```bash
aws configure sso
```

Enter these values when prompted:
```
SSO session name: bedrock_blaise
SSO start URL: https://d-9067aa9c10.awsapps.com/start
SSO region: us-east-1
SSO registration scopes: [press Enter for default]
```

It will open a browser for UW login. After auth, you'll see available accounts. Pick the Bedrock account (`183295408236`).

When asked for CLI profile name, enter: `bedrock_blaise`

### 2. Verify It Works

```bash
aws sso login --profile bedrock_blaise
aws sts get-caller-identity --profile bedrock_blaise
```

You should see your email in the output.

---

## Quick Start

### 1. Get the Vector Index

```bash
aws s3 cp s3://wattbot-nils-kohakurag/indexes/wattbot_jinav4.db artifacts/wattbot_jinav4.db --profile bedrock_blaise
```

### 2. AWS Auth

Before running anything:
```bash
aws sso login --profile bedrock_blaise
```

This gives you temporary credentials for ~8 hours.

### 3. Test It Works

```bash
python scripts/demo_bedrock_rag.py --question "What is the carbon footprint of GPT-3?"
```

If you get an answer with citations, you're good.

## Using in Streamlit

### Imports

```python
import sys
import asyncio

sys.path.insert(0, "src")
sys.path.insert(0, "KohakuRAG/src")

from llm_bedrock import BedrockChatModel
from kohakurag.datastore import KVaultNodeStore
from kohakurag.embeddings import JinaEmbeddingModel
```

### Initialize (Once at App Startup)

```python
# Bedrock client
chat = BedrockChatModel(
    profile_name="bedrock_blaise",
    region_name="us-east-2",
    model_id="us.anthropic.claude-3-haiku-20240307-v1:0",
    max_concurrent=3
)

# Vector store
store = KVaultNodeStore(
    path="artifacts/wattbot.db",
    table_prefix="wattbot",
    dimensions=1024
)

# Embeddings
embedder = JinaEmbeddingModel()
```

### Handle a Query

```python
async def answer_question(question: str) -> dict:
    # 1. Embed the question
    query_embedding = await embedder.embed([question])

    # 2. Search for relevant passages
    results = store.search_sync(query_embedding[0], k=5)

    # 3. Build context
    context = "\n\n".join([node.content for node, score in results])
    sources = [node.metadata.get("doc_id", "unknown") for node, score in results]

    # 4. Generate answer
    prompt = f"""Answer the question based on this context.

Context:
{context}

Question: {question}

Provide a clear answer with citations."""

    answer = await chat.complete(prompt)

    return {
        "answer": answer,
        "sources": list(set(sources))
    }
```

### Streamlit Integration

```python
import streamlit as st

st.title("WattBot")

question = st.text_input("Ask a question:")

if question:
    with st.spinner("Thinking..."):
        result = asyncio.run(answer_question(question))

    st.write(result["answer"])

    st.caption(f"Sources: {', '.join(result['sources'])}")
```

## API Reference

### BedrockChatModel

```python
BedrockChatModel(
    profile_name="bedrock_blaise",  # AWS SSO profile
    region_name="us-east-2",       # AWS region
    model_id="us.anthropic.claude-3-haiku-20240307-v1:0",  # Model
    max_concurrent=3,              # Parallel requests (keep low)
    max_retries=5,                 # Retry on throttle
    base_retry_delay=3.0           # Seconds between retries
)
```

**Methods:**
- `await chat.complete(prompt)` - Returns string response
- `await chat.complete_with_system(system_prompt, user_prompt)` - With system message

### Available Models

| Model ID | Name | Cost |
|----------|------|------|
| `us.anthropic.claude-3-haiku-20240307-v1:0` | Haiku | $0.003/query |
| `us.anthropic.claude-3-5-sonnet-20241022-v2:0` | Sonnet 3.5 | $0.03/query |
| `us.anthropic.claude-3-7-sonnet-20250219-v1:0` | Sonnet 3.7 | $0.03/query |

Start with Haiku. Switch to Sonnet if you need better answers.

### KVaultNodeStore

```python
store = KVaultNodeStore(
    path="artifacts/wattbot.db",   # Path to SQLite file
    table_prefix="wattbot",        # Table name prefix
    dimensions=1024                # Embedding dimensions (1024 for JinaV3)
)
```

**Methods:**
- `store.search_sync(embedding, k=5)` - Returns list of (node, score) tuples
- `await store.search(embedding, k=5)` - Async version

### JinaEmbeddingModel

```python
embedder = JinaEmbeddingModel()
```

**Methods:**
- `await embedder.embed(["text1", "text2"])` - Returns list of embeddings

## Common Issues

### "Token has expired"
Run `aws sso login --profile bedrock_blaise` again.

### "ThrottlingException"
The retry logic handles this automatically. If persistent, reduce `max_concurrent`.

### Slow first query
First query loads the embedding model (~2GB). Subsequent queries are fast.

## Files

- `src/llm_bedrock.py` - Bedrock integration
- `scripts/demo_bedrock_rag.py` - Full working example
- `artifacts/wattbot.db` - JinaV3 index (30MB)
- `artifacts/wattbot_jinav4.db` - JinaV4 index (82MB)

## Questions

Ping Nils on Slack.
