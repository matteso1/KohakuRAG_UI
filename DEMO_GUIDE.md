# Tuesday Demo Guide - Bedrock Integration

## Before You Start

1. Make sure you're logged into AWS SSO:
```bash
aws sso login --profile bedrock_nils
```

2. Navigate to the project:
```bash
cd "C:\Users\nilsm\Desktop\VSCODE PROJECTS\KohakuRAG\KohakuRAG_UI"
```

---

## The Demo

### What You're Showing
**"I ported the winning KohakuRAG solution to AWS Bedrock"**

### Run This Command
```bash
python scripts/demo_bedrock_rag.py --question "What is the carbon footprint of training GPT-3?"
```

### What Happens (narrate as it runs)
1. **"It loads the vector index"** - 82MB SQLite database with ~200 research papers
2. **"Embeds the question using JinaV3"** - Converts text to 1024-dim vector
3. **"Searches for relevant passages"** - Semantic search finds top matches
4. **"Calls Claude 3 on AWS Bedrock"** - Sends context + question
5. **"Returns structured answer with citations"** - Clean JSON output

### Expected Output
```
Answer: The carbon footprint of training GPT-3 is over 550 metric tons of CO2 equivalent.
Sources: ['jegham2025']
```

---

## Key Talking Points

### What You Built
1. **BedrockChatModel** (`src/llm_bedrock.py`)
   - Drop-in replacement for OpenRouter
   - AWS SSO authentication (no hardcoded keys)
   - Exponential backoff for rate limits
   - Async/await for concurrent requests

2. **JinaV4 Index on S3**
   - `s3://wattbot-nils-kohakurag/indexes/wattbot_jinav4.db`
   - Ready for Blaise to download
   - 512-dim embeddings, 82MB file

3. **Working End-to-End**
   - Question → Retrieval → Bedrock → Answer
   - Structured output with citations
   - Production-ready code

### For Blaise (most important)

**"Here's what you need from me:"**

1. **The Vector Index**
   ```bash
   aws s3 cp s3://wattbot-nils-kohakurag/indexes/wattbot_jinav4.db artifacts/
   ```

2. **The Code Snippet** (show from progress report)
   ```python
   from llm_bedrock import BedrockChatModel

   # Initialize once in Streamlit
   chat = BedrockChatModel(
       profile_name="bedrock_nils",
       region_name="us-east-2",
       model_id="us.anthropic.claude-3-haiku-20240307-v1:0"
   )

   # Per user query
   answer = await chat.complete(prompt)
   ```

3. **The Architecture** (point to diagram in progress report)

### Scores (if asked)
| Config | Score | Notes |
|--------|-------|-------|
| JinaV3 + Haiku | 0.665 | Current working baseline |
| Winning solution | 0.861 | Used GPT-OSS-120B + 9x ensemble |

**"We're close. The gap is mainly the model (Haiku vs larger) and ensemble voting."**

---

## Next Steps Discussion

### Immediate (This Week)
1. **Blaise**: Integrate Bedrock backend into Streamlit
2. **Team**: Review cost monitoring strategy

### Optional (Future)
1. **Score optimization**: Try Claude 3.5 Sonnet (but costs money)
2. **Ensemble voting**: Run 5x in parallel and aggregate (increases cost 5x)
3. **On-prem comparison**: Test on GB10 hardware

### Cost Reality Check
- Current demo: ~$0.01 per question
- Full ensemble (5 runs): ~$0.05 per question
- Production with Haiku: manageable
- Production with Sonnet: 5-10x more

**"We should decide as a team if score optimization is worth the cost."**

---

## If Something Goes Wrong

### AWS SSO Token Expired
```bash
aws sso login --profile bedrock_nils
```

### Can't Find the DB
```bash
ls -lh artifacts/wattbot.db
# Should show ~30MB file
```

### Demo Script Fails
Fall back to the progress report and talk through the architecture.

---

## Files to Reference

1. **Progress Report**: `docs/tuesday-progress-report.md`
2. **Demo Script**: `scripts/demo_bedrock_rag.py`
3. **Bedrock Integration**: `src/llm_bedrock.py`
4. **JinaV4 Config**: `configs/jinav4_index.py`

---

## Confidence Boosters

**What works:**
- ✅ Bedrock integration is solid
- ✅ Pipeline runs end-to-end
- ✅ Returns structured answers with citations
- ✅ Index is on S3 for team access
- ✅ Code is clean and documented

**What you delivered:**
- Production-ready backend
- Clear handoff for Blaise
- Cost-conscious approach
- Working demo

**You're ready. The hard work is done.**
