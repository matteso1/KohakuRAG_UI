# AWS Bedrock Integration Details

This document details the configuration, pricing, and technical implementation of the AWS Bedrock integration for KohakuRAG.

## Supported Models & Verified Pricing

All prices are for **US East (Ohio)** region, Standard On-Demand inference, verified against the [AWS Bedrock Pricing Page](https://aws.amazon.com/bedrock/pricing/) on **Feb 10, 2026**.

| Model Family | Model ID | Input Cost ($/1M) | Output Cost ($/1M) | Notes |
|--------------|----------|-------------------|--------------------|-------|
| **DeepSeek** | `us.deepseek.r1-v1:0` | **$1.35** | **$5.40** | Requires header-based token tracking (see below). |
| **Claude 3** | `claude-3-haiku-20240307-v1:0` | $0.80 | $4.00 | |
| **Claude 3.5** | `claude-3-5-sonnet-20240620-v1:0` | $3.00 | $15.00 | High accuracy, moderate cost. |
| **Claude 3.7** | `claude-3-7-sonnet-20250219-v1:0` | $3.00 | $15.00 | State-of-the-art reasoning. |
| **Llama 3** | `meta.llama3-70b-instruct-v1:0` | $0.72 | $0.72 | Balanced performance. |
| **Llama 4** | `meta.llama4-maverick-17b-v1:0` | $0.24 | $0.97 | "Maverick" variant. |
| **Llama 4** | `meta.llama4-scout-17b-v1:0` | $0.17 | $0.66 | "Scout" variant. |
| **GPT-OSS** | `openai.gpt-oss-120b-1:0` | $0.15 | $0.60 | Extremely cost-effective for size. |
| **GPT-OSS** | `openai.gpt-oss-20b-1:0` | $0.09 | $0.39 | Cheapest option. |

> **Note:** GPT-OSS 120B was previously estimated at ~$3.00/1M output, but verified pricing is significantly lower ($0.60/1M), making it a top contender for cost efficiency.

## DeepSeek R1 Implementation Details

DeepSeek R1 on Bedrock behaves differently from other models:

1. **Token Usage Metadata**: Standard Bedrock models return token usage in the JSON response body (`generation_token_count` etc.). DeepSeek R1 **does not**. Instead, it returns token counts in the HTTP response headers:
    * `x-amzn-bedrock-input-token-count`
    * `x-amzn-bedrock-output-token-count`

    We implemented a custom adapter in `src/llm_bedrock.py` that inspects these headers to ensure accurate cost tracking.

2. **Concurrency & Throttling**: DeepSeek R1 is a large reasoning model ("Deep Thinking"). usage on Bedrock is strictly rate-limited.
    * **Recommended Concurrency**: `1` (Sequential processing).
    * Attributes like `max_concurrent` in configs should be set to 1.
    * Higher concurrency (e.g., 2+) results in `ThrottlingException` and infinite retry loops.

## Cost Efficiency Analysis

We replaced the generic "Size vs Performance" plot with a "Cost Efficiency" ranking, using the formula:
$$ \text{Efficiency} = \frac{\text{WattBot Score (0-1)}}{\text{Total Benchmark Cost (\$)}}$$

**Key Findings:**

* **GPT-OSS 20B** and **120B** are the most cost-efficient models due to extremely low pricing.
* **DeepSeek R1** offers competitive performance but is ~3-4x more expensive per query than GPT-OSS.
* **Claude 3.5 Sonnet** remains the accuracy leader but at ~10x the cost of OSS models.
