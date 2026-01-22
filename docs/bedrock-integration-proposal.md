# AWS Bedrock Integration Proposal

**Status**: Completed ✅  
**Date**: January 22, 2026  
**Author**: Nils Matteson  

## Overview

This document outlines the design and implementation of integrating AWS Bedrock as an LLM backend for the WattBot (KohakuRAG) pipeline. This allows us to use managed foundation models (e.g., Claude 3) without managing local inference servers.

## Implementation Details

### 1. Architecture

- **Class**: `BedrockChatModel` (in `src/llm_bedrock.py`)
- **Protocol**: Implements KohakuRAG's `ChatModel` protocol (`async def complete()`)
- **SDK**: `boto3` + `bedrock-runtime` client
- **Authentication**: AWS SSO profile (`bedrock_nils`)

### 2. Key Features

- **Rate Limiting**: Custom exponential backoff algorithm with jitter to handle Bedrock throttling.
- **Concurrency Control**: Async semaphore to limit parallel requests (default: 10).
- **Security**: No long-lived keys; uses temporary SSO credentials.
- **Model Support**: Currently configured for `anthropic.claude-3-haiku-20240307-v1:0` (fast/cheap) but supports any text-generation model in Bedrock.

### 3. GPU Acceleration

- **Indexing**: Jina embeddings (v3/v4) run on local GPU (RTX 4090) via PyTorch + CUDA 12.1.
- **Inference**: Offloaded to AWS Bedrock (no local GPU load for generation).

## Setup Instructions

See [README.md](../README.md) for detailed setup steps, including CUDA installation and AWS SSO configuration.

## Verification

The integration has been verified with:

1. **Unit Tests**: `scripts/test_bedrock_model.py` checks connectivity, system prompts, and concurrency.
2. **End-to-End RAG**: `scripts/demo_bedrock_rag.py` runs a full retrieval + answer generation loop using the WattBot 2025 dataset.

## Future Work

- [ ] Add support for Bedrock Knowledge Bases (fully managed RAG)
- [ ] Experiment with Claude 3.5 Sonnet for harder queries
- [ ] Compare cost/performance vs. local llama4-8b models
