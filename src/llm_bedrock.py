"""
AWS Bedrock Integration for KohakuRAG

This module is the bridge between our RAG pipeline and AWS Bedrock. It lets us swap out
local models or OpenRouter for enterprise-grade Claude 3 models hosted on AWS.

Key features:
- Handles all the AWS auth complexity (SSO profiles, regions)
- Smart retry logic so we don't get crushed by rate limits
- Implements the standard `ChatModel` protocol so it drops right into existing pipelines

Usage:
    from src.llm_bedrock import BedrockChatModel
    
    # Initialize with your local AWS profile (setup via `aws sso login`)
    model = BedrockChatModel(
        profile_name="bedrock_nils",
        model_id="us.anthropic.claude-3-haiku-20240307-v1:0"  # Haiku is fast & cheap
    )
    
    # Fire off a query
    response = await model.complete("What's the energy cost of training GPT-3?")
"""

import asyncio
import json
import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol


@dataclass
class TokenUsage:
    """Track token usage for cost estimation."""
    input_tokens: int = 0
    output_tokens: int = 0

    def add(self, input_tokens: int, output_tokens: int) -> None:
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens

    def reset(self) -> None:
        self.input_tokens = 0
        self.output_tokens = 0


# ChatModel protocol - matches KohakuRAG's pipeline.py definition
class ChatModel(Protocol):
    """Protocol for chat backends."""

    async def complete(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
    ) -> str:  # pragma: no cover
        raise NotImplementedError


def _load_dotenv(path: str | Path = ".env") -> dict[str, str]:
    """Load environment variables from a .env file."""
    env_path = Path(path)
    if not env_path.exists():
        return {}

    env_vars: dict[str, str] = {}
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        env_vars[key.strip()] = value.strip().strip('"').strip("'")

    return env_vars


class BedrockChatModel:
    """
    A unified wrapper for AWS Bedrock's Chat API.
    
    This class handles the messy parts of working with Bedrock:
    1. Authentication via SSO profiles (so we don't need hardcoded keys)
    2. Rate limiting (exponential backoff when AWS says "slow down")
    3. Threading (boto3 is synchronous, so we offload it to keep the UI snappy)
    """

    def __init__(
        self,
        *,
        model_id: str = "us.anthropic.claude-3-haiku-20240307-v1:0",
        inference_profile_arn: str | None = None,
        profile_name: str | None = None,
        region_name: str | None = None,
        system_prompt: str | None = None,
        max_tokens: int = 4096,
        max_retries: int = 5,
        base_retry_delay: float = 3.0,
        max_concurrent: int = 10,
        experiment_tag: str | None = None,
    ) -> None:
        """
        Set up the Bedrock connection.

        Args:
            model_id: Which model to use. Used for request/response format detection.
                Defaults to Claude 3 Haiku (best balance of speed/cost).
            inference_profile_arn: Optional Application Inference Profile ARN for cost tracking.
                If set, API calls use this ARN (so AWS attributes costs to the profile's tags)
                while model_id is still used for request/response format detection.
                Example: 'arn:aws:bedrock:us-east-2:123456:application-inference-profile/abc123'
            profile_name: Your AWS SSO profile (e.g., 'bedrock_nils'). Falls back to AWS_PROFILE env var.
            region_name: The AWS region where the model is enabled. Defaults to us-east-2 (Ohio).
            system_prompt: The persona the model should adopt.
            max_tokens: Hard limit on response length. 4096 is usually plenty.
            max_retries: How many times to try again if AWS throttles us.
            base_retry_delay: Initial wait time between retries (increases exponentially).
            max_concurrent: How many parallel requests we allow before queuing. Critical for bulk processing.
            experiment_tag: Optional tag for cost tracking in AWS Cost Explorer (e.g., 'wattbot-sonnet-exp1').
        """
        # Lazy import boto3 to allow using the rest of the app without AWS dependencies
        try:
            import boto3
            self._boto3 = boto3
        except ImportError as e:
            raise ImportError(
                "Missing the 'boto3' library. You need it to talk to AWS. Run: pip install boto3"
            ) from e

        # Load environment variables
        dotenv_vars = _load_dotenv()

        # Resolve profile name
        self._profile_name = (
            profile_name
            or os.environ.get("AWS_PROFILE")
            or dotenv_vars.get("AWS_PROFILE")
        )

        # Resolve region
        self._region_name = (
            region_name
            or os.environ.get("AWS_REGION")
            or dotenv_vars.get("AWS_REGION")
            or "us-east-2"
        )

        # model_id is ALWAYS used for family detection (request/response format).
        # _invoke_model_id is what we pass to invoke_model() -- either the profile ARN
        # (for cost tracking) or the model_id itself.
        self._model_id = model_id
        self._inference_profile_arn = inference_profile_arn
        self._invoke_model_id = inference_profile_arn or model_id

        self._max_tokens = max_tokens
        self._system_prompt = system_prompt or "You are a helpful assistant."
        self._max_retries = max_retries
        self._base_retry_delay = base_retry_delay

        # Rate limiting semaphore
        self._semaphore = (
            asyncio.Semaphore(max_concurrent) if max_concurrent > 0 else None
        )

        # Experiment tag for cost tracking
        self._experiment_tag = experiment_tag

        # Token usage tracking
        self.token_usage = TokenUsage()

        # Create boto3 session and client
        self._session = self._boto3.Session(
            profile_name=self._profile_name,
            region_name=self._region_name,
        )
        self._client = self._session.client("bedrock-runtime")

    def _detect_model_family(self) -> str:
        """Detect the model family from the model ID."""
        model_lower = self._model_id.lower()

        if "anthropic" in model_lower or "claude" in model_lower:
            return "anthropic"
        elif "meta" in model_lower or "llama" in model_lower:
            return "meta"
        elif "mistral" in model_lower:
            return "mistral"
        elif "deepseek" in model_lower:
            return "deepseek"
        elif "amazon" in model_lower or "nova" in model_lower:
            return "amazon"
        elif "openai" in model_lower or "gpt" in model_lower:
            return "openai"
        elif "cohere" in model_lower:
            return "cohere"
        else:
            # Default to Anthropic format as fallback
            return "anthropic"

    def _build_request_body(self, prompt: str, system_prompt: str) -> dict:
        """Build the request body based on model family."""
        family = self._detect_model_family()

        if family == "anthropic":
            # Claude models use the Messages API
            # https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-anthropic-claude-messages.html
            return {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": self._max_tokens,
                "system": system_prompt,
                "messages": [
                    {"role": "user", "content": prompt}
                ],
            }

        elif family == "meta":
            # Llama models use a simple prompt format
            # https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-meta.html
            full_prompt = (
                "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
                f"{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
                f"{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
            )
            return {
                "prompt": full_prompt,
                "max_gen_len": self._max_tokens,
                "temperature": 0.1,
            }

        elif family == "mistral":
            # Mistral models
            # https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-mistral.html
            return {
                "prompt": f"<s>[INST] {system_prompt}\n\n{prompt} [/INST]",
                "max_tokens": self._max_tokens,
                "temperature": 0.1,
            }

        elif family == "deepseek":
            # DeepSeek models use Text Completion format
            # https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-deepseek.html
            return {
                "prompt": f"System: {system_prompt}\n\nUser: {prompt}\n\nAssistant:",
                "max_tokens": min(self._max_tokens, 8192),  # DeepSeek quality degrades above 8192
                "temperature": 0.1,
            }

        elif family == "openai":
            # OpenAI GPT-OSS models use standard OpenAI Chat Completions format
            # https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-openai.html
            return {
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                "max_completion_tokens": self._max_tokens,
                "temperature": 0.1,
            }

        elif family == "amazon":
            # Amazon Nova models use messages format
            # https://docs.aws.amazon.com/nova/latest/userguide/using-invoke-api.html
            return {
                "schemaVersion": "messages-v1",
                "messages": [
                    {"role": "user", "content": [{"text": prompt}]}
                ],
                "system": [{"text": system_prompt}],
                "inferenceConfig": {
                    "maxTokens": self._max_tokens,
                    "temperature": 0.1,
                }
            }

        else:
            # Fallback to Anthropic format
            return {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": self._max_tokens,
                "system": system_prompt,
                "messages": [
                    {"role": "user", "content": prompt}
                ],
            }

    def _parse_response(self, result: dict, headers: dict | None = None) -> tuple[str, int, int]:
        """Parse the response based on model family."""
        family = self._detect_model_family()

        if family == "anthropic":
            # Claude returns content array with text
            text = result["content"][0]["text"]
            usage = result.get("usage", {})
            input_tokens = usage.get("input_tokens", 0)
            output_tokens = usage.get("output_tokens", 0)
            return text, input_tokens, output_tokens

        elif family == "meta":
            # Llama returns generation directly
            text = result.get("generation", "")
            # Llama doesn't return token counts in the same way
            input_tokens = result.get("prompt_token_count", 0)
            output_tokens = result.get("generation_token_count", 0)
            return text, input_tokens, output_tokens

        elif family == "mistral":
            # Mistral returns outputs array
            outputs = result.get("outputs", [{}])
            text = outputs[0].get("text", "") if outputs else ""
            # Estimate tokens (Mistral doesn't always return counts)
            input_tokens = 0
            output_tokens = 0
            return text, input_tokens, output_tokens

        elif family == "deepseek":
            # DeepSeek returns choices[0].text format
            # https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-deepseek.html
            choices = result.get("choices", [])
            if choices:
                text = choices[0].get("text", "")
                # DeepSeek R1 may include <think>...</think> tags - strip them
                if "<think>" in text and "</think>" in text:
                    # Extract just the response part after </think>
                    think_end = text.find("</think>")
                    if think_end != -1:
                        text = text[think_end + len("</think>"):].strip()
            else:
                text = result.get("generation", "")
            # DeepSeek returns token counts in HTTP headers, not response body
            input_tokens = 0
            output_tokens = 0
            if headers:
                input_tokens = int(headers.get("x-amzn-bedrock-input-token-count", 0))
                output_tokens = int(headers.get("x-amzn-bedrock-output-token-count", 0))
            return text, input_tokens, output_tokens

        elif family == "openai":
            # OpenAI GPT-OSS: standard OpenAI Chat Completions response format
            # https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-openai.html
            choices = result.get("choices", [])
            if choices:
                message = choices[0].get("message", {})
                text = message.get("content", "")
                # GPT-OSS wraps chain-of-thought in <reasoning>...</reasoning> tags
                # Strip them to get the actual answer
                for tag in ["reasoning", "think"]:
                    open_tag = f"<{tag}>"
                    close_tag = f"</{tag}>"
                    if open_tag in text and close_tag in text:
                        tag_end = text.find(close_tag)
                        if tag_end != -1:
                            text = text[tag_end + len(close_tag):].strip()
            else:
                text = ""
            usage = result.get("usage", {})
            input_tokens = usage.get("prompt_tokens", 0)
            output_tokens = usage.get("completion_tokens", 0)
            return text, input_tokens, output_tokens

        elif family == "amazon":
            # Amazon Nova format - output is in message.content[0].text
            # https://docs.aws.amazon.com/nova/latest/userguide/using-invoke-api.html
            output = result.get("output", {})
            message = output.get("message", {})
            content = message.get("content", [{}])
            text = content[0].get("text", "") if content else ""
            usage = result.get("usage", {})
            input_tokens = usage.get("inputTokens", 0)
            output_tokens = usage.get("outputTokens", 0)
            return text, input_tokens, output_tokens

        else:
            # Fallback - try common patterns
            if "content" in result:
                text = result["content"][0]["text"]
            elif "generation" in result:
                text = result["generation"]
            elif "outputs" in result:
                text = result["outputs"][0].get("text", "")
            else:
                text = str(result)

            usage = result.get("usage", {})
            input_tokens = usage.get("input_tokens", 0)
            output_tokens = usage.get("output_tokens", 0)
            return text, input_tokens, output_tokens

    def _make_request_sync(self, prompt: str, system_prompt: str) -> tuple[str, int, int]:
        """
        The actual API call to Bedrock.

        Note: This is a synchronous (blocking) function because boto3 doesn't support async yet.
        We run this inside `asyncio.to_thread` elsewhere to prevent freezing the application.

        Returns:
            Tuple of (response_text, input_tokens, output_tokens)
        """
        body = self._build_request_body(prompt, system_prompt)

        response = self._client.invoke_model(
            modelId=self._invoke_model_id,
            body=json.dumps(body),
            contentType="application/json",
            accept="application/json",
        )

        result = json.loads(response["body"].read())
        # Pass HTTP headers for models that return token counts there (e.g. DeepSeek)
        headers = response.get("ResponseMetadata", {}).get("HTTPHeaders", {})
        return self._parse_response(result, headers=headers)

    async def complete(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
    ) -> str:
        """
        Send a message to Bedrock and get the response text.

        This method includes a robust retry loop to handle AWS "ThrottlingException" errors.
        It uses exponential backoff: if we get rated limited, we wait 3s, then 6s, then 12s...
        It also uses a semaphore to ensure we never have more than `max_concurrent` requests in flight.

        Token usage is accumulated in self.token_usage for cost tracking.

        Args:
            prompt: The text you want Claude to process.
            system_prompt: (Optional) Override the default system persona for this specific call.

        Returns:
            The raw text string from the model.
        """
        system = system_prompt or self._system_prompt

        for attempt in range(self._max_retries + 1):
            try:
                # Use semaphore for rate limiting if enabled
                if self._semaphore is not None:
                    async with self._semaphore:
                        response, input_tokens, output_tokens = await asyncio.to_thread(
                            self._make_request_sync, prompt, system
                        )
                else:
                    response, input_tokens, output_tokens = await asyncio.to_thread(
                        self._make_request_sync, prompt, system
                    )

                # Track token usage
                self.token_usage.add(input_tokens, output_tokens)

                return response

            except Exception as e:
                error_str = str(e).lower()

                # Check for retryable errors
                is_throttling = (
                    "throttling" in error_str
                    or "rate" in error_str
                    or "too many requests" in error_str
                    or "429" in error_str
                )
                is_server_error = any(
                    code in error_str for code in ["500", "502", "503", "504"]
                )
                is_retryable = is_throttling or is_server_error

                if not is_retryable or attempt >= self._max_retries:
                    raise  # Not retryable or exhausted retries

                # Exponential backoff with jitter
                wait_time = self._base_retry_delay * (2 ** attempt)
                jitter_factor = random.random() * 0.5 + 0.75  # 75-125%
                wait_time = wait_time * jitter_factor

                error_type = "Throttling" if is_throttling else "Server error"
                print(
                    f"Bedrock {error_type} (attempt {attempt + 1}/{self._max_retries + 1}). "
                    f"Retrying in {wait_time:.1f}s..."
                )
                await asyncio.sleep(wait_time)

        raise RuntimeError("Unexpected end of retry loop")

    def __repr__(self) -> str:
        parts = [f"model_id={self._model_id!r}"]
        if self._inference_profile_arn:
            parts.append(f"inference_profile_arn=...{self._inference_profile_arn[-20:]!r}")
        parts.append(f"region_name={self._region_name!r}")
        parts.append(f"profile_name={self._profile_name!r}")
        return f"BedrockChatModel({', '.join(parts)})"
