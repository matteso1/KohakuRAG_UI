"""Chat model integrations (OpenAI, OpenRouter, HuggingFace local, AWS Bedrock)."""

import asyncio
import base64
import json as _json
import os
import random
import re
from pathlib import Path

from openai import AsyncOpenAI, RateLimitError

try:
    from openrouter import OpenRouter

    OPENROUTER_AVAILABLE = True
except ImportError:
    OPENROUTER_AVAILABLE = False

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    HF_LOCAL_AVAILABLE = True
except ImportError:
    HF_LOCAL_AVAILABLE = False

try:
    import boto3
    from botocore.config import Config as BotoConfig
    from botocore.exceptions import ClientError as BotoClientError

    BEDROCK_AVAILABLE = True
except ImportError:
    BEDROCK_AVAILABLE = False

from .pipeline import ChatModel


def _load_dotenv(path: str | Path = ".env") -> dict[str, str]:
    """Load environment variables from a .env file."""
    env_path = Path(path)
    if not env_path.exists():
        return {}

    env_vars: dict[str, str] = {}
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()

        # Skip empty lines and comments
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        env_vars[key.strip()] = value.strip().strip('"').strip("'")

    return env_vars


class OpenAIChatModel(ChatModel):
    """Chat backend powered by OpenAI's Chat Completions API with automatic rate limit handling."""

    def __init__(
        self,
        *,
        model: str = "gpt-4o-mini",
        api_key: str | None = None,
        organization: str | None = None,
        system_prompt: str | None = None,
        max_retries: int = 5,
        base_retry_delay: float = 3.0,
        base_url: str | None = None,
        max_concurrent: int = 10,
    ) -> None:
        """Initialize OpenAI chat model with automatic rate limit retry.

        Args:
            model: OpenAI model identifier (e.g., "gpt-4o-mini")
            api_key: OpenAI API key (reads from env if not provided)
            organization: OpenAI organization ID (optional)
            system_prompt: Default system message for all completions
            max_retries: Maximum retry attempts on rate limit errors
            base_retry_delay: Base delay for exponential backoff (seconds)
            base_url: Optional override for the API base URL (e.g., for
                self-hosted or proxy OpenAI-compatible endpoints). If not
                provided, falls back to the OPENAI_BASE_URL environment
                variable or .env file when present.
            max_concurrent: Maximum number of concurrent API requests (default: 10).
                Set to 0 or negative to disable rate limiting (unlimited concurrency).
        """
        dotenv_vars: dict[str, str] | None = None

        # Try multiple sources for API key
        key = api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            dotenv_vars = _load_dotenv()
            key = dotenv_vars.get("OPENAI_API_KEY")

        if not key:
            raise ValueError("OPENAI_API_KEY is required for OpenAIChatModel.")

        # Resolve base URL for OpenAI-compatible endpoints
        resolved_base_url = base_url
        if resolved_base_url is None:
            env_base_url = os.environ.get("OPENAI_BASE_URL")
            if env_base_url is None:
                if dotenv_vars is None:
                    dotenv_vars = _load_dotenv()
                env_base_url = dotenv_vars.get("OPENAI_BASE_URL")
            resolved_base_url = env_base_url

        self._system_prompt = system_prompt or "You are a helpful assistant."
        self._client = AsyncOpenAI(
            api_key=key,
            organization=organization,
            base_url=resolved_base_url,
        )
        self._model = model
        self._max_retries = max_retries
        self._base_retry_delay = base_retry_delay
        # Only create semaphore if max_concurrent > 0 (rate limiting enabled)
        self._semaphore = (
            asyncio.Semaphore(max_concurrent) if max_concurrent > 0 else None
        )

    def _parse_retry_after(self, error_message: str) -> float | None:
        """Extract wait time from rate limit error message.

        Handles formats like:
        - "Please try again in 23ms"
        - "Please try again in 1.5s"
        - "Please try again in 2m"
        """
        patterns = [
            (r"try again in (\d+(?:\.\d+)?)ms", 0.001),  # milliseconds
            (r"try again in (\d+(?:\.\d+)?)s", 1.0),  # seconds
            (r"try again in (\d+(?:\.\d+)?)m", 60.0),  # minutes
        ]
        for pattern, multiplier in patterns:
            match = re.search(pattern, error_message, re.IGNORECASE)
            if match:
                value = float(match.group(1))
                return value * multiplier
        return None

    async def complete(self, prompt: str, *, system_prompt: str | None = None) -> str:
        """Execute chat completion with automatic rate limit retry.

        Uses intelligent retry strategy:
        1. Semaphore limits concurrent requests (if enabled)
        2. Parse server-recommended delay from error message
        3. Fall back to exponential backoff if no delay specified
        4. Apply jitter to avoid thundering herd

        Returns:
            Model's text response
        """
        system = system_prompt or self._system_prompt

        for attempt in range(self._max_retries + 1):
            try:
                # Use semaphore only if rate limiting is enabled
                if self._semaphore is not None:
                    async with self._semaphore:
                        response = await self._client.chat.completions.create(
                            model=self._model,
                            messages=[
                                {"role": "system", "content": system},
                                {"role": "user", "content": prompt},
                            ],
                        )
                else:
                    # No rate limiting - make request directly
                    response = await self._client.chat.completions.create(
                        model=self._model,
                        messages=[
                            {"role": "system", "content": system},
                            {"role": "user", "content": prompt},
                        ],
                    )

                # Handle empty/malformed response - treat as retryable
                if not response.choices:
                    raise RuntimeError(
                        "API returned empty choices - server error, will retry"
                    )
                return response.choices[0].message.content or ""

            except Exception as e:
                error_str = str(e).lower()

                # Check if it's a rate limit error (by type or message)
                is_rate_limit = isinstance(e, RateLimitError) or (
                    "rate" in error_str and "limit" in error_str
                )

                # Check for server errors (empty choices, 5xx, etc.)
                is_server_error = (
                    "server error" in error_str
                    or "empty choices" in error_str
                    or any(code in error_str for code in ["500", "502", "503", "504"])
                )

                is_retryable = is_rate_limit or is_server_error

                if not is_retryable or attempt >= self._max_retries:
                    raise  # Not retryable or exhausted retries

                # Calculate wait time: server-recommended or exponential backoff
                error_msg = str(e)
                retry_after = self._parse_retry_after(error_msg)

                if retry_after is not None:
                    # Server told us exactly how long to wait
                    wait_time = retry_after + 1  # Add 1s buffer
                else:
                    # Exponential backoff: 3s, 6s, 12s, 24s, 48s...
                    wait_time = self._base_retry_delay * (2**attempt)

                # Add jitter to prevent thundering herd (75-125% of wait_time)
                jitter_factor = random.random() * 0.5 + 0.75
                wait_time = wait_time * jitter_factor

                error_type = "Rate limit" if is_rate_limit else "Server error"
                print(
                    f"{error_type} (attempt {attempt + 1}/{self._max_retries + 1}). "
                    f"Waiting {wait_time:.2f}s before retry..."
                )
                await asyncio.sleep(wait_time)

        raise RuntimeError("Unexpected end of retry loop")


class OpenRouterChatModel(ChatModel):
    """Chat backend powered by OpenRouter's unified API using the native OpenRouter SDK.

    Supports 300+ models from various providers including OpenAI, Anthropic, Google, etc.
    Handles both text and vision requests (multi-modal content).
    """

    def __init__(
        self,
        *,
        model: str = "openai/gpt-5-nano",
        api_key: str | None = None,
        site_url: str | None = None,
        app_name: str | None = None,
        system_prompt: str | None = None,
        max_retries: int = 10,
        base_retry_delay: float = 3.0,
        max_concurrent: int = 10,
    ) -> None:
        """Initialize OpenRouter chat model.

        Args:
            model: Model identifier (e.g., "openai/gpt-5-nano", "anthropic/claude-3.5-sonnet")
            api_key: OpenRouter API key (fallback to OPENROUTER_API_KEY env var)
            site_url: Your app/site URL for OpenRouter headers (optional)
            app_name: Your app name for OpenRouter headers (optional)
            system_prompt: Default system message for all completions
            max_retries: Maximum retry attempts on rate limit errors
            base_retry_delay: Base delay for exponential backoff (seconds)
            max_concurrent: Maximum concurrent requests (0 = unlimited)
        """
        # Check if OpenRouter SDK is available
        if not OPENROUTER_AVAILABLE:
            raise ImportError(
                "openrouter package is required for OpenRouterChatModel. "
                "Install with: pip install openrouter"
            )

        # Resolve API key
        key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not key:
            dotenv_vars = _load_dotenv()
            key = dotenv_vars.get("OPENROUTER_API_KEY")

        if not key:
            raise ValueError(
                "OPENROUTER_API_KEY is required. Set via env variable or pass as api_key parameter."
            )

        # Store API key for creating clients in context managers
        self._api_key = key
        self._model = model
        self._system_prompt = system_prompt
        self._site_url = site_url or "https://github.com/KohakuBlueleaf/KohakuRAG"
        self._app_name = app_name or "KohakuRAG"
        self._max_retries = max_retries
        self._base_retry_delay = base_retry_delay

        # Rate limiting
        self._semaphore: asyncio.Semaphore | None = (
            asyncio.Semaphore(max_concurrent) if max_concurrent > 0 else None
        )

    async def complete(
        self,
        prompt: str | list[dict],
        *,
        system_prompt: str | None = None,
    ) -> str:
        """Generate completion using OpenRouter API.

        Supports both text-only and vision (multimodal) requests.

        Args:
            prompt: Either a string (text-only) or list of content parts (vision)
                   Text: "What is the capital of France?"
                   Vision: [
                       {"type": "text", "text": "What's in this image?"},
                       {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}
                   ]
            system_prompt: Optional system message override

        Returns:
            Model's text response

        Raises:
            RuntimeError: If all retries are exhausted
        """
        # Build messages
        messages = []

        # Add system prompt if provided
        effective_system_prompt = system_prompt or self._system_prompt
        if effective_system_prompt:
            messages.append({"role": "system", "content": effective_system_prompt})

        # Add user message (supports both string and vision format)
        messages.append({"role": "user", "content": prompt})

        # Retry loop with exponential backoff
        for attempt in range(self._max_retries + 1):
            try:
                # Make API call with optional rate limiting
                if self._semaphore is not None:
                    async with self._semaphore:
                        response = await self._make_request(messages)
                        return response
                else:
                    response = await self._make_request(messages)
                    return response

            except Exception as e:
                error_str = str(e).lower()

                # Check for context length errors - these should NOT be retried internally
                # They need to propagate to the outer retry mechanism which reduces context
                # OpenRouter: "This endpoint's maximum context length is X tokens"
                # OpenAI: "This model's maximum context length is X tokens" + code: "context_length_exceeded"
                is_context_overflow = (
                    "maximum context length" in error_str
                    or "context_length_exceeded" in error_str
                )

                if is_context_overflow:
                    raise  # Propagate immediately for outer retry with reduced context

                # Check if it's a retryable error
                is_rate_limit = "rate" in error_str and "limit" in error_str
                is_server_error = any(
                    code in error_str
                    for code in [
                        "500",
                        "502",
                        "503",
                        "504",
                        "429",
                        "cloudflare",
                        "server",
                    ]
                )
                is_validation_error = "validation error" in error_str
                is_connection_error = any(
                    phrase in error_str
                    for phrase in [
                        "peer closed connection",
                        "incomplete chunked read",
                        "connection reset",
                        "remotprotocolerror",
                    ]
                )
                # Check for provider errors (OpenRouter ChatError, etc.)
                is_provider_error = (
                    "provider" in error_str
                    or "chaterror" in error_str
                    or "returned error" in error_str
                )

                # Validation errors often mean API returned error response instead of success
                if is_validation_error and "error" in error_str:
                    is_server_error = True  # Treat as retryable

                is_retryable = (
                    is_rate_limit
                    or is_server_error
                    or is_connection_error
                    or is_provider_error
                )

                if not is_retryable or attempt >= self._max_retries:
                    raise  # Not retryable or exhausted retries

                # Determine error type for logging
                if is_rate_limit:
                    error_type = "Rate limit"
                elif "502" in error_str or "cloudflare" in error_str:
                    error_type = "Cloudflare/Server error (502)"
                elif "500" in error_str or "503" in error_str:
                    error_type = "Server error"
                elif is_connection_error:
                    error_type = "Connection error (peer closed)"
                elif is_validation_error:
                    error_type = "API error (validation failure)"
                elif is_provider_error:
                    error_type = "Provider error"
                else:
                    error_type = "Transient error"

                # Calculate wait time with exponential backoff
                wait_time = self._base_retry_delay * (1.414**attempt)

                # Add jitter to prevent thundering herd (75-125% of wait_time)
                jitter_factor = random.random() * 0.5 + 0.75
                wait_time = wait_time * jitter_factor

                print(
                    f"OpenRouter {error_type} (attempt {attempt + 1}/{self._max_retries + 1}). "
                    f"Retrying in {wait_time:.1f}s..."
                )
                await asyncio.sleep(wait_time)

        raise RuntimeError("Unexpected end of retry loop")

    async def _make_request(self, messages: list[dict]) -> str:
        """Make the actual API request to OpenRouter.

        Args:
            messages: List of message dictionaries

        Returns:
            Response text from the model
        """
        # Use OpenRouter SDK with context manager (required for proper auth)
        async with OpenRouter(api_key=self._api_key) as client:
            response = await client.chat.send_async(
                messages=messages,
                model=self._model,
                stream=False,
            )

            # Extract response text
            if hasattr(response, "choices") and response.choices:
                return response.choices[0].message.content or ""
            else:
                # Fallback for unexpected response format
                return str(response)


class HuggingFaceLocalChatModel(ChatModel):
    """Chat backend powered by a local Hugging Face Transformers model.

    Intended for fully local inference (no network). Uses device_map="auto"
    to distribute the model across available GPUs/CPU.
    """

    def __init__(
        self,
        *,
        model: str = "Qwen/Qwen2.5-7B-Instruct",
        system_prompt: str | None = None,
        dtype: str = "bf16",
        max_concurrent: int = 2,
        max_new_tokens: int = 512,
        temperature: float = 0.2,
    ) -> None:
        """Initialize local HuggingFace chat model.

        Args:
            model: HuggingFace model identifier or local path
            system_prompt: Default system message for all completions
            dtype: Torch dtype - "bf16", "fp16", "4bit", or "auto"
            max_concurrent: Maximum concurrent inference requests
            max_new_tokens: Maximum tokens to generate per request
            temperature: Sampling temperature (0 = greedy)
        """
        if not HF_LOCAL_AVAILABLE:
            raise ImportError(
                "transformers and torch are required for HuggingFaceLocalChatModel. "
                "Install with: pip install torch transformers accelerate"
            )

        self._model_id = model
        self._system_prompt = system_prompt or "You are a helpful assistant."
        self._max_new_tokens = max_new_tokens
        self._temperature = temperature

        self._semaphore: asyncio.Semaphore | None = (
            asyncio.Semaphore(max_concurrent) if max_concurrent > 0 else None
        )

        # Load tokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(
            self._model_id, use_fast=True
        )

        # Resolve dtype and quantization
        load_kwargs = {"device_map": "auto"}

        if dtype == "4bit":
            try:
                from transformers import BitsAndBytesConfig
            except ImportError:
                raise ImportError(
                    "4-bit quantization requires bitsandbytes. "
                    "Install with: pip install bitsandbytes"
                )
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
            )
        else:
            if dtype == "bf16":
                load_kwargs["dtype"] = torch.bfloat16
            elif dtype == "fp16":
                load_kwargs["dtype"] = torch.float16

        # Load model with automatic device placement
        self._model = AutoModelForCausalLM.from_pretrained(
            self._model_id,
            **load_kwargs,
        )

    async def complete(self, prompt: str, *, system_prompt: str | None = None) -> str:
        """Generate a chat completion using local HF model.

        Args:
            prompt: User message
            system_prompt: Optional system message override

        Returns:
            Model's text response
        """
        system = system_prompt or self._system_prompt

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ]

        async def _run() -> str:
            return await asyncio.to_thread(self._generate_sync, messages)

        if self._semaphore is not None:
            async with self._semaphore:
                return await _run()
        return await _run()

    def _generate_sync(self, messages: list[dict]) -> str:
        """Run synchronous generation (offloaded to thread)."""
        # Use chat template if the tokenizer supports it
        if hasattr(self._tokenizer, "apply_chat_template"):
            text = self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            lines = []
            for m in messages:
                lines.append(f"{m['role'].upper()}: {m['content']}")
            lines.append("ASSISTANT:")
            text = "\n".join(lines)

        inputs = self._tokenizer(text, return_tensors="pt")
        input_len = inputs["input_ids"].shape[-1]
        # For device_map="auto" models, get device from first parameter
        target_device = next(self._model.parameters()).device
        inputs = {k: v.to(target_device) for k, v in inputs.items()}

        do_sample = self._temperature > 0
        with torch.no_grad():
            out = self._model.generate(
                **inputs,
                max_new_tokens=self._max_new_tokens,
                do_sample=do_sample,
                temperature=self._temperature if do_sample else None,
            )

        # Only decode the newly generated tokens (skip the input)
        new_tokens = out[0][input_len:]
        return self._tokenizer.decode(new_tokens, skip_special_tokens=True).strip()



class BedrockChatModel(ChatModel):
    """Chat backend powered by AWS Bedrock's Converse API.

    Supports Anthropic Claude, Meta Llama, Amazon Nova, Mistral, and other
    Bedrock-hosted foundation models through a unified interface.

    Authentication:
        - AWS SSO profile (recommended): set ``profile_name``
        - Environment variables: AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY
        - IAM instance role (EC2/Lambda): automatic
    """

    def __init__(
        self,
        *,
        model_id: str = "us.anthropic.claude-3-haiku-20240307-v1:0",
        profile_name: str | None = None,
        region_name: str = "us-east-2",
        system_prompt: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.2,
        max_retries: int = 5,
        base_retry_delay: float = 2.0,
        max_concurrent: int = 5,
    ) -> None:
        """Initialize AWS Bedrock chat model.

        Args:
            model_id: Bedrock model identifier (e.g.
                ``us.anthropic.claude-3-haiku-20240307-v1:0``)
            profile_name: AWS SSO/CLI profile name.  Falls back to env vars
                or instance role when ``None``.
            region_name: AWS region with Bedrock access (default ``us-east-2``).
            system_prompt: Default system message for all completions.
            max_tokens: Maximum tokens to generate per request.
            temperature: Sampling temperature (0 = greedy).
            max_retries: Maximum retry attempts on throttle / server errors.
            base_retry_delay: Base delay in seconds for exponential backoff.
            max_concurrent: Maximum concurrent API requests (0 = unlimited).
        """
        if not BEDROCK_AVAILABLE:
            raise ImportError(
                "boto3 is required for BedrockChatModel. "
                "Install with: pip install boto3"
            )

        # Resolve profile from env if not provided directly
        if profile_name is None:
            profile_name = os.environ.get("AWS_PROFILE")

        # Resolve region from env if not provided
        env_region = os.environ.get("AWS_REGION") or os.environ.get(
            "AWS_DEFAULT_REGION"
        )
        if env_region:
            region_name = env_region

        session_kwargs: dict = {}
        if profile_name:
            session_kwargs["profile_name"] = profile_name

        boto_config = BotoConfig(
            region_name=region_name,
            retries={"max_attempts": 0},  # We handle retries ourselves
        )

        session = boto3.Session(**session_kwargs)
        self._client = session.client(
            "bedrock-runtime",
            config=boto_config,
        )

        self._model_id = model_id
        self._system_prompt = system_prompt or "You are a helpful assistant."
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._max_retries = max_retries
        self._base_retry_delay = base_retry_delay

        self._semaphore: asyncio.Semaphore | None = (
            asyncio.Semaphore(max_concurrent) if max_concurrent > 0 else None
        )

    async def complete(self, prompt: str, *, system_prompt: str | None = None) -> str:
        """Generate a chat completion via AWS Bedrock Converse API.

        Uses exponential backoff with jitter for throttling and transient
        server errors, matching the retry patterns of the other providers.

        Returns:
            Model's text response.
        """
        system = system_prompt or self._system_prompt

        for attempt in range(self._max_retries + 1):
            try:
                if self._semaphore is not None:
                    async with self._semaphore:
                        return await asyncio.to_thread(
                            self._converse_sync, system, prompt
                        )
                else:
                    return await asyncio.to_thread(
                        self._converse_sync, system, prompt
                    )

            except Exception as e:
                error_str = str(e).lower()

                is_throttle = (
                    "throttling" in error_str
                    or "too many requests" in error_str
                    or "rate exceeded" in error_str
                )
                is_server_error = any(
                    code in error_str
                    for code in ["500", "502", "503", "504", "serviceunavailable"]
                )

                is_retryable = is_throttle or is_server_error

                if not is_retryable or attempt >= self._max_retries:
                    raise

                wait_time = self._base_retry_delay * (2**attempt)
                jitter_factor = random.random() * 0.5 + 0.75
                wait_time *= jitter_factor

                error_type = "Throttle" if is_throttle else "Server error"
                print(
                    f"Bedrock {error_type} (attempt {attempt + 1}/{self._max_retries + 1}). "
                    f"Retrying in {wait_time:.1f}s..."
                )
                await asyncio.sleep(wait_time)

        raise RuntimeError("Unexpected end of retry loop")

    def _converse_sync(self, system: str, prompt: str) -> str:
        """Synchronous call to the Bedrock Converse API."""
        inference_config: dict = {"maxTokens": self._max_tokens}
        if self._temperature > 0:
            inference_config["temperature"] = self._temperature

        response = self._client.converse(
            modelId=self._model_id,
            system=[{"text": system}],
            messages=[
                {
                    "role": "user",
                    "content": [{"text": prompt}],
                }
            ],
            inferenceConfig=inference_config,
        )

        # Extract text from the response
        output = response.get("output", {})
        message = output.get("message", {})
        content_blocks = message.get("content", [])

        text_parts = []
        for block in content_blocks:
            if "text" in block:
                text_parts.append(block["text"])

        return "\n".join(text_parts).strip()

    @property
    def usage(self) -> dict:
        """Return the last request's token usage (if tracked by the caller)."""
        return {}
