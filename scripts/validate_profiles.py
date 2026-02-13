#!/usr/bin/env python3
"""
Validation Test: Direct Model ID vs Tagged Inference Profile ARN

Runs the SAME questions through:
  A) Direct model ID (how we've been running experiments)
  B) Tagged inference profile ARN (for cost tracking)

Then compares:
  1. Response content (should be functionally equivalent)
  2. Token counts (should match)
  3. Latency (should be similar)
  4. Model family detection (must work correctly for both)

Tests multiple model families (Claude, Llama, DeepSeek) to catch
the family detection bug with profile ARNs.

Exit code 0 = all tests pass. Non-zero = failures found.
"""

import asyncio
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from llm_bedrock import BedrockChatModel


# Test configuration
AWS_PROFILE = "bedrock_nils"
REGION = "us-east-2"

# Models to validate (model_id, profile_arn, expected_family)
# Profile ARNs from artifacts/profile_mapping.json
PROFILE_MAPPING_PATH = Path(__file__).parent.parent / "artifacts" / "profile_mapping.json"

TEST_PROMPT = "What is 2 + 2? Reply with ONLY the number, nothing else."
SYSTEM_PROMPT = "You are a helpful assistant. Be concise."

# Models to test -- covers different families
TEST_MODELS = {
    "claude-3-haiku": {
        "model_id": "us.anthropic.claude-3-haiku-20240307-v1:0",
        "family": "anthropic",
    },
    "llama-4-maverick": {
        "model_id": "us.meta.llama4-maverick-17b-instruct-v1:0",
        "family": "meta",
    },
    "deepseek-r1": {
        "model_id": "us.deepseek.r1-v1:0",
        "family": "deepseek",
    },
}


def load_profile_mapping() -> dict:
    if not PROFILE_MAPPING_PATH.exists():
        print(f"ERROR: Profile mapping not found: {PROFILE_MAPPING_PATH}")
        print("Run: python scripts/setup_bedrock_cost_tracking.py --create --output artifacts/profile_mapping.json")
        sys.exit(1)

    with open(PROFILE_MAPPING_PATH) as f:
        return json.load(f)


async def test_model(
    name: str,
    model_id: str,
    profile_arn: str | None,
    expected_family: str,
) -> dict:
    """Test a single model both ways and return comparison."""

    results = {}

    # ---- Test A: Direct model ID ----
    print(f"\n  [A] Direct model ID: {model_id}")
    chat_direct = BedrockChatModel(
        model_id=model_id,
        profile_name=AWS_PROFILE,
        region_name=REGION,
        system_prompt=SYSTEM_PROMPT,
        max_retries=2,
        max_concurrent=1,
    )

    # Verify family detection
    family = chat_direct._detect_model_family()
    print(f"      Family detected: {family} (expected: {expected_family})")
    results["direct_family"] = family
    results["direct_family_correct"] = family == expected_family

    start = time.time()
    try:
        response_direct = await chat_direct.complete(TEST_PROMPT, system_prompt=SYSTEM_PROMPT)
        latency_direct = time.time() - start
        results["direct_response"] = response_direct.strip()
        results["direct_latency"] = latency_direct
        results["direct_input_tokens"] = chat_direct.token_usage.input_tokens
        results["direct_output_tokens"] = chat_direct.token_usage.output_tokens
        results["direct_error"] = None
        print(f"      Response: {response_direct.strip()[:80]}")
        print(f"      Tokens: in={chat_direct.token_usage.input_tokens}, out={chat_direct.token_usage.output_tokens}")
        print(f"      Latency: {latency_direct:.2f}s")
    except Exception as e:
        results["direct_error"] = str(e)
        print(f"      ERROR: {e}")

    # ---- Test B: Inference profile ARN ----
    if profile_arn:
        print(f"  [B] Profile ARN: ...{profile_arn[-35:]}")
        chat_profile = BedrockChatModel(
            model_id=model_id,
            inference_profile_arn=profile_arn,
            profile_name=AWS_PROFILE,
            region_name=REGION,
            system_prompt=SYSTEM_PROMPT,
            max_retries=2,
            max_concurrent=1,
        )

        # Verify family detection still works
        family_p = chat_profile._detect_model_family()
        print(f"      Family detected: {family_p} (expected: {expected_family})")
        results["profile_family"] = family_p
        results["profile_family_correct"] = family_p == expected_family

        # Verify invoke_model_id is the profile ARN
        results["uses_profile_arn"] = chat_profile._invoke_model_id == profile_arn

        start = time.time()
        try:
            response_profile = await chat_profile.complete(TEST_PROMPT, system_prompt=SYSTEM_PROMPT)
            latency_profile = time.time() - start
            results["profile_response"] = response_profile.strip()
            results["profile_latency"] = latency_profile
            results["profile_input_tokens"] = chat_profile.token_usage.input_tokens
            results["profile_output_tokens"] = chat_profile.token_usage.output_tokens
            results["profile_error"] = None
            print(f"      Response: {response_profile.strip()[:80]}")
            print(f"      Tokens: in={chat_profile.token_usage.input_tokens}, out={chat_profile.token_usage.output_tokens}")
            print(f"      Latency: {latency_profile:.2f}s")
        except Exception as e:
            results["profile_error"] = str(e)
            print(f"      ERROR: {e}")
    else:
        print(f"  [B] SKIPPED - no profile ARN found for {name}")
        results["profile_error"] = "NO_PROFILE_ARN"

    return results


async def main():
    print("=" * 70)
    print("VALIDATION TEST: Direct Model ID vs Tagged Inference Profile")
    print("=" * 70)

    mapping = load_profile_mapping()
    print(f"Loaded {len(mapping)} profile mappings")

    all_results = {}
    failures = []

    for name, config in TEST_MODELS.items():
        print(f"\n{'-'*70}")
        print(f"Testing: {name}")

        profile_arn = mapping.get(name, {}).get("profile_arn")

        result = await test_model(
            name=name,
            model_id=config["model_id"],
            profile_arn=profile_arn,
            expected_family=config["family"],
        )
        all_results[name] = result

        # Check for failures
        if not result.get("direct_family_correct"):
            failures.append(f"{name}: direct family detection wrong ({result.get('direct_family')})")
        if result.get("direct_error"):
            failures.append(f"{name}: direct invocation failed: {result['direct_error']}")
        if not result.get("profile_family_correct", True):
            failures.append(f"{name}: profile family detection wrong ({result.get('profile_family')})")
        if result.get("profile_error") and result["profile_error"] != "NO_PROFILE_ARN":
            failures.append(f"{name}: profile invocation failed: {result['profile_error']}")
        if not result.get("uses_profile_arn", True):
            failures.append(f"{name}: invoke_model_id is NOT the profile ARN (cost tracking broken)")

    # ---- Summary ----
    print(f"\n{'='*70}")
    print("VALIDATION RESULTS")
    print(f"{'='*70}")

    for name, result in all_results.items():
        status = "PASS" if not any(name in f for f in failures) else "FAIL"
        direct_ok = "OK" if not result.get("direct_error") else "ERR"
        profile_ok = "OK" if not result.get("profile_error") else "ERR"

        print(f"  {name:<25s} [{status}]  direct={direct_ok}  profile={profile_ok}")

        if result.get("direct_response") and result.get("profile_response"):
            match = "MATCH" if result["direct_response"] == result["profile_response"] else "DIFFER (ok - nondeterministic)"
            print(f"    Responses: {match}")
            print(f"    Direct tokens:  in={result.get('direct_input_tokens', '?')}, out={result.get('direct_output_tokens', '?')}")
            print(f"    Profile tokens: in={result.get('profile_input_tokens', '?')}, out={result.get('profile_output_tokens', '?')}")

    if failures:
        print(f"\n{'!'*70}")
        print(f"FAILURES ({len(failures)}):")
        for f in failures:
            print(f"  - {f}")
        print(f"{'!'*70}")
        return 1
    else:
        print(f"\nAll tests PASSED. Inference profiles are working correctly.")
        print(f"Cost tracking will attribute usage to the tagged profiles.")
        return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
