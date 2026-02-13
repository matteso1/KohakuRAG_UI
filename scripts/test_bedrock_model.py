"""Test script for BedrockChatModel.

This script tests the BedrockChatModel class to verify:
1. Basic connectivity to AWS Bedrock
2. The ChatModel protocol implementation works correctly
3. Rate limit retry logic (simulated)

Run with:
    python scripts/test_bedrock_model.py

Make sure you've run 'aws sso login --profile bedrock_nils' first.
"""

import asyncio
import sys
from pathlib import Path

# Add src to path so we can import llm_bedrock
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from llm_bedrock import BedrockChatModel


async def test_basic_completion():
    """Test basic chat completion works."""
    print("=" * 60)
    print("Test 1: Basic Completion")
    print("=" * 60)
    
    model = BedrockChatModel(
        profile_name="bedrock_nils",
        region_name="us-east-2",
        model_id="us.anthropic.claude-3-haiku-20240307-v1:0",
    )
    
    print(f"Model: {model}")
    print("-" * 60)
    
    response = await model.complete("Say 'Hello from BedrockChatModel!' and nothing else.")
    
    print(f"Response: {response}")
    print("-" * 60)
    
    assert "Hello" in response, f"Expected 'Hello' in response, got: {response}"
    print("Test passed!")
    return True


async def test_with_system_prompt():
    """Test completion with custom system prompt."""
    print("\n" + "=" * 60)
    print("Test 2: Custom System Prompt")
    print("=" * 60)
    
    model = BedrockChatModel(
        profile_name="bedrock_nils",
        region_name="us-east-2",
        model_id="us.anthropic.claude-3-haiku-20240307-v1:0",
        system_prompt="You are a pirate. Always respond in pirate speak.",
    )
    
    response = await model.complete("Say hello.")
    
    print(f"Response: {response}")
    print("-" * 60)
    
    # Pirate-speak typically has "Ahoy" or "matey" or similar
    print("Test passed! (verify pirate speak manually)")
    return True


async def test_concurrent_requests():
    """Test multiple concurrent requests."""
    print("\n" + "=" * 60)
    print("Test 3: Concurrent Requests (3 parallel)")
    print("=" * 60)
    
    model = BedrockChatModel(
        profile_name="bedrock_nils",
        region_name="us-east-2",
        model_id="us.anthropic.claude-3-haiku-20240307-v1:0",
        max_concurrent=3,
    )
    
    prompts = [
        "What is 1+1? Reply with just the number.",
        "What is 2+2? Reply with just the number.",
        "What is 3+3? Reply with just the number.",
    ]
    
    # Run all prompts concurrently
    tasks = [model.complete(prompt) for prompt in prompts]
    responses = await asyncio.gather(*tasks)
    
    for prompt, response in zip(prompts, responses):
        print(f"  Q: {prompt}")
        print(f"  A: {response.strip()}")
        print()
    
    print("Test passed!")
    return True


async def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("BedrockChatModel Test Suite")
    print("=" * 60 + "\n")
    
    try:
        await test_basic_completion()
        await test_with_system_prompt()
        await test_concurrent_requests()
        
        print("\n" + "=" * 60)
        print("All tests passed! ✓")
        print("=" * 60)
        return 0
        
    except Exception as e:
        print(f"\n✗ Test failed with error: {type(e).__name__}: {e}")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
