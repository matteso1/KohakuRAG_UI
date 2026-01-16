"""
Simple test script to verify Bedrock connectivity.

Run with:
    python scripts/test_bedrock.py

Make sure you've run 'aws sso login --profile bedrock_nils' first.
"""

import json
import boto3

# Configuration
PROFILE_NAME = "bedrock_nils"
REGION = "us-east-2"
MODEL_ID = "us.anthropic.claude-3-haiku-20240307-v1:0"  # Cross-region inference profile

def test_bedrock():
    # Create a session with the SSO profile
    session = boto3.Session(profile_name=PROFILE_NAME, region_name=REGION)
    
    # Create Bedrock runtime client
    client = session.client("bedrock-runtime")
    
    # Prepare the request (Anthropic Claude format)
    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 256,
        "messages": [
            {
                "role": "user",
                "content": "Say 'Hello from Bedrock!' and nothing else."
            }
        ]
    }
    
    print(f"Testing Bedrock with model: {MODEL_ID}")
    print(f"Region: {REGION}")
    print(f"Profile: {PROFILE_NAME}")
    print("-" * 40)
    
    try:
        # Call Bedrock
        response = client.invoke_model(
            modelId=MODEL_ID,
            body=json.dumps(body),
            contentType="application/json",
            accept="application/json"
        )
        
        # Parse response
        result = json.loads(response["body"].read())
        
        # Extract the text
        output_text = result["content"][0]["text"]
        
        print(f"Response: {output_text}")
        print("-" * 40)
        print("Bedrock connection successful!")
        
        # Print usage info
        if "usage" in result:
            print(f"Input tokens: {result['usage'].get('input_tokens', 'N/A')}")
            print(f"Output tokens: {result['usage'].get('output_tokens', 'N/A')}")
        
        return True
        
    except Exception as e:
        print(f"Error: {type(e).__name__}: {e}")
        return False

if __name__ == "__main__":
    test_bedrock()
