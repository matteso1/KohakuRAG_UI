#!/usr/bin/env python3
"""
AWS Bedrock Cost Tracking Setup

Creates Application Inference Profiles with UW-Madison approved billing tags.
This enables tracking costs by experiment/model in AWS Cost Explorer.

IMPORTANT: UW-Madison manages billing tags at the payer/management account level.
Only pre-approved tag keys will appear in Cost Explorer / Cost Allocation Reports.
See: https://kb.wisc.edu/page.php?id=72454

Tag mapping (our concept -> approved UW-Madison tag key):
    project     -> Project       (e.g., "wattbot")
    project name-> ProjectName   (e.g., "KohakuRAG-WattBot")
    experiment  -> Purpose       (e.g., "wattbot-eval")
    model       -> CA001         (custom allocation code; no "Model" tag exists yet)
    owner       -> Owner         (e.g., "nils-matteson")
    environment -> Environment   (e.g., "evaluation")
    application -> Application   (e.g., "bedrock-inference")

Usage:
    # List existing inference profiles
    python scripts/setup_bedrock_cost_tracking.py --list

    # Create inference profiles for all models
    python scripts/setup_bedrock_cost_tracking.py --create

    # Create profile for specific model
    python scripts/setup_bedrock_cost_tracking.py --create --model claude-3-haiku

    # Create with a specific experiment name
    python scripts/setup_bedrock_cost_tracking.py --create --experiment wattbot-eval-v2

    # Delete old profiles (e.g., before recreating with new tags)
    python scripts/setup_bedrock_cost_tracking.py --delete <profile-arn>

After creation, tagged costs should appear in Cost Explorer within ~24 hours
(no manual tag activation needed -- these are already activated at the payer level).

References:
- https://docs.aws.amazon.com/bedrock/latest/userguide/inference-profiles-create.html
- https://kb.wisc.edu/page.php?id=72454 (UW-Madison approved billing tags)
"""

import argparse
import json
import sys
from pathlib import Path

try:
    import boto3
    from botocore.exceptions import ClientError
except ImportError:
    print("boto3 required: pip install boto3")
    sys.exit(1)


# Model configurations
# Maps friendly names to cross-region inference profile model IDs.
# The script auto-discovers the account ID and builds the full system-defined
# inference profile ARN at runtime (needed for create_inference_profile's copyFrom).
MODELS = {
    "claude-3-haiku": {
        "profile_model_id": "us.anthropic.claude-3-haiku-20240307-v1:0",
        "description": "WattBot Claude3 Haiku",
    },
    "claude-3-5-haiku": {
        "profile_model_id": "us.anthropic.claude-3-5-haiku-20241022-v1:0",
        "description": "WattBot Claude3.5 Haiku",
    },
    "claude-3-5-sonnet": {
        "profile_model_id": "us.anthropic.claude-3-5-sonnet-20241022-v2:0",
        "description": "WattBot Claude3.5 Sonnet",
    },
    "claude-3-7-sonnet": {
        "profile_model_id": "us.anthropic.claude-3-7-sonnet-20250219-v1:0",
        "description": "WattBot Claude3.7 Sonnet",
    },
    "nova-pro": {
        "profile_model_id": "us.amazon.nova-pro-v1:0",
        "description": "WattBot Amazon NovaPro",
    },
    "llama-3-70b": {
        "profile_model_id": "us.meta.llama3-1-70b-instruct-v1:0",
        "description": "WattBot Llama3 70B",
    },
    "llama-4-maverick": {
        "profile_model_id": "us.meta.llama4-maverick-17b-instruct-v1:0",
        "description": "WattBot Llama4 Maverick",
    },
    "deepseek-r1": {
        "profile_model_id": "us.deepseek.r1-v1:0",
        "description": "WattBot DeepSeek R1",
    },
}

# =============================================================================
# UW-Madison Approved Billing Tags
# =============================================================================
# These tag keys are pre-activated at the UW-Madison payer account level.
# They MUST be spelled exactly as shown (case-sensitive) or they won't
# appear in Cost Explorer / Cost Allocation Reports.
# Full list: https://kb.wisc.edu/page.php?id=72454
#
# We use CA001 for "model" since there is no dedicated Model tag yet.
# If UW-Madison adds a "Model" tag in the future, we can migrate to that.

DEFAULT_TAGS = [
    {"key": "Project", "value": "wattbot"},
    {"key": "ProjectName", "value": "KohakuRAG-WattBot"},
    {"key": "Owner", "value": "nils-matteson"},
    {"key": "Environment", "value": "evaluation"},
    {"key": "Application", "value": "bedrock-inference"},
]


def get_bedrock_client(profile_name: str, region: str):
    """Create Bedrock client and resolve account ID."""
    session = boto3.Session(profile_name=profile_name, region_name=region)
    bedrock = session.client("bedrock")

    # Get account ID for building system-defined inference profile ARNs
    sts = session.client("sts")
    account_id = sts.get_caller_identity()["Account"]

    return bedrock, account_id, region


def list_inference_profiles(client, profile_type: str | None = None) -> list:
    """List inference profiles. If profile_type is None, returns both types."""
    try:
        profiles = []
        for ptype in ([profile_type] if profile_type else ["SYSTEM_DEFINED", "APPLICATION"]):
            response = client.list_inference_profiles(typeEquals=ptype)
            profiles.extend(response.get("inferenceProfileSummaries", []))
        return profiles
    except ClientError as e:
        print(f"Error listing profiles: {e}")
        return []


def create_inference_profile(
    client,
    profile_name: str,
    model_arn: str,
    description: str,
    tags: list,
) -> dict | None:
    """Create an application inference profile with tags."""
    try:
        params = {
            "inferenceProfileName": profile_name,
            "modelSource": {"copyFrom": model_arn},
            "description": description,
            "tags": tags,
        }

        response = client.create_inference_profile(**params)

        return {
            "arn": response["inferenceProfileArn"],
            "status": response.get("status"),
            "name": profile_name,
        }

    except ClientError as e:
        error_code = e.response["Error"]["Code"]
        error_msg = e.response["Error"]["Message"]

        if error_code == "ConflictException":
            print(f"  Profile '{profile_name}' already exists")
            return None
        elif error_code == "ValidationException":
            print(f"  Validation error: {error_msg}")
            return None
        else:
            print(f"  Error ({error_code}): {error_msg}")
            return None


def delete_inference_profile(client, profile_id: str) -> bool:
    """Delete an inference profile."""
    try:
        client.delete_inference_profile(inferenceProfileIdentifier=profile_id)
        return True
    except ClientError as e:
        print(f"Error deleting profile: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Setup AWS Bedrock cost tracking with inference profiles"
    )
    parser.add_argument(
        "--profile",
        default="bedrock_nils",
        help="AWS profile name",
    )
    parser.add_argument(
        "--region",
        default="us-east-2",
        help="AWS region",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List existing inference profiles",
    )
    parser.add_argument(
        "--create",
        action="store_true",
        help="Create inference profiles for models",
    )
    parser.add_argument(
        "--model",
        help="Specific model to create profile for (default: all)",
    )
    parser.add_argument(
        "--experiment",
        default="wattbot-eval",
        help="Experiment name for tagging",
    )
    parser.add_argument(
        "--delete",
        help="Delete a specific inference profile by ID/ARN",
    )
    parser.add_argument(
        "--output",
        help="Output file for profile ARN mapping (JSON)",
    )

    args = parser.parse_args()

    print(f"Using AWS profile: {args.profile}")
    print(f"Region: {args.region}")
    print()

    client, account_id, region = get_bedrock_client(args.profile, args.region)
    print(f"Account ID: {account_id}")
    print()

    if args.list:
        # Show application profiles (our tagged ones) first, then system-defined
        app_profiles = list_inference_profiles(client, "APPLICATION")
        sys_profiles = list_inference_profiles(client, "SYSTEM_DEFINED")

        print(f"=== Application Inference Profiles ({len(app_profiles)}) ===")
        if app_profiles:
            for p in app_profiles:
                print(f"  Name: {p.get('inferenceProfileName')}")
                print(f"  ARN:  {p.get('inferenceProfileArn')}")
                print(f"  Status: {p.get('status')}")
                print()
        else:
            print("  None (run --create to set up tagged profiles)")
            print()

        print(f"=== System-Defined Inference Profiles ({len(sys_profiles)}) ===")
        if sys_profiles:
            for p in sys_profiles:
                print(f"  {p.get('inferenceProfileName')}")
        else:
            print("  None found")
        return

    if args.delete:
        print(f"Deleting profile: {args.delete}")
        if delete_inference_profile(client, args.delete):
            print("  Deleted successfully")
        return

    if args.create:
        print("=== Creating Inference Profiles ===")
        print(f"Experiment tag: {args.experiment}")
        print()

        # Determine which models to create profiles for
        if args.model:
            if args.model not in MODELS:
                print(f"Unknown model: {args.model}")
                print(f"Available: {', '.join(MODELS.keys())}")
                return
            models_to_create = {args.model: MODELS[args.model]}
        else:
            models_to_create = MODELS

        created_profiles = {}

        for model_name, model_info in models_to_create.items():
            profile_name = f"wattbot-{model_name}"
            print(f"Creating profile: {profile_name}")

            # Build the system-defined inference profile ARN.
            # create_inference_profile requires a system-defined profile ARN
            # (not a foundation model ARN) for cross-region models.
            source_arn = (
                f"arn:aws:bedrock:{region}:{account_id}"
                f":inference-profile/{model_info['profile_model_id']}"
            )
            print(f"  Source: {source_arn}")

            # Build tags using UW-Madison approved keys
            tags = DEFAULT_TAGS.copy()
            tags.append({"key": "Purpose", "value": args.experiment})  # experiment name
            tags.append({"key": "CA001", "value": model_name})         # model (custom allocation code)

            result = create_inference_profile(
                client=client,
                profile_name=profile_name,
                model_arn=source_arn,
                description=model_info["description"],
                tags=tags,
            )

            if result:
                print(f"  Created: {result['arn']}")
                created_profiles[model_name] = {
                    "profile_arn": result["arn"],
                    "original_model_id": model_info["profile_model_id"],
                }
            print()

        # Output mapping file
        if args.output and created_profiles:
            output_path = Path(args.output)
            with open(output_path, "w") as f:
                json.dump(created_profiles, f, indent=2)
            print(f"Saved profile mapping to: {output_path}")

        print()
        print("=== Next Steps ===")
        print("Tags used are from the UW-Madison approved billing tag list.")
        print("They are already activated at the payer account level (no action needed).")
        print("  - Project     = wattbot")
        print("  - ProjectName = KohakuRAG-WattBot")
        print("  - Purpose     = experiment name")
        print("  - CA001       = model name (custom allocation code)")
        print()
        print("Costs should appear in Cost Explorer within ~24 hours.")
        print()
        print("To use profiles in code, replace model_id with the profile ARN:")
        print("  model_id = 'arn:aws:bedrock:us-east-2:ACCOUNT:inference-profile/...'")
        print()
        print("Reference: https://kb.wisc.edu/page.php?id=72454")


if __name__ == "__main__":
    main()
