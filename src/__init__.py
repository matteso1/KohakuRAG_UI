"""Source package for KohakuRAG UI extensions.

This package contains extensions and integrations for the KohakuRAG pipeline,
including the BedrockChatModel for AWS Bedrock integration.
"""

from .llm_bedrock import BedrockChatModel

__all__ = ["BedrockChatModel"]
