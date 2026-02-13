"""Public package surface for KohakuRAG."""

from .datastore import (
    HierarchicalNodeStore,
    InMemoryNodeStore,
    KVaultNodeStore,
    matches_to_snippets,
)
from .embeddings import (
    EmbeddingModel,
    JinaEmbeddingModel,
    LocalHFEmbeddingModel,
    average_embeddings,
)
from .indexer import DocumentIndexer
from .llm import BedrockChatModel, HuggingFaceLocalChatModel, OpenAIChatModel
from .parsers import (
    dict_to_payload,
    markdown_to_payload,
    payload_to_dict,
    text_to_payload,
)
from .pipeline import (
    LLMQueryPlanner,
    MockChatModel,
    PromptTemplate,
    QueryPlanner,
    RAGPipeline,
    SimpleQueryPlanner,
    StructuredAnswer,
    StructuredAnswerResult,
    format_snippets,
)
from .types import (
    ContextSnippet,
    DocumentPayload,
    NodeKind,
    RetrievalMatch,
    StoredNode,
    TreeNode,
)

__all__ = [
    "average_embeddings",
    "BedrockChatModel",
    "ContextSnippet",
    "DocumentIndexer",
    "DocumentPayload",
    "EmbeddingModel",
    "HierarchicalNodeStore",
    "HuggingFaceLocalChatModel",
    "InMemoryNodeStore",
    "KVaultNodeStore",
    "JinaEmbeddingModel",
    "LLMQueryPlanner",
    "LocalHFEmbeddingModel",
    "MockChatModel",
    "NodeKind",
    "OpenAIChatModel",
    "PromptTemplate",
    "QueryPlanner",
    "RAGPipeline",
    "RetrievalMatch",
    "SimpleQueryPlanner",
    "StoredNode",
    "StructuredAnswer",
    "StructuredAnswerResult",
    "TreeNode",
    "dict_to_payload",
    "format_snippets",
    "markdown_to_payload",
    "payload_to_dict",
    "text_to_payload",
    "matches_to_snippets",
]
