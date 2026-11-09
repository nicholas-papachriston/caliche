"""
Integration adapters for popular AI frameworks.

This module provides adapters that allow you to use tools from other frameworks
(e.g., LangChain) with Caliche.
"""

from __future__ import annotations

from .langchain import (
    LangChainChainAdapter,
    LangChainMemoryAdapter,
    LangChainRetrieverAdapter,
    LangChainTool,
)
from .langchain_embeddings import LangChainEmbeddingProvider
from .langchain_llm import LangChainLLMProvider

__all__ = [
    "LangChainTool",
    "LangChainMemoryAdapter",
    "LangChainRetrieverAdapter",
    "LangChainChainAdapter",
    "LangChainLLMProvider",
    "LangChainEmbeddingProvider",
]
