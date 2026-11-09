"""
RAG Service for document indexing and retrieval.

Provides programmatic access to RAG operations including document indexing,
semantic search, hybrid search, reranking, and graph-enhanced search.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from caliche import Agent

_RAG_FEATURE_ERROR = (
    "RAG Service requires 'tools-rag' feature. "
    "Rebuild the native extension with features including tools-rag."
)

if TYPE_CHECKING:
    from caliche_runtime import _caliche_py as _runtime
else:
    try:
        from caliche._runtime import load_runtime

        _runtime = load_runtime()
    except ImportError as exc:
        raise ImportError(_RAG_FEATURE_ERROR) from exc

if not hasattr(_runtime, "PyRagService") or not hasattr(_runtime, "PyRagTool"):
    raise ImportError(_RAG_FEATURE_ERROR)

_PyRagService = _runtime.PyRagService
_PyRagTool = _runtime.PyRagTool


@dataclass
class RagIndexResult:
    """Result of indexing a document."""

    chunk_count: int
    usage: Optional[Dict[str, int]] = None


@dataclass
class RagQueryResult:
    """Result of a basic RAG query."""

    results: List[str]
    top_similarity: Optional[float] = None
    usage: Optional[Dict[str, int]] = None


@dataclass
class EnhancedQueryResult:
    """Result of an enhanced query."""

    results: List[Dict[str, Any]]
    top_similarity: Optional[float] = None
    usage: Optional[Dict[str, int]] = None
    warnings: List[str] = field(default_factory=list)
    context_stats: Optional[Dict[str, Any]] = None


@dataclass
class GraphSearchResult:
    """Result of a graph search."""

    chunks: List[Dict[str, Any]]
    entities: List[Dict[str, Any]]


@dataclass
class RagStats:
    """RAG service statistics."""

    embedding_count: int
    text_count: int


class RagService:
    """
    RAG Service for document indexing and retrieval.

    Provides programmatic access to RAG operations including document indexing,
    semantic search, hybrid search, reranking, and graph-enhanced search.

    Example:
        >>> from caliche import AgentBuilder, ProviderType, RagService
        >>> agent = AgentBuilder() \\
        ...     .provider(ProviderType.OpenAI) \\
        ...     .model("gpt-4o") \\
        ...     .api_key("sk-...") \\
        ...     .build()
        >>> rag = RagService(agent)
        >>> await rag.index_document("Python is a programming language...", "python_intro")
        >>> results = await rag.query("What is Python?", 5, 0.3)
        >>> print(results.results)
    """

    def __init__(self, agent: Agent):
        """
        Create a new RAG service with default in-memory backend.

        Args:
            agent: Agent with embedding provider configured
        """
        # Pass the agent directly - PyRagService expects a PyAgent reference
        self._service = _PyRagService(agent)

    @classmethod
    def _from_py_service(cls, py_service: _PyRagService) -> RagService:
        """
        Create a RagService from a PyRagService instance.

        This is used internally to share backends between RagTool and RagService.

        Args:
            py_service: The PyRagService instance

        Returns:
            A RagService instance wrapping the PyRagService
        """
        instance = cls.__new__(cls)
        instance._service = py_service
        return instance

    def index_document(self, content: str, doc_id: str) -> RagIndexResult:
        """
        Index a document by splitting it into chunks and generating embeddings.

        Args:
            content: Document content to index
            doc_id: Unique identifier for the document

        Returns:
            RagIndexResult with chunk count and usage
        """
        result = self._service.index_document(content, doc_id)
        return RagIndexResult(
            chunk_count=result["chunk_count"],
            usage=result.get("usage"),
        )

    def query(
        self, query: str, top_k: int, min_similarity: Optional[float] = None
    ) -> RagQueryResult:
        """
        Query for relevant chunks using semantic search.

        Args:
            query: Query string
            top_k: Number of results to return
            min_similarity: Minimum similarity threshold (default: 0.3)

        Returns:
            RagQueryResult with relevant chunks
        """
        result = self._service.query(query, top_k, min_similarity)
        return RagQueryResult(
            results=result["results"],
            top_similarity=result.get("top_similarity"),
            usage=result.get("usage"),
        )

    def enhanced_query(
        self,
        query: str,
        top_k: int,
        min_similarity: Optional[float] = None,
        search_mode: Optional[str] = None,
        dense_weight: Optional[float] = None,
        sparse_weight: Optional[float] = None,
        rerank: Optional[bool] = None,
        metadata_filter: Optional[Dict[str, Any]] = None,
        max_context_tokens: Optional[int] = None,
    ) -> EnhancedQueryResult:
        """
        Enhanced query with hybrid search, reranking, and filtering.

        Args:
            query: Query string
            top_k: Number of results to return
            min_similarity: Minimum similarity threshold (default: 0.3)
            search_mode: Retrieval mode (`dense`, `hybrid`, `full_text`, `hybrid_full_text`)
            dense_weight: Weight for dense embeddings (default: 0.7)
            sparse_weight: Weight for sparse BM25 (default: 0.3)
            rerank: Enable cross-encoder reranking
            metadata_filter: Metadata filter dictionary
            max_context_tokens: Maximum context tokens for result fitting

        Returns:
            EnhancedQueryResult with results, warnings, and context stats
        """
        result = self._service.enhanced_query(
            query,
            top_k,
            min_similarity,
            search_mode,
            dense_weight,
            sparse_weight,
            rerank,
            metadata_filter,
            max_context_tokens,
        )
        return EnhancedQueryResult(
            results=result["results"],
            top_similarity=result.get("top_similarity"),
            usage=result.get("usage"),
            warnings=result.get("warnings", []),
            context_stats=result.get("context_stats"),
        )

    def graph_search(
        self, query: str, top_k: int, max_hops: Optional[int] = None
    ) -> GraphSearchResult:
        """
        Perform graph-enhanced search (requires graph backend).

        Args:
            query: Query string
            top_k: Number of results to return
            max_hops: Maximum graph traversal hops (default: 2)

        Returns:
            GraphSearchResult with chunks and entities
        """
        result = self._service.graph_search(query, top_k, max_hops)
        return GraphSearchResult(
            chunks=result["chunks"],
            entities=result["entities"],
        )

    def stats(self) -> RagStats:
        """
        Get statistics about indexed documents.

        Returns:
            RagStats with embedding and text counts
        """
        stats = self._service.stats()
        return RagStats(
            embedding_count=stats["embedding_count"],
            text_count=stats["text_count"],
        )

    def clear(self) -> None:
        """Clear all indexed documents."""
        self._service.clear()


class RagTool:
    """
    RAG Tool for use with agents.

    This tool wraps RagService and provides a Tool interface for use in agents.
    Use this to create a shared RAG tool that can be used by multiple agents,
    ensuring they all share the same vector store backend.

    Example:
        >>> from caliche import AgentBuilder, ProviderType, RagTool
        >>> agent = AgentBuilder() \\
        ...     .provider(ProviderType.OpenAI) \\
        ...     .model("gpt-4o") \\
        ...     .api_key("sk-...") \\
        ...     .build()
        >>> rag_tool = RagTool(agent)
        >>> # Get the RagService to share the backend
        >>> rag_service = rag_tool.service()
        >>> # Use the shared tool in another agent
        >>> agent2 = AgentBuilder() \\
        ...     .provider(ProviderType.OpenAI) \\
        ...     .model("gpt-4o") \\
        ...     .api_key("sk-...") \\
        ...     .shared_rag_tool(rag_tool) \\
        ...     .build()
    """

    def __init__(self, agent: Agent):
        """
        Create a new RAG tool.

        Args:
            agent: An agent with an embedding provider configured
        """
        # Pass the agent directly - PyRagTool expects a PyAgent reference
        self._tool = _PyRagTool(agent)

    def service(self) -> RagService:
        """
        Get the underlying RAG service.

        This allows sharing the same backend between RagTool and RagService instances.

        Returns:
            A RagService instance that shares the same backend as this RagTool
        """
        py_service = self._tool.service()
        return RagService._from_py_service(py_service)
