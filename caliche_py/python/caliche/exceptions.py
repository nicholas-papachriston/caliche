"""
Custom exception hierarchy for Caliche.

Provides specific exception types for better error handling and debugging.
"""

from __future__ import annotations


class CalicheError(Exception):
    """Base exception for Caliche.

    All Caliche-specific exceptions inherit from this class.
    """

    pass


class AgentNotInitializedError(CalicheError):
    """Raised when an agent operation is attempted before the agent is initialized."""

    pass


class ProviderError(CalicheError):
    """Raised when an LLM provider operation fails.

    This includes API errors, authentication failures, rate limiting, etc.
    """

    pass


class ToolExecutionError(CalicheError):
    """Raised when a tool execution fails.

    This includes validation errors, runtime errors during tool execution,
    and other tool-related failures.
    """

    pass


class WorkflowError(CalicheError):
    """Raised when a workflow operation fails.

    This includes workflow execution errors, state transition errors,
    and other workflow-related failures.
    """

    pass


class RagError(CalicheError):
    """Raised when a RAG operation fails.

    This includes indexing errors, query errors, embedding generation failures,
    and other RAG-related errors.
    """

    pass


class ChainError(CalicheError):
    """Raised when a chain operation fails.

    This includes chain execution errors, template rendering errors,
    and other chain-related failures.
    """

    pass


class MemoryError(CalicheError):
    """Raised when a memory operation fails.

    This includes memory persistence errors, loading errors,
    and other memory-related failures.
    """

    pass


class CoordinatorError(CalicheError):
    """Raised when an agent coordinator operation fails.

    This includes agent registration errors, delegation errors,
    and other coordinator-related failures.
    """

    pass


class EvaluationError(CalicheError):
    """Raised when an evaluation operation fails.

    This includes evaluator errors, dataset loading errors,
    and other evaluation-related failures.
    """

    pass
