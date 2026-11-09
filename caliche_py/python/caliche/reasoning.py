"""
Reasoning patterns for Caliche

Provides reasoning patterns: ReAct, Plan-Execute, Self-Ask, Reflexion, Tree of Thoughts

Note: ReActPattern, PlanExecutePattern, and ReflexionPattern now use native Rust bindings
for improved performance. The API remains the same.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict

from caliche import (
    Agent,
    PlanExecutePattern as _NativePlanExecutePattern,
    ReActPattern as _NativeReActPattern,
    ReflexionPattern as _NativeReflexionPattern,
    SelfAskPattern as _NativeSelfAskPattern,
    TreeOfThoughtsPattern as _NativeTreeOfThoughtsPattern,
)


class ReasoningPattern(Enum):
    """Available reasoning patterns"""

    REACT = "react"
    PLAN_EXECUTE = "plan-execute"
    SELF_ASK = "self-ask"
    REFLEXION = "reflexion"
    TREE_OF_THOUGHTS = "tree-of-thoughts"


class ReActPattern:
    """ReAct (Reasoning + Acting) pattern

    Uses native Rust bindings for high performance.
    Interleaves reasoning with tool usage for multi-step tasks.
    """

    def __init__(self, agent: Agent, max_iterations: int = 5):
        """Create a new ReActPattern

        Args:
            agent: The agent to use for reasoning
            max_iterations: Maximum number of thought-action cycles
        """
        self._native = _NativeReActPattern(agent, max_iterations)
        self.max_iterations = max_iterations
        self.pattern = ReasoningPattern.REACT

    def execute(self, query: str) -> Dict[str, Any]:
        """Execute the ReAct pattern

        Args:
            query: The query to process

        Returns:
            Dictionary containing:
            - final_answer: The final answer
            - steps: List of reasoning steps
            - success: Whether reasoning succeeded
            - error: Optional error message
            - token_usage: Optional token usage information
        """
        return self._native.execute(query)


class PlanExecutePattern:
    """Plan-and-Execute pattern

    Uses native Rust bindings for high performance.
    Separates planning from execution for complex workflows.
    """

    def __init__(self, agent: Agent, max_iterations: int = 10):
        """Create a new PlanExecutePattern

        Args:
            agent: The agent to use for reasoning
            max_iterations: Maximum number of planning iterations
        """
        self._native = _NativePlanExecutePattern(agent, max_iterations)
        self.max_iterations = max_iterations
        self.pattern = ReasoningPattern.PLAN_EXECUTE

    def execute(self, query: str) -> Dict[str, Any]:
        """Execute the Plan-Execute pattern

        Args:
            query: The query to process

        Returns:
            Dictionary containing:
            - final_answer: The final answer
            - steps: List of reasoning steps
            - success: Whether reasoning succeeded
            - error: Optional error message
            - token_usage: Optional token usage information
        """
        return self._native.execute(query)


class SelfAskPattern:
    """Self-Ask pattern

    Uses native Rust bindings for high performance.
    Recursively decomposes complex questions into simpler sub-questions.
    """

    def __init__(self, agent: Agent, max_iterations: int = 5):
        """Create a new SelfAskPattern

        Args:
            agent: The agent to use for reasoning
            max_iterations: Maximum depth for question decomposition
        """
        self._native = _NativeSelfAskPattern(agent, max_iterations)
        self.max_iterations = max_iterations
        self.pattern = ReasoningPattern.SELF_ASK

    def execute(self, query: str) -> Dict[str, Any]:
        """Execute the Self-Ask pattern

        Args:
            query: The query to process

        Returns:
            Dictionary containing:
            - final_answer: The final answer
            - steps: List of reasoning steps
            - success: Whether reasoning succeeded
            - error: Optional error message
            - token_usage: Optional token usage information
        """
        return self._native.execute(query)


class ReflexionPattern:
    """Reflexion (Self-Critique) pattern

    Uses native Rust bindings for high performance.
    Uses self-reflection and error correction.
    """

    def __init__(self, agent: Agent, max_iterations: int = 3):
        """Create a new ReflexionPattern

        Args:
            agent: The agent to use for reasoning
            max_iterations: Maximum number of reflection iterations
        """
        self._native = _NativeReflexionPattern(agent, max_iterations)
        self.max_iterations = max_iterations
        self.pattern = ReasoningPattern.REFLEXION

    def execute(self, query: str) -> Dict[str, Any]:
        """Execute the Reflexion pattern

        Args:
            query: The query to process

        Returns:
            Dictionary containing:
            - final_answer: The final answer
            - steps: List of reasoning steps
            - success: Whether reasoning succeeded
            - error: Optional error message
            - token_usage: Optional token usage information
        """
        return self._native.execute(query)


class TreeOfThoughtsPattern:
    """Tree of Thoughts pattern

    Uses native Rust bindings for high performance.
    Explores multiple reasoning paths in parallel using BFS/DFS search.
    """

    def __init__(
        self,
        agent: Agent,
        depth: int = 3,
        branches: int = 3,
        beam_width: int | None = None,
    ) -> None:
        """Create a new TreeOfThoughtsPattern

        Args:
            agent: The agent to use for reasoning
            depth: Maximum depth of the search tree
            branches: Number of branches to explore at each level
            beam_width: Optional beam width for pruning (defaults to branches)
        """
        if beam_width is not None:
            self._native = _NativeTreeOfThoughtsPattern(agent, depth, branches, beam_width)
        else:
            self._native = _NativeTreeOfThoughtsPattern(agent, depth, branches)
        self.depth = depth
        self.branches = branches
        self.beam_width = beam_width
        self.pattern = ReasoningPattern.TREE_OF_THOUGHTS

    def execute(self, query: str) -> Dict[str, Any]:
        """Execute the Tree of Thoughts pattern

        Args:
            query: The query to process

        Returns:
            Dictionary containing:
            - final_answer: The final answer
            - steps: List of reasoning steps
            - success: Whether reasoning succeeded
            - error: Optional error message
            - token_usage: Optional token usage information
        """
        return self._native.execute(query)
