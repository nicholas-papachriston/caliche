"""
Chain patterns for Caliche

Provides chain patterns: LLMChain, SequentialChain, MapReduceChain, RetrievalChain

Note: LLMChain, SequentialChain, and MapReduceChain now use native Rust bindings
for improved performance. The API remains the same.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List

if TYPE_CHECKING:
    from caliche import Agent

from caliche import (
    Agent,
    LLMChain as _NativeLLMChain,
    MapReduceChain as _NativeMapReduceChain,
    SequentialChain as _NativeSequentialChain,
)


class LLMChain:
    """LLM Chain for combining prompt templates with LLM calls

    Uses native Rust bindings for high performance.
    """

    def __init__(self, agent: Agent, prompt_template: str):
        """Create a new LLMChain

        Args:
            agent: The agent to use for LLM calls
            prompt_template: Template string with {variable} placeholders
        """
        self._native = _NativeLLMChain(agent, prompt_template)

    def run(self, variables: Dict[str, Any]) -> Dict[str, Any]:
        """Run the chain with variables

        Args:
            variables: Dictionary of variables to substitute in the template

        Returns:
            Dictionary containing the chain output (typically has 'output' key with the result)
        """
        result = self._native.run(variables)
        # The native binding returns the ChainResult.output() which is already a dict
        # For LLMChain, this is typically {"output": "text"}
        return result


class SequentialChain:
    """Sequential chain for executing multiple chains in sequence

    Uses native Rust bindings for high performance.
    """

    def __init__(self):
        """Create a new SequentialChain"""
        self._native = _NativeSequentialChain()

    def add_chain(self, name: str, chain: LLMChain) -> SequentialChain:
        """Add a chain to the sequence

        Args:
            name: Name for this chain step
            chain: The LLMChain to add

        Returns:
            Self for method chaining
        """
        self._native.add_chain(name, chain._native)
        return self

    def run(self, initial_input: Dict[str, Any]) -> Dict[str, Any]:
        """Run all chains sequentially

        Args:
            initial_input: Initial input variables

        Returns:
            Dictionary containing the final chain output
        """
        return self._native.run(initial_input)


class MapReduceChain:
    """Map-Reduce chain for parallel processing

    Uses native Rust bindings for high performance.
    """

    def __init__(self, map_agent: Agent, map_prompt: str, reduce_agent: Agent, reduce_prompt: str):
        """Create a new MapReduceChain

        Args:
            map_agent: Agent to use for map phase
            map_prompt: Prompt template for map phase
            reduce_agent: Agent to use for reduce phase
            reduce_prompt: Prompt template for reduce phase
        """
        self._native = _NativeMapReduceChain(
            map_agent, map_prompt, reduce_agent, reduce_prompt
        )

    def run(self, items: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Run map-reduce over items

        Args:
            items: List of input dictionaries to process

        Returns:
            Dictionary containing the reduced result
        """
        return self._native.run(items)


class RetrievalChain:
    """Retrieval-Augmented Generation chain"""

    def __init__(self, agent: Agent, collection_name: str, top_k: int = 5):
        self.agent = agent
        self.collection_name = collection_name
        self.top_k = top_k

    async def run(self, query: str) -> str:
        """Run retrieval chain with query"""
        # Use RAG tool to retrieve context
        rag_prompt = f"Use the rag tool to search '{self.collection_name}' collection for: {query}. Then answer the question using the retrieved context."
        response, _, _ = self.agent.chat(rag_prompt)
        return response
