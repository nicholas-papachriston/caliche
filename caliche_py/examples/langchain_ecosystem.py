#!/usr/bin/env python3
"""
LangChain Ecosystem Integration Example

This example demonstrates how to use various LangChain components with Caliche:
- LangChain Memory adapters
- LangChain Retriever adapters
- LangChain Chain adapters

Prerequisites:
- Install LangChain dependencies: uv sync --extra langchain
- Set OPENAI_API_KEY environment variable

Run with:
```bash
uv run examples/langchain_ecosystem.py
```
"""

import os
import sys
from dataclasses import dataclass, field
from typing import Any

from caliche.integrations.langchain import (
    LangChainMemoryAdapter,
    LangChainRetrieverAdapter,
    LangChainChainAdapter
)

# Try importing LangChain - will fail gracefully if not installed
try:
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import AIMessage, HumanMessage
    from langchain_core.prompts import PromptTemplate
except ImportError as exc:
    print("Error: LangChain dependencies not installed")
    print("\nPlease install LangChain dependencies:")
    print("  uv sync --extra langchain")
    print(f"\nImport error: {exc}")
    sys.exit(1)


@dataclass
class _SimpleChatMemory:
    """Tiny in-memory chat store compatible with LangChainMemoryAdapter."""

    messages: list[Any] = field(default_factory=list)

    def add_user_message(self, content: str) -> None:
        self.messages.append(HumanMessage(content=content))

    def add_ai_message(self, content: str) -> None:
        self.messages.append(AIMessage(content=content))


@dataclass
class _SimpleConversationBufferMemory:
    """Minimal shim with `chat_memory` expected by the adapter."""

    chat_memory: _SimpleChatMemory = field(default_factory=_SimpleChatMemory)


def example_memory_adapter():
    """Example: Using LangChain memory with Caliche"""
    print("=== Example 1: LangChain Memory Adapter ===\n")
    
    if not os.environ.get("OPENAI_API_KEY"):
        print("Skipping: OPENAI_API_KEY not set")
        return
    
    # Create lightweight LangChain-compatible memory
    langchain_memory = _SimpleConversationBufferMemory()
    langchain_memory.chat_memory.add_user_message("My name is Alice")
    langchain_memory.chat_memory.add_ai_message("Hello Alice! Nice to meet you.")
    
    # Wrap it with Caliche adapter
    memory_adapter = LangChainMemoryAdapter(langchain_memory)
    
    # Get messages from LangChain memory
    messages = memory_adapter.get_messages()
    print("Messages from LangChain memory:")
    for msg in messages:
        print(f"  {msg['role']}: {msg['content']}")
    print()
    
    # Add a new message
    memory_adapter.add_message("user", "What's my name?")
    messages = memory_adapter.get_messages()
    print("After adding message:")
    for msg in messages:
        print(f"  {msg['role']}: {msg['content']}")
    print()


def example_retriever_adapter():
    """Example: Using LangChain retriever with Caliche"""
    print("=== Example 2: LangChain Retriever Adapter ===\n")
    
    if not os.environ.get("OPENAI_API_KEY"):
        print("Skipping: OPENAI_API_KEY not set")
        return
    
    # Note: This is a simplified example
    # In practice, you would create a proper LangChain retriever with a vector store
    print("LangChain Retriever Adapter can be used to:")
    print("  - Query LangChain vector stores")
    print("  - Use LangChain document loaders")
    print("  - Integrate with LangChain's retrieval chains")
    print()
    print("Example usage:")
    print("  retriever = create_langchain_retriever(...)")
    print("  adapter = LangChainRetrieverAdapter(retriever)")
    print("  docs = await adapter.get_relevant_documents('query', top_k=5)")
    print()


def example_chain_adapter():
    """Example: Using LangChain chains with Caliche"""
    print("=== Example 3: LangChain Chain Adapter ===\n")
    
    if not os.environ.get("OPENAI_API_KEY"):
        print("Skipping: OPENAI_API_KEY not set")
        return
    
    # Create a LangChain runnable chain compatible with the adapter
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
    prompt = PromptTemplate(
        input_variables=["topic"],
        template="Write a short summary about {topic}."
    )
    chain = prompt | llm
    
    # Wrap it with Caliche adapter
    chain_adapter = LangChainChainAdapter(chain)
    
    # Run the chain
    import asyncio
    result = asyncio.run(chain_adapter.run({"topic": "artificial intelligence"}))
    print("Chain result:")
    print(f"  {result.get('output', result)}")
    print()


def main():
    """Run all examples"""
    print("=" * 60)
    print("LangChain Ecosystem Integration Examples")
    print("=" * 60)
    print()
    
    if not os.environ.get("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable is not set")
        print("\nPlease set your OpenAI API key:")
        print("  export OPENAI_API_KEY='sk-...'")
        sys.exit(1)
    
    try:
        example_memory_adapter()
        example_retriever_adapter()
        example_chain_adapter()
        
        print("=" * 60)
        print("All examples completed successfully!")
        print("=" * 60)
    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

