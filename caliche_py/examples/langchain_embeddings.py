#!/usr/bin/env python3
"""
LangChain Embeddings Integration Example

This example demonstrates how to use LangChain embedding providers with Caliche
for vector memory and RAG features.

Prerequisites:
- Install LangChain dependencies: uv sync --extra langchain
- Set OPENAI_API_KEY environment variable (or other provider credentials)

Run with:
```bash
uv run examples/langchain_embeddings.py
```
"""

import os
import sys
from caliche.integrations.langchain_embeddings import LangChainEmbeddingProvider
from caliche import AgentBuilder, ProviderType

# Try importing LangChain - will fail gracefully if not installed
try:
    from langchain_openai import OpenAIEmbeddings
except ImportError:
    print("Error: LangChain dependencies not installed")
    print("\nPlease install LangChain dependencies:")
    print("  uv sync --extra langchain")
    sys.exit(1)


def example_vector_memory():
    """Example: Using LangChain embeddings for vector memory"""
    print("=== Example 1: Vector Memory with LangChain Embeddings ===\n")
    
    if not os.environ.get("OPENAI_API_KEY"):
        print("Skipping: OPENAI_API_KEY not set")
        return
    
    # Create LangChain embeddings
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    # Wrap it with Caliche adapter
    embedding_provider = LangChainEmbeddingProvider(embeddings)
    
    # Create agent with LangChain embeddings for vector memory
    builder = AgentBuilder()
    builder.provider(ProviderType.OpenAI)  # Use OpenAI for LLM
    builder.model("gpt-4o-mini")
    builder.external_embedding_provider(
        name="langchain-openai-embeddings",
        embed_fn=embedding_provider.embed
    )
    builder.vector_memory({
        "max_tokens": 8192,
        "max_memories": 1000,
        "top_k": 5,
        "similarity_threshold": 0.7,
        "buffer_tokens": 4096
    })
    
    agent = builder.build()
    
    print("Agent created with LangChain embeddings for vector memory")
    print(f"Provider: {agent.provider}, Model: {agent.model}\n")
    
    # Add some memories
    response1, _, _ = agent.chat("My favorite programming language is Rust.")
    print(f"Response 1: {response1[:100]}...\n")
    
    response2, _, _ = agent.chat("What is my favorite programming language?")
    print(f"Response 2: {response2}\n")
    
    # The agent should remember from the conversation history
    response3, _, _ = agent.chat("Why do I like Rust?")
    print(f"Response 3: {response3[:200]}...\n")


def example_rag_with_embeddings():
    """Example: Using LangChain embeddings for RAG"""
    print("=== Example 2: RAG with LangChain Embeddings ===\n")
    
    if not os.environ.get("OPENAI_API_KEY"):
        print("Skipping: OPENAI_API_KEY not set")
        return
    
    # Create LangChain embeddings
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    # Wrap it with Caliche adapter
    embedding_provider = LangChainEmbeddingProvider(embeddings)
    
    # Create agent with LangChain embeddings for RAG
    builder = AgentBuilder()
    builder.provider(ProviderType.OpenAI)
    builder.model("gpt-4o-mini")
    builder.external_embedding_provider(
        name="langchain-openai-embeddings",
        embed_fn=embedding_provider.embed
    )
    builder.enable_tool("semantic_search")  # Enable RAG tool which uses embeddings
    
    agent = builder.build()
    
    print("Agent created with LangChain embeddings for RAG")
    print(f"Provider: {agent.provider}, Model: {agent.model}\n")
    
    # Index some documents (this would typically be done via RAG service)
    # For this example, we'll just demonstrate the setup
    print("Agent is ready to use RAG with LangChain embeddings")
    print("You can now index documents and query them using the RAG tool\n")


def main():
    """Run all examples"""
    print("=" * 60)
    print("LangChain Embeddings Integration Examples")
    print("=" * 60)
    print()
    
    if not os.environ.get("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable is not set")
        print("\nPlease set your OpenAI API key:")
        print("  export OPENAI_API_KEY='sk-...'")
        sys.exit(1)
    
    try:
        example_vector_memory()
        example_rag_with_embeddings()
        
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

