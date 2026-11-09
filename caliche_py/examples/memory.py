#!/usr/bin/env python3
"""
Memory Strategies Example

This example demonstrates:
- BufferMemory: Sliding window of recent messages
- SummaryMemory: Automatic condensation of older messages
- EntityMemory: Semantic tracking of facts and entities
- VectorMemory: Semantic retrieval of relevant conversation turns

Best practices shown:
- Choosing the right memory strategy for your use case
- Token-aware memory management

Run with:
```bash
uv run examples/memory.py
```
"""

import caliche


def buffer_memory():
    """Example 1: Buffer Memory"""
    print("=== Example 1: Buffer Memory ===\n")
    print("Buffer memory keeps a sliding window of recent messages.\n")
    
    try:
        builder = caliche.AgentBuilder()
        builder.provider(caliche.ProviderType.LMStudio)
        builder.buffer_memory({
            "max_tokens": 2048,
            "min_messages": 4,
            "preserve_system": True,
            "summary_placeholder": True,
        })
        agent = builder.build()
        
        agent.chat("Remember: the launch codename is Horizon.")
        response, _, _ = agent.chat("What's our launch codename?")
        print(f"Buffer memory agent: {response[:160]}...\n")
    except Exception as e:
        print(f"(Skipped: {e})\n")


def summary_memory():
    """Example 2: Summary Memory"""
    print("\n=== Example 2: Summary Memory ===\n")
    print("Summary memory automatically condenses older turns when the context grows.\n")
    
    try:
        builder = caliche.AgentBuilder()
        builder.provider(caliche.ProviderType.LMStudio)
        builder.summary_memory({
            "max_tokens": 8192,
            "summarize_threshold_pct": 0.75,
            "chunk_size": 8,
            "max_summaries": 6,
        })
        agent = builder.build()
        
        agent.chat("Please remember that I prefer async Rust examples.")
        response, _, _ = agent.chat("What kind of examples do I like?")
        print(f"Summary memory agent: {response[:160]}...\n")
    except Exception as e:
        print(f"(Skipped: {e})\n")


def entity_memory():
    """Example 3: Entity Memory"""
    print("\n=== Example 3: Entity Memory ===\n")
    print("Entity memory keeps track of facts about people, places, and concepts.\n")
    
    try:
        builder = caliche.AgentBuilder()
        builder.provider(caliche.ProviderType.LMStudio)
        builder.entity_memory({
            "max_tokens": 4096,
            "max_entities": 32,
            "extraction_interval": 3,
            "similarity_threshold": 0.6,
        })
        agent = builder.build()
        
        agent.chat("My teammate Nova owns the analytics pipeline.")
        response, _, _ = agent.chat("Who owns the analytics pipeline?")
        print(f"Entity memory agent: {response[:160]}...\n")
    except Exception as e:
        print(f"(Skipped: {e})\n")


def vector_memory():
    """Example 4: Vector Memory"""
    print("\n=== Example 4: Vector Memory ===\n")
    print("Vector memory stores conversation turns as embeddings and retrieves semantically")
    print("relevant memories based on the current query.\n")
    
    try:
        builder = caliche.AgentBuilder()
        builder.provider(caliche.ProviderType.LMStudio)
        builder.vector_memory({
            "max_tokens": 8192,
            "max_memories": 1000,
            "top_k": 5,
            "similarity_threshold": 0.7,
            "buffer_tokens": 4096,
            "namespace": "example-conversation",
        })
        agent = builder.build()
        
        # Simulate a long conversation with multiple topics
        agent.chat("I'm planning a trip to Japan next month. I want to visit Tokyo and Kyoto.")
        agent.chat("The weather in Japan during spring is usually mild and pleasant.")
        agent.chat("I'm also working on a Rust project that uses async/await patterns.")
        agent.chat("The project involves building a distributed system with multiple agents.")
        
        # Even after many turns, VectorMemory can retrieve semantically relevant information
        response, _, _ = agent.chat("What did I say about my travel plans?")
        print(f"Vector memory agent (travel query): {response[:160]}...\n")
        
        response, _, _ = agent.chat("Tell me about the technical project I mentioned.")
        print(f"Vector memory agent (project query): {response[:160]}...\n")
    except Exception as e:
        print(f"(Skipped: {e})\n")


def main():
    """Main execution"""
    buffer_memory()
    summary_memory()
    entity_memory()
    vector_memory()
    
    print("=== Examples Complete ===")
    print("\nTips:")
    print("- BufferMemory: Simple sliding window, good for short conversations")
    print("- SummaryMemory: Automatic condensation, good for medium conversations")
    print("- EntityMemory: Fact tracking, good for structured information")
    print("- VectorMemory: Semantic retrieval, good for long conversations")


if __name__ == "__main__":
    main()

