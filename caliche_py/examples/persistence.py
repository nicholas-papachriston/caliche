"""
Example: Memory Persistence in Python

Demonstrates how to use memory persistence to save and restore conversation state.

This example uses LM Studio by default (no API key required).
LM Studio runs locally at http://localhost:1234/v1

uv run examples/memory_persistence_example.py
"""

from caliche import AgentBuilder, BufferMemoryConfig, ProviderType
from caliche.memory import MemoryPersistence, MemorySnapshot, FileMemoryPersistence


def main():
    # Use file-based persistence
    file_persistence = FileMemoryPersistence(base_path="./conversations")
    
    conversation_id = "user-123-session-1"
    
    # Create agent with persistence
    # Using LM Studio (no API key required - runs locally at http://localhost:1234/v1)
    agent = AgentBuilder() \
        .provider(ProviderType.LMStudio) \
        .model("local-model") \
        .buffer_memory(BufferMemoryConfig(max_tokens=4096)) \
        .persist_memory_with(
            persistence=file_persistence,
            conversation_id=conversation_id,
            autosave=True
        ) \
        .build()
    
    # First conversation
    print("=== First Session ===")
    response, _, _ = agent.chat("My name is Alice and I'm a software engineer.")
    print(f"Agent: {response}\n")
    
    response, _, _ = agent.chat("What's my profession?")
    print(f"Agent: {response}\n")
    
    # Later, create a new agent with the same conversation ID
    print("=== Second Session (Resumed) ===")
    agent2 = AgentBuilder() \
        .provider(ProviderType.LMStudio) \
        .model("local-model") \
        .buffer_memory(BufferMemoryConfig(max_tokens=4096)) \
        .persist_memory_with(
            persistence=file_persistence,
            conversation_id=conversation_id,
            autosave=True
        ) \
        .build()
    
    # Agent remembers previous conversation
    response, _, _ = agent2.chat("What's my name and profession?")
    print(f"Agent: {response}\n")
    
    # Clear the conversation
    file_persistence.clear_snapshot(conversation_id)
    print("Conversation cleared.")


if __name__ == "__main__":
    main()

