#!/usr/bin/env python3
"""
Agent Creation and Configuration Example

This example demonstrates:
- Creating agents with explicit configuration
- Runtime configuration through a single builder path
- Custom system prompts for specialized agents
- Explicit tool selection
- Multiple chat turns with context
- Shared state management

Best practices shown:
- Proper error handling
- Token usage tracking
- Consistent builder-first configuration
- Type-safe agent creation

Run with:
```bash
uv run examples/agent.py
```

Set environment variables and consume them via builder configuration:
```bash
export OPENAI_API_KEY=...
uv run examples/agent.py
"""

import caliche
import json


def explicit_configuration():
    """Example 1: Explicit configuration (recommended for production)"""
    print("=== Example 1: Explicit Configuration ===\n")
    
    builder = caliche.AgentBuilder()
    builder.provider(caliche.ProviderType.Bedrock)
    builder.model("us.anthropic.claude-haiku-4-5-20251001-v1:0")
    
    try:
        agent = builder.build()
        print(f"Agent created with provider: {agent.provider}, model: {agent.model}")
        
        response, usage, _ = agent.chat("What is 15 * 23?")
        print(f"\nQuery: What is 15 * 23?")
        print(f"Response: {response}")
        if usage.get("total_tokens"):
            print(f"Tokens: {usage['total_tokens']}")
    except Exception as e:
        print(f"Note: {e}\n")


def multiple_chat_turns():
    """Example 2: Multiple chat turns (maintains context)"""
    print("\n=== Example 2: Multiple Chat Turns (Context) ===\n")
    
    builder = caliche.AgentBuilder()
    builder.provider(caliche.ProviderType.Bedrock)
    builder.model("us.anthropic.claude-haiku-4-5-20251001-v1:0")
    
    try:
        agent = builder.build()
        
        # First turn
        response1, _, _ = agent.chat("Remember this number: 42. What is that number plus 10?")
        print("Query: Remember this number: 42. What is that number plus 10?")
        print(f"Response: {response1}\n")
        
        # Second turn - agent remembers context
        response2, _, _ = agent.chat("What was the number I asked you to remember?")
        print("Query: What was the number I asked you to remember?")
        print(f"Response: {response2}\n")
    except Exception as e:
        print(f"Note: {e}\n")


def runtime_config():
    """Example 3: Runtime configuration through the canonical builder path."""
    print("\n=== Example 3: Runtime Configuration ===\n")
    try:
        builder = caliche.AgentBuilder()
        builder.provider(caliche.ProviderType.Bedrock)
        builder.model("us.anthropic.claude-haiku-4-5-20251001-v1:0")
        builder.enable_tool("calculator")
        builder.enable_tool("fs")
        config_agent = builder.build()

        response, _, _ = config_agent.chat("Echo this: Hello from builder-first runtime config!")
        print("Query: Echo this: Hello from builder-first runtime config!")
        print(f"Response: {response}\n")
    except Exception as e:
        print(f"Note: {e}\n")


def custom_system_prompt():
    """Example 4: Builder with custom system prompt"""
    print("\n=== Example 4: Custom System Prompt ===\n")
    
    builder = caliche.AgentBuilder()
    builder.provider(caliche.ProviderType.Bedrock)
    builder.system_prompt(
        """You are a Rust programming expert. 
When answering questions, focus on Rust best practices, 
memory safety, and performance considerations."""
    )
    builder.enable_tool("fs")
    
    try:
        coding_agent = builder.build()
        
        response, _, _ = coding_agent.chat("What are the key principles of Rust memory safety?")
        print("Query: What are the key principles of Rust memory safety?")
        print(f"Response: {response}\n")
    except Exception as e:
        print(f"Note: {e}\n")


def all_default_tools():
    """Example 5: Agent with all default tools"""
    print("\n=== Example 5: All Default Tools ===\n")
    
    builder = caliche.AgentBuilder()
    builder.provider(caliche.ProviderType.Bedrock)
    builder.enable_all_default_tools()
    
    try:
        full_agent = builder.build()
        
        print("Available tools:")
        tools = full_agent.list_tools()
        for tool in tools:
            print(f"  - {tool['name']}")
        print()
    except Exception as e:
        print(f"Note: {e}\n")


def shared_state_management():
    """Example 6: Shared state management"""
    print("\n=== Example 6: Shared State Management ===\n")
    
    # Create a shared state that multiple agents can access
    shared_state = caliche.SharedState()
    
    # Initialize state with some data
    print("1. Setting initial state values...")
    shared_state.set("/order_id", "ORD-12345")
    shared_state.set("/customer/name", "Alice")
    shared_state.set("/customer/email", "alice@example.com")
    
    print("   Set order_id: ORD-12345")
    print("   Set customer name: Alice")
    print("   Set customer email: alice@example.com\n")
    
    # Get individual values
    print("2. Retrieving values from state...")
    order_id = shared_state.get("/order_id")
    customer_name = shared_state.get("/customer/name")
    
    print(f"   Order ID: {order_id}")
    print(f"   Customer Name: {customer_name}\n")
    
    # Get all state
    print("3. Getting entire state...")
    all_state = shared_state.get_all()
    print(f"   Full state: {json.dumps(all_state, indent=2)}\n")
    
    # Update state
    print("4. Updating order status...")
    shared_state.set("/order/status", "processing")
    status = shared_state.get("/order/status")
    print(f"   Order Status: {status}\n")
    
    # Demonstrate nested paths
    print("5. Setting nested values...")
    shared_state.set("/order/items/0/name", "Widget")
    shared_state.set("/order/items/0/quantity", 2)
    
    item_name = shared_state.get("/order/items/0/name")
    item_quantity = shared_state.get("/order/items/0/quantity")
    
    print(f"   Item Name: {item_name}")
    print(f"   Item Quantity: {item_quantity}\n")
    
    # Final state
    print("6. Final state:")
    final_state = shared_state.get_all()
    print(json.dumps(final_state, indent=2))


def main():
    """Main execution"""
    try:
        # Configuration pattern examples
        explicit_configuration()
        multiple_chat_turns()
        runtime_config()
        custom_system_prompt()
        all_default_tools()
        
        # Feature examples
        shared_state_management()
        
        print("\n=== Examples Complete ===")
        print("\nTips:")
        print("- Use one canonical path: AgentBuilder() + fluent configuration + build()")
        print("- Keep provider/model explicit in application code")
        print("- Shared state enables agents to coordinate and share data")
    except Exception as e:
        print(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

