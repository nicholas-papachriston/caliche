#!/usr/bin/env python3
"""
Use a LangChain chat model as a Caliche external provider.

Prerequisites:
- `uv sync --extra langchain`
- Provider credentials (for example `OPENAI_API_KEY`)

Run:
```bash
uv run examples/langchain_provider.py
```
"""

import os
import sys
import asyncio
from caliche.integrations.langchain_llm import LangChainLLMProvider
from caliche import AgentBuilder

# Try importing LangChain - will fail gracefully if not installed
try:
    from langchain_openai import ChatOpenAI
    from langchain_anthropic import ChatAnthropic
except ImportError:
    print("Error: LangChain dependencies not installed")
    print("\nPlease install LangChain dependencies:")
    print("  uv sync --extra langchain")
    print("\nOr install manually:")
    print("  pip install langchain-core langchain-openai langchain-anthropic")
    sys.exit(1)


def example_openai_provider():
    """Example: Using OpenAI via LangChain"""
    print("=== Example 1: OpenAI Provider via LangChain ===\n")
    
    if not os.environ.get("OPENAI_API_KEY"):
        print("Skipping: OPENAI_API_KEY not set")
        return
    
    # Create LangChain LLM
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
    
    # Wrap it with Caliche adapter
    # Keep a reference to provider to ensure it stays alive
    provider = LangChainLLMProvider(llm, supports_native_tools=True, supports_streaming=True)
    
    # Create agent with LangChain provider
    builder = AgentBuilder()
    builder.external_provider(
        name="langchain-openai",
        model_name="gpt-4o-mini",
        complete_fn=provider.complete,
        stream_fn=provider.stream,
        supports_native_tools=True,
        supports_streaming=True
    )
    builder.enable_tool("calculator")
    builder.enable_tool("fs")
    
    agent = builder.build()
    
    # Keep provider reference alive (though PyObject should handle this)
    # This is just to be explicit
    _provider_ref = provider
    
    # Use the agent
    print("Agent created with LangChain OpenAI provider")
    print(f"Provider: {agent.provider()}, Model: {agent.model()}\n")
    
    print("Sending chat message...")
    try:
        response, usage, _ = agent.chat("What is 15 * 23? Use the calculator tool if needed.")
        print(f"Response: {response}")
        print(f"Token usage: {usage}\n")
    except Exception as e:
        print(f"Error during chat: {e}")
        import traceback
        traceback.print_exc()
        raise


def example_anthropic_provider():
    """Example: Using Anthropic via LangChain"""
    print("=== Example 2: Anthropic Provider via LangChain ===\n")
    
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Skipping: ANTHROPIC_API_KEY not set")
        return
    
    # Create LangChain LLM
    llm = ChatAnthropic(model="claude-sonnet-4-20250514", temperature=0.7)
    
    # Wrap it with Caliche adapter
    provider = LangChainLLMProvider(llm, supports_native_tools=True, supports_streaming=True)
    
    # Create agent with LangChain provider
    builder = AgentBuilder()
    builder.external_provider(
        name="langchain-anthropic",
        model_name="claude-sonnet-4-20250514",
        complete_fn=provider.complete,
        stream_fn=provider.stream,
        supports_native_tools=True,
        supports_streaming=True
    )
    builder.system_prompt("You are a helpful assistant that explains concepts clearly.")
    builder.enable_tool("calculator")
    
    agent = builder.build()
    
    print("Agent created with LangChain Anthropic provider")
    print(f"Provider: {agent.provider()}, Model: {agent.model()}\n")
    
    response, usage, _ = agent.chat("Explain quantum computing in simple terms.")
    print(f"Response: {response[:200]}...")
    print(f"Token usage: {usage}\n")


def example_streaming():
    """Example: Streaming with LangChain provider"""
    print("=== Example 3: Streaming with LangChain Provider ===\n")
    
    if not os.environ.get("OPENAI_API_KEY"):
        print("Skipping: OPENAI_API_KEY not set")
        return
    
    # Create LangChain LLM
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7, streaming=True)
    
    # Wrap it with Caliche adapter
    provider = LangChainLLMProvider(llm, supports_native_tools=False, supports_streaming=True)
    
    # Create agent with LangChain provider
    builder = AgentBuilder()
    builder.external_provider(
        name="langchain-openai-streaming",
        model_name="gpt-4o-mini",
        complete_fn=provider.complete,
        stream_fn=provider.stream,
        supports_native_tools=False,
        supports_streaming=True
    )
    
    agent = builder.build()
    
    print("Streaming response:")
    print("-" * 50)
    for event in agent.stream_chat("Write a short poem about AI."):
        if isinstance(event, dict):
            text = event.get("content") or event.get("text")
            if isinstance(text, str) and text:
                print(text, end="", flush=True)
        else:
            text = getattr(event, "content", None)
            if isinstance(text, str) and text:
                print(text, end="", flush=True)
    print()
    print("-" * 50)
    print()


def example_with_tools():
    """Example: Using LangChain provider with Caliche tools"""
    print("=== Example 4: LangChain Provider with Tools ===\n")
    
    if not os.environ.get("OPENAI_API_KEY"):
        print("Skipping: OPENAI_API_KEY not set")
        return
    
    # Create LangChain LLM
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
    
    # Wrap it with Caliche adapter
    provider = LangChainLLMProvider(llm, supports_native_tools=True, supports_streaming=True)
    
    # Create agent with LangChain provider and multiple tools
    builder = AgentBuilder()
    builder.external_provider(
        name="langchain-openai-tools",
        model_name="gpt-4o-mini",
        complete_fn=provider.complete,
        stream_fn=provider.stream,
        supports_native_tools=True,
        supports_streaming=True
    )
    builder.enable_tool("calculator")
    builder.enable_tool("fs")
    builder.enable_tool("fs")
    
    agent = builder.build()
    
    print("Agent with LangChain provider and tools")
    print(f"Provider: {agent.provider()}, Model: {agent.model()}\n")
    
    # Multi-turn conversation with tool usage
    response1, usage1, _ = agent.chat("Calculate 123 * 456 using the calculator.")
    print(f"Turn 1: {response1}")
    print(f"Usage: {usage1}\n")
    
    response2, usage2, _ = agent.chat("Now echo back: 'Hello from Caliche!'")
    print(f"Turn 2: {response2}")
    print(f"Usage: {usage2}\n")


def main():
    """Run all examples"""
    print("=" * 60)
    print("LangChain Provider Integration Examples")
    print("=" * 60)
    print()
    
    # Check for at least one API key
    has_openai = bool(os.environ.get("OPENAI_API_KEY"))
    has_anthropic = bool(os.environ.get("ANTHROPIC_API_KEY"))
    
    if not has_openai and not has_anthropic:
        print("Error: No API keys found")
        print("\nPlease set at least one of:")
        print("  export OPENAI_API_KEY='<openai-api-key>'")
        print("  export ANTHROPIC_API_KEY='<anthropic-api-key>'")
        sys.exit(1)
    
    try:
        # Run examples synchronously since agent.chat() is synchronous
        # and handles async internally via Rust's runtime
        example_openai_provider()
        example_anthropic_provider()
        example_streaming()
        example_with_tools()
        
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

