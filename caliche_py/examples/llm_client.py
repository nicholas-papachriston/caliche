#!/usr/bin/env python3
"""
Non-Agentic LLM Client Example

This example demonstrates:
- Using LlmClient for simple, single-shot LLM completions
- Non-streaming completions with messages
- Streaming completions
- Convenience methods for text-only prompts
- Builder pattern configuration

Best practices shown:
- Direct LLM access without agent overhead
- Simple prompt/response workflows
- Streaming for real-time responses

Run with:
```bash
uv run examples/llm_client.py
"""

import caliche


def example_1_basic_completion():
    """Example 1: Basic Non-Streaming Completion"""
    print("=== Example 1: Basic Non-Streaming Completion ===\n")
    
    try:
        builder = caliche.LlmClient.builder()
        builder.provider(caliche.ProviderType.OpenAI)
        builder.model("gpt-4o")
        builder.temperature(0.7)
        builder.max_tokens(100)
        client = builder.build()
        
        response = client.complete_text("What is 2+2? Answer in one sentence.")
        print(f"Response: {response}\n")
    except Exception as e:
        print(f"(Skipped: {e})\n")


def example_2_completion_with_messages():
    """Example 2: Completion with Message History"""
    print("=== Example 2: Completion with Message History ===\n")
    
    try:
        builder = caliche.LlmClient.builder()
        builder.provider(caliche.ProviderType.OpenAI)
        builder.model("gpt-4o")
        client = builder.build()
        
        messages = [
            caliche.Message.system("You are a helpful assistant."),
            caliche.Message.user("What is the capital of France?"),
        ]
        
        response = client.complete(messages)
        print(f"Response: {response['content']}")
        if 'usage' in response:
            print(f"Tokens used: {response['usage']}\n")
    except Exception as e:
        print(f"(Skipped: {e})\n")


def example_3_streaming_completion():
    """Example 3: Streaming Completion"""
    print("=== Example 3: Streaming Completion ===\n")
    
    try:
        builder = caliche.LlmClient.builder()
        builder.provider(caliche.ProviderType.OpenAI)
        builder.model("gpt-4o")
        client = builder.build()
        
        print("Streaming response: ", end="", flush=True)
        stream = client.stream_text("Tell me a short story about a robot in exactly 3 sentences.")
        
        for event in stream:
            if event.get("type") == "text_chunk":
                print(event.get("content", ""), end="", flush=True)
            elif event.get("type") == "done":
                print("\nStream completed\n")
                break
    except Exception as e:
        print(f"\n(Skipped: {e})\n")


def example_4_streaming_with_messages():
    """Example 4: Streaming with Message History"""
    print("=== Example 4: Streaming with Message History ===\n")
    
    try:
        builder = caliche.LlmClient.builder()
        builder.provider(caliche.ProviderType.OpenAI)
        builder.model("gpt-4o")
        client = builder.build()
        
        messages = [
            caliche.Message.system("You are a creative writing assistant."),
            caliche.Message.user("Write a haiku about programming."),
        ]
        
        print("Streaming response: ", end="", flush=True)
        stream = client.stream(messages)
        
        for event in stream:
            if event.get("type") == "text_chunk":
                print(event.get("content", ""), end="", flush=True)
            elif event.get("type") == "done":
                print("\nStream completed\n")
                break
    except Exception as e:
        print(f"\n(Skipped: {e})\n")


def example_5_anthropic_provider():
    """Example 5: Using Anthropic Provider"""
    print("=== Example 5: Using Anthropic Provider ===\n")
    
    try:
        builder = caliche.LlmClient.builder()
        builder.provider(caliche.ProviderType.Anthropic)
        builder.model("claude-haiku-4-5-20251001")
        client = builder.build()
        
        print("Anthropic client created")
        response = client.complete_text("Say hello in one sentence.")
        print(f"Response: {response}\n")
    except Exception as e:
        print(f"(Skipped: {e})\n")


def example_6_custom_config():
    """Example 6: Custom Configuration"""
    print("=== Example 6: Custom Configuration ===\n")
    
    try:
        builder = caliche.LlmClient.builder()
        builder.provider(caliche.ProviderType.OpenAI)
        builder.model("gpt-4o")
        builder.timeout(30.0)  # 30 seconds
        builder.max_retries(3)
        builder.temperature(0.5)
        client = builder.build()
        
        print("Client created with custom timeout and retries")
        response = client.complete_text("What is the meaning of life? Answer briefly.")
        print(f"Response: {response}\n")
    except Exception as e:
        print(f"(Skipped: {e})\n")


def example_7_with_tools():
    """Example 7: Using Tools (Function Calling)"""
    print("=== Example 7: Using Tools (Function Calling) ===\n")
    
    try:
        import caliche
        
        # Define a tool schema
        weather_tool = {
            "name": "get_weather",
            "description": "Get the current weather in a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA"
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "The unit of temperature"
                    }
                },
                "required": ["location"]
            }
        }
        
        builder = caliche.LlmClient.builder()
        builder.provider(caliche.ProviderType.Anthropic)
        builder.model("claude-haiku-4-5-20251001")
        builder.tools([weather_tool])
        builder.tool_choice("auto")  # Let the LLM decide when to use tools
        client = builder.build()
        
        print("Client created with tools")
        messages = [
            caliche.Message.user("What's the weather like in San Francisco?")
        ]
        
        response = client.complete(messages)
        print(f"Response: {response['content']}")
        
        # Note: Tool calls may be present in the response but are not exposed
        # in the Python bindings. In a real application, you would check for
        # tool calls, execute them, and send the result back to the LLM
        # in a follow-up message.
        print("(Note: Tool calls are returned but not executed in this example)\n")
    except Exception as e:
        print(f"(Skipped: {e})\n")


def main():
    """Run all examples"""
    print("=== Non-Agentic LLM Client Examples ===\n")
    
    example_1_basic_completion()
    example_2_completion_with_messages()
    example_3_streaming_completion()
    example_4_streaming_with_messages()
    example_5_anthropic_provider()
    example_6_custom_config()
    example_7_with_tools()
    
    print("=== All Examples Complete ===")


if __name__ == "__main__":
    main()
