#!/usr/bin/env python3
"""
Streaming Responses Example

This example demonstrates how to use the Agent's streaming API directly
for real-time token-by-token output.

Best practices shown:
- Proper stream event handling
- Real-time token display with flushing
- Tool call detection in streams
- Graceful error handling
- Token usage tracking

Run with:
```bash
uv run examples/streaming.py
```
"""

import caliche
import sys


def basic_streaming():
    """Example 1: Basic Streaming"""
    print("=== Example 1: Basic Streaming ===\n")
    
    try:
        builder = caliche.AgentBuilder()
        builder.provider(caliche.ProviderType.Bedrock)
        builder.model("anthropic.claude-3-sonnet-20240229-v1:0")
        agent = builder.build()
        
        print("Query: Write a haiku about programming")
        print("Response (streaming):")
        
        stream = agent.stream_chat("Write a haiku about programming")
        
        for event in stream:
            if event["type"] == "text_chunk":
                print(event["content"], end="", flush=True)
            elif event["type"] == "metadata":
                usage = event.get("usage")
                if usage and usage.get("total_tokens"):
                    print(f"\n\nTokens: {usage['total_tokens']}")
        
        print("\n")
    except Exception as e:
        print(f"(Skipped: {e})\n")


def streaming_with_tools():
    """Example 2: Streaming with Tool Calls"""
    print("\n=== Example 2: Streaming with Tool Calls ===\n")
    
    try:
        builder = caliche.AgentBuilder()
        builder.provider(caliche.ProviderType.Bedrock)
        builder.model("anthropic.claude-3-sonnet-20240229-v1:0")
        builder.enable_tool("calculator")
        agent = builder.build()
        
        print("Query: What is 15 * 23? Show your work.")
        print("Response (streaming):")
        
        stream = agent.stream_chat("What is 15 * 23? Show your work.")
        
        for event in stream:
            if event["type"] == "text_chunk":
                print(event["content"], end="", flush=True)
            elif event["type"] == "tool_call_start":
                print(f"\n[Tool Call] {event['name']}")
                print(f"Arguments: {event['arguments']}\n")
            elif event["type"] == "tool_call_result":
                print(f"\n[Tool Result] {event['result']}\n")
            elif event["type"] == "metadata":
                usage = event.get("usage")
                if usage and usage.get("total_tokens"):
                    print(f"\n\nToken usage: {usage['total_tokens']}")
        
        print("\n")
    except Exception as e:
        print(f"(Skipped: {e})\n")


def main():
    """Main execution"""
    try:
        basic_streaming()
        streaming_with_tools()
        
        print("=== Examples Complete ===")
        print("\nTips:")
        print("- Use streaming for better user experience")
        print("- Handle tool calls in streams for interactive agents")
        print("- Monitor token usage for cost tracking")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

