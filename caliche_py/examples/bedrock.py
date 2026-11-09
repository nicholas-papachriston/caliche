#!/usr/bin/env python3
"""
AWS Bedrock Provider Example

This example demonstrates AWS Bedrock streaming capabilities with the fully
type-safe streaming implementation.

Best practices shown:
- Bedrock provider configuration with AWS credentials
- Real-time streaming with proper event handling
- Token usage tracking from Bedrock streaming events
- Error handling for AWS-specific errors
- Multiple Bedrock model examples
- Tool calls in streaming mode

Prerequisites:
- AWS credentials configured (via environment variables or IAM role)
- AWS_REGION set (e.g., "us-east-1")
- BEDROCK_MODEL_ID set (optional, defaults to Claude Sonnet)

Run with:
```bash
export AWS_REGION=us-east-1
export AWS_ACCESS_KEY_ID=your_key
export AWS_SECRET_ACCESS_KEY=your_secret
uv run examples/bedrock.py
```
"""

import caliche
import sys


def basic_bedrock_streaming():
    """Example 1: Basic Bedrock Streaming"""
    print("=== Example 1: Basic Bedrock Streaming ===\n")
    
    try:
        builder = caliche.AgentBuilder()
        builder.provider(caliche.ProviderType.Bedrock)
        builder.model("us.anthropic.claude-haiku-4-5-20251001-v1:0")
        agent = builder.build()
        
        print("Query: Explain quantum computing in one sentence")
        print("Response (streaming):\n")
        
        stream = agent.stream_chat("Explain quantum computing in one sentence")
        
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


def bedrock_models():
    """Example 2: Different Bedrock Models"""
    print("\n=== Example 2: Different Bedrock Models ===\n")
    
    models = [
        "us.anthropic.claude-haiku-4-5-20251001-v1:0",
        "us.anthropic.claude-sonnet-4-20250514-v1:0",
        "us.anthropic.claude-opus-4-20250514-v1:0",
    ]
    
    for model in models:
        try:
            print(f"Testing model: {model}")
            builder = caliche.AgentBuilder()
            builder.provider(caliche.ProviderType.Bedrock)
            builder.model(model)
            agent = builder.build()
            
            response, _, _ = agent.chat("Say hello in one sentence.")
            print(f"Response: {response}\n")
        except Exception as e:
            print(f"(Skipped: {e})\n")


def bedrock_streaming_with_tools():
    """Example 3: Streaming with Tools"""
    print("\n=== Example 3: Streaming with Tools ===\n")
    
    try:
        builder = caliche.AgentBuilder()
        builder.provider(caliche.ProviderType.Bedrock)
        builder.model("us.anthropic.claude-haiku-4-5-20251001-v1:0")
        builder.enable_tool("calculator")
        agent = builder.build()
        
        print("Query: What is 15 * 23? Show your work.")
        print("Response (streaming):\n")
        
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


def long_form_streaming():
    """Example 4: Long-form Content Streaming"""
    print("\n=== Example 4: Long-form Content Streaming ===\n")
    
    try:
        builder = caliche.AgentBuilder()
        builder.provider(caliche.ProviderType.Bedrock)
        builder.model("us.anthropic.claude-haiku-4-5-20251001-v1:0")
        agent = builder.build()
        
        print("Query: Write a short story about a robot.")
        print("Response (streaming):\n")
        
        stream = agent.stream_chat("Write a short story about a robot.")
        
        for event in stream:
            if event["type"] == "text_chunk":
                print(event["content"], end="", flush=True)
        
        print("\n")
    except Exception as e:
        print(f"(Skipped: {e})\n")


def main():
    """Main execution"""
    try:
        basic_bedrock_streaming()
        bedrock_models()
        bedrock_streaming_with_tools()
        long_form_streaming()
        
        print("=== Examples Complete ===")
        print("\nTips:")
        print("- Use Bedrock for AWS-native integration")
        print("- Streaming provides better user experience")
        print("- Different models have different capabilities and costs")
    except Exception as e:
        print(f"Error: {e}")
        if "credentials" in str(e).lower():
            print("\nNote: Make sure AWS credentials are configured:")
            print("  - Set AWS_REGION environment variable")
            print("  - Set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY")
            print("  - Or use IAM role if running on EC2")
        sys.exit(1)


if __name__ == "__main__":
    main()

