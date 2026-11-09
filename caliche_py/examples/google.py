#!/usr/bin/env python3
"""
Google Gemini Provider Example

This example demonstrates:
- Using Google Gemini provider with AgentBuilder
- Model selection (gemini-2.5-flash, etc.)
- Streaming support with Google provider
- Comparison with other providers

Best practices shown:
- Environment variable configuration
- Error handling
- Token usage tracking

Run with:
```bash
uv run examples/google.py
```

Prerequisites:
- Set GOOGLE_API_KEY or GEMINI_API_KEY environment variable
"""

import caliche
import sys


def basic_google_usage():
    """Example 1: Basic Google Gemini Usage"""
    print("=== Example 1: Basic Google Gemini Usage ===\n")
    
    try:
        builder = caliche.AgentBuilder()
        builder.provider(caliche.ProviderType.Google)
        builder.model("models/gemini-2.5-flash")
        builder.enable_tool("calculator")
        agent = builder.build()
        
        response, usage, _ = agent.chat("What is 15 multiplied by 23?")
        
        print("Query: What is 15 multiplied by 23?")
        print(f"Response: {response}")
        if usage.get("total_tokens"):
            print(f"Tokens used: {usage['total_tokens']}")
        print()
    except Exception as e:
        print(f"(Skipped: {e})\n")


def google_streaming():
    """Example 2: Streaming with Google Provider"""
    print("\n=== Example 2: Streaming with Google Provider ===\n")
    
    try:
        builder = caliche.AgentBuilder()
        builder.provider(caliche.ProviderType.Google)
        builder.model("models/gemini-2.5-flash")
        agent = builder.build()
        
        print("Query: Write a haiku about programming")
        print("Response (streaming):\n")
        
        stream = agent.stream_chat("Write a haiku about programming")
        
        for event in stream:
            if event.type == "text_chunk":
                print(event.content, end="", flush=True)
            elif event.type == "metadata":
                if event.usage and event.usage.get("total_tokens"):
                    print(f"\n\nTokens: {event.usage['total_tokens']}")
        
        print("\n")
    except Exception as e:
        print(f"(Skipped: {e})\n")


def provider_comparison():
    """Example 3: Provider Comparison"""
    print("\n=== Example 3: Provider Comparison ===\n")
    
    query = "Explain quantum computing in one sentence"
    
    try:
        print("Testing Google Gemini...")
        builder = caliche.AgentBuilder()
        builder.provider(caliche.ProviderType.Google)
        builder.model("models/gemini-2.5-flash")
        agent = builder.build()
        
        response, _, _ = agent.chat(query)
        print(f"Google: {response}\n")
    except Exception as e:
        print(f"Google: (not configured)\n")


def main():
    """Main execution"""
    try:
        basic_google_usage()
        google_streaming()
        provider_comparison()
        
        print("=== Examples Complete ===")
        print("\nTips:")
        print("- Google Gemini offers fast inference")
        print("- Streaming provides real-time responses")
        print("- Use GOOGLE_API_KEY or GEMINI_API_KEY environment variable")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

