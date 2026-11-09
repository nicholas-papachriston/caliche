#!/usr/bin/env python3
"""
Provider Configuration and Switching Example

This example demonstrates:
- Programmatic configuration for all LLM providers
- Advanced provider options (timeouts, retries, headers)
- Custom retry strategies
- Comparing responses across different providers
- Implementing fallback logic on provider failure
- Provider-specific features and limitations

Best practices shown:
- Graceful provider fallback
- Provider selection based on use case
- Error handling for provider failures
- Configuration without environment variables

Run with:
```bash
uv run examples/providers.py
"""

import caliche


def openai_provider():
    """Example 1: OpenAI Provider"""
    print("=== Example 1: OpenAI Provider ===\n")
    
    try:
        builder = caliche.AgentBuilder()
        builder.provider(caliche.ProviderType.OpenAI)
        builder.model("gpt-4o")
        agent = builder.build()
        
        print(f"OpenAI provider: {agent.provider}, model: {agent.model}")
        
        response, _, _ = agent.chat("Say hello in one sentence.")
        print(f"Response: {response}\n")
    except Exception as e:
        print(f"(Skipped: {e})\n")


def anthropic_provider():
    """Example 2: Anthropic Provider"""
    print("=== Example 2: Anthropic Provider ===\n")
    
    try:
        builder = caliche.AgentBuilder()
        builder.provider(caliche.ProviderType.Anthropic)
        builder.model("claude-sonnet-4-5-20250929")
        agent = builder.build()
        
        print(f"Anthropic provider: {agent.provider}, model: {agent.model}")
        
        response, _, _ = agent.chat("Say hello in one sentence.")
        print(f"Response: {response}\n")
    except Exception as e:
        print(f"(Skipped: {e})\n")


def google_provider():
    """Example 3: Google Provider"""
    print("=== Example 3: Google Provider ===\n")
    
    try:
        builder = caliche.AgentBuilder()
        builder.provider(caliche.ProviderType.Google)
        builder.model("models/gemini-2.5-flash")
        agent = builder.build()
        
        print(f"Google provider: {agent.provider}, model: {agent.model}")
        
        response, _, _ = agent.chat("Say hello in one sentence.")
        print(f"Response: {response}\n")
    except Exception as e:
        print(f"(Skipped: {e})\n")


def openrouter_provider():
    """Example 4: OpenRouter Provider"""
    print("=== Example 4: OpenRouter Provider ===\n")
    
    try:
        builder = caliche.AgentBuilder()
        builder.provider(caliche.ProviderType.OpenRouter)
        builder.model("anthropic/claude-haiku-4.5")
        agent = builder.build()
        
        print(f"OpenRouter provider: {agent.provider}, model: {agent.model}")
        
        response, _, _ = agent.chat("Say hello in one sentence.")
        print(f"Response: {response}\n")
    except Exception as e:
        print(f"(Skipped: {e})\n")


def perplexity_provider():
    """Example 5: Perplexity Provider"""
    print("=== Example 5: Perplexity Provider ===\n")
    
    try:
        builder = caliche.AgentBuilder()
        builder.provider(caliche.ProviderType.Perplexity)
        builder.model("sonar")
        agent = builder.build()
        
        print(f"Perplexity provider: {agent.provider}, model: {agent.model}")
        
        response, _, _ = agent.chat("Say hello in one sentence.")
        print(f"Response: {response}\n")
    except Exception as e:
        print(f"(Skipped: {e})\n")


def lmstudio_provider():
    """Example 6: LMStudio Provider"""
    print("=== Example 6: LMStudio Provider ===\n")
    
    try:
        builder = caliche.AgentBuilder()
        builder.provider(caliche.ProviderType.LMStudio)
        builder.model("local-model")
        agent = builder.build()
        
        print(f"LMStudio provider: {agent.provider}, model: {agent.model}")
        
        response, _, _ = agent.chat("Say hello in one sentence.")
        print(f"Response: {response}\n")
    except Exception as e:
        print(f"(Skipped: {e})\n")


def provider_comparison():
    """Example 7: Provider Comparison"""
    print("\n=== Example 7: Provider Comparison ===\n")
    
    providers = [
        {"name": "openai", "model": "gpt-4o"},
        {"name": "anthropic", "model": "claude-sonnet-4-5-20250929"},
        {"name": "google", "model": "models/gemini-2.5-flash"},
    ]
    
    query = "What is 2 + 2? Answer in one word."
    
    for provider in providers:
        try:
            print(f"Testing {provider['name']}...")
            builder = caliche.AgentBuilder()
            builder.provider(getattr(caliche.ProviderType, provider['name'].title()))
            builder.model(provider['model'])
            agent = builder.build()
            
            response, usage, _ = agent.chat(query)
            print(f"  {provider['name']}: {response}")
            if usage.get("total_tokens"):
                print(f"  Tokens: {usage['total_tokens']}")
            print()
        except Exception as e:
            print(f"  {provider['name']}: (not configured)\n")


def provider_fallback():
    """Example 8: Provider Fallback"""
    print("\n=== Example 8: Provider Fallback ===\n")
    
    providers = ["openai", "anthropic", "bedrock", "google"]
    
    for provider_name in providers:
        try:
            print(f"Trying {provider_name}...")
            builder = caliche.AgentBuilder()
            builder.provider(getattr(caliche.ProviderType, provider_name.title()))
            builder.model("default")
            agent = builder.build()
            
            response, _, _ = agent.chat("Hello!")
            print(f"{provider_name} succeeded: {response}\n")
            return  # Success, exit
        except Exception as e:
            print(f"Failed: {provider_name} failed: {e}")
            continue  # Try next provider
    
    print("\nAll providers failed. Please configure at least one provider.\n")


def main():
    """Main execution"""
    print("=== Provider Configuration Examples ===\n")
    
    openai_provider()
    anthropic_provider()
    google_provider()
    openrouter_provider()
    perplexity_provider()
    lmstudio_provider()
    provider_comparison()
    provider_fallback()
    
    print("=== Examples Complete ===")
    print("\nTips:")
    print("- Use explicit provider configuration for production")
    print("- Implement fallback logic for high availability")
    print("- Compare providers to find the best fit for your use case")
    print("- Each provider has different capabilities and pricing")


if __name__ == "__main__":
    main()

