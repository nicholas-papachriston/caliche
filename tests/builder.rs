use caliche::AgentBuilder;
/// AgentBuilder tests
use caliche::prelude::ProviderType;

#[tokio::test]
async fn test_builder_new() {
    let _builder = AgentBuilder::new();
    // Builder should be created successfully
}

#[tokio::test]
async fn test_builder_with_provider() {
    let _builder = AgentBuilder::new().provider(ProviderType::OpenAI);
    // Provider should be set successfully
}

#[tokio::test]
async fn test_builder_with_model() {
    let _builder = AgentBuilder::new().model("gpt-4".to_string());
    // Model should be set successfully
}

#[tokio::test]
async fn test_builder_with_system_prompt() {
    let prompt = "You are a helpful assistant";
    let _builder = AgentBuilder::new().system_prompt(prompt.to_string());
    // System prompt should be set successfully
}

#[tokio::test]
async fn test_builder_enable_tool() {
    let _builder = AgentBuilder::new().enable_tool("calculator");
    // Tool should be enabled successfully
}

#[tokio::test]
async fn test_builder_enable_multiple_tools() {
    let _builder = AgentBuilder::new()
        .enable_tool("calculator")
        .enable_tool("fs");

    // Multiple tools should be enabled successfully
}

#[tokio::test]
async fn test_builder_chaining() {
    let _builder = AgentBuilder::new()
        .provider(ProviderType::Anthropic)
        .model("claude-3-opus-20240229".to_string())
        .system_prompt("Test system prompt".to_string())
        .enable_tool("calculator");

    // All builder methods should chain successfully
}

#[tokio::test]
async fn test_builder_default() {
    let _builder = AgentBuilder::default();
    // Default builder should be created successfully
}

#[tokio::test]
async fn test_builder_all_providers() {
    let providers = vec![
        ProviderType::Bedrock,
        ProviderType::OpenAI,
        ProviderType::Anthropic,
        ProviderType::OpenRouter,
        ProviderType::Perplexity,
        ProviderType::LMStudio,
    ];

    for provider in providers {
        let _builder = AgentBuilder::new().provider(provider);
        // Each provider should be set successfully
    }
}
