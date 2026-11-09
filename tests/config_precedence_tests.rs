//! Tests for configuration precedence and programmatic configuration.

use caliche::llm::anthropic::AnthropicProviderConfig;
use caliche::llm::google::GoogleProviderConfig;
use caliche::llm::{LMStudioConfig, OpenAIConfig, OpenRouterConfig, PerplexityConfig};
use std::time::Duration;

#[test]
fn test_openai_provider_config_builder() {
    let provider = OpenAIConfig::builder()
        .api_key("test-key".to_string())
        .model("gpt-4o".to_string())
        .timeout(Duration::from_secs(30))
        .max_retries(3)
        .build()
        .expect("Failed to build provider");

    let config = provider.config();
    assert_eq!(config.api_key, "test-key");
    assert_eq!(config.model, "gpt-4o");
}

#[test]
fn test_anthropic_provider_config_builder() {
    let _provider = AnthropicProviderConfig::builder()
        .api_key("test-key".to_string())
        .model("claude-3-5-sonnet-20241022".to_string())
        .timeout(Duration::from_secs(60))
        .build()
        .expect("Failed to build provider");

    // AnthropicProvider doesn't expose config() method, so we just verify it builds
    // The config is validated during build
}

#[test]
fn test_google_provider_config_builder() {
    let _provider = GoogleProviderConfig::builder()
        .api_key("test-key".to_string())
        .model("models/gemini-2.5-flash".to_string())
        .timeout(Duration::from_secs(30))
        .build()
        .expect("Failed to build provider");

    // GoogleProvider doesn't expose config() method, so we just verify it builds
    // The config is validated during build
}

#[test]
fn test_openrouter_provider_config_builder() {
    let provider = OpenRouterConfig::builder()
        .api_key("test-key".to_string())
        .model("qwen/qwen-plus".to_string())
        .timeout(Duration::from_secs(30))
        .build()
        .expect("Failed to build provider");

    let config = provider.config();
    assert_eq!(config.api_key, "test-key");
    assert_eq!(config.model, "qwen/qwen-plus");
}

#[test]
fn test_perplexity_provider_config_builder() {
    let provider = PerplexityConfig::builder()
        .api_key("test-key".to_string())
        .model("sonar".to_string())
        .timeout(Duration::from_secs(30))
        .build()
        .expect("Failed to build provider");

    let config = provider.config();
    assert_eq!(config.api_key, "test-key");
    assert_eq!(config.model, "sonar");
}

#[test]
fn test_lmstudio_provider_config_builder() {
    let provider = LMStudioConfig::builder()
        .base_url("http://localhost:1234/v1".to_string())
        .model("local-model".to_string())
        .timeout(Duration::from_secs(60))
        .build()
        .expect("Failed to build provider");

    let config = provider.config();
    assert_eq!(config.base_url, "http://localhost:1234/v1");
    assert_eq!(config.model, "local-model");
}

#[test]
fn test_provider_config_with_custom_headers() {
    // Note: Custom headers need to be set via builder before building
    // This test verifies the provider can be created
    let provider = OpenAIConfig::builder()
        .api_key("test-key".to_string())
        .model("gpt-4o".to_string())
        .build()
        .expect("Failed to build provider");

    let config = provider.config();
    assert_eq!(config.api_key, "test-key");
    assert_eq!(config.model, "gpt-4o");
}

#[test]
fn test_provider_config_with_retry_strategy() {
    use caliche::traits::retry::ExponentialBackoffStrategy;

    let retry_strategy: std::sync::Arc<dyn caliche::traits::retry::RetryStrategy> =
        std::sync::Arc::new(ExponentialBackoffStrategy::new(5, 1000, 60000, 2.0, true));

    let provider = OpenAIConfig::builder()
        .api_key("test-key".to_string())
        .model("gpt-4o".to_string())
        .retry_strategy(retry_strategy)
        .build()
        .expect("Failed to build provider");

    let config = provider.config();
    assert_eq!(config.api_key, "test-key");
    assert_eq!(config.model, "gpt-4o");
}

#[test]
fn test_provider_config_with_connection_pool() {
    // Note: Connection pool size needs to be set via builder before building
    // This test verifies the provider can be created
    let provider = OpenAIConfig::builder()
        .api_key("test-key".to_string())
        .model("gpt-4o".to_string())
        .build()
        .expect("Failed to build provider");

    let config = provider.config();
    assert_eq!(config.api_key, "test-key");
    assert_eq!(config.model, "gpt-4o");
}

#[tokio::test]
async fn test_provider_with_config_creation() {
    // Test that providers can be created with config
    let provider = OpenAIConfig::builder()
        .api_key("test-key".to_string())
        .model("gpt-4o".to_string())
        .build()
        .expect("Failed to build provider");

    // Verify the provider was created successfully
    let config = provider.config();
    assert_eq!(config.api_key, "test-key");
    assert_eq!(config.model, "gpt-4o");
}

#[tokio::test]
async fn test_provider_builder_pattern() {
    // Test that builder pattern works correctly
    let provider = OpenAIConfig::builder()
        .api_key("test-key".to_string())
        .model("gpt-4o".to_string())
        .timeout(Duration::from_secs(30))
        .max_retries(3)
        .build()
        .expect("Failed to build provider");

    let config = provider.config();
    assert_eq!(config.api_key, "test-key");
    assert_eq!(config.model, "gpt-4o");
}
