use caliche::llm::azure_openai::{AzureOpenAIProvider, AzureOpenAIProviderConfig};
use caliche::llm::openai_compat::OpenAICompatConfig;

#[expect(
    clippy::expect_used,
    reason = "test helper: a failure fails the calling test"
)]
fn provider_for_deployment(deployment_id: &str) -> AzureOpenAIProvider {
    AzureOpenAIProvider::with_config(AzureOpenAIProviderConfig::new_api_key(
        "my-resource",
        deployment_id,
        "test-key",
    ))
    .expect("azure provider should build")
}

#[test]
fn azure_gpt4o_uses_max_completion_tokens() {
    let provider = provider_for_deployment("gpt-4o");
    let compat = provider.openai_compat();
    assert!(compat.uses_max_completion_tokens());
    assert!(!compat.requires_default_temperature());
}

#[test]
fn azure_o_series_requires_default_temperature() {
    let provider = provider_for_deployment("o3-mini");
    let compat = provider.openai_compat();
    assert!(compat.uses_max_completion_tokens());
    assert!(compat.requires_default_temperature());
}

#[test]
fn azure_chat_completions_url_includes_deployment_and_api_version() {
    let provider = provider_for_deployment("gpt-4o-mini");
    let compat = provider.openai_compat();
    assert_eq!(
        compat.chat_completions_url(),
        Some(
            "https://my-resource.openai.azure.com/openai/deployments/gpt-4o-mini/chat/completions?api-version=2024-10-21"
        )
    );
}

#[test]
fn azure_embeddings_url_uses_deployment_path() {
    let provider = provider_for_deployment("text-embedding-3-small");
    let compat = provider.openai_compat();
    assert_eq!(
        compat.embeddings_url(),
        Some(
            "https://my-resource.openai.azure.com/openai/deployments/text-embedding-3-small/embeddings?api-version=2024-10-21"
        )
    );
}

#[test]
fn azure_audio_deployment_supports_audio_input() {
    let provider = provider_for_deployment("gpt-4o-realtime");
    let compat = provider.openai_compat();
    assert!(compat.supports_audio_input());
}
