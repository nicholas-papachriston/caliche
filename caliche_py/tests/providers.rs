use caliche::llm::ProviderType as RustProviderType;
use caliche_py::providers::ProviderType;

#[test]
fn provider_constants_match_core_variants() {
    let cases = [
        (ProviderType::OpenAI, RustProviderType::OpenAI),
        (ProviderType::Anthropic, RustProviderType::Anthropic),
        (ProviderType::Bedrock, RustProviderType::Bedrock),
        (ProviderType::OpenRouter, RustProviderType::OpenRouter),
        (
            ProviderType::CloudflareAIGateway,
            RustProviderType::CloudflareAIGateway,
        ),
        (ProviderType::Perplexity, RustProviderType::Perplexity),
        (ProviderType::LMStudio, RustProviderType::LMStudio),
        (ProviderType::Google, RustProviderType::Google),
        (ProviderType::Cohere, RustProviderType::Cohere),
        (ProviderType::AzureOpenAI, RustProviderType::AzureOpenAI),
        (ProviderType::Ollama, RustProviderType::Ollama),
        (ProviderType::External, RustProviderType::External),
        #[cfg(any(
            feature = "local-inference",
            feature = "local-inference-candle",
            feature = "local-inference-ort",
            feature = "local-inference-llamacpp"
        ))]
        (
            ProviderType::LocalInference,
            RustProviderType::LocalInference,
        ),
    ];

    for (py_provider, rust_provider) in cases {
        let converted: RustProviderType = py_provider.into();
        assert_eq!(converted, rust_provider);
    }
}

#[test]
fn provider_from_core_round_trips() {
    let original = RustProviderType::OpenRouter;
    let py_provider = ProviderType::from(original);
    let restored: RustProviderType = py_provider.into();
    assert_eq!(restored, original);
}
