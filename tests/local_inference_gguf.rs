//! Local inference GGUF / llama.cpp provider smoke test.

#![cfg(feature = "local-inference-llamacpp")]

use caliche::llm::{
    GenerationParams, GgufBootstrapConfig, InferenceBackendType, LocalInferenceBootstrapConfig,
    LocalInferenceConfig, LocalInferenceProvider, Provider,
};
use caliche::messages::{CompletionRequest, Message};
use dripstone::inference::DeviceConfig;
use dripstone::loader::ModelSource;
use std::path::PathBuf;
use std::sync::Arc;

fn models_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("target/llamacpp-caliche-test-models")
}

#[tokio::test]
async fn local_inference_provider_runs_stories15m_gguf() {
    let local_dir = models_dir().join("stories15m-gguf");
    let bootstrap = LocalInferenceBootstrapConfig {
        continue_on_error: false,
        allow_curl_fallback: true,
        max_download_bytes: Some(25 * 1024 * 1024),
        gguf: Some(GgufBootstrapConfig {
            local_dir: local_dir.clone(),
            repo: Some("ggml-org/models-moved".to_string()),
            revision: None,
            filename: Some("tinyllamas/stories15M-q4_0.gguf".to_string()),
            redirect_source_to_local: true,
        }),
        candle: None,
        ort: None,
        ..LocalInferenceBootstrapConfig::default()
    };

    let config = LocalInferenceConfig::new(
        ModelSource::huggingface("ggml-org/models-moved", None),
        InferenceBackendType::LlamaCpp,
    )
    .with_device(DeviceConfig::Cpu)
    .with_bootstrap(bootstrap)
    .with_generation_params(
        GenerationParams::default()
            .with_max_tokens(12)
            .with_temperature(0.0),
    );

    let provider = LocalInferenceProvider::with_config(config)
        .await
        .expect("create local inference provider");

    let messages = Arc::new(vec![Message::user("Once upon a time")]);
    let mut request = CompletionRequest::new(messages);
    request.max_tokens = Some(12);
    request.temperature = Some(0.0);

    let response = provider.complete(request).await.expect("complete");
    let text = response.content.extract_text();
    assert!(
        !text.trim().is_empty(),
        "expected non-empty completion, got {text:?}"
    );
}
