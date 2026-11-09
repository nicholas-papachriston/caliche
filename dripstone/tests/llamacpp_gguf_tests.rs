//! End-to-end GGUF / llama.cpp smoke tests.
//!
//! Downloads the tiny Stories15M GGUF (~19 MiB) when missing, then runs a short
//! generation. Requires network + curl on first run.

#![cfg(feature = "inference-llamacpp")]

use dripstone::inference::{
    DeviceConfig, GenerationParams, InferenceBackend, InferenceBackendType, InferenceConfig,
    LlamaCppInference,
};
use dripstone::loader::{
    DownloadOptions, EnsureModelOptions, ModelSource, builtin_catalog, ensure_model, find_model,
};
use std::path::PathBuf;

fn models_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("target/llamacpp-test-models")
}

fn ensure_stories15m() -> PathBuf {
    let spec = find_model("stories15m-gguf").expect("catalog entry");
    let dest = models_dir().join(spec.id);
    let options = EnsureModelOptions {
        download: DownloadOptions::default()
            .with_max_bytes(Some(25 * 1024 * 1024))
            .with_reserve_free_bytes(32 * 1024 * 1024),
        prefer_hub_cache: false,
    };
    ensure_model(spec, &dest, &options, None).expect("ensure stories15m gguf");
    let artifact = spec.artifacts.first().expect("artifact");
    dest.join(artifact.remote_path)
}

#[tokio::test]
async fn stories15m_gguf_generates_tokens() {
    let gguf_path = ensure_stories15m();
    assert!(gguf_path.is_file(), "missing {}", gguf_path.display());

    let config = InferenceConfig::new(
        ModelSource::local(&gguf_path),
        InferenceBackendType::LlamaCpp,
    )
    .with_device(DeviceConfig::Cpu)
    .with_default_params(
        GenerationParams::default()
            .with_max_tokens(16)
            .with_temperature(0.0),
    );

    let backend = LlamaCppInference::new(config)
        .await
        .expect("load llama.cpp backend");
    assert_eq!(backend.model_info().architecture.as_deref(), Some("gguf"));

    let params = GenerationParams::default()
        .with_max_tokens(16)
        .with_temperature(0.0);
    let text = backend
        .generate("Once upon a time", &params)
        .await
        .expect("generate");
    assert!(
        !text.trim().is_empty(),
        "expected non-empty generation, got {text:?}"
    );

    // Second call exercises a fresh per-request context on the worker thread.
    let text2 = backend
        .generate("The little cat", &params)
        .await
        .expect("generate again");
    assert!(
        !text2.trim().is_empty(),
        "expected non-empty second generation, got {text2:?}"
    );
    assert!(!backend.supports_kv_cache());
}

#[test]
fn catalog_includes_gguf_smoke_model() {
    assert!(
        builtin_catalog()
            .iter()
            .any(|spec| spec.id == "stories15m-gguf")
    );
}
