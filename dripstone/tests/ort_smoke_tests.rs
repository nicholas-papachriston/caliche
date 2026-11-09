//! Ignored ONNX Runtime smoke tests.
//!
//! These tests intentionally require caller-provided models so the normal test
//! suite stays offline and lightweight.

#[cfg(feature = "embedding-ort")]
use dripstone::{
    Chunk, SearchResult,
    rag::rerank::{CrossEncoderConfig, CrossEncoderReranker},
};

#[cfg(feature = "inference-ort")]
use dripstone::{
    inference::{
        GenerationParams, InferenceBackend, InferenceBackendType, InferenceConfig, OrtInference,
    },
    loader::ModelSource,
};

#[cfg(feature = "embedding-ort")]
#[test]
#[ignore = "requires DRIPSTONE_ORT_RERANK_MODEL pointing at an ONNX cross-encoder with tokenizer.json beside it"]
fn ort_cross_encoder_reranks_known_model() {
    let model_path = std::env::var("DRIPSTONE_ORT_RERANK_MODEL")
        .expect("DRIPSTONE_ORT_RERANK_MODEL must be set");
    let reranker = CrossEncoderReranker::new(CrossEncoderConfig::new(model_path))
        .expect("create ORT reranker");
    let result = SearchResult::new(
        Chunk::new(
            "doc",
            0,
            "Rust is a systems programming language.",
            vec![1.0, 0.0],
            0,
            39,
            1,
            serde_json::json!({}),
            None,
        ),
        0.1,
        0,
    );

    let reranked = reranker
        .rerank("What language is Rust?", vec![result])
        .expect("rerank result");

    assert_eq!(reranked.len(), 1);
    assert!(reranked[0].score.is_finite());
}

#[cfg(feature = "inference-ort")]
#[tokio::test]
#[ignore = "requires DRIPSTONE_ORT_INFERENCE_MODEL pointing at a local ONNX model path or HuggingFace ONNX repo"]
async fn ort_inference_generates_with_known_model() {
    let source = std::env::var("DRIPSTONE_ORT_INFERENCE_MODEL")
        .expect("DRIPSTONE_ORT_INFERENCE_MODEL must be set");
    let config = InferenceConfig::new(ModelSource::from(source), InferenceBackendType::Ort);
    let backend = OrtInference::new(config)
        .await
        .expect("create ORT inference");
    let params = GenerationParams::new().with_max_tokens(4);

    let generated = backend
        .generate("The capital of France is", &params)
        .await
        .expect("generate text");

    assert!(!generated.is_empty());
}
