//! Tests for local embedding generation.

#[cfg(any(feature = "local-candle", feature = "local-ort"))]
use dripstone::VectorGenerator;
#[cfg(any(feature = "local-candle", feature = "local-ort"))]
use dripstone::generation::{EmbeddingConfig, PoolingStrategy};

#[cfg(feature = "local-candle")]
use dripstone::generation::CandleEmbedder;

#[cfg(feature = "local-ort")]
use dripstone::generation::OrtEmbedder;

#[cfg(feature = "local-candle")]
#[tokio::test]
async fn test_candle_embedder_config() {
    let config = EmbeddingConfig::default()
        .with_model_repo("sentence-transformers/all-MiniLM-L6-v2")
        .with_pooling(PoolingStrategy::Mean)
        .with_normalize(true);

    assert_eq!(config.model_repo, "sentence-transformers/all-MiniLM-L6-v2");
    assert_eq!(config.pooling, PoolingStrategy::Mean);
    assert!(config.normalize);
}

#[cfg(feature = "local-ort")]
#[tokio::test]
async fn test_ort_embedder_config() {
    let config = EmbeddingConfig::default()
        .with_model_repo("sentence-transformers/all-MiniLM-L6-v2")
        .with_pooling(PoolingStrategy::Mean)
        .with_normalize(true);

    assert_eq!(config.model_repo, "sentence-transformers/all-MiniLM-L6-v2");
    assert_eq!(config.pooling, PoolingStrategy::Mean);
    assert!(config.normalize);
}

#[cfg(feature = "local-candle")]
#[tokio::test]
#[ignore = "requires downloading models"]
async fn test_candle_embedder_generation() {
    let config =
        EmbeddingConfig::default().with_model_repo("sentence-transformers/all-MiniLM-L6-v2");

    let embedder = CandleEmbedder::new(config).expect("Failed to create embedder");

    let texts = vec!["Hello, world!".to_string(), "This is a test.".to_string()];

    let embeddings = embedder
        .generate(&texts)
        .await
        .expect("Failed to generate embeddings");

    assert_eq!(embeddings.len(), 2);
    assert_eq!(embeddings[0].len(), embedder.dimension());
    assert_eq!(embeddings[1].len(), embedder.dimension());
}

#[cfg(feature = "local-ort")]
#[tokio::test]
#[ignore = "requires downloading models"]
async fn test_ort_embedder_generation() {
    let config =
        EmbeddingConfig::default().with_model_repo("sentence-transformers/all-MiniLM-L6-v2");

    let embedder = OrtEmbedder::new(config).expect("Failed to create embedder");

    let texts = vec!["Hello, world!".to_string(), "This is a test.".to_string()];

    let embeddings = embedder
        .generate(&texts)
        .await
        .expect("Failed to generate embeddings");

    assert_eq!(embeddings.len(), 2);
    assert_eq!(embeddings[0].len(), embedder.dimension());
    assert_eq!(embeddings[1].len(), embedder.dimension());
}
