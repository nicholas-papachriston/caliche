//! Integration tests for local embedding providers.

#[cfg(feature = "embedding-candle")]
use caliche::llm::LocalCandleProvider;

#[cfg(feature = "embedding-ort")]
use caliche::llm::LocalOrtProvider;

#[cfg(feature = "embedding-candle")]
#[tokio::test]
#[ignore = "requires a local embedding model download"]
async fn test_local_candle_provider() {
    use caliche::llm::{EmbeddingProvider, EmbeddingRequest};

    let provider = LocalCandleProvider::new().expect("Failed to create provider");

    let request = EmbeddingRequest {
        texts: vec!["Hello, world!".to_string(), "This is a test.".to_string()],
        model: None,
    };

    let response = provider
        .embed(request)
        .await
        .expect("Failed to generate embeddings");

    assert_eq!(response.embeddings.len(), 2);
    assert!(!response.embeddings[0].is_empty());
    assert!(!response.embeddings[1].is_empty());
    assert_eq!(response.embeddings[0].len(), response.embeddings[1].len());
}

#[cfg(feature = "embedding-ort")]
#[tokio::test]
#[ignore = "requires a local embedding model download"]
async fn test_local_ort_provider() {
    use caliche::llm::{EmbeddingProvider, EmbeddingRequest};

    let provider = LocalOrtProvider::new().expect("Failed to create provider");

    let request = EmbeddingRequest {
        texts: vec!["Hello, world!".to_string(), "This is a test.".to_string()],
        model: None,
    };

    let response = provider
        .embed(request)
        .await
        .expect("Failed to generate embeddings");

    assert_eq!(response.embeddings.len(), 2);
    assert!(!response.embeddings[0].is_empty());
    assert!(!response.embeddings[1].is_empty());
    assert_eq!(response.embeddings[0].len(), response.embeddings[1].len());
}
