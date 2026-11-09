#![cfg(feature = "qdrant-grpc-backend")]

mod common;

use dripstone::backend::qdrant::{QdrantBackend, QdrantGrpcBackendOptions};
use dripstone::core::{Chunk, Metric, SearchQuery};
use dripstone::traits::Backend;
use std::sync::Arc;

#[tokio::test]
async fn test_qdrant_grpc_basic_operations() {
    let grpc_url =
        std::env::var("QDRANT_GRPC_URL").unwrap_or_else(|_| "http://localhost:6334".to_string());

    let mut options = QdrantGrpcBackendOptions::new(grpc_url, 3);
    options.api_key = common::qdrant_api_key();
    let backend: Arc<dyn Backend> = Arc::new(
        QdrantBackend::with_options(options)
            .await
            .expect("failed to create qdrant grpc backend"),
    );

    // Clean up test namespace
    let _ = backend.delete_namespace("test_grpc").await;

    // Upsert a chunk
    let chunk = Chunk::new(
        "test_1",
        0,
        "",
        Arc::from(vec![1.0, 0.0, 0.0].into_boxed_slice()),
        0,
        0,
        1,
        serde_json::json!({}),
        Some("test_grpc".to_string()),
    );

    backend
        .upsert(&[chunk.clone()])
        .await
        .expect("upsert failed");

    // Get the chunk back (Chunk::new uses composite id doc_id#chunk_index)
    let retrieved = backend
        .get("test_1#0", Some("test_grpc"))
        .await
        .expect("get failed")
        .expect("chunk not found");

    assert_eq!(retrieved.id, "test_1#0");
    assert_eq!(
        retrieved.vector,
        Arc::from(vec![1.0, 0.0, 0.0].into_boxed_slice())
    );

    // Search
    let query_vec: Arc<[f32]> = Arc::from(vec![1.0, 0.0, 0.0].into_boxed_slice());
    let query = SearchQuery::new(query_vec)
        .metric(Metric::Cosine)
        .limit(5)
        .namespace("test_grpc");
    let results = backend.search(&query).await.expect("search failed");

    assert_eq!(results.len(), 1);
    assert_eq!(results[0].chunk.id, "test_1#0");

    // Clean up
    backend
        .delete_namespace("test_grpc")
        .await
        .expect("flush failed");
}

#[tokio::test]
async fn test_qdrant_grpc_bulk_operations() {
    let grpc_url =
        std::env::var("QDRANT_GRPC_URL").unwrap_or_else(|_| "http://localhost:6334".to_string());

    let mut options = QdrantGrpcBackendOptions::new(grpc_url, 3);
    options.api_key = common::qdrant_api_key();
    let backend: Arc<dyn Backend> = Arc::new(
        QdrantBackend::with_options(options)
            .await
            .expect("failed to create qdrant grpc backend"),
    );

    // Clean up test namespace
    let _ = backend.delete_namespace("test_grpc_bulk").await;

    // Bulk upsert
    let chunk1 = Chunk::new(
        "bulk_1",
        0,
        "",
        Arc::from(vec![1.0, 0.0, 0.0].into_boxed_slice()),
        0,
        0,
        1,
        serde_json::json!({}),
        Some("test_grpc_bulk".to_string()),
    );
    let chunk2 = Chunk::new(
        "bulk_2",
        0,
        "",
        Arc::from(vec![0.0, 1.0, 0.0].into_boxed_slice()),
        0,
        0,
        1,
        serde_json::json!({}),
        Some("test_grpc_bulk".to_string()),
    );
    let chunk3 = Chunk::new(
        "bulk_3",
        0,
        "",
        Arc::from(vec![0.0, 0.0, 1.0].into_boxed_slice()),
        0,
        0,
        1,
        serde_json::json!({}),
        Some("test_grpc_bulk".to_string()),
    );
    let chunks = vec![chunk1, chunk2, chunk3];

    backend.upsert(&chunks).await.expect("bulk upsert failed");

    // Count
    let count = backend
        .count(Some("test_grpc_bulk"))
        .await
        .expect("count failed");

    assert_eq!(count, 3);

    // List
    let all_chunks = backend
        .list(Some("test_grpc_bulk"), None)
        .await
        .expect("list failed");

    assert_eq!(all_chunks.len(), 3);

    // Clean up
    backend
        .delete_namespace("test_grpc_bulk")
        .await
        .expect("flush failed");
}
