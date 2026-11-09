//! Integration tests for hybrid search functionality.

#![cfg(feature = "memory-backend")]

#[cfg(feature = "memory-backend")]
use dripstone::MemoryBackend;
use dripstone::{Chunk, Metric, SearchQuery, SparseVector, traits::Backend};
use serde_json::json;
use std::sync::Arc;

#[tokio::test]
async fn test_hybrid_search_basic() {
    #[cfg(feature = "memory-backend")]
    {
        let backend: Arc<dyn Backend> = Arc::new(MemoryBackend::new());

        // Insert chunks with sparse vector info in metadata
        let mut chunks = Vec::new();
        for i in 0..10 {
            let mut sparse = SparseVector::new();
            sparse.set(i, 1.0);
            sparse.set((i + 1) % 10, 0.5);

            // Store sparse vector info in metadata
            let sparse_json = serde_json::to_value(&sparse).unwrap();
            let chunk = Chunk::new(
                format!("doc:{i}"),
                0,
                "",
                vec![i as f32 / 10.0, 0.5, 0.3],
                0,
                0,
                1,
                json!({
                    "sparse_vector": sparse_json
                }),
                Some("test".to_string()),
            );
            chunks.push(chunk);
        }
        backend.upsert(&chunks).await.unwrap();

        // Perform hybrid search
        let query_dense = vec![0.5, 0.5, 0.3];
        let mut query_sparse = SparseVector::new();
        query_sparse.set(5, 1.0);

        let mut query = SearchQuery::new(query_dense)
            .metric(Metric::Cosine)
            .limit(5)
            .namespace("test");
        query.sparse_vector = Some(query_sparse);
        query.hybrid_weight = Some(0.5);

        let results = backend.search(&query).await.unwrap();

        assert!(!results.is_empty());
        assert!(results.len() <= 5);
    }
}

#[tokio::test]
async fn test_hybrid_search_dense_only() {
    #[cfg(feature = "memory-backend")]
    {
        let backend: Arc<dyn Backend> = Arc::new(MemoryBackend::new());

        let chunk = Chunk::new("doc1", 0, "", vec![1.0, 0.0, 0.0], 0, 0, 1, json!({}), None);
        backend.upsert(&[chunk]).await.unwrap();

        let query = SearchQuery::new(vec![1.0, 0.0, 0.0])
            .metric(Metric::Cosine)
            .limit(5);
        let results = backend.search(&query).await.unwrap();

        assert_eq!(results.len(), 1);
    }
}

#[tokio::test]
async fn test_hybrid_search_with_weights() {
    #[cfg(feature = "memory-backend")]
    {
        let backend: Arc<dyn Backend> = Arc::new(MemoryBackend::new());

        let mut sparse = SparseVector::new();
        sparse.set(0, 1.0);
        let sparse_json = serde_json::to_value(&sparse).unwrap();

        let chunk = Chunk::new(
            "doc1",
            0,
            "",
            vec![1.0, 0.0, 0.0],
            0,
            0,
            1,
            json!({
                "sparse_vector": sparse_json
            }),
            None,
        );
        backend.upsert(&[chunk]).await.unwrap();

        // Test with different weight distributions
        let mut query_sparse = SparseVector::new();
        query_sparse.set(0, 1.0);
        let mut query = SearchQuery::new(vec![1.0, 0.0, 0.0])
            .metric(Metric::Cosine)
            .limit(5);
        query.sparse_vector = Some(query_sparse);
        query.hybrid_weight = Some(0.9); // Favor dense

        let results1 = backend.search(&query).await.unwrap();

        assert!(!results1.is_empty());
    }
}

#[tokio::test]
async fn test_hybrid_search_with_metadata_filter() {
    #[cfg(feature = "memory-backend")]
    {
        use dripstone::search::Filter;
        let backend: Arc<dyn Backend> = Arc::new(MemoryBackend::new());

        let mut chunks = Vec::new();
        for i in 0..5 {
            let chunk = Chunk::new(
                format!("doc:{i}"),
                0,
                "",
                vec![(i as f32).max(1e-6), 1e-6, 1e-6],
                0,
                0,
                1,
                json!({"category": if i % 2 == 0 { "even" } else { "odd" }}),
                None,
            );
            chunks.push(chunk);
        }
        backend.upsert(&chunks).await.unwrap();

        let query = SearchQuery::new(vec![2.0, 1e-6, 1e-6])
            .metric(Metric::Cosine)
            .limit(10)
            .filter(Filter::Equal {
                field: "category".to_string(),
                value: json!("even"),
            });
        let results = backend.search(&query).await.unwrap();

        // Should only return even docs
        assert!(results.len() <= 3); // docs 0, 2, 4
        for result in results {
            assert_eq!(result.chunk.metadata["category"], "even");
        }
    }
}
