//! Tests for HNSW indexing functionality

use dripstone::core::Chunk;
use dripstone::embedding::{EmbeddingRecord, SimilarityMetric};
use dripstone::index::{HnswConfig, HnswIndex, VectorIndex};
use serde_json::json;

fn create_test_chunks(count: usize, dimension: usize) -> Vec<Chunk> {
    (0..count)
        .map(|i| {
            let vector: Vec<f32> = (0..dimension)
                .map(|j| ((i * dimension + j) as f32 / 1000.0).sin())
                .collect();
            Chunk::new(
                format!("doc_{i}"),
                0,
                format!("Content for chunk {i}"),
                vector,
                0,
                20,
                1,
                json!({}),
                None,
            )
        })
        .collect()
}

fn chunk_to_embedding_record(chunk: Chunk) -> EmbeddingRecord {
    let mut record = EmbeddingRecord::new(&chunk.id, chunk.vector.clone());
    if let Some(namespace) = &chunk.namespace {
        record = record.with_namespace(namespace);
    }
    record.metadata = Some(chunk.metadata);
    record
}

#[tokio::test]
async fn test_hnsw_build_and_search() {
    let config = HnswConfig {
        m: 16,
        ef_construction: 50,
        ef_search: 20,
        max_nb_connection: 16,
        ..Default::default()
    };
    let index = HnswIndex::new(config, 128);

    let records = create_test_chunks(100, 128);
    let embedding_records: Vec<EmbeddingRecord> =
        records.into_iter().map(chunk_to_embedding_record).collect();
    index.build(&embedding_records).await.unwrap();

    // Use spawn_blocking for blocking operations
    let index_arc = std::sync::Arc::new(index);
    let len = tokio::task::spawn_blocking({
        let index = std::sync::Arc::clone(&index_arc);
        move || index.len()
    })
    .await
    .unwrap();
    assert_eq!(len, 100);

    // Unwrap to use for search
    let index = match std::sync::Arc::try_unwrap(index_arc) {
        Ok(idx) => idx,
        Err(_) => panic!("Failed to unwrap Arc"),
    };
    let query: Vec<f32> = (0..128).map(|j| (j as f32 / 1000.0).sin()).collect();
    let query_vec = std::sync::Arc::from(query.into_boxed_slice());
    let results = index
        .search(&query_vec, 10, SimilarityMetric::Euclidean)
        .await
        .unwrap();

    assert!(!results.is_empty());
    assert!(results.len() <= 10);

    // Verify results are sorted by score
    for i in 1..results.len() {
        assert!(results[i - 1].score <= results[i].score);
    }
}

#[tokio::test]
async fn test_hnsw_incremental_add() {
    let config = HnswConfig::default();
    let index = HnswIndex::new(config, 128);

    let records = create_test_chunks(50, 128);
    let embedding_records: Vec<EmbeddingRecord> =
        records.into_iter().map(chunk_to_embedding_record).collect();
    index.build(&embedding_records).await.unwrap();

    // Use spawn_blocking for blocking operations
    let index_arc = std::sync::Arc::new(index);
    let len = tokio::task::spawn_blocking({
        let index = std::sync::Arc::clone(&index_arc);
        move || index.len()
    })
    .await
    .unwrap();
    assert_eq!(len, 50);

    let index = match std::sync::Arc::try_unwrap(index_arc) {
        Ok(idx) => idx,
        Err(_) => panic!("Failed to unwrap Arc"),
    };
    let new_chunk = Chunk::new(
        "new_record".to_string(),
        0,
        "New content".to_string(),
        (0..128)
            .map(|j| (j as f32 / 1000.0).sin())
            .collect::<Vec<f32>>(),
        0,
        11,
        1,
        json!({}),
        None,
    );
    index
        .add(chunk_to_embedding_record(new_chunk))
        .await
        .unwrap();

    let index_arc = std::sync::Arc::new(index);
    let len = tokio::task::spawn_blocking({
        let index = std::sync::Arc::clone(&index_arc);
        move || index.len()
    })
    .await
    .unwrap();
    assert_eq!(len, 51);
}

#[tokio::test]
async fn test_hnsw_remove() {
    let config = HnswConfig::default();
    let index = HnswIndex::new(config, 128);

    let records = create_test_chunks(50, 128);
    let embedding_records: Vec<EmbeddingRecord> =
        records.into_iter().map(chunk_to_embedding_record).collect();
    index.build(&embedding_records).await.unwrap();

    index.remove("doc_0#0").await.unwrap();

    // Use spawn_blocking for blocking operations
    let index_arc = std::sync::Arc::new(index);
    let len = tokio::task::spawn_blocking({
        let index = std::sync::Arc::clone(&index_arc);
        move || index.len()
    })
    .await
    .unwrap();
    assert_eq!(len, 49);
}

#[tokio::test(flavor = "multi_thread")]
async fn test_hnsw_serialization() {
    let config = HnswConfig {
        m: 16,
        ef_construction: 50,
        ef_search: 20,
        max_nb_connection: 16,
        ..Default::default()
    };
    let index = HnswIndex::new(config.clone(), 128);

    let records = create_test_chunks(50, 128);
    let embedding_records: Vec<EmbeddingRecord> =
        records.into_iter().map(chunk_to_embedding_record).collect();
    index.build(&embedding_records).await.unwrap();

    // Use spawn_blocking for blocking operations in to_bytes
    let index_arc = std::sync::Arc::new(index);
    let bytes = tokio::task::spawn_blocking({
        let index = std::sync::Arc::clone(&index_arc);
        move || index.to_bytes()
    })
    .await
    .unwrap()
    .unwrap();
    assert!(!bytes.is_empty());

    let _index = match std::sync::Arc::try_unwrap(index_arc) {
        Ok(idx) => idx,
        Err(_) => panic!("Failed to unwrap Arc"),
    };

    // Use spawn_blocking for from_bytes which uses blocking operations
    let bytes_clone = bytes.clone();
    let restored = tokio::task::spawn_blocking(move || HnswIndex::from_bytes(&bytes_clone))
        .await
        .unwrap()
        .unwrap();

    // Use spawn_blocking for blocking operations
    let restored_arc = std::sync::Arc::new(restored);
    let len = tokio::task::spawn_blocking({
        let index = std::sync::Arc::clone(&restored_arc);
        move || index.len()
    })
    .await
    .unwrap();
    assert_eq!(len, 50);

    // Use spawn_blocking for stats() which uses blocking operations
    let stats = tokio::task::spawn_blocking({
        let index = std::sync::Arc::clone(&restored_arc);
        move || index.stats()
    })
    .await
    .unwrap();
    assert_eq!(stats.num_records, 50);
    assert_eq!(stats.dimension, 128);
}

#[tokio::test]
async fn test_hnsw_different_metrics() {
    let config = HnswConfig {
        ef_search: 20,
        ..Default::default()
    };
    let index = HnswIndex::new(config, 128);

    let records = create_test_chunks(100, 128);
    let embedding_records: Vec<EmbeddingRecord> =
        records.into_iter().map(chunk_to_embedding_record).collect();
    index.build(&embedding_records).await.unwrap();

    let query: Vec<f32> = (0..128).map(|j| (j as f32 / 1000.0).sin()).collect();
    let query_vec = std::sync::Arc::from(query.into_boxed_slice());

    // Test cosine similarity
    let cosine_results = index
        .search(&query_vec, 5, SimilarityMetric::Cosine)
        .await
        .unwrap();
    assert!(!cosine_results.is_empty());

    // Test Euclidean distance
    let euclidean_results = index
        .search(&query_vec, 5, SimilarityMetric::Euclidean)
        .await
        .unwrap();
    assert!(!euclidean_results.is_empty());

    // Test dot product
    let dot_results = index
        .search(&query_vec, 5, SimilarityMetric::Dot)
        .await
        .unwrap();
    assert!(!dot_results.is_empty());
}

#[tokio::test(flavor = "multi_thread")]
async fn test_hnsw_clear() {
    let config = HnswConfig::default();
    let index = HnswIndex::new(config, 128);

    let records = create_test_chunks(50, 128);
    let embedding_records: Vec<EmbeddingRecord> =
        records.into_iter().map(chunk_to_embedding_record).collect();
    index.build(&embedding_records).await.unwrap();

    // Use spawn_blocking for blocking operations
    let index_arc = std::sync::Arc::new(index);
    let len = tokio::task::spawn_blocking({
        let index = std::sync::Arc::clone(&index_arc);
        move || index.len()
    })
    .await
    .unwrap();
    assert_eq!(len, 50);

    let index = match std::sync::Arc::try_unwrap(index_arc) {
        Ok(idx) => idx,
        Err(_) => panic!("Failed to unwrap Arc"),
    };

    // Use spawn_blocking for clear() which uses blocking operations
    let index = tokio::task::spawn_blocking(move || {
        index.clear();
        index
    })
    .await
    .unwrap();

    let index_arc = std::sync::Arc::new(index);
    let len = tokio::task::spawn_blocking({
        let index = std::sync::Arc::clone(&index_arc);
        move || index.len()
    })
    .await
    .unwrap();
    assert_eq!(len, 0);
}

#[tokio::test]
async fn test_hnsw_large_dataset() {
    let config = HnswConfig {
        m: 16,
        ef_construction: 100,
        ef_search: 50,
        max_nb_connection: 16,
        ..Default::default()
    };
    let index = HnswIndex::new(config, 128);

    let records = create_test_chunks(1000, 128);
    let embedding_records: Vec<EmbeddingRecord> =
        records.into_iter().map(chunk_to_embedding_record).collect();
    index.build(&embedding_records).await.unwrap();

    // Use spawn_blocking for blocking operations
    let index_arc = std::sync::Arc::new(index);
    let len = tokio::task::spawn_blocking({
        let index = std::sync::Arc::clone(&index_arc);
        move || index.len()
    })
    .await
    .unwrap();
    assert_eq!(len, 1000);

    // Unwrap to use for search
    let index = match std::sync::Arc::try_unwrap(index_arc) {
        Ok(idx) => idx,
        Err(_) => panic!("Failed to unwrap Arc"),
    };
    let query: Vec<f32> = (0..128).map(|j| (j as f32 / 1000.0).sin()).collect();
    let query_vec = std::sync::Arc::from(query.into_boxed_slice());
    let results = index
        .search(&query_vec, 10, SimilarityMetric::Euclidean)
        .await
        .unwrap();

    assert!(!results.is_empty());
    assert!(results.len() <= 10);
}
