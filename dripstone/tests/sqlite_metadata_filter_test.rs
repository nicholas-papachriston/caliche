//! `SQLite` backend: metadata filters on the in-memory search path (`LoadAll`).
//!
//! Full DB round-trips use rkyv for `Chunk` rows (metadata as JSON text in the archived layout);
//! filter semantics for `search_in_db` are covered by unit tests in `backend/sqlite/mod.rs`.

use dripstone::backend::sqlite::{MemoryStrategy, SqliteBackend, SqliteBackendConfig};
use dripstone::core::{Chunk, Metric, SearchQuery};
use dripstone::search::Filter;
use dripstone::traits::Backend;
use serde_json::json;
use tempfile::NamedTempFile;

fn sample_chunk(doc_id: &str, vec_seed: f32, category: &str) -> Chunk {
    let v: Vec<f32> = (0..8).map(|i| (i as f32 + vec_seed) / 100.0).collect();
    Chunk::new(
        doc_id,
        0,
        format!("content {doc_id}"),
        v,
        0,
        10,
        1,
        json!({ "category": category, "tier": "pro" }),
        None,
    )
}

#[tokio::test]
async fn sqlite_load_all_cache_search_respects_metadata_filter() {
    let file = NamedTempFile::new().expect("temp db");
    let path = file.path().to_path_buf();
    let config = SqliteBackendConfig {
        db_path: path,
        memory_strategy: MemoryStrategy::LoadAll,
        max_connections: 2,
        table_name: "vectors".to_string(),
        ensure_schema: true,
    };
    let backend = SqliteBackend::new(config).await.expect("backend");

    let chunks = vec![sample_chunk("x", 3.0, "A"), sample_chunk("y", 4.0, "B")];
    backend.upsert(&chunks).await.expect("upsert");

    let filter = Filter::And {
        left: Box::new(Filter::Equal {
            field: "category".to_string(),
            value: json!("A"),
        }),
        right: Box::new(Filter::Equal {
            field: "tier".to_string(),
            value: json!("pro"),
        }),
    };
    let qv: Vec<f32> = (0..8).map(|i| i as f32 / 100.0).collect();
    let query = SearchQuery {
        vector: qv.into(),
        limit: 10,
        metric: Metric::Cosine,
        namespace: None,
        filter: Some(filter),
        hybrid_weight: None,
        sparse_vector: None,
        include_content: true,
        include_vectors: false,
        query_multi_vectors: None,
        maxsim: None,
    };

    let results = backend.search(&query).await.expect("search");
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].chunk.doc_id, "x");
}

#[tokio::test]
async fn sqlite_load_all_search_without_filter_returns_all() {
    let file = NamedTempFile::new().expect("temp db");
    let config = SqliteBackendConfig {
        db_path: file.path().to_path_buf(),
        memory_strategy: MemoryStrategy::LoadAll,
        max_connections: 2,
        table_name: "vectors".to_string(),
        ensure_schema: true,
    };
    let backend = SqliteBackend::new(config).await.expect("backend");

    let chunks = vec![sample_chunk("p", 1.0, "x"), sample_chunk("q", 2.0, "y")];
    backend.upsert(&chunks).await.expect("upsert");

    let qv: Vec<f32> = (0..8).map(|i| i as f32 / 100.0).collect();
    let query = SearchQuery {
        vector: qv.into(),
        limit: 10,
        metric: Metric::Cosine,
        namespace: None,
        filter: None,
        hybrid_weight: None,
        sparse_vector: None,
        include_content: true,
        include_vectors: false,
        query_multi_vectors: None,
        maxsim: None,
    };

    let results = backend.search(&query).await.expect("search");
    assert_eq!(results.len(), 2);
}
