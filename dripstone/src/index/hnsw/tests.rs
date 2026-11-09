use super::*;
use crate::search::{Filter, compile_native_filter_plan};
use serde_json::json;
use std::sync::Arc;

fn create_test_records(count: usize, dimension: usize) -> Vec<EmbeddingRecord> {
    (0..count)
        .map(|i| {
            let vector: Vec<f32> = (0..dimension)
                .map(|_| fastrand::f32().mul_add(2.0, -1.0))
                .collect();
            EmbeddingRecord::new(format!("record_{i}"), vector)
        })
        .collect()
}

#[tokio::test(flavor = "multi_thread")]
async fn test_hnsw_build() {
    let config = HnswConfig::default();
    let index = HnswIndex::new(config, 128);

    let records = create_test_records(100, 128);
    index
        .build(&records)
        .await
        .expect("Failed to build HNSW index");

    // Use spawn_blocking for blocking operations
    let index_arc = Arc::new(index);
    let len = tokio::task::spawn_blocking({
        let index = Arc::clone(&index_arc);
        move || index.len()
    })
    .await
    .expect("Failed to get index length");
    assert_eq!(len, 100);
}

#[tokio::test]
async fn test_hnsw_search() {
    let config = HnswConfig {
        ef_search: 20,
        ..Default::default()
    };
    let index = HnswIndex::new(config, 128);

    let records = create_test_records(100, 128);
    index
        .build(&records)
        .await
        .expect("Failed to build HNSW index");

    let query: Vec<f32> = (0..128)
        .map(|_| fastrand::f32().mul_add(2.0, -1.0))
        .collect();
    let results = index
        .search(&query, 10, SimilarityMetric::Cosine)
        .await
        .expect("Failed to search HNSW index");

    assert!(!results.is_empty());
    assert!(results.len() <= 10);
}

#[tokio::test]
async fn test_hnsw_native_filter_search_request() {
    let config = HnswConfig {
        ef_search: 32,
        rerank_overfetch: 4,
        ..Default::default()
    };
    let index = HnswIndex::new(config, 4);

    let records = vec![
        EmbeddingRecord::new("match-a", vec![1.0, 0.0, 0.0, 0.0])
            .with_namespace("forge")
            .with_metadata(json!({"domain": "docs"})),
        EmbeddingRecord::new("match-b", vec![0.99, 0.01, 0.0, 0.0])
            .with_namespace("forge")
            .with_metadata(json!({"domain": "docs"})),
        EmbeddingRecord::new("other-a", vec![0.98, 0.02, 0.0, 0.0])
            .with_namespace("forge")
            .with_metadata(json!({"domain": "other"})),
        EmbeddingRecord::new("other-b", vec![0.97, 0.03, 0.0, 0.0])
            .with_namespace("forge")
            .with_metadata(json!({"domain": "other"})),
    ];
    index
        .build(&records)
        .await
        .expect("Failed to build HNSW index");

    let filter = Filter::Equal {
        field: "domain".to_string(),
        value: json!("docs"),
    };
    let native_filter_plan = compile_native_filter_plan(Some("forge"), Some(&filter));
    let results = index
        .search_request(IndexSearchRequest {
            query: &[1.0, 0.0, 0.0, 0.0],
            limit: 4,
            metric: SimilarityMetric::Cosine,
            native_filter: Some(&native_filter_plan),
        })
        .await
        .expect("Failed to search filtered HNSW index");

    assert_eq!(results.len(), 2);
    assert!(results.iter().all(|result| {
        result.record.namespace.as_deref() == Some("forge")
            && result
                .record
                .metadata
                .as_ref()
                .and_then(|m| m.get("domain"))
                == Some(&json!("docs"))
    }));
}

#[tokio::test]
async fn test_hnsw_disjunctive_native_filter_search_request() {
    let config = HnswConfig {
        ef_search: 32,
        rerank_overfetch: 4,
        ..Default::default()
    };
    let index = HnswIndex::new(config, 4);

    let records = vec![
        EmbeddingRecord::new("match-a", vec![1.0, 0.0, 0.0, 0.0])
            .with_namespace("forge")
            .with_metadata(json!({"domain": "docs"})),
        EmbeddingRecord::new("match-b", vec![0.99, 0.01, 0.0, 0.0])
            .with_namespace("forge")
            .with_metadata(json!({"domain": "code"})),
        EmbeddingRecord::new("other-a", vec![0.98, 0.02, 0.0, 0.0])
            .with_namespace("forge")
            .with_metadata(json!({"domain": "other"})),
    ];
    index
        .build(&records)
        .await
        .expect("Failed to build HNSW index");

    let filter = Filter::Or {
        left: Box::new(Filter::Equal {
            field: "domain".to_string(),
            value: json!("docs"),
        }),
        right: Box::new(Filter::Equal {
            field: "domain".to_string(),
            value: json!("code"),
        }),
    };
    let native_filter_plan = compile_native_filter_plan(Some("forge"), Some(&filter));
    let results = index
        .search_request(IndexSearchRequest {
            query: &[1.0, 0.0, 0.0, 0.0],
            limit: 4,
            metric: SimilarityMetric::Cosine,
            native_filter: Some(&native_filter_plan),
        })
        .await
        .expect("Failed to search filtered HNSW index");

    assert_eq!(results.len(), 2);
    assert!(results.iter().all(|result| {
        result.record.namespace.as_deref() == Some("forge")
            && matches!(
                result.record.metadata.as_ref().and_then(|m| m.get("domain")),
                Some(value) if value == &json!("docs") || value == &json!("code")
            )
    }));
}

#[tokio::test(flavor = "multi_thread")]
async fn test_hnsw_add_remove() {
    let config = HnswConfig::default();
    let index = HnswIndex::new(config, 128);

    let records = create_test_records(50, 128);
    index
        .build(&records)
        .await
        .expect("Failed to build HNSW index");

    let new_record = EmbeddingRecord::new(
        "new_record",
        (0..128).map(|_| fastrand::f32()).collect::<Vec<_>>(),
    );
    index
        .add(new_record)
        .await
        .expect("Failed to add record to HNSW index");

    // Use spawn_blocking for blocking operations
    let index_arc = Arc::new(index);
    let len = tokio::task::spawn_blocking({
        let index = Arc::clone(&index_arc);
        move || index.len()
    })
    .await
    .expect("Failed to get index length");
    assert_eq!(len, 51);

    let Ok(index) = Arc::try_unwrap(index_arc) else {
        panic!("Failed to unwrap Arc");
    };
    index
        .remove("new_record")
        .await
        .expect("Failed to remove record from HNSW index");

    let index_arc = Arc::new(index);
    let len = tokio::task::spawn_blocking({
        let index = Arc::clone(&index_arc);
        move || index.len()
    })
    .await
    .expect("Failed to get index length");
    assert_eq!(len, 50);
}
