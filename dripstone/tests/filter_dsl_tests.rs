//! Tests for advanced metadata filtering DSL

#![cfg(feature = "memory-backend")]

use dripstone::backend::memory::MemoryBackend;
use dripstone::core::{Chunk, Metric, SearchQuery};
use dripstone::search::{Filter, FilterBuilder};
use dripstone::traits::Backend;
use serde_json::json;
use std::sync::Arc;

#[tokio::test]
async fn test_filter_equal() {
    let filter = Filter::Equal {
        field: "category".to_string(),
        value: json!("A"),
    };

    let metadata1 = json!({"category": "A"});
    assert!(filter.matches(&metadata1));

    let metadata2 = json!({"category": "B"});
    assert!(!filter.matches(&metadata2));
}

#[tokio::test]
async fn test_filter_and() {
    let filter = Filter::And {
        left: Box::new(Filter::Equal {
            field: "category".to_string(),
            value: json!("A"),
        }),
        right: Box::new(Filter::GreaterThan {
            field: "price".to_string(),
            value: json!(100),
        }),
    };

    let metadata1 = json!({"category": "A", "price": 150});
    assert!(filter.matches(&metadata1));

    let metadata2 = json!({"category": "A", "price": 50});
    assert!(!filter.matches(&metadata2));

    let metadata3 = json!({"category": "B", "price": 150});
    assert!(!filter.matches(&metadata3));
}

#[tokio::test]
async fn test_filter_or() {
    let filter = Filter::Or {
        left: Box::new(Filter::Equal {
            field: "category".to_string(),
            value: json!("A"),
        }),
        right: Box::new(Filter::Equal {
            field: "category".to_string(),
            value: json!("B"),
        }),
    };

    let metadata1 = json!({"category": "A"});
    assert!(filter.matches(&metadata1));

    let metadata2 = json!({"category": "B"});
    assert!(filter.matches(&metadata2));

    let metadata3 = json!({"category": "C"});
    assert!(!filter.matches(&metadata3));
}

#[tokio::test]
async fn test_filter_not() {
    let filter = Filter::Not {
        filter: Box::new(Filter::Equal {
            field: "category".to_string(),
            value: json!("A"),
        }),
    };

    let metadata1 = json!({"category": "A"});
    assert!(!filter.matches(&metadata1));

    let metadata2 = json!({"category": "B"});
    assert!(filter.matches(&metadata2));
}

#[tokio::test]
async fn test_filter_in() {
    let filter = Filter::In {
        field: "category".to_string(),
        values: vec![json!("A"), json!("B")],
    };

    let metadata1 = json!({"category": "A"});
    assert!(filter.matches(&metadata1));

    let metadata2 = json!({"category": "B"});
    assert!(filter.matches(&metadata2));

    let metadata3 = json!({"category": "C"});
    assert!(!filter.matches(&metadata3));
}

#[tokio::test]
async fn test_filter_builder() {
    let filter = FilterBuilder::new()
        .equal("category", json!("A"))
        .and(Filter::GreaterThan {
            field: "price".to_string(),
            value: json!(100),
        })
        .build()
        .unwrap();

    let metadata = json!({"category": "A", "price": 150});
    assert!(filter.matches(&metadata));
}

#[tokio::test]
async fn test_filter_nested_fields() {
    let filter = Filter::Equal {
        field: "user.name".to_string(),
        value: json!("John"),
    };

    let metadata = json!({"user": {"name": "John"}});
    assert!(filter.matches(&metadata));

    let metadata2 = json!({"user": {"name": "Jane"}});
    assert!(!filter.matches(&metadata2));
}

#[tokio::test]
async fn test_filter_integration_with_backend() {
    let backend: Arc<dyn Backend> = Arc::new(MemoryBackend::new());

    // Insert chunks with metadata
    let mut chunks = Vec::new();
    for i in 0..10 {
        let chunk = Chunk::new(
            format!("doc:{i}"),
            0,
            format!("Content {i}"),
            (0..128)
                .map(|j| (i + j) as f32 / 1000.0)
                .collect::<Vec<f32>>(),
            0,
            12,
            1,
            json!({
                "category": if i % 2 == 0 { "A" } else { "B" },
                "price": i * 10,
                "active": i < 5
            }),
            None,
        );
        chunks.push(chunk);
    }
    backend.upsert(&chunks).await.unwrap();

    // Test filter with backend
    let filter = FilterBuilder::new()
        .equal("category", json!("A"))
        .and(Filter::GreaterThan {
            field: "price".to_string(),
            value: json!(20),
        })
        .build()
        .unwrap();

    let query_vector: Vec<f32> = (0..128).map(|j| j as f32 / 1000.0).collect();
    let query = SearchQuery {
        vector: query_vector.clone().into(),
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
    let results = backend.search(&query).await.unwrap();

    // Should only return chunks with category="A" and price > 20
    assert!(!results.is_empty());
    for result in &results {
        assert_eq!(result.chunk.metadata["category"], "A");
        assert!(result.chunk.metadata["price"].as_u64().unwrap() > 20);
    }
}

#[tokio::test]
async fn test_filter_complex_query() {
    // Test: (category == "A" OR category == "B") AND price > 20
    let filter = FilterBuilder::new()
        .in_values("category", vec![json!("A"), json!("B")])
        .and(Filter::GreaterThan {
            field: "price".to_string(),
            value: json!(20),
        })
        .build()
        .unwrap();

    let metadata1 = json!({"category": "A", "price": 30});
    assert!(filter.matches(&metadata1));

    let metadata2 = json!({"category": "B", "price": 30});
    assert!(filter.matches(&metadata2));

    let metadata3 = json!({"category": "A", "price": 10});
    assert!(!filter.matches(&metadata3));

    let metadata4 = json!({"category": "C", "price": 30});
    assert!(!filter.matches(&metadata4));
}
