#[path = "common/mod.rs"]
mod common;

macro_rules! outputln {
    ($($arg:tt)*) => {{
        let mut output = std::io::stdout().lock();
        let _ = std::io::Write::write_fmt(&mut output, format_args!($($arg)*));
        let _ = std::io::Write::write_all(&mut output, b"\n");
    }};
}

use common::*;
use dripstone::DripstoneError;
use dripstone::core::{Chunk, Metric, SearchQuery};
use dripstone::search::{Filter, FilterBuilder};
use dripstone::traits::Backend;
use serde_json::json;
use std::env;
use std::sync::Arc;
use tokio::task::JoinSet;

/// Run the full integration test suite on a backend
pub async fn run_integration_suite<B>(
    backend: B,
    namespace: String,
    dimension: usize,
) -> Result<(), DripstoneError>
where
    B: Backend + 'static,
{
    let backend: Arc<dyn Backend> = Arc::new(backend);

    outputln!("Running integration test: cross_namespace_isolation");
    test_cross_namespace_isolation(Arc::clone(&backend), &namespace, dimension).await?;

    outputln!("Running integration test: concurrent_upserts");
    test_concurrent_upserts(Arc::clone(&backend), &namespace, dimension).await?;

    outputln!("Running integration test: concurrent_searches");
    test_concurrent_searches(Arc::clone(&backend), &namespace, dimension).await?;

    outputln!("Running integration test: pagination_consistency");
    test_pagination_consistency(Arc::clone(&backend), &namespace, dimension).await?;

    outputln!("Running integration test: edge_case_vectors");
    test_edge_case_vectors(Arc::clone(&backend), &namespace, dimension).await?;

    outputln!("Running integration test: special_characters_in_ids");
    test_special_characters_in_ids(Arc::clone(&backend), &namespace, dimension).await?;

    outputln!("Running integration test: large_metadata");
    test_large_metadata(Arc::clone(&backend), &namespace, dimension).await?;

    outputln!("Running integration test: update_existing_records");
    test_update_existing_records(Arc::clone(&backend), &namespace, dimension).await?;

    outputln!("Running integration test: delete_during_search");
    test_delete_during_search(Arc::clone(&backend), &namespace, dimension).await?;

    outputln!("Running integration test: namespace_cleanup");
    test_namespace_cleanup(Arc::clone(&backend), &namespace, dimension).await?;

    // Note: Pipeline tests removed - pipeline API is not part of Backend trait
    // These would need to be implemented at Dripstone level if needed

    outputln!("Running integration test: advanced_filter_dsl");
    test_advanced_filter_dsl(Arc::clone(&backend), &namespace, dimension).await?;

    outputln!("All integration tests passed!");
    Ok(())
}

async fn test_cross_namespace_isolation(
    backend: Arc<dyn Backend>,
    namespace: &str,
    dimension: usize,
) -> Result<(), DripstoneError> {
    let ns1 = format!("{namespace}-ns1");
    let ns2 = format!("{namespace}-ns2");

    backend.delete_namespace(&ns1).await?;
    backend.delete_namespace(&ns2).await?;

    let chunk1 = common::build_chunk("iso-1".to_string(), &ns1, 1, dimension);
    backend.upsert(&[chunk1]).await?;

    let chunk2 = build_chunk("iso-1".to_string(), &ns2, 2, dimension);
    backend.upsert(&[chunk2]).await?;

    let from_ns1 = backend
        .get("iso-1#0", Some(&ns1))
        .await?
        .ok_or_else(|| DripstoneError::NotFound("iso-1 in ns1".to_string()))?;
    let from_ns2 = backend
        .get("iso-1#0", Some(&ns2))
        .await?
        .ok_or_else(|| DripstoneError::NotFound("iso-1 in ns2".to_string()))?;

    assert_ne!(
        from_ns1.vector, from_ns2.vector,
        "Vectors should differ across namespaces"
    );
    assert_eq!(backend.count(Some(&ns1)).await?, 1);
    assert_eq!(backend.count(Some(&ns2)).await?, 1);

    backend.delete_namespace(&ns1).await?;
    backend.delete_namespace(&ns2).await?;

    Ok(())
}

async fn test_concurrent_upserts(
    backend: Arc<dyn Backend>,
    namespace: &str,
    dimension: usize,
) -> Result<(), DripstoneError> {
    let ns = format!("{namespace}-concurrent");
    backend.delete_namespace(&ns).await?;

    let concurrency = env::var("INTEGRATION_CONCURRENCY")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(50);

    let chunks = build_concurrent_chunks(concurrency, &ns, dimension, 1000);

    let mut tasks = JoinSet::new();
    for chunk in chunks {
        let backend_clone = Arc::clone(&backend);
        tasks.spawn(async move { backend_clone.upsert(&[chunk]).await });
    }

    while let Some(result) = tasks.join_next().await {
        result.map_err(|e| DripstoneError::storage(format!("Join error: {e}")))??;
    }

    let count = backend.count(Some(&ns)).await?;
    assert_eq!(
        count, concurrency,
        "All concurrent chunks should be inserted"
    );

    backend.delete_namespace(&ns).await?;
    Ok(())
}

async fn test_concurrent_searches(
    backend: Arc<dyn Backend>,
    namespace: &str,
    dimension: usize,
) -> Result<(), DripstoneError> {
    let ns = format!("{namespace}-search-concurrent");
    backend.delete_namespace(&ns).await?;

    let initial_chunks = build_concurrent_chunks(20, &ns, dimension, 2000);
    backend.upsert(&initial_chunks).await?;

    let query_vector = Arc::from(build_vector(2010, dimension).into_boxed_slice());
    let query = SearchQuery {
        vector: query_vector,
        limit: 5,
        metric: Metric::Cosine,
        namespace: Some(ns.clone()),
        filter: None,
        hybrid_weight: None,
        sparse_vector: None,
        include_content: true,
        include_vectors: false,
        query_multi_vectors: None,
        maxsim: None,
    };

    let mut tasks = JoinSet::new();

    for i in 0..10 {
        let backend_clone = Arc::clone(&backend);
        let query_clone = query.clone();
        tasks.spawn(async move {
            backend_clone
                .search(&query_clone)
                .await
                .map(|results| (i, results))
        });
    }

    let mut search_count = 0;
    while let Some(result) = tasks.join_next().await {
        let (_, results) =
            result.map_err(|e| DripstoneError::storage(format!("Join error: {e}")))??;
        assert!(!results.is_empty(), "Search should return results");
        search_count += 1;
    }

    assert_eq!(search_count, 10, "All searches should complete");

    backend.delete_namespace(&ns).await?;
    Ok(())
}

async fn test_pagination_consistency(
    backend: Arc<dyn Backend>,
    namespace: &str,
    dimension: usize,
) -> Result<(), DripstoneError> {
    let ns = format!("{namespace}-pagination");
    backend.delete_namespace(&ns).await?;

    let chunks = build_concurrent_chunks(50, &ns, dimension, 3000);
    backend.upsert(&chunks).await?;

    let all_chunks = backend.list(Some(&ns), None).await?;
    assert_eq!(all_chunks.len(), 50, "Should have 50 total chunks");

    // Use limit parameter for pagination
    let page1 = backend.list(Some(&ns), Some(20)).await?;
    let remaining = backend.list(Some(&ns), None).await?;
    let page2_start = page1.len();
    let page2: Vec<_> = remaining
        .iter()
        .skip(page2_start)
        .take(20)
        .cloned()
        .collect();
    let page3_start = page2_start + page2.len();
    let page3: Vec<_> = remaining
        .iter()
        .skip(page3_start)
        .take(20)
        .cloned()
        .collect();

    assert_eq!(page1.len(), 20, "Page 1 should have 20 chunks");
    assert_eq!(page2.len(), 20, "Page 2 should have 20 chunks");
    assert_eq!(page3.len(), 10, "Page 3 should have 10 chunks");

    let mut all_ids: Vec<String> = page1.iter().map(|c| c.id.clone()).collect();
    all_ids.extend(page2.iter().map(|c| c.id.clone()));
    all_ids.extend(page3.iter().map(|c| c.id.clone()));
    all_ids.sort();
    all_ids.dedup();
    assert_eq!(all_ids.len(), 50, "Should have 50 unique IDs across pages");

    backend.delete_namespace(&ns).await?;
    Ok(())
}

async fn test_edge_case_vectors(
    backend: Arc<dyn Backend>,
    namespace: &str,
    dimension: usize,
) -> Result<(), DripstoneError> {
    let ns = format!("{namespace}-edge-vectors");
    backend.delete_namespace(&ns).await?;

    let edge_cases = build_edge_case_vectors();

    for (id, vector) in edge_cases {
        if vector.len() != dimension {
            continue;
        }

        let chunk = Chunk::new(
            id.clone(),
            0,
            format!("Content for {id}"),
            vector.clone(),
            0,
            20,
            1,
            json!({}),
            Some(ns.clone()),
        );

        backend.upsert(&[chunk.clone()]).await?;

        let retrieved = backend
            .get(&format!("{id}#0"), Some(&ns))
            .await?
            .ok_or_else(|| DripstoneError::NotFound(id.clone()))?;

        assert_eq!(
            retrieved.vector.len(),
            vector.len(),
            "Vector length should match for {id}"
        );
        assert_chunks_equivalent(
            &retrieved,
            &Chunk::new(
                id.clone(),
                0,
                format!("Content for {id}"),
                vector.clone(),
                0,
                20,
                1,
                json!({}),
                Some(ns.clone()),
            ),
            1e-6,
        );
    }

    backend.delete_namespace(&ns).await?;
    Ok(())
}

async fn test_special_characters_in_ids(
    backend: Arc<dyn Backend>,
    namespace: &str,
    dimension: usize,
) -> Result<(), DripstoneError> {
    let ns = format!("{namespace}-special-ids");
    backend.delete_namespace(&ns).await?;

    let special_ids = build_special_ids();

    for id in special_ids {
        let chunk = build_chunk(id.clone(), &ns, 4000, dimension);

        match backend.upsert(&[chunk.clone()]).await {
            Ok(()) => {
                let retrieved = backend.get(&format!("{id}#0"), Some(&ns)).await?;
                assert!(
                    retrieved.is_some(),
                    "Should be able to retrieve chunk with ID: {id}"
                );
            }
            Err(_) => continue,
        }
    }

    backend.delete_namespace(&ns).await?;
    Ok(())
}

async fn test_large_metadata(
    backend: Arc<dyn Backend>,
    namespace: &str,
    dimension: usize,
) -> Result<(), DripstoneError> {
    let ns = format!("{namespace}-large-metadata");
    backend.delete_namespace(&ns).await?;

    let metadata_cases = build_special_metadata();

    for (id, metadata) in metadata_cases {
        let chunk = Chunk::new(
            id.clone(),
            0,
            format!("Content for {id}"),
            build_vector(5000, dimension),
            0,
            20,
            1,
            metadata.clone(),
            Some(ns.clone()),
        );

        backend.upsert(&[chunk.clone()]).await?;

        let retrieved = backend
            .get(&format!("{id}#0"), Some(&ns))
            .await?
            .ok_or_else(|| DripstoneError::NotFound(id.clone()))?;

        assert_eq!(
            retrieved.metadata, metadata,
            "Metadata should match for {id}"
        );
    }

    backend.delete_namespace(&ns).await?;
    Ok(())
}

async fn test_update_existing_records(
    backend: Arc<dyn Backend>,
    namespace: &str,
    dimension: usize,
) -> Result<(), DripstoneError> {
    let ns = format!("{namespace}-update");
    backend.delete_namespace(&ns).await?;

    let chunk1 = build_chunk("update-test".to_string(), &ns, 6000, dimension);
    backend.upsert(&[chunk1.clone()]).await?;

    let retrieved1 = backend
        .get("update-test#0", Some(&ns))
        .await?
        .ok_or_else(|| DripstoneError::NotFound("update-test".to_string()))?;

    let chunk2 = build_chunk("update-test".to_string(), &ns, 6001, dimension);
    backend.upsert(&[chunk2.clone()]).await?;

    let retrieved2 = backend
        .get("update-test#0", Some(&ns))
        .await?
        .ok_or_else(|| DripstoneError::NotFound("update-test".to_string()))?;

    assert_ne!(
        retrieved1.vector, retrieved2.vector,
        "Vector should be updated"
    );
    assert_eq!(
        backend.count(Some(&ns)).await?,
        1,
        "Count should remain 1 after update"
    );

    backend.delete_namespace(&ns).await?;
    Ok(())
}

async fn test_delete_during_search(
    backend: Arc<dyn Backend>,
    namespace: &str,
    dimension: usize,
) -> Result<(), DripstoneError> {
    let ns = format!("{namespace}-delete-search");
    backend.delete_namespace(&ns).await?;

    let chunks = build_concurrent_chunks(30, &ns, dimension, 7000);
    backend.upsert(&chunks).await?;

    let query_vector = Arc::from(build_vector(7015, dimension).into_boxed_slice());
    let query = SearchQuery {
        vector: query_vector,
        limit: 10,
        metric: Metric::Cosine,
        namespace: Some(ns.clone()),
        filter: None,
        hybrid_weight: None,
        sparse_vector: None,
        include_content: true,
        include_vectors: false,
        query_multi_vectors: None,
        maxsim: None,
    };

    let results_before = backend.search(&query).await?;
    assert!(
        !results_before.is_empty(),
        "Should have results before delete"
    );

    for i in 0..10 {
        backend
            .delete(&format!("concurrent-{i}#0"), Some(&ns))
            .await?;
    }

    let results_after = backend.search(&query).await?;
    assert!(
        !results_after.is_empty(),
        "Should still have results after delete"
    );

    let count = backend.count(Some(&ns)).await?;
    assert_eq!(count, 20, "Count should be 20 after deleting 10 chunks");

    backend.delete_namespace(&ns).await?;
    Ok(())
}

async fn test_namespace_cleanup(
    backend: Arc<dyn Backend>,
    namespace: &str,
    dimension: usize,
) -> Result<(), DripstoneError> {
    let ns = format!("{namespace}-cleanup");

    let chunks = build_concurrent_chunks(100, &ns, dimension, 8000);
    backend.upsert(&chunks).await?;

    let count_before = backend.count(Some(&ns)).await?;
    assert_eq!(count_before, 100, "Should have 100 chunks before cleanup");

    backend.delete_namespace(&ns).await?;

    let count_after = backend.count(Some(&ns)).await?;
    assert_eq!(count_after, 0, "Should have 0 chunks after cleanup");

    let retrieved = backend.get("concurrent-0#0", Some(&ns)).await?;
    assert!(
        retrieved.is_none(),
        "Should not be able to retrieve deleted chunks"
    );

    Ok(())
}

// Pipeline tests removed - pipeline API is not part of Backend trait
// These would need to be implemented at Dripstone level if needed

// Pipeline tests removed - pipeline API is not part of Backend trait

// Pipeline tests removed - pipeline API is not part of Backend trait

#[expect(
    clippy::unwrap_used,
    reason = "test fixtures are well-formed; panicking on malformed data is desired"
)]
async fn test_advanced_filter_dsl(
    backend: Arc<dyn Backend>,
    namespace: &str,
    dimension: usize,
) -> Result<(), DripstoneError> {
    let ns = format!("{namespace}-filter");
    backend.delete_namespace(&ns).await?;

    // Insert chunks with various metadata
    let mut chunks = Vec::new();
    for i in 0..20 {
        let mut chunk = build_chunk(format!("filter-{i}"), &ns, i, dimension);
        chunk.metadata = json!({
            "category": if i % 2 == 0 { "A" } else { "B" },
            "price": i * 10,
            "active": i < 10,
            "rating": if i < 5 { 5 } else if i < 15 { 4 } else { 3 }
        });
        chunks.push(chunk);
    }
    backend.upsert(&chunks).await?;

    // Test simple equality filter
    let filter1 = Filter::Equal {
        field: "category".to_string(),
        value: serde_json::json!("A"),
    };

    let query_vector: Arc<[f32]> = Arc::from(build_vector(10, dimension).into_boxed_slice());
    let query = SearchQuery {
        vector: query_vector.clone(),
        limit: 20,
        metric: Metric::Cosine,
        namespace: Some(ns.clone()),
        filter: Some(filter1),
        hybrid_weight: None,
        sparse_vector: None,
        include_content: true,
        include_vectors: false,
        query_multi_vectors: None,
        maxsim: None,
    };
    let results = backend.search(&query).await?;

    assert_eq!(results.len(), 10); // Half should be category A
    for result in &results {
        assert_eq!(result.chunk.metadata["category"], "A");
    }

    // Test complex AND filter
    let filter2 = FilterBuilder::new()
        .equal("category", serde_json::json!("A"))
        .and(Filter::GreaterThan {
            field: "price".to_string(),
            value: serde_json::json!(50),
        })
        .build()
        .unwrap();

    let query = SearchQuery {
        vector: query_vector.clone(),
        limit: 20,
        metric: Metric::Cosine,
        namespace: Some(ns.clone()),
        filter: Some(filter2),
        hybrid_weight: None,
        sparse_vector: None,
        include_content: true,
        include_vectors: false,
        query_multi_vectors: None,
        maxsim: None,
    };
    let results = backend.search(&query).await?;

    assert!(!results.is_empty());
    for result in &results {
        assert_eq!(result.chunk.metadata["category"], "A");
        assert!(result.chunk.metadata["price"].as_u64().unwrap() > 50);
    }

    // Test OR filter
    let filter3 = FilterBuilder::new()
        .in_values(
            "category",
            vec![serde_json::json!("A"), serde_json::json!("B")],
        )
        .build()
        .unwrap();

    let query = SearchQuery {
        vector: query_vector.clone(),
        limit: 20,
        metric: Metric::Cosine,
        namespace: Some(ns.clone()),
        filter: Some(filter3),
        hybrid_weight: None,
        sparse_vector: None,
        include_content: true,
        include_vectors: false,
        query_multi_vectors: None,
        maxsim: None,
    };
    let results = backend.search(&query).await?;

    assert_eq!(results.len(), 20); // All chunks

    // Test range filter
    let filter4 = FilterBuilder::new()
        .greater_than_or_equal("price", serde_json::json!(50))
        .and(Filter::LessThanOrEqual {
            field: "price".to_string(),
            value: serde_json::json!(150),
        })
        .build()
        .unwrap();

    let query = SearchQuery {
        vector: query_vector,
        limit: 20,
        metric: Metric::Cosine,
        namespace: Some(ns.clone()),
        filter: Some(filter4),
        hybrid_weight: None,
        sparse_vector: None,
        include_content: true,
        include_vectors: false,
        query_multi_vectors: None,
        maxsim: None,
    };
    let results = backend.search(&query).await?;

    assert!(!results.is_empty());
    for result in &results {
        let price = result.chunk.metadata["price"].as_u64().unwrap();
        assert!((50..=150).contains(&price));
    }

    backend.delete_namespace(&ns).await?;
    Ok(())
}
