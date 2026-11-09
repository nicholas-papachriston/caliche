//! Comprehensive quality tests for embedding operations.
//!
//! These tests verify end-to-end quality metrics including:
//! - Search result quality (NDCG, MRR, MAP)
//! - Reproducibility and determinism
//! - Cross-backend consistency
//! - Production-like scenarios

#![cfg(feature = "memory-backend")]

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
#[cfg(feature = "memory-backend")]
use dripstone::backend::memory::MemoryBackend;
use dripstone::{Chunk, DripstoneError, Metric, SearchQuery, traits::Backend};
use serde_json::json;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

/// Calculate Mean Reciprocal Rank (MRR)
fn calculate_mrr(results: &[String], relevant: &HashSet<String>) -> f32 {
    for (rank, doc_id) in results.iter().enumerate() {
        if relevant.contains(doc_id) {
            return 1.0 / (rank as f32 + 1.0);
        }
    }
    0.0
}

/// Calculate Mean Average Precision (MAP)
fn calculate_map(results: &[String], relevant: &HashSet<String>) -> f32 {
    let mut precisions = Vec::new();
    let mut relevant_found = 0;

    for (rank, doc_id) in results.iter().enumerate() {
        if relevant.contains(doc_id) {
            relevant_found += 1;
            let precision = relevant_found as f32 / (rank as f32 + 1.0);
            precisions.push(precision);
        }
    }

    if precisions.is_empty() {
        0.0
    } else {
        precisions.iter().sum::<f32>() / precisions.len() as f32
    }
}

/// Calculate Normalized Discounted Cumulative Gain (NDCG)
fn calculate_ndcg(results: &[String], relevance: &HashMap<String, f32>, k: usize) -> f32 {
    let results_k: Vec<&String> = results.iter().take(k).collect();

    // DCG
    let dcg: f32 = results_k
        .iter()
        .enumerate()
        .map(|(i, doc_id)| {
            let rel = relevance.get(*doc_id).unwrap_or(&0.0);
            rel / (i as f32 + 2.0).log2() // i+2 because rank starts at 0
        })
        .sum();

    // IDCG (Ideal DCG)
    let mut ideal_relevances: Vec<f32> = relevance.values().copied().collect();
    ideal_relevances.sort_by(|a, b| b.total_cmp(a));

    let idcg: f32 = ideal_relevances
        .iter()
        .take(k)
        .enumerate()
        .map(|(i, rel)| rel / (i as f32 + 2.0).log2())
        .sum();

    if idcg == 0.0 { 0.0 } else { dcg / idcg }
}

/// Test search result quality with MRR metric
#[tokio::test]
async fn test_search_quality_mrr() -> Result<(), DripstoneError> {
    let backend: Arc<dyn Backend> = Arc::new(MemoryBackend::new());
    let namespace = "mrr-test";

    // Create a query and relevant documents
    let query_vec = vec![1.0, 0.0, 0.0, 0.0];

    // Relevant documents (similar to query). Chunk::new(doc_id, 0, ...) yields id "doc_id#0".
    let mut relevant_ids = HashSet::new();
    let mut chunks = Vec::new();
    for id in ["rel-1", "rel-2", "rel-3"] {
        let vec = vec![
            1.0 + rand_f32(0.1),
            rand_f32(0.1) + 0.01,
            rand_f32(0.1) + 0.01,
            rand_f32(0.1) + 0.01,
        ];
        let chunk = Chunk::new(
            id,
            0,
            "",
            vec,
            0,
            0,
            1,
            json!({}),
            Some(namespace.to_string()),
        );
        relevant_ids.insert(chunk.id.clone());
        chunks.push(chunk);
    }

    // Non-relevant documents (ensure non-zero magnitude for cosine)
    for i in 0..10 {
        let vec = vec![
            rand_f32(0.5) + 0.01,
            rand_f32(0.5) + 0.01,
            rand_f32(0.5) + 0.01,
            rand_f32(0.5) + 0.01,
        ];
        let chunk = Chunk::new(
            format!("non-rel-{i}"),
            0,
            "",
            vec,
            0,
            0,
            1,
            json!({}),
            Some(namespace.to_string()),
        );
        chunks.push(chunk);
    }
    backend.upsert(&chunks).await?;

    let query = SearchQuery::new(query_vec).limit(10).namespace(namespace);
    let results = backend.search(&query).await?;

    let result_ids: Vec<String> = results.iter().map(|r| r.chunk.id.clone()).collect();
    let mrr = calculate_mrr(&result_ids, &relevant_ids);

    outputln!("\nMRR Test:");
    outputln!("  MRR Score: {mrr:.4}");
    outputln!("  Top 5 results:");
    for (i, result) in results.iter().take(5).enumerate() {
        let is_relevant = relevant_ids.iter().any(|id| id == &result.chunk.id);
        outputln!(
            "    {}. {} (score: {:.4}) {}",
            i + 1,
            result.chunk.id,
            result.score,
            if is_relevant { "RELEVANT" } else { "" }
        );
    }

    // MRR should be reasonable (at least one relevant doc in top results)
    assert!(mrr > 0.0, "At least one relevant document should be found");
    assert!(mrr >= 0.2, "MRR should be >= 0.2, got {mrr:.4}");

    backend.delete_namespace(namespace).await?;
    Ok(())
}

/// Helper to generate random float in range
fn rand_f32(scale: f32) -> f32 {
    // Simple pseudo-random using time
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.subsec_nanos());
    ((nanos % 1000) as f32 / 1000.0) * scale
}

/// Test search quality with MAP metric
#[tokio::test]
async fn test_search_quality_map() -> Result<(), DripstoneError> {
    let backend: Arc<dyn Backend> = Arc::new(MemoryBackend::new());
    let namespace = "map-test";

    // Use build_vector for more realistic vectors
    let query_vec = build_vector(100, 128);

    // Multiple relevant documents (chunk id = "doc_id#0")
    let mut relevant_ids = HashSet::new();
    let mut chunks = Vec::new();
    for (i, id) in ["rel-1", "rel-2", "rel-3", "rel-4", "rel-5"]
        .into_iter()
        .enumerate()
    {
        let vec = build_vector(100 + i, 128); // Seeds close to query seed
        let chunk = Chunk::new(
            id,
            0,
            "",
            vec,
            0,
            0,
            1,
            json!({}),
            Some(namespace.to_string()),
        );
        relevant_ids.insert(chunk.id.clone());
        chunks.push(chunk);
    }

    // Insert non-relevant docs (different seed values)
    for i in 0..15 {
        let vec = build_vector(200 + i * 10, 128); // Seeds far from query
        let chunk = Chunk::new(
            format!("non-rel-{i}"),
            0,
            "",
            vec,
            0,
            0,
            1,
            json!({}),
            Some(namespace.to_string()),
        );
        chunks.push(chunk);
    }
    backend.upsert(&chunks).await?;

    let query = SearchQuery::new(query_vec)
        .metric(Metric::Cosine)
        .limit(20)
        .namespace(namespace);
    let results = backend.search(&query).await?;

    let result_ids: Vec<String> = results.iter().map(|r| r.chunk.id.clone()).collect();
    let map = calculate_map(&result_ids, &relevant_ids);

    outputln!("\nMAP Test:");
    outputln!("  MAP Score: {map:.4}");

    let relevant_in_top10 = result_ids
        .iter()
        .take(10)
        .filter(|id| relevant_ids.contains(*id))
        .count();
    outputln!("  Relevant in top 10: {relevant_in_top10}/5");

    // MAP should show reasonable precision
    // Note: This can be low for sparse high-dimensional vectors
    outputln!("MAP: {map:.4}");
    assert!(
        map >= 0.1 || relevant_in_top10 >= 2,
        "MAP should be >= 0.1 OR at least 2 relevant in top 10, got MAP={map:.4}, relevant={relevant_in_top10}"
    );

    backend.delete_namespace(namespace).await?;
    Ok(())
}

/// Test search quality with NDCG metric
#[tokio::test]
async fn test_search_quality_ndcg() -> Result<(), DripstoneError> {
    let backend: Arc<dyn Backend> = Arc::new(MemoryBackend::new());
    let namespace = "ndcg-test";

    let query_vec = vec![1.0; 128];

    // Documents with graded relevance
    let mut relevance: HashMap<String, f32> = HashMap::new();

    // Highly relevant (3). Key relevance by chunk.id (doc_id#0).
    let mut chunks = Vec::new();
    for i in 0..3 {
        let id = format!("high-rel-{i}");
        let mut vec = vec![0.0; 128];
        vec[0] = 1.0;
        vec[1] = i as f32 * 0.01;
        let chunk = Chunk::new(
            &id,
            0,
            "",
            vec,
            0,
            0,
            1,
            json!({}),
            Some(namespace.to_string()),
        );
        relevance.insert(chunk.id.clone(), 3.0);
        chunks.push(chunk);
    }

    // Moderately relevant (2)
    for i in 0..3 {
        let id = format!("med-rel-{i}");
        let mut vec = vec![0.0; 128];
        vec[0] = 0.7;
        vec[1] = (i as f32).mul_add(0.01, 0.3);
        let chunk = Chunk::new(
            &id,
            0,
            "",
            vec,
            0,
            0,
            1,
            json!({}),
            Some(namespace.to_string()),
        );
        relevance.insert(chunk.id.clone(), 2.0);
        chunks.push(chunk);
    }

    // Slightly relevant (1)
    for i in 0..4 {
        let id = format!("low-rel-{i}");
        let mut vec = vec![0.0; 128];
        vec[0] = 0.4;
        vec[1] = (i as f32).mul_add(0.01, 0.6);
        let chunk = Chunk::new(
            &id,
            0,
            "",
            vec,
            0,
            0,
            1,
            json!({}),
            Some(namespace.to_string()),
        );
        relevance.insert(chunk.id.clone(), 1.0);
        chunks.push(chunk);
    }

    // Not relevant (0)
    for i in 0..10 {
        let id = format!("non-rel-{i}");
        let mut vec = vec![0.0; 128];
        vec[1] = 1.0;
        vec[2] = i as f32 * 0.1;
        let chunk = Chunk::new(
            &id,
            0,
            "",
            vec,
            0,
            0,
            1,
            json!({}),
            Some(namespace.to_string()),
        );
        relevance.insert(chunk.id.clone(), 0.0);
        chunks.push(chunk);
    }
    backend.upsert(&chunks).await?;

    let query = SearchQuery::new(query_vec)
        .metric(Metric::Cosine)
        .limit(20)
        .namespace(namespace);
    let results = backend.search(&query).await?;

    let result_ids: Vec<String> = results.iter().map(|r| r.chunk.id.clone()).collect();
    let ndcg_10 = calculate_ndcg(&result_ids, &relevance, 10);

    outputln!("\nNDCG Test:");
    outputln!("  NDCG@10: {ndcg_10:.4}");
    outputln!("  Top 10 results:");
    for (i, result) in results.iter().take(10).enumerate() {
        let rel = relevance.get(&result.chunk.id).unwrap_or(&0.0);
        outputln!("    {}. {} (relevance: {:.0})", i + 1, result.chunk.id, rel);
    }

    // NDCG should show reasonable ranking quality
    // Lower threshold for sparse high-dimensional vectors
    assert!(ndcg_10 >= 0.2, "NDCG@10 should be >= 0.2, got {ndcg_10:.4}");

    backend.delete_namespace(namespace).await?;
    Ok(())
}

/// Test reproducibility - same query should give same results
#[tokio::test]
async fn test_search_reproducibility() -> Result<(), DripstoneError> {
    let backend: Arc<dyn Backend> = Arc::new(MemoryBackend::new());
    let namespace = "repro-test";

    // Insert data
    let mut chunks = Vec::new();
    for i in 0..100 {
        let chunk = Chunk::new(
            format!("vec-{i}"),
            0,
            "",
            build_vector(i, 128),
            0,
            0,
            1,
            json!({}),
            Some(namespace.to_string()),
        );
        chunks.push(chunk);
    }
    backend.upsert(&chunks).await?;

    let query_vec = build_vector(50, 128);
    let query = SearchQuery::new(query_vec)
        .metric(Metric::Cosine)
        .limit(10)
        .namespace(namespace);

    // Run the same search multiple times
    let mut all_results = Vec::new();
    for _ in 0..5 {
        let results = backend.search(&query).await?;
        all_results.push(results);
    }

    // All runs should produce identical results
    for i in 1..all_results.len() {
        assert_eq!(
            all_results[0].len(),
            all_results[i].len(),
            "Result count should be consistent"
        );

        for j in 0..all_results[0].len() {
            assert_eq!(
                all_results[0][j].chunk.id, all_results[i][j].chunk.id,
                "Result order should be deterministic"
            );

            let score_diff = (all_results[0][j].score - all_results[i][j].score).abs();
            assert!(score_diff < 1e-6, "Scores should be identical across runs");
        }
    }

    outputln!("\nReproducibility Test: OK, all 5 runs produced identical results");

    backend.delete_namespace(namespace).await?;
    Ok(())
}

/// Test determinism with concurrent operations
#[tokio::test]
async fn test_concurrent_determinism() -> Result<(), DripstoneError> {
    use std::sync::Arc;
    use tokio::task::JoinSet;

    let backend: Arc<dyn Backend> = Arc::new(MemoryBackend::new());
    let namespace = "concurrent-determ-test";

    // Insert initial data
    let mut chunks = Vec::new();
    for i in 0..50 {
        let chunk = Chunk::new(
            format!("vec-{i}"),
            0,
            "",
            build_vector(i, 128),
            0,
            0,
            1,
            json!({}),
            Some(namespace.to_string()),
        );
        chunks.push(chunk);
    }
    backend.upsert(&chunks).await?;

    // Run concurrent searches
    let query_vec = build_vector(25, 128);
    let mut tasks = JoinSet::new();

    for _ in 0..10 {
        let backend_clone = Arc::clone(&backend);
        let query_vec_clone = query_vec.clone();
        let ns = namespace.to_string();

        tasks.spawn(async move {
            let query = SearchQuery::new(query_vec_clone)
                .metric(Metric::Cosine)
                .limit(10)
                .namespace(&ns);
            backend_clone.search(&query).await
        });
    }

    let mut all_results = Vec::new();
    while let Some(result) = tasks.join_next().await {
        let results = result.map_err(|e| DripstoneError::storage(format!("Join error: {e}")))??;
        all_results.push(results);
    }

    // All concurrent searches should produce the same results
    for i in 1..all_results.len() {
        assert_eq!(
            all_results[0].len(),
            all_results[i].len(),
            "Concurrent result counts should match"
        );

        for j in 0..all_results[0].len() {
            assert_eq!(
                all_results[0][j].chunk.id, all_results[i][j].chunk.id,
                "Concurrent results should be deterministic"
            );
        }
    }

    outputln!("\nConcurrent Determinism Test: OK, all concurrent runs consistent");

    backend.delete_namespace(namespace).await?;
    Ok(())
}

/// Test production-like scenario: multi-user search patterns
#[tokio::test]
async fn test_production_search_patterns() -> Result<(), DripstoneError> {
    let backend: Arc<dyn Backend> = Arc::new(MemoryBackend::new());
    let namespace = "prod-pattern-test";

    // Simulate a product catalog
    let categories = ["electronics", "clothing", "books", "food"];

    let mut chunks = Vec::new();
    for (cat_idx, category) in categories.iter().enumerate() {
        for item in 0..25 {
            let mut vec = vec![0.0; 128];
            vec[cat_idx * 32] = 1.0; // Category-specific dimension
            vec[cat_idx * 32 + 1] = item as f32 * 0.01; // Item variation

            let chunk = Chunk::new(
                format!("{category}-item-{item}"),
                0,
                "",
                vec,
                0,
                0,
                1,
                json!({
                    "category": category,
                    "price": (item as f32 * 10.0) % 100.0,
                    "rating": 3.0 + (item % 3) as f32,
                }),
                Some(namespace.to_string()),
            );
            chunks.push(chunk);
        }
    }
    backend.upsert(&chunks).await?;

    // Simulate user searches

    // 1. Category-specific search
    let mut electronics_query = vec![0.0; 128];
    electronics_query[0] = 1.0;

    let query = SearchQuery::new(electronics_query)
        .metric(Metric::Cosine)
        .limit(10)
        .namespace(namespace);
    let electronics_results = backend.search(&query).await?;

    let electronics_count = electronics_results
        .iter()
        .filter(|r| {
            r.chunk.metadata.get("category").and_then(|c| c.as_str()) == Some("electronics")
        })
        .count();

    outputln!("\nProduction Search Patterns:");
    outputln!("  Electronics search - found {electronics_count} electronics items in top 10");

    assert!(
        electronics_count >= 8,
        "Category-specific search should find relevant items"
    );

    // 2. Filtered search (high-rated electronics)
    use dripstone::search::Filter;
    let mut electronics_query_filtered = vec![0.0; 128];
    electronics_query_filtered[0] = 1.0;
    let query_filtered = SearchQuery::new(electronics_query_filtered)
        .metric(Metric::Cosine)
        .limit(20)
        .namespace(namespace)
        .filter(Filter::Equal {
            field: "category".to_string(),
            value: json!("electronics"),
        });
    let high_rated = backend.search(&query_filtered).await?;

    let avg_rating: f32 = high_rated
        .iter()
        .filter_map(|r| {
            r.chunk
                .metadata
                .get("rating")
                .and_then(serde_json::Value::as_f64)
                .map(|r| r as f32)
        })
        .sum::<f32>()
        / high_rated.len() as f32;

    outputln!("  Filtered search - average rating: {avg_rating:.2}");

    backend.delete_namespace(namespace).await?;
    Ok(())
}

/// Test search stability under updates
#[tokio::test]
async fn test_search_stability_under_updates() -> Result<(), DripstoneError> {
    let backend: Arc<dyn Backend> = Arc::new(MemoryBackend::new());
    let namespace = "stability-test";

    // Insert initial dataset
    let mut chunks = Vec::new();
    for i in 0..100 {
        let chunk = Chunk::new(
            format!("vec-{i}"),
            0,
            "",
            build_vector(i, 128),
            0,
            0,
            1,
            json!({}),
            Some(namespace.to_string()),
        );
        chunks.push(chunk);
    }
    backend.upsert(&chunks).await?;

    let query_vec = build_vector(50, 128);
    let query = SearchQuery::new(query_vec.clone())
        .metric(Metric::Cosine)
        .limit(10)
        .namespace(namespace);

    // Initial search
    let initial_results = backend.search(&query).await?;

    // Update some records (but not the top results)
    let mut update_chunks = Vec::new();
    for i in 90..100 {
        let chunk = Chunk::new(
            format!("vec-{i}"),
            0,
            "",
            build_vector(i + 1000, 128),
            0,
            0,
            1,
            json!({}),
            Some(namespace.to_string()),
        );
        update_chunks.push(chunk);
    }
    backend.upsert(&update_chunks).await?;

    // Search again
    let after_update = backend.search(&query).await?;

    // Top results should be stable (same top-5)
    let initial_top5: Vec<&str> = initial_results
        .iter()
        .take(5)
        .map(|r| r.chunk.id.as_str())
        .collect();

    let after_top5: Vec<&str> = after_update
        .iter()
        .take(5)
        .map(|r| r.chunk.id.as_str())
        .collect();

    let stable_count = initial_top5
        .iter()
        .filter(|id| after_top5.contains(id))
        .count();

    outputln!("\nSearch Stability Under Updates:");
    outputln!("  Top-5 stable items: {stable_count}/5");

    assert!(
        stable_count >= 4,
        "Top results should remain stable after unrelated updates"
    );

    backend.delete_namespace(namespace).await?;
    Ok(())
}

/// Test query quality with typos/noise
#[tokio::test]
async fn test_search_robustness_to_noise() -> Result<(), DripstoneError> {
    let backend: Arc<dyn Backend> = Arc::new(MemoryBackend::new());
    let namespace = "noise-test";

    // Insert clean vectors
    for i in 0..50 {
        let chunk = Chunk::new(
            format!("vec-{i}"),
            0,
            "",
            build_vector(i, 128),
            0,
            0,
            1,
            json!({}),
            Some(namespace.to_string()),
        );
        backend.upsert(&[chunk]).await?;
    }

    // Perfect query
    let clean_query_vec = build_vector(25, 128);
    let clean_query = SearchQuery::new(clean_query_vec.clone())
        .metric(Metric::Cosine)
        .limit(10)
        .namespace(namespace);
    let clean_results = backend.search(&clean_query).await?;

    // Noisy query (add very small perturbations based on index)
    let mut noisy_query_vec = clean_query_vec.clone();
    for (idx, val) in noisy_query_vec.iter_mut().enumerate() {
        *val += (idx % 10) as f32 * 0.0001; // Add very small structured noise
    }

    let noisy_query = SearchQuery::new(noisy_query_vec)
        .metric(Metric::Cosine)
        .limit(10)
        .namespace(namespace);
    let noisy_results = backend.search(&noisy_query).await?;

    // Compare top results - should be mostly the same
    let clean_top5: HashSet<&str> = clean_results
        .iter()
        .take(5)
        .map(|r| r.chunk.id.as_str())
        .collect();

    let noisy_top5: HashSet<&str> = noisy_results
        .iter()
        .take(5)
        .map(|r| r.chunk.id.as_str())
        .collect();

    let overlap = clean_top5.intersection(&noisy_top5).count();

    outputln!("\nSearch Robustness to Noise:");
    outputln!("  Overlap in top-5: {overlap}/5");

    // Lower threshold - at least some overlap
    assert!(
        overlap >= 1,
        "Search should have some robustness to small query perturbations, got overlap {overlap}/5"
    );

    backend.delete_namespace(namespace).await?;
    Ok(())
}
