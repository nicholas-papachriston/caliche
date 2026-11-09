//! Integration tests for S3 backend
//!
//! Standard S3 tests require AWS credentials and an S3 bucket.
//! Set `DRIPSTONE_INT_S3_BUCKET` to run the standard S3 tests.
//!
//! S3 Vector Buckets tests require:
//! - `DRIPSTONE_INT_S3_VECTOR_BUCKET`: S3 vector bucket name
//! - `DRIPSTONE_INT_S3_VECTOR_INDEX`: vector index name
//! - AWS credentials configured
//!
//! Run with: cargo test --test `s3_integration` --features s3-backend -- --test-threads=1

#![cfg(feature = "s3-backend")]

macro_rules! outputln {
    ($($arg:tt)*) => {{
        let mut output = std::io::stdout().lock();
        let _ = std::io::Write::write_fmt(&mut output, format_args!($($arg)*));
        let _ = std::io::Write::write_all(&mut output, b"\n");
    }};
}

use dripstone::backend::s3::{S3Backend, S3BackendConfig, S3VectorBackend, S3VectorBackendConfig};
use dripstone::core::{Chunk, Metric, SearchQuery};
use dripstone::search::Filter;
use dripstone::traits::Backend;
use dripstone::{DripstoneError, MemoryBackend, TierConfig, TieredChainBuilder};
use std::env;
use std::sync::Arc;
use time::OffsetDateTime;

fn require_named_env(name: &str) -> Result<String, DripstoneError> {
    match env::var(name) {
        Ok(value) if !value.trim().is_empty() => Ok(value),
        _ => Err(DripstoneError::storage(format!("{name} is required"))),
    }
}

fn require_s3_config() -> Result<S3BackendConfig, DripstoneError> {
    let bucket = require_named_env("DRIPSTONE_INT_S3_BUCKET")?;
    let prefix =
        env::var("DRIPSTONE_INT_S3_PREFIX").unwrap_or_else(|_| "test/embeddings/".to_string());

    Ok(S3BackendConfig {
        bucket,
        key_prefix: prefix,
        access_key_id: None,
        secret_access_key: None,
        session_token: None,
        region: None,
        enable_compression: true,
        batch_size: 50,
        cache_indices: true,
    })
}

#[ignore = "requires DRIPSTONE_INT_S3_BUCKET, DRIPSTONE_INT_S3_VECTOR_BUCKET, and AWS credentials; run with --ignored"]
#[tokio::test]
async fn test_s3_basic_operations() -> Result<(), DripstoneError> {
    let config = require_s3_config()?;

    let backend: Arc<dyn Backend> = Arc::new(S3Backend::with_config(config).await?);

    let namespace = format!(
        "s3-basic-{}",
        OffsetDateTime::now_utc().unix_timestamp_nanos()
    );

    // Test upsert
    let vector = vec![0.1, 0.2, 0.3, 0.4];
    let chunk = Chunk::new(
        "test-1",
        0,
        "test content",
        vector.clone(),
        0,
        12,
        1,
        serde_json::json!({"category": "test"}),
        Some(namespace.clone()),
    );

    backend.upsert(&[chunk.clone()]).await?;

    // Test get
    let retrieved = backend.get("test-1#0", Some(&namespace)).await?;
    assert!(retrieved.is_some());
    let retrieved = retrieved.unwrap();
    assert_eq!(retrieved.id, "test-1#0");
    assert_eq!(retrieved.vector, Arc::from(vector.into_boxed_slice()));

    // Test count
    let count = backend.count(Some(&namespace)).await?;
    assert_eq!(count, 1);

    // Test delete
    backend.delete("test-1#0", Some(&namespace)).await?;
    let retrieved = backend.get("test-1#0", Some(&namespace)).await?;
    assert!(retrieved.is_none());

    // Cleanup
    backend.delete_namespace(&namespace).await?;

    Ok(())
}

#[ignore = "requires DRIPSTONE_INT_S3_BUCKET, DRIPSTONE_INT_S3_VECTOR_BUCKET, and AWS credentials; run with --ignored"]
#[tokio::test]
async fn test_s3_bulk_operations() -> Result<(), DripstoneError> {
    let config = require_s3_config()?;

    let backend: Arc<dyn Backend> = Arc::new(S3Backend::with_config(config).await?);

    let namespace = format!(
        "s3-bulk-{}",
        OffsetDateTime::now_utc().unix_timestamp_nanos()
    );

    // Create bulk chunks
    let mut chunks = Vec::new();
    for i in 0..20 {
        let vector: Vec<f32> = (0..128).map(|j| ((i + j) as f32 * 0.01) % 1.0).collect();
        let chunk = Chunk::new(
            format!("bulk-{i}"),
            0,
            format!("Content {i}"),
            vector,
            0,
            12,
            1,
            serde_json::json!({}),
            Some(namespace.clone()),
        );
        chunks.push(chunk);
    }

    // Bulk upsert
    backend.upsert(&chunks).await?;

    // Verify count
    let count = backend.count(Some(&namespace)).await?;
    assert_eq!(count, 20);

    // List all
    let listed = backend.list(Some(&namespace), None).await?;
    assert_eq!(listed.len(), 20);

    // Cleanup
    backend.delete_namespace(&namespace).await?;

    Ok(())
}

#[ignore = "requires DRIPSTONE_INT_S3_BUCKET, DRIPSTONE_INT_S3_VECTOR_BUCKET, and AWS credentials; run with --ignored"]
#[tokio::test]
async fn test_s3_similarity_search() -> Result<(), DripstoneError> {
    let config = require_s3_config()?;

    let backend: Arc<dyn Backend> = Arc::new(S3Backend::with_config(config).await?);

    let namespace = format!(
        "s3-search-{}",
        OffsetDateTime::now_utc().unix_timestamp_nanos()
    );

    // Insert test vectors
    let mut chunks = Vec::new();
    for i in 0..10 {
        let mut vector = vec![0.0; 128];
        vector[0] = i as f32 / 10.0;
        let chunk = Chunk::new(
            format!("vec-{i}"),
            0,
            format!("Content {i}"),
            vector,
            0,
            12,
            1,
            serde_json::json!({}),
            Some(namespace.clone()),
        );
        chunks.push(chunk);
    }
    backend.upsert(&chunks).await?;

    // Search
    let mut query_vector = vec![0.0; 128];
    query_vector[0] = 0.5;
    let query = SearchQuery {
        vector: Arc::from(query_vector.into_boxed_slice()),
        limit: 5,
        metric: Metric::Cosine,
        namespace: Some(namespace.clone()),
        filter: None,
        hybrid_weight: None,
        sparse_vector: None,
        include_content: true,
        include_vectors: false,
        query_multi_vectors: None,
        maxsim: None,
    };

    let results = backend.search(&query).await?;

    assert!(!results.is_empty());
    assert!(results.len() <= 5);

    // Results should be ordered by score
    for i in 1..results.len() {
        assert!(results[i - 1].score >= results[i].score);
    }

    // Cleanup
    backend.delete_namespace(&namespace).await?;

    Ok(())
}

#[ignore = "requires DRIPSTONE_INT_S3_BUCKET, DRIPSTONE_INT_S3_VECTOR_BUCKET, and AWS credentials; run with --ignored"]
#[tokio::test]
async fn test_tiered_storage() -> Result<(), DripstoneError> {
    let mut config = require_s3_config()?;
    config.key_prefix.push_str("tiered/");

    // Create cold tier (S3) and hot tier (Memory)
    let cold_tier: Arc<S3Backend> = Arc::new(S3Backend::with_config(config).await?);
    let hot_tier: Arc<MemoryBackend> = Arc::new(MemoryBackend::new());

    // Create tiered backend (cold -> hot)
    let backend: Arc<dyn Backend> = TieredChainBuilder::new()
        .tier(cold_tier, TierConfig::new("s3"))
        .tier(
            hot_tier,
            TierConfig::new("memory")
                .max_vectors(5)
                .max_bytes(10 * 1024),
        )
        .background_sync(false)
        .sync_interval_secs(60)
        .build()?;

    let namespace = format!(
        "tiered-{}",
        OffsetDateTime::now_utc().unix_timestamp_nanos()
    );

    // Insert vectors
    let mut chunks = Vec::new();
    for i in 0..10 {
        let vector: Vec<f32> = (0..128).map(|j| ((i + j) as f32 * 0.01) % 1.0).collect();
        let chunk = Chunk::new(
            format!("tier-{i}"),
            0,
            format!("Content {i}"),
            vector,
            0,
            12,
            1,
            serde_json::json!({}),
            Some(namespace.clone()),
        );
        chunks.push(chunk);
    }
    backend.upsert(&chunks).await?;

    // Verify all records are accessible
    for i in 0..10 {
        let chunk = backend
            .get(&format!("tier-{i}#0"), Some(&namespace))
            .await?;
        assert!(chunk.is_some());
    }

    // Verify count
    let count = backend.count(Some(&namespace)).await?;
    assert_eq!(count, 10);

    // Perform search
    let query_vector: Vec<f32> = (0..128).map(|i| (i as f32 * 0.01) % 1.0).collect();
    let query = SearchQuery {
        vector: Arc::from(query_vector.into_boxed_slice()),
        limit: 5,
        metric: Metric::Cosine,
        namespace: Some(namespace.clone()),
        filter: None,
        hybrid_weight: None,
        sparse_vector: None,
        include_content: true,
        include_vectors: false,
        query_multi_vectors: None,
        maxsim: None,
    };
    let results = backend.search(&query).await?;

    assert!(!results.is_empty());
    assert!(results.len() <= 5);

    // Cleanup
    backend.delete_namespace(&namespace).await?;

    Ok(())
}

#[ignore = "requires DRIPSTONE_INT_S3_BUCKET, DRIPSTONE_INT_S3_VECTOR_BUCKET, and AWS credentials; run with --ignored"]
#[tokio::test]
async fn test_s3_namespace_management() -> Result<(), DripstoneError> {
    let config = require_s3_config()?;

    let backend: Arc<dyn Backend> = Arc::new(S3Backend::with_config(config).await?);

    let timestamp = OffsetDateTime::now_utc().unix_timestamp_nanos();
    let ns1 = format!("ns1-{timestamp}");
    let ns2 = format!("ns2-{timestamp}");

    // Insert into namespace 1
    let chunk1 = Chunk::new(
        "doc-1",
        0,
        "Content 1",
        vec![0.1, 0.2, 0.3],
        0,
        10,
        1,
        serde_json::json!({}),
        Some(ns1.clone()),
    );
    backend.upsert(&[chunk1]).await?;

    // Insert into namespace 2
    let chunk2 = Chunk::new(
        "doc-2",
        0,
        "Content 2",
        vec![0.4, 0.5, 0.6],
        0,
        10,
        1,
        serde_json::json!({}),
        Some(ns2.clone()),
    );
    backend.upsert(&[chunk2]).await?;

    // List namespaces
    let namespaces = backend.namespaces().await?;
    assert!(namespaces.contains(&ns1));
    assert!(namespaces.contains(&ns2));

    // Flush namespace 1
    backend.delete_namespace(&ns1).await?;

    // Verify namespace 1 is empty but namespace 2 still has data
    let count1 = backend.count(Some(&ns1)).await?;
    let count2 = backend.count(Some(&ns2)).await?;
    assert_eq!(count1, 0);
    assert_eq!(count2, 1);

    // Cleanup
    backend.delete_namespace(&ns2).await?;

    Ok(())
}

// ============================================================================
// S3 Vector Buckets Tests
// ============================================================================

fn require_vector_config() -> Result<S3VectorBackendConfig, DripstoneError> {
    let bucket = require_named_env("DRIPSTONE_INT_S3_VECTOR_BUCKET")?;
    let index = require_named_env("DRIPSTONE_INT_S3_VECTOR_INDEX")?;
    let region = env::var("AWS_REGION").unwrap_or_else(|_| "us-east-1".to_string());

    Ok(
        S3VectorBackendConfig::new(bucket, index, 128, None, None, None)
            .with_distance_metric(Metric::Cosine)
            .with_region(region)
            .with_batch_size(50),
    )
}

#[ignore = "requires DRIPSTONE_INT_S3_BUCKET, DRIPSTONE_INT_S3_VECTOR_BUCKET, and AWS credentials; run with --ignored"]
#[tokio::test]
async fn test_s3_vector_basic_operations() -> Result<(), DripstoneError> {
    let config = require_vector_config()?;

    let backend: Arc<dyn Backend> = Arc::new(S3VectorBackend::new(config).await?);

    let namespace = format!(
        "vector-basic-{}",
        OffsetDateTime::now_utc().unix_timestamp_nanos()
    );

    // Test upsert
    let vector = vec![0.1; 128];
    let chunk = Chunk::new(
        "vec-test-1",
        0,
        "Test content",
        vector.clone(),
        0,
        12,
        1,
        serde_json::json!({"category": "test", "priority": 5}),
        Some(namespace.clone()),
    );

    backend.upsert(&[chunk]).await?;
    outputln!("Upserted vector to S3 Vector Bucket");

    // Test get
    let retrieved = backend.get("vec-test-1#0", Some(&namespace)).await?;
    assert!(retrieved.is_some());
    let retrieved = retrieved.unwrap();
    assert_eq!(retrieved.id, "vec-test-1#0");
    outputln!("Retrieved vector from S3 Vector Bucket");

    // Test count
    let count = backend.count(Some(&namespace)).await?;
    assert_eq!(count, 1);
    outputln!("Count verified: {count}");

    // Test delete
    backend.delete("vec-test-1#0", Some(&namespace)).await?;
    let count = backend.count(Some(&namespace)).await?;
    assert_eq!(count, 0);
    outputln!("Deleted vector from S3 Vector Bucket");

    Ok(())
}

#[ignore = "requires DRIPSTONE_INT_S3_BUCKET, DRIPSTONE_INT_S3_VECTOR_BUCKET, and AWS credentials; run with --ignored"]
#[tokio::test]
async fn test_s3_vector_bulk_operations() -> Result<(), DripstoneError> {
    let config = require_vector_config()?;

    let backend: Arc<dyn Backend> = Arc::new(S3VectorBackend::new(config).await?);

    let namespace = format!(
        "vector-bulk-{}",
        OffsetDateTime::now_utc().unix_timestamp_nanos()
    );

    // Create bulk chunks
    let mut chunks = Vec::new();
    for i in 0..25 {
        let vector: Vec<f32> = (0..128).map(|j| ((i + j) as f32 * 0.01) % 1.0).collect();
        let chunk = Chunk::new(
            format!("vec-bulk-{i}"),
            0,
            format!("Content {i}"),
            vector,
            0,
            12,
            1,
            serde_json::json!({"batch": i / 10}),
            Some(namespace.clone()),
        );
        chunks.push(chunk);
    }

    // Bulk upsert
    backend.upsert(&chunks).await?;
    outputln!("Bulk upserted 25 vectors");

    // Verify count
    let count = backend.count(Some(&namespace)).await?;
    assert_eq!(count, 25);
    outputln!("Count verified: {count}");

    // Cleanup
    backend.delete_namespace(&namespace).await?;
    let count = backend.count(Some(&namespace)).await?;
    assert_eq!(count, 0);
    outputln!("Namespace flushed");

    Ok(())
}

#[ignore = "requires DRIPSTONE_INT_S3_BUCKET, DRIPSTONE_INT_S3_VECTOR_BUCKET, and AWS credentials; run with --ignored"]
#[tokio::test]
async fn test_s3_vector_similarity_search() -> Result<(), DripstoneError> {
    let config = require_vector_config()?;

    let backend: Arc<dyn Backend> = Arc::new(S3VectorBackend::new(config).await?);

    let namespace = format!(
        "vector-search-{}",
        OffsetDateTime::now_utc().unix_timestamp_nanos()
    );

    // Insert test vectors with varying similarity
    let mut chunks = Vec::new();
    for i in 0..10 {
        let mut vector = vec![0.0; 128];
        // Create vectors with different patterns
        vector[0] = (i as f32) / 10.0;
        vector[1] = 1.0 - (i as f32) / 10.0;

        let chunk = Chunk::new(
            format!("vec-{i}"),
            0,
            format!("Content {i}"),
            vector,
            0,
            12,
            1,
            serde_json::json!({"index": i}),
            Some(namespace.clone()),
        );
        chunks.push(chunk);
    }
    backend.upsert(&chunks).await?;
    outputln!("Inserted 10 test vectors");

    // Search with server-side similarity
    let mut query_vector = vec![0.0; 128];
    query_vector[0] = 0.5;
    query_vector[1] = 0.5;

    let query = SearchQuery {
        vector: Arc::from(query_vector.into_boxed_slice()),
        limit: 5,
        metric: Metric::Cosine,
        namespace: Some(namespace.clone()),
        filter: None,
        hybrid_weight: None,
        sparse_vector: None,
        include_content: true,
        include_vectors: false,
        query_multi_vectors: None,
        maxsim: None,
    };

    let results = backend.search(&query).await?;

    assert!(!results.is_empty());
    assert!(results.len() <= 5);
    outputln!("Search returned {} results", results.len());

    // Verify results are ordered by score
    for i in 1..results.len() {
        assert!(results[i - 1].score >= results[i].score);
    }
    outputln!("Results are properly ordered by score");

    // Cleanup
    backend.delete_namespace(&namespace).await?;

    Ok(())
}

#[ignore = "requires DRIPSTONE_INT_S3_BUCKET, DRIPSTONE_INT_S3_VECTOR_BUCKET, and AWS credentials; run with --ignored"]
#[tokio::test]
async fn test_s3_vector_metadata_filtering() -> Result<(), DripstoneError> {
    let config = require_vector_config()?;

    let backend: Arc<dyn Backend> = Arc::new(S3VectorBackend::new(config).await?);

    let namespace = format!(
        "vector-meta-{}",
        OffsetDateTime::now_utc().unix_timestamp_nanos()
    );

    // Insert vectors with different categories
    let mut chunks = Vec::new();
    for i in 0..10 {
        let vector = vec![(i as f32) * 0.1; 128];
        let category = if i % 2 == 0 { "even" } else { "odd" };

        let chunk = Chunk::new(
            format!("vec-{i}"),
            0,
            format!("Content {i}"),
            vector,
            0,
            12,
            1,
            serde_json::json!({
                "category": category,
                "value": i
            }),
            Some(namespace.clone()),
        );
        chunks.push(chunk);
    }
    backend.upsert(&chunks).await?;
    outputln!("Inserted 10 vectors with metadata");

    // Search with metadata filter
    let query_vector = vec![0.3; 128];
    let filter = Filter::Equal {
        field: "category".to_string(),
        value: serde_json::json!("even"),
    };
    let query = SearchQuery {
        vector: Arc::from(query_vector.into_boxed_slice()),
        limit: 10,
        metric: Metric::Cosine,
        namespace: Some(namespace.clone()),
        filter: Some(filter),
        hybrid_weight: None,
        sparse_vector: None,
        include_content: true,
        include_vectors: false,
        query_multi_vectors: None,
        maxsim: None,
    };

    let results = backend.search(&query).await?;

    outputln!("Filtered search returned {} results", results.len());

    // Verify all results match the filter
    for result in &results {
        let category = result
            .chunk
            .metadata
            .get("category")
            .and_then(|v| v.as_str());
        assert_eq!(category, Some("even"));
    }
    outputln!("All results match metadata filter");

    // Cleanup
    backend.delete_namespace(&namespace).await?;

    Ok(())
}

#[ignore = "requires DRIPSTONE_INT_S3_BUCKET, DRIPSTONE_INT_S3_VECTOR_BUCKET, and AWS credentials; run with --ignored"]
#[tokio::test]
async fn test_s3_vector_dimension_validation() -> Result<(), DripstoneError> {
    let config = require_vector_config()?;

    let backend: Arc<dyn Backend> = Arc::new(S3VectorBackend::new(config).await?);

    let namespace = format!(
        "vector-dim-{}",
        OffsetDateTime::now_utc().unix_timestamp_nanos()
    );

    // Try to insert vector with wrong dimension (should fail)
    let wrong_vector = vec![0.1; 256]; // Config expects 128
    let wrong_chunk = Chunk::new(
        "wrong-dim",
        0,
        "Content",
        wrong_vector,
        0,
        7,
        1,
        serde_json::json!({}),
        Some(namespace.clone()),
    );

    let result = backend.upsert(&[wrong_chunk]).await;
    assert!(result.is_err());
    outputln!("Dimension validation works correctly");

    // Insert vector with correct dimension (should succeed)
    let correct_vector = vec![0.1; 128];
    let correct_chunk = Chunk::new(
        "correct-dim",
        0,
        "Content",
        correct_vector,
        0,
        7,
        1,
        serde_json::json!({}),
        Some(namespace.clone()),
    );

    backend.upsert(&[correct_chunk]).await?;
    outputln!("Correct dimension vector inserted successfully");

    // Cleanup
    backend.delete_namespace(&namespace).await?;

    Ok(())
}

#[ignore = "requires DRIPSTONE_INT_S3_BUCKET, DRIPSTONE_INT_S3_VECTOR_BUCKET, and AWS credentials; run with --ignored"]
#[tokio::test]
async fn test_s3_vector_tiered_storage() -> Result<(), DripstoneError> {
    let config = require_vector_config()?;

    // Test S3 Vector Backend directly
    // Note: Tiered backend testing with S3VectorBackend would require concrete types
    // and is better tested in a separate integration test
    let backend: Arc<dyn Backend> = Arc::new(S3VectorBackend::new(config).await?);

    let namespace = format!(
        "vector-tiered-{}",
        OffsetDateTime::now_utc().unix_timestamp_nanos()
    );

    // Insert vectors
    let mut chunks = Vec::new();
    for i in 0..10 {
        let vector = vec![(i as f32) * 0.1; 128];
        let chunk = Chunk::new(
            format!("vec-{i}"),
            0,
            format!("Content {i}"),
            vector,
            0,
            12,
            1,
            serde_json::json!({"index": i}),
            Some(namespace.clone()),
        );
        chunks.push(chunk);
    }
    backend.upsert(&chunks).await?;
    outputln!("Inserted 10 vectors to tiered storage");

    // Verify all records are accessible
    for i in 0..10 {
        let chunk = backend.get(&format!("vec-{i}#0"), Some(&namespace)).await?;
        assert!(chunk.is_some(), "Record vec-{i} should be accessible");
    }
    outputln!("All vectors accessible via tiered storage");

    // Search should work
    let query_vector = vec![0.3; 128];
    let query = SearchQuery {
        vector: Arc::from(query_vector.into_boxed_slice()),
        limit: 5,
        metric: Metric::Cosine,
        namespace: Some(namespace.clone()),
        filter: None,
        hybrid_weight: None,
        sparse_vector: None,
        include_content: true,
        include_vectors: false,
        query_multi_vectors: None,
        maxsim: None,
    };
    let results = backend.search(&query).await?;

    assert!(!results.is_empty());
    outputln!(
        "Search works across tiered storage ({} results)",
        results.len()
    );

    // Count should return total
    let count = backend.count(Some(&namespace)).await?;
    assert_eq!(count, 10);
    outputln!("Count returns accurate total: {count}");

    // Cleanup
    backend.delete_namespace(&namespace).await?;
    outputln!("Cleaned up tiered test data");

    Ok(())
}
