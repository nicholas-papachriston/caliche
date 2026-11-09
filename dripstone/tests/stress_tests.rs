//! Stress tests for embedding store backends
//!
//! These tests are designed to run with high concurrency and may require significant
//! database resources. If tests fail with connection pool timeouts when running all
//! tests together, consider:
//!
//! 1. Running tests serially: `cargo test --test stress_tests -- --test-threads=1`
//! 2. Increasing database `max_connections` in postgresql.conf
//! 3. Adjusting `STRESS_CONCURRENCY` environment variable to a lower value

#[path = "common/mod.rs"]
mod common;

#[cfg(any(
    feature = "redis-backend",
    feature = "postgres-backend",
    feature = "qdrant-backend"
))]
macro_rules! outputln {
    ($($arg:tt)*) => {{
        let mut output = std::io::stdout().lock();
        let _ = std::io::Write::write_fmt(&mut output, format_args!($($arg)*));
        let _ = std::io::Write::write_all(&mut output, b"\n");
    }};
}

#[cfg(any(
    feature = "redis-backend",
    feature = "postgres-backend",
    feature = "qdrant-backend"
))]
macro_rules! errorln {
    ($($arg:tt)*) => {{
        let mut output = std::io::stderr().lock();
        let _ = std::io::Write::write_fmt(&mut output, format_args!($($arg)*));
        let _ = std::io::Write::write_all(&mut output, b"\n");
    }};
}

#[cfg(any(
    feature = "redis-backend",
    feature = "postgres-backend",
    feature = "qdrant-backend"
))]
use common::*;
#[cfg(any(
    feature = "redis-backend",
    feature = "postgres-backend",
    feature = "qdrant-backend"
))]
use dripstone::{DripstoneError, Metric, SearchQuery, traits::Backend};
#[cfg(any(
    feature = "redis-backend",
    feature = "postgres-backend",
    feature = "qdrant-backend"
))]
use std::env;
#[cfg(any(
    feature = "redis-backend",
    feature = "postgres-backend",
    feature = "qdrant-backend"
))]
use std::sync::Arc;
#[cfg(any(
    feature = "redis-backend",
    feature = "postgres-backend",
    feature = "qdrant-backend"
))]
use time::OffsetDateTime;
#[cfg(any(
    feature = "redis-backend",
    feature = "postgres-backend",
    feature = "qdrant-backend"
))]
use tokio::task::JoinSet;

#[cfg(any(
    feature = "redis-backend",
    feature = "postgres-backend",
    feature = "qdrant-backend"
))]
fn get_stress_concurrency() -> usize {
    env::var("STRESS_CONCURRENCY")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(100)
}

#[cfg(any(
    feature = "redis-backend",
    feature = "postgres-backend",
    feature = "qdrant-backend"
))]
fn get_stress_batch_size() -> usize {
    env::var("STRESS_BATCH_SIZE")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(1000)
}

#[cfg(any(
    feature = "redis-backend",
    feature = "postgres-backend",
    feature = "qdrant-backend"
))]
fn get_stress_iterations() -> usize {
    env::var("STRESS_ITERATIONS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(10)
}

// High concurrency write test for Redis
#[cfg(feature = "redis-backend")]
#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn test_redis_high_concurrency_writes() -> Result<(), DripstoneError> {
    use dripstone::backend::redis::RedisBackend;

    let url = env::var("REDIS_URL").unwrap_or_else(|_| redis_url_local());
    let namespace = format!(
        "stress-redis-{}",
        OffsetDateTime::now_utc().unix_timestamp_nanos()
    );

    let backend: Arc<dyn Backend> =
        Arc::new(RedisBackend::with_prefix(url.as_str(), "stress").await?);

    backend.delete_namespace(&namespace).await?;

    let concurrency = get_stress_concurrency();
    outputln!("Running high concurrency test with {concurrency} parallel writes");

    let mut tasks = JoinSet::new();
    for i in 0..concurrency {
        let backend_clone = Arc::clone(&backend);
        let ns = namespace.clone();
        tasks.spawn(async move {
            let chunk = build_chunk(format!("stress-{i}"), &ns, i, 128);
            backend_clone.upsert(&[chunk]).await
        });
    }

    let mut success_count = 0;
    while let Some(result) = tasks.join_next().await {
        match result {
            Ok(Ok(())) => success_count += 1,
            Ok(Err(e)) => errorln!("Upsert failed: {e}"),
            Err(e) => errorln!("Task join failed: {e}"),
        }
    }

    outputln!("Successfully completed {success_count}/{concurrency} concurrent writes");
    assert!(
        success_count >= concurrency * 95 / 100,
        "At least 95% of writes should succeed"
    );

    let count = backend.count(Some(&namespace)).await?;
    outputln!("Final count: {count}");

    backend.delete_namespace(&namespace).await?;
    Ok(())
}

// High concurrency write test for Valkey
#[cfg(feature = "redis-backend")]
#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn test_valkey_high_concurrency_writes() -> Result<(), DripstoneError> {
    use dripstone::backend::redis::RedisBackend;

    let url = env::var("VALKEY_URL").unwrap_or_else(|_| valkey_url_local());
    let namespace = format!(
        "stress-valkey-{}",
        OffsetDateTime::now_utc().unix_timestamp_nanos()
    );

    let backend: Arc<dyn Backend> =
        Arc::new(RedisBackend::with_prefix(url.as_str(), "stress-valkey").await?);

    backend.delete_namespace(&namespace).await?;

    let concurrency = get_stress_concurrency();
    outputln!("Running Valkey high concurrency test with {concurrency} parallel writes");

    let mut tasks = JoinSet::new();
    for i in 0..concurrency {
        let backend_clone = Arc::clone(&backend);
        let ns = namespace.clone();
        tasks.spawn(async move {
            let chunk = build_chunk(format!("stress-valkey-{i}"), &ns, i, 128);
            backend_clone.upsert(&[chunk]).await
        });
    }

    let mut success_count = 0;
    while let Some(result) = tasks.join_next().await {
        match result {
            Ok(Ok(())) => success_count += 1,
            Ok(Err(e)) => errorln!("Valkey upsert failed: {e}"),
            Err(e) => errorln!("Valkey task join failed: {e}"),
        }
    }

    outputln!("Valkey successfully completed {success_count}/{concurrency} concurrent writes");
    assert!(
        success_count >= concurrency * 95 / 100,
        "At least 95% of Valkey writes should succeed"
    );

    let count = backend.count(Some(&namespace)).await?;
    outputln!("Valkey final count: {count}");

    backend.delete_namespace(&namespace).await?;
    Ok(())
}

// Rapid namespace cycling test for Postgres
#[cfg(feature = "postgres-backend")]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_postgres_rapid_namespace_cycling() -> Result<(), DripstoneError> {
    use dripstone::backend::postgres::{PostgresBackend, PostgresBackendOptions};

    let url = postgres_url_local();

    let iterations = get_stress_iterations();
    let options = PostgresBackendOptions {
        table_name: "embeddings_stress_cycle_128".to_string(),
        ensure_schema: true,
        max_connections: ((iterations * 2 + 10).min(50)) as u32,
        vector_dimension: Some(128),
        hnsw_index: None,
    };

    let backend: Arc<dyn Backend> = Arc::new(PostgresBackend::with_options(&url, options).await?);

    let iterations = get_stress_iterations();
    outputln!("Running namespace cycling test with {iterations} iterations");

    for i in 0..iterations {
        let namespace = format!(
            "cycle-{}-{}",
            OffsetDateTime::now_utc().unix_timestamp_nanos(),
            i
        );

        // Create
        let chunks = build_concurrent_chunks(50, &namespace, 128, i * 1000);
        backend.upsert(&chunks).await?;

        // Verify
        let count = backend.count(Some(&namespace)).await?;
        assert_eq!(count, 50, "Iteration {i}: should have 50 records");

        // Flush
        backend.delete_namespace(&namespace).await?;

        // Verify empty
        let count_after = backend.count(Some(&namespace)).await?;
        assert_eq!(
            count_after, 0,
            "Iteration {i}: should have 0 records after flush"
        );

        if i % 5 == 0 {
            outputln!("Completed {} iterations", i + 1);
        }
    }

    outputln!("Successfully completed {iterations} namespace cycles");
    Ok(())
}

// Large batch operations test for Redis
#[cfg(feature = "redis-backend")]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_redis_large_batch_operations() -> Result<(), DripstoneError> {
    use dripstone::backend::redis::RedisBackend;

    let url = env::var("REDIS_URL").unwrap_or_else(|_| redis_url_local());
    let namespace = format!(
        "batch-redis-{}",
        OffsetDateTime::now_utc().unix_timestamp_nanos()
    );

    let backend: Arc<dyn Backend> =
        Arc::new(RedisBackend::with_prefix(url.as_str(), "batch").await?);

    backend.delete_namespace(&namespace).await?;

    let batch_size = get_stress_batch_size();
    outputln!("Running large batch test with {batch_size} records");

    let chunks = build_concurrent_chunks(batch_size, &namespace, 128, 10000);

    let start = std::time::Instant::now();
    backend.upsert(&chunks).await?;
    let duration = start.elapsed();

    outputln!("Bulk upsert of {batch_size} records took {duration:?}");

    let count = backend.count(Some(&namespace)).await?;
    assert_eq!(count, batch_size, "Should have all records");

    // Test large list operation
    let start = std::time::Instant::now();
    let listed = backend.list(Some(&namespace), None).await?;
    let list_duration = start.elapsed();

    outputln!("List of {} records took {:?}", listed.len(), list_duration);
    assert_eq!(listed.len(), batch_size, "List should return all records");

    backend.delete_namespace(&namespace).await?;
    Ok(())
}

// Large batch operations test for Valkey
#[cfg(feature = "redis-backend")]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_valkey_large_batch_operations() -> Result<(), DripstoneError> {
    use dripstone::backend::redis::RedisBackend;

    let url = env::var("VALKEY_URL").unwrap_or_else(|_| valkey_url_local());
    let namespace = format!(
        "batch-valkey-{}",
        OffsetDateTime::now_utc().unix_timestamp_nanos()
    );

    let backend: Arc<dyn Backend> =
        Arc::new(RedisBackend::with_prefix(url.as_str(), "batch-valkey").await?);

    backend.delete_namespace(&namespace).await?;

    let batch_size = get_stress_batch_size();
    outputln!("Running Valkey large batch test with {batch_size} records");

    let chunks = build_concurrent_chunks(batch_size, &namespace, 128, 20000);

    let start = std::time::Instant::now();
    backend.upsert(&chunks).await?;
    let duration = start.elapsed();

    outputln!("Valkey bulk upsert of {batch_size} records took {duration:?}");

    let count = backend.count(Some(&namespace)).await?;
    assert_eq!(count, batch_size, "Valkey should have all records");

    let start = std::time::Instant::now();
    let listed = backend.list(Some(&namespace), None).await?;
    let list_duration = start.elapsed();

    outputln!(
        "Valkey list of {} records took {:?}",
        listed.len(),
        list_duration
    );
    assert_eq!(
        listed.len(),
        batch_size,
        "Valkey list should return all records"
    );

    backend.delete_namespace(&namespace).await?;
    Ok(())
}

// Search under write load test for Postgres
#[cfg(feature = "postgres-backend")]
#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn test_postgres_search_under_write_load() -> Result<(), DripstoneError> {
    use dripstone::backend::postgres::{PostgresBackend, PostgresBackendOptions};

    let url = postgres_url_local();
    let namespace = format!(
        "search-load-{}",
        OffsetDateTime::now_utc().unix_timestamp_nanos()
    );

    let concurrency = get_stress_concurrency();
    let options = PostgresBackendOptions {
        table_name: "embeddings_stress_search_128".to_string(),
        ensure_schema: true,
        max_connections: ((concurrency / 2 + 20).min(100)) as u32,
        vector_dimension: Some(128),
        hnsw_index: None,
    };

    let backend: Arc<dyn Backend> = Arc::new(PostgresBackend::with_options(&url, options).await?);

    backend.delete_namespace(&namespace).await?;

    // Insert initial data
    let initial_chunks = build_concurrent_chunks(100, &namespace, 128, 20000);
    backend.upsert(&initial_chunks).await?;

    outputln!("Starting concurrent search and write test");

    let mut tasks = JoinSet::new();

    // Spawn writers
    for i in 0..20 {
        let backend_clone = Arc::clone(&backend);
        let ns = namespace.clone();
        tasks.spawn(async move {
            let chunk = build_chunk(format!("writer-{i}"), &ns, 20100 + i, 128);
            backend_clone.upsert(&[chunk]).await.map(|()| "write")
        });
    }

    // Spawn searchers
    for i in 0..30 {
        let backend_clone = Arc::clone(&backend);
        let ns = namespace.clone();
        let query_vec = build_vector(20050 + i, 128);
        tasks.spawn(async move {
            let query = SearchQuery::new(query_vec)
                .metric(Metric::Cosine)
                .limit(10)
                .namespace(&ns);
            backend_clone.search(&query).await.map(|_| "search")
        });
    }

    let mut write_count = 0;
    let mut search_count = 0;

    while let Some(result) = tasks.join_next().await {
        match result {
            Ok(Ok("write")) => write_count += 1,
            Ok(Ok("search")) => search_count += 1,
            Ok(Ok(_)) => {}
            Ok(Err(e)) => errorln!("Operation failed: {e}"),
            Err(e) => errorln!("Task join failed: {e}"),
        }
    }

    outputln!("Completed {write_count} writes and {search_count} searches concurrently");
    assert!(write_count >= 18, "Most writes should succeed");
    assert!(search_count >= 28, "Most searches should succeed");

    backend.delete_namespace(&namespace).await?;
    Ok(())
}

// High concurrency writes for Qdrant
#[cfg(feature = "qdrant-backend")]
#[ignore = "requires a Qdrant server at EMBEDDING_QDRANT_URL or http://127.0.0.1:6333; run with --ignored"]
#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn test_qdrant_high_concurrency_writes() -> Result<(), DripstoneError> {
    use dripstone::backend::qdrant::{QdrantBackendOptions, QdrantHttpBackend};

    let url =
        env::var("EMBEDDING_QDRANT_URL").unwrap_or_else(|_| "http://127.0.0.1:6333".to_string());
    let api_key = qdrant_api_key();
    let namespace = format!(
        "stress-qdrant-{}",
        OffsetDateTime::now_utc().unix_timestamp_nanos()
    );

    let mut options = QdrantBackendOptions::new(url, 128);
    options.collection_prefix = Some("stress_".to_string());
    options.api_key = api_key;

    let backend: Arc<dyn Backend> = match QdrantHttpBackend::with_options(options).await {
        Ok(backend) => Arc::new(backend),
        Err(error) => {
            return Err(DripstoneError::storage(format!(
                "Qdrant stress test requires a server: {error}"
            )));
        }
    };

    backend.delete_namespace(&namespace).await?;

    let concurrency = get_stress_concurrency();
    outputln!("Running Qdrant high concurrency test with {concurrency} parallel writes");

    let mut tasks = JoinSet::new();
    for i in 0..concurrency {
        let backend_clone = Arc::clone(&backend);
        let ns = namespace.clone();
        tasks.spawn(async move {
            let chunk = build_chunk(format!("stress-{i}"), &ns, i, 128);
            backend_clone.upsert(&[chunk]).await
        });
    }

    let mut success_count = 0;
    while let Some(result) = tasks.join_next().await {
        match result {
            Ok(Ok(())) => success_count += 1,
            Ok(Err(e)) => errorln!("Upsert failed: {e}"),
            Err(e) => errorln!("Task join failed: {e}"),
        }
    }

    outputln!("Successfully completed {success_count}/{concurrency} concurrent writes");
    assert!(
        success_count >= concurrency * 95 / 100,
        "At least 95% of writes should succeed"
    );

    backend.delete_namespace(&namespace).await?;
    Ok(())
}

// Pipeline stress test with concurrent searches
#[cfg(feature = "redis-backend")]
#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn test_redis_pipeline_concurrent_searches() -> Result<(), DripstoneError> {
    use dripstone::backend::redis::RedisBackend;
    use dripstone::pipeline::SearchPipelineBuilder;

    let url = env::var("REDIS_URL").unwrap_or_else(|_| redis_url_local());
    let namespace = format!(
        "stress-pipeline-redis-{}",
        OffsetDateTime::now_utc().unix_timestamp_nanos()
    );

    // Use concrete type for pipeline
    let backend: Arc<RedisBackend> =
        Arc::new(RedisBackend::with_prefix(url.as_str(), "stress").await?);
    let backend_trait: Arc<dyn Backend> = backend.clone() as Arc<dyn Backend>;

    backend_trait.delete_namespace(&namespace).await?;

    // Insert test data
    let batch_size = get_stress_batch_size();
    let mut chunks = Vec::new();
    for i in 0..batch_size {
        chunks.push(build_chunk(format!("pipeline-{i}"), &namespace, i, 128));
        // Update metadata
        let last_idx = chunks.len() - 1;
        chunks[last_idx].metadata = serde_json::json!({
            "category": if i % 3 == 0 { "A" } else if i % 3 == 1 { "B" } else { "C" },
            "value": i
        });
    }
    backend_trait.upsert(&chunks).await?;

    outputln!("Inserted {batch_size} records for pipeline stress test");

    // Build a complex pipeline - need concrete type
    let pipeline = SearchPipelineBuilder::<RedisBackend>::new(Metric::Cosine)
        .coarse_ann(100)
        .metadata_filter("category", serde_json::json!("A"))
        .normalize_scores()
        .with_final_limit(10)
        .build();

    let concurrency = get_stress_concurrency();
    outputln!("Running {concurrency} concurrent pipeline searches");

    let mut tasks = JoinSet::new();
    for i in 0..concurrency {
        let backend_clone = Arc::clone(&backend);
        let ns = namespace.clone();
        let pipeline_clone = pipeline.clone();
        tasks.spawn(async move {
            let query_vec = build_vector(i, 128);
            pipeline_clone
                .execute(&*backend_clone, &query_vec, Some(&ns))
                .await
        });
    }

    let mut success_count = 0;
    while let Some(result) = tasks.join_next().await {
        match result {
            Ok(Ok(results)) => {
                success_count += 1;
                // Verify all results match the filter
                for result in results {
                    if let Some(category) = result
                        .chunk
                        .metadata
                        .get("category")
                        .and_then(|v| v.as_str())
                    {
                        assert_eq!(category, "A", "Pipeline filter should be applied");
                    }
                }
            }
            Ok(Err(e)) => errorln!("Pipeline search failed: {e}"),
            Err(e) => errorln!("Task join failed: {e}"),
        }
    }

    outputln!("Successfully completed {success_count}/{concurrency} concurrent pipeline searches");
    assert!(
        success_count >= concurrency * 95 / 100,
        "At least 95% of pipeline searches should succeed"
    );

    backend_trait.delete_namespace(&namespace).await?;
    Ok(())
}

// Pipeline stress test for Postgres
#[cfg(feature = "postgres-backend")]
#[ignore = "requires a Postgres server at the local stress URL"]
#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn test_postgres_pipeline_concurrent_searches() -> Result<(), DripstoneError> {
    use dripstone::backend::postgres::{PostgresBackend, PostgresBackendOptions};
    use dripstone::pipeline::SearchPipelineBuilder;

    let url = postgres_url_local();
    let namespace = format!(
        "stress-pipeline-pg-{}",
        OffsetDateTime::now_utc().unix_timestamp_nanos()
    );

    let concurrency = get_stress_concurrency();
    // Use concrete type for pipeline
    let backend: Arc<PostgresBackend> = Arc::new(
        PostgresBackend::with_options(
            url.as_str(),
            PostgresBackendOptions {
                table_name: "embeddings_stress_pipeline_128".to_string(),
                ensure_schema: true,
                max_connections: ((concurrency + 10).min(200)) as u32, // Ensure enough connections for concurrent ops
                vector_dimension: Some(128),
                hnsw_index: None,
            },
        )
        .await?,
    );
    let backend_trait: Arc<dyn Backend> = backend.clone() as Arc<dyn Backend>;

    backend_trait.delete_namespace(&namespace).await?;

    // Insert test data
    let batch_size = get_stress_batch_size();
    let mut chunks = Vec::new();
    for i in 0..batch_size {
        let mut chunk = build_chunk(format!("pipeline-{i}"), &namespace, i, 128);
        chunk.metadata = serde_json::json!({
            "priority": if i % 2 == 0 { "high" } else { "low" }
        });
        chunks.push(chunk);
    }
    backend_trait.upsert(&chunks).await?;

    outputln!("Inserted {batch_size} records for pipeline stress test");

    // Build pipeline with diversity reranking - need concrete type
    let pipeline = SearchPipelineBuilder::<PostgresBackend>::new(Metric::Cosine)
        .coarse_ann(50)
        .diversity_rerank(
            dripstone::pipeline::DiversityConfig::new()
                .with_limit(20)
                .with_metadata_key("priority")
                .with_max_per_value(10),
        )
        .with_final_limit(15)
        .build();

    outputln!("Running {concurrency} concurrent diversity pipeline searches");

    let mut tasks = JoinSet::new();
    for i in 0..concurrency {
        let backend_clone = Arc::clone(&backend);
        let ns = namespace.clone();
        let pipeline_clone = pipeline.clone();
        tasks.spawn(async move {
            let query_vec = build_vector(i, 128);
            pipeline_clone
                .execute(&*backend_clone, &query_vec, Some(&ns))
                .await
        });
    }

    let mut success_count = 0;
    while let Some(result) = tasks.join_next().await {
        match result {
            Ok(Ok(results)) => {
                success_count += 1;
                assert!(results.len() <= 15, "Pipeline should respect final limit");
            }
            Ok(Err(e)) => errorln!("Pipeline search failed: {e}"),
            Err(e) => errorln!("Task join failed: {e}"),
        }
    }

    outputln!(
        "Successfully completed {success_count}/{concurrency} concurrent diversity pipeline searches"
    );
    assert!(
        success_count >= concurrency * 95 / 100,
        "At least 95% of pipeline searches should succeed"
    );

    backend_trait.delete_namespace(&namespace).await?;
    Ok(())
}
