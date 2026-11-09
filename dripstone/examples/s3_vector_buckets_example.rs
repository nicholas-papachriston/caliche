//! Example demonstrating AWS S3 Vector Buckets usage
//!
//! This example shows how to use the native S3 Vector Buckets API (preview feature)
//! for optimized vector storage and server-side similarity search.
//!
//! Prerequisites:
//! 1. AWS credentials configured (via env vars, config file, or IAM role)
//! 2. S3 Vector Bucket created in your AWS account
//! 3. Vector index created within the bucket
//!
//! Create a vector bucket using AWS CLI or Console:
//! ```bash
//! aws s3vectors create-vector-bucket --bucket-name my-vector-bucket --region us-east-1
//! aws s3vectors create-index --bucket-name my-vector-bucket --index-name my-index \
//!     --dimension 384 --distance-metric Cosine
//! ```
//!
//! See: <https://docs.aws.amazon.com/AmazonS3/latest/userguide/s3-vectors-getting-started.html>

#![cfg(feature = "s3-backend")]

macro_rules! outputln {
    () => {{
        let mut output = std::io::stdout().lock();
        let _ = std::io::Write::write_all(&mut output, b"\n");
    }};
    ($($arg:tt)+) => {{
        let mut output = std::io::stdout().lock();
        let _ = std::io::Write::write_fmt(&mut output, format_args!($($arg)*));
        let _ = std::io::Write::write_all(&mut output, b"\n");
    }};
}

use dripstone::backend::s3::{S3VectorBackend, S3VectorBackendConfig};
use dripstone::core::{Chunk, Metric, SearchQuery};
use dripstone::search::Filter;
use dripstone::traits::Backend;
use serde_json::json;
use std::env;
use std::sync::Arc;
use time::OffsetDateTime;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    outputln!("=== S3 Vector Buckets Example ===\n");

    let Ok(bucket_name) = env::var("S3_VECTOR_BUCKET") else {
        outputln!(
            "SKIP: set S3_VECTOR_BUCKET and S3_VECTOR_INDEX. This example does not create a vector bucket."
        );
        return Ok(());
    };
    let Ok(index_name) = env::var("S3_VECTOR_INDEX") else {
        outputln!("SKIP: set S3_VECTOR_INDEX. This example does not create a vector index.");
        return Ok(());
    };
    if bucket_name.is_empty() || index_name.is_empty() {
        outputln!("SKIP: S3_VECTOR_BUCKET and S3_VECTOR_INDEX must be non-empty");
        return Ok(());
    }
    let region = env::var("AWS_REGION").unwrap_or_else(|_| "us-east-1".to_string());

    // Configure S3 Vector Backend
    let config = S3VectorBackendConfig::new(
        &bucket_name,
        &index_name,
        384, // Common dimension for sentence transformers
        None,
        None,
        None,
    )
    .with_distance_metric(Metric::Cosine)
    .with_region(&region)
    .with_batch_size(100);

    outputln!("Connecting to S3 Vector Bucket:");
    outputln!("  Bucket: {bucket_name}");
    outputln!("  Index: {index_name}");
    outputln!("  Region: {region}");
    outputln!("  Dimension: 384");
    outputln!();

    // Create backend
    let backend: Arc<dyn Backend> = Arc::new(S3VectorBackend::new(config).await?);

    // Example 1: Insert individual vectors
    outputln!("--- Example 1: Insert Vectors ---");
    insert_example_vectors(&backend).await?;

    // Example 2: Bulk insert
    outputln!("\n--- Example 2: Bulk Insert ---");
    bulk_insert_example(&backend).await?;

    // Example 3: Similarity search
    outputln!("\n--- Example 3: Similarity Search ---");
    search_example(&backend).await?;

    // Example 4: Advanced search with metadata filtering
    outputln!("\n--- Example 4: Metadata Filtering ---");
    filtered_search_example(&backend).await?;

    // Example 5: Namespace management
    outputln!("\n--- Example 5: Namespace Management ---");
    namespace_example(&backend).await?;

    outputln!("\n=== Examples Complete ===");
    Ok(())
}

/// Insert individual vectors with metadata
async fn insert_example_vectors(
    backend: &Arc<dyn Backend>,
) -> Result<(), Box<dyn std::error::Error>> {
    let documents = [
        ("Machine learning is transforming software", "tech", 5),
        ("Climate change requires urgent action", "environment", 8),
        ("Quantum computing shows great promise", "tech", 7),
        ("Biodiversity loss threatens ecosystems", "environment", 9),
        ("Neural networks power modern AI", "tech", 6),
    ];

    let mut chunks = Vec::new();
    for (i, (text, category, priority)) in documents.iter().enumerate() {
        // Generate a sample embedding (in production, use a real embedding model)
        let embedding = generate_sample_embedding(text, 384);

        let chunk = Chunk::new(
            format!("doc-{i}"),
            0,
            *text,
            embedding,
            0,
            text.len(),
            1,
            json!({
                "category": category,
                "priority": priority,
                "indexed_at": OffsetDateTime::now_utc().unix_timestamp(),
            }),
            Some("articles".to_string()),
        );
        chunks.push(chunk);
    }

    backend.upsert(&chunks).await?;
    for (text, _, _) in &documents {
        outputln!("Inserted: {text}");
    }

    outputln!("\nInserted {} vectors", documents.len());
    Ok(())
}

/// Bulk insert multiple vectors
async fn bulk_insert_example(backend: &Arc<dyn Backend>) -> Result<(), Box<dyn std::error::Error>> {
    let mut chunks = Vec::new();

    for i in 0..20 {
        let text = format!("Bulk document {i}");
        let embedding = generate_sample_embedding(&text, 384);
        let chunk = Chunk::new(
            format!("bulk-{i}"),
            0,
            &text,
            embedding,
            0,
            text.len(),
            1,
            json!({
                "batch": "demo",
                "index": i,
            }),
            Some("bulk".to_string()),
        );
        chunks.push(chunk);
    }

    backend.upsert(&chunks).await?;
    outputln!("Bulk inserted 20 vectors");

    let count = backend.count(Some("bulk")).await?;
    outputln!("Total vectors in 'bulk' namespace: {count}");

    Ok(())
}

/// Perform similarity search
async fn search_example(backend: &Arc<dyn Backend>) -> Result<(), Box<dyn std::error::Error>> {
    // Create a query vector
    let query_vector =
        generate_sample_embedding("artificial intelligence and machine learning", 384);

    outputln!("Searching for similar vectors...");
    let query = SearchQuery {
        vector: query_vector.into(),
        limit: 5,
        metric: Metric::Cosine,
        namespace: Some("articles".to_string()),
        filter: None,
        hybrid_weight: None,
        sparse_vector: None,
        include_content: true,
        include_vectors: false,
        query_multi_vectors: None,
        maxsim: None,
    };
    let results = backend.search(&query).await?;

    outputln!("\nTop 5 similar vectors:");
    for (i, result) in results.iter().enumerate() {
        let text = result.chunk.content.as_str();
        outputln!("  {}. Score: {:.4} - {}", i + 1, result.score, text);
    }

    Ok(())
}

/// Advanced search with metadata filtering
async fn filtered_search_example(
    backend: &Arc<dyn Backend>,
) -> Result<(), Box<dyn std::error::Error>> {
    let query_vector = generate_sample_embedding("environmental issues", 384);

    outputln!("Searching with metadata filter (category = 'environment')...");
    let filter = Filter::Equal {
        field: "category".to_string(),
        value: json!("environment"),
    };
    let query = SearchQuery {
        vector: query_vector.into(),
        limit: 10,
        metric: Metric::Cosine,
        namespace: Some("articles".to_string()),
        filter: Some(filter),
        hybrid_weight: None,
        sparse_vector: None,
        include_content: true,
        include_vectors: false,
        query_multi_vectors: None,
        maxsim: None,
    };
    let all_results = backend.search(&query).await?;

    // Filter by min_score client-side
    let results: Vec<_> = all_results.into_iter().filter(|r| r.score >= 0.5).collect();

    outputln!("\nFiltered results:");
    for (i, result) in results.iter().enumerate() {
        let text = result.chunk.content.as_str();
        let priority = result
            .chunk
            .metadata
            .get("priority")
            .and_then(serde_json::Value::as_i64)
            .unwrap_or(0);

        outputln!(
            "  {}. Score: {:.4}, Priority: {} - {}",
            i + 1,
            result.score,
            priority,
            text
        );
    }

    Ok(())
}

/// Demonstrate namespace operations
async fn namespace_example(backend: &Arc<dyn Backend>) -> Result<(), Box<dyn std::error::Error>> {
    // Insert into different namespaces
    for ns in ["ns-a", "ns-b", "ns-c"] {
        let text = format!("Document in {ns}");
        let embedding = generate_sample_embedding(&text, 384);
        let chunk = Chunk::new(
            "doc-1",
            0,
            &text,
            embedding,
            0,
            text.len(),
            1,
            json!({"namespace": ns}),
            Some(ns.to_string()),
        );

        backend.upsert(&[chunk]).await?;
    }

    outputln!("Created vectors in multiple namespaces");

    // Count vectors per namespace
    for ns in &["ns-a", "ns-b", "ns-c", "articles", "bulk"] {
        let count = backend.count(Some(ns)).await?;
        outputln!("  Namespace '{ns}': {count} vectors");
    }

    // List all namespaces
    let namespaces = backend.namespaces().await?;
    outputln!("\nAll namespaces: {namespaces:?}");

    // Cleanup example namespace
    outputln!("\nCleaning up test namespace 'ns-a'...");
    backend.delete_namespace("ns-a").await?;
    let count = backend.count(Some("ns-a")).await?;
    outputln!("Namespace 'ns-a' cleared (count: {count})");

    Ok(())
}

/// Generate a sample embedding (deterministic for demo purposes)
/// In production, use a real embedding model like sentence-transformers
fn generate_sample_embedding(text: &str, dimension: usize) -> Vec<f32> {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let mut hasher = DefaultHasher::new();
    text.hash(&mut hasher);
    let seed = hasher.finish();

    let mut embedding = Vec::with_capacity(dimension);
    let mut rng_state = seed;

    for _ in 0..dimension {
        // Simple LCG for deterministic pseudo-random numbers
        rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
        let value = ((rng_state >> 32) as f32 / u32::MAX as f32).mul_add(2.0, -1.0);
        embedding.push(value);
    }

    // Normalize to unit vector
    let magnitude: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
    if magnitude > 0.0 {
        for val in &mut embedding {
            *val /= magnitude;
        }
    }

    embedding
}
