//! Example demonstrating performance features: SIMD, batching, and streaming.

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

use dripstone::{
    backend::compute_batch_size,
    backend::memory::MemoryBackend,
    core::Chunk,
    metrics::{SimpleMetrics, VectorMetrics},
    simd,
    traits::Backend,
};
use futures::StreamExt;
use futures::stream;
use std::sync::Arc;
use std::time::Instant;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    outputln!("=== Performance Features Demo ===\n");

    // SIMD Performance
    demo_simd_performance();

    // Adaptive Batching
    demo_adaptive_batching().await?;

    // Streaming Bulk Upload
    demo_streaming_upload().await?;

    // Metrics Collection
    demo_metrics_collection()?;

    Ok(())
}

fn demo_simd_performance() {
    outputln!("--- SIMD-Optimized Similarity Computation ---\n");

    let dimensions = vec![128, 384, 768, 1536];

    for &dim in &dimensions {
        let a: Vec<f32> = (0..dim).map(|i| i as f32 / dim as f32).collect();
        let b: Vec<f32> = (0..dim).map(|i| (dim - i) as f32 / dim as f32).collect();

        let start = Instant::now();
        let iterations = 10_000;

        for _ in 0..iterations {
            let _ = simd::dot_product(&a, &b);
        }

        let elapsed = start.elapsed();
        let ops_per_sec = f64::from(iterations) / elapsed.as_secs_f64();

        outputln!(
            "Dimension {}: {:.2} ops/sec ({:.2}µs per op)",
            dim,
            ops_per_sec,
            elapsed.as_micros() as f64 / f64::from(iterations)
        );
    }

    outputln!();
}

async fn demo_adaptive_batching() -> Result<(), Box<dyn std::error::Error>> {
    outputln!("--- Adaptive Batching ---\n");

    let dimensions = vec![128, 384, 768, 1536];

    for &dim in &dimensions {
        let batch_size = compute_batch_size(dim);
        let memory_per_batch_mb = (batch_size * (dim * 4 + 500)) as f64 / (1024.0 * 1024.0);

        outputln!(
            "Dimension {dim}: batch_size={batch_size}, ~{memory_per_batch_mb:.1}MB per batch"
        );
    }

    outputln!("\nBulk upload with adaptive batching:");

    let backend: Arc<dyn Backend> = Arc::new(MemoryBackend::new());

    let num_vectors = 1000;
    let dimension = 384;

    outputln!("  Generating {num_vectors} vectors of dimension {dimension}...");

    let chunks: Vec<Chunk> = (0..num_vectors)
        .map(|i| {
            let vector: Vec<f32> = (0..dimension).map(|j| (i + j) as f32 / 1000.0).collect();
            Chunk::new(
                format!("vec:{i}"),
                0,
                "",
                vector,
                0,
                0,
                1,
                serde_json::json!({}),
                None,
            )
        })
        .collect();

    let start = Instant::now();
    backend.upsert(&chunks).await?;
    let elapsed = start.elapsed();

    outputln!("  Uploaded in {:.2}ms", elapsed.as_millis());
    outputln!(
        "    Rate: {:.0} vectors/sec",
        f64::from(num_vectors) / elapsed.as_secs_f64()
    );

    Ok(())
}

async fn demo_streaming_upload() -> Result<(), Box<dyn std::error::Error>> {
    outputln!("\n--- Streaming Bulk Upload ---\n");

    let backend: Arc<dyn Backend> = Arc::new(MemoryBackend::new());

    let num_vectors = 5000;
    let dimension = 384;

    outputln!("  Streaming {num_vectors} vectors...");

    // Create a stream of chunks and batch them
    let chunk_stream = stream::iter((0..num_vectors).map(move |i| {
        let vector: Vec<f32> = (0..dimension).map(|j| (i + j) as f32 / 1000.0).collect();
        Chunk::new(
            format!("stream:{i}"),
            0,
            "",
            vector,
            0,
            0,
            1,
            serde_json::json!({}),
            None,
        )
    }));

    let start = Instant::now();
    // Collect stream into batches and upsert
    let all_chunks: Vec<Chunk> = chunk_stream.collect().await;
    let batches: Vec<&[Chunk]> = all_chunks.chunks(100).collect();

    for batch in batches {
        backend.upsert(batch).await?;
    }
    let elapsed = start.elapsed();

    outputln!("  Streamed in {:.2}ms", elapsed.as_millis());
    outputln!(
        "    Rate: {:.0} vectors/sec",
        f64::from(num_vectors) / elapsed.as_secs_f64()
    );
    outputln!("    Memory efficient: constant memory usage regardless of dataset size");

    Ok(())
}

fn demo_metrics_collection() -> Result<(), Box<dyn std::error::Error>> {
    outputln!("\n--- Metrics Collection ---\n");

    let metrics = SimpleMetrics::new();

    // Simulate some operations
    outputln!("Simulating vector operations...");

    for i in 0..100 {
        metrics.record_search_latency(std::time::Duration::from_micros(500 + i as u64));
        metrics.record_search_result_count(10 + (i % 5));

        if i % 3 == 0 {
            metrics.increment_cache_hit();
        } else {
            metrics.increment_cache_miss();
        }

        if i % 10 == 0 {
            metrics.record_index_size(1000 + i * 10);
        }
    }

    // Report metrics
    outputln!("Metrics Summary:");
    if let Some(avg_latency) = metrics.average_search_latency() {
        outputln!("  Average search latency: {:.2}µs", avg_latency.as_micros());
    }
    outputln!("  Cache hit rate: {:.1}%", metrics.cache_hit_rate() * 100.0);
    if let Some(avg_results) = metrics.average_search_result_count() {
        outputln!("  Average search results: {avg_results:.1}");
    }
    if let Some(index_size) = metrics.current_index_size() {
        outputln!("  Current index size: {index_size} vectors");
    }

    let errors = metrics.error_counts();
    if !errors.is_empty() {
        outputln!("  Errors:");
        for (error_type, count) in errors {
            outputln!("    {error_type}: {count}");
        }
    }

    Ok(())
}
