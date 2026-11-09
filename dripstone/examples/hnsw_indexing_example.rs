//! Example demonstrating HNSW indexing for fast approximate nearest neighbor search.
//! cargo run -p dripstone --example `hnsw_indexing_example` --features="memory-backend"

macro_rules! outputln {
    ($($arg:tt)*) => {{
        let mut output = std::io::stdout().lock();
        let _ = std::io::Write::write_fmt(&mut output, format_args!($($arg)*));
        let _ = std::io::Write::write_all(&mut output, b"\n");
    }};
}

use dripstone::embedding::{EmbeddingRecord, SimilarityMetric};
use dripstone::index::{HnswConfig, HnswIndex, VectorIndex};
use std::time::Instant;

fn create_test_records(count: usize, dimension: usize) -> Vec<EmbeddingRecord> {
    (0..count)
        .map(|i| {
            let vector: Vec<f32> = (0..dimension)
                .map(|j| ((i * dimension + j) as f32 / 1000.0).sin())
                .collect();
            EmbeddingRecord::new(format!("record_{i}"), vector)
        })
        .collect()
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    outputln!("=== HNSW Indexing Example ===\n");

    // Configure HNSW index
    let config = HnswConfig {
        m: 16,                 // Maximum connections per node
        ef_construction: 200,  // Construction-time search width
        ef_search: 50,         // Search-time candidate list size
        max_nb_connection: 16, // Maximum connections
        ..Default::default()
    };

    let dimension = 128;
    let index = HnswIndex::new(config, dimension);

    outputln!("Building index with 10,000 vectors...");
    let records = create_test_records(10_000, dimension);

    let start = Instant::now();
    index.build(&records).await?;
    let build_time = start.elapsed();

    outputln!("Index built in {:.2}ms", build_time.as_millis());
    outputln!("  Index size: {} vectors\n", index.len());

    // Perform searches
    let query: Vec<f32> = (0..dimension).map(|j| (j as f32 / 1000.0).sin()).collect();

    outputln!("Searching for 10 nearest neighbors...");
    let start = Instant::now();
    let results = index
        .search(&query, 10, SimilarityMetric::Euclidean)
        .await?;
    let search_time = start.elapsed();

    outputln!("Search completed in {:.2}µs", search_time.as_micros());
    outputln!("\nTop 10 results:");
    for (i, result) in results.iter().enumerate() {
        outputln!(
            "  {}. {} (distance: {:.4})",
            i + 1,
            result.record.id,
            result.score
        );
    }

    // Demonstrate incremental updates
    outputln!("\n--- Incremental Updates ---");
    let new_record = EmbeddingRecord::new(
        "new_record",
        (0..dimension)
            .map(|j| (j as f32 / 1000.0).cos())
            .collect::<Vec<f32>>(),
    );

    let start = Instant::now();
    index.add(new_record).await?;
    let add_time = start.elapsed();

    outputln!("Added new record in {:.2}µs", add_time.as_micros());
    outputln!("  Index size: {} vectors", index.len());

    // Demonstrate serialization (use async versions when running in async context)
    outputln!("\n--- Serialization ---");
    let start = Instant::now();
    let bytes = index.to_bytes_async().await?;
    let serialize_time = start.elapsed();

    outputln!("Serialized index in {:.2}ms", serialize_time.as_millis());
    outputln!("  Size: {:.2} KB", bytes.len() as f64 / 1024.0);

    let start = Instant::now();
    let restored = HnswIndex::from_bytes_async(&bytes).await?;
    let deserialize_time = start.elapsed();

    outputln!(
        "Deserialized index in {:.2}ms",
        deserialize_time.as_millis()
    );
    outputln!("  Restored size: {} vectors", restored.len());

    // Compare search performance
    outputln!("\n--- Performance Comparison ---");
    let query2: Vec<f32> = (0..dimension).map(|j| (j as f32 / 1000.0).cos()).collect();

    let start = Instant::now();
    let _results = index
        .search(&query2, 10, SimilarityMetric::Euclidean)
        .await?;
    let hnsw_time = start.elapsed();

    outputln!("HNSW search time: {:.2}µs", hnsw_time.as_micros());
    outputln!("  (HNSW provides sub-millisecond search even with 10K+ vectors)");

    Ok(())
}
