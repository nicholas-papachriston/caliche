//! Example demonstrating Product Quantization for vector compression.

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

use dripstone::quantization::ProductQuantizer;
use std::time::Instant;

fn create_test_vectors(count: usize, dimension: usize) -> Vec<Vec<f32>> {
    (0..count)
        .map(|i| {
            (0..dimension)
                .map(|j| ((i * dimension + j) as f32 / 1000.0).sin())
                .collect()
        })
        .collect()
}

#[expect(
    clippy::expect_used,
    reason = "example code panics on encoding failures for brevity"
)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    outputln!("=== Product Quantization Example ===\n");

    // Create a Product Quantizer
    // M=8: Split vectors into 8 sub-vectors
    // K=256: Use 256 centroids per sub-vector (8 bits per sub-vector)
    // Dimension: 128 (must be divisible by M)
    let mut pq = ProductQuantizer::new(8, 256, 128)?;

    outputln!("Configuration:");
    outputln!("  M (sub-vectors): {}", pq.m());
    outputln!("  K (centroids per sub-vector): {}", pq.k());
    outputln!("  Original dimension: {}", pq.original_dim());
    outputln!("  Compression ratio: {:.1}x", pq.compression_ratio());
    outputln!();

    // Train the quantizer
    outputln!("Training quantizer on 10,000 vectors...");
    let training_vectors = create_test_vectors(10_000, 128);

    let start = Instant::now();
    pq.train(&training_vectors, 20)?;
    let train_time = start.elapsed();

    outputln!("Training completed in {:.2}ms", train_time.as_millis());
    outputln!();

    // Demonstrate encoding and decoding
    outputln!("--- Encoding and Decoding ---");
    let test_vector = &training_vectors[0];
    outputln!("Original vector:");
    outputln!("  Dimension: {}", test_vector.len());
    outputln!(
        "  Size: {} bytes ({} * 4 bytes)",
        test_vector.len() * 4,
        test_vector.len()
    );
    outputln!(
        "  Sample values: [{:.3}, {:.3}, {:.3}, ...]",
        test_vector[0],
        test_vector[1],
        test_vector[2]
    );

    let start = Instant::now();
    let codes = pq.encode(test_vector)?;
    let encode_time = start.elapsed();

    outputln!("\nEncoded codes:");
    outputln!("  Length: {} bytes", codes.len());
    outputln!(
        "  Compression: {:.1}x",
        (test_vector.len() * 4) as f32 / codes.len() as f32
    );
    outputln!("  Encoding time: {:.2}µs", encode_time.as_micros());
    outputln!(
        "  Sample codes: [{}, {}, {}, ...]",
        codes[0],
        codes[1],
        codes[2]
    );

    let start = Instant::now();
    let reconstructed = pq.decode(&codes)?;
    let decode_time = start.elapsed();

    outputln!("\nReconstructed vector:");
    outputln!("  Dimension: {}", reconstructed.len());
    outputln!("  Decoding time: {:.2}µs", decode_time.as_micros());
    outputln!(
        "  Sample values: [{:.3}, {:.3}, {:.3}, ...]",
        reconstructed[0],
        reconstructed[1],
        reconstructed[2]
    );

    // Calculate reconstruction error
    let error: f32 = test_vector
        .iter()
        .zip(reconstructed.iter())
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f32>()
        .sqrt();
    outputln!("  Reconstruction error (L2): {error:.6}");

    // Demonstrate approximate distance calculation
    outputln!("\n--- Approximate Distance Calculation ---");
    let query_vector = &training_vectors[100];

    // Calculate exact distance
    let exact_distance: f32 = test_vector
        .iter()
        .zip(query_vector.iter())
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f32>()
        .sqrt();

    // Calculate approximate distance using PQ codes
    let query_codes = pq.encode(query_vector)?;
    let start = Instant::now();
    let approx_distance = pq.approximate_distance(test_vector, &query_codes)?;
    let approx_time = start.elapsed();

    outputln!("Exact L2 distance: {exact_distance:.6}");
    outputln!("Approximate distance: {approx_distance:.6}");
    outputln!(
        "Error: {:.2}%",
        ((approx_distance - exact_distance).abs() / exact_distance) * 100.0
    );
    outputln!(
        "Approx distance calculation time: {:.2}µs",
        approx_time.as_micros()
    );
    outputln!("  (Much faster than exact calculation for large datasets)");

    // Demonstrate memory savings
    outputln!("\n--- Memory Savings ---");
    let num_vectors = 1_000_000;
    let original_memory_mb = f64::from(num_vectors * 128 * 4) / (1024.0 * 1024.0);
    let compressed_memory_mb = f64::from(num_vectors * 8) / (1024.0 * 1024.0);

    outputln!("For {num_vectors} vectors:");
    outputln!("  Original memory: {original_memory_mb:.2} MB");
    outputln!("  Compressed memory: {compressed_memory_mb:.2} MB");
    outputln!(
        "  Savings: {:.1}%",
        (1.0 - compressed_memory_mb / original_memory_mb) * 100.0
    );

    // Batch encoding example
    outputln!("\n--- Batch Encoding ---");
    let batch_size = 1000;
    let batch_vectors = &training_vectors[..batch_size];

    let start = Instant::now();
    let _batch_codes: Vec<Vec<u8>> = batch_vectors
        .iter()
        .map(|v| pq.encode(v).expect("encoding should succeed"))
        .collect();
    let batch_encode_time = start.elapsed();

    outputln!(
        "Encoded {} vectors in {:.2}ms",
        batch_size,
        batch_encode_time.as_millis()
    );
    outputln!(
        "  Rate: {:.0} vectors/sec",
        batch_size as f64 / batch_encode_time.as_secs_f64()
    );

    Ok(())
}
