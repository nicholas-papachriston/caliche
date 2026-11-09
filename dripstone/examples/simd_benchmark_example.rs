//! Example demonstrating SIMD performance improvements over scalar implementations.
//!
//! This example benchmarks various SIMD-accelerated operations and compares
//! them against scalar implementations to show the performance benefits.
//! cargo run -p dripstone --example `simd_benchmark_example` --features="simd"

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

use dripstone::simd;
use std::hint::black_box;
use std::time::Instant;

/// Scalar implementation of dot product for comparison
fn scalar_dot_product(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// Scalar implementation of euclidean distance squared
fn scalar_euclidean_distance_squared(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| {
            let diff = x - y;
            diff * diff
        })
        .sum()
}

/// Scalar implementation of L2 normalization
fn scalar_l2_normalize(vector: &mut [f32]) {
    let mag: f32 = vector.iter().map(|v| v * v).sum::<f32>().sqrt();
    if mag > f32::EPSILON {
        for val in vector.iter_mut() {
            *val /= mag;
        }
    }
}

/// Scalar implementation of vector addition
fn scalar_vector_add(a: &[f32], b: &[f32], out: &mut [f32]) {
    for ((&ai, &bi), outi) in a.iter().zip(b.iter()).zip(out.iter_mut()) {
        *outi = ai + bi;
    }
}

/// Scalar implementation of min/max finding
fn scalar_find_min_max(values: &[f32]) -> (f32, f32) {
    if values.is_empty() {
        return (f32::INFINITY, f32::NEG_INFINITY);
    }
    let mut min = f32::MAX;
    let mut max = f32::MIN;
    for &val in values {
        min = min.min(val);
        max = max.max(val);
    }
    (min, max)
}

struct BenchmarkResult {
    operation: String,
    dimension: usize,
    iterations: usize,
    scalar_time_ns: u64,
    simd_time_ns: u64,
    speedup: f64,
}

fn benchmark_operation<F, G>(
    name: &str,
    dimension: usize,
    iterations: usize,
    mut scalar_fn: F,
    mut simd_fn: G,
) -> BenchmarkResult
where
    F: FnMut(),
    G: FnMut(),
{
    // Warmup
    for _ in 0..100 {
        scalar_fn();
        simd_fn();
    }

    // Benchmark scalar
    let start = Instant::now();
    for _ in 0..iterations {
        scalar_fn();
    }
    let scalar_time = start.elapsed();

    // Benchmark SIMD
    let start = Instant::now();
    for _ in 0..iterations {
        simd_fn();
    }
    let simd_time = start.elapsed();

    let scalar_ns = scalar_time.as_nanos() as u64;
    let simd_ns = simd_time.as_nanos() as u64;
    let speedup = scalar_ns as f64 / simd_ns as f64;

    BenchmarkResult {
        operation: name.to_string(),
        dimension,
        iterations,
        scalar_time_ns: scalar_ns,
        simd_time_ns: simd_ns,
        speedup,
    }
}

fn print_result(result: &BenchmarkResult) {
    outputln!(
        "  {} (dim={}, {} iterations):",
        result.operation,
        result.dimension,
        result.iterations
    );
    outputln!(
        "    Scalar: {:.2}µs ({:.2}ns/op)",
        result.scalar_time_ns as f64 / 1000.0,
        result.scalar_time_ns as f64 / result.iterations as f64
    );
    outputln!(
        "    SIMD:   {:.2}µs ({:.2}ns/op)",
        result.simd_time_ns as f64 / 1000.0,
        result.simd_time_ns as f64 / result.iterations as f64
    );
    outputln!("    Speedup: {:.2}x", result.speedup);
    outputln!();
}

/// Prints runtime SIMD capability so target host can be validated:
/// `x86_64`: AVX-512/AVX2 detection; aarch64: NEON path (always available).
fn print_runtime_simd_status() {
    outputln!("--- Runtime SIMD validation ---");
    #[cfg(target_arch = "x86_64")]
    {
        let avx512 = std::arch::is_x86_feature_detected!("avx512f");
        let avx2 = std::arch::is_x86_feature_detected!("avx2");
        outputln!(
            "  target_arch: x86_64, AVX-512 detected: {}, AVX2 detected: {} (SIMD path: {})",
            avx512,
            avx2,
            if avx512 {
                "AVX-512"
            } else if avx2 {
                "AVX2"
            } else {
                "scalar fallback"
            }
        );
    }
    #[cfg(target_arch = "aarch64")]
    {
        outputln!("  target_arch: aarch64, NEON path: enabled (always available)");
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        outputln!(
            "  target_arch: {} (no SIMD intrinsics; scalar path only)",
            std::env::consts::ARCH
        );
    }
    outputln!();
}

#[expect(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "benchmark example panics on SIMD kernel failures for brevity"
)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    outputln!("=== SIMD vs Scalar Performance Benchmark ===\n");
    print_runtime_simd_status();

    // Test with larger vectors to better demonstrate SIMD benefits
    // Larger dimensions help amortize SIMD overhead and reduce remainder handling impact
    let dimensions = vec![512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072];

    // Adjust iterations based on dimension to keep benchmark runtime reasonable
    let base_iterations = 10_000;
    let get_iterations = |dim: usize| -> usize {
        match dim {
            d if d <= 8192 => base_iterations,
            d if d <= 16384 => base_iterations / 2,
            d if d <= 32768 => base_iterations / 4,
            d if d <= 65536 => base_iterations / 8,
            _ => base_iterations / 16,
        }
    };

    // Benchmark dot product
    outputln!("--- Dot Product ---");
    for &dim in &dimensions {
        let iterations = get_iterations(dim);
        let a: Vec<f32> = (0..dim).map(|i| (i as f32 / dim as f32).sin()).collect();
        let b: Vec<f32> = (0..dim).map(|i| (i as f32 / dim as f32).cos()).collect();

        let result = benchmark_operation(
            "dot_product",
            dim,
            iterations,
            || {
                black_box(scalar_dot_product(black_box(&a), black_box(&b)));
            },
            || {
                black_box(simd::dot_product(black_box(&a), black_box(&b)).unwrap());
            },
        );
        print_result(&result);
    }

    // Benchmark euclidean distance squared
    outputln!("--- Euclidean Distance Squared ---");
    for &dim in &dimensions {
        let iterations = get_iterations(dim);
        let a: Vec<f32> = (0..dim).map(|i| (i as f32 / dim as f32).sin()).collect();
        let b: Vec<f32> = (0..dim).map(|i| (i as f32 / dim as f32).cos()).collect();

        let result = benchmark_operation(
            "euclidean_distance_squared",
            dim,
            iterations,
            || {
                black_box(scalar_euclidean_distance_squared(
                    black_box(&a),
                    black_box(&b),
                ));
            },
            || {
                black_box(simd::euclidean_distance_squared(black_box(&a), black_box(&b)).unwrap());
            },
        );
        print_result(&result);
    }

    // Benchmark cosine similarity
    outputln!("--- Cosine Similarity ---");
    for &dim in &dimensions {
        let iterations = get_iterations(dim);
        let a: Vec<f32> = (0..dim).map(|i| (i as f32 / dim as f32).sin()).collect();
        let b: Vec<f32> = (0..dim).map(|i| (i as f32 / dim as f32).cos()).collect();

        let result = benchmark_operation(
            "cosine_similarity",
            dim,
            iterations,
            || {
                let dot = scalar_dot_product(black_box(&a), black_box(&b));
                let mag_a: f32 = black_box(&a).iter().map(|v| v * v).sum::<f32>().sqrt();
                let mag_b: f32 = black_box(&b).iter().map(|v| v * v).sum::<f32>().sqrt();
                black_box(dot / (mag_a * mag_b));
            },
            || {
                black_box(simd::cosine_similarity(black_box(&a), black_box(&b)).unwrap());
            },
        );
        print_result(&result);
    }

    // Benchmark L2 normalization
    outputln!("--- L2 Normalization ---");
    for &dim in &dimensions {
        let iterations = get_iterations(dim);
        let base_vec: Vec<f32> = (0..dim).map(|i| (i as f32 / dim as f32).sin()).collect();

        let result = benchmark_operation(
            "l2_normalize",
            dim,
            iterations,
            || {
                let mut scalar_vec = base_vec.clone();
                scalar_l2_normalize(black_box(&mut scalar_vec));
                black_box(&mut scalar_vec);
            },
            || {
                let mut simd_vec = base_vec.clone();
                simd::l2_normalize(black_box(&mut simd_vec)).unwrap();
                black_box(());
                black_box(&mut simd_vec);
            },
        );
        print_result(&result);
    }

    // Benchmark vector addition
    outputln!("--- Vector Addition ---");
    for &dim in &dimensions {
        let iterations = get_iterations(dim);
        let a: Vec<f32> = (0..dim).map(|i| (i as f32 / dim as f32).sin()).collect();
        let b: Vec<f32> = (0..dim).map(|i| (i as f32 / dim as f32).cos()).collect();

        let result = benchmark_operation(
            "vector_add",
            dim,
            iterations,
            || {
                let mut scalar_out = vec![0.0; dim];
                scalar_vector_add(black_box(&a), black_box(&b), black_box(&mut scalar_out));
                black_box(&mut scalar_out);
            },
            || {
                let mut simd_out = vec![0.0; dim];
                simd::vector_add(black_box(&a), black_box(&b), black_box(&mut simd_out))
                    .expect("SIMD vector add should succeed");
                black_box(());
                black_box(&mut simd_out);
            },
        );
        print_result(&result);
    }

    // Benchmark min/max finding
    outputln!("--- Min/Max Finding ---");
    for &dim in &dimensions {
        let iterations = get_iterations(dim);
        let values: Vec<f32> = (0..dim).map(|i| (i as f32 / dim as f32).sin()).collect();

        let result = benchmark_operation(
            "find_min_max",
            dim,
            iterations,
            || {
                black_box(scalar_find_min_max(black_box(&values)));
            },
            || {
                black_box(simd::find_min_max(black_box(&values)));
            },
        );
        print_result(&result);
    }

    // Benchmark batch operations
    outputln!("--- Batch Operations ---");
    let num_candidates = 500; // Increased for better batch performance demonstration
    for &dim in &dimensions {
        let iterations = get_iterations(dim) / 20; // Fewer iterations for batch ops (they're more expensive)
        let query: Vec<f32> = (0..dim).map(|i| (i as f32 / dim as f32).sin()).collect();
        let candidates: Vec<Vec<f32>> = (0..num_candidates)
            .map(|i| {
                (0..dim)
                    .map(|j| ((i + j) as f32 / dim as f32).cos())
                    .collect()
            })
            .collect();
        let candidate_refs: Vec<&[f32]> = candidates.iter().map(std::vec::Vec::as_slice).collect();

        // Scalar batch dot products
        let scalar_batch_fn = || {
            for candidate in &candidates {
                black_box(scalar_dot_product(black_box(&query), black_box(candidate)));
            }
        };

        // SIMD batch dot products
        let simd_batch_fn = || {
            black_box(
                simd::batch_dot_products(black_box(&query), black_box(&candidate_refs)).unwrap(),
            );
        };

        let result = benchmark_operation(
            &format!("batch_dot_products ({num_candidates} candidates)"),
            dim,
            iterations,
            scalar_batch_fn,
            simd_batch_fn,
        );
        print_result(&result);
    }

    Ok(())
}
