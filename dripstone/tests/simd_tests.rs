//! Integration tests for SIMD operations across different vector sizes.
//!
//! These tests verify that SIMD operations work correctly for common
//! embedding dimensions used in production (128, 384, 768, 1536).

use dripstone::simd;

const TEST_SIZES: &[usize] = &[128, 384, 768, 1536];

#[test]
fn test_dot_product_various_sizes() {
    for &size in TEST_SIZES {
        let a: Vec<f32> = (0..size).map(|i| i as f32 / size as f32).collect();
        let b: Vec<f32> = (0..size).map(|i| (size - i) as f32 / size as f32).collect();

        let result = simd::dot_product(&a, &b).expect("dot_product should succeed");

        // Verify result is reasonable (not NaN, not infinite)
        assert!(result.is_finite(), "size {size}: result should be finite");

        // Compute expected result manually for verification
        let expected: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let diff = (result - expected).abs();

        // Allow small floating point error (relative to magnitude)
        let tolerance = expected.abs().mul_add(1e-5, 1e-6);
        assert!(
            diff < tolerance,
            "size {size}: result {result} differs from expected {expected} by {diff} (tolerance: {tolerance})"
        );
    }
}

#[test]
fn test_dot_product_orthogonal_vectors() {
    // Test orthogonal vectors (dot product should be ~0)
    for &size in TEST_SIZES {
        let mut a = vec![0.0; size];
        let mut b = vec![0.0; size];

        // Set alternating dimensions
        for i in (0..size).step_by(2) {
            a[i] = 1.0;
        }
        for i in (1..size).step_by(2) {
            b[i] = 1.0;
        }

        let result = simd::dot_product(&a, &b).expect("dot_product should succeed");

        assert!(
            result.abs() < 1e-5,
            "size {size}: orthogonal vectors should have dot product ~0, got {result}"
        );
    }
}

#[test]
fn test_cosine_similarity_various_sizes() {
    for &size in TEST_SIZES {
        let a: Vec<f32> = (0..size).map(|i| i as f32 / size as f32).collect();
        let b: Vec<f32> = (0..size).map(|i| (size - i) as f32 / size as f32).collect();

        let result = simd::cosine_similarity(&a, &b).expect("cosine_similarity should succeed");

        // Cosine similarity should be in range [-1, 1]
        assert!(
            (-1.0..=1.0).contains(&result),
            "size {size}: cosine similarity {result} should be in [-1, 1]"
        );
        assert!(result.is_finite(), "size {size}: result should be finite");
    }
}

#[test]
fn test_cosine_similarity_identical_vectors() {
    // Identical vectors should have cosine similarity of 1.0
    for &size in TEST_SIZES {
        let a: Vec<f32> = (0..size).map(|i| i as f32 / size as f32).collect();
        let b = a.clone();

        let result = simd::cosine_similarity(&a, &b).expect("cosine_similarity should succeed");

        assert!(
            (result - 1.0).abs() < 1e-5,
            "size {size}: identical vectors should have cosine similarity ~1.0, got {result}"
        );
    }
}

#[test]
fn test_cosine_similarity_opposite_vectors() {
    // Opposite vectors should have cosine similarity of -1.0
    for &size in TEST_SIZES {
        let a: Vec<f32> = (0..size).map(|i| (i + 1) as f32).collect();
        let b: Vec<f32> = a.iter().map(|x| -x).collect();

        let result = simd::cosine_similarity(&a, &b).expect("cosine_similarity should succeed");

        assert!(
            (result + 1.0).abs() < 1e-5,
            "size {size}: opposite vectors should have cosine similarity ~-1.0, got {result}"
        );
    }
}

#[test]
fn test_euclidean_distance_various_sizes() {
    for &size in TEST_SIZES {
        let a: Vec<f32> = (0..size).map(|i| i as f32 / size as f32).collect();
        let b: Vec<f32> = (0..size).map(|i| (size - i) as f32 / size as f32).collect();

        let result = simd::euclidean_distance(&a, &b).expect("euclidean_distance should succeed");

        // Euclidean distance should be non-negative
        assert!(
            result >= 0.0,
            "size {size}: euclidean distance {result} should be non-negative"
        );
        assert!(result.is_finite(), "size {size}: result should be finite");

        // Verify against manual calculation
        let expected: f32 = a
            .iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).powi(2))
            .sum::<f32>()
            .sqrt();

        let diff = (result - expected).abs();
        let tolerance = expected.mul_add(1e-5, 1e-6);

        assert!(
            diff < tolerance,
            "size {size}: result {result} differs from expected {expected} by {diff} (tolerance: {tolerance})"
        );
    }
}

#[test]
fn test_euclidean_distance_identical_vectors() {
    // Identical vectors should have distance 0
    for &size in TEST_SIZES {
        let a: Vec<f32> = (0..size).map(|i| i as f32 / size as f32).collect();
        let b = a.clone();

        let result = simd::euclidean_distance(&a, &b).expect("euclidean_distance should succeed");

        assert!(
            result < 1e-5,
            "size {size}: identical vectors should have distance ~0, got {result}"
        );
    }
}

#[test]
fn test_magnitude_various_sizes() {
    for &size in TEST_SIZES {
        let v: Vec<f32> = (0..size).map(|i| (i + 1) as f32 / size as f32).collect();

        let result = simd::magnitude(&v).expect("magnitude should succeed");

        // Magnitude should be non-negative
        assert!(
            result >= 0.0,
            "size {size}: magnitude {result} should be non-negative"
        );
        assert!(result.is_finite(), "size {size}: result should be finite");

        // Verify against manual calculation
        let expected: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();

        let diff = (result - expected).abs();
        let tolerance = expected.mul_add(1e-5, 1e-6);

        assert!(
            diff < tolerance,
            "size {size}: result {result} differs from expected {expected} by {diff} (tolerance: {tolerance})"
        );
    }
}

#[test]
fn test_magnitude_unit_vector() {
    // Unit vectors should have magnitude 1.0
    for &size in TEST_SIZES {
        let mut v = vec![0.0; size];
        v[0] = 1.0;

        let result = simd::magnitude(&v).expect("magnitude should succeed");

        assert!(
            (result - 1.0).abs() < 1e-5,
            "size {size}: unit vector should have magnitude ~1.0, got {result}"
        );
    }
}

#[test]
fn test_manhattan_distance_various_sizes() {
    for &size in TEST_SIZES {
        let a: Vec<f32> = (0..size).map(|i| i as f32 / size as f32).collect();
        let b: Vec<f32> = (0..size).map(|i| (size - i) as f32 / size as f32).collect();

        let result = simd::manhattan_distance(&a, &b).expect("manhattan_distance should succeed");

        // Manhattan distance should be non-negative
        assert!(
            result >= 0.0,
            "size {size}: manhattan distance {result} should be non-negative"
        );
        assert!(result.is_finite(), "size {size}: result should be finite");

        // Verify against manual calculation
        let expected: f32 = a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).sum();

        let diff = (result - expected).abs();
        let tolerance = expected.mul_add(1e-5, 1e-6);

        assert!(
            diff < tolerance,
            "size {size}: result {result} differs from expected {expected} by {diff} (tolerance: {tolerance})"
        );
    }
}

#[test]
fn test_manhattan_distance_identical_vectors() {
    // Identical vectors should have Manhattan distance 0
    for &size in TEST_SIZES {
        let a: Vec<f32> = (0..size).map(|i| i as f32 / size as f32).collect();
        let b = a.clone();

        let result = simd::manhattan_distance(&a, &b).expect("manhattan_distance should succeed");

        assert!(
            result < 1e-5,
            "size {size}: identical vectors should have manhattan distance ~0, got {result}"
        );
    }
}

#[test]
fn test_manhattan_distance_known_cases() {
    // Test with known Manhattan distance values
    let test_cases = vec![
        (vec![0.0, 0.0], vec![3.0, 4.0], 7.0),   // 3 + 4 = 7
        (vec![1.0, 2.0], vec![4.0, 6.0], 7.0),   // |1-4| + |2-6| = 3 + 4 = 7
        (vec![-1.0, -2.0], vec![1.0, 2.0], 6.0), // |(-1)-1| + |(-2)-2| = 2 + 4 = 6
    ];

    for (a, b, expected) in test_cases {
        let result = simd::manhattan_distance(&a, &b).expect("manhattan_distance should succeed");
        assert!(
            (result - expected).abs() < 1e-5,
            "manhattan distance between {a:?} and {b:?} should be ~{expected}, got {result}"
        );
    }
}

#[test]
fn test_chebyshev_distance_various_sizes() {
    for &size in TEST_SIZES {
        let a: Vec<f32> = (0..size).map(|i| i as f32 / size as f32).collect();
        let b: Vec<f32> = (0..size).map(|i| (size - i) as f32 / size as f32).collect();

        let result = simd::chebyshev_distance(&a, &b).expect("chebyshev_distance should succeed");

        // Chebyshev distance should be non-negative
        assert!(
            result >= 0.0,
            "size {size}: chebyshev distance {result} should be non-negative"
        );
        assert!(result.is_finite(), "size {size}: result should be finite");

        // Verify against manual calculation
        let expected: f32 = a
            .iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max);

        let diff = (result - expected).abs();
        let tolerance = expected.mul_add(1e-5, 1e-6);

        assert!(
            diff < tolerance,
            "size {size}: result {result} differs from expected {expected} by {diff} (tolerance: {tolerance})"
        );
    }
}

#[test]
fn test_chebyshev_distance_identical_vectors() {
    // Identical vectors should have Chebyshev distance 0
    for &size in TEST_SIZES {
        let a: Vec<f32> = (0..size).map(|i| i as f32 / size as f32).collect();
        let b = a.clone();

        let result = simd::chebyshev_distance(&a, &b).expect("chebyshev_distance should succeed");

        assert!(
            result < 1e-5,
            "size {size}: identical vectors should have chebyshev distance ~0, got {result}"
        );
    }
}

#[test]
fn test_chebyshev_distance_known_cases() {
    // Test with known Chebyshev distance values
    let test_cases = vec![
        (vec![0.0, 0.0], vec![3.0, 4.0], 4.0),   // max(3, 4) = 4
        (vec![1.0, 2.0], vec![4.0, 6.0], 4.0),   // max(|1-4|, |2-6|) = max(3, 4) = 4
        (vec![-1.0, -2.0], vec![1.0, 2.0], 4.0), // max(|(-1)-1|, |(-2)-2|) = max(2, 4) = 4
        (vec![5.0, 1.0], vec![1.0, 5.0], 4.0),   // max(|5-1|, |1-5|) = max(4, 4) = 4
    ];

    for (a, b, expected) in test_cases {
        let result = simd::chebyshev_distance(&a, &b).expect("chebyshev_distance should succeed");
        assert!(
            (result - expected).abs() < 1e-5,
            "chebyshev distance between {a:?} and {b:?} should be ~{expected}, got {result}"
        );
    }
}

#[test]
fn test_chebyshev_distance_max_difference() {
    // Test that Chebyshev distance correctly identifies the maximum difference
    for &size in TEST_SIZES {
        let mut a = vec![0.0; size];
        let mut b = vec![0.0; size];

        // Set one element to have a large difference
        let max_idx = size / 2;
        a[max_idx] = 0.0;
        b[max_idx] = 10.0;

        // Set other elements to have smaller differences
        for i in 0..size {
            if i != max_idx {
                a[i] = i as f32;
                b[i] = (i + 1) as f32; // Small difference of 1.0
            }
        }

        let result = simd::chebyshev_distance(&a, &b).expect("chebyshev_distance should succeed");

        // Should be the maximum difference (10.0)
        assert!(
            (result - 10.0).abs() < 1e-5,
            "size {size}: chebyshev distance should be ~10.0 (max difference), got {result}"
        );
    }
}

#[test]
fn test_simd_error_handling() {
    // Test mismatched lengths
    let a = vec![1.0, 2.0, 3.0];
    let b = vec![1.0, 2.0];

    assert!(
        simd::dot_product(&a, &b).is_err(),
        "dot_product should error on mismatched lengths"
    );
    assert!(
        simd::cosine_similarity(&a, &b).is_err(),
        "cosine_similarity should error on mismatched lengths"
    );
    assert!(
        simd::euclidean_distance(&a, &b).is_err(),
        "euclidean_distance should error on mismatched lengths"
    );
    assert!(
        simd::manhattan_distance(&a, &b).is_err(),
        "manhattan_distance should error on mismatched lengths"
    );
    assert!(
        simd::chebyshev_distance(&a, &b).is_err(),
        "chebyshev_distance should error on mismatched lengths"
    );
}

#[test]
fn test_simd_empty_vectors() {
    let a: Vec<f32> = vec![];
    let b: Vec<f32> = vec![];

    // Empty vectors should error or return 0
    let dot_result = simd::dot_product(&a, &b);
    assert!(
        dot_result.is_err() || dot_result.unwrap().abs() < 1e-10,
        "dot_product on empty vectors should error or be ~0"
    );

    // Manhattan and Chebyshev distances on empty vectors should error or return 0
    let manhattan_result = simd::manhattan_distance(&a, &b);
    assert!(
        manhattan_result.is_err() || manhattan_result.unwrap().abs() < 1e-10,
        "manhattan_distance on empty vectors should error or be ~0"
    );

    let chebyshev_result = simd::chebyshev_distance(&a, &b);
    assert!(
        chebyshev_result.is_err() || chebyshev_result.unwrap().abs() < 1e-10,
        "chebyshev_distance on empty vectors should error or be ~0"
    );
}

#[test]
fn test_manhattan_vs_euclidean_relationship() {
    // Manhattan distance should be >= Euclidean distance (triangle inequality)
    for &size in TEST_SIZES {
        let a: Vec<f32> = (0..size).map(|i| i as f32 / size as f32).collect();
        let b: Vec<f32> = (0..size).map(|i| (size - i) as f32 / size as f32).collect();

        let manhattan =
            simd::manhattan_distance(&a, &b).expect("manhattan_distance should succeed");
        let euclidean =
            simd::euclidean_distance(&a, &b).expect("euclidean_distance should succeed");

        assert!(
            manhattan >= euclidean,
            "size {size}: manhattan distance {manhattan} should be >= euclidean distance {euclidean}"
        );
    }
}

#[test]
fn test_chebyshev_vs_euclidean_relationship() {
    // Chebyshev distance should be <= Euclidean distance
    for &size in TEST_SIZES {
        let a: Vec<f32> = (0..size).map(|i| i as f32 / size as f32).collect();
        let b: Vec<f32> = (0..size).map(|i| (size - i) as f32 / size as f32).collect();

        let chebyshev =
            simd::chebyshev_distance(&a, &b).expect("chebyshev_distance should succeed");
        let euclidean =
            simd::euclidean_distance(&a, &b).expect("euclidean_distance should succeed");

        assert!(
            chebyshev <= euclidean,
            "size {size}: chebyshev distance {chebyshev} should be <= euclidean distance {euclidean}"
        );
    }
}

#[test]
fn test_distance_metrics_single_element() {
    // Test with single-element vectors
    let a = vec![5.0];
    let b = vec![2.0];

    let manhattan = simd::manhattan_distance(&a, &b).expect("manhattan_distance should succeed");
    let euclidean = simd::euclidean_distance(&a, &b).expect("euclidean_distance should succeed");
    let chebyshev = simd::chebyshev_distance(&a, &b).expect("chebyshev_distance should succeed");

    // For single element, all distances should be the same: |5 - 2| = 3
    assert!(
        (manhattan - 3.0).abs() < 1e-5,
        "manhattan distance should be 3.0"
    );
    assert!(
        (euclidean - 3.0).abs() < 1e-5,
        "euclidean distance should be 3.0"
    );
    assert!(
        (chebyshev - 3.0).abs() < 1e-5,
        "chebyshev distance should be 3.0"
    );
}

#[test]
fn test_simd_performance_sizes() {
    // Test common embedding dimensions from various models
    let sizes = vec![
        384,  // sentence-transformers/all-MiniLM-L6-v2
        768,  // BERT base
        1024, // OpenAI ada-002
        1536, // OpenAI text-embedding-3-small
        3072, // OpenAI text-embedding-3-large
    ];

    for size in sizes {
        let a: Vec<f32> = (0..size).map(|i| i as f32 / size as f32).collect();
        let b: Vec<f32> = (0..size).map(|i| (size - i) as f32 / size as f32).collect();

        // Just verify they complete successfully for these sizes
        let _ = simd::dot_product(&a, &b).expect("dot_product should work");
        let _ = simd::cosine_similarity(&a, &b).expect("cosine_similarity should work");
        let _ = simd::euclidean_distance(&a, &b).expect("euclidean_distance should work");
        let _ = simd::manhattan_distance(&a, &b).expect("manhattan_distance should work");
        let _ = simd::chebyshev_distance(&a, &b).expect("chebyshev_distance should work");
        let _ = simd::magnitude(&a).expect("magnitude should work");
    }
}

#[test]
fn test_simd_normalized_vectors() {
    // Test with normalized (unit) vectors
    for &size in TEST_SIZES {
        let mut a: Vec<f32> = (0..size).map(|i| (i + 1) as f32).collect();

        // Normalize to unit length
        let mag: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        for x in &mut a {
            *x /= mag;
        }

        let result_mag = simd::magnitude(&a).expect("magnitude should succeed");

        assert!(
            (result_mag - 1.0).abs() < 1e-4,
            "size {size}: normalized vector should have magnitude ~1.0, got {result_mag}"
        );
    }
}
