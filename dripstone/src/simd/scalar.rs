//! Scalar (non-SIMD) implementations for baseline performance.
#![cfg_attr(
    all(target_arch = "aarch64", feature = "simd"),
    expect(
        dead_code,
        reason = "NEON dispatch replaces most scalar paths on aarch64"
    )
)]

use crate::errors::DripstoneError;

/// Scalar dot product implementation (no SIMD).
/// Simple, portable, works on all architectures.
pub fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    let mut sum = 0.0_f32;
    for i in 0..a.len() {
        sum += a[i] * b[i];
    }
    sum
}

pub fn magnitude(values: &[f32]) -> Result<f32, DripstoneError> {
    let mut squared_sum = 0.0_f32;
    for &value in values {
        squared_sum += value * value;
    }
    let magnitude = squared_sum.sqrt();
    if magnitude.is_finite() {
        Ok(magnitude)
    } else {
        Err(DripstoneError::storage("failed to compute magnitude"))
    }
}

pub fn squared_sum(values: &[f32]) -> Result<f32, DripstoneError> {
    let mut sum = 0.0_f32;
    for &value in values {
        sum += value * value;
    }
    if sum.is_finite() {
        Ok(sum)
    } else {
        Err(DripstoneError::storage("failed to compute squared sum"))
    }
}

pub fn euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
    euclidean_distance_squared(a, b).sqrt()
}

/// Scalar Manhattan distance (L1 norm) implementation.
/// Sum of absolute differences between corresponding elements.
pub fn manhattan_distance(a: &[f32], b: &[f32]) -> f32 {
    let mut sum = 0.0_f32;
    for i in 0..a.len() {
        sum += (a[i] - b[i]).abs();
    }
    sum
}

/// Scalar Chebyshev distance (L∞ norm) implementation.
/// Maximum absolute difference between corresponding elements.
pub fn chebyshev_distance(a: &[f32], b: &[f32]) -> f32 {
    let mut max_distance = 0.0_f32;
    for i in 0..a.len() {
        max_distance = max_distance.max((a[i] - b[i]).abs());
    }
    max_distance
}

/// Scalar Euclidean distance squared implementation (no sqrt).
/// Faster than `euclidean_distance` when only comparing distances.
pub fn euclidean_distance_squared(a: &[f32], b: &[f32]) -> f32 {
    let mut sum = 0.0_f32;
    for i in 0..a.len() {
        let diff = a[i] - b[i];
        sum += diff * diff;
    }
    sum
}

/// Scalar batch Euclidean distances.
pub fn batch_euclidean_distances(query: &[f32], candidates: &[&[f32]]) -> Vec<f32> {
    let mut results = Vec::with_capacity(candidates.len());
    for candidate in candidates {
        results.push(euclidean_distance(query, candidate));
    }
    results
}

/// Scalar batch dot products.
pub fn batch_dot_products(query: &[f32], candidates: &[&[f32]]) -> Vec<f32> {
    let mut results = Vec::with_capacity(candidates.len());
    for candidate in candidates {
        results.push(dot_product(query, candidate));
    }
    results
}

/// Scalar batch cosine similarities.
pub fn batch_cosine_similarities(query: &[f32], candidates: &[&[f32]]) -> Vec<f32> {
    let Ok(query_mag) = magnitude(query) else {
        return vec![0.0; candidates.len()];
    };
    let mut results = Vec::with_capacity(candidates.len());
    for candidate in candidates {
        let dot = dot_product(query, candidate);
        let Ok(mag) = magnitude(candidate) else {
            results.push(0.0);
            continue;
        };
        results.push(super::cosine_or_zero(dot, query_mag, mag));
    }
    results
}

/// Scalar batch cosine similarities with precomputed query norm.
pub fn batch_cosine_with_query_norm(
    query: &[f32],
    query_norm: f32,
    candidates: &[&[f32]],
) -> Result<Vec<f32>, DripstoneError> {
    let mut results = Vec::with_capacity(candidates.len());
    for candidate in candidates {
        let dot = dot_product(query, candidate);
        let candidate_norm = match magnitude(candidate) {
            Ok(norm) => norm,
            Err(_) => {
                results.push(0.0);
                continue;
            }
        };
        results.push(super::cosine_or_zero(dot, query_norm, candidate_norm));
    }
    Ok(results)
}

/// Scalar L2 normalization (normalizes vector to unit length in-place).
pub fn l2_normalize(vector: &mut [f32]) -> Result<(), DripstoneError> {
    let mag = magnitude(vector)?;
    if !super::usable_cosine_norm(mag) {
        return Err(DripstoneError::storage(
            "cannot normalize zero-magnitude vector",
        ));
    }
    for val in vector.iter_mut() {
        *val /= mag;
    }
    Ok(())
}

/// Scalar vector addition: out = a + b.
pub fn vector_add(a: &[f32], b: &[f32], out: &mut [f32]) {
    for i in 0..a.len() {
        out[i] = a[i] + b[i];
    }
}

/// Scalar vector subtraction: out = a - b.
pub fn vector_sub(a: &[f32], b: &[f32], out: &mut [f32]) {
    for i in 0..a.len() {
        out[i] = a[i] - b[i];
    }
}

/// Scalar vector scaling: v *= scalar.
pub fn vector_scale(v: &mut [f32], scalar: f32) {
    for val in v.iter_mut() {
        *val *= scalar;
    }
}

/// Scalar vector accumulation: accumulator += v.
pub fn vector_accumulate(accumulator: &mut [f32], v: &[f32]) {
    for i in 0..accumulator.len() {
        accumulator[i] += v[i];
    }
}

/// Scalar min/max finding.
pub fn find_min_max(values: &[f32]) -> (f32, f32) {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scalar_dot_product() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        assert!((dot_product(&a, &b) - 32.0).abs() <= f32::EPSILON);
    }

    #[test]
    fn test_scalar_magnitude() {
        let v = vec![3.0, 4.0];
        let magnitude = magnitude(&v).expect("Failed to calculate magnitude");
        assert!((magnitude - 5.0).abs() <= f32::EPSILON);
    }

    #[test]
    fn test_scalar_euclidean() {
        let a = vec![0.0, 0.0];
        let b = vec![3.0, 4.0];
        assert!((euclidean_distance(&a, &b) - 5.0).abs() <= f32::EPSILON);
    }

    #[test]
    fn test_scalar_manhattan() {
        let a = vec![0.0, 0.0];
        let b = vec![3.0, 4.0];
        assert!((manhattan_distance(&a, &b) - 7.0).abs() <= f32::EPSILON);
    }

    #[test]
    fn test_scalar_chebyshev() {
        let a = vec![0.0, 0.0];
        let b = vec![3.0, 4.0];
        assert!((chebyshev_distance(&a, &b) - 4.0).abs() <= f32::EPSILON);
    }
}
