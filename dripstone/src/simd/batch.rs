use crate::errors::DripstoneError;

/// Computes cosine similarity with a precomputed query norm.
///
/// # Errors
///
/// Returns an error when the operation cannot be completed.
pub fn cosine_with_query_norm(
    query: &[f32],
    query_norm: f32,
    candidate: &[f32],
) -> Result<f32, DripstoneError> {
    super::ensure_same_length(query, candidate)?;
    let dot = super::dot_product(query, candidate)?;
    let candidate_norm = match super::magnitude(candidate) {
        Ok(norm) => norm,
        Err(_) => return Ok(0.0),
    };
    Ok(super::cosine_or_zero(dot, query_norm, candidate_norm))
}

/// Computes Euclidean distances between a query vector and multiple candidates.
///
/// # Errors
///
/// Returns an error when the operation cannot be completed.
pub fn batch_euclidean_distances(
    query: &[f32],
    candidates: &[&[f32]],
) -> Result<Vec<f32>, DripstoneError> {
    super::ensure_batch_candidate_lengths(query, candidates)?;

    #[cfg(all(target_arch = "x86_64", feature = "simd"))]
    {
        if is_x86_feature_detected!("avx512f") {
            return Ok(unsafe {
                super::avx512::batch_euclidean_distances_avx512(query, candidates)
            });
        }
        if is_x86_feature_detected!("avx2") {
            return Ok(unsafe { super::avx2::batch_euclidean_distances_avx2(query, candidates) });
        }
    }

    #[cfg(all(target_arch = "aarch64", feature = "simd"))]
    {
        Ok(unsafe { super::neon::batch_euclidean_distances_neon(query, candidates) })
    }

    #[cfg(not(all(feature = "simd", any(target_arch = "x86_64", target_arch = "aarch64"))))]
    {
        Ok(super::scalar::batch_euclidean_distances(query, candidates))
    }

    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    {
        Ok(super::scalar::batch_euclidean_distances(query, candidates))
    }
}

/// Computes dot products between a query vector and multiple candidates.
///
/// # Errors
///
/// Returns an error when the operation cannot be completed.
pub fn batch_dot_products(
    query: &[f32],
    candidates: &[&[f32]],
) -> Result<Vec<f32>, DripstoneError> {
    super::ensure_batch_candidate_lengths(query, candidates)?;

    #[cfg(all(target_arch = "x86_64", feature = "simd"))]
    {
        if is_x86_feature_detected!("avx512f") {
            return Ok(unsafe { super::avx512::batch_dot_products_avx512(query, candidates) });
        }
        if is_x86_feature_detected!("avx2") {
            return Ok(unsafe { super::avx2::batch_dot_products_avx2(query, candidates) });
        }
    }

    #[cfg(all(target_arch = "aarch64", feature = "simd"))]
    {
        Ok(unsafe { super::neon::batch_dot_products_neon(query, candidates) })
    }

    #[cfg(not(all(feature = "simd", any(target_arch = "x86_64", target_arch = "aarch64"))))]
    {
        Ok(super::scalar::batch_dot_products(query, candidates))
    }

    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    {
        Ok(super::scalar::batch_dot_products(query, candidates))
    }
}

/// Computes cosine similarities between a query vector and multiple candidates.
///
/// # Errors
///
/// Returns an error when the operation cannot be completed.
pub fn batch_cosine_similarities(
    query: &[f32],
    candidates: &[&[f32]],
) -> Result<Vec<f32>, DripstoneError> {
    super::ensure_batch_candidate_lengths(query, candidates)?;

    #[cfg(all(target_arch = "x86_64", feature = "simd"))]
    {
        if is_x86_feature_detected!("avx512f") {
            return Ok(unsafe {
                super::avx512::batch_cosine_similarities_avx512(query, candidates)
            });
        }
        if is_x86_feature_detected!("avx2") {
            return Ok(unsafe { super::avx2::batch_cosine_similarities_avx2(query, candidates) });
        }
    }

    #[cfg(all(target_arch = "aarch64", feature = "simd"))]
    {
        Ok(unsafe { super::neon::batch_cosine_similarities_neon(query, candidates) })
    }

    #[cfg(not(all(feature = "simd", any(target_arch = "x86_64", target_arch = "aarch64"))))]
    {
        Ok(super::scalar::batch_cosine_similarities(query, candidates))
    }

    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    {
        Ok(super::scalar::batch_cosine_similarities(query, candidates))
    }
}

/// Computes cosine similarities between a query vector and multiple candidates using
/// a precomputed query norm.
///
/// # Errors
///
/// Returns an error when the operation cannot be completed.
pub fn batch_cosine_with_query_norm(
    query: &[f32],
    query_norm: f32,
    candidates: &[&[f32]],
) -> Result<Vec<f32>, DripstoneError> {
    super::ensure_batch_candidate_lengths(query, candidates)?;

    #[cfg(all(target_arch = "x86_64", feature = "simd"))]
    {
        if is_x86_feature_detected!("avx512f") {
            return Ok(unsafe {
                super::avx512::batch_cosine_with_query_norm_avx512(query, query_norm, candidates)
            });
        }
        if is_x86_feature_detected!("avx2") {
            return Ok(unsafe {
                super::avx2::batch_cosine_with_query_norm_avx2(query, query_norm, candidates)
            });
        }
    }

    #[cfg(all(target_arch = "aarch64", feature = "simd"))]
    {
        Ok(
            unsafe {
                super::neon::batch_cosine_with_query_norm_neon(query, query_norm, candidates)
            },
        )
    }

    #[cfg(not(all(feature = "simd", any(target_arch = "x86_64", target_arch = "aarch64"))))]
    {
        super::scalar::batch_cosine_with_query_norm(query, query_norm, candidates)
    }

    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    {
        super::scalar::batch_cosine_with_query_norm(query, query_norm, candidates)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_batch_euclidean_distances() {
        let query = vec![0.0, 0.0];
        let candidates = vec![&[3.0, 4.0][..], &[5.0, 12.0][..]];
        let results =
            batch_euclidean_distances(&query, &candidates).expect("batch euclidean should succeed");
        assert_eq!(results.len(), 2);
        assert!((results[0] - 5.0).abs() < 1e-5);
        assert!((results[1] - 13.0).abs() < 1e-5);
    }

    #[test]
    fn test_batch_dot_products() {
        let query = vec![1.0, 2.0];
        let candidates = vec![&[3.0, 4.0][..], &[5.0, 6.0][..]];
        let results =
            batch_dot_products(&query, &candidates).expect("batch dot products should succeed");
        assert_eq!(results.len(), 2);
        assert!((results[0] - 11.0).abs() < 1e-5);
        assert!((results[1] - 17.0).abs() < 1e-5);
    }

    #[test]
    fn test_batch_cosine_with_query_norm() {
        let query = vec![1.0, 0.0];
        let candidates = vec![&[1.0, 0.0][..], &[0.0, 1.0][..]];
        let results = batch_cosine_with_query_norm(&query, 1.0, &candidates)
            .expect("batch cosine with query norm should succeed");
        assert!((results[0] - 1.0).abs() < 1e-5);
        assert!(results[1].abs() < 1e-5);
    }

    #[test]
    fn test_batch_dot_products_mismatched_candidate_length() {
        let query = vec![1.0, 2.0];
        let candidates = vec![&[3.0, 4.0][..], &[5.0][..]];
        assert!(batch_dot_products(&query, &candidates).is_err());
    }

    #[test]
    fn test_batch_euclidean_distances_longer_candidate_rejected() {
        let query = vec![1.0, 2.0];
        let candidates = vec![&[1.0, 2.0, 3.0][..]];
        assert!(batch_euclidean_distances(&query, &candidates).is_err());
    }

    #[test]
    fn test_batch_cosine_similarities_shorter_candidate_rejected() {
        let query = vec![1.0, 2.0];
        let candidates = vec![&[1.0][..]];
        assert!(batch_cosine_similarities(&query, &candidates).is_err());
    }
}
