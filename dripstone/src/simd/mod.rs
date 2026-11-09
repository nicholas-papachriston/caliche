//! SIMD-optimized vector similarity computations.
//!
//! This module provides high-performance similarity calculations using SIMD instructions
//! when available, with automatic fallback to scalar implementations.

#[cfg(all(feature = "simd", any(target_arch = "aarch64", target_arch = "x86_64")))]
mod accum;
mod batch;
mod dispatch;
mod ops;
mod scalar;
mod scorer;

#[cfg(all(target_arch = "x86_64", feature = "simd"))]
mod avx2;

#[cfg(all(target_arch = "x86_64", feature = "simd"))]
mod avx512;

#[cfg(all(target_arch = "aarch64", feature = "simd"))]
mod neon;

pub use batch::{
    batch_cosine_similarities, batch_cosine_with_query_norm, batch_dot_products,
    batch_euclidean_distances, cosine_with_query_norm,
};
pub use ops::{
    find_min_max, l2_normalize, vector_accumulate, vector_add, vector_scale, vector_sub,
};
pub use scorer::{
    DenseMetric, PreparedQueryScorer, chebyshev_distance, cosine_similarity, dot_product,
    euclidean_distance, euclidean_distance_squared, magnitude, manhattan_distance, squared_sum,
};

pub(crate) use dispatch::{ensure_batch_candidate_lengths, ensure_same_length};

/// A norm at or below this value has no direction, so cosine is `0.0`.
pub const ZERO_NORM_EPSILON: f32 = f32::EPSILON;

/// Cosine of two norms.
///
/// A norm at or below [`ZERO_NORM_EPSILON`], or a non-finite norm, dot, or
/// quotient, is `0.0`. The check is on each norm, so a tiny but finite pair
/// such as `[1e-4, 0.0]` keeps its cosine.
#[inline]
pub fn cosine_or_zero(dot: f32, query_norm: f32, candidate_norm: f32) -> f32 {
    if !dot.is_finite() || !usable_cosine_norm(query_norm) || !usable_cosine_norm(candidate_norm) {
        return 0.0;
    }
    let score = dot / (query_norm * candidate_norm);
    if score.is_finite() { score } else { 0.0 }
}

#[inline]
pub(crate) fn usable_cosine_norm(norm: f32) -> bool {
    norm.is_finite() && norm > ZERO_NORM_EPSILON
}

/// Divide by the L2 norm when it is finite and above [`ZERO_NORM_EPSILON`].
#[cfg(feature = "local")]
pub(crate) fn scale_to_unit_norm(values: &mut [f32]) {
    let mut sum = 0.0_f32;
    for value in &*values {
        sum = value.mul_add(*value, sum);
    }
    let norm = sum.sqrt();
    if usable_cosine_norm(norm) {
        for value in values {
            *value /= norm;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use half::f16;

    #[cfg(target_endian = "little")]
    #[test]
    fn prepared_f32_byte_scoring_accepts_aligned_storage() {
        let query = [1.0_f32, 2.0];
        let candidate = [3.0_f32, 4.0];
        let candidate_bytes = unsafe {
            // SAFETY: the byte slice covers the initialized `f32` array and
            // inherits its alignment and lifetime.
            std::slice::from_raw_parts(
                candidate.as_ptr().cast::<u8>(),
                std::mem::size_of_val(&candidate),
            )
        };
        let scorer = PreparedQueryScorer::new(&query, DenseMetric::Dot)
            .expect("prepared scorer should be valid");

        let score = scorer
            .score_f32_le_bytes(candidate_bytes)
            .expect("aligned bytes should score");
        assert!((score - 11.0).abs() < f32::EPSILON);
    }

    #[cfg(target_endian = "little")]
    #[test]
    fn prepared_f32_byte_scoring_rejects_unaligned_storage() {
        let query = [1.0_f32, 2.0];
        let bytes = [0_u8; 12];
        let (prefix, _, _) = unsafe {
            // SAFETY: this call only identifies the aligned middle region.
            bytes.align_to::<f32>()
        };
        let start = usize::from(prefix.is_empty());
        let unaligned = &bytes[start..start + std::mem::size_of_val(&query)];
        let scorer = PreparedQueryScorer::new(&query, DenseMetric::Dot)
            .expect("prepared scorer should be valid");

        assert!(scorer.score_f32_le_bytes(unaligned).is_err());
    }

    #[test]
    fn test_dot_product() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        let result = dot_product(&a, &b).expect("dot product should succeed");
        assert!((result - 32.0).abs() < 1e-5);
    }

    #[test]
    fn test_magnitude() {
        let v = vec![3.0, 4.0];
        let result = magnitude(&v).expect("magnitude should succeed");
        assert!((result - 5.0).abs() < 1e-5);
    }

    #[test]
    fn test_cosine_similarity() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        let result = cosine_similarity(&a, &b).expect("cosine should succeed");
        assert!((result - 1.0).abs() < 1e-5);
    }

    #[test]
    fn zero_norm_cosine_is_zero() {
        let query = cosine_similarity(&[0.0, 0.0], &[1.0, 0.0]).expect("zero query");
        let candidate = cosine_similarity(&[1.0, 0.0], &[0.0, 0.0]).expect("zero candidate");
        assert_eq!(query, 0.0);
        assert_eq!(candidate, 0.0);
    }

    #[test]
    fn zero_and_overflow_cosine_is_zero_on_every_compiled_path() {
        let zero = [0.0_f32, 0.0];
        let unit = [1.0_f32, 0.0];
        let huge = [f32::MAX, f32::MAX];
        assert_eq!(cosine_similarity(&zero, &unit).expect("zero query"), 0.0);
        assert_eq!(
            cosine_similarity(&unit, &zero).expect("zero candidate"),
            0.0
        );
        assert_eq!(cosine_similarity(&huge, &huge).expect("overflow"), 0.0);
        assert_eq!(
            cosine_similarity(&unit, &huge).expect("overflow candidate"),
            0.0
        );
        assert!((cosine_similarity(&unit, &unit).expect("unit") - 1.0).abs() < 1e-5);

        assert_eq!(
            crate::core::Metric::Cosine
                .compute(&zero, &unit)
                .expect("metric zero"),
            0.0
        );
        assert_eq!(
            crate::core::Metric::Cosine
                .compute(&huge, &huge)
                .expect("metric overflow"),
            0.0
        );
        assert_eq!(
            crate::embedding::SimilarityMetric::Cosine
                .compare(&zero, &unit)
                .expect("similarity zero"),
            0.0
        );
        assert_eq!(
            crate::embedding::SimilarityMetric::Cosine
                .compare(&huge, &huge)
                .expect("similarity overflow"),
            0.0
        );

        let zero_scorer = PreparedQueryScorer::new(&zero, DenseMetric::Cosine).expect("prepare");
        assert_eq!(zero_scorer.score_dense(&unit).expect("score"), 0.0);
        let huge_scorer = PreparedQueryScorer::new(&huge, DenseMetric::Cosine).expect("prepare");
        assert_eq!(huge_scorer.score_dense(&unit).expect("score"), 0.0);

        assert_eq!(
            cosine_with_query_norm(&zero, 0.0, &unit).expect("batch helper"),
            0.0
        );
        assert_eq!(
            cosine_with_query_norm(&unit, 1.0, &zero).expect("batch helper"),
            0.0
        );
        assert_eq!(
            cosine_with_query_norm(&huge, f32::INFINITY, &huge).expect("batch helper"),
            0.0
        );

        let candidates = [&zero[..], &huge[..]];
        assert_eq!(
            batch_cosine_with_query_norm(&unit, 1.0, &candidates).expect("batch"),
            vec![0.0, 0.0]
        );
        assert_eq!(
            batch_cosine_with_query_norm(&zero, 0.0, &[&unit[..]]).expect("zero query batch"),
            vec![0.0]
        );
        assert_eq!(
            batch_cosine_similarities(&huge, &[&huge[..]]).expect("batch similarities"),
            vec![0.0]
        );
        assert_eq!(
            super::scalar::batch_cosine_with_query_norm(&unit, 1.0, &candidates).expect("scalar"),
            vec![0.0, 0.0]
        );
        assert_eq!(
            super::scalar::batch_cosine_with_query_norm(&zero, 0.0, &[&unit[..]])
                .expect("scalar zero"),
            vec![0.0]
        );
        assert_eq!(
            super::scalar::batch_cosine_similarities(&huge, &[&huge[..]]),
            vec![0.0]
        );
    }

    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    #[test]
    fn avx_batch_cosine_matches_the_zero_and_overflow_rule() {
        let zero = [0.0_f32, 0.0];
        let unit = [1.0_f32, 0.0];
        let huge = [f32::MAX, f32::MAX];
        let candidates = [&zero[..], &huge[..], &unit[..]];
        if is_x86_feature_detected!("avx2") {
            let scores =
                unsafe { super::avx2::batch_cosine_with_query_norm_avx2(&unit, 1.0, &candidates) };
            assert_eq!(scores, vec![0.0, 0.0, 1.0]);
            let overflow =
                unsafe { super::avx2::batch_cosine_similarities_avx2(&huge, &[&huge[..]]) };
            assert_eq!(overflow, vec![0.0]);
        }
        if is_x86_feature_detected!("avx512f") {
            let scores = unsafe {
                super::avx512::batch_cosine_with_query_norm_avx512(&unit, 1.0, &candidates)
            };
            assert_eq!(scores, vec![0.0, 0.0, 1.0]);
            let overflow =
                unsafe { super::avx512::batch_cosine_similarities_avx512(&huge, &[&huge[..]]) };
            assert_eq!(overflow, vec![0.0]);
        }
    }

    #[test]
    fn cosine_paths_agree_on_zero_and_tiny_vectors() {
        let zero = [0.0_f32, 0.0];
        let unit = [1.0_f32, 0.0];
        let tiny = [1.0e-4_f32, 0.0];
        assert_cosine_paths_agree(&zero, &unit);
        assert_cosine_paths_agree(&unit, &zero);
        assert_cosine_paths_agree(&tiny, &tiny);
    }

    #[test]
    fn test_euclidean_distance() {
        let a = vec![0.0, 0.0];
        let b = vec![3.0, 4.0];
        let result = euclidean_distance(&a, &b).expect("euclidean should succeed");
        assert!((result - 5.0).abs() < 1e-5);
    }

    #[test]
    fn test_manhattan_distance() {
        let a = vec![0.0, 0.0];
        let b = vec![3.0, 4.0];
        let result = manhattan_distance(&a, &b).expect("manhattan should succeed");
        assert!((result - 7.0).abs() < 1e-5);
    }

    #[test]
    fn test_chebyshev_distance() {
        let a = vec![0.0, 0.0];
        let b = vec![3.0, 4.0];
        let result = chebyshev_distance(&a, &b).expect("chebyshev should succeed");
        assert!((result - 4.0).abs() < 1e-5);
    }

    #[test]
    fn test_euclidean_distance_squared() {
        let a = vec![0.0, 0.0];
        let b = vec![3.0, 4.0];
        let result =
            euclidean_distance_squared(&a, &b).expect("euclidean distance squared should succeed");
        assert!((result - 25.0).abs() < 1e-5);
    }

    #[test]
    fn test_squared_sum() {
        let values = vec![1.0, 2.0, 3.0];
        let result = squared_sum(&values).expect("squared sum should succeed");
        assert!((result - 14.0).abs() < 1e-5);
    }

    #[test]
    fn test_prepared_query_scorer_reuses_query_state() {
        let query = vec![1.0, 0.0, 0.0];
        let scorer =
            PreparedQueryScorer::new(&query, DenseMetric::Cosine).expect("prepared scorer exists");
        let score = scorer
            .score_dense(&[1.0, 0.0, 0.0])
            .expect("prepared scorer should score");
        assert!((score - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_prepared_query_scorer_mapped_dense() {
        let query = vec![1.0, 2.0];
        let scorer =
            PreparedQueryScorer::new(&query, DenseMetric::Dot).expect("prepared scorer exists");
        let mut scratch = Vec::with_capacity(query.len());
        let score = scorer
            .score_mapped_dense([3.0, 4.0], &mut scratch)
            .expect("mapped scorer should score");
        assert!((score - 11.0).abs() < 1e-5);
    }

    #[test]
    fn test_prepared_query_scorer_f16_bytes_matches_dense() {
        let query = vec![1.0, 2.0, -3.0, 4.5, 0.25, -0.75, 8.0, -1.25];
        let candidate = vec![0.5, 1.5, -2.5, 4.0, 0.5, -0.5, 7.5, -1.0];
        let bytes: Vec<u8> = candidate
            .iter()
            .flat_map(|value| f16::from_f32(*value).to_bits().to_le_bytes())
            .collect();

        for metric in [
            DenseMetric::Cosine,
            DenseMetric::Dot,
            DenseMetric::Euclidean,
            DenseMetric::Manhattan,
            DenseMetric::Chebyshev,
        ] {
            let scorer = PreparedQueryScorer::new(&query, metric).expect("prepared scorer exists");
            let dense = scorer.score_dense(&candidate).expect("dense scoring");
            let encoded = scorer.score_f16_le_bytes(&bytes).expect("f16 byte scoring");
            assert!(
                (dense - encoded).abs() < 0.01,
                "metric {metric:?} mismatch: dense={dense}, encoded={encoded}"
            );
        }
    }

    #[test]
    fn test_prepared_query_scorer_f16_bytes_with_candidate_norm_matches_dense() {
        let query = vec![1.0, 2.0, -3.0, 4.5, 0.25, -0.75, 8.0, -1.25];
        let candidate = vec![0.5, 1.5, -2.5, 4.0, 0.5, -0.5, 7.5, -1.0];
        let candidate_norm = candidate
            .iter()
            .map(|value| value * value)
            .sum::<f32>()
            .sqrt();
        let bytes: Vec<u8> = candidate
            .iter()
            .flat_map(|value| f16::from_f32(*value).to_bits().to_le_bytes())
            .collect();

        let scorer =
            PreparedQueryScorer::new(&query, DenseMetric::Cosine).expect("prepared scorer exists");
        let dense = scorer.score_dense(&candidate).expect("dense scoring");
        let encoded = scorer
            .score_f16_le_bytes_with_candidate_norm(&bytes, Some(candidate_norm))
            .expect("f16 byte scoring with norm");
        assert!((dense - encoded).abs() < 0.01);
    }

    #[test]
    fn test_prepared_query_scorer_i8_bytes_with_candidate_norm_matches_dense() {
        let query = vec![1.0, -2.0, 3.0, 0.5, -0.25, 1.5, -1.0, 2.25];
        let scale = 0.125_f32;
        let zero_point = 3_i8;
        let encoded_values = [11_i8, -5, 19, 7, 1, 15, -3, 21];
        let candidate = encoded_values
            .iter()
            .map(|value| (f32::from(*value) - f32::from(zero_point)) * scale)
            .collect::<Vec<_>>();
        let candidate_norm = candidate
            .iter()
            .map(|value| value * value)
            .sum::<f32>()
            .sqrt();
        let bytes = encoded_values
            .iter()
            .map(|value| value.to_le_bytes()[0])
            .collect::<Vec<_>>();

        for metric in [
            DenseMetric::Cosine,
            DenseMetric::Dot,
            DenseMetric::Euclidean,
            DenseMetric::Manhattan,
            DenseMetric::Chebyshev,
        ] {
            let scorer = PreparedQueryScorer::new(&query, metric).expect("prepared scorer exists");
            let dense = scorer.score_dense(&candidate).expect("dense scoring");
            let encoded = scorer
                .score_i8_le_bytes_with_candidate_norm(
                    &bytes,
                    Some(candidate_norm),
                    scale,
                    zero_point,
                )
                .expect("i8 byte scoring");
            assert!(
                (dense - encoded).abs() < 1e-5,
                "metric {metric:?} mismatch: dense={dense}, encoded={encoded}"
            );
        }
    }

    #[test]
    fn test_mismatched_lengths() {
        let a = vec![1.0, 2.0];
        let b = vec![1.0, 2.0, 3.0];
        assert!(dot_product(&a, &b).is_err());
        assert!(euclidean_distance(&a, &b).is_err());
    }

    fn assert_cosine_paths_agree(query: &[f32], candidate: &[f32]) {
        let dot = query
            .iter()
            .zip(candidate)
            .map(|(left, right)| left * right)
            .sum();
        let query_norm = query.iter().map(|value| value * value).sum::<f32>().sqrt();
        let candidate_norm = candidate
            .iter()
            .map(|value| value * value)
            .sum::<f32>()
            .sqrt();
        let expected = cosine_or_zero(dot, query_norm, candidate_norm);

        assert_eq!(
            cosine_similarity(query, candidate).expect("single"),
            expected
        );
        assert_eq!(
            crate::core::Metric::Cosine
                .compute(query, candidate)
                .expect("metric"),
            expected
        );
        assert_eq!(
            crate::embedding::SimilarityMetric::Cosine
                .compare(query, candidate)
                .expect("similarity"),
            expected
        );
        let prepared = PreparedQueryScorer::new(query, DenseMetric::Cosine).expect("prepare");
        assert_eq!(prepared.score_dense(candidate).expect("dense"), expected);
        assert_eq!(
            prepared
                .score_dense_batch(&[candidate])
                .expect("prepared batch"),
            vec![expected]
        );

        let stored = candidate.to_vec();
        let f32_bytes = unsafe {
            // SAFETY: the byte slice covers the initialized `f32` vector and
            // inherits its alignment and lifetime.
            std::slice::from_raw_parts(
                stored.as_ptr().cast::<u8>(),
                std::mem::size_of_val(stored.as_slice()),
            )
        };
        assert_eq!(
            prepared.score_f32_le_bytes(f32_bytes).expect("f32 bytes"),
            expected
        );
        assert_eq!(
            prepared
                .score_f32_le_bytes_with_candidate_norm(f32_bytes, Some(candidate_norm))
                .expect("f32 bytes with norm"),
            expected
        );
        assert_eq!(
            prepared
                .score_f32_le_bytes_with_candidate_norm(f32_bytes, None)
                .expect("f32 bytes without norm"),
            expected
        );

        let decoded_f16: Vec<f32> = candidate
            .iter()
            .map(|value| f16::from_f32(*value).to_f32())
            .collect();
        let f16_norm = decoded_f16
            .iter()
            .map(|value| value * value)
            .sum::<f32>()
            .sqrt();
        let f16_dot = query
            .iter()
            .zip(&decoded_f16)
            .map(|(left, right)| left * right)
            .sum();
        let f16_expected = cosine_or_zero(f16_dot, query_norm, f16_norm);
        let f16_bytes = le_f16_bytes(candidate);
        assert_eq!(
            prepared.score_f16_le_bytes(&f16_bytes).expect("f16"),
            f16_expected
        );
        assert_eq!(
            prepared
                .score_f16_le_bytes_with_candidate_norm(&f16_bytes, Some(f16_norm))
                .expect("f16 with norm"),
            f16_expected
        );
        assert_eq!(
            prepared
                .score_f16_le_bytes_with_candidate_norm(&f16_bytes, None)
                .expect("f16 without norm"),
            f16_expected
        );

        let scale = if candidate
            .iter()
            .any(|value| value.abs() > 0.0 && value.abs() < 1.0e-3)
        {
            1.0e-4
        } else {
            1.0
        };
        let i8_bytes: Vec<u8> = candidate
            .iter()
            .map(|value| (value / scale).round() as i8)
            .map(|value| value.to_le_bytes()[0])
            .collect();
        assert_eq!(
            prepared
                .score_i8_le_bytes_with_candidate_norm(&i8_bytes, Some(candidate_norm), scale, 0)
                .expect("i8 with norm"),
            expected
        );
        assert_eq!(
            prepared
                .score_i8_le_bytes_with_candidate_norm(&i8_bytes, None, scale, 0)
                .expect("i8 without norm"),
            expected
        );

        assert_eq!(
            cosine_with_query_norm(query, query_norm, candidate).expect("query norm"),
            expected
        );
        assert_eq!(
            batch_cosine_with_query_norm(query, query_norm, &[candidate]).expect("batch norm"),
            vec![expected]
        );
        assert_eq!(
            batch_cosine_similarities(query, &[candidate]).expect("batch"),
            vec![expected]
        );
        assert_eq!(
            super::scalar::batch_cosine_with_query_norm(query, query_norm, &[candidate])
                .expect("scalar norm"),
            vec![expected]
        );
        assert_eq!(
            super::scalar::batch_cosine_similarities(query, &[candidate]),
            vec![expected]
        );
        assert_eq!(
            sparse_from(query)
                .cosine_similarity(&sparse_from(candidate))
                .expect("sparse"),
            expected
        );

        #[cfg(all(feature = "simd", target_arch = "x86_64"))]
        {
            if is_x86_feature_detected!("avx2") {
                let scores = unsafe {
                    super::avx2::batch_cosine_with_query_norm_avx2(query, query_norm, &[candidate])
                };
                assert_eq!(scores, vec![expected]);
            }
            if is_x86_feature_detected!("avx512f") {
                let scores = unsafe {
                    super::avx512::batch_cosine_with_query_norm_avx512(
                        query,
                        query_norm,
                        &[candidate],
                    )
                };
                assert_eq!(scores, vec![expected]);
                let with_norms = unsafe {
                    super::avx512::batch_cosine_with_query_norm_and_candidate_norms_avx512(
                        query,
                        query_norm,
                        &[candidate],
                        Some(&[candidate_norm]),
                    )
                };
                assert_eq!(with_norms, vec![expected]);
            }
        }
    }

    fn le_f16_bytes(values: &[f32]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|value| f16::from_f32(*value).to_bits().to_le_bytes())
            .collect()
    }

    fn sparse_from(values: &[f32]) -> crate::sparse::SparseVector {
        let mut sparse = crate::sparse::SparseVector::new();
        for (index, value) in values.iter().copied().enumerate() {
            sparse.set(u32::try_from(index).expect("index"), value);
        }
        sparse
    }
}
