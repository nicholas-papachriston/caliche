//! AVX2 SIMD implementations for x86_64.

use super::DenseMetric;
use crate::errors::DripstoneError;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// AVX2-optimized dot product.
///
/// # Safety
/// Caller must ensure AVX2 is available on the CPU and both slices have equal length.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn dot_product_avx2(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len(), "AVX2 dot product requires equal lengths");
    let len = a.len();
    let mut sum = _mm256_setzero_ps();

    let chunks = len / 8;

    // Process 8 floats at a time
    for i in 0..chunks {
        let idx = i * 8;
        // SAFETY: the loop bound keeps this read below `len`, this slice's length.
        let va = unsafe { _mm256_loadu_ps(a.as_ptr().add(idx)) };
        // SAFETY: the loop bound keeps this read below `len`; the contract requires `b.len() == a.len()`.
        let vb = unsafe { _mm256_loadu_ps(b.as_ptr().add(idx)) };
        let prod = _mm256_mul_ps(va, vb);
        sum = _mm256_add_ps(sum, prod);
    }

    // Horizontal sum of AVX2 register
    // SAFETY: this `#[target_feature]` function enables what the helper needs.
    let mut result = unsafe { horizontal_sum_avx2(sum) };

    // Handle remainder
    for i in (chunks * 8)..len {
        result += a[i] * b[i];
    }

    result
}

/// Computes the sum of squared values using AVX2.
///
/// # Safety
/// Caller must ensure AVX2 is available on the CPU and both slices have equal length.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn squared_sum_avx2(values: &[f32]) -> f32 {
    let len = values.len();
    let mut sum = _mm256_setzero_ps();

    let chunks = len / 8;

    for i in 0..chunks {
        let idx = i * 8;
        // SAFETY: the loop bound keeps this read below `len`, this slice's length.
        let v = unsafe { _mm256_loadu_ps(values.as_ptr().add(idx)) };
        let squared = _mm256_mul_ps(v, v);
        sum = _mm256_add_ps(sum, squared);
    }

    // SAFETY: this `#[target_feature]` function enables what the helper needs.
    let mut result = unsafe { horizontal_sum_avx2(sum) };

    for i in (chunks * 8)..len {
        result += values[i] * values[i];
    }

    result
}

/// AVX2-optimized Euclidean distance.
///
/// # Safety
/// Caller must ensure AVX2 is available on the CPU.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn euclidean_distance_avx2(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(
        a.len(),
        b.len(),
        "AVX2 Euclidean distance requires equal lengths"
    );
    let len = a.len();
    let mut sum = _mm256_setzero_ps();

    let chunks = len / 8;

    for i in 0..chunks {
        let idx = i * 8;
        // SAFETY: the loop bound keeps this read below `len`, this slice's length.
        let va = unsafe { _mm256_loadu_ps(a.as_ptr().add(idx)) };
        // SAFETY: the loop bound keeps this read below `len`; the contract requires `b.len() == a.len()`.
        let vb = unsafe { _mm256_loadu_ps(b.as_ptr().add(idx)) };
        let diff = _mm256_sub_ps(va, vb);
        let squared = _mm256_mul_ps(diff, diff);
        sum = _mm256_add_ps(sum, squared);
    }

    // SAFETY: this `#[target_feature]` function enables what the helper needs.
    let mut result = unsafe { horizontal_sum_avx2(sum) };

    for i in (chunks * 8)..len {
        let diff = a[i] - b[i];
        result += diff * diff;
    }

    result.sqrt()
}

/// AVX2-optimized Manhattan distance (L1 norm).
///
/// # Safety
/// Caller must ensure AVX2 is available on the CPU and both slices have equal length.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn manhattan_distance_avx2(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(
        a.len(),
        b.len(),
        "AVX2 Manhattan distance requires equal lengths"
    );
    let len = a.len();
    let mut sum = _mm256_setzero_ps();
    let sign_mask = _mm256_set1_ps(-0.0f32); // 0x80000000

    let chunks = len / 8;

    // Process 8 floats at a time
    for i in 0..chunks {
        let idx = i * 8;
        // SAFETY: the loop bound keeps this read below `len`, this slice's length.
        let va = unsafe { _mm256_loadu_ps(a.as_ptr().add(idx)) };
        // SAFETY: the loop bound keeps this read below `len`; the contract requires `b.len() == a.len()`.
        let vb = unsafe { _mm256_loadu_ps(b.as_ptr().add(idx)) };
        let diff = _mm256_sub_ps(va, vb);
        // Absolute value: clear sign bit
        let abs_diff = _mm256_andnot_ps(sign_mask, diff);
        sum = _mm256_add_ps(sum, abs_diff);
    }

    // Horizontal sum of AVX2 register
    // SAFETY: this `#[target_feature]` function enables what the helper needs.
    let mut result = unsafe { horizontal_sum_avx2(sum) };

    // Handle remainder
    for i in (chunks * 8)..len {
        result += (a[i] - b[i]).abs();
    }

    result
}

/// AVX2-optimized Chebyshev distance (L∞ norm).
///
/// # Safety
/// Caller must ensure AVX2 is available on the CPU and both slices have equal length.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn chebyshev_distance_avx2(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(
        a.len(),
        b.len(),
        "AVX2 Chebyshev distance requires equal lengths"
    );
    let len = a.len();
    let mut max_abs_diff = _mm256_setzero_ps();
    let sign_mask = _mm256_set1_ps(-0.0f32); // 0x80000000

    let chunks = len / 8;

    // Process 8 floats at a time
    for i in 0..chunks {
        let idx = i * 8;
        // SAFETY: the loop bound keeps this read below `len`, this slice's length.
        let va = unsafe { _mm256_loadu_ps(a.as_ptr().add(idx)) };
        // SAFETY: the loop bound keeps this read below `len`; the contract requires `b.len() == a.len()`.
        let vb = unsafe { _mm256_loadu_ps(b.as_ptr().add(idx)) };
        let diff = _mm256_sub_ps(va, vb);
        // Absolute value: clear sign bit
        let abs_diff = _mm256_andnot_ps(sign_mask, diff);
        max_abs_diff = _mm256_max_ps(max_abs_diff, abs_diff);
    }

    // Horizontal max of AVX2 register
    // SAFETY: this `#[target_feature]` function enables what the helper needs.
    let mut result = unsafe { horizontal_max_avx2(max_abs_diff) };

    // Handle remainder
    for i in (chunks * 8)..len {
        result = result.max((a[i] - b[i]).abs());
    }

    result
}

/// Performs horizontal sum of an AVX2 register.
///
/// # Safety
/// Caller must ensure AVX2 is available on the CPU and both slices have equal length.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn horizontal_sum_avx2(v: __m256) -> f32 {
    // Extract high and low 128-bit lanes
    let high = _mm256_extractf128_ps(v, 1);
    let low = _mm256_castps256_ps128(v);

    // Add the two 128-bit lanes
    let sum128 = _mm_add_ps(high, low);

    // Horizontal add within 128-bit
    let shuf = _mm_movehdup_ps(sum128);
    let sums = _mm_add_ps(sum128, shuf);
    let shuf = _mm_movehl_ps(shuf, sums);
    let sums = _mm_add_ss(sums, shuf);

    _mm_cvtss_f32(sums)
}

/// Performs horizontal max of an AVX2 register.
///
/// # Safety
/// Caller must ensure AVX2 is available on the CPU.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn horizontal_max_avx2(v: __m256) -> f32 {
    // Extract high and low 128-bit lanes
    let high = _mm256_extractf128_ps(v, 1);
    let low = _mm256_castps256_ps128(v);

    // Max of the two 128-bit lanes
    let max128 = _mm_max_ps(high, low);

    // Horizontal max within 128-bit
    let shuf = _mm_movehl_ps(max128, max128);
    let maxes = _mm_max_ps(max128, shuf);
    let shuf = _mm_shuffle_ps(maxes, maxes, 0x1);
    let maxes = _mm_max_ss(maxes, shuf);

    _mm_cvtss_f32(maxes)
}

/// AVX2-optimized Euclidean distance squared (no sqrt).
///
/// # Safety
/// Caller must ensure AVX2 is available on the CPU.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn euclidean_distance_squared_avx2(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(
        a.len(),
        b.len(),
        "AVX2 Euclidean distance squared requires equal lengths"
    );
    let len = a.len();
    let mut sum = _mm256_setzero_ps();

    let chunks = len / 8;

    for i in 0..chunks {
        let idx = i * 8;
        // SAFETY: the loop bound keeps this read below `len`, this slice's length.
        let va = unsafe { _mm256_loadu_ps(a.as_ptr().add(idx)) };
        // SAFETY: the loop bound keeps this read below `len`; the contract requires `b.len() == a.len()`.
        let vb = unsafe { _mm256_loadu_ps(b.as_ptr().add(idx)) };
        let diff = _mm256_sub_ps(va, vb);
        let squared = _mm256_mul_ps(diff, diff);
        sum = _mm256_add_ps(sum, squared);
    }

    // SAFETY: this `#[target_feature]` function enables what the helper needs.
    let mut result = unsafe { horizontal_sum_avx2(sum) };

    for i in (chunks * 8)..len {
        let diff = a[i] - b[i];
        result += diff * diff;
    }

    result
}

/// Scores a little-endian `f16` byte slice using x86 AVX/F16C instructions.
///
/// # Safety
/// Caller must ensure AVX and F16C are available on the CPU and the byte slice
/// length matches `query.len() * size_of::<u16>()`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,f16c")]
pub unsafe fn score_f16_bytes_avx_f16c(
    query: &[f32],
    candidate_bytes: &[u8],
    metric: DenseMetric,
    query_norm: Option<f32>,
    candidate_norm: Option<f32>,
) -> Result<f32, DripstoneError> {
    let len = query.len();
    let expected_len = len
        .checked_mul(std::mem::size_of::<u16>())
        .ok_or_else(|| DripstoneError::storage("candidate byte length overflow"))?;
    if candidate_bytes.len() != expected_len {
        return Err(DripstoneError::InvalidVectorLength {
            expected: expected_len,
            actual: candidate_bytes.len(),
        });
    }

    let chunks = len / 8;
    let sign_mask = _mm256_set1_ps(-0.0f32);
    let mut dot_sum = _mm256_setzero_ps();
    let mut candidate_sq_sum = _mm256_setzero_ps();
    let mut euclidean_sum = _mm256_setzero_ps();
    let mut manhattan_sum = _mm256_setzero_ps();
    let mut chebyshev_max = _mm256_setzero_ps();

    for chunk_idx in 0..chunks {
        let idx = chunk_idx * 8;
        // SAFETY: the loop bound keeps this read below `len`, this slice's length.
        let query_vec = unsafe { _mm256_loadu_ps(query.as_ptr().add(idx)) };
        // SAFETY: the length check above and the loop bound keep this read inside `candidate_bytes`.
        let encoded =
            unsafe { _mm_loadu_si128(candidate_bytes.as_ptr().add(idx * 2).cast::<__m128i>()) };
        let candidate_vec = _mm256_cvtph_ps(encoded);
        match metric {
            DenseMetric::Cosine => {
                let prod = _mm256_mul_ps(query_vec, candidate_vec);
                dot_sum = _mm256_add_ps(dot_sum, prod);
                if candidate_norm.is_none() {
                    let squared = _mm256_mul_ps(candidate_vec, candidate_vec);
                    candidate_sq_sum = _mm256_add_ps(candidate_sq_sum, squared);
                }
            }
            DenseMetric::Dot => {
                let prod = _mm256_mul_ps(query_vec, candidate_vec);
                dot_sum = _mm256_add_ps(dot_sum, prod);
            }
            DenseMetric::Euclidean => {
                let diff = _mm256_sub_ps(query_vec, candidate_vec);
                let squared = _mm256_mul_ps(diff, diff);
                euclidean_sum = _mm256_add_ps(euclidean_sum, squared);
            }
            DenseMetric::Manhattan => {
                let diff = _mm256_sub_ps(query_vec, candidate_vec);
                let abs_diff = _mm256_andnot_ps(sign_mask, diff);
                manhattan_sum = _mm256_add_ps(manhattan_sum, abs_diff);
            }
            DenseMetric::Chebyshev => {
                let diff = _mm256_sub_ps(query_vec, candidate_vec);
                let abs_diff = _mm256_andnot_ps(sign_mask, diff);
                chebyshev_max = _mm256_max_ps(chebyshev_max, abs_diff);
            }
        }
    }

    let mut parts = super::accum::ScoreParts {
        // SAFETY: this `#[target_feature]` function enables what the helper needs.
        dot: unsafe { horizontal_sum_avx(dot_sum) },
        // SAFETY: this `#[target_feature]` function enables what the helper needs.
        euclidean: unsafe { horizontal_sum_avx(euclidean_sum) },
        // SAFETY: this `#[target_feature]` function enables what the helper needs.
        manhattan: unsafe { horizontal_sum_avx(manhattan_sum) },
        // SAFETY: this `#[target_feature]` function enables what the helper needs.
        chebyshev: unsafe { horizontal_max_avx(chebyshev_max) },
    };
    let mut candidate_squared_sum =
        // SAFETY: this `#[target_feature]` function enables what the helper needs.
        candidate_norm.map_or_else(|| unsafe { horizontal_sum_avx(candidate_sq_sum) }, |norm| norm * norm);
    let track_squared = candidate_norm.is_none();

    for idx in (chunks * 8)..len {
        let byte_offset = idx * 2;
        let encoded = u16::from_le_bytes([
            candidate_bytes[byte_offset],
            candidate_bytes[byte_offset + 1],
        ]);
        let candidate_value = half::f16::from_bits(encoded).to_f32();
        super::accum::accumulate_pair(
            metric,
            query[idx],
            candidate_value,
            &mut parts,
            &mut candidate_squared_sum,
            track_squared,
        );
    }

    super::accum::finish_scored(
        metric,
        parts,
        query_norm,
        Some(candidate_squared_sum.sqrt()),
        "missing candidate norm for cosine scoring",
    )
}

/// Scores a little-endian `i8` byte slice using AVX2 dequantization.
///
/// # Safety
/// Caller must ensure AVX2 is available on the CPU and the byte slice length
/// matches `query.len()`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn score_i8_bytes_avx2(
    query: &[f32],
    candidate_bytes: &[u8],
    metric: DenseMetric,
    query_norm: Option<f32>,
    candidate_norm: Option<f32>,
    quant_scale: f32,
    quant_zero_point: i8,
) -> Result<f32, DripstoneError> {
    let len = query.len();
    if candidate_bytes.len() != len {
        return Err(DripstoneError::InvalidVectorLength {
            expected: len,
            actual: candidate_bytes.len(),
        });
    }
    if !quant_scale.is_finite() || quant_scale <= 0.0 {
        return Err(DripstoneError::storage(
            "int8 quantization scale must be positive and finite",
        ));
    }

    let chunks = len / 8;
    let sign_mask = _mm256_set1_ps(-0.0f32);
    let zero_point = _mm256_set1_ps(f32::from(quant_zero_point));
    let scale = _mm256_set1_ps(quant_scale);
    let mut dot_sum = _mm256_setzero_ps();
    let mut euclidean_sum = _mm256_setzero_ps();
    let mut manhattan_sum = _mm256_setzero_ps();
    let mut chebyshev_max = _mm256_setzero_ps();

    for chunk_idx in 0..chunks {
        let idx = chunk_idx * 8;
        // SAFETY: the loop bound keeps this read below `len`, this slice's length.
        let query_vec = unsafe { _mm256_loadu_ps(query.as_ptr().add(idx)) };
        // SAFETY: the length check above and the loop bound keep this read inside `candidate_bytes`.
        let encoded =
            unsafe { _mm_loadl_epi64(candidate_bytes.as_ptr().add(idx).cast::<__m128i>()) };
        let widened = _mm256_cvtepi8_epi32(encoded);
        let candidate_vec = _mm256_mul_ps(
            _mm256_sub_ps(_mm256_cvtepi32_ps(widened), zero_point),
            scale,
        );

        match metric {
            DenseMetric::Cosine | DenseMetric::Dot => {
                let prod = _mm256_mul_ps(query_vec, candidate_vec);
                dot_sum = _mm256_add_ps(dot_sum, prod);
            }
            DenseMetric::Euclidean => {
                let diff = _mm256_sub_ps(query_vec, candidate_vec);
                let squared = _mm256_mul_ps(diff, diff);
                euclidean_sum = _mm256_add_ps(euclidean_sum, squared);
            }
            DenseMetric::Manhattan => {
                let diff = _mm256_sub_ps(query_vec, candidate_vec);
                let abs_diff = _mm256_andnot_ps(sign_mask, diff);
                manhattan_sum = _mm256_add_ps(manhattan_sum, abs_diff);
            }
            DenseMetric::Chebyshev => {
                let diff = _mm256_sub_ps(query_vec, candidate_vec);
                let abs_diff = _mm256_andnot_ps(sign_mask, diff);
                chebyshev_max = _mm256_max_ps(chebyshev_max, abs_diff);
            }
        }
    }

    let mut parts = super::accum::ScoreParts {
        // SAFETY: this `#[target_feature]` function enables what the helper needs.
        dot: unsafe { horizontal_sum_avx(dot_sum) },
        // SAFETY: this `#[target_feature]` function enables what the helper needs.
        euclidean: unsafe { horizontal_sum_avx(euclidean_sum) },
        // SAFETY: this `#[target_feature]` function enables what the helper needs.
        manhattan: unsafe { horizontal_sum_avx(manhattan_sum) },
        // SAFETY: this `#[target_feature]` function enables what the helper needs.
        chebyshev: unsafe { horizontal_max_avx(chebyshev_max) },
    };
    let mut ignored_squared = 0.0_f32;
    let zero_point = f32::from(quant_zero_point);

    for idx in (chunks * 8)..len {
        let candidate_value =
            (f32::from(i8::from_le_bytes([candidate_bytes[idx]])) - zero_point) * quant_scale;
        super::accum::accumulate_pair(
            metric,
            query[idx],
            candidate_value,
            &mut parts,
            &mut ignored_squared,
            false,
        );
    }

    super::accum::finish_scored(
        metric,
        parts,
        query_norm,
        candidate_norm,
        "missing candidate norm for int8 cosine scoring",
    )
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
unsafe fn horizontal_sum_avx(v: __m256) -> f32 {
    let mut lanes = [0.0_f32; 8];
    unsafe { _mm256_storeu_ps(lanes.as_mut_ptr(), v) };
    lanes.into_iter().sum()
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
unsafe fn horizontal_max_avx(v: __m256) -> f32 {
    let mut lanes = [0.0_f32; 8];
    unsafe { _mm256_storeu_ps(lanes.as_mut_ptr(), v) };
    lanes.into_iter().fold(f32::NEG_INFINITY, f32::max)
}

#[cfg(test)]
#[cfg(target_arch = "x86_64")]
mod tests {
    use super::*;

    #[test]
    fn test_avx2_dot_product() {
        if is_x86_feature_detected!("avx2") {
            let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
            let b = vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
            let result = unsafe { dot_product_avx2(&a, &b) };
            assert_eq!(result, 36.0); // 1+2+3+4+5+6+7+8
        }
    }

    #[test]
    fn test_avx2_squared_sum() {
        if is_x86_feature_detected!("avx2") {
            let v = vec![1.0, 2.0, 3.0, 4.0];
            let result = unsafe { squared_sum_avx2(&v) };
            assert_eq!(result, 30.0); // 1+4+9+16
        }
    }

    #[test]
    fn test_avx2_euclidean() {
        if is_x86_feature_detected!("avx2") {
            let a = vec![0.0, 0.0, 0.0, 0.0];
            let b = vec![3.0, 4.0, 0.0, 0.0];
            let result = unsafe { euclidean_distance_avx2(&a, &b) };
            assert!((result - 5.0).abs() < 1e-5);
        }
    }

    #[test]
    fn test_avx2_manhattan() {
        if is_x86_feature_detected!("avx2") {
            let a = vec![0.0, 0.0, 0.0, 0.0];
            let b = vec![3.0, 4.0, 0.0, 0.0];
            let result = unsafe { manhattan_distance_avx2(&a, &b) };
            assert!((result - 7.0).abs() < 1e-5); // 3 + 4 = 7
        }
    }

    #[test]
    fn test_avx2_chebyshev() {
        if is_x86_feature_detected!("avx2") {
            let a = vec![0.0, 0.0, 0.0, 0.0];
            let b = vec![3.0, 4.0, 0.0, 0.0];
            let result = unsafe { chebyshev_distance_avx2(&a, &b) };
            assert!((result - 4.0).abs() < 1e-5); // max(3, 4) = 4
        }
    }

    #[test]
    fn test_avx2_euclidean_squared() {
        if is_x86_feature_detected!("avx2") {
            let a = vec![0.0, 0.0, 0.0, 0.0];
            let b = vec![3.0, 4.0, 0.0, 0.0];
            let result = unsafe { euclidean_distance_squared_avx2(&a, &b) };
            assert!((result - 25.0).abs() < 1e-5); // 3^2 + 4^2 = 25
        }
    }
}

/// AVX2-optimized batch Euclidean distances.
///
/// # Safety
/// Caller must ensure AVX2 is available on the CPU.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn batch_euclidean_distances_avx2(query: &[f32], candidates: &[&[f32]]) -> Vec<f32> {
    let mut results = Vec::with_capacity(candidates.len());
    for candidate in candidates {
        results.push(unsafe { euclidean_distance_avx2(query, candidate) });
    }
    results
}

/// AVX2-optimized batch dot products.
///
/// # Safety
/// Caller must ensure AVX2 is available on the CPU.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn batch_dot_products_avx2(query: &[f32], candidates: &[&[f32]]) -> Vec<f32> {
    let mut results = Vec::with_capacity(candidates.len());
    for candidate in candidates {
        results.push(unsafe { dot_product_avx2(query, candidate) });
    }
    results
}

/// AVX2-optimized batch cosine similarities.
///
/// # Safety
/// Caller must ensure AVX2 is available on the CPU and each candidate length
/// matches the query length.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn batch_cosine_similarities_avx2(query: &[f32], candidates: &[&[f32]]) -> Vec<f32> {
    let query_mag = unsafe { squared_sum_avx2(query).sqrt() };
    unsafe { batch_cosine_with_query_norm_avx2(query, query_mag, candidates) }
}

/// AVX2-optimized batch cosine similarities with precomputed query norm.
///
/// # Safety
/// Caller must ensure AVX2 is available on the CPU and each candidate length
/// matches the query length.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn batch_cosine_with_query_norm_avx2(
    query: &[f32],
    query_norm: f32,
    candidates: &[&[f32]],
) -> Vec<f32> {
    let mut results = Vec::with_capacity(candidates.len());
    for candidate in candidates {
        let dot = unsafe { dot_product_avx2(query, candidate) };
        let mag = unsafe { squared_sum_avx2(candidate).sqrt() };
        results.push(super::cosine_or_zero(dot, query_norm, mag));
    }
    results
}

/// AVX2-optimized L2 normalization (normalizes vector to unit length in-place).
///
/// # Safety
/// Caller must ensure AVX2 is available on the CPU.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn l2_normalize_avx2(vector: &mut [f32]) -> Result<(), crate::errors::DripstoneError> {
    let mag = unsafe { squared_sum_avx2(vector).sqrt() };
    if !super::usable_cosine_norm(mag) {
        return Err(crate::errors::DripstoneError::storage(
            "cannot normalize zero-magnitude vector",
        ));
    }

    let inv_mag = 1.0 / mag;
    let scale = _mm256_set1_ps(inv_mag);
    let len = vector.len();
    let chunks = len / 8;

    // Process 8 floats at a time
    for i in 0..chunks {
        let idx = i * 8;
        unsafe {
            let v = _mm256_loadu_ps(vector.as_ptr().add(idx));
            let normalized = _mm256_mul_ps(v, scale);
            _mm256_storeu_ps(vector.as_mut_ptr().add(idx), normalized);
        }
    }

    // Handle remainder
    for i in (chunks * 8)..len {
        vector[i] *= inv_mag;
    }

    Ok(())
}

/// AVX2-optimized vector addition: out = a + b.
///
/// # Safety
/// Caller must ensure AVX2 is available on the CPU and lengths match exactly.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn vector_add_avx2(a: &[f32], b: &[f32], out: &mut [f32]) {
    debug_assert_eq!(a.len(), b.len(), "AVX2 vector add requires equal lengths");
    debug_assert_eq!(a.len(), out.len(), "AVX2 vector add requires equal lengths");
    let len = a.len().min(b.len()).min(out.len());
    let chunks = len / 8;

    for i in 0..chunks {
        let idx = i * 8;
        unsafe {
            let va = _mm256_loadu_ps(a.as_ptr().add(idx));
            let vb = _mm256_loadu_ps(b.as_ptr().add(idx));
            let sum = _mm256_add_ps(va, vb);
            _mm256_storeu_ps(out.as_mut_ptr().add(idx), sum);
        }
    }

    for i in (chunks * 8)..len {
        out[i] = a[i] + b[i];
    }
}

/// AVX2-optimized vector subtraction: out = a - b.
///
/// # Safety
/// Caller must ensure AVX2 is available on the CPU and lengths match exactly.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn vector_sub_avx2(a: &[f32], b: &[f32], out: &mut [f32]) {
    debug_assert_eq!(a.len(), b.len(), "AVX2 vector sub requires equal lengths");
    debug_assert_eq!(a.len(), out.len(), "AVX2 vector sub requires equal lengths");
    let len = a.len().min(b.len()).min(out.len());
    let chunks = len / 8;

    for i in 0..chunks {
        let idx = i * 8;
        unsafe {
            let va = _mm256_loadu_ps(a.as_ptr().add(idx));
            let vb = _mm256_loadu_ps(b.as_ptr().add(idx));
            let diff = _mm256_sub_ps(va, vb);
            _mm256_storeu_ps(out.as_mut_ptr().add(idx), diff);
        }
    }

    for i in (chunks * 8)..len {
        out[i] = a[i] - b[i];
    }
}

/// AVX2-optimized vector scaling: v *= scalar.
///
/// # Safety
/// Caller must ensure AVX2 is available on the CPU.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn vector_scale_avx2(v: &mut [f32], scalar: f32) {
    let scale = _mm256_set1_ps(scalar);
    let len = v.len();
    let chunks = len / 8;

    for i in 0..chunks {
        let idx = i * 8;
        unsafe {
            let val = _mm256_loadu_ps(v.as_ptr().add(idx));
            let scaled = _mm256_mul_ps(val, scale);
            _mm256_storeu_ps(v.as_mut_ptr().add(idx), scaled);
        }
    }

    for i in (chunks * 8)..len {
        v[i] *= scalar;
    }
}

/// AVX2-optimized vector accumulation: accumulator += v.
///
/// # Safety
/// Caller must ensure AVX2 is available on the CPU and lengths match exactly.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn vector_accumulate_avx2(accumulator: &mut [f32], v: &[f32]) {
    debug_assert_eq!(
        accumulator.len(),
        v.len(),
        "AVX2 vector accumulate requires equal lengths"
    );
    let len = accumulator.len().min(v.len());
    let chunks = len / 8;

    for i in 0..chunks {
        let idx = i * 8;
        unsafe {
            let acc = _mm256_loadu_ps(accumulator.as_ptr().add(idx));
            let val = _mm256_loadu_ps(v.as_ptr().add(idx));
            let sum = _mm256_add_ps(acc, val);
            _mm256_storeu_ps(accumulator.as_mut_ptr().add(idx), sum);
        }
    }

    for i in (chunks * 8)..len {
        accumulator[i] += v[i];
    }
}

/// AVX2-optimized min/max finding.
///
/// # Safety
/// Caller must ensure AVX2 is available on the CPU.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn find_min_max_avx2(values: &[f32]) -> (f32, f32) {
    if values.is_empty() {
        return (f32::INFINITY, f32::NEG_INFINITY);
    }

    let len = values.len();
    let chunks = len / 8;

    let mut min_vec = _mm256_set1_ps(f32::MAX);
    let mut max_vec = _mm256_set1_ps(f32::MIN);

    for i in 0..chunks {
        let idx = i * 8;
        unsafe {
            let v = _mm256_loadu_ps(values.as_ptr().add(idx));
            min_vec = _mm256_min_ps(min_vec, v);
            max_vec = _mm256_max_ps(max_vec, v);
        }
    }

    let mut min = unsafe { horizontal_min_avx2(min_vec) };
    let mut max = unsafe { horizontal_max_avx2(max_vec) };

    // Handle remainder
    for i in (chunks * 8)..len {
        min = min.min(values[i]);
        max = max.max(values[i]);
    }

    (min, max)
}

/// Performs horizontal min of an AVX2 register.
///
/// # Safety
/// Caller must ensure AVX2 is available on the CPU.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn horizontal_min_avx2(v: __m256) -> f32 {
    // Extract high and low 128-bit lanes
    let high = _mm256_extractf128_ps(v, 1);
    let low = _mm256_castps256_ps128(v);

    // Min of the two 128-bit lanes
    let min128 = _mm_min_ps(high, low);

    // Horizontal min within 128-bit
    let shuf = _mm_movehl_ps(min128, min128);
    let mins = _mm_min_ps(min128, shuf);
    let shuf = _mm_shuffle_ps(mins, mins, 0x1);
    let mins = _mm_min_ss(mins, shuf);

    _mm_cvtss_f32(mins)
}
