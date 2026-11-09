//! NEON SIMD implementations for ARM aarch64.

use super::DenseMetric;
use crate::errors::DripstoneError;

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::{
    vabdq_f32, vaddq_f32, vaddvq_f32, vcvtq_f32_s32, vdupq_n_f32, vget_high_s16, vget_low_s16,
    vld1_s8, vld1q_f32, vmaxq_f32, vmaxvq_f32, vminq_f32, vminvq_f32, vmlaq_f32, vmovl_s8,
    vmovl_s16, vmulq_f32, vst1q_f32, vsubq_f32,
};

/// NEON-optimized dot product.
///
/// # Safety
/// Caller must ensure NEON is available (always true on aarch64) and
/// both slices have equal length.
#[cfg(target_arch = "aarch64")]
pub unsafe fn dot_product_neon(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len(), "NEON dot product requires equal lengths");
    let len = a.len();
    let mut sum = unsafe { vdupq_n_f32(0.0) };

    let chunks = len / 4;

    // Process 4 floats at a time
    for i in 0..chunks {
        let idx = i * 4;
        let va = unsafe { vld1q_f32(a.as_ptr().add(idx)) };
        let vb = unsafe { vld1q_f32(b.as_ptr().add(idx)) };
        sum = unsafe { vmlaq_f32(sum, va, vb) }; // sum += va * vb
    }

    // Horizontal sum
    let mut result = unsafe { vaddvq_f32(sum) };

    // Handle remainder
    for i in (chunks * 4)..len {
        result += a[i] * b[i];
    }

    result
}

/// Computes the sum of squared values using NEON.
///
/// # Safety
/// Caller must ensure NEON is available (always true on aarch64).
#[cfg(target_arch = "aarch64")]
pub unsafe fn squared_sum_neon(values: &[f32]) -> f32 {
    let len = values.len();
    let mut sum = unsafe { vdupq_n_f32(0.0) };

    let chunks = len / 4;

    for i in 0..chunks {
        let idx = i * 4;
        let v = unsafe { vld1q_f32(values.as_ptr().add(idx)) };
        sum = unsafe { vmlaq_f32(sum, v, v) }; // sum += v * v
    }

    let mut result = unsafe { vaddvq_f32(sum) };

    for i in values.iter().take(len).skip(chunks * 4) {
        result += *i * *i;
    }

    result
}

/// NEON-optimized Euclidean distance.
///
/// # Safety
/// Caller must ensure NEON is available (always true on aarch64) and
/// both slices have equal length.
#[cfg(target_arch = "aarch64")]
pub unsafe fn euclidean_distance_neon(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(
        a.len(),
        b.len(),
        "NEON Euclidean distance requires equal lengths"
    );
    let len = a.len();
    let mut sum = unsafe { vdupq_n_f32(0.0) };

    let chunks = len / 4;

    for i in 0..chunks {
        let idx = i * 4;
        let va = unsafe { vld1q_f32(a.as_ptr().add(idx)) };
        let vb = unsafe { vld1q_f32(b.as_ptr().add(idx)) };
        let diff = unsafe { vsubq_f32(va, vb) };
        sum = unsafe { vmlaq_f32(sum, diff, diff) }; // sum += diff * diff
    }

    let mut result = unsafe { vaddvq_f32(sum) };

    for i in (chunks * 4)..len {
        let diff = a[i] - b[i];
        result += diff * diff;
    }

    result.sqrt()
}

/// NEON-optimized Manhattan distance (L1 norm).
///
/// # Safety
/// Caller must ensure NEON is available (always true on aarch64) and
/// both slices have equal length.
#[cfg(target_arch = "aarch64")]
pub unsafe fn manhattan_distance_neon(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(
        a.len(),
        b.len(),
        "NEON Manhattan distance requires equal lengths"
    );
    let len = a.len();
    let mut sum = unsafe { vdupq_n_f32(0.0) };

    let chunks = len / 4;

    // Process 4 floats at a time
    for i in 0..chunks {
        let idx = i * 4;
        let va = unsafe { vld1q_f32(a.as_ptr().add(idx)) };
        let vb = unsafe { vld1q_f32(b.as_ptr().add(idx)) };
        // Absolute difference: |va - vb|
        let abs_diff = unsafe { vabdq_f32(va, vb) };
        sum = unsafe { vaddq_f32(sum, abs_diff) };
    }

    // Horizontal sum
    let mut result = unsafe { vaddvq_f32(sum) };

    // Handle remainder
    for i in (chunks * 4)..len {
        result += (a[i] - b[i]).abs();
    }

    result
}

/// NEON-optimized Chebyshev distance (L∞ norm).
///
/// # Safety
/// Caller must ensure NEON is available (always true on aarch64) and
/// both slices have equal length.
#[cfg(target_arch = "aarch64")]
pub unsafe fn chebyshev_distance_neon(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(
        a.len(),
        b.len(),
        "NEON Chebyshev distance requires equal lengths"
    );
    let len = a.len();
    let mut max_abs_diff = unsafe { vdupq_n_f32(0.0) };

    let chunks = len / 4;

    // Process 4 floats at a time
    for i in 0..chunks {
        let idx = i * 4;
        let va = unsafe { vld1q_f32(a.as_ptr().add(idx)) };
        let vb = unsafe { vld1q_f32(b.as_ptr().add(idx)) };
        // Absolute difference: |va - vb|
        let abs_diff = unsafe { vabdq_f32(va, vb) };
        max_abs_diff = unsafe { vmaxq_f32(max_abs_diff, abs_diff) };
    }

    // Horizontal max
    let mut result = unsafe { vmaxvq_f32(max_abs_diff) };

    // Handle remainder
    for i in (chunks * 4)..len {
        result = result.max((a[i] - b[i]).abs());
    }

    result
}

/// NEON-optimized Euclidean distance squared (no sqrt).
///
/// # Safety
/// Caller must ensure NEON is available (always true on aarch64) and
/// both slices have equal length.
#[cfg(target_arch = "aarch64")]
pub unsafe fn euclidean_distance_squared_neon(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(
        a.len(),
        b.len(),
        "NEON Euclidean distance squared requires equal lengths"
    );
    let len = a.len();
    let mut sum = unsafe { vdupq_n_f32(0.0) };

    let chunks = len / 4;

    for i in 0..chunks {
        let idx = i * 4;
        let va = unsafe { vld1q_f32(a.as_ptr().add(idx)) };
        let vb = unsafe { vld1q_f32(b.as_ptr().add(idx)) };
        let diff = unsafe { vsubq_f32(va, vb) };
        sum = unsafe { vmlaq_f32(sum, diff, diff) }; // sum += diff * diff
    }

    let mut result = unsafe { vaddvq_f32(sum) };

    for i in (chunks * 4)..len {
        let diff = a[i] - b[i];
        result += diff * diff;
    }

    result
}

#[cfg(target_arch = "aarch64")]
fn validate_i8_score_input(
    query: &[f32],
    candidate_bytes: &[u8],
    quant_scale: f32,
) -> Result<(), DripstoneError> {
    if candidate_bytes.len() != query.len() {
        return Err(DripstoneError::InvalidVectorLength {
            expected: query.len(),
            actual: candidate_bytes.len(),
        });
    }
    if !quant_scale.is_finite() || quant_scale <= 0.0 {
        return Err(DripstoneError::storage(
            "int8 quantization scale must be positive and finite",
        ));
    }
    Ok(())
}

/// Scores a little-endian `i8` byte slice using NEON dequantization.
///
/// # Safety
/// Caller must ensure NEON is available (always true on aarch64) and the byte
/// slice length matches `query.len()`.
#[cfg(target_arch = "aarch64")]
pub unsafe fn score_i8_bytes_neon(
    query: &[f32],
    candidate_bytes: &[u8],
    metric: DenseMetric,
    query_norm: Option<f32>,
    candidate_norm: Option<f32>,
    quant_scale: f32,
    quant_zero_point: i8,
) -> Result<f32, DripstoneError> {
    let len = query.len();
    validate_i8_score_input(query, candidate_bytes, quant_scale)?;

    let chunks = len / 8;
    let zero_point = unsafe { vdupq_n_f32(f32::from(quant_zero_point)) };
    let scale = unsafe { vdupq_n_f32(quant_scale) };
    let mut dot_sum_low = unsafe { vdupq_n_f32(0.0) };
    let mut dot_sum_high = unsafe { vdupq_n_f32(0.0) };
    let mut euclidean_sum_low = unsafe { vdupq_n_f32(0.0) };
    let mut euclidean_sum_high = unsafe { vdupq_n_f32(0.0) };
    let mut manhattan_sum_low = unsafe { vdupq_n_f32(0.0) };
    let mut manhattan_sum_high = unsafe { vdupq_n_f32(0.0) };
    let mut chebyshev_max_low = unsafe { vdupq_n_f32(0.0) };
    let mut chebyshev_max_high = unsafe { vdupq_n_f32(0.0) };

    for chunk_idx in 0..chunks {
        let idx = chunk_idx * 8;
        let query_low = unsafe { vld1q_f32(query.as_ptr().add(idx)) };
        let query_high = unsafe { vld1q_f32(query.as_ptr().add(idx + 4)) };
        let encoded = unsafe { vld1_s8(candidate_bytes.as_ptr().add(idx).cast::<i8>()) };
        let widened16 = unsafe { vmovl_s8(encoded) };
        let widened32_low = unsafe { vmovl_s16(vget_low_s16(widened16)) };
        let widened32_high = unsafe { vmovl_s16(vget_high_s16(widened16)) };
        let candidate_low =
            unsafe { vmulq_f32(vsubq_f32(vcvtq_f32_s32(widened32_low), zero_point), scale) };
        let candidate_high =
            unsafe { vmulq_f32(vsubq_f32(vcvtq_f32_s32(widened32_high), zero_point), scale) };

        match metric {
            DenseMetric::Cosine | DenseMetric::Dot => {
                dot_sum_low = unsafe { vmlaq_f32(dot_sum_low, query_low, candidate_low) };
                dot_sum_high = unsafe { vmlaq_f32(dot_sum_high, query_high, candidate_high) };
            }
            DenseMetric::Euclidean => {
                let diff_low = unsafe { vsubq_f32(query_low, candidate_low) };
                let diff_high = unsafe { vsubq_f32(query_high, candidate_high) };
                euclidean_sum_low = unsafe { vmlaq_f32(euclidean_sum_low, diff_low, diff_low) };
                euclidean_sum_high = unsafe { vmlaq_f32(euclidean_sum_high, diff_high, diff_high) };
            }
            DenseMetric::Manhattan => {
                manhattan_sum_low =
                    unsafe { vaddq_f32(manhattan_sum_low, vabdq_f32(query_low, candidate_low)) };
                manhattan_sum_high =
                    unsafe { vaddq_f32(manhattan_sum_high, vabdq_f32(query_high, candidate_high)) };
            }
            DenseMetric::Chebyshev => {
                chebyshev_max_low =
                    unsafe { vmaxq_f32(chebyshev_max_low, vabdq_f32(query_low, candidate_low)) };
                chebyshev_max_high =
                    unsafe { vmaxq_f32(chebyshev_max_high, vabdq_f32(query_high, candidate_high)) };
            }
        }
    }

    let mut dot = unsafe { vaddvq_f32(dot_sum_low) + vaddvq_f32(dot_sum_high) };
    let mut euclidean = unsafe { vaddvq_f32(euclidean_sum_low) + vaddvq_f32(euclidean_sum_high) };
    let mut manhattan = unsafe { vaddvq_f32(manhattan_sum_low) + vaddvq_f32(manhattan_sum_high) };
    let mut chebyshev =
        unsafe { vmaxvq_f32(chebyshev_max_low).max(vmaxvq_f32(chebyshev_max_high)) };
    let zero_point = f32::from(quant_zero_point);

    for idx in (chunks * 8)..len {
        let candidate_value =
            (f32::from(i8::from_le_bytes([candidate_bytes[idx]])) - zero_point) * quant_scale;
        match metric {
            DenseMetric::Cosine | DenseMetric::Dot => {
                dot += query[idx] * candidate_value;
            }
            DenseMetric::Euclidean => {
                let diff = query[idx] - candidate_value;
                euclidean += diff * diff;
            }
            DenseMetric::Manhattan => {
                manhattan += (query[idx] - candidate_value).abs();
            }
            DenseMetric::Chebyshev => {
                chebyshev = chebyshev.max((query[idx] - candidate_value).abs());
            }
        }
    }

    match metric {
        DenseMetric::Cosine => {
            let query_norm = query_norm.ok_or_else(|| {
                DripstoneError::storage("missing prepared query norm for cosine scoring")
            })?;
            let candidate_norm = candidate_norm.ok_or_else(|| {
                DripstoneError::storage("missing candidate norm for int8 cosine scoring")
            })?;
            Ok(super::cosine_or_zero(dot, query_norm, candidate_norm))
        }
        DenseMetric::Dot => Ok(dot),
        DenseMetric::Euclidean => Ok(euclidean.sqrt()),
        DenseMetric::Manhattan => Ok(manhattan),
        DenseMetric::Chebyshev => Ok(chebyshev),
    }
}

/// NEON-optimized batch Euclidean distances.
///
/// # Safety
/// Caller must ensure NEON is available (always true on aarch64).
#[cfg(target_arch = "aarch64")]
pub unsafe fn batch_euclidean_distances_neon(query: &[f32], candidates: &[&[f32]]) -> Vec<f32> {
    candidates
        .iter()
        .map(|candidate| unsafe { euclidean_distance_neon(query, candidate) })
        .collect()
}

/// NEON-optimized batch dot products.
///
/// # Safety
/// Caller must ensure NEON is available (always true on aarch64).
#[cfg(target_arch = "aarch64")]
pub unsafe fn batch_dot_products_neon(query: &[f32], candidates: &[&[f32]]) -> Vec<f32> {
    candidates
        .iter()
        .map(|candidate| unsafe { dot_product_neon(query, candidate) })
        .collect()
}

/// NEON-optimized batch cosine similarities.
///
/// # Safety
/// Caller must ensure NEON is available (always true on aarch64).
#[cfg(target_arch = "aarch64")]
pub unsafe fn batch_cosine_similarities_neon(query: &[f32], candidates: &[&[f32]]) -> Vec<f32> {
    let query_mag = unsafe { squared_sum_neon(query) }.sqrt();
    unsafe { batch_cosine_with_query_norm_neon(query, query_mag, candidates) }
}

/// NEON-optimized batch cosine similarities with precomputed query norm.
///
/// # Safety
/// Caller must ensure NEON is available (always true on aarch64) and each candidate
/// length matches the query length.
#[cfg(target_arch = "aarch64")]
pub unsafe fn batch_cosine_with_query_norm_neon(
    query: &[f32],
    query_norm: f32,
    candidates: &[&[f32]],
) -> Vec<f32> {
    candidates
        .iter()
        .map(|candidate| {
            let dot = unsafe { dot_product_neon(query, candidate) };
            let mag = unsafe { squared_sum_neon(candidate) }.sqrt();
            super::cosine_or_zero(dot, query_norm, mag)
        })
        .collect()
}

/// NEON-optimized L2 normalization (normalizes vector to unit length in-place).
///
/// # Safety
/// Caller must ensure NEON is available (always true on aarch64).
#[cfg(target_arch = "aarch64")]
pub unsafe fn l2_normalize_neon(vector: &mut [f32]) -> Result<(), crate::errors::DripstoneError> {
    let mag = unsafe { squared_sum_neon(vector) }.sqrt();
    if !super::usable_cosine_norm(mag) {
        return Err(crate::errors::DripstoneError::storage(
            "cannot normalize zero-magnitude vector",
        ));
    }

    let inv_mag = 1.0 / mag;
    let scale = unsafe { vdupq_n_f32(inv_mag) };
    let len = vector.len();
    let chunks = len / 4;

    // Process 4 floats at a time
    for i in 0..chunks {
        let idx = i * 4;
        let v = unsafe { vld1q_f32(vector.as_ptr().add(idx)) };
        let normalized = unsafe { vmulq_f32(v, scale) };
        unsafe { vst1q_f32(vector.as_mut_ptr().add(idx), normalized) };
    }

    // Handle remainder
    for x in vector
        .iter_mut()
        .skip(chunks * 4)
        .take(len.saturating_sub(chunks * 4))
    {
        *x *= inv_mag;
    }

    Ok(())
}

/// NEON-optimized vector addition: out = a + b.
///
/// # Safety
/// Caller must ensure NEON is available (always true on aarch64) and lengths match exactly.
#[cfg(target_arch = "aarch64")]
pub unsafe fn vector_add_neon(a: &[f32], b: &[f32], out: &mut [f32]) {
    debug_assert_eq!(a.len(), b.len(), "NEON vector add requires equal lengths");
    debug_assert_eq!(a.len(), out.len(), "NEON vector add requires equal lengths");
    let len = a.len().min(b.len()).min(out.len());
    let chunks = len / 4;

    for i in 0..chunks {
        let idx = i * 4;
        let va = unsafe { vld1q_f32(a.as_ptr().add(idx)) };
        let vb = unsafe { vld1q_f32(b.as_ptr().add(idx)) };
        let sum = unsafe { vaddq_f32(va, vb) };
        unsafe { vst1q_f32(out.as_mut_ptr().add(idx), sum) };
    }

    for (out_val, (a_val, b_val)) in out[chunks * 4..len]
        .iter_mut()
        .zip(a[chunks * 4..len].iter().zip(b[chunks * 4..len].iter()))
    {
        *out_val = *a_val + *b_val;
    }
}

/// NEON-optimized vector subtraction: out = a - b.
///
/// # Safety
/// Caller must ensure NEON is available (always true on aarch64) and lengths match exactly.
#[cfg(target_arch = "aarch64")]
pub unsafe fn vector_sub_neon(a: &[f32], b: &[f32], out: &mut [f32]) {
    debug_assert_eq!(a.len(), b.len(), "NEON vector sub requires equal lengths");
    debug_assert_eq!(a.len(), out.len(), "NEON vector sub requires equal lengths");
    let len = a.len().min(b.len()).min(out.len());
    let chunks = len / 4;

    for i in 0..chunks {
        let idx = i * 4;
        let va = unsafe { vld1q_f32(a.as_ptr().add(idx)) };
        let vb = unsafe { vld1q_f32(b.as_ptr().add(idx)) };
        let diff = unsafe { vsubq_f32(va, vb) };
        unsafe { vst1q_f32(out.as_mut_ptr().add(idx), diff) };
    }

    for i in (chunks * 4)..len {
        out[i] = a[i] - b[i];
    }
}

/// NEON-optimized vector scaling: v *= scalar.
///
/// # Safety
/// Caller must ensure NEON is available (always true on aarch64).
#[cfg(target_arch = "aarch64")]
pub unsafe fn vector_scale_neon(v: &mut [f32], scalar: f32) {
    let scale = unsafe { vdupq_n_f32(scalar) };
    let len = v.len();
    let chunks = len / 4;

    for i in 0..chunks {
        let idx = i * 4;
        let val = unsafe { vld1q_f32(v.as_ptr().add(idx)) };
        let scaled = unsafe { vmulq_f32(val, scale) };
        unsafe { vst1q_f32(v.as_mut_ptr().add(idx), scaled) };
    }

    for x in v
        .iter_mut()
        .skip(chunks * 4)
        .take(len.saturating_sub(chunks * 4))
    {
        *x *= scalar;
    }
}

/// NEON-optimized vector accumulation: accumulator += v.
///
/// # Safety
/// Caller must ensure NEON is available (always true on aarch64) and lengths match exactly.
#[cfg(target_arch = "aarch64")]
pub unsafe fn vector_accumulate_neon(accumulator: &mut [f32], v: &[f32]) {
    debug_assert_eq!(
        accumulator.len(),
        v.len(),
        "NEON vector accumulate requires equal lengths"
    );
    let len = accumulator.len().min(v.len());
    let chunks = len / 4;

    for i in 0..chunks {
        let idx = i * 4;
        let acc = unsafe { vld1q_f32(accumulator.as_ptr().add(idx)) };
        let val = unsafe { vld1q_f32(v.as_ptr().add(idx)) };
        let sum = unsafe { vaddq_f32(acc, val) };
        unsafe { vst1q_f32(accumulator.as_mut_ptr().add(idx), sum) };
    }

    for (acc_val, v_val) in accumulator[chunks * 4..len]
        .iter_mut()
        .zip(v[chunks * 4..len].iter())
    {
        *acc_val += *v_val;
    }
}

/// NEON-optimized min/max finding.
///
/// # Safety
/// Caller must ensure NEON is available (always true on aarch64).
#[cfg(target_arch = "aarch64")]
pub unsafe fn find_min_max_neon(values: &[f32]) -> (f32, f32) {
    if values.is_empty() {
        return (f32::INFINITY, f32::NEG_INFINITY);
    }

    let len = values.len();
    let chunks = len / 4;

    let mut min_vec = unsafe { vdupq_n_f32(f32::MAX) };
    let mut max_vec = unsafe { vdupq_n_f32(f32::MIN) };

    for i in 0..chunks {
        let idx = i * 4;
        let v = unsafe { vld1q_f32(values.as_ptr().add(idx)) };
        min_vec = unsafe { vminq_f32(min_vec, v) };
        max_vec = unsafe { vmaxq_f32(max_vec, v) };
    }

    let mut min = unsafe { vminvq_f32(min_vec) };
    let mut max = unsafe { vmaxvq_f32(max_vec) };

    // Handle remainder
    for val in values
        .iter()
        .skip(chunks * 4)
        .take(len.saturating_sub(chunks * 4))
    {
        min = min.min(*val);
        max = max.max(*val);
    }

    (min, max)
}

#[cfg(test)]
#[cfg(target_arch = "aarch64")]
mod tests {
    use super::*;

    #[test]
    fn test_neon_dot_product() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![1.0, 1.0, 1.0, 1.0];
        let result = unsafe { dot_product_neon(&a, &b) };
        assert!((result - 10.0).abs() <= f32::EPSILON);
    }

    #[test]
    fn test_neon_squared_sum() {
        let v = vec![1.0, 2.0, 3.0, 4.0];
        let result = unsafe { squared_sum_neon(&v) };
        assert!((result - 30.0).abs() <= f32::EPSILON);
    }

    #[test]
    fn test_neon_euclidean() {
        let a = vec![0.0, 0.0];
        let b = vec![3.0, 4.0];
        let result = unsafe { euclidean_distance_neon(&a, &b) };
        assert!((result - 5.0).abs() < 1e-5);
    }

    #[test]
    fn test_neon_manhattan() {
        let a = vec![0.0, 0.0];
        let b = vec![3.0, 4.0];
        let result = unsafe { manhattan_distance_neon(&a, &b) };
        assert!((result - 7.0).abs() < 1e-5); // 3 + 4 = 7
    }

    #[test]
    fn test_neon_chebyshev() {
        let a = vec![0.0, 0.0];
        let b = vec![3.0, 4.0];
        let result = unsafe { chebyshev_distance_neon(&a, &b) };
        assert!((result - 4.0).abs() < 1e-5); // max(3, 4) = 4
    }

    #[test]
    fn test_neon_euclidean_squared() {
        let a = vec![0.0, 0.0];
        let b = vec![3.0, 4.0];
        let result = unsafe { euclidean_distance_squared_neon(&a, &b) };
        assert!((result - 25.0).abs() < 1e-5); // 3^2 + 4^2 = 25
    }
}
