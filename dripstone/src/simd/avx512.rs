//! AVX-512 SIMD implementations for x86_64.

use super::DenseMetric;
use crate::errors::DripstoneError;
use half::f16;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// AVX-512 optimized dot product.
///
/// # Safety
/// Caller must ensure AVX-512F is available on the CPU and both slices have equal length.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
pub unsafe fn dot_product_avx512(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(
        a.len(),
        b.len(),
        "AVX-512 dot product requires equal lengths"
    );
    let len = a.len();
    let chunks = len / 16;
    let mut sum = _mm512_setzero_ps();

    for i in 0..chunks {
        let idx = i * 16;
        unsafe {
            let va = _mm512_loadu_ps(a.as_ptr().add(idx));
            let vb = _mm512_loadu_ps(b.as_ptr().add(idx));
            let prod = _mm512_mul_ps(va, vb);
            sum = _mm512_add_ps(sum, prod);
        }
    }

    let mut result = unsafe { horizontal_sum_avx512(sum) };
    for i in (chunks * 16)..len {
        result += a[i] * b[i];
    }
    result
}

/// Computes the sum of squared values using AVX-512.
///
/// # Safety
/// Caller must ensure AVX-512F is available on the CPU.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
pub unsafe fn squared_sum_avx512(values: &[f32]) -> f32 {
    let len = values.len();
    let chunks = len / 16;
    let mut sum = _mm512_setzero_ps();

    for i in 0..chunks {
        let idx = i * 16;
        unsafe {
            let v = _mm512_loadu_ps(values.as_ptr().add(idx));
            let squared = _mm512_mul_ps(v, v);
            sum = _mm512_add_ps(sum, squared);
        }
    }

    let mut result = unsafe { horizontal_sum_avx512(sum) };
    for i in (chunks * 16)..len {
        result += values[i] * values[i];
    }
    result
}

/// AVX-512 optimized Euclidean distance.
///
/// # Safety
/// Caller must ensure AVX-512F is available on the CPU and both slices have equal length.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
pub unsafe fn euclidean_distance_avx512(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(
        a.len(),
        b.len(),
        "AVX-512 Euclidean distance requires equal lengths"
    );
    let len = a.len();
    let chunks = len / 16;
    let mut sum = _mm512_setzero_ps();

    for i in 0..chunks {
        let idx = i * 16;
        unsafe {
            let va = _mm512_loadu_ps(a.as_ptr().add(idx));
            let vb = _mm512_loadu_ps(b.as_ptr().add(idx));
            let diff = _mm512_sub_ps(va, vb);
            let squared = _mm512_mul_ps(diff, diff);
            sum = _mm512_add_ps(sum, squared);
        }
    }

    let mut result = unsafe { horizontal_sum_avx512(sum) };
    for i in (chunks * 16)..len {
        let diff = a[i] - b[i];
        result += diff * diff;
    }
    result.sqrt()
}

/// AVX-512 optimized Manhattan distance.
///
/// # Safety
/// Caller must ensure AVX-512F is available on the CPU and both slices have equal length.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
pub unsafe fn manhattan_distance_avx512(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(
        a.len(),
        b.len(),
        "AVX-512 Manhattan distance requires equal lengths"
    );
    let len = a.len();
    let chunks = len / 16;
    let mut sum = _mm512_setzero_ps();

    for i in 0..chunks {
        let idx = i * 16;
        unsafe {
            let va = _mm512_loadu_ps(a.as_ptr().add(idx));
            let vb = _mm512_loadu_ps(b.as_ptr().add(idx));
            let diff = _mm512_sub_ps(va, vb);
            // _mm512_abs_ps requires only avx512f (unlike _mm512_andnot_ps which needs avx512dq)
            let abs_diff = _mm512_abs_ps(diff);
            sum = _mm512_add_ps(sum, abs_diff);
        }
    }

    let mut result = unsafe { horizontal_sum_avx512(sum) };
    for i in (chunks * 16)..len {
        result += (a[i] - b[i]).abs();
    }
    result
}

/// AVX-512 optimized Chebyshev distance.
///
/// # Safety
/// Caller must ensure AVX-512F is available on the CPU and both slices have equal length.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
pub unsafe fn chebyshev_distance_avx512(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(
        a.len(),
        b.len(),
        "AVX-512 Chebyshev distance requires equal lengths"
    );
    let len = a.len();
    let chunks = len / 16;
    let mut max_abs_diff = _mm512_setzero_ps();

    for i in 0..chunks {
        let idx = i * 16;
        unsafe {
            let va = _mm512_loadu_ps(a.as_ptr().add(idx));
            let vb = _mm512_loadu_ps(b.as_ptr().add(idx));
            let diff = _mm512_sub_ps(va, vb);
            // _mm512_abs_ps requires only avx512f (unlike _mm512_andnot_ps which needs avx512dq)
            let abs_diff = _mm512_abs_ps(diff);
            max_abs_diff = _mm512_max_ps(max_abs_diff, abs_diff);
        }
    }

    let mut result = unsafe { horizontal_max_avx512(max_abs_diff) };
    for i in (chunks * 16)..len {
        result = result.max((a[i] - b[i]).abs());
    }
    result
}

/// AVX-512 optimized Euclidean distance squared.
///
/// # Safety
/// Caller must ensure AVX-512F is available on the CPU and both slices have equal length.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
pub unsafe fn euclidean_distance_squared_avx512(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(
        a.len(),
        b.len(),
        "AVX-512 Euclidean distance squared requires equal lengths"
    );
    let len = a.len();
    let chunks = len / 16;
    let mut sum = _mm512_setzero_ps();

    for i in 0..chunks {
        let idx = i * 16;
        unsafe {
            let va = _mm512_loadu_ps(a.as_ptr().add(idx));
            let vb = _mm512_loadu_ps(b.as_ptr().add(idx));
            let diff = _mm512_sub_ps(va, vb);
            let squared = _mm512_mul_ps(diff, diff);
            sum = _mm512_add_ps(sum, squared);
        }
    }

    let mut result = unsafe { horizontal_sum_avx512(sum) };
    for i in (chunks * 16)..len {
        let diff = a[i] - b[i];
        result += diff * diff;
    }
    result
}

/// AVX-512 optimized batch Euclidean distances.
///
/// # Safety
/// Caller must ensure AVX-512F is available and each candidate length matches query length.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
pub unsafe fn batch_euclidean_distances_avx512(query: &[f32], candidates: &[&[f32]]) -> Vec<f32> {
    let mut results = Vec::with_capacity(candidates.len());
    for candidate in candidates {
        results.push(unsafe { euclidean_distance_avx512(query, candidate) });
    }
    results
}

/// AVX-512 optimized batch dot products.
///
/// # Safety
/// Caller must ensure AVX-512F is available and each candidate length matches query length.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
pub unsafe fn batch_dot_products_avx512(query: &[f32], candidates: &[&[f32]]) -> Vec<f32> {
    let mut results = Vec::with_capacity(candidates.len());
    for candidate in candidates {
        results.push(unsafe { dot_product_avx512(query, candidate) });
    }
    results
}

/// AVX-512 optimized batch cosine similarities.
///
/// # Safety
/// Caller must ensure AVX-512F is available and each candidate length matches query length.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
pub unsafe fn batch_cosine_similarities_avx512(query: &[f32], candidates: &[&[f32]]) -> Vec<f32> {
    let query_norm = unsafe { squared_sum_avx512(query).sqrt() };
    unsafe { batch_cosine_with_query_norm_avx512(query, query_norm, candidates) }
}

/// AVX-512 optimized batch cosine similarities with precomputed query norm.
///
/// # Safety
/// Caller must ensure AVX-512F is available and each candidate length matches query length.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
pub unsafe fn batch_cosine_with_query_norm_avx512(
    query: &[f32],
    query_norm: f32,
    candidates: &[&[f32]],
) -> Vec<f32> {
    unsafe {
        batch_cosine_with_query_norm_and_candidate_norms_avx512(query, query_norm, candidates, None)
    }
}

/// AVX-512 optimized batch cosine similarities with precomputed query norm and
/// optional precomputed candidate norms.
///
/// # Safety
/// Caller must ensure AVX-512F is available and each candidate length matches query length.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
pub unsafe fn batch_cosine_with_query_norm_and_candidate_norms_avx512(
    query: &[f32],
    query_norm: f32,
    candidates: &[&[f32]],
    candidate_norms: Option<&[f32]>,
) -> Vec<f32> {
    let mut results = Vec::with_capacity(candidates.len());
    for (idx, candidate) in candidates.iter().enumerate() {
        let dot = unsafe { dot_product_avx512(query, candidate) };
        let mag = candidate_norms
            .and_then(|norms| norms.get(idx).copied())
            .unwrap_or_else(|| unsafe { squared_sum_avx512(candidate).sqrt() });
        results.push(super::cosine_or_zero(dot, query_norm, mag));
    }
    results
}

/// AVX-512-dispatch `f16` byte scoring using `vcvtph2ps` decode and FMA accumulation.
///
/// This keeps the stored layout in packed `f16` for maximum bandwidth savings while
/// still accumulating in `f32`.
///
/// # Safety
/// Caller must ensure AVX-512F, AVX, F16C, and FMA are available on the CPU and
/// the byte slice length matches `query.len() * size_of::<u16>()`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx,f16c,fma")]
pub unsafe fn score_f16_bytes_avx512_f16c(
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

    let chunks = len / 16;
    let sign_mask = _mm256_set1_ps(-0.0f32);
    let mut dot_sum_lo = _mm256_setzero_ps();
    let mut dot_sum_hi = _mm256_setzero_ps();
    let mut candidate_sq_sum_lo = _mm256_setzero_ps();
    let mut candidate_sq_sum_hi = _mm256_setzero_ps();
    let mut euclidean_sum_lo = _mm256_setzero_ps();
    let mut euclidean_sum_hi = _mm256_setzero_ps();
    let mut manhattan_sum_lo = _mm256_setzero_ps();
    let mut manhattan_sum_hi = _mm256_setzero_ps();
    let mut chebyshev_max_lo = _mm256_setzero_ps();
    let mut chebyshev_max_hi = _mm256_setzero_ps();

    for chunk_idx in 0..chunks {
        let idx = chunk_idx * 16;
        // SAFETY: the loop bound keeps this read below `len`, this slice's length.
        let query_lo = unsafe { _mm256_loadu_ps(query.as_ptr().add(idx)) };
        // SAFETY: the loop bound keeps this read below `len`, this slice's length.
        let query_hi = unsafe { _mm256_loadu_ps(query.as_ptr().add(idx + 8)) };
        let encoded_lo =
            // SAFETY: the length check above and the loop bound keep this read inside `candidate_bytes`.
            unsafe { _mm_loadu_si128(candidate_bytes.as_ptr().add(idx * 2).cast::<__m128i>()) };
        // SAFETY: the length check above and the loop bound keep this read inside `candidate_bytes`.
        let encoded_hi = unsafe {
            _mm_loadu_si128(
                candidate_bytes
                    .as_ptr()
                    .add((idx + 8) * 2)
                    .cast::<__m128i>(),
            )
        };
        let candidate_lo = _mm256_cvtph_ps(encoded_lo);
        let candidate_hi = _mm256_cvtph_ps(encoded_hi);

        match metric {
            DenseMetric::Cosine => {
                dot_sum_lo = _mm256_fmadd_ps(query_lo, candidate_lo, dot_sum_lo);
                dot_sum_hi = _mm256_fmadd_ps(query_hi, candidate_hi, dot_sum_hi);
                if candidate_norm.is_none() {
                    candidate_sq_sum_lo =
                        _mm256_fmadd_ps(candidate_lo, candidate_lo, candidate_sq_sum_lo);
                    candidate_sq_sum_hi =
                        _mm256_fmadd_ps(candidate_hi, candidate_hi, candidate_sq_sum_hi);
                }
            }
            DenseMetric::Dot => {
                dot_sum_lo = _mm256_fmadd_ps(query_lo, candidate_lo, dot_sum_lo);
                dot_sum_hi = _mm256_fmadd_ps(query_hi, candidate_hi, dot_sum_hi);
            }
            DenseMetric::Euclidean => {
                let diff_lo = _mm256_sub_ps(query_lo, candidate_lo);
                let diff_hi = _mm256_sub_ps(query_hi, candidate_hi);
                euclidean_sum_lo = _mm256_fmadd_ps(diff_lo, diff_lo, euclidean_sum_lo);
                euclidean_sum_hi = _mm256_fmadd_ps(diff_hi, diff_hi, euclidean_sum_hi);
            }
            DenseMetric::Manhattan => {
                let diff_lo = _mm256_sub_ps(query_lo, candidate_lo);
                let diff_hi = _mm256_sub_ps(query_hi, candidate_hi);
                let abs_diff_lo = _mm256_andnot_ps(sign_mask, diff_lo);
                let abs_diff_hi = _mm256_andnot_ps(sign_mask, diff_hi);
                manhattan_sum_lo = _mm256_add_ps(manhattan_sum_lo, abs_diff_lo);
                manhattan_sum_hi = _mm256_add_ps(manhattan_sum_hi, abs_diff_hi);
            }
            DenseMetric::Chebyshev => {
                let diff_lo = _mm256_sub_ps(query_lo, candidate_lo);
                let diff_hi = _mm256_sub_ps(query_hi, candidate_hi);
                let abs_diff_lo = _mm256_andnot_ps(sign_mask, diff_lo);
                let abs_diff_hi = _mm256_andnot_ps(sign_mask, diff_hi);
                chebyshev_max_lo = _mm256_max_ps(chebyshev_max_lo, abs_diff_lo);
                chebyshev_max_hi = _mm256_max_ps(chebyshev_max_hi, abs_diff_hi);
            }
        }
    }

    let mut parts = super::accum::ScoreParts {
        dot: unsafe { horizontal_sum_avx256(dot_sum_lo) + horizontal_sum_avx256(dot_sum_hi) },
        euclidean: unsafe {
            horizontal_sum_avx256(euclidean_sum_lo) + horizontal_sum_avx256(euclidean_sum_hi)
        },
        manhattan: unsafe {
            horizontal_sum_avx256(manhattan_sum_lo) + horizontal_sum_avx256(manhattan_sum_hi)
        },
        chebyshev: unsafe {
            horizontal_max_avx256(chebyshev_max_lo).max(horizontal_max_avx256(chebyshev_max_hi))
        },
    };
    let mut candidate_squared_sum = candidate_norm.map_or_else(
        || unsafe {
            horizontal_sum_avx256(candidate_sq_sum_lo) + horizontal_sum_avx256(candidate_sq_sum_hi)
        },
        |norm| norm * norm,
    );
    let track_squared = candidate_norm.is_none();

    for idx in (chunks * 16)..len {
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

/// AVX-512 FP16 `f16` byte scoring using 16-lane widen to `f32`.
///
/// This keeps the stored layout in packed `f16` while widening an entire 16-value
/// chunk at once before accumulating in `f32`.
///
/// Rust's stable AVX-512 FP16 vector intrinsics around `__m512h` are still unstable,
/// so this path uses the stable 16-half conversion intrinsic `_mm512_cvtph_ps`
/// instead of the nightly-only `__m512h` load/convert APIs.
///
/// # Safety
/// Caller must ensure AVX-512F and AVX-512FP16 are available on the CPU and the
/// byte slice length matches `query.len() * size_of::<u16>()`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512fp16")]
pub unsafe fn score_f16_bytes_avx512fp16(
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

    let chunks = len / 16;
    let mut dot_sum = _mm512_setzero_ps();
    let mut candidate_sq_sum = _mm512_setzero_ps();
    let mut euclidean_sum = _mm512_setzero_ps();
    let mut manhattan_sum = _mm512_setzero_ps();
    let mut chebyshev_max = _mm512_setzero_ps();

    for chunk_idx in 0..chunks {
        let idx = chunk_idx * 16;
        // SAFETY: the loop bound keeps this read below `len`, this slice's length.
        let query_vec = unsafe { _mm512_loadu_ps(query.as_ptr().add(idx)) };
        // SAFETY: the length check above and the loop bound keep this read inside `candidate_bytes`.
        let encoded = unsafe {
            candidate_bytes
                .as_ptr()
                .add(idx * 2)
                .cast::<__m256i>()
                .read_unaligned()
        };
        let candidate_vec = _mm512_cvtph_ps(encoded);

        match metric {
            DenseMetric::Cosine => {
                dot_sum = _mm512_fmadd_ps(query_vec, candidate_vec, dot_sum);
                if candidate_norm.is_none() {
                    candidate_sq_sum =
                        _mm512_fmadd_ps(candidate_vec, candidate_vec, candidate_sq_sum);
                }
            }
            DenseMetric::Dot => {
                dot_sum = _mm512_fmadd_ps(query_vec, candidate_vec, dot_sum);
            }
            DenseMetric::Euclidean => {
                let diff = _mm512_sub_ps(query_vec, candidate_vec);
                euclidean_sum = _mm512_fmadd_ps(diff, diff, euclidean_sum);
            }
            DenseMetric::Manhattan => {
                let diff = _mm512_sub_ps(query_vec, candidate_vec);
                let abs_diff = _mm512_abs_ps(diff);
                manhattan_sum = _mm512_add_ps(manhattan_sum, abs_diff);
            }
            DenseMetric::Chebyshev => {
                let diff = _mm512_sub_ps(query_vec, candidate_vec);
                let abs_diff = _mm512_abs_ps(diff);
                chebyshev_max = _mm512_max_ps(chebyshev_max, abs_diff);
            }
        }
    }

    let mut parts = super::accum::ScoreParts {
        // SAFETY: this `#[target_feature]` function enables what the helper needs.
        dot: unsafe { horizontal_sum_avx512(dot_sum) },
        // SAFETY: this `#[target_feature]` function enables what the helper needs.
        euclidean: unsafe { horizontal_sum_avx512(euclidean_sum) },
        // SAFETY: this `#[target_feature]` function enables what the helper needs.
        manhattan: unsafe { horizontal_sum_avx512(manhattan_sum) },
        // SAFETY: this `#[target_feature]` function enables what the helper needs.
        chebyshev: unsafe { horizontal_max_avx512(chebyshev_max) },
    };
    let mut candidate_squared_sum = candidate_norm.map_or_else(
        // SAFETY: this `#[target_feature]` function enables what the helper needs.
        || unsafe { horizontal_sum_avx512(candidate_sq_sum) },
        |norm| norm * norm,
    );
    let track_squared = candidate_norm.is_none();

    for idx in (chunks * 16)..len {
        let byte_offset = idx * 2;
        let encoded = u16::from_le_bytes([
            candidate_bytes[byte_offset],
            candidate_bytes[byte_offset + 1],
        ]);
        let candidate_value = f16::from_bits(encoded).to_f32();
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

/// AVX-512 optimized L2 normalization (in-place).
///
/// # Safety
/// Caller must ensure AVX-512F is available on the CPU.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
pub unsafe fn l2_normalize_avx512(vector: &mut [f32]) -> Result<(), crate::errors::DripstoneError> {
    let mag = unsafe { squared_sum_avx512(vector).sqrt() };
    if !super::usable_cosine_norm(mag) {
        return Err(crate::errors::DripstoneError::storage(
            "cannot normalize zero-magnitude vector",
        ));
    }

    let inv_mag = 1.0 / mag;
    let scale = _mm512_set1_ps(inv_mag);
    let len = vector.len();
    let chunks = len / 16;

    for i in 0..chunks {
        let idx = i * 16;
        unsafe {
            let v = _mm512_loadu_ps(vector.as_ptr().add(idx));
            let normalized = _mm512_mul_ps(v, scale);
            _mm512_storeu_ps(vector.as_mut_ptr().add(idx), normalized);
        }
    }

    for x in vector.iter_mut().skip(chunks * 16) {
        *x *= inv_mag;
    }
    Ok(())
}

/// AVX-512 optimized vector addition: `out = a + b`.
///
/// # Safety
/// Caller must ensure AVX-512F is available and all slices have equal length.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
pub unsafe fn vector_add_avx512(a: &[f32], b: &[f32], out: &mut [f32]) {
    debug_assert_eq!(
        a.len(),
        b.len(),
        "AVX-512 vector add requires equal lengths"
    );
    debug_assert_eq!(
        a.len(),
        out.len(),
        "AVX-512 vector add requires equal lengths"
    );
    let len = a.len().min(b.len()).min(out.len());
    let chunks = len / 16;

    for i in 0..chunks {
        let idx = i * 16;
        unsafe {
            let va = _mm512_loadu_ps(a.as_ptr().add(idx));
            let vb = _mm512_loadu_ps(b.as_ptr().add(idx));
            let sum = _mm512_add_ps(va, vb);
            _mm512_storeu_ps(out.as_mut_ptr().add(idx), sum);
        }
    }

    for i in (chunks * 16)..len {
        out[i] = a[i] + b[i];
    }
}

/// AVX-512 optimized vector subtraction: `out = a - b`.
///
/// # Safety
/// Caller must ensure AVX-512F is available and all slices have equal length.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
pub unsafe fn vector_sub_avx512(a: &[f32], b: &[f32], out: &mut [f32]) {
    debug_assert_eq!(
        a.len(),
        b.len(),
        "AVX-512 vector sub requires equal lengths"
    );
    debug_assert_eq!(
        a.len(),
        out.len(),
        "AVX-512 vector sub requires equal lengths"
    );
    let len = a.len().min(b.len()).min(out.len());
    let chunks = len / 16;

    for i in 0..chunks {
        let idx = i * 16;
        unsafe {
            let va = _mm512_loadu_ps(a.as_ptr().add(idx));
            let vb = _mm512_loadu_ps(b.as_ptr().add(idx));
            let diff = _mm512_sub_ps(va, vb);
            _mm512_storeu_ps(out.as_mut_ptr().add(idx), diff);
        }
    }

    for i in (chunks * 16)..len {
        out[i] = a[i] - b[i];
    }
}

/// AVX-512 optimized vector scaling: `v *= scalar`.
///
/// # Safety
/// Caller must ensure AVX-512F is available.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
pub unsafe fn vector_scale_avx512(v: &mut [f32], scalar: f32) {
    let scale = _mm512_set1_ps(scalar);
    let len = v.len();
    let chunks = len / 16;

    for i in 0..chunks {
        let idx = i * 16;
        unsafe {
            let val = _mm512_loadu_ps(v.as_ptr().add(idx));
            let scaled = _mm512_mul_ps(val, scale);
            _mm512_storeu_ps(v.as_mut_ptr().add(idx), scaled);
        }
    }

    for x in v.iter_mut().skip(chunks * 16) {
        *x *= scalar;
    }
}

/// AVX-512 optimized vector accumulation: `accumulator += v`.
///
/// # Safety
/// Caller must ensure AVX-512F is available and both slices have equal length.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
pub unsafe fn vector_accumulate_avx512(accumulator: &mut [f32], v: &[f32]) {
    debug_assert_eq!(
        accumulator.len(),
        v.len(),
        "AVX-512 vector accumulate requires equal lengths"
    );
    let len = accumulator.len().min(v.len());
    let chunks = len / 16;

    for i in 0..chunks {
        let idx = i * 16;
        unsafe {
            let acc = _mm512_loadu_ps(accumulator.as_ptr().add(idx));
            let val = _mm512_loadu_ps(v.as_ptr().add(idx));
            let sum = _mm512_add_ps(acc, val);
            _mm512_storeu_ps(accumulator.as_mut_ptr().add(idx), sum);
        }
    }

    for i in (chunks * 16)..len {
        accumulator[i] += v[i];
    }
}

/// AVX-512 optimized min/max finding.
///
/// # Safety
/// Caller must ensure AVX-512F is available.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
pub unsafe fn find_min_max_avx512(values: &[f32]) -> (f32, f32) {
    if values.is_empty() {
        return (f32::INFINITY, f32::NEG_INFINITY);
    }

    let len = values.len();
    let chunks = len / 16;
    let mut min_vec = _mm512_set1_ps(f32::MAX);
    let mut max_vec = _mm512_set1_ps(f32::MIN);

    for i in 0..chunks {
        let idx = i * 16;
        unsafe {
            let v = _mm512_loadu_ps(values.as_ptr().add(idx));
            min_vec = _mm512_min_ps(min_vec, v);
            max_vec = _mm512_max_ps(max_vec, v);
        }
    }

    let mut min = unsafe { horizontal_min_avx512(min_vec) };
    let mut max = unsafe { horizontal_max_avx512(max_vec) };
    for &value in values.iter().skip(chunks * 16) {
        min = min.min(value);
        max = max.max(value);
    }
    (min, max)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn horizontal_sum_avx512(v: __m512) -> f32 {
    let mut values = [0.0f32; 16];
    unsafe { _mm512_storeu_ps(values.as_mut_ptr(), v) };
    values.iter().sum()
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
unsafe fn horizontal_sum_avx256(v: __m256) -> f32 {
    let mut values = [0.0f32; 8];
    unsafe { _mm256_storeu_ps(values.as_mut_ptr(), v) };
    values.into_iter().sum()
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn horizontal_min_avx512(v: __m512) -> f32 {
    let mut values = [0.0f32; 16];
    unsafe { _mm512_storeu_ps(values.as_mut_ptr(), v) };
    values.into_iter().fold(f32::INFINITY, f32::min)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn horizontal_max_avx512(v: __m512) -> f32 {
    let mut values = [0.0f32; 16];
    unsafe { _mm512_storeu_ps(values.as_mut_ptr(), v) };
    values.into_iter().fold(f32::NEG_INFINITY, f32::max)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
unsafe fn horizontal_max_avx256(v: __m256) -> f32 {
    let mut values = [0.0f32; 8];
    unsafe { _mm256_storeu_ps(values.as_mut_ptr(), v) };
    values.into_iter().fold(f32::NEG_INFINITY, f32::max)
}

#[cfg(test)]
#[cfg(target_arch = "x86_64")]
mod tests {
    use super::*;

    #[test]
    fn test_avx512_dot_product() {
        if is_x86_feature_detected!("avx512f") {
            let a = vec![1.0; 16];
            let b = vec![2.0; 16];
            let result = unsafe { dot_product_avx512(&a, &b) };
            assert!((result - 32.0).abs() < 1e-5);
        }
    }

    #[test]
    fn test_avx512_squared_sum() {
        if is_x86_feature_detected!("avx512f") {
            let values = vec![2.0; 16];
            let result = unsafe { squared_sum_avx512(&values) };
            assert!((result - 64.0).abs() < 1e-5);
        }
    }

    #[test]
    fn test_avx512_euclidean_squared() {
        if is_x86_feature_detected!("avx512f") {
            let a = vec![0.0; 16];
            let b = vec![1.0; 16];
            let result = unsafe { euclidean_distance_squared_avx512(&a, &b) };
            assert!((result - 16.0).abs() < 1e-5);
        }
    }

    #[test]
    fn test_avx512_f16_scoring_with_candidate_norm() {
        if is_x86_feature_detected!("avx512f")
            && is_x86_feature_detected!("f16c")
            && is_x86_feature_detected!("fma")
        {
            let query = vec![1.0_f32; 16];
            let candidate = vec![0.5_f32; 16];
            let bytes: Vec<u8> = candidate
                .iter()
                .flat_map(|value| half::f16::from_f32(*value).to_bits().to_le_bytes())
                .collect();
            let score = unsafe {
                score_f16_bytes_avx512_f16c(
                    &query,
                    &bytes,
                    DenseMetric::Cosine,
                    Some(4.0),
                    Some(2.0),
                )
            }
            .expect("avx512 f16 scoring");
            assert!((score - 1.0).abs() < 0.01);
        }
    }

    #[test]
    fn avx512fp16_manhattan_and_chebyshev_use_absolute_differences() {
        if !(is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512fp16")) {
            return;
        }
        let query = [
            1.0_f32, -2.0, 3.0, -4.0, 0.5, -0.5, 2.0, -3.0, 0.0, 1.0, -1.0, 6.0, -6.0, 2.5, -2.5,
            0.0,
        ];
        let candidate = [
            -1.0_f32, 2.0, -3.0, 4.0, 0.5, 0.5, -2.0, 3.0, 0.0, -1.0, 1.0, -6.0, 6.0, -2.5, 2.5,
            -0.0,
        ];
        let bytes: Vec<u8> = candidate
            .iter()
            .flat_map(|value| f16::from_f32(*value).to_bits().to_le_bytes())
            .collect();
        let score = |metric| {
            unsafe { score_f16_bytes_avx512fp16(&query, &bytes, metric, None, None) }
                .expect("avx512fp16 f16 scoring")
        };
        assert_eq!(score(DenseMetric::Manhattan), 69.0);
        assert_eq!(score(DenseMetric::Chebyshev), 12.0);
    }

    #[test]
    fn test_avx512fp16_f16_scoring_with_candidate_norm() {
        if is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512fp16") {
            let query = vec![1.0_f32; 16];
            let candidate = vec![0.5_f32; 16];
            let bytes: Vec<u8> = candidate
                .iter()
                .flat_map(|value| f16::from_f32(*value).to_bits().to_le_bytes())
                .collect();
            let score = unsafe {
                score_f16_bytes_avx512fp16(
                    &query,
                    &bytes,
                    DenseMetric::Cosine,
                    Some(4.0),
                    Some(2.0),
                )
            }
            .expect("avx512fp16 f16 scoring");
            assert!((score - 1.0).abs() < 0.01);
        }
    }
}
