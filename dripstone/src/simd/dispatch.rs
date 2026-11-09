//! Runtime SIMD dispatch and public distance helpers.

use crate::errors::DripstoneError;
#[cfg(all(target_arch = "x86_64", feature = "simd"))]
use crate::simd::avx2;
#[cfg(all(target_arch = "x86_64", feature = "simd"))]
use crate::simd::avx512;
#[cfg(all(target_arch = "aarch64", feature = "simd"))]
use crate::simd::neon;
// The scalar fallback is compiled out only when every distance goes through NEON.
#[cfg(not(all(target_arch = "aarch64", feature = "simd")))]
use crate::simd::scalar;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SimdDispatch {
    #[cfg(not(all(target_arch = "aarch64", feature = "simd")))]
    Scalar,
    #[cfg(all(target_arch = "x86_64", feature = "simd"))]
    Avx2,
    #[cfg(all(target_arch = "x86_64", feature = "simd"))]
    Avx512,
    #[cfg(all(target_arch = "aarch64", feature = "simd"))]
    Neon,
}
#[inline]
pub(crate) const fn ensure_same_length(a: &[f32], b: &[f32]) -> Result<(), DripstoneError> {
    if a.len() != b.len() {
        return Err(DripstoneError::InvalidVectorLength {
            expected: a.len(),
            actual: b.len(),
        });
    }

    Ok(())
}

#[inline]
pub(crate) fn ensure_batch_candidate_lengths(
    query: &[f32],
    candidates: &[&[f32]],
) -> Result<(), DripstoneError> {
    for candidate in candidates {
        if candidate.len() != query.len() {
            return Err(DripstoneError::InvalidVectorLength {
                expected: query.len(),
                actual: candidate.len(),
            });
        }
    }

    Ok(())
}

#[inline]
pub(crate) fn detect_dispatch() -> SimdDispatch {
    #[cfg(all(target_arch = "x86_64", feature = "simd"))]
    {
        if is_x86_feature_detected!("avx512f") {
            return SimdDispatch::Avx512;
        }
        if is_x86_feature_detected!("avx2") {
            return SimdDispatch::Avx2;
        }
        return SimdDispatch::Scalar;
    }

    #[cfg(all(target_arch = "aarch64", feature = "simd"))]
    {
        SimdDispatch::Neon
    }

    #[cfg(not(any(
        all(feature = "simd", target_arch = "x86_64"),
        all(feature = "simd", target_arch = "aarch64")
    )))]
    {
        SimdDispatch::Scalar
    }
}

#[inline]
pub(crate) fn dot_product_with_dispatch(
    a: &[f32],
    b: &[f32],
    dispatch: SimdDispatch,
) -> Result<f32, DripstoneError> {
    ensure_same_length(a, b)?;

    let sum = match dispatch {
        #[cfg(all(target_arch = "x86_64", feature = "simd"))]
        SimdDispatch::Avx512 => unsafe { avx512::dot_product_avx512(a, b) },
        #[cfg(all(target_arch = "x86_64", feature = "simd"))]
        SimdDispatch::Avx2 => unsafe { avx2::dot_product_avx2(a, b) },
        #[cfg(all(target_arch = "aarch64", feature = "simd"))]
        SimdDispatch::Neon => unsafe { neon::dot_product_neon(a, b) },
        #[cfg(not(all(target_arch = "aarch64", feature = "simd")))]
        SimdDispatch::Scalar => scalar::dot_product(a, b),
    };

    Ok(sum)
}

#[inline]
pub(crate) fn squared_sum_with_dispatch(
    values: &[f32],
    dispatch: SimdDispatch,
) -> Result<f32, DripstoneError> {
    let sum = match dispatch {
        #[cfg(all(target_arch = "x86_64", feature = "simd"))]
        SimdDispatch::Avx512 => unsafe { avx512::squared_sum_avx512(values) },
        #[cfg(all(target_arch = "x86_64", feature = "simd"))]
        SimdDispatch::Avx2 => unsafe { avx2::squared_sum_avx2(values) },
        #[cfg(all(target_arch = "aarch64", feature = "simd"))]
        SimdDispatch::Neon => unsafe { neon::squared_sum_neon(values) },
        #[cfg(not(all(target_arch = "aarch64", feature = "simd")))]
        SimdDispatch::Scalar => scalar::squared_sum(values)?,
    };

    if sum.is_finite() {
        Ok(sum)
    } else {
        Err(DripstoneError::storage("failed to compute squared sum"))
    }
}

#[inline]
pub(crate) fn magnitude_with_dispatch(
    values: &[f32],
    dispatch: SimdDispatch,
) -> Result<f32, DripstoneError> {
    let norm = squared_sum_with_dispatch(values, dispatch)?.sqrt();
    if norm.is_finite() {
        Ok(norm)
    } else {
        Err(DripstoneError::storage("failed to compute magnitude"))
    }
}

#[inline]
pub(crate) fn cosine_with_query_norm_dispatch(
    query: &[f32],
    query_norm: f32,
    candidate: &[f32],
    dispatch: SimdDispatch,
) -> Result<f32, DripstoneError> {
    ensure_same_length(query, candidate)?;
    let dot = dot_product_with_dispatch(query, candidate, dispatch)?;
    let candidate_norm = match magnitude_with_dispatch(candidate, dispatch) {
        Ok(norm) => norm,
        Err(_) => return Ok(0.0),
    };
    Ok(super::cosine_or_zero(dot, query_norm, candidate_norm))
}

#[inline]
pub(crate) fn euclidean_distance_with_dispatch(
    a: &[f32],
    b: &[f32],
    dispatch: SimdDispatch,
) -> Result<f32, DripstoneError> {
    ensure_same_length(a, b)?;

    let distance = match dispatch {
        #[cfg(all(target_arch = "x86_64", feature = "simd"))]
        SimdDispatch::Avx512 => unsafe { avx512::euclidean_distance_avx512(a, b) },
        #[cfg(all(target_arch = "x86_64", feature = "simd"))]
        SimdDispatch::Avx2 => unsafe { avx2::euclidean_distance_avx2(a, b) },
        #[cfg(all(target_arch = "aarch64", feature = "simd"))]
        SimdDispatch::Neon => unsafe { neon::euclidean_distance_neon(a, b) },
        #[cfg(not(all(target_arch = "aarch64", feature = "simd")))]
        SimdDispatch::Scalar => scalar::euclidean_distance(a, b),
    };

    Ok(distance)
}

#[inline]
pub(crate) fn euclidean_distance_squared_with_dispatch(
    a: &[f32],
    b: &[f32],
    dispatch: SimdDispatch,
) -> Result<f32, DripstoneError> {
    ensure_same_length(a, b)?;

    let distance = match dispatch {
        #[cfg(all(target_arch = "x86_64", feature = "simd"))]
        SimdDispatch::Avx512 => unsafe { avx512::euclidean_distance_squared_avx512(a, b) },
        #[cfg(all(target_arch = "x86_64", feature = "simd"))]
        SimdDispatch::Avx2 => unsafe { avx2::euclidean_distance_squared_avx2(a, b) },
        #[cfg(all(target_arch = "aarch64", feature = "simd"))]
        SimdDispatch::Neon => unsafe { neon::euclidean_distance_squared_neon(a, b) },
        #[cfg(not(all(target_arch = "aarch64", feature = "simd")))]
        SimdDispatch::Scalar => scalar::euclidean_distance_squared(a, b),
    };

    Ok(distance)
}

#[inline]
pub(crate) fn manhattan_distance_with_dispatch(
    a: &[f32],
    b: &[f32],
    dispatch: SimdDispatch,
) -> Result<f32, DripstoneError> {
    ensure_same_length(a, b)?;

    let distance = match dispatch {
        #[cfg(all(target_arch = "x86_64", feature = "simd"))]
        SimdDispatch::Avx512 => unsafe { avx512::manhattan_distance_avx512(a, b) },
        #[cfg(all(target_arch = "x86_64", feature = "simd"))]
        SimdDispatch::Avx2 => unsafe { avx2::manhattan_distance_avx2(a, b) },
        #[cfg(all(target_arch = "aarch64", feature = "simd"))]
        SimdDispatch::Neon => unsafe { neon::manhattan_distance_neon(a, b) },
        #[cfg(not(all(target_arch = "aarch64", feature = "simd")))]
        SimdDispatch::Scalar => scalar::manhattan_distance(a, b),
    };

    Ok(distance)
}

#[inline]
pub(crate) fn chebyshev_distance_with_dispatch(
    a: &[f32],
    b: &[f32],
    dispatch: SimdDispatch,
) -> Result<f32, DripstoneError> {
    ensure_same_length(a, b)?;

    let distance = match dispatch {
        #[cfg(all(target_arch = "x86_64", feature = "simd"))]
        SimdDispatch::Avx512 => unsafe { avx512::chebyshev_distance_avx512(a, b) },
        #[cfg(all(target_arch = "x86_64", feature = "simd"))]
        SimdDispatch::Avx2 => unsafe { avx2::chebyshev_distance_avx2(a, b) },
        #[cfg(all(target_arch = "aarch64", feature = "simd"))]
        SimdDispatch::Neon => unsafe { neon::chebyshev_distance_neon(a, b) },
        #[cfg(not(all(target_arch = "aarch64", feature = "simd")))]
        SimdDispatch::Scalar => scalar::chebyshev_distance(a, b),
    };

    Ok(distance)
}

#[inline]
pub(crate) fn dot_products_batch_with_dispatch(
    query: &[f32],
    candidates: &[&[f32]],
    dispatch: SimdDispatch,
) -> Result<Vec<f32>, DripstoneError> {
    ensure_batch_candidate_lengths(query, candidates)?;

    let scores = match dispatch {
        #[cfg(all(target_arch = "x86_64", feature = "simd"))]
        SimdDispatch::Avx512 => unsafe { avx512::batch_dot_products_avx512(query, candidates) },
        #[cfg(all(target_arch = "x86_64", feature = "simd"))]
        SimdDispatch::Avx2 => unsafe { avx2::batch_dot_products_avx2(query, candidates) },
        #[cfg(all(target_arch = "aarch64", feature = "simd"))]
        SimdDispatch::Neon => unsafe { neon::batch_dot_products_neon(query, candidates) },
        #[cfg(not(all(target_arch = "aarch64", feature = "simd")))]
        SimdDispatch::Scalar => scalar::batch_dot_products(query, candidates),
    };

    Ok(scores)
}

#[inline]
pub(crate) fn euclidean_distances_batch_with_dispatch(
    query: &[f32],
    candidates: &[&[f32]],
    dispatch: SimdDispatch,
) -> Result<Vec<f32>, DripstoneError> {
    ensure_batch_candidate_lengths(query, candidates)?;

    let scores = match dispatch {
        #[cfg(all(target_arch = "x86_64", feature = "simd"))]
        SimdDispatch::Avx512 => unsafe {
            avx512::batch_euclidean_distances_avx512(query, candidates)
        },
        #[cfg(all(target_arch = "x86_64", feature = "simd"))]
        SimdDispatch::Avx2 => unsafe { avx2::batch_euclidean_distances_avx2(query, candidates) },
        #[cfg(all(target_arch = "aarch64", feature = "simd"))]
        SimdDispatch::Neon => unsafe { neon::batch_euclidean_distances_neon(query, candidates) },
        #[cfg(not(all(target_arch = "aarch64", feature = "simd")))]
        SimdDispatch::Scalar => scalar::batch_euclidean_distances(query, candidates),
    };

    Ok(scores)
}
