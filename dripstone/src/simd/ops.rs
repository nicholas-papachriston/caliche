use crate::errors::DripstoneError;

use super::scalar;

/// Finds the minimum and maximum values in a slice.
#[allow(unreachable_code)]
#[must_use]
pub fn find_min_max(values: &[f32]) -> (f32, f32) {
    #[cfg(all(target_arch = "x86_64", feature = "simd"))]
    {
        if is_x86_feature_detected!("avx512f") {
            return unsafe { super::avx512::find_min_max_avx512(values) };
        }
        if is_x86_feature_detected!("avx2") {
            return unsafe { super::avx2::find_min_max_avx2(values) };
        }
    }

    #[cfg(all(target_arch = "aarch64", feature = "simd"))]
    {
        return unsafe { super::neon::find_min_max_neon(values) };
    }

    scalar::find_min_max(values)
}

#[inline]
const fn ensure_triplet_lengths(a: &[f32], b: &[f32], out: &[f32]) -> Result<(), DripstoneError> {
    if a.len() != b.len() {
        return Err(DripstoneError::InvalidVectorLength {
            expected: a.len(),
            actual: b.len(),
        });
    }
    if out.len() != a.len() {
        return Err(DripstoneError::InvalidVectorLength {
            expected: a.len(),
            actual: out.len(),
        });
    }
    Ok(())
}

#[inline]
const fn ensure_pair_lengths(accumulator: &[f32], v: &[f32]) -> Result<(), DripstoneError> {
    if accumulator.len() != v.len() {
        return Err(DripstoneError::InvalidVectorLength {
            expected: accumulator.len(),
            actual: v.len(),
        });
    }
    Ok(())
}

/// Adds two vectors element-wise: `out = a + b`.
#[allow(unreachable_code)]
///
/// # Errors
///
/// Returns an error when the operation cannot be completed.
pub fn vector_add(a: &[f32], b: &[f32], out: &mut [f32]) -> Result<(), DripstoneError> {
    ensure_triplet_lengths(a, b, out)?;

    #[cfg(all(target_arch = "x86_64", feature = "simd"))]
    {
        if is_x86_feature_detected!("avx512f") {
            unsafe { super::avx512::vector_add_avx512(a, b, out) };
            return Ok(());
        }
        if is_x86_feature_detected!("avx2") {
            unsafe { super::avx2::vector_add_avx2(a, b, out) };
            return Ok(());
        }
    }

    #[cfg(all(target_arch = "aarch64", feature = "simd"))]
    {
        unsafe { super::neon::vector_add_neon(a, b, out) };
        return Ok(());
    }

    scalar::vector_add(a, b, out);
    Ok(())
}

/// Subtracts two vectors element-wise: `out = a - b`.
#[allow(unreachable_code)]
///
/// # Errors
///
/// Returns an error when the operation cannot be completed.
pub fn vector_sub(a: &[f32], b: &[f32], out: &mut [f32]) -> Result<(), DripstoneError> {
    ensure_triplet_lengths(a, b, out)?;

    #[cfg(all(target_arch = "x86_64", feature = "simd"))]
    {
        if is_x86_feature_detected!("avx512f") {
            unsafe { super::avx512::vector_sub_avx512(a, b, out) };
            return Ok(());
        }
        if is_x86_feature_detected!("avx2") {
            unsafe { super::avx2::vector_sub_avx2(a, b, out) };
            return Ok(());
        }
    }

    #[cfg(all(target_arch = "aarch64", feature = "simd"))]
    {
        unsafe { super::neon::vector_sub_neon(a, b, out) };
        return Ok(());
    }

    scalar::vector_sub(a, b, out);
    Ok(())
}

/// Scales a vector by a scalar: `v *= scalar`.
#[allow(unreachable_code)]
pub fn vector_scale(v: &mut [f32], scalar: f32) {
    #[cfg(all(target_arch = "x86_64", feature = "simd"))]
    {
        if is_x86_feature_detected!("avx512f") {
            unsafe { super::avx512::vector_scale_avx512(v, scalar) };
            return;
        }
        if is_x86_feature_detected!("avx2") {
            unsafe { super::avx2::vector_scale_avx2(v, scalar) };
            return;
        }
    }

    #[cfg(all(target_arch = "aarch64", feature = "simd"))]
    {
        unsafe { super::neon::vector_scale_neon(v, scalar) };
        return;
    }

    scalar::vector_scale(v, scalar);
}

/// Accumulates a vector into an accumulator: `accumulator += v`.
#[allow(unreachable_code)]
///
/// # Errors
///
/// Returns an error when the operation cannot be completed.
pub fn vector_accumulate(accumulator: &mut [f32], v: &[f32]) -> Result<(), DripstoneError> {
    ensure_pair_lengths(accumulator, v)?;

    #[cfg(all(target_arch = "x86_64", feature = "simd"))]
    {
        if is_x86_feature_detected!("avx512f") {
            unsafe { super::avx512::vector_accumulate_avx512(accumulator, v) };
            return Ok(());
        }
        if is_x86_feature_detected!("avx2") {
            unsafe { super::avx2::vector_accumulate_avx2(accumulator, v) };
            return Ok(());
        }
    }

    #[cfg(all(target_arch = "aarch64", feature = "simd"))]
    {
        unsafe { super::neon::vector_accumulate_neon(accumulator, v) };
        return Ok(());
    }

    scalar::vector_accumulate(accumulator, v);
    Ok(())
}

/// Normalizes a vector to unit length (L2 normalization) in-place.
///
/// # Errors
///
/// Returns an error when the operation cannot be completed.
pub fn l2_normalize(vector: &mut [f32]) -> Result<(), DripstoneError> {
    #[cfg(all(target_arch = "x86_64", feature = "simd"))]
    {
        if is_x86_feature_detected!("avx512f") {
            return unsafe { super::avx512::l2_normalize_avx512(vector) };
        }
        if is_x86_feature_detected!("avx2") {
            return unsafe { super::avx2::l2_normalize_avx2(vector) };
        }
    }

    #[cfg(all(target_arch = "aarch64", feature = "simd"))]
    {
        unsafe { super::neon::l2_normalize_neon(vector) }
    }

    #[cfg(not(all(feature = "simd", any(target_arch = "x86_64", target_arch = "aarch64"))))]
    {
        scalar::l2_normalize(vector)
    }

    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    {
        scalar::l2_normalize(vector)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vector_add() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        let mut out = vec![0.0; 3];
        vector_add(&a, &b, &mut out).expect("vector add should succeed");
        assert_eq!(out, vec![5.0, 7.0, 9.0]);
    }

    #[test]
    fn test_vector_add_mismatched_lengths() {
        let a = vec![1.0, 2.0];
        let b = vec![3.0, 4.0, 5.0];
        let mut out = vec![0.0; 2];
        assert!(vector_add(&a, &b, &mut out).is_err());
    }

    #[test]
    fn test_vector_sub() {
        let a = vec![5.0, 7.0, 9.0];
        let b = vec![1.0, 2.0, 3.0];
        let mut out = vec![0.0; 3];
        vector_sub(&a, &b, &mut out).expect("vector sub should succeed");
        assert_eq!(out, vec![4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_vector_scale() {
        let mut v = vec![1.0, 2.0, 3.0];
        vector_scale(&mut v, 2.0);
        assert_eq!(v, vec![2.0, 4.0, 6.0]);
    }

    #[test]
    fn test_vector_accumulate() {
        let mut acc = vec![1.0, 2.0, 3.0];
        let v = vec![4.0, 5.0, 6.0];
        vector_accumulate(&mut acc, &v).expect("vector accumulate should succeed");
        assert_eq!(acc, vec![5.0, 7.0, 9.0]);
    }

    #[test]
    fn test_vector_accumulate_mismatched_lengths() {
        let mut acc = vec![1.0, 2.0, 3.0];
        let v = vec![4.0, 5.0];
        assert!(vector_accumulate(&mut acc, &v).is_err());
    }

    #[test]
    fn test_find_min_max() {
        let values = vec![3.0, 1.0, 4.0, 2.0, 5.0];
        let (min, max) = find_min_max(&values);
        assert!((min - 1.0).abs() <= f32::EPSILON);
        assert!((max - 5.0).abs() <= f32::EPSILON);
    }

    #[test]
    fn test_find_min_max_empty() {
        let values: Vec<f32> = vec![];
        let (min, max) = find_min_max(&values);
        assert!(min.is_infinite() && min.is_sign_positive());
        assert!(max.is_infinite() && max.is_sign_negative());
    }
}
