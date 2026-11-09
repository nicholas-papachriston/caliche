/// Converts a finite `f64` into `f32` when its magnitude is representable.
///
/// The conversion may round to the nearest `f32`; callers use this for wire
/// formats and scoring APIs that are defined in terms of 32-bit floats.
#[cfg(any(
    test,
    feature = "qdrant-backend",
    feature = "qdrant-grpc-backend",
    feature = "pinecone-backend",
    feature = "chroma-backend",
    feature = "weaviate-backend",
    feature = "milvus-backend",
    feature = "search-advanced"
))]
pub fn checked_f64_to_f32(value: f64) -> Option<f32> {
    if !value.is_finite() || value < f64::from(f32::MIN) || value > f64::from(f32::MAX) {
        return None;
    }

    #[expect(
        clippy::cast_possible_truncation,
        reason = "range and finiteness are checked before the required f64-to-f32 representation conversion"
    )]
    Some(value as f32)
}

/// Represents an integer count as `f32` for scoring formulas.
pub const fn usize_to_f32(value: usize) -> f32 {
    #[expect(
        clippy::cast_precision_loss,
        reason = "BM25 and vector scoring are intentionally approximate f32 calculations"
    )]
    {
        value as f32
    }
}

/// Represents a 32-bit count as `f32` for scoring formulas.
pub const fn u32_to_f32(value: u32) -> f32 {
    #[expect(
        clippy::cast_precision_loss,
        reason = "BM25 and vector scoring are intentionally approximate f32 calculations"
    )]
    {
        value as f32
    }
}

/// Represents a signed quantization value as `f32`.
#[cfg(feature = "mmap-backend")]
pub const fn i32_to_f32(value: i32) -> f32 {
    #[expect(
        clippy::cast_precision_loss,
        reason = "quantization reconstruction is defined in the f32 domain"
    )]
    {
        value as f32
    }
}

/// Represents a `u64` counter as `f64` without an unchecked cast.
pub fn u64_to_f64(value: u64) -> f64 {
    const U32_RADIX: f64 = 4_294_967_296.0;
    let high = u32::try_from(value >> 32).unwrap_or(u32::MAX);
    let low = u32::try_from(value & u64::from(u32::MAX)).unwrap_or(u32::MAX);
    f64::from(high).mul_add(U32_RADIX, f64::from(low))
}

/// Represents a platform-sized count as `f64`.
pub fn usize_to_f64(value: usize) -> f64 {
    u64_to_f64(u64::try_from(value).unwrap_or(u64::MAX))
}

/// Represents a signed 64-bit count as `f64`.
#[cfg(feature = "search-advanced")]
pub fn i64_to_f64(value: i64) -> f64 {
    let magnitude = u64_to_f64(value.unsigned_abs());
    if value.is_negative() {
        -magnitude
    } else {
        magnitude
    }
}

#[cfg(test)]
mod tests {
    use super::{checked_f64_to_f32, u64_to_f64};

    #[test]
    fn checked_float_conversion_rejects_non_finite_and_out_of_range_values() {
        assert!(checked_f64_to_f32(f64::NAN).is_none());
        assert!(checked_f64_to_f32(f64::INFINITY).is_none());
        assert!(checked_f64_to_f32(f64::MAX).is_none());
        assert!(checked_f64_to_f32(1.5).is_some_and(|value| (value - 1.5).abs() <= f32::EPSILON));
    }

    #[test]
    fn u64_to_f64_preserves_exact_integer_boundary() {
        const MAX_EXACT_F64_INTEGER: u64 = 1_u64 << 53;
        assert!(
            u64_to_f64(MAX_EXACT_F64_INTEGER)
                .total_cmp(&9_007_199_254_740_992.0)
                .is_eq()
        );
    }
}
