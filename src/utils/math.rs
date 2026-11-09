/// Compute cosine similarity between two f32 slices.
///
/// Returns `0.0` when either vector has zero norm, dimensions do not match,
/// or dripstone rejects the input.
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(feature = "vector-memory")]
    {
        dripstone::simd::cosine_similarity(a, b).unwrap_or(0.0)
    }

    #[cfg(not(feature = "vector-memory"))]
    {
        cosine_similarity_scalar(a, b)
    }
}

#[cfg(not(feature = "vector-memory"))]
fn cosine_similarity_scalar(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }

    let mut dot = 0.0f32;
    let mut norm_a = 0.0f32;
    let mut norm_b = 0.0f32;
    for (&x, &y) in a.iter().zip(b.iter()) {
        dot = x.mul_add(y, dot);
        norm_a = x.mul_add(x, norm_a);
        norm_b = y.mul_add(y, norm_b);
    }

    if norm_a == 0.0 || norm_b == 0.0 {
        0.0
    } else {
        (dot / (norm_a.sqrt() * norm_b.sqrt())).clamp(-1.0, 1.0)
    }
}
