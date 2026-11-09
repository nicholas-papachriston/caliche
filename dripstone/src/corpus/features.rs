//! Deterministic feature vectors for the code corpus.
//!
//! CoIR CoSQA ships source text, not embeddings. The live scene upserts the
//! same `vectors.f32.bin` layout as the ANN catalogs, so each chunk and query
//! is hashed into a fixed-width signed bag of tokens and L2-normalized.
//! Cosine on these vectors is the soak metric. A neural embedder is not required.

use super::catalog::CODE_FEATURE_DIMENSION;

const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;
const DIMENSION_U64: u64 = CODE_FEATURE_DIMENSION as u64;

/// Signed token-hash features for `text`, length [`CODE_FEATURE_DIMENSION`].
///
/// Empty or whitespace-only input is the zero vector so cosine normalization
/// can drop it. Any other input is L2-normalized.
#[must_use]
pub fn code_feature_vector(text: &str) -> Vec<f32> {
    let mut bins = vec![0.0_f32; CODE_FEATURE_DIMENSION];
    let mut tokens = 0_u32;
    for_each_token(text, |token| {
        let hash = fnv1a_ascii_lower(token.as_bytes());
        let index = usize::try_from(hash % DIMENSION_U64).unwrap_or(0);
        let sign = if hash & (1_u64 << 63) == 0 { 1.0 } else { -1.0 };
        if let Some(bin) = bins.get_mut(index) {
            *bin += sign;
        }
        tokens = tokens.saturating_add(1);
    });
    if tokens == 0 {
        return bins;
    }
    let mut squared = 0.0_f32;
    for value in &bins {
        squared += value * value;
    }
    if !squared.is_finite() || squared <= 0.0 {
        if let Some(bin) = bins.first_mut() {
            *bin = 1.0;
        }
        return bins;
    }
    let norm = squared.sqrt();
    if !norm.is_finite() || norm == 0.0 {
        return bins;
    }
    for value in &mut bins {
        *value /= norm;
    }
    bins
}

fn for_each_token(text: &str, mut visit: impl FnMut(&str)) {
    let mut start = None;
    for (index, ch) in text.char_indices() {
        if ch.is_ascii_alphanumeric() || ch == '_' {
            if start.is_none() {
                start = Some(index);
            }
        } else if let Some(begin) = start.take() {
            visit(&text[begin..index]);
        }
    }
    if let Some(begin) = start {
        visit(&text[begin..]);
    }
}

fn fnv1a_ascii_lower(bytes: &[u8]) -> u64 {
    let mut hash = FNV_OFFSET;
    for byte in bytes {
        hash ^= u64::from(byte.to_ascii_lowercase());
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn features_are_stable_unit_vectors() {
        let left = code_feature_vector("def alpha():\n    return 1");
        let right = code_feature_vector("def alpha():\n    return 1");
        assert_eq!(left.len(), CODE_FEATURE_DIMENSION);
        assert_eq!(left, right);
        assert_ne!(left, code_feature_vector("def beta():\n    return 2"));
        let squared: f32 = left.iter().map(|value| value * value).sum();
        assert!((squared - 1.0).abs() < 1e-5, "norm squared {squared}");
    }

    #[test]
    fn blank_text_is_the_zero_vector() {
        let vector = code_feature_vector("   \n");
        assert!(vector.iter().all(|value| *value == 0.0));
        assert_eq!(vector.len(), CODE_FEATURE_DIMENSION);
    }
}
