//! Deterministic hashing embedder for the default local store.
//!
//! Tokens are bucketed with FNV-1a into a fixed vector and L2-normalized.
//! The same function embeds documents and queries. No model file or network
//! call is required, so this can ship in the default build.

use async_trait::async_trait;

use crate::errors::DripstoneError;
use crate::search::tokenizer::{SimpleTokenizer, Tokenizer};
use crate::traits::Embedder;

/// Width of [`LocalEmbedder`] vectors.
pub const LOCAL_EMBED_DIMENSION: usize = 32;

/// Stable name recorded in an opened store snapshot.
pub const LOCAL_EMBEDDER_NAME: &str = "local-hash";

const FNV_OFFSET: u32 = 2_166_136_261;
const FNV_PRIME: u32 = 16_777_619;

/// In-process embedder that hashes tokens into a dense vector.
#[derive(Debug, Clone)]
pub struct LocalEmbedder {
    tokenizer: SimpleTokenizer,
}

impl LocalEmbedder {
    /// Build the default hashing embedder.
    #[must_use]
    pub fn new() -> Self {
        Self {
            tokenizer: SimpleTokenizer::new().with_english_stopwords(),
        }
    }

    fn embed_text(&self, text: &str) -> Vec<f32> {
        let mut acc = vec![0.0; LOCAL_EMBED_DIMENSION];
        accumulate_tokens(&self.tokenizer, text, &mut acc);
        l2_normalize(&mut acc);
        acc
    }
}

impl Default for LocalEmbedder {
    fn default() -> Self {
        Self::new()
    }
}

fn accumulate_tokens(tokenizer: &SimpleTokenizer, text: &str, acc: &mut [f32]) {
    for token in tokenizer.tokenize(text) {
        let hash = fnv1a32(token.as_bytes());
        let bucket = (hash as usize) % acc.len();
        let delta: f32 = if hash & 1 == 0 { 1.0 } else { -1.0 };
        acc[bucket] += delta;
    }
}

fn fnv1a32(bytes: &[u8]) -> u32 {
    let mut hash = FNV_OFFSET;
    for byte in bytes {
        hash ^= u32::from(*byte);
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}

fn l2_normalize(values: &mut [f32]) {
    crate::simd::scale_to_unit_norm(values);
}

#[async_trait]
impl Embedder for LocalEmbedder {
    async fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, DripstoneError> {
        Ok(texts
            .iter()
            .copied()
            .map(|text| self.embed_text(text))
            .collect())
    }

    async fn embed_query(&self, query: &str) -> Result<Vec<f32>, DripstoneError> {
        Ok(self.embed_text(query))
    }

    fn dimension(&self) -> usize {
        LOCAL_EMBED_DIMENSION
    }

    fn name(&self) -> &str {
        LOCAL_EMBEDDER_NAME
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn same_text_embeds_to_the_same_unit_vector() {
        let embedder = LocalEmbedder::new();
        let first = embedder.embed_query("penguins swim").await.expect("embed");
        let second = embedder.embed_query("penguins swim").await.expect("embed");
        assert_eq!(first.len(), LOCAL_EMBED_DIMENSION);
        assert_eq!(first, second);
        let norm: f32 = first.iter().map(|value| value * value).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5);
    }
}
