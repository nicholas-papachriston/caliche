//! Local embedding engines (Candle, ONNX Runtime, MLX).
//!
//! [`crate::Embedder`] is the public contract used by [`crate::Dripstone`].
//! [`VectorGenerator`] is the low-level engine trait for local model backends.
//! Concrete embedders implement both so either API works.

use std::fmt::Debug;

use crate::embedding::EmbeddingVector;
use crate::errors::DripstoneError;
use crate::traits::Embedder;

pub mod config;

pub use config::{EmbeddingConfig, PoolingStrategy};

/// Low-level local embedding engine.
///
/// Prefer [`Embedder`] at the `Dripstone` boundary. Implement this trait for
/// model runtimes; a blanket [`Embedder`] adapter is provided below.
#[async_trait::async_trait]
pub trait VectorGenerator: Send + Sync {
    /// Generate embeddings for the given texts.
    async fn generate(&self, texts: &[String]) -> Result<Vec<EmbeddingVector>, DripstoneError>;

    /// Dimension of embeddings produced by this generator.
    fn dimension(&self) -> usize;

    /// Name/identifier of this generator.
    fn name(&self) -> &str;
}

/// Adapter: every local [`VectorGenerator`] that is `Debug` is also an [`Embedder`].
#[async_trait::async_trait]
impl<T> Embedder for T
where
    T: VectorGenerator + Debug,
{
    async fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, DripstoneError> {
        let owned: Vec<String> = texts.iter().map(|text| (*text).to_owned()).collect();
        let vectors = self.generate(&owned).await?;
        Ok(vectors.into_iter().map(|vector| vector.to_vec()).collect())
    }

    async fn embed_query(&self, query: &str) -> Result<Vec<f32>, DripstoneError> {
        let vectors = self.generate(&[query.to_owned()]).await?;
        vectors
            .into_iter()
            .next()
            .map(|vector| vector.to_vec())
            .ok_or_else(|| DripstoneError::pipeline("local embedder returned no vector"))
    }

    fn dimension(&self) -> usize {
        VectorGenerator::dimension(self)
    }

    fn name(&self) -> &str {
        VectorGenerator::name(self)
    }
}

#[cfg(feature = "local-candle")]
pub mod candle;

#[cfg(feature = "local-ort")]
pub mod ort;

#[cfg(feature = "local-mlx")]
pub mod mlx;

#[cfg(feature = "local-candle")]
pub use candle::CandleEmbedder;

#[cfg(feature = "local-ort")]
pub use ort::OrtEmbedder;

#[cfg(feature = "local-mlx")]
pub use mlx::MlxEmbedder;
