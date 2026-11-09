//! MLX-based embedding generation (Apple Silicon).
//!
//! Feature-gated on `local-mlx` and only active when `target_vendor = "apple"`.
//! Uses the shared runtime config for device selection.

use crate::embedding::EmbeddingVector;
use crate::errors::DripstoneError;
use crate::generation::{EmbeddingConfig, VectorGenerator};
use crate::loader::{HuggingFaceLoader, load_tokenizer_from_path};
use crate::runtime::RuntimeConfig;
use std::sync::Arc;

#[cfg(target_vendor = "apple")]
use tokenizers::Tokenizer;

/// MLX-based embedder for generating embeddings on Apple Silicon.
///
/// When runtime is `Auto` or `Mlx`, uses MLX; otherwise creation fails with a
/// suggestion to use Candle or ORT.
#[cfg(all(feature = "local-mlx", target_vendor = "apple"))]
#[allow(
    dead_code,
    reason = "config and tokenizer are kept for the MLX forward pass; this build host cannot compile the MLX features, so an expect here could not be checked"
)]
pub struct MlxEmbedder {
    config: EmbeddingConfig,
    tokenizer: Arc<Tokenizer>,
    dimension: usize,
}

#[cfg(feature = "local-mlx")]
impl std::fmt::Debug for MlxEmbedder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MlxEmbedder")
            .field("dimension", &self.dimension)
            .finish_non_exhaustive()
    }
}

#[cfg(all(feature = "local-mlx", target_vendor = "apple"))]
impl MlxEmbedder {
    /// Create a new MLX embedder with the given configuration.
    ///
    /// Resolves `config.runtime`: only `RuntimeConfig::Auto` or `RuntimeConfig::Mlx`
    /// are accepted when using this backend.
    pub fn new(config: EmbeddingConfig) -> Result<Self, DripstoneError> {
        let resolved = config.runtime.resolve_auto();
        match resolved.selected {
            RuntimeConfig::Mlx | RuntimeConfig::Auto => {}
            other => {
                return Err(DripstoneError::Misconfiguration(format!(
                    "MlxEmbedder requires runtime Auto or Mlx, got {:?}; use Candle or ORT embedder for other runtimes",
                    other
                )));
            }
        }
        if let Some(reason) = &resolved.reason {
            tracing::debug!("mlx embedder runtime: {}", reason);
        }

        let loader = HuggingFaceLoader::new(&config.model_repo, config.revision.as_deref())?;
        let tokenizer_path = loader.get_file("tokenizer.json")?;
        let tokenizer = load_tokenizer_from_path(&tokenizer_path)?;

        // Dimension from config or common default (BERT base)
        let dimension = 384;

        Ok(Self {
            config,
            tokenizer: Arc::new(tokenizer),
            dimension,
        })
    }
}

#[cfg(all(feature = "local-mlx", target_vendor = "apple"))]
#[async_trait::async_trait]
impl VectorGenerator for MlxEmbedder {
    async fn generate(&self, texts: &[String]) -> Result<Vec<EmbeddingVector>, DripstoneError> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        // MLX BERT forward not yet implemented; use Candle or ORT for embeddings.
        // Tokenizer is loaded so we could add a minimal forward (embedding layer + pool) later.
        let _ = texts;
        Err(DripstoneError::Misconfiguration(
            "MLX embedding forward pass is not yet implemented; use embedding-candle (Metal) or embedding-ort for local embeddings".to_string(),
        ))
    }

    fn dimension(&self) -> usize {
        self.dimension
    }

    fn name(&self) -> &str {
        "mlx"
    }
}

/// Stub when MLX is enabled but not on Apple (e.g. Linux with feature for build consistency).
#[cfg(all(feature = "local-mlx", not(target_vendor = "apple")))]
pub struct MlxEmbedder;

#[cfg(all(feature = "local-mlx", not(target_vendor = "apple")))]
impl MlxEmbedder {
    pub fn new(_config: EmbeddingConfig) -> Result<Self, DripstoneError> {
        Err(DripstoneError::Misconfiguration(
            "MlxEmbedder is only available on Apple Silicon (target_vendor = apple)".to_string(),
        ))
    }
}
