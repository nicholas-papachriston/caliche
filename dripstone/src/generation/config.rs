//! Configuration for local embedding generation.

use crate::runtime::RuntimeConfig;
use serde::{Deserialize, Serialize};

/// Pooling strategy for generating embeddings from token embeddings.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum PoolingStrategy {
    /// Mean pooling (average of all token embeddings)
    #[default]
    Mean,
    /// CLS token pooling (use the [CLS] token embedding)
    Cls,
    /// Max pooling (element-wise maximum)
    Max,
}

/// Configuration for local embedding generation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingConfig {
    /// `HuggingFace` model repository ID (e.g., "sentence-transformers/all-MiniLM-L6-v2")
    pub model_repo: String,
    /// Optional revision/branch/tag (defaults to "main")
    pub revision: Option<String>,
    /// Pooling strategy for generating embeddings
    pub pooling: PoolingStrategy,
    /// Maximum sequence length (defaults to 512)
    pub max_length: usize,
    /// Whether to normalize embeddings (L2 normalization)
    pub normalize: bool,
    /// Runtime/device preference (Auto, Mlx, Cuda, Metal, Cpu)
    pub runtime: RuntimeConfig,
}

impl Default for EmbeddingConfig {
    fn default() -> Self {
        Self {
            model_repo: "sentence-transformers/all-MiniLM-L6-v2".to_string(),
            revision: None,
            pooling: PoolingStrategy::default(),
            max_length: 512,
            normalize: true,
            runtime: RuntimeConfig::default(),
        }
    }
}

impl EmbeddingConfig {
    /// Create a new embedding configuration with default settings.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the model repository ID.
    #[must_use]
    pub fn with_model_repo(mut self, repo: impl Into<String>) -> Self {
        self.model_repo = repo.into();
        self
    }

    /// Set the revision/branch/tag.
    #[must_use]
    pub fn with_revision(mut self, revision: impl Into<String>) -> Self {
        self.revision = Some(revision.into());
        self
    }

    /// Set the pooling strategy.
    #[must_use]
    pub const fn with_pooling(mut self, pooling: PoolingStrategy) -> Self {
        self.pooling = pooling;
        self
    }

    /// Set the maximum sequence length.
    #[must_use]
    pub const fn with_max_length(mut self, max_length: usize) -> Self {
        self.max_length = max_length;
        self
    }

    /// Set whether to normalize embeddings.
    #[must_use]
    pub const fn with_normalize(mut self, normalize: bool) -> Self {
        self.normalize = normalize;
        self
    }

    /// Set runtime/device preference (Auto, Mlx, Cuda, Metal, Cpu).
    #[must_use]
    pub const fn with_runtime(mut self, runtime: RuntimeConfig) -> Self {
        self.runtime = runtime;
        self
    }
}
