//! Local embedding providers using Candle or ONNX Runtime.

use async_trait::async_trait;

use crate::llm::config::CommonProviderOptions;
use crate::llm::embedding::{EmbeddingProvider, EmbeddingRequest, EmbeddingResponse};
use crate::messages::TokenUsage;
use std::sync::Arc;

#[cfg(any(feature = "embedding-candle", feature = "embedding-ort"))]
use dripstone::generation::{EmbeddingConfig as DripstoneEmbeddingConfig, VectorGenerator};

#[cfg(feature = "embedding-candle")]
use dripstone::generation::CandleEmbedder;

#[cfg(feature = "embedding-ort")]
use dripstone::generation::OrtEmbedder;

/// Configuration for LocalCandleProvider
#[cfg(feature = "embedding-candle")]
#[derive(Debug, Clone)]
pub struct LocalCandleProviderConfig {
    /// Embedding model configuration
    pub embedding_config: DripstoneEmbeddingConfig,
    /// Model name identifier
    pub model_name: String,
    /// Common provider options
    pub common: CommonProviderOptions,
}

#[cfg(feature = "embedding-candle")]
impl LocalCandleProviderConfig {
    /// Create a new LocalCandleProviderConfig with default settings
    pub fn new() -> Self {
        Self {
            embedding_config: DripstoneEmbeddingConfig::default(),
            model_name: "local-candle".to_string(),
            common: CommonProviderOptions::default(),
        }
    }

    /// Set the model repository ID
    pub fn with_model_repo(mut self, repo: impl Into<String>) -> Self {
        self.embedding_config = self.embedding_config.with_model_repo(repo);
        self.model_name = self.embedding_config.model_repo.clone();
        self
    }

    /// Set the revision/branch/tag
    pub fn with_revision(mut self, revision: impl Into<String>) -> Self {
        self.embedding_config = self.embedding_config.with_revision(revision);
        self
    }

    /// Set runtime/device preference (Auto, Mlx, Cuda, Metal, Cpu)
    pub fn with_runtime(mut self, runtime: dripstone::RuntimeConfig) -> Self {
        self.embedding_config = self.embedding_config.with_runtime(runtime);
        self
    }
}

#[cfg(feature = "embedding-candle")]
impl Default for LocalCandleProviderConfig {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(feature = "embedding-candle")]
crate::impl_provider_config!(LocalCandleProviderConfig);

/// Local embedding provider using Candle
#[cfg(feature = "embedding-candle")]
#[derive(Debug)]
pub struct LocalCandleProvider {
    embedder: Arc<CandleEmbedder>,
    model_name: String,
}

#[cfg(feature = "embedding-candle")]
impl LocalCandleProvider {
    /// Create a new LocalCandleProvider with configuration
    pub fn with_config(config: LocalCandleProviderConfig) -> anyhow::Result<Self> {
        let embedder = CandleEmbedder::new(config.embedding_config)
            .map_err(|e| anyhow::anyhow!("Failed to create Candle embedder: {}", e))?;

        Ok(Self {
            embedder: Arc::new(embedder),
            model_name: config.model_name,
        })
    }

    /// Create a new LocalCandleProvider with default settings
    pub fn new() -> anyhow::Result<Self> {
        Self::with_config(LocalCandleProviderConfig::default())
    }
}

#[cfg(feature = "embedding-candle")]
#[async_trait]
impl EmbeddingProvider for LocalCandleProvider {
    async fn embed(&self, request: EmbeddingRequest) -> anyhow::Result<EmbeddingResponse> {
        if request.texts.is_empty() {
            return Ok(EmbeddingResponse {
                embeddings: Vec::new(),
                usage: None,
                model: self.model_name.clone(),
            });
        }

        // Use the model from request if provided, otherwise use default
        let model_name = request.model.as_ref().unwrap_or(&self.model_name).clone();

        // Generate embeddings
        let embeddings = self
            .embedder
            .generate(&request.texts)
            .await
            .map_err(|e| anyhow::anyhow!("Failed to generate embeddings: {}", e))?;
        let embeddings = crate::llm::embedding::owned_embedding_rows(embeddings);

        // Estimate token usage (rough approximation: ~4 chars per token)
        let total_chars: usize = request.texts.iter().map(|t| t.len()).sum();
        let estimated_tokens = (total_chars / 4).max(1);

        Ok(EmbeddingResponse {
            embeddings,
            usage: Some(TokenUsage::from_embedding_tokens(
                estimated_tokens as u32,
                Some(estimated_tokens as u32),
            )),
            model: model_name,
        })
    }

    fn name(&self) -> &str {
        "local-candle"
    }
}

/// Configuration for LocalOrtProvider
#[cfg(feature = "embedding-ort")]
#[derive(Debug, Clone)]
pub struct LocalOrtProviderConfig {
    /// Embedding model configuration
    pub embedding_config: DripstoneEmbeddingConfig,
    /// Model name identifier
    pub model_name: String,
    /// Common provider options
    pub common: CommonProviderOptions,
}

#[cfg(feature = "embedding-ort")]
impl LocalOrtProviderConfig {
    /// Create a new LocalOrtProviderConfig with default settings
    pub fn new() -> Self {
        Self {
            embedding_config: DripstoneEmbeddingConfig::default(),
            model_name: "local-ort".to_string(),
            common: CommonProviderOptions::default(),
        }
    }

    /// Set the model repository ID
    pub fn with_model_repo(mut self, repo: impl Into<String>) -> Self {
        self.embedding_config = self.embedding_config.with_model_repo(repo);
        self.model_name = self.embedding_config.model_repo.clone();
        self
    }

    /// Set the revision/branch/tag
    pub fn with_revision(mut self, revision: impl Into<String>) -> Self {
        self.embedding_config = self.embedding_config.with_revision(revision);
        self
    }

    /// Set runtime/device preference (Auto, Mlx, Cuda, Metal, Cpu)
    pub fn with_runtime(mut self, runtime: dripstone::RuntimeConfig) -> Self {
        self.embedding_config = self.embedding_config.with_runtime(runtime);
        self
    }
}

#[cfg(feature = "embedding-ort")]
impl Default for LocalOrtProviderConfig {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(feature = "embedding-ort")]
crate::impl_provider_config!(LocalOrtProviderConfig);

/// Local embedding provider using ONNX Runtime
#[cfg(feature = "embedding-ort")]
#[derive(Debug)]
pub struct LocalOrtProvider {
    embedder: Arc<OrtEmbedder>,
    model_name: String,
}

#[cfg(feature = "embedding-ort")]
impl LocalOrtProvider {
    /// Create a new LocalOrtProvider with configuration
    pub fn with_config(config: LocalOrtProviderConfig) -> anyhow::Result<Self> {
        let embedder = OrtEmbedder::new(config.embedding_config)
            .map_err(|e| anyhow::anyhow!("Failed to create ORT embedder: {}", e))?;

        Ok(Self {
            embedder: Arc::new(embedder),
            model_name: config.model_name,
        })
    }

    /// Create a new LocalOrtProvider with default settings
    pub fn new() -> anyhow::Result<Self> {
        Self::with_config(LocalOrtProviderConfig::default())
    }
}

#[cfg(feature = "embedding-ort")]
#[async_trait]
impl EmbeddingProvider for LocalOrtProvider {
    async fn embed(&self, request: EmbeddingRequest) -> anyhow::Result<EmbeddingResponse> {
        if request.texts.is_empty() {
            return Ok(EmbeddingResponse {
                embeddings: Vec::new(),
                usage: None,
                model: self.model_name.clone(),
            });
        }

        // Use the model from request if provided, otherwise use default
        let model_name = request.model.as_ref().unwrap_or(&self.model_name).clone();

        // Generate embeddings
        let embeddings = self
            .embedder
            .generate(&request.texts)
            .await
            .map_err(|e| anyhow::anyhow!("Failed to generate embeddings: {}", e))?;
        let embeddings = crate::llm::embedding::owned_embedding_rows(embeddings);

        // Estimate token usage (rough approximation: ~4 chars per token)
        let total_chars: usize = request.texts.iter().map(|t| t.len()).sum();
        let estimated_tokens = (total_chars / 4).max(1);

        Ok(EmbeddingResponse {
            embeddings,
            usage: Some(TokenUsage::from_embedding_tokens(
                estimated_tokens as u32,
                Some(estimated_tokens as u32),
            )),
            model: model_name,
        })
    }

    fn name(&self) -> &str {
        "local-ort"
    }
}
