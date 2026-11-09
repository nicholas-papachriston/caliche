use async_trait::async_trait;
use serde::{Deserialize, Serialize};

/// Request for generating embeddings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingRequest {
    /// Texts to embed
    pub texts: Vec<String>,
    /// Optional model name (if not provided, provider will use default)
    pub model: Option<String>,
}

/// Response containing embeddings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingResponse {
    /// Vector embeddings, one per input text
    pub embeddings: Vec<Vec<f32>>,
    /// Token usage information
    pub usage: Option<crate::messages::TokenUsage>,
    /// Model used for generating embeddings
    pub model: String,
}

/// Copy shared embedding rows into the owned vectors this crate returns.
#[cfg(any(feature = "embedding-candle", feature = "embedding-ort"))]
pub(crate) fn owned_embedding_rows(vectors: Vec<std::sync::Arc<[f32]>>) -> Vec<Vec<f32>> {
    vectors
        .into_iter()
        .map(|vector| vector.as_ref().to_vec())
        .collect()
}

/// Trait for providers that support text embeddings
#[async_trait]
pub trait EmbeddingProvider: Send + Sync + std::fmt::Debug {
    /// Generate embeddings for the given texts
    async fn embed(&self, request: EmbeddingRequest) -> anyhow::Result<EmbeddingResponse>;

    /// Get the name of this embedding provider
    fn name(&self) -> &str;

    /// Why this provider sends no more requests this session, if it stopped
    /// (see [`StopOnPermanentError`]).
    fn stopped_reason(&self) -> Option<String> {
        None
    }
}

/// Embeddings stopped for the session after an error a retry cannot fix.
#[derive(Debug, Clone)]
pub struct EmbeddingStopped {
    /// The provider's first permanent error.
    pub reason: String,
}

impl std::fmt::Display for EmbeddingStopped {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.reason)
    }
}

impl std::error::Error for EmbeddingStopped {}

/// True when a retry cannot fix `error`: the server answered with a client
/// error other than a timeout or a rate limit (for example 400 or 404 for a
/// model that is not there), or embeddings already stopped.
#[must_use]
pub fn is_permanent_embedding_error(error: &anyhow::Error) -> bool {
    error.chain().any(|cause| {
        cause.is::<EmbeddingStopped>()
            || cause
                .downcast_ref::<crate::traits::retry::RetryError>()
                .is_some_and(|retry| !retry.is_retryable())
    })
}

/// An embedding provider that stops for the session after its first
/// permanent error: every later call fails at once with [`EmbeddingStopped`]
/// and sends nothing.
#[derive(Debug)]
pub struct StopOnPermanentError {
    inner: std::sync::Arc<dyn EmbeddingProvider>,
    stopped: std::sync::OnceLock<String>,
}

impl StopOnPermanentError {
    #[must_use]
    pub fn new(inner: std::sync::Arc<dyn EmbeddingProvider>) -> Self {
        Self {
            inner,
            stopped: std::sync::OnceLock::new(),
        }
    }
}

#[async_trait]
impl EmbeddingProvider for StopOnPermanentError {
    async fn embed(&self, request: EmbeddingRequest) -> anyhow::Result<EmbeddingResponse> {
        if let Some(reason) = self.stopped.get() {
            return Err(EmbeddingStopped {
                reason: reason.clone(),
            }
            .into());
        }
        let result = self.inner.embed(request).await;
        if let Err(error) = &result
            && is_permanent_embedding_error(error)
            && self.stopped.set(error.to_string()).is_ok()
        {
            tracing::warn!(
                "{} embeddings stopped for this session: {error}",
                self.inner.name()
            );
        }
        result
    }

    fn name(&self) -> &str {
        self.inner.name()
    }

    fn stopped_reason(&self) -> Option<String> {
        self.stopped
            .get()
            .cloned()
            .or_else(|| self.inner.stopped_reason())
    }
}
