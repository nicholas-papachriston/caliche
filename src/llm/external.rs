//! External provider implementation for callback-based LLM providers.
//!
//! This module provides `ExternalProvider` and `ExternalEmbeddingProvider` that
//! accept completion/embedding callbacks from Python/JS bindings, enabling
//! integration with `LangChain` and other external LLM libraries.

use crate::llm::embedding::{EmbeddingProvider, EmbeddingRequest, EmbeddingResponse};
use crate::llm::provider::Provider;
use crate::messages::{CompletionRequest, CompletionResponse, StreamEvent};
use async_trait::async_trait;
use futures_util::Stream;
use std::pin::Pin;
use std::sync::Arc;

/// Trait for external completion callbacks
///
/// This trait allows Python/JS bindings to provide completion functions
/// that will be called by the `ExternalProvider`.
#[async_trait]
pub trait ExternalCompletionCallback: Send + Sync {
    /// Perform a completion request
    async fn complete(&self, request: CompletionRequest) -> anyhow::Result<CompletionResponse>;

    /// Stream completion tokens
    fn stream(
        &self,
        request: CompletionRequest,
    ) -> Pin<Box<dyn Stream<Item = anyhow::Result<StreamEvent>> + Send + '_>>;
}

/// Trait for external embedding callbacks
#[async_trait]
pub trait ExternalEmbeddingCallback: Send + Sync {
    /// Generate embeddings for the given texts
    async fn embed(&self, request: EmbeddingRequest) -> anyhow::Result<EmbeddingResponse>;
}

/// External provider that delegates to callback functions
pub struct ExternalProvider {
    name: String,
    completion_callback: Arc<dyn ExternalCompletionCallback>,
    supports_native_tools: bool,
    supports_streaming: bool,
}

impl std::fmt::Debug for ExternalProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ExternalProvider")
            .field("name", &self.name)
            .field("supports_native_tools", &self.supports_native_tools)
            .field("supports_streaming", &self.supports_streaming)
            .field("completion_callback", &"<callback>")
            .finish()
    }
}

impl ExternalProvider {
    /// Create a new `ExternalProvider` with a completion callback
    pub fn new(
        name: impl Into<String>,
        completion_callback: Arc<dyn ExternalCompletionCallback>,
        supports_native_tools: bool,
        supports_streaming: bool,
    ) -> Self {
        Self {
            name: name.into(),
            completion_callback,
            supports_native_tools,
            supports_streaming,
        }
    }
}

#[async_trait]
impl Provider for ExternalProvider {
    async fn complete(&self, request: CompletionRequest) -> anyhow::Result<CompletionResponse> {
        self.completion_callback.complete(request).await
    }

    fn stream(
        &self,
        request: CompletionRequest,
    ) -> Pin<Box<dyn Stream<Item = anyhow::Result<StreamEvent>> + Send + '_>> {
        self.completion_callback.stream(request)
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn supports_native_tools(&self) -> bool {
        self.supports_native_tools
    }

    fn supports_streaming(&self) -> bool {
        self.supports_streaming
    }
}

/// External embedding provider that delegates to callback functions
pub struct ExternalEmbeddingProvider {
    name: String,
    embedding_callback: Arc<dyn ExternalEmbeddingCallback>,
}

impl std::fmt::Debug for ExternalEmbeddingProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ExternalEmbeddingProvider")
            .field("name", &self.name)
            .field("embedding_callback", &"<callback>")
            .finish()
    }
}

impl ExternalEmbeddingProvider {
    /// Create a new `ExternalEmbeddingProvider` with an embedding callback
    pub fn new(
        name: impl Into<String>,
        embedding_callback: Arc<dyn ExternalEmbeddingCallback>,
    ) -> Self {
        Self {
            name: name.into(),
            embedding_callback,
        }
    }
}

#[async_trait]
impl EmbeddingProvider for ExternalEmbeddingProvider {
    async fn embed(&self, request: EmbeddingRequest) -> anyhow::Result<EmbeddingResponse> {
        self.embedding_callback.embed(request).await
    }

    fn name(&self) -> &str {
        &self.name
    }
}
