#[cfg(feature = "tools-rag")]
use super::{
    AgentBuilder, Arc, CacheBackend, RagTool, WorkspaceEmbeddingMode, WorkspaceIndexerConfig,
};

impl AgentBuilder {
    #[must_use]
    pub fn shared_rag_tool(mut self, rag_tool: Arc<RagTool>) -> Self {
        self.shared_rag_tool = Some(rag_tool);
        self
    }

    /// Set the RAG namespace for document isolation (e.g. per-session). Applied to workspace RAG when set.
    #[cfg(feature = "tools-rag")]
    pub fn rag_namespace(mut self, namespace: impl Into<String>) -> Self {
        self.rag_namespace = Some(namespace.into());
        self
    }

    /// Set workspace RAG `top_k` (number of chunks to retrieve). Default 12.
    #[cfg(feature = "tools-rag")]
    #[must_use]
    pub const fn workspace_rag_top_k(mut self, top_k: usize) -> Self {
        self.workspace_rag_top_k = Some(top_k);
        self
    }

    /// Set workspace RAG minimum similarity threshold (0.0–1.0). Default 0.25.
    #[cfg(feature = "tools-rag")]
    #[must_use]
    pub const fn workspace_rag_min_similarity(mut self, min_similarity: f32) -> Self {
        self.workspace_rag_min_similarity = Some(min_similarity);
        self
    }

    /// Set optional short-TTL cache for workspace RAG context (key: `query+top_k+min_sim`).
    #[cfg(feature = "tools-rag")]
    pub fn workspace_rag_context_cache(
        mut self,
        cache: Arc<dyn CacheBackend>,
        ttl_secs: u64,
    ) -> Self {
        self.workspace_rag_context_cache = Some(cache);
        self.workspace_rag_context_cache_ttl_secs = ttl_secs;
        self
    }

    /// Enable automatic workspace embedding with auto-detected embedding provider.
    /// Indexes the current directory (respecting .gitignore), then uses embeddings to guide the agent.
    /// If no embedding provider is available, falls back to buffer memory.
    #[cfg(feature = "tools-rag")]
    #[must_use]
    pub fn with_auto_workspace_embedding(mut self) -> Self {
        self.workspace_embedding = Some(WorkspaceEmbeddingMode::Auto);
        self
    }

    /// Enable workspace embedding with a specific config.
    #[cfg(feature = "tools-rag")]
    #[must_use]
    pub fn with_workspace_embedding(mut self, config: WorkspaceIndexerConfig) -> Self {
        self.workspace_embedding = Some(WorkspaceEmbeddingMode::Custom(config));
        self
    }

    /// Explicitly disable workspace embedding.
    #[cfg(feature = "tools-rag")]
    #[must_use]
    pub fn without_workspace_embedding(mut self) -> Self {
        self.workspace_embedding = None;
        self
    }

    /// Supply a pre-discovered embedding provider so that `build()` can skip
    /// the potentially slow provider-probing step during workspace embedding setup.
    ///
    /// Useful when the caller has already run `discover_best_embedding_provider()`
    /// concurrently with other startup work.
    #[cfg(feature = "tools-rag")]
    #[must_use]
    pub fn with_pre_discovered_embedding(
        mut self,
        discovery: crate::discovery::embedding::EmbeddingProviderDiscovery,
    ) -> Self {
        self.pre_discovered_embedding = Some(discovery);
        self
    }
}
