#[cfg(feature = "tools-rag")]
use super::{AgentService, Arc, PathBuf, RagHashlineOptions, RagTool};

impl AgentService {
    /// Check if workspace RAG is configured for context enhancement.
    #[cfg(feature = "tools-rag")]
    #[must_use]
    pub const fn has_workspace_rag(&self) -> bool {
        self.workspace_indexing_ready.is_some() || self.agent.workspace_rag.is_some()
    }

    /// Returns configured workspace root when workspace indexing is enabled.
    #[cfg(feature = "tools-rag")]
    #[must_use]
    pub fn workspace_root(&self) -> Option<std::path::PathBuf> {
        self.workspace_config.as_ref().map(|c| c.root_path.clone())
    }

    /// Get the workspace indexing readiness handle, if workspace RAG is enabled.
    /// Use this to wait for or poll indexing completion (e.g. to show "Workspace indexed.").
    #[cfg(feature = "tools-rag")]
    #[must_use]
    pub fn workspace_indexing_handle(
        &self,
    ) -> Option<std::sync::Arc<crate::agent::workspace_indexing::WorkspaceIndexingReady>> {
        self.workspace_indexing_ready.clone()
    }

    /// Clear the current workspace index and start a fresh background re-index.
    ///
    /// Returns the workspace root that is being indexed.
    #[cfg(feature = "tools-rag")]
    pub async fn reindex_current_workspace_background(&mut self) -> anyhow::Result<PathBuf> {
        let config = match self.workspace_config.clone() {
            Some(config) => config,
            None => Self::default_workspace_config_for_reindex()?,
        };
        let rag = if let Some(rag) = self.rag_service() {
            rag
        } else {
            let rag_tool = self.create_workspace_rag_tool(&config).await?;
            let rag = rag_tool.service();
            self.rag_tool = Some(rag_tool);
            rag
        };

        tracing::info!(
            "Workspace re-index requested for {}",
            config.root_path.display()
        );

        rag.clear().await?;

        let fresh_manifest = crate::workspace::manifest::IndexManifest::with_fingerprint(
            config.index_fingerprint.clone(),
        );
        if let crate::workspace::PersistenceConfig::Disk { path } = &config.persistence {
            fresh_manifest.save(path).await?;
        }
        let manifest = Arc::new(tokio::sync::Mutex::new(fresh_manifest));
        let indexing_ready =
            Arc::new(crate::agent::workspace_indexing::WorkspaceIndexingReady::new());
        let ready = Arc::clone(&indexing_ready);
        let indexer = crate::workspace::WorkspaceIndexer::new(Arc::clone(&rag), config.clone());
        let root = config.root_path.clone();

        self.workspace_manifest = Some(manifest);
        self.workspace_indexing_ready = Some(Arc::clone(&indexing_ready));
        self.agent_mut()
            .set_workspace_rag(Some(Arc::clone(&rag)), Some(indexing_ready));
        self.agent_mut()
            .set_workspace_rag_retrieval(config.line_level_bm25, config.rag_search_options.clone());

        // Restart the watcher so subsequent live updates use the fresh manifest.
        self.file_watcher = None;
        if let Err(e) = self.start_file_watching() {
            tracing::warn!("File watcher restart after /reindex failed: {}", e);
        }

        tokio::spawn(async move {
            match indexer.index().await {
                Ok(res) => {
                    tracing::info!(
                        "Workspace re-index completed: {} files, {} chunks, {} failures",
                        res.files_indexed,
                        res.total_chunks,
                        res.failures.len()
                    );
                }
                Err(e) => {
                    tracing::warn!(
                        "Workspace re-index failed: {}. Continuing without fresh index.",
                        e
                    );
                }
            }
            ready.notify_done();
        });

        Ok(root)
    }

    #[cfg(feature = "tools-rag")]
    fn default_workspace_config_for_reindex()
    -> anyhow::Result<crate::workspace::WorkspaceIndexerConfig> {
        let root =
            std::env::current_dir().map_err(|e| anyhow::anyhow!("Current dir unavailable: {e}"))?;
        Ok(crate::workspace::WorkspaceIndexerConfig::new(root)
            .line_level_bm25(true)
            .with_env_overrides())
    }

    #[cfg(feature = "tools-rag")]
    async fn create_workspace_rag_tool(
        &self,
        config: &crate::workspace::WorkspaceIndexerConfig,
    ) -> anyhow::Result<Arc<RagTool>> {
        let embedding_provider = self.embedding_provider().ok_or_else(|| {
            anyhow::anyhow!(
                "No embedding provider configured. Run /embedding openai, /embedding bedrock, or /embedding lmstudio before /reindex."
            )
        })?;
        Self::build_workspace_rag_tool(config, embedding_provider).await
    }

    /// Build the workspace RAG tool for `config`.
    ///
    /// This probes the embedding provider once and, with disk persistence,
    /// lists the repositories under the root, so it can take seconds. It
    /// needs no service: interactive hosts run it off the input path and
    /// install the result with [`Self::attach_workspace_rag`].
    ///
    /// The tool's provider stops for the session after its first permanent
    /// error ([`crate::llm::StopOnPermanentError`]).
    ///
    /// # Errors
    ///
    /// Returns [`crate::llm::EmbeddingStopped`] when the probe gets an error
    /// a retry cannot fix, such as a model that is not there. Nothing else
    /// is sent to the provider.
    #[cfg(feature = "tools-rag")]
    pub async fn build_workspace_rag_tool(
        config: &crate::workspace::WorkspaceIndexerConfig,
        embedding_provider: Arc<dyn crate::llm::EmbeddingProvider>,
    ) -> anyhow::Result<Arc<RagTool>> {
        let embedding_provider: Arc<dyn crate::llm::EmbeddingProvider> =
            Arc::new(crate::llm::StopOnPermanentError::new(embedding_provider));
        let dimension =
            crate::discovery::embedding::probe_embedding_dimension_or_stop(&embedding_provider)
                .await
                .map_err(|error| crate::llm::EmbeddingStopped {
                    reason: error.to_string(),
                })?;
        let mut hashline_options = RagHashlineOptions::for_workspace(true, config.line_level_bm25);
        if std::env::var("CALICHE_RAG_LATE_INTERACTION").as_deref() == Ok("true") {
            hashline_options = hashline_options.late_interaction(true);
        }
        let tool = match &config.persistence {
            crate::workspace::PersistenceConfig::Disk { .. } => {
                let backend = crate::workspace::build_workspace_backend(config, dimension).await?;
                RagTool::with_backend_and_hashline_options(
                    embedding_provider,
                    backend,
                    None::<String>,
                    false,
                    hashline_options,
                )
            }
            crate::workspace::PersistenceConfig::InMemory => RagTool::new_with_hashline_options(
                embedding_provider,
                None::<String>,
                hashline_options,
            ),
        };

        let tool = if let Some(options) = config.rag_search_options.clone() {
            tool.with_default_search_options(options)
        } else {
            tool
        };

        Ok(Arc::new(tool))
    }

    /// Attach workspace RAG after the service is already built.
    ///
    /// Indexing runs in the background, but building the tool waits on the
    /// embedding provider. Hosts that must keep input responsive build the
    /// tool elsewhere and call [`Self::attach_workspace_rag`].
    #[cfg(feature = "tools-rag")]
    pub async fn enable_workspace_embedding(
        &mut self,
        config: crate::workspace::WorkspaceIndexerConfig,
        embedding: Arc<dyn crate::llm::EmbeddingProvider>,
    ) -> anyhow::Result<PathBuf> {
        let config = Self::workspace_embedding_config(config, embedding.as_ref());
        let rag_tool = Self::build_workspace_rag_tool(&config, Arc::clone(&embedding)).await?;
        self.attach_workspace_rag(config, embedding, rag_tool).await
    }

    /// `config` with the index fingerprint workspace embedding uses when the
    /// caller set none. Pass the result to both
    /// [`Self::build_workspace_rag_tool`] and [`Self::attach_workspace_rag`].
    #[cfg(feature = "tools-rag")]
    #[must_use]
    pub fn workspace_embedding_config(
        mut config: crate::workspace::WorkspaceIndexerConfig,
        embedding: &dyn crate::llm::EmbeddingProvider,
    ) -> crate::workspace::WorkspaceIndexerConfig {
        if config.index_fingerprint.is_none() {
            config.index_fingerprint = Some(format!(
                "provider={};model=deferred;line_bm25={}",
                embedding.name(),
                config.line_level_bm25
            ));
        }
        config
    }

    /// Install a tool from [`Self::build_workspace_rag_tool`] and start
    /// indexing in the background. Does not call the embedding provider.
    #[cfg(feature = "tools-rag")]
    pub async fn attach_workspace_rag(
        &mut self,
        config: crate::workspace::WorkspaceIndexerConfig,
        embedding: Arc<dyn crate::llm::EmbeddingProvider>,
        rag_tool: Arc<RagTool>,
    ) -> anyhow::Result<PathBuf> {
        let provider_id = self
            .embedding_provider_id()
            .map(str::to_owned)
            .or_else(|| Some(embedding.name().to_string()));
        let model_id = self.embedding_model_id().map(str::to_owned);
        self.set_embedding_selection(Some(embedding), provider_id, model_id);
        self.workspace_config = Some(config);
        self.rag_tool = Some(rag_tool);
        self.reindex_current_workspace_background().await
    }

    /// Set workspace RAG on the agent for context enhancement (query before each turn).
    /// Call this when workspace embedding was used so the agent injects relevant chunks.
    /// If `indexing_ready` is provided, the first RAG fetch will wait for indexing (with timeout).
    #[cfg(feature = "tools-rag")]
    pub fn set_workspace_rag_for_context(
        &mut self,
        indexing_ready: Option<Arc<crate::agent::workspace_indexing::WorkspaceIndexingReady>>,
        workspace_config: Option<crate::workspace::WorkspaceIndexerConfig>,
        workspace_manifest: Option<
            Arc<tokio::sync::Mutex<crate::workspace::manifest::IndexManifest>>,
        >,
    ) {
        if let Some(rag) = self.rag_service() {
            self.workspace_indexing_ready = indexing_ready.as_ref().map(Arc::clone);
            self.workspace_config = workspace_config;
            self.workspace_manifest = workspace_manifest;
            let retrieval = self
                .workspace_config
                .as_ref()
                .map(|config| (config.line_level_bm25, config.rag_search_options.clone()));
            self.agent_mut()
                .set_workspace_rag(Some(rag), indexing_ready);
            if let Some((line_level_bm25, search_options)) = retrieval {
                self.agent_mut()
                    .set_workspace_rag_retrieval(line_level_bm25, search_options);
            }
        }
    }

    /// Start file watching for workspace changes (REPL/TUI mode).
    ///
    /// Spawns a background file watcher that detects changes and re-indexes modified
    /// files. With Disk persistence the manifest is saved to disk; with `InMemory`
    /// the manifest is kept in memory only for the session.
    #[cfg(feature = "tools-rag")]
    pub fn start_file_watching(&mut self) -> anyhow::Result<()> {
        // Check prerequisites
        if self.rag_tool.is_none() {
            tracing::debug!("File watcher: no RAG tool configured, skipping");
            return Ok(());
        }

        let Some(config) = self.workspace_config.as_ref() else {
            tracing::debug!("File watcher: no workspace config, skipping");
            return Ok(());
        };

        let Some(manifest) = self.workspace_manifest.as_ref() else {
            tracing::debug!("File watcher: no manifest, skipping");
            return Ok(());
        };

        let Some(rag_tool) = self.rag_tool.as_ref() else {
            return Ok(());
        };

        // Validate root path exists before handing to notify
        if !config.root_path.exists() {
            tracing::debug!(
                "File watcher: root path does not exist ({}), skipping",
                config.root_path.display()
            );
            return Ok(());
        }

        // Load manifest for watcher (block_in_place to avoid blocking async runtime)
        let manifest_clone = tokio::task::block_in_place(|| {
            let manifest_guard = manifest.blocking_lock();
            manifest_guard.clone()
        });

        // Create watcher
        let watcher = crate::workspace::WorkspaceWatcher::new(
            rag_tool.service(),
            config.clone(),
            manifest_clone,
        )?;

        self.file_watcher = Some(Arc::new(tokio::sync::Mutex::new(watcher)));
        tracing::info!("File watcher: started for {}", config.root_path.display());

        Ok(())
    }

    /// Get a reference to the RAG service if available.
    /// Returns None if the RAG tool is not registered or not available.
    ///
    /// This provides direct access to the `RagService` API for use in pipelines,
    /// builders, workflows, etc.
    ///
    /// Note: For now, use the convenience methods like `rag_index_document`, `rag_query`, etc.
    /// Direct access to `RagService` requires storing it separately or implementing proper downcasting.
    #[cfg(feature = "tools-rag")]
    #[must_use]
    pub fn rag_service(&self) -> Option<Arc<crate::rag::RagService>> {
        self.rag_tool.as_ref().map(|t| t.service())
    }

    /// Index a document into the RAG system.
    /// Returns the number of chunks created and token usage.
    ///
    /// # Errors
    /// Returns an error if the RAG tool is not available or indexing fails.
    pub async fn rag_index_document(
        &self,
        document: &str,
        doc_id: &str,
    ) -> anyhow::Result<(usize, Option<crate::messages::TokenUsage>)> {
        let rag_service = self
            .rag_service()
            .ok_or_else(|| anyhow::anyhow!("RAG service not available"))?;
        rag_service.index_document(document, doc_id).await
    }

    /// Batch index multiple documents into the RAG system.
    /// Returns total chunks, successful document count, failures, and token usage.
    pub async fn rag_index_batch(
        &self,
        documents: Vec<(String, String)>, // (doc_id, document)
    ) -> anyhow::Result<(
        usize,
        usize,
        Vec<String>,
        Option<crate::messages::TokenUsage>,
    )> {
        let rag_service = self
            .rag_service()
            .ok_or_else(|| anyhow::anyhow!("RAG service not available"))?;
        rag_service.index_batch(documents).await
    }

    /// Query the RAG system for relevant documents.
    /// Returns formatted results, top similarity score, and token usage.
    pub async fn rag_query(
        &self,
        query: &str,
        top_k: Option<usize>,
        min_similarity: Option<f32>,
    ) -> anyhow::Result<(
        Vec<String>,
        Option<f32>,
        Option<crate::messages::TokenUsage>,
    )> {
        let rag_service = self
            .rag_service()
            .ok_or_else(|| anyhow::anyhow!("RAG service not available"))?;
        rag_service
            .query(query, top_k.unwrap_or(5), min_similarity.unwrap_or(0.3))
            .await
    }

    /// Get RAG system statistics.
    /// Returns (`total_chunks`, `total_documents`).
    pub async fn rag_stats(&self) -> anyhow::Result<(usize, usize)> {
        let rag_service = self
            .rag_service()
            .ok_or_else(|| anyhow::anyhow!("RAG service not available"))?;
        rag_service.stats().await
    }

    /// Clear all documents from the RAG system.
    pub async fn rag_clear(&self) -> anyhow::Result<()> {
        let rag_service = self
            .rag_service()
            .ok_or_else(|| anyhow::anyhow!("RAG service not available"))?;
        rag_service.clear().await
    }
}
