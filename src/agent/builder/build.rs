#[cfg(feature = "learning-loop")]
use super::Arc;
use super::{AgentBuilder, AgentService};

impl AgentBuilder {
    /// Build the `AgentService` with the configured options.
    ///
    /// This method is async because it may need to initialize providers and load resources.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Provider credentials are not configured
    /// - The specified model is not available
    /// - Required environment variables are missing
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use caliche::{AgentBuilder, ProviderType};
    ///
    /// #[tokio::main]
    /// async fn main() -> anyhow::Result<()> {
    ///     let agent = AgentBuilder::new()
    ///         .provider(ProviderType::OpenAI)
    ///         .model("gpt-4o".to_string())
    ///         .build()
    ///         .await?;
    ///     
    ///     Ok(())
    /// }
    /// ```
    pub async fn build(mut self) -> anyhow::Result<AgentService> {
        self.init_durable_stream_backends().await?;
        self.apply_provider_model_from_env();

        #[cfg(feature = "tools-rag")]
        let used_workspace_embedding = Box::pin(self.setup_workspace_embedding()).await;

        #[cfg(feature = "vector-memory")]
        let prebuilt_vector_backend = self.prebuild_vector_backend().await?;

        let mut service = Box::pin(self.construct_service()).await?;
        self.apply_prompt_and_limits(&mut service)?;
        self.apply_hooks_and_permissions(&mut service);
        self.apply_session_store(&mut service).await?;
        self.apply_driver_and_skills(&mut service);
        #[cfg(feature = "tools-rag")]
        self.apply_workspace_rag(&mut service, used_workspace_embedding);
        self.apply_output_and_compaction(&mut service)?;
        self.attach_in_process_mcp(&mut service);
        self.attach_remote_mcp(&mut service).await?;
        #[cfg(feature = "learning-loop")]
        let mut plugin_store_shared: Option<Arc<crate::plugins::persistence::PluginStore>> = None;
        self.attach_sql_and_plugins(
            &mut service,
            #[cfg(feature = "learning-loop")]
            &mut plugin_store_shared,
        )
        .await?;
        self.apply_learning_loop(
            &mut service,
            #[cfg(feature = "learning-loop")]
            plugin_store_shared,
        );
        self.apply_memory(
            &mut service,
            #[cfg(feature = "vector-memory")]
            prebuilt_vector_backend,
        )
        .await?;
        Ok(service)
    }
}
