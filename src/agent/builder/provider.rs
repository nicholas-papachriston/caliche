#[cfg(feature = "vector-memory")]
use super::VectorBackendConfig;
use super::{
    AgentBuilder, Arc, CredentialProvider, Credentials, ProviderType, SqlToolsConfig,
    StaticCredentialProvider,
};

impl AgentBuilder {
    #[must_use]
    pub const fn provider(mut self, provider: ProviderType) -> Self {
        self.provider = Some(provider);
        self
    }

    /// Provide a fully constructed provider instance (advanced).
    ///
    /// This is primarily intended for external integrations (e.g. JS bindings) that
    /// need to supply a custom provider implementation without going through built-in
    /// credential loading and provider factories.
    ///
    /// When set, `build()` will use this provider instance directly.
    pub fn provider_instance(mut self, provider: Arc<dyn crate::llm::Provider>) -> Self {
        self.provider_instance = Some(provider);
        // Default provider type for externally supplied providers.
        if self.provider.is_none() {
            self.provider = Some(ProviderType::External);
        }
        self
    }

    /// Provide a fully constructed embedding provider instance (advanced).
    ///
    /// When set, `build()` will pass it into tool registry construction (RAG) and
    /// into the created `AgentService` instance.
    pub fn embedding_provider_instance(
        mut self,
        embedding_provider: Arc<dyn crate::llm::EmbeddingProvider>,
    ) -> Self {
        self.embedding_provider_instance = Some(embedding_provider);
        self
    }

    /// Provide a fully constructed image generation provider instance.
    pub fn image_generation_provider_instance(
        mut self,
        image_generation_provider: Arc<dyn crate::llm::ImageGenerationProvider>,
    ) -> Self {
        self.image_generation_provider_instance = Some(image_generation_provider);
        self
    }

    /// Set the artifact store used for generated outputs such as images.
    #[must_use]
    pub fn artifact_store(mut self, artifact_store: Arc<crate::artifacts::ArtifactStore>) -> Self {
        self.artifact_store = Some(artifact_store);
        self
    }

    /// Set the model to use. Provider-specific model identifiers.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use caliche::AgentBuilder;
    ///
    /// let builder = AgentBuilder::new()
    ///     .model("gpt-4o");
    /// ```
    pub fn model(mut self, model: impl Into<String>) -> Self {
        self.model = Some(model.into());
        self
    }

    /// Set a custom system prompt for the agent.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use caliche::AgentBuilder;
    ///
    /// let builder = AgentBuilder::new()
    ///     .system_prompt("You are a helpful coding assistant.");
    /// ```
    pub fn system_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.system_prompt = Some(prompt.into());
        self
    }

    pub fn api_key(mut self, api_key: impl Into<String>) -> Self {
        self.credential_provider = Some(Arc::new(StaticCredentialProvider::new(
            Credentials::ApiKey(api_key.into()),
        )));
        self
    }

    /// Set a custom credential provider.
    ///
    /// This allows you to use custom authentication sources like Vault,
    /// AWS Secrets Manager, or other credential management systems.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use caliche::{AgentBuilder, ProviderType};
    /// use caliche::traits::credentials::{CredentialProvider, Credentials, StaticCredentialProvider};
    /// use std::sync::Arc;
    ///
    /// # async fn example() -> anyhow::Result<()> {
    /// let cred_provider = Arc::new(StaticCredentialProvider::new(
    ///     Credentials::ApiKey("sk-...".to_string())
    /// ));
    /// let agent = AgentBuilder::new()
    ///     .provider(ProviderType::OpenAI)
    ///     .credential_provider(cred_provider)
    ///     .build()
    ///     .await?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn credential_provider(mut self, provider: Arc<dyn CredentialProvider>) -> Self {
        self.credential_provider = Some(provider);
        self
    }

    /// Keep building when the selected provider has no model key or its base
    /// URL is egress-blocked.
    ///
    /// Completions fail later with that provider's credential or egress
    /// error. The builder does not switch to another provider.
    #[must_use]
    pub const fn defer_missing_credentials(mut self, defer: bool) -> Self {
        self.defer_missing_credentials = defer;
        self
    }

    /// Set AWS credentials for Bedrock provider.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use caliche::{AgentBuilder, ProviderType};
    ///
    /// # async fn example() -> anyhow::Result<()> {
    /// let agent = AgentBuilder::new()
    ///     .provider(ProviderType::Bedrock)
    ///     .aws_credentials(
    ///         "AWS_ACCESS_KEY_ID_EXAMPLE".to_string(),
    ///         concat!("wJalrXUtnFEMI/K7MDENG", "/bPxRfiCYEXAMPLEKEY").to_string(),
    ///         "us-east-1".to_string(),
    ///     )
    ///     .build()
    ///     .await?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn aws_credentials(
        mut self,
        access_key_id: impl Into<String>,
        secret_access_key: impl Into<String>,
        region: impl Into<String>,
    ) -> Self {
        self.credential_provider = Some(Arc::new(StaticCredentialProvider::new(
            Credentials::AwsAuth {
                access_key: access_key_id.into(),
                secret_key: secret_access_key.into(),
                session_token: None,
                region: Some(region.into()),
            },
        )));
        self
    }

    /// Set AWS credentials with session token for Bedrock provider.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use caliche::{AgentBuilder, ProviderType};
    ///
    /// # async fn example() -> anyhow::Result<()> {
    /// let agent = AgentBuilder::new()
    ///     .provider(ProviderType::Bedrock)
    ///     .aws_credentials_with_token(
    ///         "AWS_ACCESS_KEY_ID_EXAMPLE".to_string(),
    ///         concat!("wJalrXUtnFEMI/K7MDENG", "/bPxRfiCYEXAMPLEKEY").to_string(),
    ///         "session-token".to_string(),
    ///         "us-east-1".to_string(),
    ///     )
    ///     .build()
    ///     .await?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn aws_credentials_with_token(
        mut self,
        access_key_id: impl Into<String>,
        secret_access_key: impl Into<String>,
        session_token: impl Into<String>,
        region: impl Into<String>,
    ) -> Self {
        self.credential_provider = Some(Arc::new(StaticCredentialProvider::new(
            Credentials::AwsAuth {
                access_key: access_key_id.into(),
                secret_key: secret_access_key.into(),
                session_token: Some(session_token.into()),
                region: Some(region.into()),
            },
        )));
        self
    }

    /// Set AWS bearer token for Bedrock provider.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use caliche::{AgentBuilder, ProviderType};
    ///
    /// # async fn example() -> anyhow::Result<()> {
    /// let agent = AgentBuilder::new()
    ///     .provider(ProviderType::Bedrock)
    ///     .aws_bearer_token("bearer-token".to_string(), "us-east-1".to_string())
    ///     .build()
    ///     .await?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn aws_bearer_token(
        mut self,
        bearer_token: impl Into<String>,
        _region: impl Into<String>,
    ) -> Self {
        self.credential_provider = Some(Arc::new(StaticCredentialProvider::new(
            Credentials::Bearer(bearer_token.into()),
        )));
        self
    }

    /// Set the base URL for LM Studio provider.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use caliche::{AgentBuilder, ProviderType};
    ///
    /// # async fn example() -> anyhow::Result<()> {
    /// let agent = AgentBuilder::new()
    ///     .provider(ProviderType::LMStudio)
    ///     .base_url("http://localhost:1234/v1".to_string())
    ///     .build()
    ///     .await?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = Some(base_url.into());
        self
    }

    /// Set a specific vector store configuration for RAG.
    ///
    /// This allows you to use persistent vector stores like Qdrant, Pinecone,
    /// `PostgreSQL` with pgvector, Redis, Chroma, or S3 instead of the default
    /// in-memory backend.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use caliche::{AgentBuilder, ProviderType};
    /// # #[cfg(feature = "vector-store-qdrant")]
    /// use caliche::vecstore::VectorBackendConfig;
    ///
    /// # async fn example() -> anyhow::Result<()> {
    /// # #[cfg(feature = "vector-store-qdrant")]
    /// let agent = AgentBuilder::new()
    ///     .provider(ProviderType::OpenAI)
    ///     .vector_backend_config(VectorBackendConfig::Qdrant {
    ///         url: "http://localhost:6333".to_string(),
    ///         collection_prefix: Some("kb_".to_string()),
    ///         vector_dimension: 1536,
    ///         api_key: None,
    ///         batch_size: None,
    ///         timeout: None,
    ///     })
    ///     .enable_tool("fs")
    ///     .build()
    ///     .await?;
    /// # Ok(())
    /// # }
    /// ```
    #[cfg(feature = "vector-memory")]
    #[must_use]
    pub fn vector_backend_config(mut self, config: VectorBackendConfig) -> Self {
        self.vector_backend_config = Some(config);
        self
    }

    /// Auto-detect vector store configuration from environment variables.
    ///
    /// Enable metrics collection for vector operations
    #[must_use]
    pub const fn with_metrics(mut self, enabled: bool) -> Self {
        self.vector_metrics_enabled = enabled;
        self
    }

    /// Reads the `VECTOR_BACKEND` environment variable to determine
    /// which backend to use, along with backend-specific configuration.
    ///
    /// Falls back to in-memory backend if not specified.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use caliche::{AgentBuilder, ProviderType};
    ///
    /// # async fn example() -> anyhow::Result<()> {
    /// // Set VECTOR_BACKEND=qdrant and QDRANT_URL=http://localhost:6333
    /// let agent = AgentBuilder::new()
    ///     .provider(ProviderType::OpenAI)
    ///     .vector_backend_from_env()
    ///     .enable_tool("fs")
    ///     .build()
    ///     .await?;
    /// # Ok(())
    /// # }
    /// ```
    #[cfg(feature = "vector-memory")]
    pub fn vector_backend_from_env(mut self) -> Self {
        match VectorBackendConfig::from_env() {
            Ok(config) => {
                tracing::info!("Vector store configured from environment: {:?}", config);
                self.vector_backend_config = Some(config);
            }
            Err(e) => {
                tracing::warn!(
                    "Failed to configure vector store from environment: {}. Using default in-memory backend.",
                    e
                );
                self.vector_backend_config = Some(VectorBackendConfig::memory());
            }
        }
        self
    }

    /// Enable SQL generation and execution tools backed by `PostgreSQL`.
    ///
    /// This connects to the provided database, normalizes the schema into embeddings,
    /// and wires SQL tools that can generate queries using the loaded schema context.
    pub fn enable_sql_tools_with_postgres(
        mut self,
        database_url: impl Into<String>,
        namespace: impl Into<String>,
    ) -> Self {
        self.sql_tools = Some(SqlToolsConfig {
            database_url: database_url.into(),
            namespace: namespace.into(),
            top_k: 5,
            max_rows: 200,
        });
        self
    }

    /// Override the number of schema chunks retrieved for SQL generation context.
    pub fn sql_tools_top_k(mut self, top_k: usize) -> Self {
        if let Some(ref mut config) = self.sql_tools {
            config.top_k = top_k.clamp(1, 12);
        } else {
            tracing::warn!(
                "sql_tools_top_k called before enable_sql_tools_with_postgres; configuration ignored"
            );
        }
        self
    }

    /// Override the maximum number of rows returned by the SQL execution tool.
    pub fn sql_tools_max_rows(mut self, max_rows: usize) -> Self {
        if let Some(ref mut config) = self.sql_tools {
            config.max_rows = max_rows.max(1);
        } else {
            tracing::warn!(
                "sql_tools_max_rows called before enable_sql_tools_with_postgres; configuration ignored"
            );
        }
        self
    }

    /// Set the maximum number of tool call iterations (default: 50).
    ///
    /// By default, agents have a limit of 50 tool call iterations. Use this method
    /// to set a different limit, or set to `usize::MAX` for unlimited iterations.
    /// Each iteration consists of one LLM call and its tool executions.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use caliche::{AgentBuilder, ProviderType};
    ///
    /// # async fn example() -> anyhow::Result<()> {
    /// // Default limit of 50 iterations
    /// let agent1 = AgentBuilder::new()
    ///     .provider(ProviderType::OpenAI)
    ///     .build()
    ///     .await?;
    ///
    /// // Limited to 20 iterations
    /// let agent2 = AgentBuilder::new()
    ///     .provider(ProviderType::OpenAI)
    ///     .max_iterations(20) // Limit to 20 iterations
    ///     .build()
    ///     .await?;
    ///
    /// // Unlimited iterations
    /// let agent3 = AgentBuilder::new()
    ///     .provider(ProviderType::OpenAI)
    ///     .max_iterations(usize::MAX) // Unlimited iterations
    ///     .build()
    ///     .await?;
    /// # Ok(())
    /// # }
    /// ```
    #[must_use]
    pub const fn max_iterations(mut self, max_iterations: usize) -> Self {
        self.max_iterations = Some(max_iterations);
        self
    }

    /// Set the tool choice preference (default: Auto)
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use caliche::{AgentBuilder, ProviderType};
    /// use caliche::messages::ToolChoice;
    ///
    /// # async fn example() -> anyhow::Result<()> {
    /// let agent = AgentBuilder::new()
    ///     .provider(ProviderType::Anthropic)
    ///     .tool_choice(Some(ToolChoice::Required))
    ///     .build()
    ///     .await?;
    /// # Ok(())
    /// # }
    /// ```
    #[must_use]
    pub fn tool_choice(mut self, tool_choice: Option<crate::messages::ToolChoice>) -> Self {
        self.tool_choice = tool_choice;
        self
    }

    /// Set response format for structured outputs
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use caliche::{AgentBuilder, ProviderType};
    /// use caliche::output::ResponseFormat;
    ///
    /// # async fn example() -> anyhow::Result<()> {
    /// let agent = AgentBuilder::new()
    ///     .provider(ProviderType::OpenAI)
    ///     .model("gpt-4o")
    ///     .response_format(ResponseFormat::JsonObject)
    ///     .build()
    ///     .await?;
    /// # Ok(())
    /// # }
    /// ```
    #[must_use]
    pub fn response_format(mut self, format: crate::output::ResponseFormat) -> Self {
        self.response_format = Some(format);
        self
    }
}
