use crate::agent::tools::common::params;
use crate::agent::tools::config::CommonToolOptions;
use crate::agent::tools::credentials::{CredentialResolver, ToolCredential};
use crate::agent::tools::{Tool, ToolMeta, ToolResult};
use crate::llm::openai_compat::PerplexityConfig;
use crate::llm::{PerplexityProvider, Provider};
use crate::messages::{CompletionRequest, Message, MessageRole};
use crate::utils::user_context::UserContext;
use async_trait::async_trait;
use serde_json::json;
use std::sync::Arc;
use std::time::Duration;

/// Configuration for `PerplexitySearchTool`
#[derive(Debug, Clone)]
pub struct PerplexitySearchToolConfig {
    /// Perplexity API key
    pub api_key: Option<String>,
    /// Perplexity model
    pub model: Option<String>,
    /// Common tool options
    pub common: CommonToolOptions,
}

impl Default for PerplexitySearchToolConfig {
    fn default() -> Self {
        Self::new()
    }
}

impl PerplexitySearchToolConfig {
    /// Create a new `PerplexitySearchToolConfig`
    #[must_use]
    pub fn new() -> Self {
        Self {
            api_key: None,
            model: None,
            common: CommonToolOptions::default(),
        }
    }

    /// Create a builder for `PerplexitySearchToolConfig`
    #[must_use]
    pub fn builder() -> PerplexitySearchToolConfigBuilder {
        PerplexitySearchToolConfigBuilder::default()
    }
}

/// Builder for `PerplexitySearchToolConfig`
#[derive(Debug, Default)]
pub struct PerplexitySearchToolConfigBuilder {
    api_key: Option<String>,
    model: Option<String>,
    common: CommonToolOptions,
}

impl PerplexitySearchToolConfigBuilder {
    /// Set the API key
    pub fn api_key(mut self, api_key: impl Into<String>) -> Self {
        self.api_key = Some(api_key.into());
        self
    }

    /// Set the model
    pub fn model(mut self, model: impl Into<String>) -> Self {
        self.model = Some(model.into());
        self
    }

    /// Set request timeout
    #[must_use]
    pub fn timeout(mut self, timeout: Duration) -> Self {
        self.common = self.common.with_timeout(timeout);
        self
    }

    /// Set maximum retries
    #[must_use]
    pub fn max_retries(mut self, max_retries: usize) -> Self {
        self.common = self.common.with_max_retries(max_retries);
        self
    }

    /// Build the configuration
    #[must_use]
    pub fn build(self) -> PerplexitySearchToolConfig {
        PerplexitySearchToolConfig {
            api_key: self.api_key,
            model: self.model,
            common: self.common,
        }
    }
}

#[derive(Debug)]
pub struct PerplexitySearchTool {
    provider: Arc<PerplexityProvider>,
    /// Construction-time credentials (highest priority)
    credentials: Option<ToolCredential>,
}

impl PerplexitySearchTool {
    /// Create a new `PerplexitySearchTool` with a provider
    #[must_use]
    pub const fn new(provider: Arc<PerplexityProvider>) -> Self {
        Self {
            provider,
            credentials: None,
        }
    }

    /// Create a new `PerplexitySearchTool` with a provider and credentials
    #[must_use]
    pub const fn with_credentials(
        provider: Arc<PerplexityProvider>,
        credentials: ToolCredential,
    ) -> Self {
        Self {
            provider,
            credentials: Some(credentials),
        }
    }

    /// Create a `PerplexitySearchTool` with configuration
    pub fn with_config(config: PerplexitySearchToolConfig) -> anyhow::Result<Self> {
        let mut builder = PerplexityConfig::builder();

        if let Some(api_key) = config.api_key {
            builder = builder.api_key(api_key);
        }

        if let Some(model) = config.model {
            builder = builder.model(model);
        }

        if let Some(timeout) = config.common.timeout {
            builder = builder.timeout(timeout);
        }

        if let Some(max_retries) = config.common.max_retries {
            builder = builder.max_retries(max_retries);
        }

        // Note: retry_strategy not supported in new builder - using max_retries instead

        let provider = builder.build()?;
        let mut tool = Self::new(Arc::new(provider));

        // Set construction-time credentials if available
        if let Some(cred) = config.common.credentials {
            tool.credentials = Some(cred);
        }

        Ok(tool)
    }

    /// Create a `PerplexitySearchTool` from environment variables if available
    pub async fn from_env() -> Option<Self> {
        // Check if API key is available
        if caliche_contracts::credential_env::var("PERPLEXITY_API_KEY").is_err() {
            tracing::debug!(
                "PerplexitySearchTool::from_env: PERPLEXITY_API_KEY not found in environment"
            );
            return None;
        }

        // Try to create provider
        match PerplexityProvider::from_env(None).await {
            Ok(provider) => {
                tracing::debug!("PerplexitySearchTool::from_env: Successfully created tool");
                Some(Self::new(Arc::new(provider)))
            }
            Err(e) => {
                tracing::debug!(
                    "PerplexitySearchTool::from_env: Failed to create provider: {}",
                    e
                );
                None
            }
        }
    }
}

const NAME: &str = "perplexity_search";
const DESCRIPTION: &str = "Search the web and get real-time information using Perplexity AI. Use this tool when you need current information, facts, news, or to search the internet. The tool can answer questions about current events, look up information, and provide up-to-date data.";

impl ToolMeta for PerplexitySearchTool {
    fn description(&self) -> &str {
        DESCRIPTION
    }

    fn description_as_cow(&self) -> beef::lean::Cow<'static, str> {
        beef::lean::Cow::borrowed(DESCRIPTION)
    }

    fn name_as_cow(&self) -> beef::lean::Cow<'static, str> {
        beef::lean::Cow::borrowed(NAME)
    }

    fn name(&self) -> &str {
        NAME
    }

    fn parameters(&self) -> serde_json::Value {
        // Merge credential parameters
        let mut props = json!({
            "query": {
                "type": "string",
                "description": "The search query or question to ask Perplexity. This should be a clear, specific question or search term."
            }
        });

        let credential_params = crate::agent::tools::credential_parameters();
        if let (Some(props_obj), Some(cred_obj)) =
            (props.as_object_mut(), credential_params.as_object())
        {
            for (key, value) in cred_obj {
                props_obj.insert(key.clone(), value.clone());
            }
        }

        json!({
            "type": "object",
            "properties": props,
            "required": ["query"]
        })
    }
}

#[async_trait]
impl Tool for PerplexitySearchTool {
    async fn execute(&self, arguments: serde_json::Value) -> ToolResult {
        self.execute_with_context(arguments, None).await
    }

    async fn execute_with_context(
        &self,
        arguments: serde_json::Value,
        context: Option<&UserContext>,
    ) -> ToolResult {
        tracing::debug!("Perplexity search called with arguments: {:?}", arguments);

        let query = match params::extract_string(&arguments, "query", None) {
            Ok(q) => q,
            Err(e) => {
                tracing::warn!(
                    "Perplexity search tool called without query parameter. Arguments received: {:?}",
                    arguments
                );
                return ToolResult::error(e);
            }
        };

        // Resolve credentials - check if runtime credentials differ from construction-time
        let resolved_credential =
            CredentialResolver::resolve(self.credentials.as_ref(), &arguments, context);

        // If runtime credentials are provided and differ from construction-time, create a new provider
        let provider_to_use = if let Some(cred) = &resolved_credential {
            // Check if this is different from construction-time credential
            let needs_new_provider = self.credentials.as_ref().map_or(true, |c| c != cred);

            if needs_new_provider {
                // Runtime credential differs from construction-time, extract API key and create new provider
                let api_key = match cred {
                    ToolCredential::ApiKey { value, .. } => Some(value.clone()),
                    ToolCredential::BearerToken(token) => Some(token.clone()),
                    _ => {
                        // For other credential types, use default provider
                        tracing::warn!(
                            "Perplexity only supports API key/bearer token credentials, ignoring credential type"
                        );
                        None
                    }
                };

                if let Some(key) = api_key {
                    tracing::debug!("Using runtime credential for Perplexity search");
                    let builder = PerplexityConfig::builder().api_key(key);
                    match builder.build() {
                        Ok(provider) => Arc::new(provider),
                        Err(e) => {
                            tracing::warn!(
                                "Failed to create provider with runtime credential, using default: {}",
                                e
                            );
                            Arc::clone(&self.provider)
                        }
                    }
                } else {
                    Arc::clone(&self.provider)
                }
            } else {
                // Same as construction-time, use existing provider
                Arc::clone(&self.provider)
            }
        } else {
            // No runtime credentials, use existing provider
            Arc::clone(&self.provider)
        };

        tracing::info!("Executing Perplexity search with query: {}", query);

        // Create a completion request for Perplexity
        let request = CompletionRequest {
            messages: std::sync::Arc::new(vec![Message {
                role: MessageRole::User,
                content: crate::messages::MessageContent::text(query.clone()),
                metadata: None,
            }]),
            max_tokens: Some(2000),
            temperature: Some(0.7),
            reasoning_effort: None,
            tools: None,
            tool_choice: None,
            response_format: None,
        };

        match provider_to_use.as_ref().complete(request).await {
            Ok(response) => {
                let result_text = response.content.as_text().to_string();

                // Track token usage if available
                if let Some(ref usage) = response.usage {
                    tracing::info!(
                        "Perplexity search token usage - prompt: {:?}, completion: {:?}, total: {:?}",
                        usage.prompt_tokens,
                        usage.completion_tokens,
                        usage.total_tokens
                    );

                    tracing::info!(
                        "Perplexity search completed successfully, response length: {}",
                        result_text.len()
                    );
                    ToolResult::success_with_usage(result_text, *usage)
                } else {
                    tracing::info!(
                        "Perplexity search completed successfully, response length: {}",
                        result_text.len()
                    );
                    ToolResult::success(result_text)
                }
            }
            Err(e) => {
                tracing::error!("Perplexity search failed: {}", e);
                ToolResult::error(format!("Failed to search with Perplexity: {e}"))
            }
        }
    }
}
