//! Non-agentic LLM client for simple, single-shot completions.
//!
//! This module provides a lightweight client for direct LLM interactions
//! without the overhead of memory or agent orchestration.
//! Tools are optionally supported for function calling.

use crate::llm::Provider;
use crate::llm::middleware::{MiddlewareProvider, ProviderMiddleware};
use crate::llm::provider_factory::{ProviderFactoryFn, default_provider_factory};
use crate::messages::{
    CompletionRequest, CompletionResponse, Message, StreamEvent, ToolChoice, ToolSchema,
};
use crate::traits::credentials::{
    CredentialProvider, Credentials, EnvironmentCredentialProvider, StaticCredentialProvider,
};
use crate::{agent::AwsCredentials, llm::ProviderType};
use futures_util::Stream;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

#[cfg(feature = "failover")]
use crate::llm::{
    AuthProfile, AuthProfileManager, FailoverConfig, FailoverEntry, FailoverEvent,
    FailoverProvider, ProviderModelPair, RotationStrategy,
};
#[cfg(feature = "failover")]
fn default_auth_profiles(provider: ProviderType, api_key: Option<&str>) -> Vec<AuthProfile> {
    if let Some(key) = api_key
        && !key.is_empty()
    {
        return vec![AuthProfile::ApiKey {
            name: "default".into(),
            key: key.to_string(),
        }];
    }
    let var_name = match provider {
        ProviderType::OpenAI => "OPENAI_API_KEY",
        ProviderType::Anthropic => "ANTHROPIC_API_KEY",
        ProviderType::OpenRouter => "OPENROUTER_API_KEY",
        ProviderType::Google => "GOOGLE_API_KEY",
        ProviderType::Cohere => "COHERE_API_KEY",
        ProviderType::Perplexity => "PERPLEXITY_API_KEY",
        ProviderType::LMStudio => "LMSTUDIO_BASE_URL",
        ProviderType::Ollama => "OLLAMA_BASE_URL",
        ProviderType::CloudflareAIGateway => "CLOUDFLARE_AIG_API_KEY",
        _ => "CALICHE_API_KEY",
    };
    vec![AuthProfile::EnvVar {
        name: "default".into(),
        var_name: var_name.into(),
    }]
}

/// A lightweight client for non-agentic LLM completions.
///
/// This client provides direct access to LLM providers without
/// memory or orchestration overhead. Tools are optionally supported
/// for function calling.
///
/// # Examples
///
/// Basic usage without tools:
///
/// ```rust,no_run
/// use caliche::{LlmClient, ProviderType};
///
/// # async fn example() -> anyhow::Result<()> {
/// let client = LlmClient::builder()
///     .provider(ProviderType::OpenAI)
///     .model("gpt-4o".to_string())
///     .api_key("sk-...".to_string())
///     .build()
///     .await?;
///
/// // Non-streaming completion
/// let response = client.complete_text("What is 2+2?").await?;
/// println!("{}", response);
///
/// // Streaming completion
/// use caliche::messages::StreamEvent;
/// use futures_util::StreamExt;
///
/// let mut stream = client.stream_text("Tell me a story");
/// while let Some(event) = stream.next().await {
///     if let Ok(StreamEvent::TextChunk { content }) = event {
///         print!("{}", content);
///     }
/// }
/// # Ok(())
/// # }
/// ```
///
/// Usage with tools:
///
/// ```rust,no_run
/// use caliche::{LlmClient, ProviderType, ToolChoice};
/// use caliche::messages::{ToolSchema, ParametersSchema};
///
/// # async fn example() -> anyhow::Result<()> {
/// let tools = vec![ToolSchema {
///     name: beef::lean::Cow::borrowed("get_weather"),
///     description: beef::lean::Cow::borrowed("Get the current weather"),
///     parameters: ParametersSchema {
///         schema_type: beef::lean::Cow::borrowed("object"),
///         properties: serde_json::json!({
///             "location": {
///                 "type": "string",
///                 "description": "The city and state"
///             }
///         }),
///         required: vec![beef::lean::Cow::owned("location".to_string())],
///     },
/// }];
///
/// let client = LlmClient::builder()
///     .provider(ProviderType::OpenAI)
///     .model("gpt-4o".to_string())
///     .api_key("sk-...".to_string())
///     .tools(tools)
///     .tool_choice(ToolChoice::Auto)
///     .build()
///     .await?;
///
/// let messages = vec![caliche::Message::user("What's the weather in San Francisco?")];
/// let response = client.complete(messages).await?;
/// # Ok(())
/// # }
/// ```
pub struct LlmClient {
    provider: Arc<dyn Provider>,
    model: String,
    default_max_tokens: Option<u32>,
    default_temperature: Option<f32>,
    default_tools: Option<Arc<Vec<ToolSchema>>>,
    default_tool_choice: Option<ToolChoice>,
}

impl LlmClient {
    /// Create a new builder for constructing an `LlmClient`.
    #[must_use]
    pub fn builder() -> LlmClientBuilder {
        LlmClientBuilder::new()
    }

    /// Complete a conversation with messages (non-streaming).
    ///
    /// Returns the full completion response including content, usage, and finish reason.
    /// # Errors
    ///
    /// Returns an error if the language-model operation cannot be completed.
    pub async fn complete(&self, messages: Vec<Message>) -> anyhow::Result<CompletionResponse> {
        let request = CompletionRequest {
            messages: Arc::new(messages),
            max_tokens: self.default_max_tokens,
            temperature: self.default_temperature,
            reasoning_effort: None,
            tools: self.default_tools.clone(),
            tool_choice: self.default_tool_choice.clone(),
            response_format: None,
        };

        self.provider.complete(request).await
    }

    /// Stream a conversation with messages.
    ///
    /// Returns a stream of `StreamEvent` items.
    #[must_use]
    pub fn stream(
        &self,
        messages: Vec<Message>,
    ) -> std::pin::Pin<Box<dyn Stream<Item = anyhow::Result<StreamEvent>> + Send + '_>> {
        let request = CompletionRequest {
            messages: Arc::new(messages),
            max_tokens: self.default_max_tokens,
            temperature: self.default_temperature,
            reasoning_effort: None,
            tools: self.default_tools.clone(),
            tool_choice: self.default_tool_choice.clone(),
            response_format: None,
        };

        self.provider.stream(request)
    }

    /// Complete a single text prompt (convenience method).
    ///
    /// Creates a user message from the prompt and returns just the text content.
    /// # Errors
    ///
    /// Returns an error if the language-model operation cannot be completed.
    pub async fn complete_text(&self, prompt: &str) -> anyhow::Result<String> {
        let messages = vec![Message::user(prompt)];
        let response = self.complete(messages).await?;
        Ok(response.content.extract_text())
    }

    /// Stream a single text prompt (convenience method).
    ///
    /// Creates a user message from the prompt and returns a stream.
    #[must_use]
    pub fn stream_text(
        &self,
        prompt: &str,
    ) -> std::pin::Pin<Box<dyn Stream<Item = anyhow::Result<StreamEvent>> + Send + '_>> {
        let messages = vec![Message::user(prompt)];
        self.stream(messages)
    }

    /// Get the provider name.
    #[must_use]
    pub fn provider_name(&self) -> &str {
        self.provider.name()
    }

    /// Access the underlying provider (for advanced integrations).
    #[must_use]
    pub fn provider(&self) -> &Arc<dyn Provider> {
        &self.provider
    }

    /// Get the model name.
    #[must_use]
    pub fn model(&self) -> &str {
        &self.model
    }
}

/// Builder for constructing `LlmClient` instances.
///
/// # Examples
///
/// ```rust,no_run
/// use caliche::{LlmClientBuilder, ProviderType};
///
/// # async fn example() -> anyhow::Result<()> {
/// let client = LlmClientBuilder::new()
///     .provider(ProviderType::OpenAI)
///     .model("gpt-4o".to_string())
///     .api_key("sk-...".to_string())
///     .temperature(0.7)
///     .max_tokens(1000)
///     .build()
///     .await?;
/// # Ok(())
/// # }
/// ```
#[derive(Clone)]
pub struct LlmClientBuilder {
    provider: Option<ProviderType>,
    model: Option<String>,
    api_key: Option<String>,
    scoped_api_keys: HashMap<ProviderType, String>,
    aws_credentials: Option<AwsCredentials>,
    base_url: Option<String>,
    temperature: Option<f32>,
    max_tokens: Option<u32>,
    timeout: Option<Duration>,
    max_retries: Option<usize>,
    tools: Option<Arc<Vec<ToolSchema>>>,
    tool_choice: Option<ToolChoice>,
    middleware: Vec<Arc<dyn ProviderMiddleware>>,
    #[cfg(feature = "failover")]
    fallbacks: Vec<ProviderModelPair>,
    #[cfg(feature = "failover")]
    auth_profiles: HashMap<ProviderType, Vec<AuthProfile>>,
    #[cfg(feature = "failover")]
    cooldown_base_seconds: Option<u64>,
    #[cfg(feature = "failover")]
    failover_tx: Option<tokio::sync::broadcast::Sender<FailoverEvent>>,
    #[cfg(feature = "failover")]
    failover_config: Option<FailoverConfig>,
    provider_factory: Option<ProviderFactoryFn>,
}

impl LlmClientBuilder {
    /// Create a new builder.
    #[must_use]
    pub fn new() -> Self {
        Self {
            provider: None,
            model: None,
            api_key: None,
            scoped_api_keys: HashMap::new(),
            aws_credentials: None,
            base_url: None,
            temperature: None,
            max_tokens: None,
            timeout: None,
            max_retries: None,
            tools: None,
            tool_choice: None,
            middleware: Vec::new(),
            #[cfg(feature = "failover")]
            fallbacks: Vec::new(),
            #[cfg(feature = "failover")]
            auth_profiles: HashMap::new(),
            #[cfg(feature = "failover")]
            cooldown_base_seconds: None,
            #[cfg(feature = "failover")]
            failover_tx: None,
            #[cfg(feature = "failover")]
            failover_config: None,
            provider_factory: None,
        }
    }

    /// Set the provider type.
    #[must_use]
    pub const fn provider(mut self, provider: ProviderType) -> Self {
        self.provider = Some(provider);
        self
    }

    /// Set the model name.
    #[must_use]
    pub fn model(mut self, model: impl Into<String>) -> Self {
        self.model = Some(model.into());
        self
    }

    /// Set the API key for authentication.
    #[must_use]
    pub fn api_key(mut self, key: impl Into<String>) -> Self {
        self.api_key = Some(key.into());
        self
    }

    /// Set an API key scoped to a specific provider.
    ///
    /// This is useful when a single process uses multiple providers and each provider
    /// requires a distinct credential.
    #[must_use]
    pub fn api_key_for(mut self, provider: ProviderType, key: impl Into<String>) -> Self {
        self.scoped_api_keys.insert(provider, key.into());
        self
    }

    /// Set AWS credentials for Bedrock provider.
    #[must_use]
    pub fn aws_credentials(mut self, credentials: AwsCredentials) -> Self {
        self.aws_credentials = Some(credentials);
        self
    }

    /// Set base URL (for `LMStudio` or custom endpoints).
    #[must_use]
    pub fn base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = Some(url.into());
        self
    }

    /// Set default temperature for completions.
    #[must_use]
    pub const fn temperature(mut self, temp: f32) -> Self {
        self.temperature = Some(temp);
        self
    }

    /// Set default max tokens for completions.
    #[must_use]
    pub const fn max_tokens(mut self, tokens: u32) -> Self {
        self.max_tokens = Some(tokens);
        self
    }

    /// Set request timeout.
    #[must_use]
    pub const fn timeout(mut self, timeout: Duration) -> Self {
        self.timeout = Some(timeout);
        self
    }

    /// Set maximum retry attempts.
    #[must_use]
    pub const fn max_retries(mut self, retries: usize) -> Self {
        self.max_retries = Some(retries);
        self
    }

    /// Set default tools for completions.
    ///
    /// Tools are optional and can be used for function calling.
    /// If not set, no tools will be sent to the LLM.
    #[must_use]
    pub fn tools(mut self, tools: Vec<ToolSchema>) -> Self {
        self.tools = Some(Arc::new(tools));
        self
    }

    /// Set default tool choice for completions.
    ///
    /// Controls how the LLM uses tools:
    /// - `Auto`: LLM decides whether to use tools (default)
    /// - `None`: Disable tool usage
    /// - `Required`: Force tool usage
    /// - `Specific(String)`: Force use of a specific tool
    #[must_use]
    pub fn tool_choice(mut self, choice: ToolChoice) -> Self {
        self.tool_choice = Some(choice);
        self
    }

    /// Add a provider middleware layer.
    #[must_use]
    pub fn middleware(mut self, middleware: Arc<dyn ProviderMiddleware>) -> Self {
        self.middleware.push(middleware);
        self
    }

    /// Add multiple provider middleware layers.
    #[must_use]
    pub fn middleware_layers(mut self, middleware: Vec<Arc<dyn ProviderMiddleware>>) -> Self {
        self.middleware.extend(middleware);
        self
    }

    /// Override provider factory implementation.
    #[must_use]
    pub fn provider_factory(mut self, provider_factory: ProviderFactoryFn) -> Self {
        self.provider_factory = Some(provider_factory);
        self
    }

    /// Add a fallback provider+model to the failover chain (requires `failover` feature).
    #[cfg(feature = "failover")]
    #[must_use]
    pub fn fallback(mut self, provider: ProviderType, model: impl Into<String>) -> Self {
        self.fallbacks.push(ProviderModelPair::new(provider, model));
        self
    }

    /// Configure auth profiles for a specific provider (requires `failover` feature).
    #[cfg(feature = "failover")]
    #[must_use]
    pub fn auth_profiles(mut self, provider: ProviderType, profiles: Vec<AuthProfile>) -> Self {
        self.auth_profiles.insert(provider, profiles);
        self
    }

    /// Set cooldown base duration in seconds for rate-limited profiles (requires `failover` feature).
    #[cfg(feature = "failover")]
    #[must_use]
    pub const fn cooldown_base(mut self, seconds: u64) -> Self {
        self.cooldown_base_seconds = Some(seconds);
        self
    }

    /// Subscribe to failover events (requires `failover` feature).
    #[cfg(feature = "failover")]
    #[must_use]
    pub fn on_failover(mut self, tx: tokio::sync::broadcast::Sender<FailoverEvent>) -> Self {
        self.failover_tx = Some(tx);
        self
    }

    /// Set failover from a single config (primary + fallbacks + auth + cooldown). Applied at `build()`.
    #[cfg(feature = "failover")]
    #[must_use]
    pub fn failover_config(mut self, config: FailoverConfig) -> Self {
        self.failover_config = Some(config);
        self
    }

    #[cfg(feature = "failover")]
    fn failover_entries(
        &self,
        provider_type: ProviderType,
        model: &str,
    ) -> Option<Vec<FailoverEntry>> {
        let has_profiles = self
            .auth_profiles
            .values()
            .any(|profiles| !profiles.is_empty());
        if self.fallbacks.is_empty() && !has_profiles {
            return None;
        }
        let mut entries = Vec::with_capacity(self.fallbacks.len() + 1);
        let primary_key = self
            .scoped_api_keys
            .get(&provider_type)
            .cloned()
            .or_else(|| self.api_key.clone());
        let primary_profiles = self
            .auth_profiles
            .get(&provider_type)
            .cloned()
            .unwrap_or_else(|| default_auth_profiles(provider_type, primary_key.as_deref()));
        entries.push(FailoverEntry {
            pair: ProviderModelPair::new(provider_type, model.to_string()),
            auth_manager: AuthProfileManager::new(primary_profiles, RotationStrategy::Priority)
                .cooldown_base(self.cooldown_base_seconds.unwrap_or(60)),
        });
        entries.extend(self.fallbacks.iter().map(|pair| {
            let profiles = self
                .auth_profiles
                .get(&pair.provider)
                .cloned()
                .unwrap_or_else(|| {
                    let key = self
                        .scoped_api_keys
                        .get(&pair.provider)
                        .cloned()
                        .or_else(|| self.api_key.clone());
                    default_auth_profiles(pair.provider, key.as_deref())
                });
            FailoverEntry {
                pair: pair.clone(),
                auth_manager: AuthProfileManager::new(profiles, RotationStrategy::Priority)
                    .cooldown_base(self.cooldown_base_seconds.unwrap_or(60)),
            }
        }));
        Some(entries)
    }

    fn credential_provider(&mut self, provider_type: ProviderType) -> Arc<dyn CredentialProvider> {
        let api_key = self
            .scoped_api_keys
            .get(&provider_type)
            .cloned()
            .or_else(|| self.api_key.clone());
        let default: Arc<dyn CredentialProvider> = if let Some(key) = api_key {
            Arc::new(StaticCredentialProvider::new(Credentials::ApiKey(key)))
        } else {
            Arc::new(EnvironmentCredentialProvider::new())
        };
        if provider_type != ProviderType::Bedrock {
            return default;
        }
        #[cfg(feature = "bedrock")]
        if let Some(aws) = self.aws_credentials.take()
            && let (Some(access_key), Some(secret_key)) = (aws.access_key_id, aws.secret_access_key)
            && !access_key.is_empty()
            && !secret_key.is_empty()
        {
            return Arc::new(StaticCredentialProvider::new(Credentials::AwsAuth {
                access_key,
                secret_key,
                session_token: aws.session_token,
                region: aws.region.or_else(|| Some("us-east-1".to_string())),
            }));
        }
        default
    }

    /// Build the `LlmClient`.
    #[cfg_attr(not(feature = "failover"), allow(unused_mut))]
    /// # Errors
    ///
    /// Returns an error if the language-model operation cannot be completed.
    pub async fn build(mut self) -> anyhow::Result<LlmClient> {
        #[cfg(feature = "failover")]
        if let Some(c) = self.failover_config.take() {
            self.provider = Some(c.primary.provider);
            self.model = Some(c.primary.model);
            self.fallbacks = c.fallbacks;
            self.auth_profiles = c.auth_profiles;
            self.cooldown_base_seconds = c.cooldown_base_seconds;
            self.failover_tx = c.failover_tx;
        }

        let provider_type = self
            .provider
            .take()
            .ok_or_else(|| anyhow::anyhow!("provider is required. Use .provider()"))?;

        let model = self
            .model
            .take()
            .ok_or_else(|| anyhow::anyhow!("model is required. Use .model()"))?;

        #[cfg(feature = "failover")]
        {
            if let Some(entries) = self.failover_entries(provider_type, &model) {
                let failover = FailoverProvider::new(
                    entries,
                    self.base_url.clone(),
                    self.timeout,
                    self.max_retries,
                    self.failover_tx,
                    self.provider_factory.clone(),
                );
                let provider: Arc<dyn Provider> = if self.middleware.is_empty() {
                    Arc::new(failover)
                } else {
                    Arc::new(MiddlewareProvider::new(
                        Arc::new(failover),
                        self.middleware.clone(),
                    ))
                };
                return Ok(LlmClient {
                    provider,
                    model,
                    default_max_tokens: self.max_tokens,
                    default_temperature: self.temperature,
                    default_tools: self.tools,
                    default_tool_choice: self.tool_choice,
                });
            }
        }

        let credential_provider = self.credential_provider(provider_type);

        let provider_factory = self
            .provider_factory
            .clone()
            .unwrap_or_else(default_provider_factory);
        let provider = provider_factory(
            provider_type,
            model.clone(),
            credential_provider,
            self.base_url.clone(),
            self.timeout,
            self.max_retries,
        )
        .await?;

        let provider: Arc<dyn Provider> = if self.middleware.is_empty() {
            provider
        } else {
            Arc::new(MiddlewareProvider::new(provider, self.middleware))
        };

        Ok(LlmClient {
            provider,
            model,
            default_max_tokens: self.max_tokens,
            default_temperature: self.temperature,
            default_tools: self.tools,
            default_tool_choice: self.tool_choice,
        })
    }
}

impl Default for LlmClientBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_builder_creation() {
        let builder = LlmClientBuilder::new()
            .provider(ProviderType::OpenAI)
            .model("gpt-4o".to_string());
        assert!(builder.provider.is_some());
        assert!(builder.model.is_some());
    }
}
