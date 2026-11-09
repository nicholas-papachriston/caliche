//! Service factory trait for creating agent services.
//!
//! Provides a unified interface for creating agent services and harnesses,
//! enabling dependency injection and testability.

use crate::errors::CliResult;
use async_trait::async_trait;
use caliche::AgentService;
use caliche::harness::{AgentHarness, HarnessConfig};
use caliche::llm::ProviderType;
use std::str::FromStr;
use std::sync::Arc;

/// Configuration for creating an agent service.
#[derive(Debug, Clone, Default)]
pub struct ServiceConfig {
    /// Provider to use (e.g., "openai", "anthropic")
    pub provider: Option<String>,

    /// Model to use (e.g., "gpt-4", "claude-3-opus")
    pub model: Option<String>,

    /// Tools to enable
    pub tools: Option<Vec<String>>,

    /// System prompt override
    pub system_prompt: Option<String>,

    /// Temperature for generation
    pub temperature: Option<f32>,

    /// Maximum tokens for response
    pub max_tokens: Option<u32>,

    /// Whether to enable streaming
    pub streaming: bool,

    /// Working directory for file operations
    pub working_dir: Option<std::path::PathBuf>,
}

impl ServiceConfig {
    /// Create a new service config with defaults.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the provider.
    pub fn provider(mut self, provider: impl Into<String>) -> Self {
        self.provider = Some(provider.into());
        self
    }

    /// Set the model.
    pub fn model(mut self, model: impl Into<String>) -> Self {
        self.model = Some(model.into());
        self
    }

    /// Set the tools to enable.
    pub fn tools(mut self, tools: Vec<String>) -> Self {
        self.tools = Some(tools);
        self
    }

    /// Set the system prompt.
    pub fn system_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.system_prompt = Some(prompt.into());
        self
    }

    /// Set the temperature.
    pub fn temperature(mut self, temp: f32) -> Self {
        self.temperature = Some(temp);
        self
    }

    /// Set max tokens.
    pub fn max_tokens(mut self, tokens: u32) -> Self {
        self.max_tokens = Some(tokens);
        self
    }

    /// Enable streaming.
    pub fn streaming(mut self, enabled: bool) -> Self {
        self.streaming = enabled;
        self
    }

    /// Set working directory.
    pub fn working_dir(mut self, dir: std::path::PathBuf) -> Self {
        self.working_dir = Some(dir);
        self
    }
}

/// Trait for creating agent services.
///
/// Implementations can provide different strategies for service creation,
/// such as pooling, caching, or mock services for testing.
#[async_trait]
pub trait ServiceFactory: Send + Sync {
    /// Create an agent service with the given configuration.
    async fn create_service(&self, config: ServiceConfig) -> CliResult<AgentService>;

    /// Create an agent harness with the given configuration.
    async fn create_harness(&self, config: HarnessConfig) -> CliResult<AgentHarness>;

    /// Get the default provider name.
    fn default_provider(&self) -> Option<&str> {
        None
    }

    /// Get the default model for a provider.
    fn default_model(&self, _provider: &str) -> Option<&str> {
        None
    }

    /// List available providers.
    fn available_providers(&self) -> Vec<&str> {
        vec![]
    }
}

/// Default service factory implementation.
///
/// Creates services directly using the caliche library.
pub struct DefaultServiceFactory {
    /// Default provider to use
    default_provider: Option<String>,

    /// Default models per provider
    default_models: std::collections::HashMap<String, String>,
}

impl Default for DefaultServiceFactory {
    fn default() -> Self {
        Self::new()
    }
}

impl DefaultServiceFactory {
    /// Create a new default service factory.
    pub fn new() -> Self {
        Self {
            default_provider: None,
            default_models: std::collections::HashMap::new(),
        }
    }

    /// Set the default provider.
    pub fn with_default_provider(mut self, provider: impl Into<String>) -> Self {
        self.default_provider = Some(provider.into());
        self
    }

    /// Set a default model for a provider.
    pub fn with_default_model(
        mut self,
        provider: impl Into<String>,
        model: impl Into<String>,
    ) -> Self {
        self.default_models.insert(provider.into(), model.into());
        self
    }
}

#[async_trait]
impl ServiceFactory for DefaultServiceFactory {
    async fn create_service(&self, config: ServiceConfig) -> CliResult<AgentService> {
        let provider_str = config.provider.or_else(|| self.default_provider.clone());
        let provider_type = provider_str
            .as_ref()
            .and_then(|p| ProviderType::from_str(p).ok());

        let model = config.model.or_else(|| {
            provider_str
                .as_ref()
                .and_then(|p| self.default_models.get(p).cloned())
        });

        let service_config = caliche::AgentServiceConfig::new()
            .with_provider(provider_type, model)
            .with_tool_names_or_defaults(config.tools);
        #[cfg(feature = "lsp")]
        let service_config = service_config.with_lsp_config(caliche::lsp::LspConfig::default());

        let mut service = AgentService::from_config(service_config)
            .await
            .map_err(|e| crate::errors::CliError::Agent(e.to_string()))?;
        crate::bootstrap::finalize_cli_service(&mut service, false);

        Ok(service)
    }

    async fn create_harness(&self, config: HarnessConfig) -> CliResult<AgentHarness> {
        let harness = AgentHarness::new(config)
            .await
            .map_err(|e| crate::errors::CliError::Agent(e.to_string()))?;

        Ok(harness)
    }

    fn default_provider(&self) -> Option<&str> {
        self.default_provider.as_deref()
    }

    fn default_model(&self, provider: &str) -> Option<&str> {
        self.default_models.get(provider).map(|s| s.as_str())
    }
}

/// Create a shared service factory.
pub fn create_service_factory() -> Arc<dyn ServiceFactory> {
    Arc::new(DefaultServiceFactory::new())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_service_config_builder() {
        let config = ServiceConfig::new()
            .provider("openai")
            .model("gpt-4")
            .temperature(0.7)
            .streaming(true);

        assert_eq!(config.provider, Some("openai".to_string()));
        assert_eq!(config.model, Some("gpt-4".to_string()));
        assert_eq!(config.temperature, Some(0.7));
        assert!(config.streaming);
    }

    #[test]
    fn test_default_factory_defaults() {
        let factory = DefaultServiceFactory::new()
            .with_default_provider("anthropic")
            .with_default_model("anthropic", "claude-3-opus");

        assert_eq!(factory.default_provider(), Some("anthropic"));
        assert_eq!(factory.default_model("anthropic"), Some("claude-3-opus"));
        assert_eq!(factory.default_model("openai"), None);
    }
}
