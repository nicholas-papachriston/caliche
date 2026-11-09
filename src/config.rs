//! Configuration support for agent settings.
//!
//! The `Config` struct and related types are always available.
//! File loading functionality requires the `cli` feature.
//!
//! # Configuration Hierarchy
//!
//! The configuration is organized into a clear hierarchy:
//!
//! - **`Config`**: Top-level configuration container
//!   - **`ProviderConfig`**: LLM provider settings (type, model, temperature, etc.)
//!   - **`ReasoningConfig`**: Reasoning pattern configuration
//!   - **`ChainsConfig`**: Chain operation settings
//!   - **`RagConfig`**: RAG (Retrieval Augmented Generation) settings
//!   - **`OrchestrationConfig`**: Multi-agent orchestration settings
//!   - **`ToolsConfig`**: Tool enablement and configuration
//!   - **`ApiConfig`**: API server settings
//!
//! # Examples
//!
//! ## Load from File
//!
//! ```rust,no_run
//! # #[cfg(feature = "templates")]
//! use caliche::Config;
//!
//! # #[cfg(feature = "templates")]
//! # async fn example() -> anyhow::Result<()> {
//! let config = Config::from_file("config.toml")?;
//! # Ok(())
//! # }
//! ```
//!
//! ## Create Programmatically
//!
//! ```rust,no_run
//! use caliche::Config;
//!
//! let mut config = Config::default();
//! config.provider.provider_type = Some("openai".to_string());
//! config.provider.model = Some("gpt-4o".to_string());
//! config.tools.enabled = Some(vec!["calculator".to_string()]);
//! ```

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[cfg(feature = "templates")]
use anyhow::{Context, Result};
#[cfg(feature = "templates")]
use std::path::Path;

/// Top-level application configuration.
///
/// Contains all configuration sections for the agent framework.
/// Each section can be configured independently and has sensible defaults.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Config {
    /// LLM provider configuration
    #[serde(default)]
    pub provider: ProviderConfig,
    /// Reasoning pattern configuration
    #[serde(default)]
    pub reasoning: ReasoningConfig,
    /// Chain operation configuration
    #[serde(default)]
    pub chains: ChainsConfig,
    /// RAG (Retrieval Augmented Generation) configuration
    #[serde(default)]
    pub rag: RagConfig,
    /// Multi-agent orchestration configuration
    #[serde(default)]
    pub orchestration: OrchestrationConfig,
    /// Tool enablement and configuration
    #[serde(default)]
    pub tools: ToolsConfig,
    /// API server configuration
    #[serde(default)]
    pub api: ApiConfig,
}

/// LLM provider configuration.
///
/// Configures which LLM provider to use and model parameters.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ProviderConfig {
    #[serde(rename = "type")]
    pub provider_type: Option<String>,
    pub model: Option<String>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub max_tokens: Option<u32>,
    pub timeout: Option<u64>,
}

/// Reasoning pattern configuration.
///
/// Configures advanced reasoning patterns like ReAct, Plan-and-Execute, etc.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReasoningConfig {
    pub default_pattern: Option<String>,
    pub max_iterations: Option<usize>,
    pub tree_depth: Option<usize>,
    pub tree_branches: Option<usize>,
}

impl Default for ReasoningConfig {
    fn default() -> Self {
        Self {
            default_pattern: None,
            max_iterations: Some(5),
            tree_depth: None,
            tree_branches: None,
        }
    }
}

/// Chain operation configuration.
///
/// Configures chain operations like LLM chains, sequential chains, etc.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChainsConfig {
    pub default_chain_type: Option<String>,
    pub chunk_size: Option<usize>,
    pub chunk_overlap: Option<usize>,
}

impl Default for ChainsConfig {
    fn default() -> Self {
        Self {
            default_chain_type: None,
            chunk_size: Some(500),
            chunk_overlap: Some(50),
        }
    }
}

/// RAG (Retrieval Augmented Generation) configuration.
///
/// Configures document indexing, retrieval, and hybrid search settings.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RagConfig {
    pub default_collection: Option<String>,
    pub chunk_size: Option<usize>,
    pub chunk_overlap: Option<usize>,
    pub top_k: Option<usize>,
    pub vector_weight: Option<f32>,
    pub keyword_weight: Option<f32>,
}

impl Default for RagConfig {
    fn default() -> Self {
        Self {
            default_collection: None,
            chunk_size: Some(500),
            chunk_overlap: Some(50),
            top_k: Some(5),
            vector_weight: Some(0.7),
            keyword_weight: Some(0.3),
        }
    }
}

/// Supervision restart strategy (OTP-style).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum SupervisionStrategy {
    /// Restart only the failed agent.
    #[default]
    OneForOne,
    /// Restart all agents in the group when one fails.
    OneForAll,
    /// Restart the failed agent and all registered after it (ordered pipelines).
    RestForOne,
}

/// Backoff type for restart delays.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum BackoffType {
    /// No delay between restarts.
    #[default]
    Immediate,
    /// Linear backoff: base_ms * restart_count.
    Linear,
    /// Exponential backoff: base_ms * 2^restart_count.
    Exponential,
}

/// Supervision configuration for per-agent task isolation.
///
/// Enables OTP-style supervision: agents run in isolated tokio tasks with
/// automatic restart on panic or error, configurable strategies and backoff.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SupervisionConfig {
    pub strategy: SupervisionStrategy,
    pub max_restarts: u32,
    pub restart_window_secs: u64,
    pub backoff: BackoffType,
    pub backoff_base_ms: u64,
}

impl Default for SupervisionConfig {
    fn default() -> Self {
        Self {
            strategy: SupervisionStrategy::OneForOne,
            max_restarts: 3,
            restart_window_secs: 60,
            backoff: BackoffType::Exponential,
            backoff_base_ms: 500,
        }
    }
}

/// Default hop limit for [`crate::orchestration::AgentFlow`] handoff chains.
pub const DEFAULT_MAX_FLOW_HOPS: usize = 50;

/// Default cap on concurrent branches inside one orchestration pattern.
pub const DEFAULT_PATTERN_CONCURRENCY: usize = 8;

/// Multi-agent orchestration configuration.
///
/// Configures multi-agent workflows, coordinators, agent coordination, and the multi-agent
/// safety controls: task invocation ACLs, privilege ceilings, hop limits, and budgets.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrchestrationConfig {
    pub enable_multi_agent: Option<bool>,
    pub coordinator_enabled: Option<bool>,
    pub default_pattern: Option<String>,
    pub supervision: Option<SupervisionConfig>,
    /// Default task invocation ACL applied to agents without their own policy.
    ///
    /// Keys are glob patterns over target agent names, values are `allow` or `deny`, for
    /// example `{"*": "deny", "reviewer": "allow"}`. The most specific pattern wins. When
    /// unset, every registered target is allowed.
    #[serde(default)]
    pub task_acl: Option<BTreeMap<String, String>>,
    /// Refuse `transfer_to` targets that are not registered with the coordinator or flow.
    /// Defaults to `true`.
    #[serde(default)]
    pub require_registered_targets: Option<bool>,
    /// Wrap subagent output in a labeled untrusted block before a parent agent reads it.
    /// Defaults to `true`.
    #[serde(default)]
    pub quarantine_subagent_output: Option<bool>,
    /// Maximum handoff hops in one `AgentFlow` run. Defaults to
    /// [`DEFAULT_MAX_FLOW_HOPS`].
    #[serde(default)]
    pub max_flow_hops: Option<usize>,
    /// Timeout in seconds for one coordinator dispatch. When unset, dispatch is unbounded.
    #[serde(default)]
    pub dispatch_timeout_secs: Option<u64>,
    /// Maximum branches that one orchestration pattern may run at once. Defaults to
    /// [`DEFAULT_PATTERN_CONCURRENCY`].
    #[serde(default)]
    pub pattern_concurrency: Option<usize>,
    /// Maximum agent nodes one pattern run may execute, as a cost ceiling. When unset, no
    /// node budget applies.
    #[serde(default)]
    pub pattern_max_nodes: Option<usize>,
    /// Maximum total tokens one pattern run may spend. When unset, no token budget applies.
    #[serde(default)]
    pub pattern_max_tokens: Option<u64>,
    /// Retry attempts for a failed workflow agent node when the node error policy is
    /// `Retry`. Defaults to `2`.
    #[serde(default)]
    pub node_retry_attempts: Option<u32>,
    /// Base backoff in milliseconds between workflow node retries. Defaults to `250`.
    #[serde(default)]
    pub node_retry_backoff_ms: Option<u64>,
}

impl Default for OrchestrationConfig {
    fn default() -> Self {
        Self {
            enable_multi_agent: Some(true),
            coordinator_enabled: Some(true),
            default_pattern: None,
            supervision: None,
            task_acl: None,
            require_registered_targets: Some(true),
            quarantine_subagent_output: Some(true),
            max_flow_hops: Some(DEFAULT_MAX_FLOW_HOPS),
            dispatch_timeout_secs: None,
            pattern_concurrency: Some(DEFAULT_PATTERN_CONCURRENCY),
            pattern_max_nodes: None,
            pattern_max_tokens: None,
            node_retry_attempts: Some(2),
            node_retry_backoff_ms: Some(250),
        }
    }
}

impl OrchestrationConfig {
    /// Effective flow hop limit.
    pub fn max_flow_hops_or_default(&self) -> usize {
        self.max_flow_hops
            .filter(|hops| *hops > 0)
            .unwrap_or(DEFAULT_MAX_FLOW_HOPS)
    }

    /// Effective pattern concurrency limit.
    pub fn pattern_concurrency_or_default(&self) -> usize {
        self.pattern_concurrency
            .filter(|limit| *limit > 0)
            .unwrap_or(DEFAULT_PATTERN_CONCURRENCY)
    }

    /// Whether unknown handoff targets must be refused.
    pub fn require_registered_targets_or_default(&self) -> bool {
        self.require_registered_targets.unwrap_or(true)
    }

    /// Whether subagent output must be quarantined before re-injection.
    pub fn quarantine_subagent_output_or_default(&self) -> bool {
        self.quarantine_subagent_output.unwrap_or(true)
    }

    /// Effective dispatch timeout, if any.
    pub fn dispatch_timeout(&self) -> Option<std::time::Duration> {
        self.dispatch_timeout_secs
            .filter(|secs| *secs > 0)
            .map(std::time::Duration::from_secs)
    }

    /// Effective retry attempts for workflow agent nodes.
    pub fn node_retry_attempts_or_default(&self) -> u32 {
        self.node_retry_attempts.unwrap_or(2)
    }

    /// Effective retry backoff for workflow agent nodes.
    pub fn node_retry_backoff(&self) -> std::time::Duration {
        std::time::Duration::from_millis(self.node_retry_backoff_ms.unwrap_or(250))
    }
}

/// Tool enablement and configuration.
///
/// Controls which tools are available to agents.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ToolsConfig {
    pub enabled: Option<Vec<String>>,
}

/// API server configuration.
///
/// Configures HTTP API server settings like bind address, port, and CORS.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiConfig {
    pub bind: Option<String>,
    pub port: Option<u16>,
    pub cors_enabled: Option<bool>,
}

impl Default for ApiConfig {
    fn default() -> Self {
        Self {
            bind: Some("127.0.0.1".to_owned()),
            port: Some(3000),
            cors_enabled: Some(true),
        }
    }
}

impl Config {
    /// Load configuration from a file
    ///
    /// Supports both TOML and YAML formats based on file extension.
    /// Requires the `templates` feature for YAML support.
    #[cfg(feature = "templates")]
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path = path.as_ref();
        let content = std::fs::read_to_string(path)
            .with_context(|| format!("Failed to read config file: {}", path.display()))?;

        let config = if path.extension().and_then(|s| s.to_str()) == Some("toml") {
            // TOML support is always available (toml crate is in dependencies)
            toml::from_str(&content)
                .with_context(|| format!("Failed to parse TOML config: {}", path.display()))?
        } else if let Some(ext) = path.extension().and_then(|s| s.to_str()) {
            if ext == "yaml" || ext == "yml" {
                serde_norway::from_str(&content)
                    .with_context(|| format!("Failed to parse YAML config: {}", path.display()))?
            } else {
                anyhow::bail!(
                    "Unsupported config file format: {}. Use .toml, .yaml, or .yml",
                    ext
                )
            }
        } else {
            anyhow::bail!("Config file must have .toml, .yaml, or .yml extension")
        };

        Ok(config)
    }

    /// Merge with CLI arguments (CLI args take precedence)
    pub fn merge_with_cli(
        &mut self,
        provider_type: Option<String>,
        model: Option<String>,
        temperature: Option<f32>,
        max_tokens: Option<u32>,
    ) {
        if provider_type.is_some() {
            self.provider.provider_type = provider_type;
        }
        if model.is_some() {
            self.provider.model = model;
        }
        if temperature.is_some() {
            self.provider.temperature = temperature;
        }
        if max_tokens.is_some() {
            self.provider.max_tokens = max_tokens;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = Config::default();
        assert_eq!(config.reasoning.max_iterations, Some(5));
        assert_eq!(config.rag.top_k, Some(5));
        assert_eq!(config.orchestration.enable_multi_agent, Some(true));
        assert_eq!(config.api.bind.as_deref(), Some("127.0.0.1"));
        assert_eq!(config.api.port, Some(3000));
    }

    #[cfg(feature = "templates")]
    #[test]
    fn test_toml_parsing() {
        let toml_str = r#"
[provider]
type = "openai"
model = "gpt-4o"
temperature = 0.7

[reasoning]
default_pattern = "react"
max_iterations = 10

[rag]
default_collection = "docs"
top_k = 5
"#;
        let config: Config = toml::from_str(toml_str).expect("Failed to parse TOML");
        assert_eq!(config.provider.provider_type, Some("openai".to_string()));
        assert_eq!(config.reasoning.max_iterations, Some(10));
        assert_eq!(config.rag.top_k, Some(5));
    }

    #[cfg(feature = "templates")]
    #[test]
    fn test_yaml_parsing() {
        let yaml_str = r#"
provider:
  type: openai
  model: gpt-4o
  temperature: 0.7

reasoning:
  default_pattern: react
  max_iterations: 10
"#;
        let config: Config = serde_norway::from_str(yaml_str).expect("Failed to parse YAML");
        assert_eq!(config.provider.provider_type, Some("openai".to_string()));
        assert_eq!(config.reasoning.max_iterations, Some(10));
    }
}
