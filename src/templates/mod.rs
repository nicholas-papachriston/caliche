//! Agent template system for reusable agent configurations.
//!
//! Templates allow you to define agent configurations in YAML/JSON format,
//! share them with others, and quickly instantiate agents from templates.
//!
//! # Note: Difference from Prompt Templates
//!
//! This module (`templates`) provides **agent configuration templates** - complete
//! YAML/JSON definitions that specify provider, model, tools, memory, and other
//! agent settings. For simple **text prompt templates** with variable substitution,
//! see [`crate::prompts`].
//!
//! # Examples
//!
//! ```rust,no_run
//! use caliche::templates::{TemplateRegistry, AgentTemplate};
//!
//! # async fn example() -> anyhow::Result<()> {
//! let mut registry = TemplateRegistry::new();
//! registry.add_template_dir("templates");
//! registry.load_templates().await?;
//! let Some(template) = registry.get("rag_agent") else {
//!     return Err(anyhow::anyhow!("Template not found"));
//! };
//! let agent = template.instantiate().await?; // Arc<AgentTemplate> derefs to &AgentTemplate
//! # Ok(())
//! # }
//! ```

pub mod loader;
pub mod registry;
#[cfg(feature = "s3")]
pub mod storage;
pub mod validator;
pub mod variables;

pub use loader::TemplateLoader;
pub use registry::{TemplateRegistry, TemplateSearchFilters};
#[cfg(feature = "s3")]
pub use storage::{create_storage_backend, load_template_from_storage, save_template_to_storage};
pub use validator::TemplateValidator;
pub use variables::TemplateVariables;

use crate::agent::builder::AgentBuilder;
use crate::agent::memory::{BufferMemoryConfig, EntityMemoryConfig, SummaryMemoryConfig};
use crate::llm::ProviderType;
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;

/// Agent template definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentTemplate {
    /// Template metadata
    pub name: String,
    pub version: String,
    pub description: String,
    pub author: Option<String>,
    pub tags: Vec<String>,
    /// Template category (customer-service, coding, data-analysis, etc.)
    pub category: Option<String>,
    /// Template configuration
    pub config: TemplateConfig,
    /// Template variables/parameters with validation schemas
    pub variables: Option<TemplateVariables>,
    /// Example conversations and expected behaviors
    pub examples: Option<Vec<TemplateExample>>,
    /// Template dependencies (other templates or features)
    pub dependencies: Option<Vec<String>>,
    /// Performance and cost profile
    pub cost_profile: Option<CostProfile>,
    /// Template file path (if loaded from file)
    #[serde(skip)]
    pub file_path: Option<PathBuf>,
    /// Usage statistics (runtime only)
    #[serde(skip)]
    pub usage_stats: Option<UsageStats>,
}

/// Example conversation for template
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemplateExample {
    /// Example input message
    pub input: String,
    /// Expected output (optional)
    pub expected_output: Option<String>,
    /// Description of what this example demonstrates
    pub description: Option<String>,
}

/// Cost and performance profile
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CostProfile {
    /// Estimated tokens per request
    pub estimated_tokens_per_request: Option<usize>,
    /// Recommended provider tier
    pub recommended_tier: Option<String>,
    /// Estimated cost per 1000 requests (USD)
    pub estimated_cost_per_1k_requests: Option<f64>,
}

/// Usage statistics (runtime only)
#[derive(Debug, Clone, Default)]
pub struct UsageStats {
    /// Number of times template was used
    pub usage_count: usize,
    /// Average rating (1-5)
    pub average_rating: Option<f64>,
}

/// Template configuration that maps to AgentBuilder options
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemplateConfig {
    /// LLM provider (openai, anthropic, bedrock, etc.)
    pub provider: String,
    /// Model identifier
    pub model: Option<String>,
    /// System prompt
    pub system_prompt: Option<String>,
    /// List of tool names to enable
    pub tools: Option<Vec<String>>,
    /// Memory configuration
    pub memory: Option<MemoryConfig>,
    /// Vector store configuration (for RAG)
    #[cfg(feature = "tools-rag")]
    pub vector_backend: Option<VectorBackendConfig>,
    /// SQL tools configuration
    pub sql_tools: Option<SqlToolsConfig>,
}

/// Memory configuration in template
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryConfig {
    /// Memory type: buffer, summary, or entity
    #[serde(rename = "type")]
    pub memory_type: String,
    /// Maximum tokens for buffer memory
    pub max_tokens: Option<usize>,
    /// Summary memory configuration
    pub summary_config: Option<SummaryMemoryConfig>,
    /// Entity memory configuration
    pub entity_config: Option<EntityMemoryConfig>,
}

/// Vector store configuration in template
#[cfg(feature = "tools-rag")]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorBackendConfig {
    /// Backend type: memory, redis, postgres, qdrant, etc.
    pub backend: String,
    /// Backend-specific configuration
    pub config: HashMap<String, serde_json::Value>,
}

/// SQL tools configuration in template
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SqlToolsConfig {
    /// Database connection URL
    pub database_url: String,
    /// Namespace for schema embeddings
    pub namespace: String,
    /// Top K schema chunks to retrieve
    pub top_k: Option<usize>,
    /// Maximum rows to return
    pub max_rows: Option<usize>,
}

impl AgentTemplate {
    /// Instantiate an agent from this template with variable values
    #[expect(
        clippy::unused_async,
        reason = "public API consumed as async across caliche_server/cli/py bindings and integration tests; changing the signature is a cross-repo break out of scope here"
    )]
    pub async fn instantiate_with_variables(
        &self,
        variable_values: Option<HashMap<String, serde_json::Value>>,
    ) -> Result<AgentBuilder> {
        // Validate and apply variables if present
        let mut values = variable_values.unwrap_or_default();
        if let Some(ref vars) = self.variables {
            vars.apply_defaults(&mut values);
            vars.validate(&values)?;
        }

        // Interpolate variables into configuration
        let config = self.interpolate_config(&values)?;

        self.build_agent_from_config(config)
    }

    /// Instantiate an agent from this template
    pub async fn instantiate(&self) -> Result<AgentBuilder> {
        self.instantiate_with_variables(None).await
    }

    /// Interpolate variable values into configuration strings
    fn interpolate_config(
        &self,
        values: &HashMap<String, serde_json::Value>,
    ) -> Result<TemplateConfig> {
        let mut config = self.config.clone();

        // Interpolate system prompt
        if let Some(ref prompt) = config.system_prompt {
            config.system_prompt = Some(self.interpolate_string(prompt, values)?);
        }

        // Interpolate SQL tools database URL and namespace
        if let Some(ref mut sql_config) = config.sql_tools {
            sql_config.database_url = self.interpolate_string(&sql_config.database_url, values)?;
            sql_config.namespace = self.interpolate_string(&sql_config.namespace, values)?;
        }

        Ok(config)
    }

    /// Interpolate variables in a string (supports `{variable_name}` syntax).
    ///
    /// Thin wrapper around the free [`interpolate_string`] helper so other
    /// modules (e.g. the MCP `prompts/get` capability) can reuse the same
    /// substitution rules without re-implementing them.
    fn interpolate_string(
        &self,
        template: &str,
        values: &HashMap<String, serde_json::Value>,
    ) -> Result<String> {
        Ok(interpolate_string(
            template,
            values.iter().map(|(k, v)| (k.as_str(), v)),
        ))
    }

    /// Build agent from interpolated configuration
    fn build_agent_from_config(&self, config: TemplateConfig) -> Result<AgentBuilder> {
        let mut builder = AgentBuilder::new();

        // Set provider
        let provider: ProviderType = config
            .provider
            .parse()
            .map_err(|e| anyhow::anyhow!("Invalid provider '{}': {}", config.provider, e))?;
        builder = builder.provider(provider);

        // Set model if specified
        if let Some(ref model) = config.model {
            builder = builder.model(model.clone());
        }

        // Set system prompt if specified
        if let Some(ref prompt) = config.system_prompt {
            builder = builder.system_prompt(prompt.clone());
        }

        // Enable tools
        if let Some(ref tools) = config.tools {
            for tool in tools {
                builder = builder.enable_tool(tool);
            }
        }

        builder = apply_memory_config(builder, config.memory.as_ref())?;
        #[cfg(feature = "tools-rag")]
        {
            builder = apply_vector_backend_config(builder, config.vector_backend.as_ref())?;
        }
        builder = apply_sql_tools_config(builder, config.sql_tools.as_ref());
        Ok(builder)
    }

    /// Get template ID (derived from name and version)
    pub fn id(&self) -> String {
        format!(
            "{}:{}",
            self.name.to_lowercase().replace(' ', "_"),
            self.version
        )
    }
}

fn apply_memory_config(
    mut builder: AgentBuilder,
    memory_config: Option<&MemoryConfig>,
) -> Result<AgentBuilder> {
    let Some(memory_config) = memory_config else {
        return Ok(builder);
    };
    match memory_config.memory_type.as_str() {
        "buffer" => {
            let config = BufferMemoryConfig {
                max_tokens: memory_config.max_tokens.unwrap_or(4000),
                ..Default::default()
            };
            builder = builder.buffer_memory(config);
        }
        "summary" => {
            let config =
                memory_config
                    .summary_config
                    .clone()
                    .unwrap_or_else(|| SummaryMemoryConfig {
                        max_tokens: memory_config.max_tokens.unwrap_or(4000),
                        ..Default::default()
                    });
            builder = builder.summary_memory(config);
        }
        "entity" => {
            let config =
                memory_config
                    .entity_config
                    .clone()
                    .unwrap_or_else(|| EntityMemoryConfig {
                        max_tokens: memory_config.max_tokens.unwrap_or(4000),
                        ..Default::default()
                    });
            builder = builder.entity_memory(config);
        }
        _ => {
            return Err(anyhow::anyhow!(
                "Invalid memory type: {}",
                memory_config.memory_type
            ));
        }
    }
    Ok(builder)
}

#[cfg(feature = "tools-rag")]
fn apply_vector_backend_config(
    builder: AgentBuilder,
    vs_config: Option<&VectorBackendConfig>,
) -> Result<AgentBuilder> {
    let Some(vs_config) = vs_config else {
        return Ok(builder);
    };
    let backend_config = template_vector_backend(vs_config)?;
    Ok(builder.vector_backend_config(backend_config))
}

#[cfg(feature = "tools-rag")]
fn template_vector_backend(
    vs_config: &VectorBackendConfig,
) -> Result<crate::data::storage::vector::VectorBackendConfig> {
    use crate::data::storage::vector::VectorBackendConfig as VSConfig;
    match vs_config.backend.as_str() {
        "memory" => Ok(VSConfig::memory()),
        #[cfg(feature = "vector-store-redis")]
        "redis" => redis_backend_from_template(vs_config),
        #[cfg(feature = "vector-store-postgres")]
        "postgres" => postgres_backend_from_template(vs_config),
        #[cfg(feature = "vector-store-qdrant-grpc")]
        "qdrant" | "qdrant-grpc" => qdrant_backend_from_template(vs_config),
        other => Err(anyhow::anyhow!(
            "Unsupported vector store backend: {}",
            other
        )),
    }
}

#[cfg(all(feature = "tools-rag", feature = "vector-store-redis"))]
fn redis_backend_from_template(
    vs_config: &VectorBackendConfig,
) -> Result<crate::data::storage::vector::VectorBackendConfig> {
    use crate::data::storage::vector::VectorBackendConfig as VSConfig;
    let url = vs_config
        .config
        .get("url")
        .and_then(|v| v.as_str())
        .ok_or_else(|| anyhow::anyhow!("Redis backend requires 'url' config"))?;
    let prefix = vs_config
        .config
        .get("prefix")
        .and_then(|v| v.as_str())
        .unwrap_or("embeddings:");
    Ok(VSConfig::Redis {
        url: url.to_string(),
        prefix: prefix.to_string(),
    })
}

#[cfg(all(feature = "tools-rag", feature = "vector-store-postgres"))]
fn postgres_backend_from_template(
    vs_config: &VectorBackendConfig,
) -> Result<crate::data::storage::vector::VectorBackendConfig> {
    use crate::data::storage::vector::VectorBackendConfig as VSConfig;
    let connection_string = vs_config
        .config
        .get("url")
        .or_else(|| vs_config.config.get("connection_string"))
        .and_then(|v| v.as_str())
        .ok_or_else(|| {
            anyhow::anyhow!("Postgres backend requires 'url' or 'connection_string' config")
        })?;
    Ok(VSConfig::Postgres {
        connection_string: connection_string.to_string(),
        table_name: vs_config
            .config
            .get("table")
            .or_else(|| vs_config.config.get("table_name"))
            .and_then(|v| v.as_str())
            .unwrap_or("embeddings")
            .to_string(),
        vector_dimension: vs_config
            .config
            .get("dimension")
            .or_else(|| vs_config.config.get("vector_dimension"))
            .and_then(|v| v.as_u64())
            .map(|d| d as usize),
        max_connections: vs_config
            .config
            .get("max_connections")
            .and_then(|v| v.as_u64())
            .map(|n| n as u32)
            .unwrap_or(10),
        hnsw_config: None,
    })
}

#[cfg(all(feature = "tools-rag", feature = "vector-store-qdrant-grpc"))]
fn qdrant_backend_from_template(
    vs_config: &VectorBackendConfig,
) -> Result<crate::data::storage::vector::VectorBackendConfig> {
    use crate::data::storage::vector::VectorBackendConfig as VSConfig;
    let grpc_url = vs_config
        .config
        .get("grpc_url")
        .or_else(|| vs_config.config.get("url"))
        .and_then(|v| v.as_str())
        .ok_or_else(|| anyhow::anyhow!("Qdrant backend requires 'grpc_url' or 'url' config"))?;
    Ok(VSConfig::QdrantGrpc {
        grpc_url: grpc_url.to_string(),
        collection_prefix: vs_config
            .config
            .get("collection_prefix")
            .and_then(|v| v.as_str())
            .map(str::to_string),
        vector_dimension: vs_config
            .config
            .get("vector_dimension")
            .and_then(|v| v.as_u64())
            .map(|n| n as usize)
            .unwrap_or(1536),
        api_key: vs_config
            .config
            .get("api_key")
            .and_then(|v| v.as_str())
            .map(str::to_string),
        batch_size: vs_config
            .config
            .get("batch_size")
            .and_then(|v| v.as_u64())
            .map(|n| n as usize),
    })
}

fn apply_sql_tools_config(
    mut builder: AgentBuilder,
    sql_config: Option<&SqlToolsConfig>,
) -> AgentBuilder {
    let Some(sql_config) = sql_config else {
        return builder;
    };
    builder = builder.enable_sql_tools_with_postgres(
        sql_config.database_url.clone(),
        sql_config.namespace.clone(),
    );
    if let Some(top_k) = sql_config.top_k {
        builder = builder.sql_tools_top_k(top_k);
    }
    if let Some(max_rows) = sql_config.max_rows {
        builder = builder.sql_tools_max_rows(max_rows);
    }
    builder
}

/// Substitute `{variable}` placeholders in `template` using the provided
/// iterator of `(name, value)` pairs.
///
/// `serde_json::Value` inputs are flattened the same way `TemplateRegistry`
/// interpolates agent configuration: strings are inlined verbatim, numbers
/// and booleans use their canonical textual form, and anything else is
/// serialized via `Value::to_string()`.
pub(crate) fn interpolate_string<'a, I>(template: &str, values: I) -> String
where
    I: IntoIterator<Item = (&'a str, &'a serde_json::Value)>,
{
    let mut result = template.to_string();
    for (key, value) in values {
        let placeholder = format!("{{{}}}", key);
        let value_str = match value {
            serde_json::Value::String(s) => s.clone(),
            serde_json::Value::Number(n) => n.to_string(),
            serde_json::Value::Bool(b) => b.to_string(),
            _ => value.to_string(),
        };
        result = result.replace(&placeholder, &value_str);
    }
    result
}
