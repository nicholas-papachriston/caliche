//! Template validator for ensuring template correctness.

use crate::templates::AgentTemplate;
use anyhow::Result;

/// Validates agent templates for correctness
pub struct TemplateValidator;

impl TemplateValidator {
    /// Create a new template validator
    pub fn new() -> Self {
        Self
    }

    /// Validate a template
    pub fn validate(&self, template: &AgentTemplate) -> Result<()> {
        // Validate name
        if template.name.is_empty() {
            return Err(anyhow::anyhow!("Template name cannot be empty"));
        }

        // Validate version
        if template.version.is_empty() {
            return Err(anyhow::anyhow!("Template version cannot be empty"));
        }

        // Validate description
        if template.description.is_empty() {
            return Err(anyhow::anyhow!("Template description cannot be empty"));
        }

        // Validate provider
        template
            .config
            .provider
            .parse::<crate::llm::ProviderType>()
            .map_err(|e| {
                anyhow::anyhow!("Invalid provider '{}': {}", template.config.provider, e)
            })?;

        // Validate memory configuration if present
        if let Some(ref memory) = template.config.memory {
            match memory.memory_type.as_str() {
                "buffer" | "summary" | "entity" => {}
                _ => {
                    return Err(anyhow::anyhow!(
                        "Invalid memory type: {}. Must be one of: buffer, summary, entity",
                        memory.memory_type
                    ));
                }
            }
        }

        // Validate vector store configuration if present
        #[cfg(feature = "tools-rag")]
        if let Some(ref vs_config) = template.config.vector_backend {
            match vs_config.backend.as_str() {
                "memory" | "redis" | "postgres" | "qdrant" | "pinecone" | "chroma" | "s3" => {}
                _ => {
                    return Err(anyhow::anyhow!(
                        "Invalid vector store backend: {}. Must be one of: memory, redis, postgres, qdrant, pinecone, chroma, s3",
                        vs_config.backend
                    ));
                }
            }
        }

        // Validate SQL tools configuration if present
        if let Some(ref sql_config) = template.config.sql_tools {
            if sql_config.database_url.is_empty() {
                return Err(anyhow::anyhow!("SQL tools database_url cannot be empty"));
            }
            if sql_config.namespace.is_empty() {
                return Err(anyhow::anyhow!("SQL tools namespace cannot be empty"));
            }
        }

        Ok(())
    }
}

impl Default for TemplateValidator {
    fn default() -> Self {
        Self::new()
    }
}
