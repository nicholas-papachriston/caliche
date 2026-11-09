//! Prompt template storage operations for loading and saving prompts from storage backends

use crate::data::storage::object::StorageConfig;
use crate::data::storage::object::backend::StorageBackend;
use crate::prompts::PromptTemplate;
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;

/// Metadata for a stored prompt template
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PromptTemplateMetadata {
    /// Template name
    pub name: String,
    /// Template string
    pub template: String,
    /// Variables required by the template
    pub variables: Vec<String>,
    /// Partial variables (pre-filled)
    pub partial_variables: HashMap<String, String>,
    /// Description (optional)
    pub description: Option<String>,
    /// Created timestamp
    pub created_at: Option<String>,
}

impl From<&PromptTemplate> for PromptTemplateMetadata {
    fn from(template: &PromptTemplate) -> Self {
        Self {
            name: String::new(), // Will be set when saving
            template: template.template().to_string(),
            variables: template.variables().iter().cloned().collect(),
            partial_variables: template.partial_variables().clone(),
            description: None,
            created_at: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs().to_string())
                .ok(),
        }
    }
}

impl TryFrom<PromptTemplateMetadata> for PromptTemplate {
    type Error = anyhow::Error;

    fn try_from(metadata: PromptTemplateMetadata) -> Result<Self> {
        let mut template = PromptTemplate::new(metadata.template);

        // Restore partial variables
        for (key, value) in metadata.partial_variables {
            template = template.with_partial(key, value);
        }

        Ok(template)
    }
}

/// Prompt template store for managing saved prompts
pub struct PromptTemplateStore {
    backend: Arc<dyn StorageBackend>,
}

impl PromptTemplateStore {
    /// Create a new prompt template store
    pub async fn new(config: StorageConfig) -> Result<Self> {
        let backend = crate::templates::storage::create_storage_backend(&config).await?;
        Ok(Self { backend })
    }

    /// Create from an existing backend
    pub fn from_backend(backend: Arc<dyn StorageBackend>) -> Self {
        Self { backend }
    }

    /// Save a prompt template to storage
    pub async fn save(&self, name: &str, template: &PromptTemplate) -> Result<()> {
        let mut metadata: PromptTemplateMetadata = template.into();
        metadata.name = name.to_string();

        let key = prompt_storage_key(name);
        let json = serde_json::to_string_pretty(&metadata)
            .context("Failed to serialize prompt template")?;

        self.backend.write(&key, &json).await
    }

    /// Load a prompt template from storage
    pub async fn load(&self, name: &str) -> Result<PromptTemplate> {
        let key = prompt_storage_key(name);
        let content = self.backend.read(&key).await?;

        let metadata: PromptTemplateMetadata =
            serde_json::from_str(&content).context("Failed to deserialize prompt template")?;

        PromptTemplate::try_from(metadata)
    }

    /// List all prompt template names in storage
    pub async fn list(&self) -> Result<Vec<String>> {
        let keys = self.backend.list("prompts/").await?;

        // Extract names from keys (prompts/{name}.json)
        let names: Vec<String> = keys
            .iter()
            .filter_map(|key| {
                key.strip_prefix("prompts/")
                    .and_then(|s| s.strip_suffix(".json"))
                    .map(|s| s.to_string())
            })
            .collect();

        Ok(names)
    }

    /// Delete a prompt template from storage
    pub async fn delete(&self, name: &str) -> Result<()> {
        let key = prompt_storage_key(name);
        self.backend.delete(&key).await
    }

    /// Check if a prompt template exists
    pub async fn exists(&self, name: &str) -> Result<bool> {
        let key = prompt_storage_key(name);
        self.backend.exists(&key).await
    }
}

/// Generate storage key for a prompt template
fn prompt_storage_key(name: &str) -> String {
    format!("prompts/{}.json", name.to_lowercase().replace(' ', "_"))
}
