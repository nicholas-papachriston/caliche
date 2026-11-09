//! Template loader for loading templates from files or strings.

#[cfg(feature = "s3")]
use crate::data::storage::object::backend::StorageBackend;
use crate::templates::{AgentTemplate, TemplateValidator};
use anyhow::{Context, Result};
use std::path::{Path, PathBuf};

/// Loads agent templates from various sources
pub struct TemplateLoader {
    validator: TemplateValidator,
}

impl TemplateLoader {
    /// Create a new template loader
    pub fn new() -> Self {
        Self {
            validator: TemplateValidator::new(),
        }
    }

    /// Load a template from a file path
    pub async fn load_template<P: AsRef<Path>>(&self, path: P) -> Result<AgentTemplate> {
        let path = path.as_ref();
        let content = tokio::fs::read_to_string(path)
            .await
            .with_context(|| format!("Failed to read template file: {}", path.display()))?;

        self.load_template_from_str(&content, Some(path.to_path_buf()))
    }

    /// Load a template from a string
    pub fn load_template_from_str(
        &self,
        content: &str,
        file_path: Option<PathBuf>,
    ) -> Result<AgentTemplate> {
        // Try YAML first, then JSON
        let template: AgentTemplate = if content.trim_start().starts_with('{') {
            // JSON format
            serde_json::from_str(content).context("Failed to parse template as JSON")?
        } else {
            // YAML format
            #[cfg(feature = "templates")]
            {
                serde_norway::from_str(content).context("Failed to parse template as YAML")?
            }
            #[cfg(not(feature = "templates"))]
            {
                return Err(anyhow::anyhow!(
                    "YAML parsing requires the 'templates' feature to be enabled"
                ));
            }
        };

        // Validate the template
        self.validator.validate(&template)?;

        // Set file path if provided
        let mut template = template;
        template.file_path = file_path;

        Ok(template)
    }

    /// Load all templates from a directory (supports nested structure: prompts/<name>/<version>.yaml)
    pub async fn load_templates_from_dir<P: AsRef<Path>>(
        &self,
        dir: P,
    ) -> Result<Vec<AgentTemplate>> {
        let dir = dir.as_ref();
        let mut templates = Vec::new();
        let mut dirs_to_process = vec![dir.to_path_buf()];

        // Use iterative approach to avoid recursion issues
        while let Some(current_dir) = dirs_to_process.pop() {
            let mut entries = tokio::fs::read_dir(&current_dir)
                .await
                .with_context(|| format!("Failed to read directory: {}", current_dir.display()))?;

            while let Some(entry) = entries.next_entry().await? {
                let path = entry.path();
                if path.is_file() {
                    let ext = path.extension().and_then(|s| s.to_str());
                    if ext == Some("yaml") || ext == Some("yml") || ext == Some("json") {
                        match self.load_template(&path).await {
                            Ok(template) => templates.push(template),
                            Err(e) => {
                                tracing::warn!("Failed to load template {}: {}", path.display(), e);
                            }
                        }
                    }
                } else if path.is_dir() {
                    // Add subdirectory to queue for processing
                    dirs_to_process.push(path);
                }
            }
        }

        Ok(templates)
    }

    /// Load templates from a storage backend
    #[cfg(all(feature = "s3", feature = "templates"))]
    pub async fn load_templates_from_storage(
        &self,
        backend: &dyn StorageBackend,
        prefix: Option<&str>,
    ) -> Result<Vec<AgentTemplate>> {
        use crate::templates::storage::load_template_from_storage;

        let prefix = prefix.unwrap_or("templates/");
        let keys = backend.list(prefix).await?;

        let mut templates = Vec::new();
        for key in keys {
            // Filter for template files
            if key.ends_with(".yaml") || key.ends_with(".yml") || key.ends_with(".json") {
                match load_template_from_storage(backend, &key).await {
                    Ok(template) => templates.push(template),
                    Err(e) => {
                        tracing::warn!("Failed to load template {}: {}", key, e);
                    }
                }
            }
        }

        Ok(templates)
    }
}

impl Default for TemplateLoader {
    fn default() -> Self {
        Self::new()
    }
}
