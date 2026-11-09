//! Template storage operations for loading and saving templates from storage backends

use crate::data::storage::object::StorageConfig;
use crate::data::storage::object::backend::StorageBackend;
use crate::templates::{AgentTemplate, TemplateLoader};
use anyhow::{Context, Result};
use std::sync::Arc;

/// Helper to create a storage backend from config
pub async fn create_storage_backend(config: &StorageConfig) -> Result<Arc<dyn StorageBackend>> {
    use crate::data::storage::object::StorageBackendType;
    use crate::data::storage::object::backend::{FilesystemBackend, ObjectStoreBackend};
    use crate::data::storage::object::cache::CachedStorageBackend;

    let backend: Arc<dyn StorageBackend> = match &config.backend {
        StorageBackendType::Filesystem => {
            let base_path = config
                .base_path
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("Filesystem backend requires base_path"))?
                .clone();
            Arc::new(FilesystemBackend::new(base_path))
        }
        #[cfg(feature = "s3")]
        StorageBackendType::ObjectStore => {
            let object_store_config = config.object_store.as_ref().ok_or_else(|| {
                anyhow::anyhow!("ObjectStore backend requires object_store config")
            })?;
            let bucket = config
                .default_bucket
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("ObjectStore backend requires default_bucket"))?
                .clone();
            Arc::new(ObjectStoreBackend::new(object_store_config.clone(), bucket).await?)
        }
        #[cfg(not(feature = "s3"))]
        StorageBackendType::ObjectStore => {
            return Err(anyhow::anyhow!(
                "ObjectStore backend requires 's3' feature to be enabled"
            ));
        }
    };

    // Wrap with cache if enabled
    if config.cache.enabled {
        Ok(Arc::new(CachedStorageBackend::new(
            backend,
            config.cache.clone(),
        )))
    } else {
        Ok(backend)
    }
}

/// Load a template from storage by key
pub async fn load_template_from_storage(
    backend: &dyn StorageBackend,
    key: &str,
) -> Result<AgentTemplate> {
    let content = backend.read(key).await?;
    let loader = TemplateLoader::new();
    loader.load_template_from_str(&content, None)
}

/// Save a template to storage
pub async fn save_template_to_storage(
    backend: &dyn StorageBackend,
    template: &AgentTemplate,
    key: &str,
) -> Result<()> {
    // Serialize template to YAML
    #[cfg(feature = "templates")]
    {
        let yaml =
            serde_norway::to_string(template).context("Failed to serialize template to YAML")?;
        backend.write(key, &yaml).await
    }
    #[cfg(not(feature = "templates"))]
    {
        // Fallback to JSON if YAML feature not available
        let json = serde_json::to_string_pretty(template)
            .context("Failed to serialize template to JSON")?;
        backend.write(key, &json).await
    }
}

/// Generate storage key for a template
pub fn template_storage_key(name: &str, version: &str) -> String {
    format!(
        "templates/{}/{}.yaml",
        name.to_lowercase().replace(' ', "_"),
        version
    )
}

/// List all templates in storage
pub async fn list_templates_from_storage(
    backend: &dyn StorageBackend,
    prefix: Option<&str>,
) -> Result<Vec<String>> {
    let prefix = prefix.unwrap_or("templates/");
    backend.list(prefix).await
}
