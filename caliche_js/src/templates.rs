//! JavaScript bindings for Caliche Templates

use caliche::templates::{
    AgentTemplate as RustAgentTemplate, TemplateRegistry as RustTemplateRegistry,
};
use napi_derive::napi;
use std::sync::Arc;

/// JavaScript wrapper for `TemplateRegistry`
#[napi]
pub struct TemplateRegistry {
    inner: Arc<tokio::sync::Mutex<caliche::templates::TemplateRegistry>>,
}

#[napi]
impl TemplateRegistry {
    /// Create a new `TemplateRegistry`
    #[napi(constructor, catch_unwind)]
    pub fn new() -> Self {
        Self {
            inner: Arc::new(tokio::sync::Mutex::new(RustTemplateRegistry::new())),
        }
    }
}

/// Storage-backed factory constructors, split into their own `impl` block so the
/// `#[napi]` macro only ever sees this method when the `s3` feature is enabled
/// (per-method `#[cfg]` inside a shared napi `impl` block is not reliably
/// respected by napi-rs's codegen).
#[cfg(feature = "s3")]
#[napi]
impl TemplateRegistry {
    /// Create a TemplateRegistry with storage config
    /// storage_type: "filesystem", "aws_s3", or "cloudflare_r2"
    #[napi(factory)]
    ///
    /// # Errors
    ///
    /// Returns a NAPI error when validation, synchronization, serialization,
    /// or an underlying runtime operation fails.
    pub async fn with_storage(
        storage_type: String,
        storage_params: serde_json::Value,
    ) -> napi::Result<Self> {
        let config = match storage_type.as_str() {
            "filesystem" => {
                let base_path = storage_params
                    .get("base_path")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| {
                        napi::Error::from_reason("Missing 'base_path' for filesystem storage")
                    })?
                    .to_string();
                caliche::storage::StorageConfig::filesystem(std::path::PathBuf::from(base_path))
            }
            "aws_s3" => {
                let bucket = storage_params
                    .get("bucket")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| napi::Error::from_reason("Missing 'bucket' for aws_s3 storage"))?
                    .to_string();
                let region = storage_params
                    .get("region")
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string());
                let access_key_id = storage_params
                    .get("access_key_id")
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string());
                let secret_access_key = storage_params
                    .get("secret_access_key")
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string());
                let session_token = storage_params
                    .get("session_token")
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string());
                caliche::storage::StorageConfig::aws_s3(
                    bucket,
                    region,
                    access_key_id,
                    secret_access_key,
                    session_token,
                )
            }
            "cloudflare_r2" => {
                let bucket = storage_params
                    .get("bucket")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| {
                        napi::Error::from_reason("Missing 'bucket' for cloudflare_r2 storage")
                    })?
                    .to_string();
                let account_id = storage_params
                    .get("account_id")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| {
                        napi::Error::from_reason("Missing 'account_id' for cloudflare_r2 storage")
                    })?
                    .to_string();
                let access_key_id = storage_params
                    .get("access_key_id")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| {
                        napi::Error::from_reason(
                            "Missing 'access_key_id' for cloudflare_r2 storage",
                        )
                    })?
                    .to_string();
                let secret_access_key = storage_params
                    .get("secret_access_key")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| {
                        napi::Error::from_reason(
                            "Missing 'secret_access_key' for cloudflare_r2 storage",
                        )
                    })?
                    .to_string();
                let endpoint = storage_params
                    .get("endpoint")
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string());
                caliche::storage::StorageConfig::cloudflare_r2(
                    bucket,
                    account_id,
                    access_key_id,
                    secret_access_key,
                    endpoint,
                )
            }
            _ => {
                return Err(napi::Error::from_reason(format!(
                    "Unknown storage type: {}",
                    storage_type
                )));
            }
        };
        let registry = RustTemplateRegistry::with_storage(config)
            .await
            .map_err(|e| {
                napi::Error::from_reason(format!("Failed to create registry with storage: {}", e))
            })?;
        Ok(Self {
            inner: Arc::new(tokio::sync::Mutex::new(registry)),
        })
    }
}

#[napi]
impl TemplateRegistry {
    /// Add a template directory to the registry
    #[napi]
    ///
    /// # Errors
    ///
    /// Returns a NAPI error when validation, synchronization, serialization,
    /// or an underlying runtime operation fails.
    pub async fn add_template_dir(&self, dir: String) -> napi::Result<()> {
        let mut reg = self.inner.lock().await;
        reg.add_template_dir(dir);
        drop(reg);
        Ok(())
    }

    /// Load templates from configured sources
    #[napi]
    pub async fn load_templates(&self) -> napi::Result<u32> {
        let reg = self.inner.lock().await;
        let count = reg
            .load_templates()
            .await
            .map_err(|e| napi::Error::from_reason(format!("Failed to load templates: {e}")))?;
        drop(reg);
        u32::try_from(count).map_err(|_| {
            napi::Error::from_reason("template count exceeds the JavaScript API limit")
        })
    }

    /// Get a template by ID
    #[napi]
    ///
    /// # Errors
    ///
    /// Returns a NAPI error when validation, synchronization, serialization,
    /// or an underlying runtime operation fails.
    pub async fn get(&self, id: String) -> napi::Result<Option<AgentTemplate>> {
        let reg = self.inner.lock().await;
        let template = reg.get(&id).map(Into::into);
        drop(reg);
        Ok(template)
    }

    /// List all templates
    #[napi]
    pub async fn list(&self) -> Vec<AgentTemplate> {
        let reg = self.inner.lock().await;
        reg.list().into_iter().map(Into::into).collect()
    }
}

/// JavaScript wrapper for `AgentTemplate`
#[napi]
pub struct AgentTemplate {
    inner: RustAgentTemplate,
}

#[napi]
impl AgentTemplate {
    /// Get template ID
    #[napi(catch_unwind)]
    pub fn id(&self) -> String {
        self.inner.id()
    }

    /// Get template name
    #[napi(catch_unwind)]
    pub fn name(&self) -> String {
        self.inner.name.clone()
    }

    /// Get template description
    #[napi(catch_unwind)]
    pub fn description(&self) -> String {
        self.inner.description.clone()
    }

    /// Get template version
    #[napi(catch_unwind)]
    pub fn version(&self) -> String {
        self.inner.version.clone()
    }
}

impl From<RustAgentTemplate> for AgentTemplate {
    fn from(template: RustAgentTemplate) -> Self {
        Self { inner: template }
    }
}

impl From<Arc<RustAgentTemplate>> for AgentTemplate {
    fn from(template: Arc<RustAgentTemplate>) -> Self {
        Self {
            inner: (*template).clone(),
        }
    }
}
