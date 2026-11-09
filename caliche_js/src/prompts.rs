//! JavaScript bindings for Caliche Prompt Templates
#![expect(
    clippy::needless_pass_by_value,
    reason = "NAPI must own values converted from JavaScript arguments at the ABI boundary"
)]

use caliche::prompts::PromptTemplate as RustPromptTemplate;
use napi_derive::napi;
use serde_json::Value;
#[cfg(feature = "s3")]
use std::sync::Arc;

/// JavaScript wrapper for `PromptTemplate`
#[napi]
pub struct PromptTemplate {
    inner: RustPromptTemplate,
}

#[napi]
impl PromptTemplate {
    /// Create a new `PromptTemplate`
    #[napi(constructor, catch_unwind)]
    pub fn new(template: String) -> Self {
        Self {
            inner: RustPromptTemplate::new(&template),
        }
    }

    /// Format a `PromptTemplate` with variables
    #[napi(catch_unwind)]
    ///
    /// # Errors
    ///
    /// Returns a NAPI error when validation, synchronization, serialization,
    /// or an underlying runtime operation fails.
    pub fn format(&self, variables: Value) -> napi::Result<String> {
        self.inner
            .format(&variables)
            .map_err(|e| napi::Error::from_reason(format!("Template formatting failed: {e}")))
    }

    /// Get variables from a `PromptTemplate`
    #[napi(catch_unwind)]
    pub fn variables(&self) -> Vec<String> {
        self.inner.variables().iter().cloned().collect()
    }

    /// Create a summarization template
    #[napi(factory, catch_unwind)]
    pub fn summarize() -> Self {
        Self {
            inner: RustPromptTemplate::summarize(),
        }
    }

    /// Create an analysis template
    #[napi(factory, catch_unwind)]
    pub fn analyze() -> Self {
        Self {
            inner: RustPromptTemplate::analyze(),
        }
    }

    /// Create a translation template
    #[napi(factory, catch_unwind)]
    pub fn translate() -> Self {
        Self {
            inner: RustPromptTemplate::translate(),
        }
    }
}

/// JavaScript wrapper for PromptTemplateStore
#[cfg(feature = "s3")]
#[napi]
pub struct PromptTemplateStore {
    inner: Arc<tokio::sync::Mutex<caliche::prompts::PromptTemplateStore>>,
}

#[cfg(feature = "s3")]
#[napi]
impl PromptTemplateStore {
    /// Create a PromptTemplateStore with storage config
    /// storage_type: "filesystem", "aws_s3", or "cloudflare_r2"
    /// For filesystem: pass base_path in storage_params
    /// For aws_s3: pass bucket, region, access_key_id, secret_access_key, session_token in storage_params
    /// For cloudflare_r2: pass bucket, account_id, access_key_id, secret_access_key, endpoint in storage_params
    #[napi(factory)]
    ///
    /// # Errors
    ///
    /// Returns a NAPI error when validation, synchronization, serialization,
    /// or an underlying runtime operation fails.
    pub async fn new(
        storage_type: String,
        storage_params: serde_json::Value,
    ) -> napi::Result<Self> {
        let storage_config = match storage_type.as_str() {
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

        let store = caliche::prompts::PromptTemplateStore::new(storage_config)
            .await
            .map_err(|e| {
                napi::Error::from_reason(format!("Failed to create prompt store: {}", e))
            })?;
        Ok(Self {
            inner: Arc::new(tokio::sync::Mutex::new(store)),
        })
    }

    /// Save a prompt template
    #[napi]
    ///
    /// # Errors
    ///
    /// Returns a NAPI error when validation, synchronization, serialization,
    /// or an underlying runtime operation fails.
    pub async fn save(&self, name: String, template: String) -> napi::Result<()> {
        let template_inner = RustPromptTemplate::new(&template);
        let store_guard = self.inner.lock().await;
        store_guard
            .save(&name, &template_inner)
            .await
            .map_err(|e| napi::Error::from_reason(format!("Failed to save prompt: {}", e)))
    }

    /// Load a prompt template
    #[napi]
    pub async fn load(&self, name: String) -> napi::Result<PromptTemplate> {
        let store_guard = self.inner.lock().await;
        let template = store_guard
            .load(&name)
            .await
            .map_err(|e| napi::Error::from_reason(format!("Failed to load prompt: {}", e)))?;
        Ok(PromptTemplate { inner: template })
    }

    /// List all prompt templates
    #[napi]
    ///
    /// # Errors
    ///
    /// Returns a NAPI error when validation, synchronization, serialization,
    /// or an underlying runtime operation fails.
    pub async fn list(&self) -> napi::Result<Vec<String>> {
        let store_guard = self.inner.lock().await;
        store_guard
            .list()
            .await
            .map_err(|e| napi::Error::from_reason(format!("Failed to list prompts: {}", e)))
    }
}
