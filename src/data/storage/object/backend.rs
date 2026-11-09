//! Storage backend abstraction for templates and prompts
//!
//! Provides a unified interface for reading/writing templates and prompts
//! from various storage backends (filesystem, S3, R2, etc.)

use async_trait::async_trait;
use std::path::PathBuf;

/// Unified storage backend trait for templates and prompts
#[async_trait]
pub trait StorageBackend: Send + Sync {
    /// Read content from storage by key
    async fn read(&self, key: &str) -> Result<String, anyhow::Error>;

    /// Write content to storage by key
    async fn write(&self, key: &str, content: &str) -> Result<(), anyhow::Error>;

    /// List all keys with the given prefix
    async fn list(&self, prefix: &str) -> Result<Vec<String>, anyhow::Error>;

    /// Delete a key from storage
    async fn delete(&self, key: &str) -> Result<(), anyhow::Error>;

    /// Check if a key exists in storage
    async fn exists(&self, key: &str) -> Result<bool, anyhow::Error>;
}

/// Filesystem-based storage backend
pub struct FilesystemBackend {
    base_path: PathBuf,
}

impl FilesystemBackend {
    /// Create a new filesystem backend with a base path
    pub fn new(base_path: impl Into<PathBuf>) -> Self {
        Self {
            base_path: base_path.into(),
        }
    }

    /// Get the full path for a key
    fn key_to_path(&self, key: &str) -> PathBuf {
        // Normalize the key to prevent directory traversal
        let mut parts = Vec::new();
        for part in key.split('/') {
            if !part.is_empty() && part != "." && part != ".." {
                parts.push(part);
            }
        }
        let normalized_key = parts.join("/");
        self.base_path.join(normalized_key)
    }
}

#[async_trait]
impl StorageBackend for FilesystemBackend {
    async fn read(&self, key: &str) -> Result<String, anyhow::Error> {
        let path = self.key_to_path(key);
        tokio::fs::read_to_string(&path)
            .await
            .with_context(|| format!("Failed to read file: {}", path.display()))
    }

    async fn write(&self, key: &str, content: &str) -> Result<(), anyhow::Error> {
        let path = self.key_to_path(key);

        // Create parent directories if they don't exist
        if let Some(parent) = path.parent() {
            tokio::fs::create_dir_all(parent)
                .await
                .with_context(|| format!("Failed to create directory: {}", parent.display()))?;
        }

        tokio::fs::write(&path, content)
            .await
            .with_context(|| format!("Failed to write file: {}", path.display()))
    }

    async fn list(&self, prefix: &str) -> Result<Vec<String>, anyhow::Error> {
        let prefix_path = self.key_to_path(prefix);

        if !prefix_path.exists() {
            return Ok(Vec::new());
        }

        let mut keys = Vec::new();
        let mut dirs_to_process = vec![prefix_path.clone()];

        while let Some(current_dir) = dirs_to_process.pop() {
            let mut entries = tokio::fs::read_dir(&current_dir)
                .await
                .with_context(|| format!("Failed to read directory: {}", current_dir.display()))?;

            while let Some(entry) = entries.next_entry().await? {
                let path = entry.path();
                if path.is_file() {
                    // Convert absolute path back to relative key
                    if let Ok(relative) = path.strip_prefix(&self.base_path)
                        && let Some(key) = relative.to_str()
                    {
                        keys.push(key.replace('\\', "/"));
                    }
                } else if path.is_dir() {
                    dirs_to_process.push(path);
                }
            }
        }

        Ok(keys)
    }

    async fn delete(&self, key: &str) -> Result<(), anyhow::Error> {
        let path = self.key_to_path(key);
        tokio::fs::remove_file(&path)
            .await
            .with_context(|| format!("Failed to delete file: {}", path.display()))
    }

    async fn exists(&self, key: &str) -> Result<bool, anyhow::Error> {
        let path = self.key_to_path(key);
        Ok(path.exists() && path.is_file())
    }
}

use anyhow::Context;
use std::sync::Arc;

#[cfg(feature = "s3")]
use crate::data::storage::object::types::ObjectStoreError;
#[cfg(feature = "s3")]
use crate::data::storage::object::{ObjectStore, ObjectStoreConfig, create_object_store};

/// Object storage backend (S3/R2/generic S3)
#[cfg(feature = "s3")]
pub struct ObjectStoreBackend {
    object_store: Arc<dyn ObjectStore>,
    bucket: String,
}

#[cfg(feature = "s3")]
impl ObjectStoreBackend {
    /// Create a new object store backend
    /// # Errors
    ///
    /// Returns an error if the persistence or storage operation fails.
    pub async fn new(
        config: ObjectStoreConfig,
        bucket: impl Into<String>,
    ) -> Result<Self, anyhow::Error> {
        let object_store = create_object_store(config).await?;
        Ok(Self {
            object_store,
            bucket: bucket.into(),
        })
    }
}

/// Infer an HTTP content-type from a storage key's extension.
///
/// Used by `ObjectStoreBackend::write` so JSON/YAML/Markdown payloads are
/// advertised correctly to downstream consumers instead of always being
/// labelled `text/plain`.
#[cfg(feature = "s3")]
fn infer_content_type(key: &str) -> &'static str {
    let lower = key.rsplit('.').next().map(str::to_ascii_lowercase);
    match lower.as_deref() {
        Some("json") => "application/json",
        Some("yaml" | "yml") => "application/yaml",
        Some("md" | "markdown") => "text/markdown",
        Some("html" | "htm") => "text/html",
        Some("csv") => "text/csv",
        Some("xml") => "application/xml",
        Some("toml") => "application/toml",
        _ => "text/plain",
    }
}

#[cfg(feature = "s3")]
#[async_trait]
impl StorageBackend for ObjectStoreBackend {
    async fn read(&self, key: &str) -> Result<String, anyhow::Error> {
        let bytes = self.object_store.get_object(&self.bucket, key).await?;
        String::from_utf8(bytes.into()).map_err(|e| anyhow::anyhow!("Failed to decode UTF-8: {e}"))
    }

    async fn write(&self, key: &str, content: &str) -> Result<(), anyhow::Error> {
        let bytes = Bytes::copy_from_slice(content.as_bytes());
        let content_type = infer_content_type(key);
        self.object_store
            .put_object(&self.bucket, key, bytes, Some(content_type))
            .await
    }

    async fn list(&self, prefix: &str) -> Result<Vec<String>, anyhow::Error> {
        let objects = self
            .object_store
            .list_objects(&self.bucket, Some(prefix))
            .await?;
        Ok(objects.into_iter().map(|obj| obj.key).collect())
    }

    async fn delete(&self, key: &str) -> Result<(), anyhow::Error> {
        self.object_store.delete_object(&self.bucket, key).await
    }

    async fn exists(&self, key: &str) -> Result<bool, anyhow::Error> {
        match self.object_store.head_object(&self.bucket, key).await {
            Ok(_) => Ok(true),
            Err(e) => match ObjectStoreError::classify(key, &e) {
                ObjectStoreError::NotFound { .. } => Ok(false),
                _ => Err(e),
            },
        }
    }
}

#[cfg(feature = "s3")]
use bytes::Bytes;
