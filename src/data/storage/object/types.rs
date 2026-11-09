//! Object storage types and trait definitions

use async_trait::async_trait;
use bytes::Bytes;
use std::time::SystemTime;
use thiserror::Error;

/// Information about an object in storage
#[derive(Debug, Clone)]
pub struct ObjectInfo {
    pub key: String,
    pub size: u64,
    pub last_modified: Option<SystemTime>,
    pub etag: Option<String>,
}

/// Metadata about an object in storage
#[derive(Debug, Clone)]
pub struct ObjectMetadata {
    pub size: u64,
    pub content_type: Option<String>,
    pub last_modified: Option<SystemTime>,
    pub etag: Option<String>,
}

/// Structured object storage errors for backend-agnostic handling.
#[derive(Debug, Error)]
pub enum ObjectStoreError {
    #[error("Object not found: {key}")]
    NotFound { key: String },
    #[error("Access denied: {message}")]
    AccessDenied { message: String },
    #[error("Object storage error: {message}")]
    Other { message: String },
}

impl ObjectStoreError {
    pub fn classify(key: &str, error: &anyhow::Error) -> Self {
        let message = error.to_string();
        if message.contains("404")
            || message.contains("NotFound")
            || message.contains("NoSuchKey")
            || message.contains("not found")
        {
            return Self::NotFound {
                key: key.to_string(),
            };
        }
        if message.contains("403")
            || message.contains("AccessDenied")
            || message.contains("access denied")
        {
            return Self::AccessDenied { message };
        }
        Self::Other { message }
    }
}

/// Trait for object storage backends (S3-compatible)
#[async_trait]
pub trait ObjectStore: Send + Sync {
    /// Get an object from storage
    async fn get_object(&self, bucket: &str, key: &str) -> Result<Bytes, anyhow::Error>;

    /// Put an object to storage
    async fn put_object(
        &self,
        bucket: &str,
        key: &str,
        data: Bytes,
        content_type: Option<&str>,
    ) -> Result<(), anyhow::Error>;

    /// Delete an object from storage
    async fn delete_object(&self, bucket: &str, key: &str) -> Result<(), anyhow::Error>;

    /// Get object metadata (HEAD request)
    async fn head_object(&self, bucket: &str, key: &str) -> Result<ObjectMetadata, anyhow::Error>;

    /// List objects in a bucket with optional prefix
    async fn list_objects(
        &self,
        bucket: &str,
        prefix: Option<&str>,
    ) -> Result<Vec<ObjectInfo>, anyhow::Error>;
}
