//! Object-store helpers (bytes in S3/R2/memory), not vector backends.
//!
//! Use [`crate::backend`] for embedding indexes and ANN search. This module
//! only stores opaque objects (snapshots, WAL segments, artifacts). The S3
//! vector backend lives under `backend::s3`; `storage::s3_store` is the
//! generic object API those backends may use.
//!
pub mod config;
pub mod memory_store;
#[cfg(feature = "s3-backend")]
pub mod s3_store;

pub use memory_store::MemoryObjectStore;

use async_trait::async_trait;
use bytes::Bytes;
#[cfg(feature = "s3-backend")]
use std::time::Duration;
use std::time::SystemTime;

pub use config::{ObjectStoreConfig, ObjectStoreProvider};

/// Metadata about an object in storage
#[derive(Debug, Clone)]
pub struct ObjectMetadata {
    pub size: u64,
    pub content_type: Option<String>,
    pub last_modified: Option<SystemTime>,
    pub etag: Option<String>,
}

/// Information about an object in a list operation
#[derive(Debug, Clone)]
pub struct ObjectInfo {
    pub key: String,
    pub size: u64,
    pub last_modified: Option<SystemTime>,
    pub etag: Option<String>,
}

/// Trait for object storage operations
#[async_trait]
pub trait ObjectStore: Send + Sync {
    /// Get an object from storage
    async fn get_object(
        &self,
        bucket: &str,
        key: &str,
    ) -> Result<Bytes, crate::errors::DripstoneError>;

    /// Put an object into storage
    async fn put_object(
        &self,
        bucket: &str,
        key: &str,
        data: Bytes,
        content_type: Option<&str>,
    ) -> Result<(), crate::errors::DripstoneError>;

    /// Delete an object from storage
    async fn delete_object(
        &self,
        bucket: &str,
        key: &str,
    ) -> Result<(), crate::errors::DripstoneError>;

    /// Get object metadata without downloading the object
    async fn head_object(
        &self,
        bucket: &str,
        key: &str,
    ) -> Result<ObjectMetadata, crate::errors::DripstoneError>;

    /// List objects in a bucket with a prefix
    async fn list_objects(
        &self,
        bucket: &str,
        prefix: Option<&str>,
    ) -> Result<Vec<ObjectInfo>, crate::errors::DripstoneError>;
}

#[cfg(feature = "s3-backend")]
pub(crate) const OBJECT_STORE_PUT_MAX_ATTEMPTS: u32 = 4;

#[cfg(feature = "s3-backend")]
pub(crate) fn is_transient_object_store_error(message: &str) -> bool {
    let lower = message.to_ascii_lowercase();
    [
        "dispatch failure",
        "timeout",
        "temporarily unavailable",
        "connection reset",
        "connection refused",
        "broken pipe",
        "request timed out",
        "service unavailable",
        "internalerror",
        "slowdown",
        "too many requests",
        "throttl",
        "503",
        "504",
    ]
    .iter()
    .any(|needle| lower.contains(needle))
}

#[cfg(feature = "s3-backend")]
pub(crate) fn object_store_put_backoff(attempt: u32) -> Duration {
    let base_ms = 200_u64.saturating_mul(1_u64 << attempt.min(5));
    Duration::from_millis(base_ms.min(3_000))
}

#[cfg(feature = "s3-backend")]
pub use config::parse_object_store_path_style;
#[cfg(feature = "s3-backend")]
pub use s3_store::S3Store;

/// Create an `ObjectStore` instance from configuration
///
/// # Errors
///
/// Returns an error when the operation cannot be completed.
#[cfg(feature = "s3-backend")]
pub async fn create_object_store(
    config: ObjectStoreConfig,
) -> Result<std::sync::Arc<dyn ObjectStore>, crate::errors::DripstoneError> {
    match config.provider {
        ObjectStoreProvider::AwsS3 {
            region,
            access_key_id,
            secret_access_key,
            session_token,
        } => {
            let store =
                S3Store::aws(region, access_key_id, secret_access_key, session_token).await?;
            Ok(std::sync::Arc::new(store))
        }
        ObjectStoreProvider::CloudflareR2 {
            account_id,
            access_key_id,
            secret_access_key,
            endpoint,
        } => {
            let store =
                S3Store::cloudflare_r2(account_id, access_key_id, secret_access_key, endpoint)
                    .await?;
            Ok(std::sync::Arc::new(store))
        }
        ObjectStoreProvider::GenericS3 {
            endpoint,
            region,
            path_style,
            access_key_id,
            secret_access_key,
            session_token,
        } => {
            let store = S3Store::generic(
                endpoint,
                region,
                path_style,
                access_key_id,
                secret_access_key,
                session_token,
            )
            .await?;
            Ok(std::sync::Arc::new(store))
        }
    }
}
