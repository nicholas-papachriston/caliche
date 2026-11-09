//! JavaScript bindings for Caliche Storage

use caliche::storage::StorageConfig as RustStorageConfig;
use napi_derive::napi;
use std::path::PathBuf;

/// JavaScript wrapper for StorageConfig
#[napi]
pub struct StorageConfig {
    pub(crate) inner: RustStorageConfig,
}

#[napi]
impl StorageConfig {
    /// Create a filesystem storage config
    #[napi(constructor, catch_unwind)]
    pub fn filesystem(base_path: String) -> Self {
        Self {
            inner: RustStorageConfig::filesystem(PathBuf::from(base_path)),
        }
    }

    /// Create an AWS S3 storage config
    #[napi(factory, catch_unwind)]
    pub fn aws_s3(
        bucket: String,
        region: Option<String>,
        access_key_id: Option<String>,
        secret_access_key: Option<String>,
        session_token: Option<String>,
    ) -> Self {
        Self {
            inner: caliche::storage::StorageConfig::aws_s3(
                bucket,
                region,
                access_key_id,
                secret_access_key,
                session_token,
            ),
        }
    }

    /// Create a Cloudflare R2 storage config
    #[napi(factory, catch_unwind)]
    pub fn cloudflare_r2(
        bucket: String,
        account_id: String,
        access_key_id: String,
        secret_access_key: String,
        endpoint: Option<String>,
    ) -> Self {
        Self {
            inner: RustStorageConfig::cloudflare_r2(
                bucket,
                account_id,
                access_key_id,
                secret_access_key,
                endpoint,
            ),
        }
    }

    /// Enable/disable caching on storage config
    #[napi(catch_unwind)]
    pub fn with_cache(&self, enabled: bool) -> Self {
        Self {
            inner: self.inner.clone().with_cache(enabled),
        }
    }

    /// Set cache TTL on storage config
    #[napi(catch_unwind)]
    pub fn with_cache_ttl(&self, ttl_seconds: f64) -> Self {
        Self {
            inner: self.inner.clone().with_cache_ttl(ttl_seconds as u64),
        }
    }

    /// Set cache max size on storage config
    #[napi(catch_unwind)]
    pub fn with_cache_max_size(&self, max_size_mb: u32) -> Self {
        Self {
            inner: self.inner.clone().with_cache_max_size(max_size_mb as u64),
        }
    }
}
