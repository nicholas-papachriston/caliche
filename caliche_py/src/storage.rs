//! Python bindings for Caliche Storage

use caliche::storage::StorageConfig;
use pyo3::prelude::*;
use std::path::PathBuf;

/// Python wrapper for `StorageConfig`
#[pyclass]
pub struct PyStorageConfig {
    inner: StorageConfig,
}

#[pymethods]
impl PyStorageConfig {
    /// Create a filesystem storage config
    #[new]
    #[pyo3(signature = (base_path))]
    fn new(base_path: String) -> Self {
        Self {
            inner: StorageConfig::filesystem(PathBuf::from(base_path)),
        }
    }

    /// Create an AWS S3 storage config
    #[staticmethod]
    #[pyo3(signature = (bucket, region=None, access_key_id=None, secret_access_key=None, session_token=None))]
    fn aws_s3(
        bucket: String,
        region: Option<String>,
        access_key_id: Option<String>,
        secret_access_key: Option<String>,
        session_token: Option<String>,
    ) -> Self {
        Self {
            inner: StorageConfig::aws_s3(
                bucket,
                region,
                access_key_id,
                secret_access_key,
                session_token,
            ),
        }
    }

    /// Create a Cloudflare R2 storage config
    #[staticmethod]
    #[pyo3(signature = (bucket, account_id, access_key_id, secret_access_key, endpoint=None))]
    fn cloudflare_r2(
        bucket: String,
        account_id: String,
        access_key_id: String,
        secret_access_key: String,
        endpoint: Option<String>,
    ) -> Self {
        Self {
            inner: StorageConfig::cloudflare_r2(
                bucket,
                account_id,
                access_key_id,
                secret_access_key,
                endpoint,
            ),
        }
    }

    /// Enable or disable caching
    fn with_cache(&self, enabled: bool) -> Self {
        Self {
            inner: self.inner.clone().with_cache(enabled),
        }
    }

    /// Set cache TTL in seconds
    fn with_cache_ttl(&self, ttl_seconds: u64) -> Self {
        Self {
            inner: self.inner.clone().with_cache_ttl(ttl_seconds),
        }
    }

    /// Set cache max size in MB
    fn with_cache_max_size(&self, max_size_mb: u64) -> Self {
        Self {
            inner: self.inner.clone().with_cache_max_size(max_size_mb),
        }
    }
}

impl PyStorageConfig {
    pub const fn inner(&self) -> &StorageConfig {
        &self.inner
    }
}
