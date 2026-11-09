//! In-memory [`ObjectStore`] for tests and local single-WAL durability.

use async_trait::async_trait;
use bytes::Bytes;
use dashmap::DashMap;
use std::sync::Arc;
use std::time::SystemTime;

use super::{ObjectInfo, ObjectMetadata, ObjectStore};
use crate::errors::DripstoneError;

/// Process-local object store keyed by `{bucket}/{key}`.
#[derive(Debug, Default)]
pub struct MemoryObjectStore {
    objects: DashMap<String, Bytes>,
}

impl MemoryObjectStore {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    #[must_use]
    pub fn shared() -> Arc<Self> {
        Arc::new(Self::new())
    }

    fn composite(bucket: &str, key: &str) -> String {
        format!("{bucket}/{key}")
    }
}

#[async_trait]
impl ObjectStore for MemoryObjectStore {
    async fn get_object(&self, bucket: &str, key: &str) -> Result<Bytes, DripstoneError> {
        self.objects
            .get(&Self::composite(bucket, key))
            .map(|v| v.clone())
            .ok_or_else(|| DripstoneError::storage(format!("object not found: {bucket}/{key}")))
    }

    async fn put_object(
        &self,
        bucket: &str,
        key: &str,
        data: Bytes,
        _content_type: Option<&str>,
    ) -> Result<(), DripstoneError> {
        self.objects.insert(Self::composite(bucket, key), data);
        Ok(())
    }

    async fn delete_object(&self, bucket: &str, key: &str) -> Result<(), DripstoneError> {
        self.objects.remove(&Self::composite(bucket, key));
        Ok(())
    }

    async fn head_object(&self, bucket: &str, key: &str) -> Result<ObjectMetadata, DripstoneError> {
        let bytes = self.get_object(bucket, key).await?;
        Ok(ObjectMetadata {
            size: bytes.len() as u64,
            content_type: Some("application/octet-stream".to_string()),
            last_modified: Some(SystemTime::now()),
            etag: None,
        })
    }

    async fn list_objects(
        &self,
        bucket: &str,
        prefix: Option<&str>,
    ) -> Result<Vec<ObjectInfo>, DripstoneError> {
        let bucket_prefix = format!("{bucket}/");
        let key_prefix = prefix.unwrap_or("");
        let mut out = Vec::new();
        for entry in &self.objects {
            let composite = entry.key();
            let Some(key) = composite.strip_prefix(&bucket_prefix) else {
                continue;
            };
            if !key.starts_with(key_prefix) {
                continue;
            }
            out.push(ObjectInfo {
                key: key.to_string(),
                size: entry.value().len() as u64,
                last_modified: Some(SystemTime::now()),
                etag: None,
            });
        }
        out.sort_by(|a, b| a.key.cmp(&b.key));
        Ok(out)
    }
}
