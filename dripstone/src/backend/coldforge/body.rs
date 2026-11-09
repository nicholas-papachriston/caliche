//! Chunk-id envelopes for ColdForge.
//!
//! Object keys are `{prefix}bodies/{namespace}/{chunk_id}`, the same identity
//! as [`crate::backend::s3::S3EmbeddingBackend`] record keys (`namespace` + chunk
//! id). A cluster index is never part of the key. The in-process map is the
//! read path; the object is the durable envelope. Neither is the LRU cache, and
//! neither is the scoring map on `CentroidIndex`.

use std::collections::HashMap;
use std::sync::Arc;

use bytes::Bytes;
use futures::stream::{self, StreamExt};
use tokio::sync::RwLock;

use crate::embedding::EmbeddingRecord;
use crate::errors::DripstoneError;
use crate::storage::ObjectStore;

/// Path segment between the ColdForge prefix and `{namespace}/{chunk_id}`.
pub(crate) const BODY_KEY_SEGMENT: &str = "bodies";

const BODY_CONTENT_TYPE: &str = "application/json";

/// Concurrent object GETs when cold open has membership but no in-memory envelopes.
///
/// A compacted namespace reloads centroids and chunk ids from the snapshot, then
/// reads each body. WAL replay does not use this path: the segment already
/// carries the envelopes, and replaying them must not PUT the body keys again.
const BODY_HYDRATE_PARALLELISM: usize = super::DEFAULT_BODY_PUT_CONCURRENCY;

/// Envelope store keyed by chunk id (`{doc_id}#{chunk_index}`).
pub(super) struct ChunkBodyStore {
    store: Arc<dyn ObjectStore>,
    bucket: String,
    key_prefix: String,
    records: RwLock<HashMap<String, HashMap<String, EmbeddingRecord>>>,
}

impl ChunkBodyStore {
    pub(super) fn new(store: Arc<dyn ObjectStore>, bucket: String, key_prefix: String) -> Self {
        Self {
            store,
            bucket,
            key_prefix,
            records: RwLock::new(HashMap::new()),
        }
    }

    /// `{prefix}bodies/{namespace}/{chunk_id}`.
    #[must_use]
    pub(crate) fn object_key(key_prefix: &str, namespace: &str, chunk_id: &str) -> String {
        crate::backend::s3::layout::chunk_body_object_key(key_prefix, namespace, chunk_id)
    }

    fn key(&self, namespace: &str, chunk_id: &str) -> String {
        Self::object_key(&self.key_prefix, namespace, chunk_id)
    }

    /// PUT distinct chunk bodies with bounded concurrency, then remember them.
    ///
    /// Duplicate ids in `records` PUT only the last value so a slow write cannot
    /// leave an older object. Callers still append every WAL record in order
    /// after this returns. A failed PUT does not update the in-memory map.
    pub(super) async fn put_many(
        &self,
        namespace: &str,
        records: &[EmbeddingRecord],
        concurrency: usize,
    ) -> Result<(), DripstoneError> {
        if records.is_empty() {
            return Ok(());
        }
        let selected = last_record_per_id(records);
        let concurrency = concurrency.max(1).min(selected.len());
        let store = Arc::clone(&self.store);
        let bucket = self.bucket.clone();
        let key_prefix = self.key_prefix.clone();
        let namespace_owned = namespace.to_string();
        let mut stream = stream::iter(selected)
            .map(|record| {
                let store = Arc::clone(&store);
                let bucket = bucket.clone();
                let key_prefix = key_prefix.clone();
                let namespace = namespace_owned.clone();
                async move { write_body_object(store, bucket, key_prefix, namespace, record).await }
            })
            .buffer_unordered(concurrency);
        while let Some(result) = stream.next().await {
            result?;
        }
        self.remember_all(namespace, records).await;
        Ok(())
    }

    /// Install an envelope in the in-process map. Does not write object storage.
    ///
    /// WAL replay already read the record from the segment. The durable body was
    /// stored by upsert before that segment was appended.
    pub(super) async fn remember(&self, namespace: &str, record: &EmbeddingRecord) {
        self.remember_record(namespace, record).await;
    }

    /// Drop cached envelopes. Durable objects stay where upsert left them.
    pub(super) async fn forget_cached(&self) {
        self.records.write().await.clear();
    }

    /// Load many envelopes, using bounded parallel GETs for cache misses.
    pub(super) async fn get_many(
        &self,
        namespace: &str,
        ids: &[String],
    ) -> Result<Vec<EmbeddingRecord>, DripstoneError> {
        if ids.is_empty() {
            return Ok(Vec::new());
        }

        let (cached, missing) = self.split_cached(namespace, ids).await;
        let fetched = self.fetch_missing(namespace, &missing).await?;
        self.remember_all(namespace, &fetched).await;

        let mut records = cached;
        records.extend(fetched);
        Ok(records)
    }

    async fn remember_record(&self, namespace: &str, record: &EmbeddingRecord) {
        let mut records = self.records.write().await;
        records
            .entry(namespace.to_string())
            .or_default()
            .insert(record.id.to_string(), record.clone());
    }

    async fn remember_all(&self, namespace: &str, records: &[EmbeddingRecord]) {
        if records.is_empty() {
            return;
        }
        let mut map = self.records.write().await;
        let namespace_records = map.entry(namespace.to_string()).or_default();
        for record in records {
            namespace_records.insert(record.id.to_string(), record.clone());
        }
    }

    async fn split_cached(
        &self,
        namespace: &str,
        ids: &[String],
    ) -> (Vec<EmbeddingRecord>, Vec<String>) {
        let records = self.records.read().await;
        let namespace_records = records.get(namespace);
        let mut cached = Vec::new();
        let mut missing = Vec::new();
        for id in ids {
            match namespace_records.and_then(|entries| entries.get(id)) {
                Some(record) => cached.push(record.clone()),
                None => missing.push(id.clone()),
            }
        }
        (cached, missing)
    }

    async fn fetch_missing(
        &self,
        namespace: &str,
        ids: &[String],
    ) -> Result<Vec<EmbeddingRecord>, DripstoneError> {
        if ids.is_empty() {
            return Ok(Vec::new());
        }
        let concurrency = BODY_HYDRATE_PARALLELISM.min(ids.len()).max(1);
        let store = Arc::clone(&self.store);
        let bucket = self.bucket.clone();
        let key_prefix = self.key_prefix.clone();
        let namespace = namespace.to_string();
        let mut stream = stream::iter(ids.iter().cloned())
            .map(|chunk_id| {
                let store = Arc::clone(&store);
                let bucket = bucket.clone();
                let key_prefix = key_prefix.clone();
                let namespace = namespace.clone();
                async move { read_body_object(store, bucket, key_prefix, namespace, chunk_id).await }
            })
            .buffer_unordered(concurrency);

        let mut fetched = Vec::with_capacity(ids.len());
        while let Some(record) = stream.next().await {
            fetched.push(record?);
        }
        Ok(fetched)
    }

    pub(super) async fn get(
        &self,
        namespace: &str,
        chunk_id: &str,
    ) -> Result<Option<EmbeddingRecord>, DripstoneError> {
        {
            let records = self.records.read().await;
            if let Some(record) = records.get(namespace).and_then(|ns| ns.get(chunk_id)) {
                return Ok(Some(record.clone()));
            }
        }

        let key = self.key(namespace, chunk_id);
        match self.store.get_object(&self.bucket, &key).await {
            Ok(bytes) => {
                let record = decode_body(&key, &bytes)?;
                self.remember_record(namespace, &record).await;
                Ok(Some(record))
            }
            Err(err) if err.is_not_found() => Ok(None),
            Err(err) => Err(DripstoneError::storage(format!(
                "failed to read chunk body {key}: {err}"
            ))),
        }
    }

    pub(super) async fn remove(
        &self,
        namespace: &str,
        chunk_id: &str,
    ) -> Result<(), DripstoneError> {
        {
            let mut records = self.records.write().await;
            if let Some(ns) = records.get_mut(namespace) {
                ns.remove(chunk_id);
                if ns.is_empty() {
                    records.remove(namespace);
                }
            }
        }
        let key = self.key(namespace, chunk_id);
        self.store
            .delete_object(&self.bucket, &key)
            .await
            .map_err(|err| {
                DripstoneError::storage(format!("failed to delete chunk body {key}: {err}"))
            })?;
        Ok(())
    }

    pub(super) async fn clear_namespace(&self, namespace: &str) -> Result<(), DripstoneError> {
        {
            let mut records = self.records.write().await;
            records.remove(namespace);
        }
        let prefix = format!("{}{BODY_KEY_SEGMENT}/{namespace}/", self.key_prefix);
        let listed = self
            .store
            .list_objects(&self.bucket, Some(&prefix))
            .await
            .map_err(|err| {
                DripstoneError::storage(format!(
                    "failed to list chunk bodies for {namespace}: {err}"
                ))
            })?;
        for object in listed {
            self.store
                .delete_object(&self.bucket, &object.key)
                .await
                .map_err(|err| {
                    DripstoneError::storage(format!(
                        "failed to delete chunk body {}: {err}",
                        object.key
                    ))
                })?;
        }
        Ok(())
    }
}

fn decode_body(key: &str, bytes: &[u8]) -> Result<EmbeddingRecord, DripstoneError> {
    serde_json::from_slice(bytes)
        .map_err(|err| DripstoneError::storage(format!("failed to decode chunk body {key}: {err}")))
}

fn last_record_per_id(records: &[EmbeddingRecord]) -> Vec<EmbeddingRecord> {
    let mut index_of = HashMap::with_capacity(records.len());
    for (index, record) in records.iter().enumerate() {
        index_of.insert(record.id.as_ref(), index);
    }
    let mut order: Vec<usize> = index_of.into_values().collect();
    order.sort_unstable();
    order
        .into_iter()
        .map(|index| records[index].clone())
        .collect()
}

async fn write_body_object(
    store: Arc<dyn ObjectStore>,
    bucket: String,
    key_prefix: String,
    namespace: String,
    record: EmbeddingRecord,
) -> Result<(), DripstoneError> {
    let chunk_id = record.id.to_string();
    let key = ChunkBodyStore::object_key(&key_prefix, &namespace, &chunk_id);
    let bytes = serde_json::to_vec(&record)?;
    store
        .put_object(&bucket, &key, Bytes::from(bytes), Some(BODY_CONTENT_TYPE))
        .await
        .map_err(|err| DripstoneError::storage(format!("failed to store chunk body {key}: {err}")))
}

async fn read_body_object(
    store: Arc<dyn ObjectStore>,
    bucket: String,
    key_prefix: String,
    namespace: String,
    chunk_id: String,
) -> Result<EmbeddingRecord, DripstoneError> {
    let key = ChunkBodyStore::object_key(&key_prefix, &namespace, &chunk_id);
    match store.get_object(&bucket, &key).await {
        Ok(bytes) => decode_body(&key, &bytes),
        Err(err) if err.is_not_found() => Err(DripstoneError::storage(format!(
            "ColdForge membership for {chunk_id} has no chunk body in {namespace}"
        ))),
        Err(err) => Err(DripstoneError::storage(format!(
            "failed to read chunk body {key}: {err}"
        ))),
    }
}
