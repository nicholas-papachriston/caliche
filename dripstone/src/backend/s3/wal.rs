//! Write-Ahead Log (WAL) for S3-backed vector storage
//!
//! Logs vector operations as batches of full [`EmbeddingRecord`](crate::embedding::EmbeddingRecord)
//! values so ColdForge can replay them into the in-memory SPFresh index on load.
//! Object keys are `{prefix}{namespace}/{unix_ms}_{writer_id}_{sequence}.wal`
//! (time-ordered segments). The writer id and sequence keep two flushes in the
//! same millisecond from overwriting each other.
//!
//! [`WalCompactor`] deletes a prefix of `.wal` segments only when an index
//! snapshot's membership is a fixed point of the log and every live envelope
//! whose latest upsert is in that prefix matches a chunk-id body object.
//! The snapshot stores centroids and membership, not bodies. Scoring vectors
//! for compacted ids are loaded from those body objects. A missing body, a
//! mismatched body, or a snapshot that still needs the log deletes nothing.
//!
//! `append` buffers entries. They become durable only after `flush_batch` puts
//! an object successfully. A failed put restores the batch to the front of the
//! queue. Object keys include a writer id and sequence so two flushes in the
//! same millisecond do not overwrite each other.

use crate::embedding::EmbeddingRecord;
use crate::errors::DripstoneError;
use crate::index::spfresh::CentroidIndex;

use super::layout::{
    body_namespace_for_wal, chunk_body_object_key, index_snapshot_key, root_prefix_from_wal,
};
use bytes::Bytes;
use parking_lot::Mutex;
use std::collections::VecDeque;
use std::hash::{BuildHasher, Hasher};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;
use time::OffsetDateTime;
use tokio::sync::{Mutex as AsyncMutex, Notify};
use tokio::time::interval;

use crate::backend::shared::BackgroundTask;
use crate::storage::ObjectStore;

/// WAL entry operation type
#[derive(Debug, Clone)]
pub enum WalOperation {
    /// Insert or update a vector
    Upsert(EmbeddingRecord),
    /// Delete a vector
    Delete {
        id: String,
        namespace: Option<String>,
    },
    /// Clear all vectors in a namespace
    Clear { namespace: Option<String> },
}

/// A single WAL entry
#[derive(Debug, Clone)]
pub struct WalEntry {
    /// Entry timestamp
    pub timestamp: i64,
    /// Sequential entry number
    pub sequence: u64,
    /// Operation to perform
    pub operation: WalOperation,
}

impl WalEntry {
    /// Create a new WAL entry
    #[must_use]
    pub fn new(sequence: u64, operation: WalOperation) -> Self {
        Self {
            timestamp: OffsetDateTime::now_utc().unix_timestamp(),
            sequence,
            operation,
        }
    }

    /// Serialize entry to bytes using `serde_json`
    ///
    /// # Errors
    ///
    /// Returns an error when the operation cannot be completed.
    pub fn to_bytes(&self) -> Result<Vec<u8>, DripstoneError> {
        serde_json::to_vec(self)
            .map_err(|e| DripstoneError::storage(format!("Failed to serialize WAL entry: {e}")))
    }

    /// Deserialize entry from bytes using `serde_json`
    ///
    /// # Errors
    ///
    /// Returns an error when the operation cannot be completed.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, DripstoneError> {
        serde_json::from_slice(bytes)
            .map_err(|e| DripstoneError::storage(format!("Failed to deserialize WAL entry: {e}")))
    }
}

impl serde::Serialize for WalEntry {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut state = serializer.serialize_struct("WalEntry", 3)?;
        state.serialize_field("timestamp", &self.timestamp)?;
        state.serialize_field("sequence", &self.sequence)?;
        state.serialize_field("operation", &self.operation)?;
        state.end()
    }
}

impl<'de> serde::Deserialize<'de> for WalEntry {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        use serde::de::{self, MapAccess, Visitor};
        use std::fmt;

        #[derive(serde::Deserialize)]
        #[serde(field_identifier, rename_all = "lowercase")]
        enum Field {
            Timestamp,
            Sequence,
            Operation,
        }

        struct WalEntryVisitor;

        impl<'de> Visitor<'de> for WalEntryVisitor {
            type Value = WalEntry;

            fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                formatter.write_str("struct WalEntry")
            }

            fn visit_map<V>(self, mut map: V) -> Result<WalEntry, V::Error>
            where
                V: MapAccess<'de>,
            {
                let mut timestamp = None;
                let mut sequence = None;
                let mut operation = None;
                while let Some(key) = map.next_key()? {
                    match key {
                        Field::Timestamp => {
                            if timestamp.is_some() {
                                return Err(de::Error::duplicate_field("timestamp"));
                            }
                            timestamp = Some(map.next_value()?);
                        }
                        Field::Sequence => {
                            if sequence.is_some() {
                                return Err(de::Error::duplicate_field("sequence"));
                            }
                            sequence = Some(map.next_value()?);
                        }
                        Field::Operation => {
                            if operation.is_some() {
                                return Err(de::Error::duplicate_field("operation"));
                            }
                            operation = Some(map.next_value()?);
                        }
                    }
                }
                let timestamp = timestamp.ok_or_else(|| de::Error::missing_field("timestamp"))?;
                let sequence = sequence.ok_or_else(|| de::Error::missing_field("sequence"))?;
                let operation = operation.ok_or_else(|| de::Error::missing_field("operation"))?;
                Ok(WalEntry {
                    timestamp,
                    sequence,
                    operation,
                })
            }
        }

        const FIELDS: &[&str] = &["timestamp", "sequence", "operation"];
        deserializer.deserialize_struct("WalEntry", FIELDS, WalEntryVisitor)
    }
}

impl serde::Serialize for WalOperation {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeTupleVariant;
        match self {
            Self::Upsert(record) => {
                let mut tv = serializer.serialize_tuple_variant("WalOperation", 0, "Upsert", 1)?;
                tv.serialize_field(record)?;
                tv.end()
            }
            Self::Delete { id, namespace } => {
                let mut tv = serializer.serialize_tuple_variant("WalOperation", 1, "Delete", 2)?;
                tv.serialize_field(id)?;
                tv.serialize_field(namespace)?;
                tv.end()
            }
            Self::Clear { namespace } => {
                let mut tv = serializer.serialize_tuple_variant("WalOperation", 2, "Clear", 1)?;
                tv.serialize_field(namespace)?;
                tv.end()
            }
        }
    }
}

impl<'de> serde::Deserialize<'de> for WalOperation {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        use serde::de::{self, Visitor};
        use std::fmt;

        struct WalOperationVisitor;

        impl<'de> Visitor<'de> for WalOperationVisitor {
            type Value = WalOperation;

            fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                formatter.write_str("enum WalOperation")
            }

            fn visit_map<V>(self, mut map: V) -> Result<WalOperation, V::Error>
            where
                V: de::MapAccess<'de>,
            {
                let mut variant = None;
                while let Some(key) = map.next_key::<String>()? {
                    match key.as_str() {
                        "Upsert" => {
                            // Tuple variant serializes as {"Upsert": [record]}
                            // Deserialize as Vec<EmbeddingRecord> and take first element
                            let vec: Vec<EmbeddingRecord> = map.next_value()?;
                            let record = vec
                                .into_iter()
                                .next()
                                .ok_or_else(|| de::Error::invalid_length(0, &"1 element"))?;
                            variant = Some(WalOperation::Upsert(record));
                            break;
                        }
                        "Delete" => {
                            // Tuple variant serializes as {"Delete": [id, namespace]}
                            // Deserialize as Vec<serde_json::Value> and extract fields
                            let vec: Vec<serde_json::Value> = map.next_value()?;
                            if vec.is_empty() {
                                return Err(de::Error::invalid_length(
                                    vec.len(),
                                    &"at least 1 element",
                                ));
                            }
                            let id = serde_json::from_value(vec[0].clone())
                                .map_err(|_| de::Error::custom("invalid id type"))?;
                            let namespace = if vec.len() > 1 {
                                serde_json::from_value(vec[1].clone()).ok()
                            } else {
                                None
                            };
                            variant = Some(WalOperation::Delete { id, namespace });
                            break;
                        }
                        "Clear" => {
                            // Tuple variant serializes as {"Clear": [namespace]}
                            // namespace is Option<String>, which serializes as null or "string"
                            let vec: Vec<Option<String>> = map.next_value()?;
                            let namespace = vec
                                .into_iter()
                                .next()
                                .ok_or_else(|| de::Error::invalid_length(0, &"1 element"))?;
                            variant = Some(WalOperation::Clear { namespace });
                            break;
                        }
                        _ => {
                            let _ = map.next_value::<de::IgnoredAny>()?;
                        }
                    }
                }
                variant.ok_or_else(|| de::Error::invalid_type(de::Unexpected::Map, &self))
            }
        }

        deserializer.deserialize_map(WalOperationVisitor)
    }
}

/// Configuration for WAL writer
#[derive(Debug, Clone)]
pub struct WalConfig {
    /// S3 bucket name
    pub bucket: String,
    /// S3 key prefix for WAL entries
    pub key_prefix: String,
    /// Batch interval in milliseconds (default: 1000ms).
    ///
    /// This is the delay before the background task attempts a flush. It is not
    /// a durability guarantee: `append` returns while the batch is still buffered,
    /// and a crash before a successful `flush_batch` drops those entries.
    /// `0` disables the background task; callers must `flush_batch` themselves.
    pub batch_interval_ms: u64,
    /// Maximum batch size before forcing a write
    pub max_batch_size: usize,
    /// Enable compression for WAL entries
    pub enable_compression: bool,
    /// Spawn the interval flush task. Tests that inject put failures turn this off
    /// so a retry cannot hide a failed flush.
    pub background_flush: bool,
}

impl Default for WalConfig {
    fn default() -> Self {
        Self {
            bucket: String::new(),
            key_prefix: "wal/".to_string(),
            batch_interval_ms: 1000,
            max_batch_size: 100,
            enable_compression: true,
            background_flush: true,
        }
    }
}

/// WAL writer for batching and writing entries to object storage.
///
/// This type is Send because all its fields are Send:
/// - `Arc<dyn ObjectStore>` is Send
/// - `Arc<Mutex<T>>` is Send if T is Send
/// - `parking_lot::Mutex` itself is Send (only guards aren't, but we don't hold them across await)
pub struct WalWriter {
    /// Object store
    store: Arc<dyn ObjectStore>,
    /// Configuration
    config: WalConfig,
    /// Pending entries waiting to be written
    pending_batch: Arc<Mutex<VecDeque<WalEntry>>>,
    /// Next sequence number
    next_sequence: Arc<Mutex<u64>>,
    /// Namespace for this WAL writer
    namespace: Option<String>,
    /// Distinguishes this process's segments from another writer's same-millisecond flush.
    writer_id: u64,
    /// One flush at a time so a failed put cannot be reordered behind a later success.
    flush_gate: Arc<AsyncMutex<()>>,
    /// Wakes the background task so drop does not wait out the full interval.
    shutdown: Arc<Notify>,
    running: Arc<AtomicBool>,
}

impl WalWriter {
    /// Create a new WAL writer
    pub fn new(store: Arc<dyn ObjectStore>, config: WalConfig, namespace: Option<String>) -> Self {
        let writer = Self {
            store,
            config,
            pending_batch: Arc::new(Mutex::new(VecDeque::new())),
            next_sequence: Arc::new(Mutex::new(0)),
            namespace,
            writer_id: fresh_writer_id(),
            flush_gate: Arc::new(AsyncMutex::new(())),
            shutdown: Arc::new(Notify::new()),
            running: Arc::new(AtomicBool::new(true)),
        };

        writer.start_background_flush();

        writer
    }

    /// Append an operation to the WAL.
    ///
    /// The entry is buffered. It is durable only after [`Self::flush_batch`]
    /// puts the segment successfully.
    ///
    /// # Errors
    ///
    /// Returns an error when a size-triggered flush cannot persist the batch.
    /// The batch stays queued for a later flush.
    pub async fn append(&self, operation: WalOperation) -> Result<u64, DripstoneError> {
        let sequence = {
            let mut seq = self.next_sequence.lock();
            let current = *seq;
            *seq += 1;
            current
        };

        let entry = WalEntry::new(sequence, operation);

        let should_flush = {
            let mut batch = self.pending_batch.lock();
            batch.push_back(entry);
            batch.len() >= self.config.max_batch_size
        };

        if should_flush {
            self.flush_batch().await?;
        }

        Ok(sequence)
    }

    /// Flush pending entries to object storage.
    ///
    /// # Errors
    ///
    /// Returns an error when the put fails. The drained entries are restored
    /// ahead of anything appended while the put was in flight.
    pub async fn flush_batch(&self) -> Result<(), DripstoneError> {
        flush_pending(
            &self.store,
            &self.config,
            &self.pending_batch,
            self.namespace.as_deref(),
            self.writer_id,
            &self.flush_gate,
        )
        .await
    }

    /// Start background task to flush batches periodically.
    ///
    /// The first interval tick is consumed so a writer does not flush immediately
    /// on construction. `batch_interval_ms == 0` or `background_flush == false`
    /// leaves flushing to the caller.
    fn start_background_flush(&self) {
        if !self.config.background_flush || self.config.batch_interval_ms == 0 {
            return;
        }

        let pending_batch = Arc::clone(&self.pending_batch);
        let config = self.config.clone();
        let namespace = self.namespace.clone();
        let store = Arc::clone(&self.store);
        let writer_id = self.writer_id;
        let flush_gate = Arc::clone(&self.flush_gate);
        let shutdown = Arc::clone(&self.shutdown);
        let running = Arc::clone(&self.running);

        tokio::spawn(async move {
            let mut tick = interval(Duration::from_millis(config.batch_interval_ms));
            tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
            tick.tick().await;

            loop {
                tokio::select! {
                    _ = shutdown.notified() => break,
                    _ = tick.tick() => {}
                }
                if !running.load(Ordering::Acquire) {
                    break;
                }

                let has_entries = {
                    let batch = pending_batch.lock();
                    !batch.is_empty()
                };
                if !has_entries {
                    continue;
                }

                if let Err(e) = flush_pending(
                    &store,
                    &config,
                    &pending_batch,
                    namespace.as_deref(),
                    writer_id,
                    &flush_gate,
                )
                .await
                {
                    tracing::error!("Failed to flush WAL batch in background: {e}");
                }
            }
        });
    }

    /// Get the current sequence number
    #[must_use]
    pub fn current_sequence(&self) -> u64 {
        *self.next_sequence.lock()
    }
}

impl Drop for WalWriter {
    fn drop(&mut self) {
        self.running.store(false, Ordering::Release);
        self.shutdown.notify_one();
    }
}

fn fresh_writer_id() -> u64 {
    let mut hasher = std::collections::hash_map::RandomState::new().build_hasher();
    hasher.write_u64(0xC0FF_EE00_D15C_A11E);
    let mixed = hasher.finish();
    if mixed == 0 { 1 } else { mixed }
}

pub(crate) fn wal_object_key(
    prefix: &str,
    namespace: Option<&str>,
    timestamp_ms: i128,
    writer_id: u64,
    sequence: u64,
) -> String {
    let namespace_part = namespace.map_or(String::new(), |ns| format!("{ns}/"));
    format!("{prefix}{namespace_part}{timestamp_ms}_{writer_id:016x}_{sequence:020}.wal")
}

fn restore_entries(pending: &Mutex<VecDeque<WalEntry>>, entries: Vec<WalEntry>) {
    let mut batch = pending.lock();
    for entry in entries.into_iter().rev() {
        batch.push_front(entry);
    }
}

async fn flush_pending(
    store: &Arc<dyn ObjectStore>,
    config: &WalConfig,
    pending_batch: &Mutex<VecDeque<WalEntry>>,
    namespace: Option<&str>,
    writer_id: u64,
    flush_gate: &AsyncMutex<()>,
) -> Result<(), DripstoneError> {
    let _gate = flush_gate.lock().await;
    let entries: Vec<WalEntry> = {
        let mut batch = pending_batch.lock();
        if batch.is_empty() {
            return Ok(());
        }
        batch.drain(..).collect()
    };

    if let Err(err) = write_wal_entries(store, config, namespace, writer_id, &entries).await {
        restore_entries(pending_batch, entries);
        return Err(err);
    }
    Ok(())
}

async fn write_wal_entries(
    store: &Arc<dyn ObjectStore>,
    config: &WalConfig,
    namespace: Option<&str>,
    writer_id: u64,
    entries: &[WalEntry],
) -> Result<(), DripstoneError> {
    if entries.is_empty() {
        return Ok(());
    }

    let batch_data = WalBatch {
        entries: entries.to_vec(),
    };
    let bytes = serde_json::to_vec(&batch_data)
        .map_err(|e| DripstoneError::storage(format!("Failed to serialize WAL batch: {e}")))?;
    let final_bytes = if config.enable_compression {
        compress_bytes(&bytes)?
    } else {
        bytes
    };

    let timestamp_ms = OffsetDateTime::now_utc().unix_timestamp_nanos() / 1_000_000;
    let first_sequence = entries.first().map_or(0, |entry| entry.sequence);
    let key = wal_object_key(
        &config.key_prefix,
        namespace,
        timestamp_ms,
        writer_id,
        first_sequence,
    );

    store
        .put_object(
            &config.bucket,
            &key,
            Bytes::from(final_bytes),
            Some("application/octet-stream"),
        )
        .await
        .map_err(|e| {
            DripstoneError::storage(format!("Failed to write WAL batch to storage: {key}: {e}"))
        })?;

    tracing::debug!(
        "Flushed WAL batch with {} entries to {}",
        entries.len(),
        key
    );
    Ok(())
}

/// Batch of WAL entries
#[derive(Debug)]
struct WalBatch {
    entries: Vec<WalEntry>,
}

impl serde::Serialize for WalBatch {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut state = serializer.serialize_struct("WalBatch", 1)?;
        state.serialize_field("entries", &self.entries)?;
        state.end()
    }
}

impl<'de> serde::Deserialize<'de> for WalBatch {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        use serde::de::{self, MapAccess, Visitor};
        use std::fmt;

        struct WalBatchVisitor;

        impl<'de> Visitor<'de> for WalBatchVisitor {
            type Value = WalBatch;

            fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                formatter.write_str("struct WalBatch")
            }

            fn visit_map<V>(self, mut map: V) -> Result<WalBatch, V::Error>
            where
                V: MapAccess<'de>,
            {
                let mut entries = None;
                while let Some(key) = map.next_key::<String>()? {
                    if key == "entries" {
                        entries = Some(map.next_value()?);
                    } else {
                        let _ = map.next_value::<de::IgnoredAny>()?;
                    }
                }
                let entries = entries.ok_or_else(|| de::Error::missing_field("entries"))?;
                Ok(WalBatch { entries })
            }
        }

        const FIELDS: &[&str] = &["entries"];
        deserializer.deserialize_struct("WalBatch", FIELDS, WalBatchVisitor)
    }
}

/// WAL reader for replaying entries from object storage
pub struct WalReader {
    /// Object store
    store: Arc<dyn ObjectStore>,
    /// Configuration
    config: WalConfig,
    /// Namespace to read from
    namespace: Option<String>,
}

impl WalReader {
    /// Create a new WAL reader
    pub fn new(store: Arc<dyn ObjectStore>, config: WalConfig, namespace: Option<String>) -> Self {
        Self {
            store,
            config,
            namespace,
        }
    }

    /// List all WAL files in object storage for this namespace
    ///
    /// # Errors
    ///
    /// Returns an error when the operation cannot be completed.
    pub async fn list_wal_files(&self) -> Result<Vec<String>, DripstoneError> {
        let namespace_part = self
            .namespace
            .as_ref()
            .map(|ns| format!("{ns}/"))
            .unwrap_or_default();
        let prefix = format!("{}{}", self.config.key_prefix, namespace_part);

        let objects = self
            .store
            .list_objects(&self.config.bucket, Some(&prefix))
            .await
            .map_err(|e| DripstoneError::storage(format!("Failed to list WAL files: {e}")))?;

        let mut wal_files: Vec<String> = objects
            .into_iter()
            .filter(|obj| {
                std::path::Path::new(&obj.key)
                    .extension()
                    .is_some_and(|extension| extension.eq_ignore_ascii_case("wal"))
            })
            .map(|obj| obj.key)
            .collect();

        wal_files.sort();
        Ok(wal_files)
    }

    /// Read and deserialize a WAL batch from object storage
    ///
    /// # Errors
    ///
    /// Returns an error when the operation cannot be completed.
    pub async fn read_wal_batch(&self, key: &str) -> Result<Vec<WalEntry>, DripstoneError> {
        let bytes = self
            .store
            .get_object(&self.config.bucket, key)
            .await
            .map_err(|e| DripstoneError::storage(format!("Failed to read WAL file: {key}: {e}")))?;

        // Decompress if needed
        let final_bytes = if self.config.enable_compression {
            decompress_bytes(&bytes)?
        } else {
            bytes.to_vec()
        };

        let batch: WalBatch = serde_json::from_slice(&final_bytes).map_err(|e| {
            DripstoneError::storage(format!("Failed to deserialize WAL batch: {e}"))
        })?;

        Ok(batch.entries)
    }

    /// Replay all WAL entries
    ///
    /// # Errors
    ///
    /// Returns an error when the operation cannot be completed.
    pub async fn replay<F>(&self, mut callback: F) -> Result<u64, DripstoneError>
    where
        F: FnMut(WalEntry) -> Result<(), DripstoneError>,
    {
        let wal_files = self.list_wal_files().await?;
        let num_files = wal_files.len();
        let mut total_entries = 0u64;

        for wal_file in &wal_files {
            let entries = self.read_wal_batch(wal_file).await?;

            for entry in entries {
                callback(entry)?;
                total_entries += 1;
            }
        }

        tracing::info!(
            "Replayed {} WAL entries from {} files",
            total_entries,
            num_files
        );

        Ok(total_entries)
    }
}

/// Deletes old WAL segment files. Does not fold entries into the SPFresh snapshot.
pub struct WalCompactor {
    /// Object store
    store: Arc<dyn ObjectStore>,
    /// Configuration
    config: WalConfig,
    /// Namespace
    namespace: Option<String>,
}

impl WalCompactor {
    /// Create a new WAL compactor
    pub fn new(store: Arc<dyn ObjectStore>, config: WalConfig, namespace: Option<String>) -> Self {
        Self {
            store,
            config,
            namespace,
        }
    }

    /// Delete the oldest WAL segments when bodies and the membership snapshot
    /// already cover them.
    ///
    /// Returns `Ok(0)` when `keep_latest_n` already retains every segment.
    /// When older segments would be deleted, this reads them and the snapshot
    /// first. It deletes nothing if the snapshot is missing, membership is not
    /// a fixed point of the log, or a live envelope in the dropped prefix is
    /// missing or does not match its body object.
    ///
    /// # Errors
    ///
    /// Returns an error when a segment cannot be dropped safely, or when
    /// listing, reading, or deleting an object fails. A safety refusal deletes
    /// nothing.
    pub async fn compact(&self, keep_latest_n: usize) -> Result<usize, DripstoneError> {
        let reader = WalReader::new(
            Arc::clone(&self.store),
            self.config.clone(),
            self.namespace.clone(),
        );
        let wal_files = reader.list_wal_files().await?;
        if wal_files.len() <= keep_latest_n {
            return Ok(0);
        }
        let drop_count = wal_files.len() - keep_latest_n;
        let segments = self.read_segments(&reader, &wal_files).await?;
        let drop_keys = self.keys_safe_to_drop(&segments, drop_count).await?;
        self.delete_segments(&drop_keys).await
    }

    async fn read_segments(
        &self,
        reader: &WalReader,
        keys: &[String],
    ) -> Result<Vec<LoadedWalSegment>, DripstoneError> {
        let mut segments = Vec::with_capacity(keys.len());
        for key in keys {
            let entries = reader.read_wal_batch(key).await?;
            segments.push(LoadedWalSegment {
                key: key.clone(),
                entries,
            });
        }
        Ok(segments)
    }

    async fn keys_safe_to_drop(
        &self,
        segments: &[LoadedWalSegment],
        drop_count: usize,
    ) -> Result<Vec<String>, DripstoneError> {
        let root = root_prefix_from_wal(&self.config.key_prefix).ok_or_else(|| {
            refuse_compaction(
                "WAL prefix is not a ColdForge wal/ layout, so body and snapshot keys cannot be checked",
            )
        })?;
        let wal_namespace = self
            .namespace
            .as_deref()
            .unwrap_or(super::layout::WAL_NAMESPACE_DEFAULT);
        let snapshot_key = index_snapshot_key(root, wal_namespace);
        let membership = match self
            .store
            .get_object(&self.config.bucket, &snapshot_key)
            .await
        {
            Ok(bytes) => CentroidIndex::snapshot_membership(&bytes).map_err(|err| {
                refuse_compaction(&format!(
                    "index snapshot {snapshot_key} is not a membership checkpoint: {err}"
                ))
            })?,
            Err(err) if err.is_not_found() => {
                return Err(refuse_compaction(
                    "index snapshot is missing; membership still needs the log",
                ));
            }
            Err(err) => return Err(err),
        };

        let folded = fold_wal_onto_membership(&membership, segments);
        if folded != membership {
            return Err(refuse_compaction(
                "index snapshot membership is not a fixed point of the log; membership still needs the log",
            ));
        }

        let body_namespace = body_namespace_for_wal(wal_namespace);
        let latest = latest_upserts(segments);
        for (chunk_id, upsert) in &latest {
            if upsert.segment_index >= drop_count || !membership.contains(chunk_id) {
                continue;
            }
            let key = chunk_body_object_key(root, body_namespace, chunk_id);
            self.require_matching_body(&key, &upsert.record).await?;
        }

        Ok(segments
            .iter()
            .take(drop_count)
            .map(|segment| segment.key.clone())
            .collect())
    }

    async fn require_matching_body(
        &self,
        key: &str,
        expected: &EmbeddingRecord,
    ) -> Result<(), DripstoneError> {
        match self.store.get_object(&self.config.bucket, key).await {
            Ok(bytes) => {
                let stored: EmbeddingRecord = serde_json::from_slice(&bytes).map_err(|err| {
                    refuse_compaction(&format!(
                        "chunk body {key} is not a durable envelope ({err})"
                    ))
                })?;
                if stored.id.as_ref() != expected.id.as_ref()
                    || stored.vector.as_ref() != expected.vector.as_ref()
                {
                    return Err(refuse_compaction(&format!(
                        "chunk body {key} does not match the WAL envelope"
                    )));
                }
                Ok(())
            }
            Err(err) if err.is_not_found() => Err(refuse_compaction(&format!(
                "chunk body {key} is missing; a body put failed or the envelope is not durable"
            ))),
            Err(err) => Err(err),
        }
    }

    async fn delete_segments(&self, keys: &[String]) -> Result<usize, DripstoneError> {
        for key in keys {
            self.store
                .delete_object(&self.config.bucket, key)
                .await
                .map_err(|err| {
                    DripstoneError::storage(format!("Failed to delete WAL segment {key}: {err}"))
                })?;
        }
        Ok(keys.len())
    }

    /// Start background compaction every `interval_secs`.
    ///
    /// The task holds its own clone of the store and runs until the returned
    /// guard drops.
    pub fn start_background_compaction(
        &self,
        interval_secs: u64,
        keep_latest_n: usize,
    ) -> BackgroundTask {
        let config = self.config.clone();
        let namespace = self.namespace.clone();

        let store = Arc::clone(&self.store);
        BackgroundTask::spawn(async move {
            let mut tick = interval(Duration::from_secs(interval_secs));

            loop {
                tick.tick().await;

                let compactor = Self::new(Arc::clone(&store), config.clone(), namespace.clone());

                match compactor.compact(keep_latest_n).await {
                    Ok(deleted) => {
                        if deleted > 0 {
                            tracing::info!("Background compaction: deleted {} files", deleted);
                        }
                    }
                    Err(e) => {
                        tracing::error!("Background compaction failed: {}", e);
                    }
                }
            }
        })
    }
}

struct LoadedWalSegment {
    key: String,
    entries: Vec<WalEntry>,
}

struct LatestUpsert {
    segment_index: usize,
    record: EmbeddingRecord,
}

fn refuse_compaction(reason: &str) -> DripstoneError {
    DripstoneError::storage(format!(
        "WalCompactor::compact refuses to delete WAL segments: {reason}"
    ))
}

fn fold_wal_onto_membership(
    snapshot: &std::collections::BTreeSet<String>,
    segments: &[LoadedWalSegment],
) -> std::collections::BTreeSet<String> {
    let mut live = snapshot.clone();
    for segment in segments {
        for entry in &segment.entries {
            match &entry.operation {
                WalOperation::Upsert(record) => {
                    live.insert(record.id.to_string());
                }
                WalOperation::Delete { id, .. } => {
                    live.remove(id);
                }
                WalOperation::Clear { .. } => {
                    live.clear();
                }
            }
        }
    }
    live
}

fn latest_upserts(
    segments: &[LoadedWalSegment],
) -> std::collections::HashMap<String, LatestUpsert> {
    let mut latest = std::collections::HashMap::new();
    for (segment_index, segment) in segments.iter().enumerate() {
        for entry in &segment.entries {
            if let WalOperation::Upsert(record) = &entry.operation {
                latest.insert(
                    record.id.to_string(),
                    LatestUpsert {
                        segment_index,
                        record: record.clone(),
                    },
                );
            }
        }
    }
    latest
}

/// Compress bytes using flate2
fn compress_bytes(data: &[u8]) -> Result<Vec<u8>, DripstoneError> {
    use flate2::Compression;
    use flate2::write::GzEncoder;
    use std::io::Write;

    let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
    encoder
        .write_all(data)
        .map_err(|e| DripstoneError::storage(format!("Compression failed: {e}")))?;
    encoder
        .finish()
        .map_err(|e| DripstoneError::storage(format!("Compression failed: {e}")))
}

/// Decompress bytes using flate2
fn decompress_bytes(data: &[u8]) -> Result<Vec<u8>, DripstoneError> {
    use flate2::read::GzDecoder;
    use std::io::Read;

    let mut decoder = GzDecoder::new(data);
    let mut decompressed = Vec::new();
    decoder
        .read_to_end(&mut decompressed)
        .map_err(|e| DripstoneError::storage(format!("Decompression failed: {e}")))?;
    Ok(decompressed)
}

#[cfg(test)]
mod background_compaction_tests {
    use super::*;
    use crate::storage::{MemoryObjectStore, ObjectInfo, ObjectMetadata};
    use async_trait::async_trait;
    use std::sync::atomic::AtomicUsize;

    const INTERVAL_SECS: u64 = 60;

    /// A memory store that counts compaction passes: each pass lists the WAL.
    #[derive(Default)]
    struct CountingStore {
        inner: MemoryObjectStore,
        lists: AtomicUsize,
    }

    #[async_trait]
    impl ObjectStore for CountingStore {
        async fn get_object(&self, bucket: &str, key: &str) -> Result<Bytes, DripstoneError> {
            self.inner.get_object(bucket, key).await
        }
        async fn put_object(
            &self,
            bucket: &str,
            key: &str,
            data: Bytes,
            content_type: Option<&str>,
        ) -> Result<(), DripstoneError> {
            self.inner.put_object(bucket, key, data, content_type).await
        }
        async fn delete_object(&self, bucket: &str, key: &str) -> Result<(), DripstoneError> {
            self.inner.delete_object(bucket, key).await
        }
        async fn head_object(
            &self,
            bucket: &str,
            key: &str,
        ) -> Result<ObjectMetadata, DripstoneError> {
            self.inner.head_object(bucket, key).await
        }
        async fn list_objects(
            &self,
            bucket: &str,
            prefix: Option<&str>,
        ) -> Result<Vec<ObjectInfo>, DripstoneError> {
            self.lists.fetch_add(1, Ordering::SeqCst);
            self.inner.list_objects(bucket, prefix).await
        }
    }

    fn compactor(store: &Arc<CountingStore>) -> WalCompactor {
        let store: Arc<dyn ObjectStore> = store.clone();
        WalCompactor::new(store, WalConfig::default(), None)
    }

    #[tokio::test(start_paused = true)]
    async fn a_held_guard_compacts_once_per_interval() {
        let store = Arc::new(CountingStore::default());
        let task = compactor(&store).start_background_compaction(INTERVAL_SECS, 1);

        tokio::time::sleep(Duration::from_secs(3 * INTERVAL_SECS + 1)).await;

        // The first tick fires at once, then one per interval.
        assert_eq!(store.lists.load(Ordering::SeqCst), 4);
        drop(task);
    }

    #[tokio::test(start_paused = true)]
    async fn a_dropped_guard_compacts_nothing_after_three_intervals() {
        let store = Arc::new(CountingStore::default());
        let task = compactor(&store).start_background_compaction(INTERVAL_SECS, 1);

        drop(task);
        tokio::time::sleep(Duration::from_secs(3 * INTERVAL_SECS + 1)).await;

        assert_eq!(store.lists.load(Ordering::SeqCst), 0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wal_entry_serialization() {
        let record = EmbeddingRecord::new("test_id", vec![1.0, 2.0, 3.0]);
        let operation = WalOperation::Upsert(record);
        let entry = WalEntry::new(42, operation);

        let bytes = entry.to_bytes().expect("Failed to serialize WalEntry");
        let restored = WalEntry::from_bytes(&bytes).expect("Failed to deserialize WalEntry");

        assert_eq!(entry.sequence, restored.sequence);
    }

    #[test]
    fn test_compression() {
        let data = vec![0u8; 1000]; // Highly compressible
        let compressed = compress_bytes(&data).expect("Failed to compress bytes");
        let decompressed = decompress_bytes(&compressed).expect("Failed to decompress bytes");

        assert_eq!(data, decompressed);
        assert!(compressed.len() < data.len());
    }

    #[test]
    fn wal_keys_in_the_same_millisecond_stay_distinct_and_sort_before_the_next() {
        let left = super::wal_object_key("wal/", Some("ns"), 1_700_000_000_000, 1, 4);
        let right = super::wal_object_key("wal/", Some("ns"), 1_700_000_000_000, 2, 4);
        let next = super::wal_object_key("wal/", Some("ns"), 1_700_000_000_001, 1, 0);
        assert_ne!(left, right);
        assert!(left < next);
        assert!(right < next);
        assert!(left.ends_with(".wal"));
        assert!(right.ends_with(".wal"));
    }
}
