//! Shared namespace write-ahead log for single-WAL multi-shard backends.
//!
//! All writes for a logical namespace land in one ordered sequence. Shard index
//! consumers apply `1/N` of the log (`hash(id) % num_shards`) without a
//! two-phase commit across shards — the turbopuffer-style scale pattern.

use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use time::OffsetDateTime;

use crate::core::Chunk;
use crate::errors::DripstoneError;
use crate::storage::ObjectStore;

/// Operation recorded in the shared namespace WAL.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NamespaceWalOp {
    Upsert(Chunk),
    Delete {
        id: String,
        namespace: Option<String>,
    },
    Clear {
        namespace: Option<String>,
    },
}

/// One sequenced entry in the shared WAL.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NamespaceWalEntry {
    pub sequence: u64,
    pub timestamp: i64,
    pub operation: NamespaceWalOp,
}

impl NamespaceWalEntry {
    #[must_use]
    pub fn new(sequence: u64, operation: NamespaceWalOp) -> Self {
        Self {
            sequence,
            timestamp: OffsetDateTime::now_utc().unix_timestamp(),
            operation,
        }
    }
}

/// Configuration for durable WAL segments on an [`ObjectStore`].
#[derive(Debug, Clone)]
pub struct SharedWalConfig {
    pub bucket: String,
    pub key_prefix: String,
}

impl Default for SharedWalConfig {
    fn default() -> Self {
        Self {
            bucket: "dripstone".to_string(),
            key_prefix: "namespaces/default/wal/".to_string(),
        }
    }
}

/// Single ordered WAL for one logical namespace.
pub struct SharedNamespaceWal {
    next_sequence: AtomicU64,
    entries: RwLock<Vec<NamespaceWalEntry>>,
    store: Option<Arc<dyn ObjectStore>>,
    config: SharedWalConfig,
}

impl std::fmt::Debug for SharedNamespaceWal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SharedNamespaceWal")
            .field("len", &self.len())
            .field("durable", &self.store.is_some())
            .field("bucket", &self.config.bucket)
            .field("key_prefix", &self.config.key_prefix)
            .finish()
    }
}

impl SharedNamespaceWal {
    /// In-memory WAL only (still provides a total write order for shards).
    #[must_use]
    pub fn in_memory() -> Self {
        Self {
            next_sequence: AtomicU64::new(0),
            entries: RwLock::new(Vec::new()),
            store: None,
            config: SharedWalConfig::default(),
        }
    }

    /// WAL with durable segments under `config.key_prefix` on `store`.
    #[must_use]
    pub fn with_object_store(store: Arc<dyn ObjectStore>, config: SharedWalConfig) -> Self {
        Self {
            next_sequence: AtomicU64::new(0),
            entries: RwLock::new(Vec::new()),
            store: Some(store),
            config,
        }
    }

    /// Append an operation, assigning the next sequence number.
    ///
    /// # Errors
    ///
    /// Returns when durable object-store persistence fails.
    pub async fn append(&self, operation: NamespaceWalOp) -> Result<u64, DripstoneError> {
        let sequence = self.next_sequence.fetch_add(1, Ordering::SeqCst);
        let entry = NamespaceWalEntry::new(sequence, operation);
        {
            let mut guard = self.entries.write();
            guard.push(entry.clone());
        }
        if let Some(store) = &self.store {
            let bytes = serde_json::to_vec(&entry).map_err(|e| {
                DripstoneError::storage(format!("Failed to serialize WAL entry: {e}"))
            })?;
            let key = format!("{}{:020}.json", self.config.key_prefix, sequence);
            store
                .put_object(
                    &self.config.bucket,
                    &key,
                    bytes::Bytes::from(bytes),
                    Some("application/json"),
                )
                .await?;
        }
        Ok(sequence)
    }

    /// Snapshot of entries currently held in process memory (ordered by sequence).
    #[must_use]
    pub fn entries(&self) -> Vec<NamespaceWalEntry> {
        self.entries.read().clone()
    }

    /// Number of entries currently held in process memory.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.read().len()
    }

    /// Returns true when the in-memory log is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.read().is_empty()
    }

    /// Load durable segments from object storage into memory (replacing the buffer).
    ///
    /// # Errors
    ///
    /// Returns when listing or deserializing WAL objects fails.
    pub async fn load_from_store(&self) -> Result<usize, DripstoneError> {
        let Some(store) = &self.store else {
            return Ok(0);
        };
        let mut objects = store
            .list_objects(&self.config.bucket, Some(&self.config.key_prefix))
            .await?;
        objects.sort_by(|a, b| a.key.cmp(&b.key));
        let mut loaded = Vec::new();
        let mut max_seq = 0_u64;
        for obj in objects {
            let bytes = store.get_object(&self.config.bucket, &obj.key).await?;
            let entry: NamespaceWalEntry = serde_json::from_slice(&bytes).map_err(|e| {
                DripstoneError::storage(format!("Failed to deserialize WAL entry {}: {e}", obj.key))
            })?;
            max_seq = max_seq.max(entry.sequence.saturating_add(1));
            loaded.push(entry);
        }
        loaded.sort_by_key(|e| e.sequence);
        {
            let mut guard = self.entries.write();
            *guard = loaded;
        }
        self.next_sequence.store(max_seq, Ordering::SeqCst);
        Ok(self.len())
    }
}
