//! Chunk hash storage for incremental indexing.
//!
//! When hashline incremental indexing is enabled, a `ChunkHashStore` records
//! content hashes per chunk. On re-ingest, unchanged chunks are skipped
//! (no re-embedding, no upsert) to save compute and storage.

use crate::errors::DripstoneError;
use async_trait::async_trait;
use parking_lot::RwLock;
use std::collections::HashMap;

/// Key for document+namespace in the hash store
fn doc_key(doc_id: &str, namespace: Option<&str>) -> String {
    namespace.map_or_else(|| doc_id.to_string(), |ns| format!("{doc_id}:{ns}"))
}

/// Optional store for chunk content hashes.
///
/// Used when `HashlineConfig::incremental_indexing` is enabled. Before
/// embedding, existing hashes are loaded. Chunks whose content hash matches
/// are skipped. After upsert, new hashes are saved.
#[async_trait]
pub trait ChunkHashStore: Send + Sync {
    /// Load chunk hashes for a document.
    ///
    /// Returns `Ok(HashMap::new())` if the document has no stored hashes.
    async fn get_doc_chunk_hashes(
        &self,
        doc_id: &str,
        namespace: Option<&str>,
    ) -> Result<HashMap<usize, String>, DripstoneError>;

    /// Save chunk hashes for a document.
    ///
    /// Replaces any existing hashes for this `doc_id+namespace`.
    async fn set_doc_chunk_hashes(
        &self,
        doc_id: &str,
        namespace: Option<&str>,
        hashes: HashMap<usize, String>,
    ) -> Result<(), DripstoneError>;

    /// Clear hashes for a document (e.g. after delete).
    async fn delete_doc(
        &self,
        doc_id: &str,
        namespace: Option<&str>,
    ) -> Result<(), DripstoneError> {
        self.set_doc_chunk_hashes(doc_id, namespace, HashMap::new())
            .await
    }
}

/// In-memory chunk hash store.
///
/// Suitable for single-process use. Data is lost when the process ends.
#[derive(Debug, Default)]
pub struct MemoryChunkHashStore {
    inner: RwLock<HashMap<String, HashMap<usize, String>>>,
}

impl MemoryChunkHashStore {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Drop every stored document hash. Used when the backing vector store is cleared.
    pub fn clear(&self) {
        self.inner.write().clear();
    }
}

#[async_trait]
impl ChunkHashStore for MemoryChunkHashStore {
    async fn get_doc_chunk_hashes(
        &self,
        doc_id: &str,
        namespace: Option<&str>,
    ) -> Result<HashMap<usize, String>, DripstoneError> {
        let key = doc_key(doc_id, namespace);
        let guard = self.inner.read();
        Ok(guard.get(&key).cloned().unwrap_or_default())
    }

    async fn set_doc_chunk_hashes(
        &self,
        doc_id: &str,
        namespace: Option<&str>,
        hashes: HashMap<usize, String>,
    ) -> Result<(), DripstoneError> {
        let key = doc_key(doc_id, namespace);
        let mut guard = self.inner.write();
        if hashes.is_empty() {
            guard.remove(&key);
        } else {
            guard.insert(key, hashes);
        }
        drop(guard);
        Ok(())
    }
}
