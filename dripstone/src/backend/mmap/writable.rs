//! Writable in-memory store for [`super::MmapBackend`] with periodic mmap snapshots.
//!
//! Authoritative data lives in a hash map keyed by composite `(namespace,id)` keys.
//! On flush, vectors are sorted and written via the selected mmap writer.

use std::collections::HashMap;
use std::path::Path;

use crate::backend::mmap::codec::write_dripstone_mmap_with_encoding;
use crate::backend::mmap::format::VectorEncoding;
use crate::backend::shared::build_composite_key;
use crate::core::{Chunk, SearchQuery, SearchResult};
use crate::errors::DripstoneError;

/// In-memory authoritative store for read-write mmap backends.
pub struct WritableStore {
    chunks: HashMap<String, Chunk>,
    ops_since_flush: usize,
    flush_every: usize,
    file_path: String,
    dimension: usize,
    vector_encoding: VectorEncoding,
}

impl WritableStore {
    /// Create an empty store. `flush_every` of `0` disables periodic flushing.
    pub fn new(
        file_path: String,
        dimension: usize,
        flush_every: usize,
        vector_encoding: VectorEncoding,
    ) -> Self {
        Self {
            chunks: HashMap::new(),
            ops_since_flush: 0,
            flush_every,
            file_path,
            dimension,
            vector_encoding,
        }
    }

    /// Add chunks read from an existing snapshot. They count as already flushed.
    pub fn load(&mut self, chunks: Vec<Chunk>) {
        for c in chunks {
            let k = build_composite_key(c.namespace.as_deref(), &c.id);
            self.chunks.insert(k, c);
        }
    }

    /// Last snapshot size on disk after flush, or `None` if never flushed.
    pub fn flush_snapshot(path: &Path) -> Result<u64, DripstoneError> {
        let meta = std::fs::metadata(path)
            .map_err(|e| DripstoneError::storage_with_source("mmap writable: stat snapshot", e))?;
        Ok(meta.len())
    }

    fn validate_dimension(&self, chunks: &[Chunk]) -> Result<(), DripstoneError> {
        for c in chunks {
            if c.dimension() != self.dimension {
                return Err(DripstoneError::storage(format!(
                    "mmap writable: expected dimension {}, got {}",
                    self.dimension,
                    c.dimension()
                )));
            }
        }
        Ok(())
    }

    /// Insert or replace chunks. Returns snapshot byte length if a flush ran.
    pub fn upsert(&mut self, new_chunks: &[Chunk]) -> Result<Option<u64>, DripstoneError> {
        if new_chunks.is_empty() {
            return Ok(None);
        }
        self.validate_dimension(new_chunks)?;
        for c in new_chunks {
            let k = build_composite_key(c.namespace.as_deref(), &c.id);
            self.chunks.insert(k, c.clone());
        }
        self.add_ops_and_maybe_flush(new_chunks.len())
    }

    fn add_ops_and_maybe_flush(&mut self, n: usize) -> Result<Option<u64>, DripstoneError> {
        self.ops_since_flush = self.ops_since_flush.saturating_add(n);
        if self.flush_every == 0 || self.ops_since_flush < self.flush_every {
            return Ok(None);
        }
        self.flush_to_disk()
    }

    /// Write all chunks to the configured mmap file and reset the flush counter.
    pub fn flush_to_disk(&mut self) -> Result<Option<u64>, DripstoneError> {
        if self.chunks.is_empty() {
            self.ops_since_flush = 0;
            return Ok(None);
        }
        let mut list: Vec<Chunk> = self.chunks.values().cloned().collect();
        list.sort_by(|a, b| {
            let ka = build_composite_key(a.namespace.as_deref(), &a.id);
            let kb = build_composite_key(b.namespace.as_deref(), &b.id);
            ka.cmp(&kb)
        });
        let path = Path::new(&self.file_path);
        if let Some(parent) = path.parent()
            && !parent.as_os_str().is_empty()
        {
            std::fs::create_dir_all(parent).map_err(|e| {
                DripstoneError::storage_with_source("mmap writable: create parent dir", e)
            })?;
        }
        write_dripstone_mmap_with_encoding(path, &list, self.vector_encoding)?;
        self.ops_since_flush = 0;
        let len = Self::flush_snapshot(path)?;
        Ok(Some(len))
    }

    pub fn delete(
        &mut self,
        id: &str,
        namespace: Option<&str>,
    ) -> Result<Option<u64>, DripstoneError> {
        let k = build_composite_key(namespace, id);
        let removed = self.chunks.remove(&k).is_some();
        if removed {
            self.add_ops_and_maybe_flush(1)
        } else {
            Ok(None)
        }
    }

    pub fn delete_document(
        &mut self,
        doc_id: &str,
        namespace: Option<&str>,
    ) -> Result<(usize, Option<u64>), DripstoneError> {
        let keys: Vec<String> = self
            .chunks
            .iter()
            .filter(|(_, c)| {
                if c.doc_id != doc_id {
                    return false;
                }
                match (namespace, c.namespace.as_deref()) {
                    (None | Some("default"), None) => true,
                    (Some(ns), Some(cn)) => ns == cn,
                    (None, Some(_)) | (Some(_), None) => false,
                }
            })
            .map(|(k, _)| k.clone())
            .collect();
        let n = keys.len();
        for k in keys {
            self.chunks.remove(&k);
        }
        let flushed = if n > 0 {
            self.add_ops_and_maybe_flush(n)?
        } else {
            None
        };
        Ok((n, flushed))
    }

    pub fn delete_namespace(&mut self, namespace: &str) -> Result<Option<u64>, DripstoneError> {
        let keys: Vec<String> = self
            .chunks
            .keys()
            .filter(|key| {
                let (ns, _) = crate::backend::shared::parse_composite_key(key);
                ns.map_or_else(|| namespace == "default", |ns_val| ns_val == namespace)
            })
            .cloned()
            .collect();
        if keys.is_empty() {
            return Ok(None);
        }
        let n = keys.len();
        for k in keys {
            self.chunks.remove(&k);
        }
        self.add_ops_and_maybe_flush(n)
    }

    pub fn get(&self, id: &str, namespace: Option<&str>) -> Option<Chunk> {
        let k = build_composite_key(namespace, id);
        self.chunks.get(&k).cloned()
    }

    pub fn count(&self, namespace: Option<&str>) -> usize {
        namespace.map_or(self.chunks.len(), |ns| {
            self.chunks
                .values()
                .filter(|c| c.namespace.as_deref().map_or(ns == "default", |n| n == ns))
                .count()
        })
    }

    pub fn list(&self, namespace: Option<&str>, limit: Option<usize>) -> Vec<Chunk> {
        let mut out: Vec<Chunk> = self
            .chunks
            .values()
            .filter(|c| {
                namespace
                    .is_none_or(|ns| c.namespace.as_deref().map_or(ns == "default", |n| n == ns))
            })
            .cloned()
            .collect();
        out.sort_by(|a, b| {
            let ka = build_composite_key(a.namespace.as_deref(), &a.id);
            let kb = build_composite_key(b.namespace.as_deref(), &b.id);
            ka.cmp(&kb)
        });
        if let Some(lim) = limit {
            out.truncate(lim);
        }
        out
    }

    pub fn namespaces(&self) -> Vec<String> {
        let mut out: Vec<String> = self
            .chunks
            .values()
            .filter_map(|c| c.namespace.clone())
            .collect();
        out.sort();
        out.dedup();
        out
    }

    /// Brute-force search over stored chunks (same semantics as read-only mmap search).
    pub fn search(&self, query: &SearchQuery) -> Result<Vec<SearchResult>, DripstoneError> {
        let q = query.query();
        let mut results: Vec<SearchResult> = Vec::new();
        for chunk in self.chunks.values() {
            if let Some(ref ns) = query.namespace {
                let ok = chunk.namespace.as_deref() == Some(ns.as_str());
                if !ok {
                    continue;
                }
            }
            if let Some(ref filter) = query.filter
                && !filter.matches(&chunk.metadata)
            {
                continue;
            }
            let slice = chunk.vector.as_ref();
            let score = query.metric.compute(q, slice)?;
            let mut c = chunk.clone();
            if !query.include_content {
                c.content.clear();
            }
            if !query.include_vectors {
                c.vector = std::sync::Arc::from(Vec::new());
            }
            results.push(SearchResult::new(c, score, 0));
        }
        if query.metric.higher_is_better() {
            results.sort_by(|a, b| {
                b.score
                    .partial_cmp(&a.score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
        } else {
            results.sort_by(|a, b| {
                a.score
                    .partial_cmp(&b.score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
        }
        results.truncate(query.limit);
        for (i, r) in results.iter_mut().enumerate() {
            r.rank = i;
        }
        Ok(results)
    }
}
