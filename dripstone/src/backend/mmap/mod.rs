//! Memory-mapped storage backend for high-performance vector serving.
//!
//! Read-only mode maps an existing `Dripstone` mmap file (aligned v1 or int8 v2 slab layout).
//!
//! Read-write mode keeps an authoritative in-memory map and periodically rewrites a v1 snapshot
//! with the configured vector encoding.

mod backend;
mod chunk;
mod codec;
mod format;
mod ops;
mod search;
mod writable;

use std::fs::File;
use std::sync::Arc;
use std::sync::atomic::AtomicU64;

use memmap2::Mmap;
use parking_lot::RwLock;

pub use codec::{
    write_dripstone_mmap, write_dripstone_mmap_v1, write_dripstone_mmap_v2,
    write_dripstone_mmap_with_encoding,
};
pub use format::VectorEncoding as MmapVectorEncoding;
use format::{ParsedMmapHeaderV1, ParsedMmapHeaderV2, default_mmap_vector_encoding};
use writable::WritableStore;

/// Configuration for the memory-mapped backend.
#[derive(Debug, Clone)]
pub struct MmapBackendConfig {
    /// Path to the memory-mapped file.
    pub file_path: String,
    /// Expected embedding dimension (must match the file header).
    pub dimension: usize,
    /// `false` opens read-write: an existing index is loaded into memory (never truncated),
    /// and flushes replace the file atomically.
    pub read_only: bool,
    /// After this many mutating operations, flush a snapshot to disk (`0` = never auto-flush).
    pub flush_every_ops: usize,
    /// Encoding used when this backend writes v1 snapshots.
    pub vector_encoding: MmapVectorEncoding,
}

impl MmapBackendConfig {
    pub fn new(file_path: impl Into<String>, dimension: usize) -> Self {
        Self {
            file_path: file_path.into(),
            dimension,
            read_only: true,
            flush_every_ops: 512,
            vector_encoding: default_mmap_vector_encoding(),
        }
    }

    #[must_use]
    pub const fn with_read_write(mut self, read_write: bool) -> Self {
        self.read_only = !read_write;
        self
    }

    #[must_use]
    pub const fn with_flush_every(mut self, ops: usize) -> Self {
        self.flush_every_ops = ops;
        self
    }

    #[must_use]
    pub const fn with_vector_encoding(mut self, vector_encoding: MmapVectorEncoding) -> Self {
        self.vector_encoding = vector_encoding;
        self
    }
}

pub(super) enum ActiveMmapData {
    /// No mapped data (read-write mode keeps its chunks in memory).
    Empty,
    AlignedV1 {
        mmap: Mmap,
        header: ParsedMmapHeaderV1,
    },
    AlignedV2 {
        mmap: Mmap,
        header: ParsedMmapHeaderV2,
    },
}

/// Memory-mapped embedding storage backend implementing [`Backend`].
pub struct MmapBackend {
    pub(super) config: MmapBackendConfig,
    pub(super) data: ActiveMmapData,
    /// Keep the file handle open so the mapping stays valid on all platforms (read-only mode).
    pub(super) _file: Option<File>,
    /// In-memory authoritative store when opened read-write.
    pub(super) writable: Option<Arc<RwLock<WritableStore>>>,
    /// Byte length of the last mmap snapshot written in read-write mode.
    pub(super) snapshot_bytes: AtomicU64,
}

impl std::fmt::Debug for MmapBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        use std::sync::atomic::Ordering;
        f.debug_struct("MmapBackend")
            .field("file_path", &self.config.file_path)
            .field("dimension", &self.config.dimension)
            .field("read_only", &self.config.read_only)
            .field("flush_every_ops", &self.config.flush_every_ops)
            .field("vector_encoding", &self.config.vector_encoding)
            .field("writable", &self.writable.is_some())
            .field(
                "snapshot_bytes",
                &self.snapshot_bytes.load(Ordering::Relaxed),
            )
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
#[cfg(feature = "mmap-backend")]
mod tests;
