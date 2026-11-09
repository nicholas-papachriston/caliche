//! ColdForge object-key segments.
//!
//! WAL, chunk bodies, and the membership snapshot share one prefix:
//! `{prefix}wal/{namespace}/…`, `{prefix}bodies/{namespace}/{chunk_id}`,
//! `{prefix}indexes/{namespace}/index.bin`. A cluster index is never a path
//! segment. Body keys use the embedding namespace (`default`), while the WAL
//! and snapshot use `_default` for the unnamed namespace.

use crate::embedding::DEFAULT_NAMESPACE;

/// Path segment under the ColdForge prefix.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColdForgeSegment {
    /// Write-ahead log segments.
    Wal,
    /// Chunk envelopes keyed by chunk id.
    Body,
    /// Membership snapshot (`index.bin`).
    Index,
}

impl ColdForgeSegment {
    /// Directory name for this segment.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Wal => "wal",
            Self::Body => "bodies",
            Self::Index => "indexes",
        }
    }
}

/// Snapshot object name under `indexes/{namespace}/`.
pub const INDEX_SNAPSHOT_FILE: &str = "index.bin";

/// WAL and snapshot namespace when the caller did not set one.
///
/// Body objects for that namespace use [`DEFAULT_NAMESPACE`] (`default`).
pub const WAL_NAMESPACE_DEFAULT: &str = "_default";

/// `{root}wal/`.
#[must_use]
pub fn wal_key_prefix(root_prefix: &str) -> String {
    format!("{root_prefix}{}/", ColdForgeSegment::Wal.as_str())
}

/// `{root}bodies/{namespace}/{chunk_id}`.
#[must_use]
pub fn chunk_body_object_key(root_prefix: &str, namespace: &str, chunk_id: &str) -> String {
    format!(
        "{root_prefix}{}/{namespace}/{chunk_id}",
        ColdForgeSegment::Body.as_str()
    )
}

/// `{root}indexes/{namespace}/index.bin`.
#[must_use]
pub fn index_snapshot_key(root_prefix: &str, namespace: &str) -> String {
    format!(
        "{root_prefix}{}/{namespace}/{INDEX_SNAPSHOT_FILE}",
        ColdForgeSegment::Index.as_str()
    )
}

/// Root prefix that owns `wal/`, `bodies/`, and `indexes/`.
///
/// `wal_key_prefix` is the value stored on [`super::wal::WalConfig::key_prefix`].
#[must_use]
pub fn root_prefix_from_wal(wal_key_prefix: &str) -> Option<&str> {
    let suffix = format!("{}/", ColdForgeSegment::Wal.as_str());
    wal_key_prefix.strip_suffix(&suffix)
}

/// Body-store namespace for a WAL namespace segment.
#[must_use]
pub fn body_namespace_for_wal(wal_namespace: &str) -> &str {
    match wal_namespace {
        WAL_NAMESPACE_DEFAULT => DEFAULT_NAMESPACE,
        other => other,
    }
}

#[cfg(test)]
mod tests {
    use super::{
        ColdForgeSegment, body_namespace_for_wal, chunk_body_object_key, index_snapshot_key,
        root_prefix_from_wal, wal_key_prefix,
    };

    #[test]
    fn keys_use_chunk_id_and_not_a_cluster_index() {
        let root = "cf/";
        assert_eq!(wal_key_prefix(root), "cf/wal/");
        assert_eq!(root_prefix_from_wal("cf/wal/"), Some("cf/"));
        let body = chunk_body_object_key(root, "demo", "doc#3");
        let snapshot = index_snapshot_key(root, "demo");
        assert_eq!(body, "cf/bodies/demo/doc#3");
        assert_eq!(snapshot, "cf/indexes/demo/index.bin");
        assert!(body.ends_with("doc#3"));
        assert!(!body.contains("cluster"));
        assert!(!snapshot.contains("cluster"));
        assert_eq!(ColdForgeSegment::Body.as_str(), "bodies");
        assert_eq!(body_namespace_for_wal("_default"), "default");
        assert_eq!(body_namespace_for_wal("demo"), "demo");
    }
}
