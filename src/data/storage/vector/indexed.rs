//! ANN index helpers for Caliche vector backends.
//!
//! Applies Dripstone [`IndexedBackend`] with optional Superstore-style deferred
//! indexing when `CALICHE_DEFERRED_ANN_INDEX` is enabled.

use std::sync::Arc;

use anyhow::{Context, Result};
use dripstone::index::HnswIndex;
use dripstone::{Backend, HnswConfig, IndexUpdateMode, IndexedBackend, IndexedBackendConfig};

const DEFAULT_VECTOR_DIMENSION: usize = 1536;

fn positive_usize_from_env(key: &str) -> Option<usize> {
    std::env::var(key)
        .ok()
        .and_then(|value| value.trim().parse().ok())
        .filter(|&value| value > 0)
}

/// Return true when deferred ANN indexing is enabled via environment.
#[must_use]
pub fn deferred_ann_index_enabled_from_env() -> bool {
    match std::env::var("CALICHE_DEFERRED_ANN_INDEX") {
        Ok(value) => matches!(value.trim(), "1" | "true" | "TRUE" | "yes" | "YES"),
        Err(_) => false,
    }
}

/// Read `VECTOR_BACKEND_DIMENSION` or return the Caliche default (1536).
#[must_use]
pub fn vector_dimension_from_env() -> usize {
    std::env::var("VECTOR_BACKEND_DIMENSION")
        .ok()
        .and_then(|value| value.trim().parse().ok())
        .filter(|&dimension: &usize| dimension > 0)
        .unwrap_or(DEFAULT_VECTOR_DIMENSION)
}

/// Build [`IndexedBackendConfig`] from environment.
///
/// When [`deferred_ann_index_enabled_from_env`] is true, uses
/// [`IndexUpdateMode::Deferred`] with tail scan enabled.
#[must_use]
pub fn indexed_backend_config_from_env() -> IndexedBackendConfig {
    if deferred_ann_index_enabled_from_env() {
        let mut config = IndexedBackendConfig {
            index_update_mode: IndexUpdateMode::Deferred,
            scan_unindexed_tail: true,
            ..IndexedBackendConfig::default()
        };
        if let Some(threshold) = positive_usize_from_env("CALICHE_DEFERRED_SEARCH_FLUSH_THRESHOLD")
        {
            config.deferred_search_flush_threshold = threshold;
        }
        if let Some(threshold) = positive_usize_from_env("CALICHE_DEFERRED_FLUSH_THRESHOLD") {
            config.deferred_flush_threshold = threshold;
        }
        if let Some(max_tail) = positive_usize_from_env("CALICHE_DEFERRED_MAX_TAIL_SCAN") {
            config.max_tail_scan_records = max_tail;
        }
        config
    } else {
        IndexedBackendConfig::default()
    }
}

/// Wrap `backend` with an in-memory HNSW [`IndexedBackend`] when deferred ANN
/// indexing is enabled and `dimension` is known.
pub fn wrap_with_deferred_ann_index(
    backend: Arc<dyn Backend + Send + Sync>,
    dimension: usize,
    hnsw_config: Option<HnswConfig>,
) -> Result<Arc<dyn Backend + Send + Sync>> {
    if !deferred_ann_index_enabled_from_env() {
        return Ok(backend);
    }
    if dimension == 0 {
        anyhow::bail!("deferred ANN index requires a positive embedding dimension");
    }

    let inner: Arc<dyn Backend> = backend;
    let indexed = IndexedBackend::new(
        inner,
        Box::new(HnswIndex::new(hnsw_config.unwrap_or_default(), dimension)),
        indexed_backend_config_from_env(),
    );
    Ok(Arc::new(indexed) as Arc<dyn Backend + Send + Sync>)
}

/// Resolve embedding dimension for workspace ANN wrapping.
///
/// Prefers a probed dimension from the embedding provider, then
/// `VECTOR_BACKEND_DIMENSION`.
#[must_use]
pub fn resolve_ann_index_dimension(probed_dimension: Option<usize>) -> Option<usize> {
    probed_dimension
        .filter(|&dimension| dimension > 0)
        .or_else(|| {
            if deferred_ann_index_enabled_from_env() {
                Some(vector_dimension_from_env())
            } else {
                None
            }
        })
}

/// Wrap a workspace store backend when deferred ANN indexing is enabled.
pub fn wrap_workspace_backend_with_ann_index(
    backend: Arc<dyn Backend + Send + Sync>,
    probed_dimension: Option<usize>,
) -> Result<Arc<dyn Backend + Send + Sync>> {
    let Some(dimension) = resolve_ann_index_dimension(probed_dimension) else {
        return Ok(backend);
    };
    wrap_with_deferred_ann_index(backend, dimension, None)
        .context("failed to wrap workspace backend with deferred ANN index")
}

/// Drain deferred ANN inserts on `backend` when supported.
pub async fn flush_deferred_ann_index(backend: &dyn Backend) -> Result<()> {
    backend
        .flush_pending_ann_index()
        .await
        .context("failed to flush deferred ANN index")
}

#[cfg(test)]
mod tests {
    use super::*;
    use dripstone::MemoryBackend;

    #[test]
    #[serial_test::serial(env)]
    fn deferred_env_parses_truthy_values() {
        let mut env = crate::test_env::EnvGuard::acquire();
        env.set("CALICHE_DEFERRED_ANN_INDEX", "true");
        assert!(deferred_ann_index_enabled_from_env());
        let config = indexed_backend_config_from_env();
        assert_eq!(config.index_update_mode, IndexUpdateMode::Deferred);
        assert!(config.scan_unindexed_tail);
    }

    #[test]
    #[serial_test::serial(env)]
    fn deferred_env_reads_search_flush_and_tail_cap() {
        let mut env = crate::test_env::EnvGuard::acquire();
        env.set("CALICHE_DEFERRED_ANN_INDEX", "true");
        env.set("CALICHE_DEFERRED_SEARCH_FLUSH_THRESHOLD", "512");
        env.set("CALICHE_DEFERRED_FLUSH_THRESHOLD", "64");
        env.set("CALICHE_DEFERRED_MAX_TAIL_SCAN", "128");
        let config = indexed_backend_config_from_env();
        assert_eq!(config.deferred_search_flush_threshold, 512);
        assert_eq!(config.deferred_flush_threshold, 64);
        assert_eq!(config.max_tail_scan_records, 128);
    }

    #[test]
    #[serial_test::serial(env)]
    fn resolve_dimension_uses_probe_then_env_default() {
        let mut env = crate::test_env::EnvGuard::acquire();
        env.set("CALICHE_DEFERRED_ANN_INDEX", "true");
        env.set("VECTOR_BACKEND_DIMENSION", "384");
        assert_eq!(resolve_ann_index_dimension(Some(768)), Some(768));
        assert_eq!(resolve_ann_index_dimension(None), Some(384));
    }

    #[tokio::test]
    #[serial_test::serial(env)]
    async fn wrap_is_noop_when_env_disabled() {
        let mut env = crate::test_env::EnvGuard::acquire();
        env.remove("CALICHE_DEFERRED_ANN_INDEX");
        let inner: Arc<dyn Backend + Send + Sync> = Arc::new(MemoryBackend::new());
        let wrapped = wrap_with_deferred_ann_index(inner, 8, None).expect("wrap");
        assert_eq!(wrapped.count(None).await.expect("count"), 0);
    }

    #[tokio::test]
    #[serial_test::serial(env)]
    async fn wrap_adds_deferred_index_layer_when_enabled() {
        let mut env = crate::test_env::EnvGuard::acquire();
        env.set("CALICHE_DEFERRED_ANN_INDEX", "true");
        let inner: Arc<dyn Backend + Send + Sync> = Arc::new(MemoryBackend::new());
        let wrapped = wrap_with_deferred_ann_index(inner, 4, None).expect("wrap");
        let chunk = dripstone::Chunk::new(
            "doc",
            0,
            "hello",
            vec![1.0_f32, 0.0, 0.0, 0.0],
            0,
            5,
            1,
            serde_json::json!({}),
            None,
        );
        wrapped.upsert(&[chunk]).await.expect("upsert");
        assert_eq!(wrapped.count(None).await.expect("count"), 1);
    }
}
