//! `SPFresh` index and WAL orchestration for `ColdForge`.

use super::ColdForgeBackend;

/// Statistics from cache warming
#[derive(Debug, Clone)]
pub struct CacheWarmStats {
    pub namespace: Option<String>,
    pub bytes_loaded: usize,
    pub time_ms: u64,
    pub vectors_cached: usize,
    pub centroids_loaded: usize,
}
use crate::backend::s3::layout::{index_snapshot_key, wal_key_prefix};
use crate::backend::s3::wal::{WalCompactor, WalConfig, WalOperation, WalReader, WalWriter};
use crate::errors::DripstoneError;
use crate::index::VectorIndex;
use crate::index::spfresh::{CentroidIndex, SPFreshConfig};
use crate::traits::Backend;
use bytes::Bytes;
use std::collections::HashSet;
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::RwLock;

impl ColdForgeBackend {
    /// Get or create WAL writer for a namespace
    pub(super) async fn get_wal_writer(&self, namespace: Option<&str>) -> Arc<WalWriter> {
        let ns_key = namespace.unwrap_or("_default").to_string();
        let wal_config = self.wal_config();
        let store = Arc::clone(&self.store);
        let mut writers = self.wal_writers.write().await;

        writers
            .entry(ns_key.clone())
            .or_insert_with(|| Arc::new(WalWriter::new(store, wal_config, Some(ns_key))))
            .clone()
    }

    /// Get or create index for a namespace
    pub(super) async fn get_index(&self, namespace: Option<&str>) -> Arc<RwLock<CentroidIndex>> {
        let ns_key = namespace.unwrap_or("_default").to_string();
        let mut indexes = self.indexes.write().await;

        indexes
            .entry(ns_key)
            .or_insert_with(|| {
                let spfresh_config = SPFreshConfig {
                    num_centroids: self.config.num_centroids,
                    ..Default::default()
                };

                Arc::new(RwLock::new(CentroidIndex::new(
                    spfresh_config,
                    self.config.vector_dimension,
                )))
            })
            .clone()
    }

    fn wal_config(&self) -> WalConfig {
        WalConfig {
            bucket: self.config.bucket.clone(),
            key_prefix: wal_key_prefix(&self.config.key_prefix),
            batch_interval_ms: self.config.wal_batch_interval_ms,
            max_batch_size: self.config.wal_max_batch_size.max(1),
            enable_compression: self.config.enable_compression,
            background_flush: true,
        }
    }

    fn namespace_key(namespace: Option<&str>) -> String {
        namespace.unwrap_or("_default").to_string()
    }

    fn refresh_enabled(&self) -> bool {
        match self.config.refresh_shared_wal {
            Some(enabled) => enabled,
            None => self.hot_tier.is_some(),
        }
    }

    /// Load the namespace snapshot (if any) and replay WAL segments.
    ///
    /// The first call always replays. Later calls replay again only when shared
    /// WAL refresh is enabled and the refresh interval has elapsed.
    pub(super) async fn sync_namespace(
        &self,
        namespace: Option<&str>,
    ) -> Result<(), DripstoneError> {
        let _gate = self.wal_gate.lock().await;
        self.sync_namespace_locked(namespace, false).await
    }

    /// Replay any WAL segments this process has not applied yet.
    ///
    /// Ignores [`ColdForgeConfig::refresh_shared_wal`](super::ColdForgeConfig::refresh_shared_wal)
    /// and the refresh interval. Use this after another writer `flush`es when
    /// this backend was opened as a single writer.
    ///
    /// # Errors
    ///
    /// Returns an error when listing or reading a segment fails. A missing
    /// object is not marked applied, so a later call retries it.
    pub async fn refresh_shared_wal(&self, namespace: Option<&str>) -> Result<(), DripstoneError> {
        let _gate = self.wal_gate.lock().await;
        self.sync_namespace_locked(namespace, true).await
    }

    async fn sync_namespace_locked(
        &self,
        namespace: Option<&str>,
        force: bool,
    ) -> Result<(), DripstoneError> {
        let ns_key = Self::namespace_key(namespace);
        self.load_snapshot_if_needed(namespace, &ns_key).await?;

        let initial_done = {
            let durability = self.durability.read();
            durability
                .get(&ns_key)
                .is_some_and(|state| state.initial_replay_done)
        };
        let refresh_due = force || self.refresh_due(&ns_key);
        if !initial_done || (self.refresh_enabled() && refresh_due) || force {
            self.replay_unseen(namespace, &ns_key).await?;
            self.hydrate_unscored_members(namespace).await?;
            let mut durability = self.durability.write();
            let state = durability.entry(ns_key).or_default();
            state.initial_replay_done = true;
            state.last_refresh = Some(Instant::now());
        }
        Ok(())
    }

    fn refresh_due(&self, ns_key: &str) -> bool {
        if self.config.wal_refresh_interval_ms == 0 {
            return true;
        }
        let durability = self.durability.read();
        let Some(state) = durability.get(ns_key) else {
            return true;
        };
        state.last_refresh.is_none_or(|at| {
            at.elapsed() >= std::time::Duration::from_millis(self.config.wal_refresh_interval_ms)
        })
    }

    async fn load_snapshot_if_needed(
        &self,
        namespace: Option<&str>,
        ns_key: &str,
    ) -> Result<(), DripstoneError> {
        {
            let durability = self.durability.read();
            if durability
                .get(ns_key)
                .is_some_and(|state| state.snapshot_loaded)
            {
                return Ok(());
            }
        }

        let index_key = index_snapshot_key(&self.config.key_prefix, ns_key);
        let index_bytes: Option<Bytes> =
            match self.store.get_object(&self.config.bucket, &index_key).await {
                Ok(bytes) => Some(bytes),
                Err(e) if e.is_not_found() => None,
                Err(e) => {
                    return Err(DripstoneError::storage(format!(
                        "Failed to load index: {e}"
                    )));
                }
            };

        if let Some(bytes) = index_bytes {
            let spfresh_config = SPFreshConfig {
                num_centroids: self.config.num_centroids,
                ..Default::default()
            };
            let index = CentroidIndex::from_bytes(&bytes, spfresh_config)?;
            let mut indexes = self.indexes.write().await;
            indexes.insert(ns_key.to_string(), Arc::new(RwLock::new(index)));
            drop(indexes);
            tracing::info!("Loaded index for namespace {} from object storage", ns_key);
        } else {
            // Create the empty index once so later replay does not race a second insert.
            let _index = self.get_index(namespace).await;
            tracing::debug!("No existing index found for namespace {}", ns_key);
        }

        let mut durability = self.durability.write();
        durability
            .entry(ns_key.to_string())
            .or_default()
            .snapshot_loaded = true;
        Ok(())
    }

    /// Replay WAL segments that this process has not applied.
    ///
    /// A segment is recorded only after its entries apply. A not-found or
    /// corrupt segment fails the call and stays eligible for the next refresh.
    async fn replay_unseen(
        &self,
        namespace: Option<&str>,
        ns_key: &str,
    ) -> Result<(), DripstoneError> {
        let reader = WalReader::new(
            Arc::clone(&self.store),
            self.wal_config(),
            Some(ns_key.to_string()),
        );
        let wal_files = reader.list_wal_files().await?;
        let applied = {
            let durability = self.durability.read();
            durability
                .get(ns_key)
                .map(|state| state.applied_wal_keys.clone())
                .unwrap_or_default()
        };

        for file in wal_files {
            if applied.contains(&file) {
                continue;
            }
            let entries = reader.read_wal_batch(&file).await?;
            self.apply_wal_entries(namespace, entries).await?;
            let mut durability = self.durability.write();
            durability
                .entry(ns_key.to_string())
                .or_default()
                .applied_wal_keys
                .insert(file);
        }
        Ok(())
    }

    async fn apply_wal_entries(
        &self,
        namespace: Option<&str>,
        entries: Vec<crate::backend::s3::wal::WalEntry>,
    ) -> Result<(), DripstoneError> {
        let index = self.get_index(namespace).await;
        for entry in entries {
            match entry.operation {
                WalOperation::Upsert(record) => {
                    let body_namespace = namespace.unwrap_or(crate::embedding::DEFAULT_NAMESPACE);
                    // Upsert already PUT this envelope before the segment was appended.
                    // Putting again on replay rewrites every body on each cold open.
                    self.bodies.remember(body_namespace, &record).await;
                    self.cache.put(record.clone());
                    let idx = index.write().await;
                    VectorIndex::add(&*idx, record).await?;
                    drop(idx);
                }
                WalOperation::Delete {
                    id,
                    namespace: op_namespace,
                } => {
                    let cache_namespace = op_namespace.as_deref().or(namespace);
                    self.cache.remove_in_namespace(&id, cache_namespace);
                    let idx = index.write().await;
                    VectorIndex::remove(&*idx, &id).await?;
                    drop(idx);
                    if let Some(body_namespace) = cache_namespace {
                        self.bodies.remove(body_namespace, &id).await?;
                    }
                }
                WalOperation::Clear {
                    namespace: op_namespace,
                } => {
                    let clear_namespace = op_namespace.as_deref().or(namespace);
                    if let Some(hot_tier) = &self.hot_tier
                        && let Some(ns) = clear_namespace
                    {
                        hot_tier.delete_namespace(ns).await?;
                    }
                    self.cache.clear_namespace(clear_namespace);
                    if let Some(ns) = clear_namespace {
                        self.bodies.clear_namespace(ns).await?;
                    }
                    let idx = index.write().await;
                    idx.build(&[]).await?;
                    drop(idx);
                }
            }
        }
        Ok(())
    }

    pub(super) async fn index_contains(&self, namespace: &str, id: &str) -> bool {
        let index = self.get_index(Some(namespace)).await;
        let idx = index.read().await;
        idx.get_record(id).is_some()
    }

    /// Save the membership snapshot, then delete WAL segments whose envelopes
    /// are already in the chunk-id body store.
    ///
    /// # Errors
    ///
    /// Returns an error when the snapshot cannot be saved or a segment is not
    /// safe to delete. A refusal deletes nothing.
    pub async fn compact_wal(
        &self,
        namespace: Option<&str>,
        keep_latest_n: usize,
    ) -> Result<usize, DripstoneError> {
        self.flush().await?;
        self.sync_namespace(namespace).await?;
        self.save_index(namespace).await?;
        let compactor = WalCompactor::new(
            std::sync::Arc::clone(&self.store),
            self.wal_config(),
            Some(Self::namespace_key(namespace)),
        );
        compactor.compact(keep_latest_n).await
    }

    async fn hydrate_unscored_members(
        &self,
        namespace: Option<&str>,
    ) -> Result<(), DripstoneError> {
        let index = self.get_index(namespace).await;
        let unscored = {
            let idx = index.read().await;
            let mut ids = HashSet::new();
            for id in idx
                .clusters
                .read()
                .iter()
                .flat_map(|cluster| cluster.record_ids.iter())
            {
                if idx.get_record(id).is_none() {
                    ids.insert(id.clone());
                }
            }
            ids.into_iter().collect::<Vec<_>>()
        };
        if unscored.is_empty() {
            return Ok(());
        }
        let body_namespace = namespace.unwrap_or(crate::embedding::DEFAULT_NAMESPACE);
        let records = self.bodies.get_many(body_namespace, &unscored).await?;
        for record in records {
            let idx = index.write().await;
            VectorIndex::add(&*idx, record).await?;
        }
        Ok(())
    }

    /// Save index to object storage.
    ///
    /// The snapshot stores centroids and membership, not chunk envelopes.
    async fn save_index(&self, namespace: Option<&str>) -> Result<(), DripstoneError> {
        let ns_key = namespace.unwrap_or("_default");
        let index = self.get_index(namespace).await;

        let bytes = {
            let idx = index.read().await;
            idx.to_bytes()?
        };

        let index_key = index_snapshot_key(&self.config.key_prefix, ns_key);

        self.store
            .put_object(
                &self.config.bucket,
                &index_key,
                Bytes::from(bytes),
                Some("application/octet-stream"),
            )
            .await
            .map_err(|e| {
                DripstoneError::storage(format!("Failed to save index: {index_key}: {e}"))
            })?;

        tracing::info!("Saved index for namespace {} to S3", ns_key);

        Ok(())
    }

    /// Warm cache by loading index and frequently accessed clusters
    ///
    /// # Errors
    ///
    /// Returns an error when the operation cannot be completed.
    pub async fn warm_cache(
        &self,
        namespace: Option<&str>,
    ) -> Result<CacheWarmStats, DripstoneError> {
        let start = std::time::Instant::now();

        self.sync_namespace(namespace).await?;

        let index = self.get_index(namespace).await;
        let stats = index.read().await.stats();

        let elapsed = start.elapsed();

        Ok(CacheWarmStats {
            namespace: namespace.map(std::string::ToString::to_string),
            bytes_loaded: stats.num_records * self.config.vector_dimension * 4,
            time_ms: u64::try_from(elapsed.as_millis()).unwrap_or(u64::MAX),
            vectors_cached: stats.num_records,
            centroids_loaded: stats.num_centroids,
        })
    }
}
