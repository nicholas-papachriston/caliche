//! Scaling and distributed-correctness coverage that does not need live AWS.
//!
//! ColdForge cases use [`MemoryObjectStore`]. Redis SCAN coverage runs only when
//! `REDIS_URL` is set. Tiered recall cases are in-process.

use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use dripstone::core::{Capabilities, Chunk, Metric, SearchQuery, SearchResult};
use dripstone::errors::DripstoneError;
use dripstone::traits::Backend;
use dripstone::{SearchStrategy, TierConfig, TieredChainBuilder, WriteStrategy};

fn chunk(doc_id: &str, namespace: &str, vector: Vec<f32>) -> Chunk {
    Chunk::new(
        doc_id,
        0,
        doc_id,
        vector,
        0,
        0,
        1,
        serde_json::json!({ "doc_id": doc_id }),
        Some(namespace.to_string()),
    )
}

fn cosine(left: &[f32], right: &[f32]) -> f32 {
    let mut dot = 0.0;
    let mut left_norm = 0.0;
    let mut right_norm = 0.0;
    for (a, b) in left.iter().zip(right.iter()) {
        dot += a * b;
        left_norm += a * a;
        right_norm += b * b;
    }
    let denom = left_norm.sqrt() * right_norm.sqrt();
    if denom == 0.0 { 0.0 } else { dot / denom }
}

#[derive(Debug)]
struct MapBackend {
    rows: Mutex<Vec<Chunk>>,
    fail_search: bool,
}

impl MapBackend {
    fn new() -> Self {
        Self {
            rows: Mutex::new(Vec::new()),
            fail_search: false,
        }
    }

    fn failing_search() -> Self {
        Self {
            rows: Mutex::new(Vec::new()),
            fail_search: true,
        }
    }

    fn rows(&self) -> Result<std::sync::MutexGuard<'_, Vec<Chunk>>, DripstoneError> {
        self.rows
            .lock()
            .map_err(|_| DripstoneError::storage("map backend lock poisoned"))
    }
}

#[async_trait]
impl Backend for MapBackend {
    fn capabilities(&self) -> Capabilities {
        Capabilities::memory()
    }

    async fn upsert(&self, chunks: &[Chunk]) -> Result<(), DripstoneError> {
        let mut rows = self.rows()?;
        for chunk in chunks {
            rows.retain(|existing| {
                existing.id != chunk.id || existing.namespace != chunk.namespace
            });
            rows.push(chunk.clone());
        }
        Ok(())
    }

    async fn get(
        &self,
        id: &str,
        namespace: Option<&str>,
    ) -> Result<Option<Chunk>, DripstoneError> {
        let rows = self.rows()?;
        Ok(rows
            .iter()
            .find(|row| row.id == id && row.namespace.as_deref() == namespace)
            .cloned())
    }

    async fn delete(&self, id: &str, namespace: Option<&str>) -> Result<(), DripstoneError> {
        let mut rows = self.rows()?;
        rows.retain(|row| row.id != id || row.namespace.as_deref() != namespace);
        Ok(())
    }

    async fn delete_document(
        &self,
        doc_id: &str,
        namespace: Option<&str>,
    ) -> Result<usize, DripstoneError> {
        let mut rows = self.rows()?;
        let before = rows.len();
        rows.retain(|row| row.doc_id != doc_id || row.namespace.as_deref() != namespace);
        Ok(before - rows.len())
    }

    async fn search(&self, query: &SearchQuery) -> Result<Vec<SearchResult>, DripstoneError> {
        if self.fail_search {
            return Err(DripstoneError::storage("search failed"));
        }
        let rows = self.rows()?;
        let mut scored = Vec::new();
        for row in rows.iter() {
            if row.namespace.as_deref() != query.namespace.as_deref() {
                continue;
            }
            let score = cosine(query.vector.as_ref(), row.vector.as_ref());
            scored.push(SearchResult::new(row.clone(), score, 0));
        }
        drop(rows);
        scored.sort_by(|left, right| {
            right
                .score
                .partial_cmp(&left.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        scored.truncate(query.limit);
        for (rank, result) in scored.iter_mut().enumerate() {
            result.rank = rank;
        }
        Ok(scored)
    }

    async fn namespaces(&self) -> Result<Vec<String>, DripstoneError> {
        let rows = self.rows()?;
        let mut names = Vec::new();
        for row in rows.iter() {
            if let Some(namespace) = row.namespace.clone()
                && !names.contains(&namespace)
            {
                names.push(namespace);
            }
        }
        Ok(names)
    }

    async fn delete_namespace(&self, namespace: &str) -> Result<(), DripstoneError> {
        let mut rows = self.rows()?;
        rows.retain(|row| row.namespace.as_deref() != Some(namespace));
        Ok(())
    }

    async fn count(&self, namespace: Option<&str>) -> Result<usize, DripstoneError> {
        let rows = self.rows()?;
        Ok(rows
            .iter()
            .filter(|row| row.namespace.as_deref() == namespace)
            .count())
    }

    async fn list(
        &self,
        namespace: Option<&str>,
        limit: Option<usize>,
    ) -> Result<Vec<Chunk>, DripstoneError> {
        let rows = self.rows()?;
        let mut listed = Vec::new();
        for row in rows
            .iter()
            .filter(|row| row.namespace.as_deref() == namespace)
        {
            if limit.is_some_and(|cap| listed.len() >= cap) {
                break;
            }
            listed.push(row.clone());
        }
        Ok(listed)
    }
}

fn chain(
    cold: Arc<dyn Backend>,
    hot: Arc<dyn Backend>,
    hot_cap: Option<usize>,
    strategy: WriteStrategy,
) -> Result<Arc<dyn Backend>, DripstoneError> {
    let mut hot_config = TierConfig::new("hot");
    if let Some(cap) = hot_cap {
        hot_config = hot_config.max_vectors(cap);
    }
    TieredChainBuilder::new()
        .tier(cold, TierConfig::new("cold"))
        .tier(hot, hot_config)
        .write_strategy(strategy)
        .search_strategy(SearchStrategy::HotThenSourceFallback)
        .background_sync(false)
        .build()
}

#[tokio::test]
async fn partial_hot_tier_cannot_hide_a_nearer_cold_neighbor() {
    let cold: Arc<dyn Backend> = Arc::new(MapBackend::new());
    let hot: Arc<dyn Backend> = Arc::new(MapBackend::new());
    let near = chunk("near", "ns", vec![1.0, 0.0]);
    let far = chunk("far", "ns", vec![0.0, 1.0]);
    cold.upsert(&[near.clone(), far.clone()])
        .await
        .expect("seed cold");
    hot.upsert(std::slice::from_ref(&far))
        .await
        .expect("seed partial hot");

    let backend = chain(
        Arc::clone(&cold),
        hot,
        Some(1),
        WriteStrategy::WriteToSource,
    )
    .expect("tiered chain");
    let results = backend
        .search(
            &SearchQuery::new(vec![1.0, 0.0])
                .metric(Metric::Cosine)
                .limit(1)
                .namespace("ns"),
        )
        .await
        .expect("search");
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].chunk.id, near.id);
}

#[tokio::test]
async fn complete_hot_tier_does_not_require_a_failing_cold_tier() {
    let cold: Arc<dyn Backend> = Arc::new(MapBackend::failing_search());
    let hot: Arc<dyn Backend> = Arc::new(MapBackend::new());
    let backend = chain(
        Arc::clone(&cold),
        Arc::clone(&hot),
        None,
        WriteStrategy::WriteToSourceAndHot,
    )
    .expect("tiered chain");
    let row = chunk("only", "ns", vec![1.0, 0.0]);
    backend
        .upsert(std::slice::from_ref(&row))
        .await
        .expect("upsert");
    let results = backend
        .search(
            &SearchQuery::new(vec![1.0, 0.0])
                .metric(Metric::Cosine)
                .limit(1)
                .namespace("ns"),
        )
        .await
        .expect("complete hot tier should satisfy the limit");
    assert_eq!(results[0].chunk.id, row.id);
}

#[tokio::test]
async fn capped_hot_tier_still_consults_a_failing_cold_tier() {
    let cold: Arc<dyn Backend> = Arc::new(MapBackend::failing_search());
    let hot: Arc<dyn Backend> = Arc::new(MapBackend::new());
    hot.upsert(std::slice::from_ref(&chunk("cached", "ns", vec![1.0, 0.0])))
        .await
        .expect("seed hot");
    let backend =
        chain(cold, hot, Some(1), WriteStrategy::WriteToSourceAndHot).expect("tiered chain");
    let err = backend
        .search(
            &SearchQuery::new(vec![1.0, 0.0])
                .metric(Metric::Cosine)
                .limit(1)
                .namespace("ns"),
        )
        .await
        .expect_err("a capped hot page is not a complete corpus");
    assert!(err.to_string().contains("search failed"));
}

#[cfg(feature = "coldforge-backend")]
mod coldforge {
    use super::*;
    use bytes::Bytes;
    use dripstone::backend::coldforge::{ColdForgeBackend, ColdForgeConfig};
    use dripstone::backend::s3::wal::{
        WalCompactor, WalConfig, WalOperation, WalReader, WalWriter,
    };
    use dripstone::embedding::EmbeddingRecord;
    use dripstone::storage::{MemoryObjectStore, ObjectInfo, ObjectMetadata, ObjectStore};
    use std::sync::atomic::{AtomicUsize, Ordering};

    fn cold_config(hot: Option<Arc<dyn Backend + Send + Sync>>) -> ColdForgeConfig {
        ColdForgeConfig {
            bucket: "bucket".to_string(),
            key_prefix: "cf/".to_string(),
            wal_batch_interval_ms: 60_000,
            wal_max_batch_size: 10_000,
            enable_compression: false,
            vector_dimension: 4,
            num_centroids: 1,
            cache_size_bytes: 8 * 1024 * 1024,
            hot_tier: hot,
            refresh_shared_wal: None,
            // 0 lists the WAL on every refreshing read. The product default is
            // 1000 ms; these tests check that a flush is visible on the next read.
            wal_refresh_interval_ms: 0,
            ..ColdForgeConfig::default()
        }
    }

    fn pair(
        hot: Arc<dyn Backend + Send + Sync>,
    ) -> (ColdForgeBackend, ColdForgeBackend, Arc<dyn ObjectStore>) {
        let store: Arc<dyn ObjectStore> = MemoryObjectStore::shared();
        let writer = ColdForgeBackend::with_object_store(
            Arc::clone(&store),
            cold_config(Some(Arc::clone(&hot))),
        );
        let reader =
            ColdForgeBackend::with_object_store(Arc::clone(&store), cold_config(Some(hot)));
        (writer, reader, store)
    }

    fn query(vector: Vec<f32>, namespace: &str) -> SearchQuery {
        SearchQuery::new(vector)
            .metric(Metric::Cosine)
            .limit(10)
            .namespace(namespace)
    }

    #[tokio::test]
    async fn second_instance_sees_flushed_wal_and_ignores_unflushed_hot_hits() {
        let hot: Arc<dyn Backend + Send + Sync> = Arc::new(MapBackend::new());
        let (writer, reader, _) = pair(Arc::clone(&hot));
        let row = chunk("doc", "demo", vec![1.0, 0.0, 0.0, 0.0]);
        writer
            .upsert(std::slice::from_ref(&row))
            .await
            .expect("upsert");

        let early = reader
            .search(&query(vec![1.0, 0.0, 0.0, 0.0], "demo"))
            .await
            .expect("search before flush");
        assert!(
            early.is_empty(),
            "unflushed writes must not show up on another instance"
        );
        assert!(
            reader
                .get("doc#0", Some("demo"))
                .await
                .expect("get")
                .is_none(),
            "redis/hot hit without a WAL segment must not resurrect the id"
        );
        assert!(
            writer
                .get("doc#0", Some("demo"))
                .await
                .expect("writer get")
                .is_some(),
            "the writer still reads its own index"
        );

        writer.flush().await.expect("flush");
        let found = reader
            .search(&query(vec![1.0, 0.0, 0.0, 0.0], "demo"))
            .await
            .expect("search after flush");
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].chunk.id, "doc#0");
        assert_eq!(reader.count(Some("demo")).await.expect("count"), 1);
        assert!(
            reader
                .get("doc#0", Some("demo"))
                .await
                .expect("get after flush")
                .is_some()
        );
    }

    #[tokio::test]
    async fn get_ignores_hot_tier_after_the_index_deletes_the_id() {
        let hot: Arc<dyn Backend + Send + Sync> = Arc::new(MapBackend::new());
        let (writer, reader, _) = pair(Arc::clone(&hot));
        let row = chunk("doc", "demo", vec![1.0, 0.0, 0.0, 0.0]);
        writer
            .upsert(std::slice::from_ref(&row))
            .await
            .expect("upsert");
        writer.flush().await.expect("flush upsert");
        writer.delete("doc#0", Some("demo")).await.expect("delete");
        writer.flush().await.expect("flush delete");
        hot.upsert(std::slice::from_ref(&row))
            .await
            .expect("reintroduce stale hot copy");

        assert!(
            reader
                .get("doc#0", Some("demo"))
                .await
                .expect("get")
                .is_none()
        );
        let results = reader
            .search(&query(vec![1.0, 0.0, 0.0, 0.0], "demo"))
            .await
            .expect("search");
        assert!(results.is_empty());
        assert_eq!(reader.count(Some("demo")).await.expect("count"), 0);
    }

    #[tokio::test]
    async fn get_reads_the_body_after_the_local_cache_evicts() {
        let store: Arc<dyn ObjectStore> = MemoryObjectStore::shared();
        let mut config = cold_config(None);
        config.cache_size_bytes = 1;
        config.refresh_shared_wal = Some(false);
        let backend = ColdForgeBackend::with_object_store(store, config);
        let first = chunk("first", "demo", vec![1.0, 0.0, 0.0, 0.0]);
        let second = chunk("second", "demo", vec![0.0, 1.0, 0.0, 0.0]);
        backend
            .upsert(&[first.clone(), second])
            .await
            .expect("upsert");
        let fetched = backend
            .get("first#0", Some("demo"))
            .await
            .expect("get")
            .expect("body store still has the evicted cache entry");
        assert_eq!(fetched.id, first.id);
        assert_eq!(fetched.content, first.content);
        let hits = backend
            .search(&query(vec![1.0, 0.0, 0.0, 0.0], "demo"))
            .await
            .expect("search");
        let hit = hits
            .iter()
            .find(|hit| hit.chunk.id == first.id)
            .expect("search still returns the evicted id");
        assert_eq!(hit.chunk.content, fetched.content);
    }

    #[tokio::test]
    async fn one_upsert_batch_keeps_namespaces_apart() {
        let hot: Arc<dyn Backend + Send + Sync> = Arc::new(MapBackend::new());
        let (writer, reader, _) = pair(hot);
        writer
            .upsert(&[
                chunk("alpha-doc", "alpha", vec![1.0, 0.0, 0.0, 0.0]),
                chunk("beta-doc", "beta", vec![0.0, 1.0, 0.0, 0.0]),
            ])
            .await
            .expect("upsert");
        writer.flush().await.expect("flush");
        assert_eq!(reader.count(Some("alpha")).await.expect("alpha"), 1);
        assert_eq!(reader.count(Some("beta")).await.expect("beta"), 1);
        let beta = reader
            .search(&query(vec![0.0, 1.0, 0.0, 0.0], "beta"))
            .await
            .expect("beta search");
        assert_eq!(beta.len(), 1);
        assert_eq!(beta[0].chunk.namespace.as_deref(), Some("beta"));
        let alpha = reader
            .search(&query(vec![1.0, 0.0, 0.0, 0.0], "alpha"))
            .await
            .expect("alpha search");
        assert_eq!(alpha.len(), 1);
        assert_eq!(alpha[0].chunk.id, "alpha-doc#0");
    }

    #[tokio::test]
    async fn single_writer_mode_ignores_later_segments_until_refresh() {
        let store: Arc<dyn ObjectStore> = MemoryObjectStore::shared();
        let mut config = cold_config(None);
        config.refresh_shared_wal = Some(false);
        let writer = ColdForgeBackend::with_object_store(Arc::clone(&store), config.clone());
        let reader = ColdForgeBackend::with_object_store(store, config);
        let _ = reader
            .search(&query(vec![1.0, 0.0, 0.0, 0.0], "demo"))
            .await
            .expect("initial load");
        writer
            .upsert(std::slice::from_ref(&chunk(
                "doc",
                "demo",
                vec![1.0, 0.0, 0.0, 0.0],
            )))
            .await
            .expect("upsert");
        writer.flush().await.expect("flush");
        let stale = reader
            .search(&query(vec![1.0, 0.0, 0.0, 0.0], "demo"))
            .await
            .expect("stale search");
        assert!(stale.is_empty());
        reader
            .refresh_shared_wal(Some("demo"))
            .await
            .expect("refresh");
        let fresh = reader
            .search(&query(vec![1.0, 0.0, 0.0, 0.0], "demo"))
            .await
            .expect("refreshed search");
        assert_eq!(fresh.len(), 1);
    }

    struct FlakyStore {
        inner: Arc<MemoryObjectStore>,
        remaining_failures: AtomicUsize,
    }

    #[async_trait]
    impl ObjectStore for FlakyStore {
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
            loop {
                let left = self.remaining_failures.load(Ordering::SeqCst);
                if left == 0 {
                    break;
                }
                if self
                    .remaining_failures
                    .compare_exchange(left, left - 1, Ordering::SeqCst, Ordering::SeqCst)
                    .is_ok()
                {
                    return Err(DripstoneError::storage("injected put failure"));
                }
            }
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
            self.inner.list_objects(bucket, prefix).await
        }
    }

    fn wal_config() -> WalConfig {
        WalConfig {
            bucket: "bucket".to_string(),
            key_prefix: "wal/".to_string(),
            batch_interval_ms: 60_000,
            max_batch_size: 100,
            enable_compression: false,
            background_flush: false,
        }
    }

    fn wal_record(id: &str) -> EmbeddingRecord {
        EmbeddingRecord::new(id, vec![1.0, 0.0, 0.0, 0.0])
    }

    #[tokio::test]
    async fn failed_wal_flush_keeps_the_batch_for_retry() {
        let inner = MemoryObjectStore::shared();
        let store: Arc<dyn ObjectStore> = Arc::new(FlakyStore {
            inner: Arc::clone(&inner),
            remaining_failures: AtomicUsize::new(1),
        });
        let writer = WalWriter::new(Arc::clone(&store), wal_config(), Some("ns".to_string()));
        writer
            .append(WalOperation::Upsert(wal_record("kept")))
            .await
            .expect("append");
        let err = writer.flush_batch().await.expect_err("first put fails");
        assert!(err.to_string().contains("injected put failure"));
        writer.flush_batch().await.expect("retry flush");
        let reader = WalReader::new(store, wal_config(), Some("ns".to_string()));
        let mut ids = Vec::new();
        reader
            .replay(|entry| {
                if let WalOperation::Upsert(record) = entry.operation {
                    ids.push(record.id.to_string());
                }
                Ok(())
            })
            .await
            .expect("replay");
        assert_eq!(ids, vec!["kept".to_string()]);
    }

    #[tokio::test]
    async fn concurrent_flushes_do_not_overwrite_each_other() {
        let store: Arc<dyn ObjectStore> = MemoryObjectStore::shared();
        let left = WalWriter::new(Arc::clone(&store), wal_config(), Some("ns".to_string()));
        let right = WalWriter::new(Arc::clone(&store), wal_config(), Some("ns".to_string()));
        left.append(WalOperation::Upsert(wal_record("left")))
            .await
            .expect("left append");
        right
            .append(WalOperation::Upsert(wal_record("right")))
            .await
            .expect("right append");
        tokio::try_join!(left.flush_batch(), right.flush_batch()).expect("flush both");
        let reader = WalReader::new(store, wal_config(), Some("ns".to_string()));
        let files = reader.list_wal_files().await.expect("list");
        assert_eq!(
            files.len(),
            2,
            "same-millisecond flushes must not share a key"
        );
        let mut ids = Vec::new();
        reader
            .replay(|entry| {
                if let WalOperation::Upsert(record) = entry.operation {
                    ids.push(record.id.to_string());
                }
                Ok(())
            })
            .await
            .expect("replay");
        ids.sort();
        assert_eq!(ids, vec!["left".to_string(), "right".to_string()]);
    }

    #[tokio::test]
    async fn compactor_does_not_delete_the_only_envelopes() {
        let store: Arc<dyn ObjectStore> = MemoryObjectStore::shared();
        let mut config = wal_config();
        config.max_batch_size = 1;
        let writer = WalWriter::new(Arc::clone(&store), config.clone(), Some("ns".to_string()));
        for id in ["a", "b", "c"] {
            writer
                .append(WalOperation::Upsert(wal_record(id)))
                .await
                .expect("append");
        }
        let compactor =
            WalCompactor::new(Arc::clone(&store), config.clone(), Some("ns".to_string()));
        let err = compactor.compact(1).await.expect_err("compact must refuse");
        assert!(err.to_string().contains("refuses to delete"));
        assert_eq!(compactor.compact(3).await.expect("nothing to delete"), 0);
        let reader = WalReader::new(store, config, Some("ns".to_string()));
        assert_eq!(reader.list_wal_files().await.expect("list").len(), 3);
        let mut count = 0;
        reader
            .replay(|_| {
                count += 1;
                Ok(())
            })
            .await
            .expect("replay");
        assert_eq!(count, 3);
    }

    async fn put_membership(
        store: &Arc<dyn ObjectStore>,
        ids: &[&str],
    ) -> Result<(), DripstoneError> {
        use dripstone::index::VectorIndex;
        use dripstone::index::spfresh::{CentroidIndex, SPFreshConfig};

        let index = CentroidIndex::new(
            SPFreshConfig {
                num_centroids: 1,
                max_cluster_records: None,
                ..SPFreshConfig::default()
            },
            4,
        );
        for id in ids {
            VectorIndex::add(&index, wal_record(id)).await?;
        }
        let bytes = index.to_bytes()?;
        store
            .put_object(
                "bucket",
                "indexes/ns/index.bin",
                Bytes::from(bytes),
                Some("application/octet-stream"),
            )
            .await?;
        for id in ids {
            let body = serde_json::to_vec(&wal_record(id))
                .map_err(|err| DripstoneError::storage(format!("body json: {err}")))?;
            store
                .put_object(
                    "bucket",
                    &format!("bodies/ns/{id}"),
                    Bytes::from(body),
                    Some("application/json"),
                )
                .await?;
        }
        Ok(())
    }

    async fn append_upserts(
        store: &Arc<dyn ObjectStore>,
        ids: &[&str],
    ) -> Result<(), DripstoneError> {
        let mut config = wal_config();
        config.max_batch_size = 1;
        let writer = WalWriter::new(Arc::clone(store), config, Some("ns".to_string()));
        for id in ids {
            writer.append(WalOperation::Upsert(wal_record(id))).await?;
        }
        Ok(())
    }

    #[tokio::test]
    async fn compactor_deletes_prefix_when_snapshot_and_bodies_match() {
        let store: Arc<dyn ObjectStore> = MemoryObjectStore::shared();
        append_upserts(&store, &["a", "b", "c"])
            .await
            .expect("append");
        put_membership(&store, &["a", "b", "c"])
            .await
            .expect("durable membership");
        let compactor = WalCompactor::new(Arc::clone(&store), wal_config(), Some("ns".to_string()));
        assert_eq!(compactor.compact(1).await.expect("compact"), 2);
        let reader = WalReader::new(store, wal_config(), Some("ns".to_string()));
        let files = reader.list_wal_files().await.expect("list");
        assert_eq!(files.len(), 1);
        let mut ids = Vec::new();
        reader
            .replay(|entry| {
                if let WalOperation::Upsert(record) = entry.operation {
                    ids.push(record.id.to_string());
                }
                Ok(())
            })
            .await
            .expect("replay");
        assert_eq!(ids, vec!["c".to_string()]);
    }

    #[tokio::test]
    async fn compactor_keeps_segments_when_a_live_body_is_missing() {
        let store: Arc<dyn ObjectStore> = MemoryObjectStore::shared();
        append_upserts(&store, &["a", "b", "c"])
            .await
            .expect("append");
        put_membership(&store, &["a", "b", "c"])
            .await
            .expect("membership");
        store
            .delete_object("bucket", "bodies/ns/a")
            .await
            .expect("drop body a");
        let compactor = WalCompactor::new(Arc::clone(&store), wal_config(), Some("ns".to_string()));
        let err = compactor.compact(1).await.expect_err("missing body");
        assert!(err.to_string().contains("refuses to delete"));
        let reader = WalReader::new(store, wal_config(), Some("ns".to_string()));
        assert_eq!(reader.list_wal_files().await.expect("list").len(), 3);
    }

    #[tokio::test]
    async fn compactor_drops_an_upsert_whose_id_was_deleted() {
        let store: Arc<dyn ObjectStore> = MemoryObjectStore::shared();
        let mut config = wal_config();
        config.max_batch_size = 1;
        let writer = WalWriter::new(Arc::clone(&store), config.clone(), Some("ns".to_string()));
        writer
            .append(WalOperation::Upsert(wal_record("a")))
            .await
            .expect("upsert a");
        writer
            .append(WalOperation::Upsert(wal_record("b")))
            .await
            .expect("upsert b");
        writer
            .append(WalOperation::Delete {
                id: "a".to_string(),
                namespace: Some("ns".to_string()),
            })
            .await
            .expect("delete a");
        put_membership(&store, &["b"]).await.expect("snapshot is b");
        let compactor =
            WalCompactor::new(Arc::clone(&store), config.clone(), Some("ns".to_string()));
        assert_eq!(compactor.compact(1).await.expect("compact"), 2);
        let reader = WalReader::new(store, config, Some("ns".to_string()));
        assert_eq!(reader.list_wal_files().await.expect("list").len(), 1);
    }

    #[tokio::test]
    async fn compact_wal_reload_searches_bodies_without_duplicating_membership() {
        let store: Arc<dyn ObjectStore> = MemoryObjectStore::shared();
        let mut config = cold_config(None);
        config.refresh_shared_wal = Some(false);
        config.wal_max_batch_size = 1;
        let writer = ColdForgeBackend::with_object_store(Arc::clone(&store), config.clone());
        writer
            .upsert(&[
                chunk("alpha", "demo", vec![1.0, 0.0, 0.0, 0.0]),
                chunk("beta", "demo", vec![0.0, 1.0, 0.0, 0.0]),
            ])
            .await
            .expect("upsert");
        let deleted = writer.compact_wal(Some("demo"), 0).await.expect("compact");
        assert!(
            deleted >= 1,
            "segments should be deleted after the snapshot"
        );
        let reader = ColdForgeBackend::with_object_store(store, config);
        assert_eq!(reader.count(Some("demo")).await.expect("count"), 2);
        let hits = reader
            .search(&query(vec![1.0, 0.0, 0.0, 0.0], "demo"))
            .await
            .expect("search");
        assert_eq!(hits.len(), 2);
        let fetched = reader
            .get("alpha#0", Some("demo"))
            .await
            .expect("get")
            .expect("body");
        assert_eq!(fetched.id, "alpha#0");
    }
}

#[cfg(feature = "redis-backend")]
mod redis_live {
    use super::chunk;
    use dripstone::backend::redis::RedisBackend;
    use dripstone::traits::Backend;

    #[ignore = "requires REDIS_URL; run with --ignored"]
    #[tokio::test]
    async fn scan_round_trip_keeps_doc_prefixed_ids() {
        let url =
            std::env::var("REDIS_URL").expect("REDIS_URL must be set when this ignored test runs");
        let namespace = format!(
            "dist-{}",
            time::OffsetDateTime::now_utc().unix_timestamp_nanos()
        );
        let backend = RedisBackend::with_prefix(&url, "dripstone:dist:")
            .await
            .expect("redis");
        let row = chunk("doc:1", &namespace, vec![0.2, 0.4, 0.6, 0.8]);
        backend
            .upsert(std::slice::from_ref(&row))
            .await
            .expect("upsert");
        let fetched = backend
            .get("doc:1#0", Some(&namespace))
            .await
            .expect("get")
            .expect("id doc:1 is an embedding, not a doc-set key");
        assert_eq!(fetched.id, "doc:1#0");
        assert_eq!(backend.count(Some(&namespace)).await.expect("count"), 1);
        backend
            .delete("doc:1#0", Some(&namespace))
            .await
            .expect("delete");
        assert_eq!(
            backend
                .delete_document("doc:1", Some(&namespace))
                .await
                .expect("delete_document"),
            0,
            "delete must drop the id from the document set"
        );
        backend.delete_namespace(&namespace).await.expect("cleanup");
    }
}
