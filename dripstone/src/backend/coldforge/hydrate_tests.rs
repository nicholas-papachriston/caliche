use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use async_trait::async_trait;
use bytes::Bytes;
use serde_json::json;

use super::live::{self, LiveAwsOptions, LiveBackend};
use super::{ColdForgeBackend, ColdForgeConfig};
use crate::backend::chunk_envelope::KEY_CONTENT;
use crate::backend::coldforge::body::{BODY_KEY_SEGMENT, ChunkBodyStore};
use crate::backend::s3::layout::ColdForgeSegment;
use crate::core::{Chunk, Metric, SearchQuery};
use crate::errors::DripstoneError;
use crate::storage::{MemoryObjectStore, ObjectInfo, ObjectMetadata, ObjectStore};
use crate::traits::Backend;

fn config() -> ColdForgeConfig {
    ColdForgeConfig {
        bucket: "bucket".to_string(),
        key_prefix: "cf/".to_string(),
        region: "us-east-1".to_string(),
        cache_size_bytes: 1,
        num_centroids: 1,
        wal_batch_interval_ms: 60_000,
        wal_max_batch_size: 100,
        enable_compression: false,
        vector_dimension: 4,
        hot_tier: None,
        refresh_shared_wal: Some(false),
        wal_refresh_interval_ms: 0,
        access_key_id: None,
        secret_access_key: None,
        session_token: None,
        object_store_config: None,
        body_put_concurrency: super::DEFAULT_BODY_PUT_CONCURRENCY,
    }
}

fn chunk(doc_id: &str, chunk_index: usize, content: &str, vector: Vec<f32>) -> Chunk {
    Chunk::new(
        doc_id,
        chunk_index,
        content,
        vector,
        0,
        content.len(),
        2,
        json!({ "kind": "note" }),
        Some("demo".to_string()),
    )
}

#[test]
fn chunk_body_key_is_namespace_and_chunk_id() {
    let key = ChunkBodyStore::object_key("cf/", "demo", "doc#0");
    assert_eq!(key, format!("cf/{BODY_KEY_SEGMENT}/demo/doc#0"));
    assert!(key.ends_with("doc#0"));
    assert!(!key.contains("cluster"));
}

#[tokio::test]
async fn search_get_and_list_agree_after_cache_eviction() {
    let store: Arc<dyn ObjectStore> = MemoryObjectStore::shared();
    let backend = ColdForgeBackend::with_object_store(Arc::clone(&store), config());
    let first = chunk("doc", 0, "alpha body", vec![1.0, 0.0, 0.0, 0.0]);
    let second = chunk("doc", 1, "beta body", vec![0.0, 1.0, 0.0, 0.0]);
    let other = chunk("other", 0, "gamma body", vec![0.0, 0.0, 1.0, 0.0]);
    backend
        .upsert(&[first.clone(), second.clone(), other.clone()])
        .await
        .expect("upsert");

    let fetched = backend
        .get(&first.id, Some("demo"))
        .await
        .expect("get")
        .expect("evicted cache entry is still in the body store");
    assert_eq!(fetched.content, first.content);
    assert_eq!(fetched.doc_id, "doc");

    let hits = backend
        .search(
            &SearchQuery::new(vec![1.0, 0.0, 0.0, 0.0])
                .metric(Metric::Cosine)
                .limit(10)
                .namespace("demo"),
        )
        .await
        .expect("search");
    let hit = hits
        .iter()
        .find(|hit| hit.chunk.id == first.id)
        .expect("search still returns the evicted id");
    assert_eq!(hit.chunk.content, fetched.content);
    assert_eq!(hit.chunk.metadata, fetched.metadata);

    let listed = backend.list(Some("demo"), None).await.expect("list");
    assert_eq!(listed.len(), 3);
    assert!(listed.iter().any(|chunk| chunk.content == "alpha body"));
    assert!(listed.iter().any(|chunk| chunk.content == "beta body"));
    assert!(listed.iter().any(|chunk| chunk.content == "gamma body"));

    let index = backend.get_index(Some("demo")).await;
    let idx = index.read().await;
    for id in [&first.id, &second.id, &other.id] {
        let stored = idx.get_record(id).expect("scoring record");
        assert!(
            stored
                .metadata
                .as_ref()
                .is_none_or(|metadata| metadata.get(KEY_CONTENT).is_none()),
            "index record embedded content for {id}"
        );
        assert_eq!(stored.id.as_ref(), id.as_str());
    }
    drop(idx);

    let prefix = format!("cf/{BODY_KEY_SEGMENT}/demo/");
    let objects = store
        .list_objects("bucket", Some(&prefix))
        .await
        .expect("list bodies");
    assert_eq!(objects.len(), 3);
    for object in &objects {
        assert!(object.key.starts_with(&prefix));
        assert!(
            object.key.ends_with("doc#0")
                || object.key.ends_with("doc#1")
                || object.key.ends_with("other#0")
        );
        assert!(!object.key.contains("cluster"));
    }

    let removed = backend
        .delete_document("doc", Some("demo"))
        .await
        .expect("delete document");
    assert_eq!(removed, 2);
    let listed = backend
        .list(Some("demo"), None)
        .await
        .expect("list after delete");
    assert_eq!(listed.len(), 1);
    assert_eq!(listed[0].content, "gamma body");
    assert!(
        backend
            .get(&first.id, Some("demo"))
            .await
            .expect("get deleted")
            .is_none()
    );
    let hits = backend
        .search(
            &SearchQuery::new(vec![1.0, 0.0, 0.0, 0.0])
                .metric(Metric::Cosine)
                .limit(10)
                .namespace("demo"),
        )
        .await
        .expect("search after delete");
    assert!(hits.iter().all(|hit| hit.chunk.doc_id != "doc"));

    let objects = store
        .list_objects("bucket", Some(&prefix))
        .await
        .expect("list bodies after delete");
    assert_eq!(objects.len(), 1);
    assert!(objects[0].key.ends_with("other#0"));
}

#[tokio::test]
async fn cold_open_search_and_get_do_not_rewrite_bodies() {
    let probe = Arc::new(ProbeStore::new(Duration::ZERO));
    let store: Arc<dyn ObjectStore> = Arc::clone(&probe) as Arc<dyn ObjectStore>;
    let writer = ColdForgeBackend::with_object_store(Arc::clone(&store), config());
    let chunks = [
        chunk("doc", 0, "alpha body", vec![1.0, 0.0, 0.0, 0.0]),
        chunk("doc", 1, "beta body", vec![0.0, 1.0, 0.0, 0.0]),
        chunk("other", 0, "gamma body", vec![0.0, 0.0, 1.0, 0.0]),
    ];
    writer.upsert(&chunks).await.expect("upsert");
    writer.flush().await.expect("flush");
    let before = probe.snapshot_bodies("bucket", "cf/bodies/").await;
    assert_eq!(before.len(), 3);
    probe.reset();

    let reader = ColdForgeBackend::with_object_store(Arc::clone(&store), config());
    let hits = reader
        .search(
            &SearchQuery::new(vec![1.0, 0.0, 0.0, 0.0])
                .metric(Metric::Cosine)
                .limit(10)
                .namespace("demo"),
        )
        .await
        .expect("cold search");
    assert!(hits.iter().any(|hit| hit.chunk.content == "alpha body"));
    assert!(hits.iter().any(|hit| hit.chunk.content == "beta body"));
    assert!(hits.iter().any(|hit| hit.chunk.content == "gamma body"));
    let fetched = reader
        .get(&chunks[0].id, Some("demo"))
        .await
        .expect("get")
        .expect("body");
    assert_eq!(fetched.content, "alpha body");
    assert_eq!(probe.body_puts(), 0, "cold open must not PUT chunk bodies");
    assert_eq!(
        probe.body_gets(),
        0,
        "WAL replay already holds the envelopes"
    );
    assert_eq!(probe.snapshot_bodies("bucket", "cf/bodies/").await, before);
}

#[tokio::test]
async fn compacted_cold_open_fetches_bodies_in_parallel_without_rewriting() {
    let probe = Arc::new(ProbeStore::new(Duration::from_millis(40)));
    let store: Arc<dyn ObjectStore> = Arc::clone(&probe) as Arc<dyn ObjectStore>;
    let mut cold_config = config();
    cold_config.wal_max_batch_size = 100;
    let writer = ColdForgeBackend::with_object_store(Arc::clone(&store), cold_config.clone());
    let mut chunks = Vec::with_capacity(8);
    for index in 0..8 {
        let mut vector = vec![0.0; 4];
        vector[index % 4] = 1.0;
        let bump = u32::try_from(index).map_or(0.0, |value| (value as f32) * 0.01);
        vector[0] += bump;
        chunks.push(chunk(
            &format!("doc{index}"),
            0,
            &format!("body-{index}"),
            vector,
        ));
    }
    writer.upsert(&chunks).await.expect("upsert");
    let deleted = writer.compact_wal(Some("demo"), 0).await.expect("compact");
    assert!(deleted >= 1, "snapshot should allow the segment to drop");
    let before = probe.snapshot_bodies("bucket", "cf/bodies/").await;
    assert_eq!(before.len(), chunks.len());
    probe.reset();

    let reader = ColdForgeBackend::with_object_store(store, cold_config);
    let started = std::time::Instant::now();
    let hits = reader
        .search(
            &SearchQuery::new(vec![1.0, 0.0, 0.0, 0.0])
                .metric(Metric::Cosine)
                .limit(10)
                .namespace("demo"),
        )
        .await
        .expect("cold search after compact");
    let elapsed = started.elapsed();
    assert_eq!(hits.len(), chunks.len());
    for index in 0..chunks.len() {
        let content = format!("body-{index}");
        assert!(
            hits.iter().any(|hit| hit.chunk.content == content),
            "missing {content}"
        );
    }
    assert_eq!(probe.body_puts(), 0, "hydrate must not PUT chunk bodies");
    assert_eq!(probe.body_gets(), chunks.len());
    assert!(
        probe.peak_in_flight() >= 2,
        "body GETs stayed serial (peak {})",
        probe.peak_in_flight()
    );
    assert!(
        elapsed < Duration::from_secs(2),
        "parallel hydrate took {elapsed:?}"
    );
    assert_eq!(probe.snapshot_bodies("bucket", "cf/bodies/").await, before);
}

#[tokio::test]
async fn live_get_samples_each_hydrate_from_object_storage() {
    let probe = Arc::new(ProbeStore::new(Duration::ZERO));
    let store: Arc<dyn ObjectStore> = Arc::clone(&probe) as Arc<dyn ObjectStore>;
    let options = LiveAwsOptions {
        backend: LiveBackend::S3,
        bucket: "bucket".to_string(),
        region: "us-east-1".to_string(),
        key_prefix: "coldforge-live/test/".to_string(),
        redis_url: None,
    };
    let prefix = "coldforge-live/test/n8/";
    let n = 8;
    let writer = live::open(
        &store,
        &options,
        prefix,
        8 * 1024 * 1024,
        None,
        live::DIMENSION,
        super::DEFAULT_BODY_PUT_CONCURRENCY,
    );
    writer
        .upsert(&live::sample_chunks(n))
        .await
        .expect("upsert");
    writer.flush().await.expect("flush");
    probe.reset();

    let ids = live::hydrate_ids(n);
    assert!(!ids.is_empty());
    let primer = format!("row-{}#0", n.saturating_sub(1));
    let samples = live::time_gets(
        &store,
        &options,
        prefix,
        None,
        live::DIMENSION,
        super::DEFAULT_BODY_PUT_CONCURRENCY,
        &primer,
        &ids,
    )
    .await
    .expect("timed gets");
    assert_eq!(samples.len(), ids.len());
    assert_eq!(probe.body_puts(), 0);
    assert_eq!(
        probe.body_gets(),
        ids.len(),
        "each timed get must read the durable body"
    );
}

#[tokio::test]
async fn concurrent_upsert_puts_bodies_in_parallel_and_cold_open_does_not_rewrite() {
    let probe = Arc::new(ProbeStore::with_delays(
        Duration::ZERO,
        Duration::from_millis(40),
    ));
    let store: Arc<dyn ObjectStore> = Arc::clone(&probe) as Arc<dyn ObjectStore>;
    let mut cold_config = config();
    cold_config.body_put_concurrency = 32;
    let writer = ColdForgeBackend::with_object_store(Arc::clone(&store), cold_config.clone());
    let mut chunks = Vec::with_capacity(8);
    for index in 0..8 {
        let mut vector = vec![0.0; 4];
        vector[index % 4] = 1.0;
        let bump = u32::try_from(index).map_or(0.0, |value| (value as f32) * 0.01);
        vector[0] += bump;
        chunks.push(chunk(
            &format!("doc{index}"),
            0,
            &format!("body-{index}"),
            vector,
        ));
    }
    let started = std::time::Instant::now();
    writer.upsert(&chunks).await.expect("upsert");
    let elapsed = started.elapsed();
    writer.flush().await.expect("flush");
    assert_eq!(probe.body_puts(), chunks.len());
    assert!(
        probe.put_peak_in_flight() >= 2,
        "body PUTs stayed serial (peak {})",
        probe.put_peak_in_flight()
    );
    assert!(
        elapsed < Duration::from_secs(2),
        "parallel upsert took {elapsed:?}"
    );
    let hits = writer
        .search(
            &SearchQuery::new(vec![1.0, 0.0, 0.0, 0.0])
                .metric(Metric::Cosine)
                .limit(10)
                .namespace("demo"),
        )
        .await
        .expect("search");
    assert_eq!(hits.len(), chunks.len());
    for index in 0..chunks.len() {
        let content = format!("body-{index}");
        assert!(
            hits.iter().any(|hit| hit.chunk.content == content),
            "missing {content}"
        );
    }
    let before = probe.snapshot_bodies("bucket", "cf/bodies/").await;
    assert_eq!(before.len(), chunks.len());
    probe.reset();

    let reader = ColdForgeBackend::with_object_store(store, cold_config);
    let cold_hits = reader
        .search(
            &SearchQuery::new(vec![1.0, 0.0, 0.0, 0.0])
                .metric(Metric::Cosine)
                .limit(10)
                .namespace("demo"),
        )
        .await
        .expect("cold search");
    assert_eq!(cold_hits.len(), chunks.len());
    assert_eq!(probe.body_puts(), 0, "cold open must not PUT chunk bodies");
    assert_eq!(probe.snapshot_bodies("bucket", "cf/bodies/").await, before);
}

fn is_chunk_body_key(key: &str) -> bool {
    let segment = ColdForgeSegment::Body.as_str();
    key.split('/').any(|part| part == segment)
}

struct ProbeStore {
    inner: Arc<MemoryObjectStore>,
    body_puts: AtomicUsize,
    body_gets: AtomicUsize,
    in_flight: AtomicUsize,
    peak_in_flight: AtomicUsize,
    put_in_flight: AtomicUsize,
    put_peak_in_flight: AtomicUsize,
    get_delay: Duration,
    put_delay: Duration,
}

impl ProbeStore {
    fn new(get_delay: Duration) -> Self {
        Self::with_delays(get_delay, Duration::ZERO)
    }

    fn with_delays(get_delay: Duration, put_delay: Duration) -> Self {
        Self {
            inner: MemoryObjectStore::shared(),
            body_puts: AtomicUsize::new(0),
            body_gets: AtomicUsize::new(0),
            in_flight: AtomicUsize::new(0),
            peak_in_flight: AtomicUsize::new(0),
            put_in_flight: AtomicUsize::new(0),
            put_peak_in_flight: AtomicUsize::new(0),
            get_delay,
            put_delay,
        }
    }

    fn reset(&self) {
        self.body_puts.store(0, Ordering::SeqCst);
        self.body_gets.store(0, Ordering::SeqCst);
        self.in_flight.store(0, Ordering::SeqCst);
        self.peak_in_flight.store(0, Ordering::SeqCst);
        self.put_in_flight.store(0, Ordering::SeqCst);
        self.put_peak_in_flight.store(0, Ordering::SeqCst);
    }

    fn body_puts(&self) -> usize {
        self.body_puts.load(Ordering::SeqCst)
    }

    fn body_gets(&self) -> usize {
        self.body_gets.load(Ordering::SeqCst)
    }

    fn peak_in_flight(&self) -> usize {
        self.peak_in_flight.load(Ordering::SeqCst)
    }

    fn put_peak_in_flight(&self) -> usize {
        self.put_peak_in_flight.load(Ordering::SeqCst)
    }

    async fn snapshot_bodies(&self, bucket: &str, prefix: &str) -> Vec<(String, Bytes)> {
        let mut listed = self
            .inner
            .list_objects(bucket, Some(prefix))
            .await
            .expect("list bodies");
        listed.sort_by(|left, right| left.key.cmp(&right.key));
        let mut out = Vec::with_capacity(listed.len());
        for object in listed {
            let bytes = self
                .inner
                .get_object(bucket, &object.key)
                .await
                .expect("read body");
            out.push((object.key, bytes));
        }
        out
    }
}

#[async_trait]
impl ObjectStore for ProbeStore {
    async fn get_object(&self, bucket: &str, key: &str) -> Result<Bytes, DripstoneError> {
        let body = is_chunk_body_key(key);
        if body {
            self.body_gets.fetch_add(1, Ordering::SeqCst);
            let now = self.in_flight.fetch_add(1, Ordering::SeqCst) + 1;
            self.peak_in_flight.fetch_max(now, Ordering::SeqCst);
            if !self.get_delay.is_zero() {
                tokio::time::sleep(self.get_delay).await;
            }
        }
        let result = self.inner.get_object(bucket, key).await;
        if body {
            self.in_flight.fetch_sub(1, Ordering::SeqCst);
        }
        result
    }

    async fn put_object(
        &self,
        bucket: &str,
        key: &str,
        data: Bytes,
        content_type: Option<&str>,
    ) -> Result<(), DripstoneError> {
        let body = is_chunk_body_key(key);
        if body {
            self.body_puts.fetch_add(1, Ordering::SeqCst);
            let now = self.put_in_flight.fetch_add(1, Ordering::SeqCst) + 1;
            self.put_peak_in_flight.fetch_max(now, Ordering::SeqCst);
            if !self.put_delay.is_zero() {
                tokio::time::sleep(self.put_delay).await;
            }
        }
        let result = self.inner.put_object(bucket, key, data, content_type).await;
        if body {
            self.put_in_flight.fetch_sub(1, Ordering::SeqCst);
        }
        result
    }

    async fn delete_object(&self, bucket: &str, key: &str) -> Result<(), DripstoneError> {
        self.inner.delete_object(bucket, key).await
    }

    async fn head_object(&self, bucket: &str, key: &str) -> Result<ObjectMetadata, DripstoneError> {
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
