//! MinIO / S3-compatible ObjectStore coverage for ColdForge WAL-sharded backends.
//!
//! Soft-skips unless an S3-compatible endpoint is configured. Start MinIO or
//! another S3-compatible service on `:9000`, then set the env vars below.
//!
//! Env (any of these unlocks the suite):
//! - `OBJECT_STORE_ENDPOINT` or `DRIPSTONE_MINIO_ENDPOINT` (default probe: `http://127.0.0.1:9000`)
//! - `DRIPSTONE_MINIO_ACCESS_KEY` / `DRIPSTONE_MINIO_SECRET_KEY` (default `minioadmin`)
//! - `DRIPSTONE_MINIO_BUCKET` (default `example-wal`)
//! - `DRIPSTONE_MINIO_FORCE=1` to attempt even when health check fails
//!
//! ```text
//! cargo test --test minio_coldforge_wal_tests --features "coldforge-backend,memory-backend,tokio" -- --test-threads=1
//! ```

#![cfg(all(
    feature = "coldforge-backend",
    feature = "memory-backend",
    feature = "tokio",
    feature = "s3-backend"
))]

macro_rules! outputln {
    ($($arg:tt)*) => {{
        let mut output = std::io::stdout().lock();
        let _ = std::io::Write::write_fmt(&mut output, format_args!($($arg)*));
        let _ = std::io::Write::write_all(&mut output, b"\n");
    }};
}

use std::env;
use std::sync::Arc;
use std::time::Instant;

use aws_sdk_s3::Client;
use aws_sdk_s3::config::{BehaviorVersion, Credentials as AwsCredentials, Region};
use dripstone::storage::{ObjectStore, ObjectStoreConfig, create_object_store};
use dripstone::{
    Backend, Chunk, ColdForgeShardOptions, MaxSimOptions, Metric, SearchQuery,
    build_coldforge_wal_sharded, multi_vectors_from_owned, pooled_vector_from_multi,
};
use serde_json::json;
use time::OffsetDateTime;

struct MinioEnv {
    endpoint: String,
    access_key: String,
    secret_key: String,
    bucket: String,
}

fn minio_env() -> MinioEnv {
    let endpoint = env::var("OBJECT_STORE_ENDPOINT")
        .or_else(|_| env::var("DRIPSTONE_MINIO_ENDPOINT"))
        .expect(
            "OBJECT_STORE_ENDPOINT or DRIPSTONE_MINIO_ENDPOINT must be set when this ignored test runs",
        );
    MinioEnv {
        endpoint,
        access_key: env::var("DRIPSTONE_MINIO_ACCESS_KEY")
            .or_else(|_| env::var("AWS_ACCESS_KEY_ID"))
            .unwrap_or_else(|_| "minioadmin".to_string()),
        secret_key: env::var("DRIPSTONE_MINIO_SECRET_KEY")
            .or_else(|_| env::var("AWS_SECRET_ACCESS_KEY"))
            .unwrap_or_else(|_| "minioadmin".to_string()),
        bucket: env::var("DRIPSTONE_MINIO_BUCKET")
            .unwrap_or_else(|_| "example-wal-bucket".to_string()),
    }
}

async fn ensure_bucket(env: &MinioEnv) -> Result<(), String> {
    let creds = AwsCredentials::new(
        &env.access_key,
        &env.secret_key,
        None,
        None,
        "dripstone-minio-test",
    );
    let conf = aws_sdk_s3::config::Builder::new()
        .behavior_version(BehaviorVersion::latest())
        .region(Region::new("us-east-1"))
        .endpoint_url(&env.endpoint)
        .force_path_style(true)
        .credentials_provider(creds)
        .build();
    let client = Client::from_conf(conf);

    match client.head_bucket().bucket(&env.bucket).send().await {
        Ok(_) => Ok(()),
        Err(_) => client
            .create_bucket()
            .bucket(&env.bucket)
            .send()
            .await
            .map(|_| ())
            .map_err(|e| format!("create_bucket {}: {e}", env.bucket)),
    }
}

async fn connect_store(env: &MinioEnv) -> Result<Arc<dyn ObjectStore>, String> {
    let cfg = ObjectStoreConfig::generic_s3(
        env.endpoint.clone(),
        "us-east-1".to_string(),
        true,
        Some(env.access_key.clone()),
        Some(env.secret_key.clone()),
        None,
    );
    create_object_store(cfg)
        .await
        .map_err(|e| format!("create_object_store: {e}"))
}

fn make_chunk(doc: &str, dim: usize, namespace: Option<&str>) -> Chunk {
    let mut vector = vec![0.0_f32; dim];
    for (i, b) in doc.bytes().enumerate() {
        vector[i % dim] += f32::from(b) / 255.0;
    }
    let norm = vector.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-6);
    for v in &mut vector {
        *v /= norm;
    }
    Chunk::new(
        doc,
        0,
        format!("content-{doc}"),
        vector,
        0,
        8,
        1,
        json!({ "doc": doc }),
        namespace.map(str::to_string),
    )
}

fn make_multi_chunk(doc: &str, tokens: Vec<Vec<f32>>, namespace: Option<&str>) -> Chunk {
    let mv = multi_vectors_from_owned(tokens);
    let pooled = pooled_vector_from_multi(&mv);
    Chunk::new(
        doc,
        0,
        format!("content-{doc}"),
        pooled,
        0,
        8,
        1,
        json!({ "doc": doc }),
        namespace.map(str::to_string),
    )
    .with_multi_vectors(mv)
}

#[ignore = "requires OBJECT_STORE_ENDPOINT or DRIPSTONE_MINIO_ENDPOINT; run with --ignored"]
#[tokio::test]
async fn minio_coldforge_wal_sharded_upsert_search_roundtrip() {
    let env = minio_env();
    ensure_bucket(&env).await.expect("minio bucket");
    let store = connect_store(&env).await.expect("minio store");

    // Smoke put to confirm credentials/bucket before building shards.
    let probe_key = format!(
        "probes/dripstone-{}.txt",
        OffsetDateTime::now_utc().unix_timestamp_nanos()
    );
    store
        .put_object(
            &env.bucket,
            &probe_key,
            bytes::Bytes::from_static(b"ok"),
            Some("text/plain"),
        )
        .await
        .expect("minio put_object");
    let _ = store.delete_object(&env.bucket, &probe_key).await;

    const DIM: usize = 32;
    const N: usize = 80;
    let ns = format!(
        "minio-wal-{}",
        OffsetDateTime::now_utc().unix_timestamp_nanos()
    );

    let backend = build_coldforge_wal_sharded(
        store,
        ColdForgeShardOptions {
            shard_count: 4,
            vector_dimension: DIM,
            num_centroids: 8,
            bucket: env.bucket.clone(),
            shared_wal_prefix: format!("namespaces/{ns}/wal/"),
            memory_hot_tier: true,
        },
    )
    .expect("minio coldforge build");

    let chunks: Vec<_> = (0..N)
        .map(|i| make_chunk(&format!("doc-{i}"), DIM, Some(&ns)))
        .collect();

    let start = Instant::now();
    for window in chunks.chunks(40) {
        backend.upsert(window).await.expect("upsert");
    }
    backend.flush().await.expect("flush");
    let write_elapsed = start.elapsed();

    assert_eq!(backend.count(Some(&ns)).await.expect("count"), N);
    let shard_counts = backend.shard_counts(Some(&ns)).await.expect("shard_counts");
    assert_eq!(shard_counts.iter().sum::<usize>(), N);
    assert!(
        shard_counts.iter().all(|&c| c > 0),
        "expected balanced shards: {shard_counts:?}"
    );

    let probe = &chunks[17];
    let query = SearchQuery::new(probe.vector.clone())
        .limit(3)
        .metric(Metric::Cosine)
        .namespace(ns.clone());
    let search_start = Instant::now();
    let hits = backend.search(&query).await.expect("search");
    let search_elapsed = search_start.elapsed();
    assert_eq!(hits[0].chunk.id, probe.id);

    outputln!(
        "MinIO ColdForge WAL-sharded: N={N} write={write_elapsed:?} search={search_elapsed:?} shards={shard_counts:?}"
    );

    // Keep runs under interactive budgets on local MinIO.
    assert!(
        write_elapsed.as_secs_f64() < 60.0,
        "upsert too slow on MinIO: {write_elapsed:?}"
    );
    assert!(
        search_elapsed.as_millis() < 5_000,
        "search too slow on MinIO: {search_elapsed:?}"
    );

    backend.delete_namespace(&ns).await.expect("cleanup");
}

#[ignore = "requires OBJECT_STORE_ENDPOINT or DRIPSTONE_MINIO_ENDPOINT; run with --ignored"]
#[tokio::test]
async fn minio_coldforge_wal_sharded_maxsim_persists_multi_vectors() {
    let env = minio_env();
    ensure_bucket(&env).await.expect("minio bucket");
    let store = connect_store(&env).await.expect("minio store");

    const DIM: usize = 16;
    const TOKENS: usize = 4;
    let ns = format!(
        "minio-maxsim-{}",
        OffsetDateTime::now_utc().unix_timestamp_nanos()
    );

    let backend = build_coldforge_wal_sharded(
        store,
        ColdForgeShardOptions {
            shard_count: 2,
            vector_dimension: DIM,
            num_centroids: 4,
            bucket: env.bucket.clone(),
            shared_wal_prefix: format!("namespaces/{ns}/wal/"),
            memory_hot_tier: true,
        },
    )
    .expect("minio coldforge build");

    let mut chunks = Vec::new();
    for i in 0..40 {
        let tokens = (0..TOKENS)
            .map(|t| {
                let mut v = vec![0.0_f32; DIM];
                v[(DIM / 2 + (i + t) % (DIM / 2)) % DIM] = 1.0;
                v
            })
            .collect();
        chunks.push(make_multi_chunk(&format!("doc-{i}"), tokens, Some(&ns)));
    }
    chunks[7] = make_multi_chunk(
        "doc-7",
        (0..TOKENS)
            .map(|t| {
                let mut v = vec![0.0_f32; DIM];
                v[t % DIM] = 1.0;
                v
            })
            .collect(),
        Some(&ns),
    );

    backend.upsert(&chunks).await.expect("upsert");
    backend.flush().await.expect("flush");

    let q_tokens = multi_vectors_from_owned(
        (0..TOKENS)
            .map(|t| {
                let mut v = vec![0.0_f32; DIM];
                v[t % DIM] = 1.0;
                v
            })
            .collect(),
    );
    let query = SearchQuery::new(pooled_vector_from_multi(&q_tokens))
        .limit(1)
        .metric(Metric::DotProduct)
        .namespace(ns.clone())
        .with_multi_vectors(q_tokens)
        .with_maxsim(MaxSimOptions::new(Metric::DotProduct).overfetch(8));

    let hits = backend.search(&query).await.expect("search");
    assert_eq!(hits[0].chunk.id, "doc-7#0");
    assert!(
        hits[0].chunk.multi_vectors.is_some(),
        "multi_vectors must survive MinIO WAL / hot-tier path"
    );

    backend.delete_namespace(&ns).await.expect("cleanup");
}
