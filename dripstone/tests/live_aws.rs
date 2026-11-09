//! Opt-in ColdForge live scene.
//!
//! One process, one backend, one TSV and one markdown table.
//! Scene: dimension 128, cosine, namespace `live`, N in {50, 200, 1000}.
//! Cold search is 2 samples on AWS and 50 on MinIO. Warm search is 50.
//! Prefer `--release` for numbers you keep.
//!
//! Skips unless `DRIPSTONE_LIVE_AWS=1` and `COLDFORGE_BUCKET` are set.
//! `DRIPSTONE_LIVE_BACKEND` swaps the backend in this same test binary:
//! `minio` (OBJECT_STORE_ENDPOINT), `s3`, or `s3-redis` (`REDIS_URL` and the
//! `redis-backend` feature). Valkey, S3 Vectors, and DynamoDB are not rows.
//!
//! `wal_refresh_interval_ms` is 1000, the product default. `0` would list the
//! WAL on every refreshing read, including get, before Redis. Each timed get
//! drops the in-process body map and the LRU after one priming read.

#![cfg(feature = "coldforge-backend")]

use dripstone::backend::coldforge::live::{self, LiveAwsPlan};

#[tokio::test]
async fn coldforge_live_scene() {
    match live::plan_from_env().expect("plan") {
        LiveAwsPlan::Skip { reason } => {
            eprintln!("skip live scene: {reason}");
        }
        LiveAwsPlan::Run(options) => {
            let report = live::run(&options).await.expect("live scene");
            println!("{}", report.markdown);
            let header = report.tsv.lines().next().expect("tsv header");
            assert!(header.contains("upsert_flush_ms"));
            assert!(header.contains("search_cold_p99_ms"));
            assert!(header.contains("head_bucket_p50_ms"));
            assert!(header.contains("recall_at_k"));
            assert_eq!(report.rows.len(), 3, "scene is N=50,200,1000");
            assert!(
                report.rows.iter().all(|row| row.hits > 0),
                "warm search returned no hits"
            );
        }
    }
}
