//! One ColdForge scene, three backends, one table.
//!
//! Scene: dimension 128, cosine, namespace `live`, N in {50, 200, 1000}.
//! Cold search is 2 samples on AWS (`s3`, `s3-redis`) and 50 on MinIO. Warm
//! search is 50 samples on the last cold backend. Prints one markdown table
//! and one TSV from this process. Use `--release` before comparing backends.
//!
//! `wal_refresh_interval_ms` is 1000, the product default. `0` lists the
//! object-store WAL on every refreshing read, including get, before Redis.
//! This scene does not use `0`. Each timed get drops the in-process body map
//! and the LRU, so S3 samples measure object hydrate and S3+Redis samples
//! measure the hot tier when it still holds the id.
//!
//! Same binary. Swap `DRIPSTONE_LIVE_BACKEND`:
//!
//! ```bash
//! # 1. MinIO local floor (compose endpoint)
//! DRIPSTONE_LIVE_AWS=1 DRIPSTONE_LIVE_BACKEND=minio \
//! OBJECT_STORE_ENDPOINT=http://127.0.0.1:9000 \
//! COLDFORGE_BUCKET=example-wal \
//! cargo run --release --example coldforge_live_aws_example \
//!   --features coldforge-backend,redis-backend
//!
//! # 2. AWS S3 cold-only
//! AWS_PROFILE=example AWS_REGION=us-east-1 \
//! DRIPSTONE_LIVE_AWS=1 DRIPSTONE_LIVE_BACKEND=s3 \
//! COLDFORGE_BUCKET=example-bucket \
//! cargo run --release --example coldforge_live_aws_example \
//!   --features coldforge-backend,redis-backend
//!
//! # 3. AWS S3 + Redis hot
//! AWS_PROFILE=example AWS_REGION=us-east-1 \
//! DRIPSTONE_LIVE_AWS=1 DRIPSTONE_LIVE_BACKEND=s3-redis \
//! COLDFORGE_BUCKET=example-bucket \
//! REDIS_URL=redis://127.0.0.1:6379 \
//! cargo run --release --example coldforge_live_aws_example \
//!   --features coldforge-backend,redis-backend
//! ```
//!
//! Without `DRIPSTONE_LIVE_AWS=1` the process prints SKIP and exits 0.
//! Valkey, S3 Vectors, and DynamoDB are not part of this matrix.
//!
//! Leave `DRIPSTONE_CORPUS` unset for the synthetic microbench (dim 128, cosine,
//! N in {50, 200, 1000}). Set it to ingest a real corpus into the same backend:
//!
//! ```bash
//! DRIPSTONE_LIVE_AWS=1 DRIPSTONE_LIVE_BACKEND=s3 \
//! DRIPSTONE_CORPUS=sift-128-euclidean \
//! DRIPSTONE_CORPUS_TRAIN_MAX=10000 DRIPSTONE_CORPUS_QUERY_MAX=128 \
//! COLDFORGE_BUCKET=example-bucket \
//! cargo run --release --example coldforge_live_aws_example \
//!   --features coldforge-backend,redis-backend
//! ```
//!
//! `nytimes-256-angular`, `sift-128-euclidean`, and `gist-960-euclidean` download
//! through the resumable cache. `cohere-wiki-en` uses English `en/` Parquet
//! shards (not Simple English). Stage that directory and set
//! `DRIPSTONE_CORPUS_PARQUET_DIR` when you do not want the helper to fetch shards.
//! `code` downloads pinned CoIR CoSQA (Python), splits functions with tree-sitter,
//! and upserts hashed cosine features. Cap it the same way:
//!
//! ```bash
//! DRIPSTONE_LIVE_AWS=1 DRIPSTONE_LIVE_BACKEND=minio \
//! DRIPSTONE_CORPUS=code \
//! DRIPSTONE_CORPUS_TRAIN_MAX=10000 DRIPSTONE_CORPUS_QUERY_MAX=128 \
//! OBJECT_STORE_ENDPOINT=http://127.0.0.1:9000 \
//! COLDFORGE_BUCKET=example-wal \
//! cargo run --release --example coldforge_live_aws_example \
//!   --features coldforge-backend,redis-backend
//! ```
//!
//! Stage the CoSQA tree and set `DRIPSTONE_CORPUS_CODE_DIR` to skip that download.
//! `DRIPSTONE_CORPUS_PUT_CONCURRENCY` defaults to 32 body PUTs.

#![cfg(feature = "coldforge-backend")]

macro_rules! outputln {
    ($($arg:tt)*) => {{
        let mut output = std::io::stdout().lock();
        let _ = std::io::Write::write_fmt(&mut output, format_args!($($arg)*));
        let _ = std::io::Write::write_all(&mut output, b"\n");
    }};
}

use dripstone::backend::coldforge::live::{self, LiveAwsPlan};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    match live::plan_from_env()? {
        LiveAwsPlan::Skip { reason } => {
            outputln!("SKIP: {reason}");
            Ok(())
        }
        LiveAwsPlan::Run(options) => {
            let report = live::run(&options).await?;
            outputln!("{}", report.markdown);
            if report.any_empty_search() {
                return Err("warm search returned 0 hits; recall_at_k is 0".into());
            }
            Ok(())
        }
    }
}
