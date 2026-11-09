# Dripstone

Dripstone stores embeddings, runs vector similarity search, and provides graph backends for RAG and knowledge-graph workloads.

## Quick start

The default build includes `local`: an in-process directory store and a hashing embedder. No model download and no extra feature flags.

`open` searches text with dense vectors plus BM25, fused by reciprocal rank. `.dense_only()` keeps dense ranking for one query. `Dripstone::new` and `Dripstone::builder` stay on dense search.

A second `open` of the same directory fails until the first handle is dropped. A snapshot whose version, embedder, or dimension is not supported is left unchanged. A non-finite dense vector or multi-vector is rejected and does not replace the snapshot.

An unreadable `store.json` is quarantined as `store.json.corrupt-<timestamp>` and is not overwritten. A later `open` refuses while that quarantine file remains and no snapshot was restored.

On Windows the directory is flushed through a write handle opened with backup semantics, and the backup file is flushed through a write handle. A replace or backup copy that hits access denied or a sharing violation is retried. If it still fails, the temporary snapshot is removed and the previous `store.json` stays.

```rust
use dripstone::Dripstone;

#[tokio::main]
async fn main() -> Result<(), dripstone::DripstoneError> {
    let store = Dripstone::open("./store")?;
    store.add("alpha", "penguins swim in cold water").await?;
    let hits = store.search("penguins").limit(1).await?;
    println!("{}", hits[0].chunk.doc_id);
    Ok(())
}
```

```bash
cargo run --example quickstart
```

The hashing embedder is for the zero-config path. It does not replace a neural model. Use `embedding-ort` or `embedding-candle` with `Dripstone::new` when you need one. Those features stay out of the default build so it does not pull ONNX, Candle, tokenizers, or an HTTP client.

## Installation

```toml
[dependencies]
dripstone = { git = "https://github.com/nicholas-papachriston/caliche", features = ["backends"] }
```

Enable only the backends you need:

```toml
# Common local and service backends
dripstone = { git = "https://github.com/nicholas-papachriston/caliche", features = ["backends"] }

# AWS managed vectors (S3 / ColdForge / DynamoDB SearchVectors)
dripstone = { git = "https://github.com/nicholas-papachriston/caliche", features = ["backends-aws"] }

# DynamoDB native vector search only
dripstone = { git = "https://github.com/nicholas-papachriston/caliche", features = ["dynamodb-backend"] }

# All backend groups
dripstone = { git = "https://github.com/nicholas-papachriston/caliche", features = ["backends-all"] }

# Embeddings, RAG helpers, inference, and backend groups
dripstone = { git = "https://github.com/nicholas-papachriston/caliche", features = ["full"] }

# Safe model download + catalog only (no Candle/ORT native deps)
dripstone = { git = "https://github.com/nicholas-papachriston/caliche", features = ["model-download"] }

# Lean local generation (prefer a single backend; Hub fetches use curl)
dripstone = { git = "https://github.com/nicholas-papachriston/caliche", features = ["inference-candle"] }
dripstone = { git = "https://github.com/nicholas-papachriston/caliche", features = ["inference-ort"] }

# Optional ureq-based Hugging Face Hub client (otherwise curl via model-download)
dripstone = { git = "https://github.com/nicholas-papachriston/caliche", features = ["inference-candle", "hf-hub"] }

# GGUF via llama.cpp (opt-in; compiles native llama.cpp)
dripstone = { git = "https://github.com/nicholas-papachriston/caliche", features = ["inference-llamacpp"] }
```

`model-download` provides resumable curl downloads with lockfiles, size caps, optional
SHA-256 verification, path sanitization, disk free-space preflight
(`DRIPSTONE_DISK_RESERVE_BYTES`), RAM/VRAM budgets
(`DRIPSTONE_RESERVE_RAM_BYTES`), and a small built-in `loader::catalog`
(including a tiny GGUF smoke model). Override the artifact cache with
`DRIPSTONE_CACHE_DIR`.

Use `loader::CapabilityProbe` / `fit_catalog` (re-exported from `runtime` when
`model-download` is enabled) to decide which catalog models fit the host before
download or load. `RuntimeConfig::resolve_auto` only picks a device from `cfg` /
features — it does not measure memory.

Graph backends are feature-gated:

```toml
dripstone = { git = "https://github.com/nicholas-papachriston/caliche", features = ["graph-age"] }
dripstone = { git = "https://github.com/nicholas-papachriston/caliche", features = ["graph-neo4j"] }
```

## Explicit backend and embedder

The snippets below need more than the default `core` and `local` features. Ingestion with a
local ONNX embedder needs `memory-backend` and `embedding-ort` (that feature
pulls Tokio). A directory passed to `EmbeddingConfig::with_model_repo` must
contain `model.onnx` (or another ONNX filename the loader accepts) and
`tokenizer.json`.

Use `Dripstone` when you want ingestion and query embedding handled together:

```rust
use serde_json::json;
use dripstone::{EmbeddingConfig, MemoryBackend, OrtEmbedder, Dripstone};

#[tokio::main]
async fn main() -> Result<(), dripstone::DripstoneError> {
    let backend = MemoryBackend::new();
    let embedder = OrtEmbedder::new(
        EmbeddingConfig::new().with_model_repo("./models/minilm"),
    )?;
    let vf = Dripstone::new(backend, embedder);

    vf.ingest("doc:1", "Hello, world!")
        .namespace("knowledge-base")
        .metadata(json!({ "title": "Getting Started" }))
        .await?;

    let results = vf
        .search("greeting")
        .namespace("knowledge-base")
        .limit(5)
        .await?;

    println!("{results:?}");
    Ok(())
}
```

Use `Backend` directly when embeddings are already available. This snippet needs
the `memory-backend` feature:

```rust
use serde_json::json;
use std::sync::Arc;
use dripstone::{Backend, Chunk, MemoryBackend, Metric, SearchQuery};

#[tokio::main]
async fn main() -> Result<(), dripstone::DripstoneError> {
    let backend: Arc<dyn Backend> = Arc::new(MemoryBackend::new());
    let chunk = Chunk::new(
        "doc:1",
        0,
        "Hello, world!",
        vec![0.12, 0.87, 0.44],
        0,
        0,
        1,
        json!({ "title": "Getting Started" }),
        Some("knowledge-base".to_string()),
    );

    backend.upsert(&[chunk]).await?;

    let query = SearchQuery::new(vec![0.1, 0.9, 0.4])
        .metric(Metric::Cosine)
        .limit(5)
        .namespace("knowledge-base");
    let results = backend.search(&query).await?;

    println!("{results:?}");
    Ok(())
}
```

## Backend Selection

| Use case | Backends |
| -------- | -------- |
| Development and tests | Memory |
| Service-backed search | Redis, PostgreSQL, Qdrant (`backends`); Chroma, Weaviate, Milvus (`backends-http`) |
| Managed vector services | Pinecone (`backends-http`), S3 Vector Buckets, DynamoDB native vectors (`backends-aws`) |
| Object-storage-backed retrieval | S3, ColdForge |
| Local persistent index | Mmap |
| Hot/warm/cold layouts | Tiered chain |
| Large / distributed namespaces | `WalShardedBackend` (single WAL, multi-shard) |
| Object-storage shards at scale | `build_coldforge_wal_sharded` (ColdForge + shared WAL + hot tiers) |
| ANN over any backend | `IndexedBackend` with HNSW or SPFresh |
| Deferred ANN ingest | `IndexUpdateMode::Deferred` on `IndexedBackend` (commit now, index later) |
| Late-interaction MaxSim | `search_with_maxsim_fallback` / `LateInteractionIndexedBackend` (PLAID) |

See [backends](docs/backends.md) and [indexing](docs/indexing.md) for usage details.

## Distributed reads and durability

ColdForge search runs against an in-process SPFresh index. `upsert` is not a durability barrier: the WAL buffer reaches object storage on `flush`, on a full batch (`wal_max_batch_size`), or after `wal_batch_interval_ms` (default 1000). A failed flush keeps the buffer and returns the error. `WalCompactor::compact` deletes older segments only when a membership snapshot is a fixed point of the log and every live envelope in the dropped prefix matches a chunk-id body. `ColdForgeBackend::compact_wal` saves that snapshot first. Direct `compact` without a snapshot still refuses and deletes nothing. Split and merge move chunk ids between clusters; body keys stay `{prefix}bodies/{namespace}/{chunk_id}`.

Another `ColdForgeBackend` on the same bucket sees flushed segments on `get`, `search`, `count`, and `list` when a hot tier is set (`refresh_shared_wal` defaults to that) or when refresh is `Some(true)`. `wal_refresh_interval_ms` defaults to 1000. `0` lists the WAL on every refreshing read, and that list runs before the hot tier on `get`. The hot tier is not searched. `get` returns a hot-tier hit only if the namespace index still contains the id, so a delete or an unflushed write does not resurrect from Redis. `refresh_shared_wal(namespace)` forces a reload when the backend was opened as a single writer.

The gated live scene is `coldforge_live_aws_example` ([AWS integration](docs/aws-integration.md)). With `DRIPSTONE_CORPUS` unset it is dimension 128, cosine, namespace `live`, N in {50, 200, 1000}. Set `DRIPSTONE_CORPUS` to ingest a catalog corpus (`nytimes-256-angular`, `sift-128-euclidean`, `gist-960-euclidean`, `cohere-wiki-en`, or `code`) at that corpus's dimension and metric, capped by `DRIPSTONE_CORPUS_TRAIN_MAX` (soak default 10_000). `code` is CoIR CoSQA: tree-sitter function chunks and hashed cosine features. Binary qrels are answer-class judgments; `answer_at_k` and `support_at_k` sit beside brute-force `recall_at_k` (`-` when that class is empty). Cosine corpora drop all-zero and near-zero train and query rows before those vectors are written and upserted; the manifest records the dropped counts. Euclidean, dot, and Manhattan keep zero rows. Cold search is 2 samples on AWS and 50 on MinIO; warm search is 50. Get samples drop the in-process body map so they measure object hydrate or the Redis hot tier. Upsert PUTs chunk bodies with `DRIPSTONE_CORPUS_PUT_CONCURRENCY` (default 32). The scene sets `wal_refresh_interval_ms` to 1000. One binary, `DRIPSTONE_LIVE_BACKEND` of `minio`, `s3`, or `s3-redis`. It prints one table from that process. Use `--release`.

`SearchStrategy::HotThenSourceFallback` stops after a tier only when that tier is a complete copy: no `max_vectors` / `max_bytes` cap, and the write strategy fills it. A capped hot tier cannot hide a nearer cold neighbor. Tiered `background_sync` is off by default because each tick lists every warmer-tier vector.

Redis embedding keys are `{prefix}{namespace}:vec:{id}`. Document sets stay `{prefix}{namespace}:doc:{doc_id}`, so an id such as `doc:1` is not skipped. Listing uses `SCAN`. `delete` also removes the id from the document set. Existing Redis hashes written with the old `{prefix}{namespace}:{id}` layout are not read.

## Documentation

- [Backends](docs/backends.md)
- [Indexing](docs/indexing.md)
- [Graph backends](docs/graph.md)
- [Benchmarks](docs/benchmarks.md)
- [AWS integration benchmarks](docs/aws-integration.md)

## Examples

Examples live in `examples/`.

```bash
cargo run --example quickstart
cargo run --example filter_dsl_example --features memory-backend
cargo run --example hybrid_search_example --features memory-backend
cargo run --example coldforge_example --features coldforge-backend
cargo run --example tiered_redis_example --features "redis-backend,s3-backend"
cargo run --example dynamodb_vector_tiered_example --features "dynamodb-backend,memory-backend"
cargo run --example graph_age_neo4j_example --features "graph-age,graph-neo4j"
```


## Extras kept out of the default build

Default features are `core` and `local`. `cargo tree -p dripstone` and
`cargo tree -p dripstone --no-default-features` must not contain the crates
below. `scripts/check-default-deps.sh` checks both sets and fails if one of
them appears. `local` adds Tokio for the in-process store. It does
not add a remote backend, a model runtime, or an HTTP client.

| Extra | Crates | Feature |
| --- | --- | --- |
| HTTP client | `reqwest`, `hyper`, `http` | `qdrant-backend`, `pinecone-backend`, `chroma-backend`, `weaviate-backend`, `milvus-backend`, `s3-backend`, `dynamodb-backend`, `video-processing` |
| ONNX Runtime | `ort` | `embedding-ort`, `local-ort`, `inference-ort` |
| Candle | `candle-core`, `candle-nn`, `candle-transformers` | `embedding-candle`, `local-candle`, `inference-candle` |
| Tokenizers and model download | `tokenizers`, `hf-hub` | `embedding-ort`, `embedding-candle`, `model-download`, `hf-hub` |
| Remote and service backends | `redis`, `sqlx`, `qdrant-client`, `aws-sdk-s3`, `aws-config`, `pgvector` | `redis-backend`, `postgres-backend`, `sqlite-backend`, `qdrant-grpc-backend`, `s3-backend`, `dynamodb-backend` |
| Other model runtimes | `llama-cpp-2`, `mlx-rs`, `image` | `inference-llamacpp`, `inference-mlx`, `vision-ort`, `vision-candle` |
| Corpus readers | `hdf5-reader`, `parquet` | `corpus` |
| Graph databases | `neo4rs`, `sqlx` | `graph-neo4j`, `graph-age` |

## Feature map

| Goal | Feature |
| --- | --- |
| Zero-config `open` / `add` / `search` | default (`local`) |
| Primary app API | default + choose backends |
| Neural local embeddings | `embedding-ort` / `embedding-candle` / `embedding-mlx` |
| Text generation | `inference-ort` / `inference-candle` / `inference-llamacpp` |
| HTTP vector DBs | `qdrant-backend`, `chroma-backend`, `pinecone-backend`, … |
| gRPC Qdrant | `qdrant-grpc-backend` |
| Object + vector S3 | `s3-backend`, `coldforge-backend` |
| DynamoDB vectors | `dynamodb-backend` |
| ANN on any backend | `tokio` + indexed wrappers (see `backend::indexed`) |

`local-ort` / `local-candle` are lean runtime pieces pulled by `embedding-*`.
Prefer the `embedding-*` / `inference-*` flags in application manifests.

## Development

Toolchain: Rust 1.92.0 (`rust-toolchain.toml`, edition 2024). This repository
is a single package. Build and test it from the repo root; `./caliche` is not
shipped here.

```bash
cargo test --locked --features memory-backend,mmap-backend,sqlite-backend
cargo clippy --locked --features memory-backend --all-targets -- -D warnings
```

`s3-backend` compiles the AWS SDK and does not need live credentials for
`cargo test` of the in-memory paths. Service integration tests are opt-in and
skip when their process is down.

Local backends are in [`docker-compose.example.yml`](docker-compose.example.yml).
Passwords there are development defaults only.

```bash
docker compose -f docker-compose.example.yml up -d
docker compose -f docker-compose.example.yml --profile graph up -d
docker compose -f docker-compose.example.yml --profile search up -d
docker compose -f docker-compose.example.yml --profile milvus up -d
```

Published ports bind to `127.0.0.1` only. Default services: Redis
(`127.0.0.1:6379`), PostgreSQL with pgvector (`127.0.0.1:5433`), MinIO
(`127.0.0.1:9000`, digest-pinned Chainguard image), Qdrant (`127.0.0.1:6333`).
`graph` adds Apache AGE (`127.0.0.1:5455`) and Neo4j Bolt (`127.0.0.1:7687`).
`search` adds Valkey (`127.0.0.1:6380`), Chroma (`127.0.0.1:8000`), and
Weaviate (`127.0.0.1:8081`, container port 8080).
`milvus` adds Milvus gRPC and REST v2 (`127.0.0.1:19530`) and metrics
`GET /healthz` (`127.0.0.1:9091`). `GET /healthz` on `:19530` is 404.

`scripts/check-default-deps.sh` reads this crate's dependency tree with
`--no-default-features` and with the default features, prints each dependency
count, and fails if a remote backend, ONNX/model, or HTTP crate appears.
`./caliche rust clippy` runs it. Inside the Caliche workspace the crate is
gated with:

```bash
./caliche rust clippy
./caliche rust test
./caliche rust complexity --gate
```

`./caliche rust complexity --gate` scans `dripstone/src` in that workspace. It does
not run from this repository alone.

Audit McCabe complexity locally if needed:

```bash
uvx lizard src -l rust -C 20 -L 150 -a 8 -w -s cyclomatic_complexity -x '*/tests/*'
```

Build API docs:

```bash
cargo doc --features backends-all --no-deps --open
```

## Security

Report vulnerabilities privately. See [SECURITY.md](SECURITY.md).

## License

Licensed under either of [Apache-2.0](../LICENSE-APACHE) or [MIT](../LICENSE-MIT), at your option.
