# Dripstone Backends

Dripstone backends implement storage and search for chunks with vector embeddings.

## Memory

```rust
use std::sync::Arc;
use dripstone::traits::Backend;
use dripstone::MemoryBackend;

let backend: Arc<dyn Backend> = Arc::new(MemoryBackend::new());
```

## Redis Or Valkey

```rust
use std::sync::Arc;
use dripstone::traits::Backend;
use dripstone::RedisBackend;

let backend: Arc<dyn Backend> = Arc::new(
    RedisBackend::new("redis://127.0.0.1/", "embeddings").await?
);
```

## PostgreSQL And pgvector

```rust
use std::sync::Arc;
use dripstone::backend::postgres::PostgresBackendOptions;
use dripstone::traits::Backend;
use dripstone::PostgresBackend;

let backend: Arc<dyn Backend> = Arc::new(
    PostgresBackend::with_options(
        "postgres://localhost/embeddings",
        PostgresBackendOptions {
            table_name: "embeddings".to_string(),
            vector_dimension: Some(1536),
            ..Default::default()
        },
    ).await?
);
```

## Qdrant

```rust
use std::sync::Arc;
use dripstone::backend::qdrant::QdrantGrpcBackendOptions;
use dripstone::traits::Backend;
use dripstone::QdrantBackend;

let backend: Arc<dyn Backend> = Arc::new(
    QdrantBackend::with_options(QdrantGrpcBackendOptions {
        grpc_url: "http://127.0.0.1:6334".to_string(),
        vector_dimension: 384,
        ..Default::default()
    }).await?
);
```

## HTTP Backends

Supported HTTP service backends include Pinecone, Chroma, Weaviate, and Milvus. Enable the matching feature and construct the backend with the service URL, dimension, and optional API key.

## S3

```rust
use dripstone::backend::s3::{S3Backend, S3BackendConfig};

let backend = S3Backend::with_config(S3BackendConfig {
    bucket: "my-vectors".to_string(),
    key_prefix: "embeddings/".to_string(),
    cache_indices: true,
    ..Default::default()
}).await?;
```

## S3 Vector Buckets

Native AWS S3 Vector Buckets ANN (`PutVectors` / `QueryVectors`). Enable `s3-backend`.

```rust
use dripstone::backend::s3::{S3VectorBackend, S3VectorBackendConfig};
use dripstone::Metric;

let backend = S3VectorBackend::new(
    S3VectorBackendConfig::new("my-vector-bucket", "embeddings", 384, None, None, None)
        .with_distance_metric(Metric::Cosine)
        .with_region("us-east-1"),
).await?;
```

## DynamoDB Native Vectors

Server-side ANN via DynamoDB vector indexes (`SearchVectors`). Enable `dynamodb-backend` (also included in `backends-aws` / `backends-all`).

Item schema: `pk` (namespace), `sk` (chunk id), `embedding` (list of numbers), `envelope` (chunk JSON), `doc_id` (inline filter), optional `multi_vectors` (MaxSim tokens).

```rust
use std::sync::Arc;
use dripstone::{
    Backend, DynamoDbVectorBackend, DynamoDbVectorConfig, Metric, TierConfig, TieredChainBuilder,
    MemoryBackend,
};

let ddb = Arc::new(
    DynamoDbVectorBackend::new(
        DynamoDbVectorConfig::new("my-table", "embeddings", 384)
            .with_distance_metric(Metric::Cosine)
            .with_region("us-east-1"),
    )
    .await?,
);

// Standalone
let standalone: Arc<dyn Backend> = ddb.clone();

// Or as a cold tier
let tiered: Arc<dyn Backend> = TieredChainBuilder::new()
    .tier(ddb, TierConfig::new("dynamodb"))
    .tier(Arc::new(MemoryBackend::new()), TierConfig::new("memory"))
    .build()?;
```

Capabilities: native ANN + namespace HASH filter; `native_maxsim: false` so combine with `search_with_maxsim_fallback`, `IndexedBackend`, or `LateInteractionIndexedBackend`.

## ColdForge

ColdForge stores vector data in S3-compatible object storage and searches an in-process SPFresh index rebuilt from the WAL.

`upsert` updates that index and a WAL buffer. The buffer is durable after `flush`, after `wal_max_batch_size` entries, or after `wal_batch_interval_ms` (default 1000; the background task waits one full interval). Set the interval to `0` to flush each upsert batch before returning. `WalCompactor` does not delete segments: those segments are the chunk bodies.

A hot tier is a `get` cache, not the search index. With a hot tier configured, reads reload flushed WAL segments so another process sharing the bucket can see them. `get` ignores a hot-tier or local-cache hit whose id is gone from the namespace index. Pass `refresh_shared_wal: Some(false)` for a single-writer process that should not list the WAL on every read.

```rust
use dripstone::backend::coldforge::{ColdForgeBackend, ColdForgeConfig};

let backend = ColdForgeBackend::new(ColdForgeConfig {
    bucket: "my-vectors".to_string(),
    region: "us-east-1".to_string(),
    cache_size_bytes: 2 * 1024 * 1024 * 1024,
    num_centroids: 256,
    vector_dimension: 1536,
    ..Default::default()
}).await?;
```

## Tiered Chain

Tiered chains order backends from cold source to hot cache. `get()` walks hottest-to-coldest and promotes frequently accessed records.

```rust
use std::sync::Arc;
use dripstone::backend::memory::MemoryBackend;
use dripstone::backend::s3::{S3Backend, S3BackendConfig};
use dripstone::{TierConfig, TieredChainBuilder, traits::Backend};

let cold_tier = Arc::new(S3Backend::with_config(config).await?);
let hot_tier = Arc::new(MemoryBackend::new());
let backend: Arc<dyn Backend> = TieredChainBuilder::new()
    .tier(cold_tier, TierConfig::new("s3"))
    .tier(hot_tier, TierConfig::new("memory").max_vectors(10_000))
    .build()?;
```

Search strategies:

- `SearchAllTiers`: search every tier and merge results.
- `HotThenSourceFallback`: search the hottest tier first. Stop early only after a tier that holds a complete copy has returned enough hits. A `max_vectors` or `max_bytes` cap, or `WriteToSource` on a warmer tier, means that tier is partial and colder tiers are still searched. `background_sync` is off unless you enable it; a tick lists every vector on the warmer tiers.

## WAL-Sharded Scale-Out

`WalShardedBackend` routes namespaces across shards behind a single shared WAL. Helpers:

- `build_memory_coldforge_wal_sharded` — memory hot tiers + ColdForge cold shards
- `build_coldforge_wal_sharded` — ColdForge shards with shared WAL options

Enable `tokio` plus the shard backends you need (`memory-backend`, `coldforge-backend`). Soft-skip MinIO coverage lives in `tests/minio_coldforge_wal_tests.rs`.

## Environment Variables

| Variable | Backend |
|----------|---------|
| `REDIS_URL` | Redis |
| `POSTGRES_URL` | PostgreSQL |
| `QDRANT_URL` | Qdrant HTTP |
| `QDRANT_GRPC_URL` | Qdrant gRPC |
| `QDRANT_API_KEY` | Qdrant |
| `PINECONE_API_KEY` | Pinecone |
| `PINECONE_INDEX_HOST` | Pinecone |
| `CHROMA_URL` | Chroma |
| `WEAVIATE_URL` | Weaviate |
| `WEAVIATE_API_KEY` | Weaviate |
| `MILVUS_URL` | Milvus |
| `AWS_REGION` | S3, ColdForge, S3 Vectors, DynamoDB |
| `S3_BUCKET` | S3 |
| `S3_PREFIX` | S3 |
| `S3_VECTOR_BUCKET` / `S3_VECTOR_INDEX` | S3 Vector Buckets examples |
| `COLDFORGE_BUCKET` | ColdForge |
| `COLDFORGE_PREFIX` | ColdForge |
| `DRIPSTONE_DYNAMODB_TABLE` | DynamoDB vector backend (required for live tests) |
| `DRIPSTONE_DYNAMODB_INDEX` | DynamoDB vector index name (default `embeddings`) |
| `DRIPSTONE_DYNAMODB_ENDPOINT` | Optional LocalStack / custom endpoint |
| `DRIPSTONE_DYNAMODB_ENSURE` | Create table/index when missing (`1`) |
