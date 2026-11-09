# Dripstone Indexing

Dripstone can wrap any backend with an ANN index while keeping full chunk content in the inner backend.

`HnswIndex`, `HnswConfig`, `HnswGraphSpace`, and `IndexedBuilder::with_hnsw` need the `hnsw` feature. The default build leaves it out, so it does not compile `hnsw_rs`. `full` includes it.

## Indexed Backend

`IndexedBackend` stores vector IDs and embeddings in an index and fetches chunk content from the inner backend.

```rust
use std::sync::Arc;
use dripstone::index::{HnswConfig, HnswGraphSpace, HnswIndex};
use dripstone::traits::Backend;
use dripstone::{IndexedBackend, IndexedBackendConfig, MemoryBackend};

let inner = MemoryBackend::new();
let hnsw = HnswIndex::new(
    HnswConfig {
        graph_space: HnswGraphSpace::CosineCompatible,
        rerank_overfetch: 4,
        ef_search: 200,
        ..Default::default()
    },
    384,
);

let backend: Arc<dyn Backend> = Arc::new(IndexedBackend::new(
    inner,
    Box::new(hnsw),
    IndexedBackendConfig {
        auto_rebuild_threshold: 1000,
        fallback_to_brute_force: true,
        overfetch_factor: 2,
    },
));
```

Behavior:

- `upsert()` writes to the inner backend and updates the ANN index.
- `search()` queries the index, fetches full chunks from the inner backend, and applies namespace and metadata filters.
- `delete()` and `delete_document()` remove records from both the inner backend and index.
- Empty indexes delegate to the inner backend when `fallback_to_brute_force` is `true`.
- After `auto_rebuild_threshold` incremental upserts, the index is rebuilt from stored chunks.

## Deferred index updates (Superstore pattern)

By default, `IndexedBackend` updates the ANN index in the same `upsert` call as the inner
backend write ([`IndexUpdateMode::Synchronous`](../src/backend/indexed/config.rs)). For ingest-heavy
workloads you can opt into **commit now, index later**:

```rust
use dripstone::{HnswConfig, HnswIndex, IndexUpdateMode, IndexedBackend, IndexedBackendConfig, MemoryBackend};

let inner = MemoryBackend::new();
let hnsw = HnswIndex::new(HnswConfig::default(), 384);

let backend = IndexedBackend::new(
    inner,
    Box::new(hnsw),
    IndexedBackendConfig {
        index_update_mode: IndexUpdateMode::Deferred,
        scan_unindexed_tail: true,          // read-your-write via tail scan
        deferred_flush_threshold: 256,      // auto-flush pending queue; 0 = manual only
        deferred_search_flush_threshold: 0,   // flush at search when pending >= N; 0 = off
        max_tail_scan_records: 0,             // cap tail scan to last N pending; 0 = unlimited
        ..Default::default()
    },
);
```

Deferred mode behavior:

1. `upsert()` commits vectors to the inner backend and queues ANN inserts.
2. `search()` merges index hits with a brute-force scan of the pending tail when
   `scan_unindexed_tail` is `true` (default). Disable tail scan only when callers accept
   bounded visibility lag and call [`flush_pending_index`](../src/backend/indexed/backend.rs) themselves.
3. `flush_pending_index()`, `flush()`, and `close()` drain the pending queue into the ANN index.
4. `pending_index_count()` reports how many records are committed but not yet indexed.
5. `deferred_search_flush_threshold` flushes the pending queue at the start of `search()` when
   pending count reaches the threshold (default `0` = disabled).
6. `max_tail_scan_records` limits tail scan to the last N pending records (default `0` = full queue).

Use `IndexedBuilder::with_deferred_index_updates()` for the same configuration on `Arc<dyn Backend>`.

**Do not** enable deferred indexing for RAG paths that need immediate ANN recall without tail scan.
Document the lag at the call site when `scan_unindexed_tail` is `false`.

## HNSW

Use `HnswGraphSpace::CosineCompatible` for cosine workloads with unnormalized vectors. The graph stores normalized vectors for traversal and reranks full-precision records after candidate fetch.

```rust
use dripstone::index::{HnswConfig, HnswGraphSpace, HnswIndex};

let config = HnswConfig {
    m: 32,
    ef_construction: 400,
    ef_search: 200,
    rerank_overfetch: 4,
    graph_space: HnswGraphSpace::CosineCompatible,
    ..Default::default()
};

let index = HnswIndex::new(config, 128);
index.build(&chunks).await?;
let results = index.search(&query, 10, Metric::Cosine).await?;
```

## Search Pipeline

`SearchPipeline` exposes an `index_ann()` stage for direct vector index integration.

```rust
use dripstone::pipeline::SearchPipelineBuilder;
use dripstone::Metric;

let pipeline = SearchPipelineBuilder::new(Metric::Cosine)
    .index_ann(indexed_backend.index(), 100)
    .normalize_scores()
    .build();
```

## Late Interaction (MaxSim / PLAID)

Dense ANN backends store a pooled vector for recall. Late-interaction multi-vectors (`Chunk::multi_vectors` / `SearchQuery::with_multi_vectors`) enable MaxSim rerank:

- `search_with_maxsim_fallback` — client MaxSim when `Capabilities::native_maxsim` is false (S3 Vectors, DynamoDB, most remote stores).
- `LateInteractionIndexedBackend` — PLAID-style candidate generation over an inner backend (`PlaidIndex`).
- Persistence: S3 (`S3V1`) and SQLite (`CKV1`) round-trip multi-vectors; DynamoDB stores them in the optional `multi_vectors` item attribute.

```rust
use dripstone::{search_with_maxsim_fallback, MaxSimOptions, Metric, SearchQuery};

let query = SearchQuery::new(pooled)
    .metric(Metric::Cosine)
    .limit(10)
    .with_multi_vectors(query_tokens)
    .with_maxsim(MaxSimOptions::new(Metric::DotProduct));
let hits = search_with_maxsim_fallback(backend.as_ref(), &query).await?;
```

## Mmap Format

The mmap vector index uses the `DRIPST01` (aligned vector slab) or `DRIPST02` (int8-quantized slab) magic prefixes. Write files with `write_dripstone_mmap` or `write_dripstone_mmap_with_encoding`.
