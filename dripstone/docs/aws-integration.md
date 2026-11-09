# AWS Integration Benchmarks

Dripstone has opt-in integration benchmarks for S3, S3 Vector Buckets, ColdForge, DynamoDB native vectors, and tiered layouts. If AWS credentials (or the relevant env vars) are unavailable, the test prints a skip message and exits successfully. Redis-dependent scenarios are skipped when Redis is unavailable.

## Run

From this repository root:

```bash
cargo test --test integration \
  --features coldforge-backend,mmap-backend,memory-backend,redis-backend \
  -- --nocapture --test-threads=1
```

DynamoDB native vector soft-skip suite:

```bash
DRIPSTONE_DYNAMODB_TABLE=my-table \
cargo test --test dynamodb_vector_tests \
  --features "dynamodb-backend,memory-backend,tokio" -- --test-threads=1
```

The main integration test can:

- create or reuse a standard S3 bucket
- create or reuse an S3 Vector bucket and index
- load an ann-benchmarks HDF5 corpus
- load local Parquet shards
- compare ColdForge and S3 Vector Buckets as cold storage
- exercise S3 Vector plus Redis plus mmap/HNSW tiering
- write JSON summaries under `target/dripstone/integration/results/`

## Variables

| Variable | Purpose |
|----------|---------|
| `REDIS_URL` | Redis connection string for Redis tier scenarios |
| `DRIPSTONE_INT_S3_BUCKET` | Standard S3 bucket for ColdForge |
| `DRIPSTONE_INT_S3_VECTOR_SIDECAR_BUCKET` | Standard S3 bucket for S3 Vector namespace key lists |
| `DRIPSTONE_INT_S3_PREFIX` | Key prefix for ColdForge objects |
| `DRIPSTONE_INT_S3_VECTOR_BUCKET` | S3 Vector bucket name |
| `DRIPSTONE_INT_S3_VECTOR_INDEX` | S3 Vector index name |
| `DRIPSTONE_INT_S3_VECTORS_REGION` | S3 Vector Buckets region |
| `DRIPSTONE_S3_VECTORS_ENDPOINT` | Optional SigV4 HTTP base URL override |
| `DRIPSTONE_CORPUS` | Catalog id: `nytimes-256-angular`, `sift-128-euclidean`, `gist-960-euclidean`, `cohere-wiki-en`, `code` (CoIR CoSQA) |
| `DRIPSTONE_CORPUS_URL` | HDF5 download URL override. `DRIPSTONE_INT_DATASET_URL` is the alias |
| `DRIPSTONE_CORPUS_PARQUET_DIR` | Staged Parquet file or directory. `DRIPSTONE_INT_PARQUET_DIR` is the alias |
| `DRIPSTONE_CORPUS_PARQUET_RECURSIVE` | Recursively scan for Parquet files. `DRIPSTONE_INT_PARQUET_RECURSIVE` is the alias |
| `DRIPSTONE_CORPUS_EMB_COLUMN` | Arrow column name for embeddings (default `emb`). `DRIPSTONE_INT_PARQUET_EMB_COLUMN` is the alias |
| `DRIPSTONE_INT_DATASET_NAME` | Label for logs and JSON summaries |
| `DRIPSTONE_CORPUS_TRAIN_MAX` | Max training vectors materialized. Soak default 10_000. `DRIPSTONE_INT_TRAIN_MAX` is the alias; integration default stays 4_000 when both are unset |
| `DRIPSTONE_CORPUS_QUERY_MAX` | Max query vectors. Soak default 128. `DRIPSTONE_INT_QUERY_MAX` is the alias; integration default stays 64 |
| `DRIPSTONE_CORPUS_NEIGHBORS_K` | Max neighbor columns read from HDF5. `DRIPSTONE_INT_NEIGHBORS_K` is the alias (default 100) |
| `DRIPSTONE_CORPUS_BATCH` | Live-scene upsert batch size (default 256) |
| `DRIPSTONE_CORPUS_PUT_CONCURRENCY` | Concurrent ColdForge body PUTs (default 32) |
| `DRIPSTONE_CORPUS_SHARD_MAX` | Cohere `en/NNNN.parquet` shards to download (default 4) |
| `DRIPSTONE_CORPUS_CODE_DIR` | Staged CoIR CoSQA directory (`corpus.jsonl` or `corpus/*.parquet`, plus queries and optional qrels). Used when `DRIPSTONE_CORPUS=code` or when that variable is unset |
| `DRIPSTONE_CORPUS_GRADE_POLICY` | `binary` (default: every positive is answer), `explicit` (`class` / `role` / `relevance`), or `threshold[:N]` (score `>= N` is answer, lower positives are support; default N is 2) |
| `DRIPSTONE_CORPUS_TECHNIQUE` | Plugin id: `hard_neighbors` (default), `graded_context`, `contextual_embed`, `late_chunking`, or `token_chunk_teacher`. The last three keep chunk text unchanged |
| `DRIPSTONE_CORPUS_METRIC` | Metric override: `cosine`, `euclidean`, `dot`, `manhattan` |
| `DRIPSTONE_CACHE_DIR` | Cache root. Corpora land in `<root>/corpora/<name>/<rev>/` |
| `DRIPSTONE_INT_GROUND_TRUTH_SOURCE` | Recall ground truth source |
| `DRIPSTONE_INT_SEARCH_METRIC` | Distance metric override |
| `DRIPSTONE_INT_UNTIMED_PREWARM` | Run one untimed search before measured first search |
| `DRIPSTONE_INT_MEMORY_HOT_PROMOTION_THRESHOLD` | Promotion threshold for memory hot tier |
| `DRIPSTONE_DYNAMODB_TABLE` | DynamoDB table with a vector index (unlocks DynamoDB suite) |
| `DRIPSTONE_DYNAMODB_INDEX` | Vector index name (default `embeddings`) |
| `DRIPSTONE_DYNAMODB_REGION` / `AWS_REGION` | DynamoDB region |
| `DRIPSTONE_DYNAMODB_ENDPOINT` | Optional LocalStack endpoint |
| `DRIPSTONE_DYNAMODB_ENSURE` | Create table + vector index when missing (`1`) |
| `DRIPSTONE_DYNAMODB_FORCE` | Fail instead of soft-skip when connect fails (`1`) |
| `DRIPSTONE_DYNAMODB_SEARCH_ENDPOINT` | Override the SearchVectors host. Live AWS uses `https://search-dynamodb.{region}.api.aws` |
| `DRIPSTONE_INT_S3_VECTORS_ENSURE` | Create the S3 Vector bucket and index when `1`. Otherwise missing indexes are skipped |
| `DRIPSTONE_LIVE_AWS` | Run the ColdForge live scene when `1` |
| `DRIPSTONE_LIVE_BACKEND` | `minio`, `s3`, or `s3-redis`. Unset infers MinIO from `OBJECT_STORE_ENDPOINT`, else S3+Redis from `REDIS_URL`, else S3 |
| `COLDFORGE_BUCKET` | Bucket for ColdForge examples and the live scene |
| `AWS_PROFILE` | Default credential chain profile. `example` is an example profile for `example-bucket` |

## Corpora

Shared download and normalize code lives in `dripstone::corpus`. Integration tests and the live scene both call it. The normalized layout is:

```text
$DRIPSTONE_CACHE_DIR/corpora/<name>/<rev>/
  manifest.json
  vectors.f32.bin
  ids.jsonl
  meta.jsonl            optional
  queries.f32.bin       optional
  query_ids.jsonl       optional
  neighbors.i32.bin     optional, row-major little-endian i32 (answer and support indices, -1 pad)
  qrels.jsonl           optional graded judgments (`class` answer/support/irrelevant, optional score)
  source/               resumable original (HDF5, en/*.parquet, or CoSQA)
```

`DRIPSTONE_CACHE_DIR` falls back to `$HF_HOME/dripstone`, then `~/.cache/huggingface/dripstone`. Downloads use the resumable curl helper (`loader::download`), including a partial-file resume. That matters for `gist-960-euclidean`, whose HDF5 is about 3.6 GB.

Normalization reads a hyperslab of the first `TRAIN_MAX` / `QUERY_MAX` rows. The on-disk `vectors.f32.bin` holds that cap. Cosine and angular corpora, including `nytimes-256-angular`, drop train and query rows whose L2 norm is below the shared cosine minimum before that file is written. The manifest records `dropped_train_rows` and `dropped_query_rows`. A MinIO soak of the first 10,000 NYTimes rows otherwise upserts three all-zero train vectors and cosine search fails. Euclidean, dot, and Manhattan keep zero rows.

A later run with a larger cap re-reads the source. A smaller cap reuses a layout whose row policy matches and slices the kept rows. A cosine layout written before that policy is rebuilt from the source.

Catalog entries:

| Name | Source | Dim | Metric |
|------|--------|-----|--------|
| `nytimes-256-angular` | ann-benchmarks HDF5 | 256 | cosine |
| `sift-128-euclidean` | ann-benchmarks HDF5 | 128 | euclidean |
| `gist-960-euclidean` | ann-benchmarks HDF5 | 960 | euclidean |
| `cohere-wiki-en` | `CohereLabs/wikipedia-2023-11-embed-multilingual-v3` Parquet under `en/` | 1024 | dot |
| `code` | CoIR CoSQA (`CoIR-Retrieval/cosqa`, pinned revision) | 256 | cosine |

`code` is the CoIR CoSQA corpus (`CoIR-Retrieval/cosqa` at revision `0846fa3b963a21bead36e9fab61451fc83b777a6`). The helper downloads `corpus/corpus-00000-of-00001.parquet`, `queries/queries-00000-of-00001.parquet`, and test qrels `data/test-00000-of-00001.parquet` with the resumable curl loader, then stops materializing at `DRIPSTONE_CORPUS_TRAIN_MAX` and `DRIPSTONE_CORPUS_QUERY_MAX`. The three files together are a few megabytes; the soak cap does not pull a larger code-search collection. Each code row is split at Python function boundaries with tree-sitter (`ChunkStrategy::AstAware`, feature `ast`, pulled in by `corpus`). Queries stay whole. Vectors are 256-dimensional signed token hashes, L2-normalized, and stored under the same layout as the other catalogs. Cosine drops blank rows. Test qrels that hit a kept row become `neighbors.i32.bin` and `qrels.jsonl`. Binary scores are answer-class and leave support empty. `explicit` or `threshold` policies can label support. A layout is reused only when `grade_policy` matches. `meta.jsonl` carries language, symbol, line span, source id, and the chunk text. The live scene copies that text onto the chunk body and leaves the symbol on metadata. Set `DRIPSTONE_CORPUS_CODE_DIR` to a staged tree of those parquet paths, or of `corpus.jsonl` / `queries.jsonl` / `qrels.jsonl`, to skip the download. Aliases: `cosqa`, `coir-cosqa`, `coir`.

Ingest uses the same chunker when `ChunkingConfig` strategy is `AstAware`. Set `code_language` (`python`, `rust`, `javascript`, `go`, `java`) or pass `language` / `path` metadata. Without a grammar the strategy falls back to sentence chunks. Search returns those symbol and line fields on chunk metadata.

Cohere's default language shard is English (`en/0000.parquet`, `en/0001.parquet`, ...), not Simple English. The helper downloads shards until the row cap is filled, stopping at `DRIPSTONE_CORPUS_SHARD_MAX` (default 4). One English shard is about 200 MB and is enough for the soak cap. To skip the download, stage the `en/` directory yourself and set `DRIPSTONE_CORPUS_PARQUET_DIR` (or `DRIPSTONE_INT_PARQUET_DIR`) to that directory. Full-page text is not copied into `meta.jsonl`; title, url, and source id are.

Integration CI should leave the train cap unset or at 4_000. The soak default of 10_000 applies only when the live scene sees `DRIPSTONE_CORPUS` (or a URL / Parquet alias) and neither cap variable is set.

`ColdForge` upsert PUTs chunk bodies with `body_put_concurrency` (default 32, override `DRIPSTONE_CORPUS_PUT_CONCURRENCY`) and only then appends the WAL in input order. A cold open replays the WAL and does not PUT those bodies again.

Cost notes:

- GIST's source download is about 3.6 GB and needs that much free space plus the cache reserve. The normalized prefix at 10_000 rows is about 38 MB (`10000 * 960 * 4`).
- Each vector is one S3 PUT. 10_000 vectors is 10_000 PUTs; concurrency 32 is what keeps that off the serial ~30 ms/PUT path. Do not raise the cap to the full GIST or Wikipedia set on a paid bucket without planning the request cost.
- Cohere shard downloads are Hugging Face bandwidth. Stop at the cap. Do not point the helper at `simple/`.

## ColdForge live scene

One binary measures one backend and prints one markdown table plus one TSV. With `DRIPSTONE_CORPUS` unset the scene is dimension 128, cosine, namespace `live`, N in {50, 200, 1000}. When `DRIPSTONE_CORPUS`, `DRIPSTONE_CORPUS_URL`, `DRIPSTONE_CORPUS_PARQUET_DIR`, or their `DRIPSTONE_INT_*` aliases are set, that corpus is what gets upserted: its dimension, its metric, and one N equal to the capped training rows (soak default 10_000). The same `DRIPSTONE_LIVE_AWS`, `DRIPSTONE_LIVE_BACKEND`, `COLDFORGE_BUCKET`, and `REDIS_URL` surface selects MinIO, S3, or S3+Redis. Cold search opens a fresh backend per sample: 2 samples on `s3` and `s3-redis`, 50 on MinIO. Warm search is 50 samples on the last cold backend. Columns are upsert wall including flush, cold and warm search, get hydrate p50/p90, HeadBucket p50 as network context, and recall@k against brute force on that same N using the scene metric. `answer_at_k` and `support_at_k` are set recall of those judgment classes for the measured query. `-` means that class has no positives, which is the synthetic scene and any binary qrel set. An empty search is hits 0 and recall 0.

`S3Store::aws` uses `aws_config` defaults and `Client::new`, so an SSO profile works when access and secret keys are unset. Run with `--release`. A debug build is labeled in the header and is not the comparison to keep.

`wal_refresh_interval_ms` in the scene is 1000, the product default. `0` lists the object-store WAL on every refreshing read, including `get`, before Redis. This scene does not use `0`. Get samples prime the index once, then drop the in-process body map and the LRU before each timed get. S3 samples measure object hydrate. S3+Redis samples measure the hot tier when it still holds the id. MinIO is the local floor, not an AWS network number.

```bash
# MinIO local floor
DRIPSTONE_LIVE_AWS=1 DRIPSTONE_LIVE_BACKEND=minio \
OBJECT_STORE_ENDPOINT=http://127.0.0.1:9000 \
COLDFORGE_BUCKET=example-wal \
cargo run --release --example coldforge_live_aws_example \
  --features coldforge-backend,redis-backend

# AWS S3 cold-only
AWS_PROFILE=example AWS_REGION=us-east-1 \
DRIPSTONE_LIVE_AWS=1 DRIPSTONE_LIVE_BACKEND=s3 \
COLDFORGE_BUCKET=example-bucket \
cargo run --release --example coldforge_live_aws_example \
  --features coldforge-backend,redis-backend

# AWS S3 + Redis
AWS_PROFILE=example AWS_REGION=us-east-1 \
DRIPSTONE_LIVE_AWS=1 DRIPSTONE_LIVE_BACKEND=s3-redis \
COLDFORGE_BUCKET=example-bucket \
REDIS_URL=redis://127.0.0.1:6379 \
cargo run --release --example coldforge_live_aws_example \
  --features coldforge-backend,redis-backend
```

The same scene is the gated test. It skips unless `DRIPSTONE_LIVE_AWS=1`:

```bash
cargo test --release --test live_aws \
  --features coldforge-backend,memory-backend -- --nocapture
```

That test target does not enable `redis-backend`. `s3-redis` is the example above. Add `redis-backend` to the test features only when you intend to select that backend.

The scene does not include Valkey, S3 Vectors, or DynamoDB. Those stay separate soft-skip examples. The harness prints only samples from the process that ran it.

S3 Vector examples require `S3_VECTOR_BUCKET` and `S3_VECTOR_INDEX` and exit 0 with SKIP when those are unset. They do not create vector buckets. The integration matrix creates them only when `DRIPSTONE_INT_S3_VECTORS_ENSURE=1`.

`s3_storage_example` requires `S3_BUCKET` and fails if search or count stays 0 after upsert. `dynamodb_vector_tiered_example` requires `DRIPSTONE_DYNAMODB_TABLE` and retries SearchVectors a few times, then fails if the upsert is still invisible.

## IAM Permissions

The integration tests require permissions for:

- `sts:GetCallerIdentity`
- creating and inspecting S3 buckets used by the test
- object read, write, and list access on the standard bucket
- `s3vectors:CreateVectorBucket`
- `s3vectors:CreateIndex`
- `s3vectors:GetIndex`
- `s3vectors:DeleteIndex`
- S3 Vector data-plane access in the chosen region
- For DynamoDB suite: `dynamodb:DescribeTable`, `dynamodb:PutItem` / `BatchWriteItem`, `dynamodb:GetItem`, `dynamodb:Query`, `dynamodb:Scan`, `dynamodb:DeleteItem`, `dynamodb:SearchVectors`, and optionally `CreateTable` / `UpdateTable` when `DRIPSTONE_DYNAMODB_ENSURE=1`

Use a vectors-supported region for `DRIPSTONE_INT_S3_VECTORS_REGION`.
