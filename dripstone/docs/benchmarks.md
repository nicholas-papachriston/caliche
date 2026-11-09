# Dripstone Benchmarks

Dripstone benchmark tests are opt-in. The default `cargo test` run keeps runtime bounded.

## Speed Benchmarks

```bash
cargo test -p dripstone --test speed_benchmarks --features "memory-backend,mmap-backend" -- --nocapture
```

Run larger cases:

```bash
DRIPSTONE_BENCH_LONG=1 \
  cargo test -p dripstone --test speed_benchmarks --features full -- --nocapture
```

Run selected backends:

```bash
DRIPSTONE_BENCH_BACKENDS=memory,mmap,redis,valkey,postgres,qdrant,chroma \
  cargo test -p dripstone --test speed_benchmarks --features full -- --nocapture
```

## HNSW Scale

```bash
DRIPSTONE_BENCH_HNSW_SCALE=1 DRIPSTONE_BENCH_MAX_POINTS=120000 \
  cargo test -p dripstone --release --test speed_benchmarks --features memory-backend,corpus,hnsw \
  test_hnsw_vs_brute_scale -- --nocapture --test-threads=1
```

## HNSW Parameter Sweep

```bash
DRIPSTONE_BENCH_HNSW_SWEEP=1 DRIPSTONE_BENCH_MAX_POINTS=120000 \
  cargo test -p dripstone --release --test speed_benchmarks --features memory-backend,corpus,hnsw \
  test_hnsw_parameter_sweep -- --nocapture --test-threads=1
```

## Variables

| Variable | Purpose |
|----------|---------|
| `DRIPSTONE_BENCH_LONG` | Enables larger benchmark cases and wider concurrency sweeps |
| `DRIPSTONE_BENCH_MAX_POINTS` | Upper bound on corpus rows for long cases |
| `DRIPSTONE_BENCH_HNSW_SCALE` | Runs the HNSW vs brute-force scale test |
| `DRIPSTONE_BENCH_HNSW_SWEEP` | Runs the HNSW parameter sweep |
| `DRIPSTONE_BENCH_BACKENDS` | Comma-separated backend matrix |

The recall harness compares HNSW against runtime exact top-k on the same upserted chunks.
