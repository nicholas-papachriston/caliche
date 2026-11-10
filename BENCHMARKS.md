# Benchmarks

Wall-clock numbers for Dripstone. These tests do not run in the merge gate,
because their time bounds fail on a loaded machine and in debug builds. Run
them alone, in release, on one thread:

```bash
./caliche rust bench
```

Each test still asserts its bound (for example, search p99 under 100 ms), so
a large regression fails the run.

## Machines

| machine | hardware | OS |
| --- | --- | --- |
| m4max | Apple M4 Max (Mac16,5), 128 GB | macOS (Darwin 27.0.0) |

## Data

| data | description |
| --- | --- |
| integration | The Dripstone integration corpus profile with default caps: up to 4,000 train vectors and 64 queries, with no `DRIPSTONE_*` overrides. In-memory backend. |

## Results

| scenario | date | machine | data | metric | result |
| --- | --- | --- | --- | --- | --- |
| search latency (`test_search_latency_percentiles`, 64 samples) | 2026-11-10 | m4max | integration | p50 / p95 / p99 | 0.185 ms / 0.203 ms / 0.225 ms |
| upsert latency (`test_upsert_latency_percentiles`, 200 samples) | 2026-11-10 | m4max | integration | p50 / p95 / p99 | 250 ns / 458 ns / 958 ns |
| search throughput (`test_search_throughput`, 5 s window) | 2026-11-10 | m4max | integration | queries per second | 1,166 |
| concurrent reads (`test_concurrent_read_throughput`, 10 tasks, 500 queries) | 2026-11-10 | m4max | integration | queries per second | 1,624 |
| search scaling (`test_performance_scaling`, 500 to 3,998 vectors) | 2026-11-10 | m4max | integration | search time growth for 8.00x data | 9.22x (median of 9 searches per size) |
