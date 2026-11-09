# Dripstone integration tests

Naming convention:

| Pattern | Role |
| --- | --- |
| `<backend>_smoke.rs` | Minimal connectivity / upsert+search |
| `<backend>_suite.rs` | Shared suite via `integration_suite` helpers |
| `<backend>_bench.rs` | Throughput / latency benches |
| `integration_*.rs` / `integration/` | Cross-backend or AWS corpus harnesses |

Prefer one smoke + one suite file per backend. Avoid a bare `<backend>.rs`
unless it is the only test for that backend.
