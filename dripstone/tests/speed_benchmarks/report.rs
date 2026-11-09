//! Aggregate JSON report for `speed_benchmarks` (written under `target/dripstone/speed_benchmarks/results/`).
//!
//! Tests in this crate use `#[serial_test::serial]` so the finalize test runs after other tests
//! and the on-disk JSON includes every benchmark row.

macro_rules! outputln {
    () => {{
        let mut output = std::io::stdout().lock();
        let _ = std::io::Write::write_all(&mut output, b"\n");
    }};
    ($($arg:tt)+) => {{
        let mut output = std::io::stdout().lock();
        let _ = std::io::Write::write_fmt(&mut output, format_args!($($arg)*));
        let _ = std::io::Write::write_all(&mut output, b"\n");
    }};
}

use std::path::PathBuf;
use std::sync::Mutex;
use std::time::Duration;

use serde_json::{Value, json};
use time::OffsetDateTime;

static ENTRIES: Mutex<Vec<Value>> = Mutex::new(Vec::new());
const RESULT_FILE_NAME: &str = "speed_benchmarks.json";

/// Append one benchmark row (must include a `"test"` string field for stable sorting/filtering).
pub fn record(entry: Value) {
    if let Ok(mut g) = ENTRIES.lock() {
        g.push(entry);
    }
}

pub fn duration_ms(d: Duration) -> f64 {
    d.as_secs_f64() * 1000.0
}

pub fn results_dir() -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("target");
    p.push("dripstone");
    p.push("speed_benchmarks");
    p.push("results");
    p
}

/// Write `speed_benchmarks.json` and print the same JSON to stdout.
pub fn write_and_print_aggregate() -> Result<(), dripstone::DripstoneError> {
    let entries: Vec<Value> = ENTRIES
        .lock()
        .map_err(|e| dripstone::DripstoneError::storage(format!("speed bench report lock: {e}")))?
        .clone();

    let ts = OffsetDateTime::now_utc().unix_timestamp();
    let profile = if cfg!(debug_assertions) {
        "debug"
    } else {
        "release"
    };

    let aggregate = json!({
        "kind": "dripstone_speed_benchmarks",
        "generated_at_unix": ts,
        "profile": profile,
        "entry_count": entries.len(),
        "env": {
            "DRIPSTONE_BENCH_MAX_POINTS": std::env::var("DRIPSTONE_BENCH_MAX_POINTS").ok(),
            "DRIPSTONE_BENCH_LONG": std::env::var("DRIPSTONE_BENCH_LONG").ok(),
            "DRIPSTONE_BENCH_BACKENDS": std::env::var("DRIPSTONE_BENCH_BACKENDS").ok(),
            "DRIPSTONE_BENCH_HNSW_SCALE": std::env::var("DRIPSTONE_BENCH_HNSW_SCALE").ok(),
            "DRIPSTONE_BENCH_HNSW_SWEEP": std::env::var("DRIPSTONE_BENCH_HNSW_SWEEP").ok(),
        },
        "entries": entries,
    });

    let pretty = serde_json::to_string_pretty(&aggregate)
        .map_err(|e| dripstone::DripstoneError::storage(format!("speed bench json: {e}")))?;

    outputln!(
        "\n=== dripstone speed_benchmarks aggregate (JSON) ===\n{pretty}\n=== end aggregate ===\n"
    );

    let dir = results_dir();
    std::fs::create_dir_all(&dir).map_err(|e| {
        dripstone::DripstoneError::storage_with_source("speed_benchmarks results mkdir", e)
    })?;
    let path = dir.join(RESULT_FILE_NAME);
    std::fs::write(&path, pretty.as_bytes()).map_err(|e| {
        dripstone::DripstoneError::storage_with_source("speed_benchmarks results write", e)
    })?;
    outputln!("wrote {}", path.display());

    Ok(())
}
