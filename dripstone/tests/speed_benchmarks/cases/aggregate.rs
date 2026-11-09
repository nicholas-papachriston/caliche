//! Speed benchmark cases (see parent `speed_benchmarks.rs`).

#[cfg(any(feature = "memory-backend", feature = "mmap-backend"))]
use crate::speed_bench_report;
use dripstone::DripstoneError;
#[cfg(any(feature = "memory-backend", feature = "mmap-backend"))]
use serial_test::serial;

#[tokio::test]
#[serial]
async fn test_zzzz_speed_benchmarks_write_aggregate_json() -> Result<(), DripstoneError> {
    speed_bench_report::write_and_print_aggregate()
}
