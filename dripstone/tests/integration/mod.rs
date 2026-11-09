//! Integration benchmark harness modules.

mod aws;
mod datasets;
mod ground_truth;
mod metrics;
mod parquet_corpus;
mod scenarios;
#[cfg(feature = "memory-backend")]
mod spfresh_filters;

use dripstone::DripstoneError;

/// Run the full benchmark matrix. The caller must opt in with the DRIPSTONE_INT_* buckets.
pub async fn run_all() -> Result<(), DripstoneError> {
    aws::require_integration_opt_in()?;
    let aws_ok = aws::probe_credentials().await;
    if !aws_ok {
        return Err(DripstoneError::storage(
            "AWS credentials are not available or invalid".to_string(),
        ));
    }

    scenarios::run_matrix().await
}

#[cfg(feature = "memory-backend")]
pub async fn run_spfresh_filter_benchmark() -> Result<(), DripstoneError> {
    spfresh_filters::run().await
}
