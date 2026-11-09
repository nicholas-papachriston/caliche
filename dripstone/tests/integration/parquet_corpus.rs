//! Parquet integration corpora.
//!
//! Loading lives in [`dripstone::corpus`]. This module remains so speed benchmarks
//! can keep the historical module path. Point `DRIPSTONE_INT_PARQUET_DIR` or
//! `DRIPSTONE_CORPUS_PARQUET_DIR` at staged English shards (`en/`), not `simple/`.
//!
//! [CohereLabs/wikipedia-2023-11-embed-multilingual-v3](https://huggingface.co/datasets/CohereLabs/wikipedia-2023-11-embed-multilingual-v3)

use std::path::PathBuf;

use dripstone::DripstoneError;
use dripstone::corpus::{self, CorpusProfile};

use super::datasets::{DatasetCorpus, dataset_from_prepared};

/// Load a staged Parquet directory through the shared corpus normalizer.
///
/// # Errors
///
/// Returns an error when the directory has no shards or fewer rows than the caps.
pub fn load_parquet_integration_corpus(root: PathBuf) -> Result<DatasetCorpus, DripstoneError> {
    // The process env still supplies caps and the embedding column. The path
    // argument wins over `DRIPSTONE_INT_PARQUET_DIR` so callers can pass a fixture.
    let mut request = corpus::request_from_env(CorpusProfile::Integration)?;
    let name = match &request.selector {
        corpus::CorpusSelector::ParquetDir { name, .. }
        | corpus::CorpusSelector::CodeDir { name, .. }
        | corpus::CorpusSelector::Hdf5Url { name, .. } => name.clone(),
        corpus::CorpusSelector::Catalog(catalog) => catalog.as_str().to_string(),
    };
    request.selector = corpus::CorpusSelector::ParquetDir { name, path: root };
    let prepared = corpus::prepare_corpus_blocking(&request)?;
    dataset_from_prepared(prepared)
}

#[test]
fn missing_parquet_directory_is_an_error() {
    let dir = tempfile::tempdir().expect("temp");
    let error = match load_parquet_integration_corpus(dir.path().to_path_buf()) {
        Ok(_) => panic!("empty parquet dir must fail"),
        Err(error) => error,
    };
    let message = error.to_string();
    assert!(
        message.contains("parquet") || message.contains("no .parquet"),
        "{message}"
    );
}
