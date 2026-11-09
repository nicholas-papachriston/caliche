//! Tokenizer loading utilities.

use crate::errors::DripstoneError;
use crate::loader::{HuggingFaceLoader, ModelSource};
use std::path::Path;
#[cfg(any(
    feature = "embedding-ort",
    feature = "embedding-candle",
    feature = "embedding-mlx",
    feature = "inference-mlx"
))]
use tokenizers::Tokenizer;

/// Candidate `SentencePiece` sidecar file names seen in the wild.
#[cfg(any(
    feature = "embedding-ort",
    feature = "embedding-candle",
    feature = "embedding-mlx",
    feature = "inference-mlx"
))]
const SENTENCEPIECE_CANDIDATES: &[&str] =
    &["tokenizer.model", "spiece.model", "sentencepiece.bpe.model"];

/// Load a tokenizer from a model source (local path or `HuggingFace` Hub).
///
/// Tries `tokenizer.json` first, then falls back to any `SentencePiece`
/// sidecar file (`tokenizer.model`, `spiece.model`, or
/// `sentencepiece.bpe.model`). The `SentencePiece` path produces a best-effort
/// `tokenizers::Tokenizer` by loading the `SentencePiece` model as a Unigram
/// tokenizer; callers that need exact parity with a reference tokenizer
/// should still supply an explicit `tokenizer.json`.
#[cfg(any(
    feature = "embedding-ort",
    feature = "embedding-candle",
    feature = "embedding-mlx",
    feature = "inference-mlx"
))]
///
/// # Errors
///
/// Returns an error when the operation cannot be completed.
pub fn load_tokenizer(source: &ModelSource) -> Result<Tokenizer, DripstoneError> {
    match source {
        ModelSource::Local { path } => {
            let base_dir = if path.is_file() {
                path.parent()
                    .unwrap_or_else(|| Path::new("."))
                    .to_path_buf()
            } else {
                path.clone()
            };

            let tokenizer_path = base_dir.join("tokenizer.json");
            if tokenizer_path.exists() {
                return load_tokenizer_from_path(&tokenizer_path);
            }

            for candidate in SENTENCEPIECE_CANDIDATES {
                let sp_path = base_dir.join(candidate);
                if sp_path.exists() {
                    return load_sentencepiece_tokenizer(&sp_path);
                }
            }

            Err(DripstoneError::storage(format!(
                "tokenizer file not found in '{}': expected one of tokenizer.json, {}",
                base_dir.display(),
                SENTENCEPIECE_CANDIDATES.join(", "),
            )))
        }
        ModelSource::HuggingFace { repo, revision } => {
            let loader = HuggingFaceLoader::new(repo, revision.as_deref())?;
            if let Ok(tokenizer_path) = loader.get_file("tokenizer.json") {
                return load_tokenizer_from_path(&tokenizer_path);
            }
            for candidate in SENTENCEPIECE_CANDIDATES {
                if let Ok(sp_path) = loader.get_file(candidate) {
                    return load_sentencepiece_tokenizer(&sp_path);
                }
            }
            Err(DripstoneError::storage(format!(
                "no tokenizer file found in HuggingFace repo '{repo}': expected tokenizer.json or one of {SENTENCEPIECE_CANDIDATES:?}",
            )))
        }
    }
}

/// Attempt to load a `SentencePiece` `.model` file as a `tokenizers::Tokenizer`.
///
/// This uses the Unigram model path from the `tokenizers` crate, which covers
/// the most common `SentencePiece` Unigram format (e.g. T5, ALBERT, XLM-R).
/// BPE `SentencePiece` models are not universally supported by this fallback;
/// when loading fails the caller gets a descriptive error pointing at
/// exporting a `tokenizer.json` with the `HuggingFace` tokenizers converter.
#[cfg(any(
    feature = "embedding-ort",
    feature = "embedding-candle",
    feature = "embedding-mlx",
    feature = "inference-mlx"
))]
fn load_sentencepiece_tokenizer(path: &Path) -> Result<Tokenizer, DripstoneError> {
    use tokenizers::models::unigram::Unigram;
    use tokenizers::tokenizer::Tokenizer as TkTokenizer;

    let model = Unigram::load(path).map_err(|e| {
        DripstoneError::storage(format!(
            "failed to load SentencePiece tokenizer from '{}': {}; \
             consider exporting a tokenizer.json via `transformers.convert_slow_tokenizer`",
            path.display(),
            e
        ))
    })?;
    Ok(TkTokenizer::new(model))
}

/// Load a tokenizer from a file path.
#[cfg(any(
    feature = "embedding-ort",
    feature = "embedding-candle",
    feature = "embedding-mlx",
    feature = "inference-mlx"
))]
///
/// # Errors
///
/// Returns an error when the operation cannot be completed.
pub fn load_tokenizer_from_path(path: &Path) -> Result<Tokenizer, DripstoneError> {
    Tokenizer::from_file(path).map_err(|e| {
        DripstoneError::storage(format!(
            "failed to load tokenizer from '{}': {}",
            path.display(),
            e
        ))
    })
}
