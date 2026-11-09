//! `HuggingFace` Hub loader for downloading models and files.

use crate::errors::DripstoneError;
use crate::loader::download::{DownloadOptions, ensure_downloaded, huggingface_resolve_url};
#[cfg(all(
    feature = "hf-hub",
    any(
        feature = "embedding-ort",
        feature = "embedding-candle",
        feature = "embedding-mlx",
        feature = "inference-mlx",
        feature = "inference-llamacpp"
    )
))]
use hf_hub::api::sync::{Api, ApiBuilder};
#[cfg(any(
    feature = "embedding-ort",
    feature = "embedding-candle",
    feature = "embedding-mlx",
    feature = "inference-mlx",
    feature = "inference-llamacpp"
))]
use std::path::{Path, PathBuf};

#[cfg(any(
    feature = "embedding-ort",
    feature = "embedding-candle",
    feature = "embedding-mlx",
    feature = "inference-mlx",
    feature = "inference-llamacpp"
))]
const BASE_ONNX_MODEL_CANDIDATES: &[&str] = &[
    "model.onnx",
    "model_optimized.onnx",
    "onnx/model.onnx",
    "onnx/model_optimized.onnx",
];

#[cfg(any(
    feature = "embedding-ort",
    feature = "embedding-candle",
    feature = "embedding-mlx",
    feature = "inference-mlx",
    feature = "inference-llamacpp"
))]
const HUGGINGFACE_TOKEN_ENV_VARS: &[&str] = &[
    "HF_TOKEN",
    "HUGGING_FACE_HUB_TOKEN",
    "HUGGINGFACE_TOKEN",
    "HF_HUB_TOKEN",
];

#[cfg(any(
    feature = "embedding-ort",
    feature = "embedding-candle",
    feature = "embedding-mlx",
    feature = "inference-mlx",
    feature = "inference-llamacpp"
))]
#[must_use]
pub fn huggingface_token_from_env() -> Option<String> {
    huggingface_token_from_pairs(
        HUGGINGFACE_TOKEN_ENV_VARS
            .iter()
            .filter_map(|name| std::env::var(name).ok().map(|value| (*name, value))),
    )
}

#[cfg(any(
    feature = "embedding-ort",
    feature = "embedding-candle",
    feature = "embedding-mlx",
    feature = "inference-mlx",
    feature = "inference-llamacpp"
))]
fn huggingface_token_from_pairs<I, K, V>(vars: I) -> Option<String>
where
    I: IntoIterator<Item = (K, V)>,
    K: AsRef<str>,
    V: AsRef<str>,
{
    for (name, value) in vars {
        if !HUGGINGFACE_TOKEN_ENV_VARS.contains(&name.as_ref()) {
            continue;
        }
        if let Some(token) = normalize_huggingface_token(value.as_ref()) {
            return Some(token);
        }
    }
    None
}

#[cfg(any(
    feature = "embedding-ort",
    feature = "embedding-candle",
    feature = "embedding-mlx",
    feature = "inference-mlx",
    feature = "inference-llamacpp"
))]
pub fn normalize_huggingface_token(token: impl AsRef<str>) -> Option<String> {
    let token = token.as_ref().trim();
    (!token.is_empty()).then(|| token.to_string())
}

#[cfg(any(
    feature = "embedding-ort",
    feature = "embedding-candle",
    feature = "embedding-mlx",
    feature = "inference-mlx",
    feature = "inference-llamacpp"
))]
#[must_use]
pub fn onnx_model_candidates<'a>(task_specific: &'a [&'a str]) -> Vec<&'a str> {
    let mut candidates = BASE_ONNX_MODEL_CANDIDATES.to_vec();
    for candidate in task_specific {
        if !candidates.contains(candidate) {
            candidates.push(candidate);
        }
    }
    candidates
}

#[cfg(any(
    feature = "embedding-ort",
    feature = "embedding-candle",
    feature = "embedding-mlx",
    feature = "inference-mlx",
    feature = "inference-llamacpp"
))]
///
/// # Errors
///
/// Returns an error when the operation cannot be completed.
pub fn resolve_local_onnx_model_path(
    path: &Path,
    candidates: &[&str],
) -> Result<PathBuf, DripstoneError> {
    if path.is_file() {
        return Ok(path.to_path_buf());
    }

    for candidate in candidates {
        let resolved = path.join(candidate);
        if resolved.is_file() {
            return Ok(resolved);
        }
    }

    Err(DripstoneError::storage(format!(
        "failed to find ONNX model under '{}'; tried: {}",
        path.display(),
        candidates.join(", ")
    )))
}

/// Loader for fetching files from `HuggingFace` Hub.
///
/// By default this uses curl via `model-download`. Enable the `hf-hub` feature
/// to prefer the ureq-based Hub client (with curl as fallback).
#[cfg(any(
    feature = "embedding-ort",
    feature = "embedding-candle",
    feature = "embedding-mlx",
    feature = "inference-mlx",
    feature = "inference-llamacpp"
))]
pub struct HuggingFaceLoader {
    #[cfg(feature = "hf-hub")]
    api: Api,
    repo_id: String,
    revision: Option<String>,
    token: Option<String>,
    has_token: bool,
    #[cfg(feature = "hf-hub")]
    allow_curl_fallback: bool,
    download_options: DownloadOptions,
}

#[cfg(any(
    feature = "embedding-ort",
    feature = "embedding-candle",
    feature = "embedding-mlx",
    feature = "inference-mlx",
    feature = "inference-llamacpp"
))]
impl HuggingFaceLoader {
    /// Create a new `HuggingFace` loader for the specified repository.
    ///
    /// # Errors
    ///
    /// Returns an error when the operation cannot be completed.
    pub fn new(repo_id: &str, revision: Option<&str>) -> Result<Self, DripstoneError> {
        Self::new_with_token(repo_id, revision, huggingface_token_from_env())
    }

    /// Create a new `HuggingFace` loader with an explicit bearer token.
    ///
    /// # Errors
    ///
    /// Returns an error when the operation cannot be completed.
    pub fn new_with_token(
        repo_id: &str,
        revision: Option<&str>,
        token: Option<String>,
    ) -> Result<Self, DripstoneError> {
        let token = token.and_then(normalize_huggingface_token);
        let has_token = token.is_some();
        #[cfg(feature = "hf-hub")]
        let api = ApiBuilder::new()
            .with_token(token.clone())
            .build()
            .map_err(|e| {
                DripstoneError::storage_with_source("failed to create HuggingFace API", e)
            })?;

        let download_options = DownloadOptions::default().with_huggingface_token(token.as_deref());
        Ok(Self {
            #[cfg(feature = "hf-hub")]
            api,
            repo_id: repo_id.to_string(),
            revision: revision.map(std::string::ToString::to_string),
            token,
            has_token,
            #[cfg(feature = "hf-hub")]
            allow_curl_fallback: true,
            download_options,
        })
    }

    /// Enable or disable the curl fallback used when `hf_hub` fetch fails.
    ///
    /// When the `hf-hub` feature is disabled, downloads always use curl and this
    /// setting has no effect.
    #[must_use]
    pub fn with_curl_fallback(self, enabled: bool) -> Self {
        #[cfg(feature = "hf-hub")]
        {
            let mut this = self;
            this.allow_curl_fallback = enabled;
            this
        }
        #[cfg(not(feature = "hf-hub"))]
        {
            let _ = enabled;
            self
        }
    }

    /// Override download safety options used by the curl download path.
    #[must_use]
    pub fn with_download_options(mut self, mut options: DownloadOptions) -> Self {
        if options
            .headers
            .iter()
            .all(|(name, _)| !name.eq_ignore_ascii_case("authorization"))
        {
            options = options.with_huggingface_token(self.token.as_deref());
        }
        self.download_options = options;
        self
    }

    /// Get a file from the repository.
    ///
    /// # Errors
    ///
    /// Returns an error when the operation cannot be completed.
    pub fn get_file(&self, filename: &str) -> Result<std::path::PathBuf, DripstoneError> {
        #[cfg(feature = "hf-hub")]
        {
            use hf_hub::{Repo, RepoType};

            let repo = self.revision.as_ref().map_or_else(
                || self.api.model(self.repo_id.clone()),
                |revision| {
                    self.api.repo(Repo::with_revision(
                        self.repo_id.clone(),
                        RepoType::Model,
                        revision.clone(),
                    ))
                },
            );

            match repo.get(filename) {
                Ok(path) => return Ok(path),
                Err(hub_error) => {
                    if !self.allow_curl_fallback {
                        return Err(DripstoneError::storage_with_source(
                            format!(
                                "failed to download file '{}' from HuggingFace repo '{}' at revision '{}' (token provided: {}; curl fallback disabled)",
                                filename,
                                self.repo_id,
                                self.revision.as_deref().unwrap_or("main"),
                                if self.has_token { "yes" } else { "no" },
                            ),
                            hub_error,
                        ));
                    }
                    return self.download_file_with_curl(filename).map_err(|curl_error| {
                        DripstoneError::storage_with_source(
                            format!(
                                "failed to download file '{}' from HuggingFace repo '{}' at revision '{}' (token provided: {}); curl fallback failed: {}",
                                filename,
                                self.repo_id,
                                self.revision.as_deref().unwrap_or("main"),
                                if self.has_token { "yes" } else { "no" },
                                curl_error
                            ),
                            hub_error,
                        )
                    });
                }
            }
        }

        #[cfg(not(feature = "hf-hub"))]
        {
            self.download_file_with_curl(filename).map_err(|curl_error| {
                DripstoneError::storage(format!(
                    "failed to download file '{}' from HuggingFace repo '{}' at revision '{}' (token provided: {}): {}",
                    filename,
                    self.repo_id,
                    self.revision.as_deref().unwrap_or("main"),
                    if self.has_token { "yes" } else { "no" },
                    curl_error
                ))
            })
        }
    }

    fn download_file_with_curl(&self, filename: &str) -> Result<PathBuf, DripstoneError> {
        let target = self.cache_fallback_path(filename)?;
        let url = huggingface_resolve_url(&self.repo_id, self.revision.as_deref(), filename);
        ensure_downloaded(&url, &target, &self.download_options)
    }

    fn cache_fallback_path(&self, filename: &str) -> Result<PathBuf, DripstoneError> {
        let mut path = crate::loader::cache_root()
            .join(sanitize_cache_component(&self.repo_id))
            .join(sanitize_cache_component(
                self.revision.as_deref().unwrap_or("main"),
            ));

        for component in filename.split('/') {
            if component.is_empty() || component == "." || component == ".." {
                return Err(DripstoneError::storage(format!(
                    "invalid HuggingFace filename '{filename}'"
                )));
            }
            path.push(component);
        }

        Ok(path)
    }

    /// Try to get a file using multiple fallback filenames.
    ///
    /// Returns the first file that exists, or an error if none are found.
    ///
    /// # Errors
    ///
    /// Returns an error when the operation cannot be completed.
    pub fn get_file_with_fallbacks(
        &self,
        filenames: &[&str],
    ) -> Result<std::path::PathBuf, DripstoneError> {
        let mut last_error = None;
        for filename in filenames {
            match self.get_file(filename) {
                Ok(path) => return Ok(path),
                Err(e) => {
                    last_error = Some(e);
                }
            }
        }

        Err(DripstoneError::storage(format!(
            "failed to find any of the following files in HuggingFace repository: {}. Last error: {}",
            filenames.join(", "),
            last_error.map_or_else(|| "unknown".to_string(), |e| e.to_string())
        )))
    }

    /// Resolve an ONNX model from common `HuggingFace` export locations plus
    /// task-specific candidates.
    ///
    /// # Errors
    ///
    /// Returns an error when the operation cannot be completed.
    pub fn get_onnx_model_file(
        &self,
        task_specific: &[&str],
    ) -> Result<std::path::PathBuf, DripstoneError> {
        let candidates = onnx_model_candidates(task_specific);
        self.get_file_with_fallbacks(&candidates)
    }
}

#[cfg(any(
    feature = "embedding-ort",
    feature = "embedding-candle",
    feature = "embedding-mlx",
    feature = "inference-mlx",
    feature = "inference-llamacpp"
))]
fn sanitize_cache_component(component: &str) -> String {
    let sanitized: String = component
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || matches!(ch, '.' | '_' | '-') {
                ch
            } else {
                '-'
            }
        })
        .collect();

    if sanitized.is_empty() {
        "unknown".to_string()
    } else {
        sanitized
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn onnx_candidates_keep_common_layouts_first() {
        let candidates = onnx_model_candidates(&["generation_model.onnx", "onnx/model.onnx"]);

        assert_eq!(
            &candidates[..4],
            [
                "model.onnx",
                "model_optimized.onnx",
                "onnx/model.onnx",
                "onnx/model_optimized.onnx",
            ]
        );
        assert!(candidates.contains(&"generation_model.onnx"));
        assert_eq!(
            candidates
                .iter()
                .filter(|candidate| **candidate == "onnx/model.onnx")
                .count(),
            1
        );
    }

    #[test]
    fn local_onnx_resolution_uses_nested_exports() {
        let temp = tempfile::tempdir().expect("tempdir");
        let onnx_dir = temp.path().join("onnx");
        std::fs::create_dir(&onnx_dir).expect("create onnx dir");
        let model = onnx_dir.join("model.onnx");
        std::fs::write(&model, b"not a real model").expect("write model");

        let candidates = onnx_model_candidates(&[]);
        let resolved =
            resolve_local_onnx_model_path(temp.path(), &candidates).expect("resolve local model");

        assert_eq!(resolved, model);
    }

    #[test]
    fn huggingface_token_uses_first_supported_non_empty_value() {
        let token = huggingface_token_from_pairs([
            ("UNRELATED", "ignored"),
            ("HF_TOKEN", "   "),
            ("HUGGING_FACE_HUB_TOKEN", "hf_test"),
            ("HF_HUB_TOKEN", "hf_later"),
        ]);

        assert_eq!(token.as_deref(), Some("hf_test"));
    }

    #[test]
    fn explicit_huggingface_token_is_trimmed_or_ignored() {
        assert_eq!(
            normalize_huggingface_token("  hf_explicit  ").as_deref(),
            Some("hf_explicit")
        );
        assert_eq!(normalize_huggingface_token("   "), None);
    }
}
