//! Discoverable model catalog and provisioning helpers.
//!
//! The catalog is pure metadata plus thin ensure helpers so callers can
//! list/download models without baking repo URLs into bootstrap code.

use crate::errors::DripstoneError;
use crate::loader::download::{
    DownloadOptions, atomic_copy, ensure_downloaded, huggingface_resolve_url,
    sanitize_relative_path,
};
use crate::loader::resources::ModelResourceHint;
use std::path::{Path, PathBuf};

/// Backend family a catalog entry targets.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelKind {
    /// Candle / SafeTensors causal LM artifacts.
    Candle,
    /// ONNX Runtime generation artifacts.
    Ort,
    /// GGUF / llama.cpp generation artifacts.
    Gguf,
    /// Embedding model artifacts.
    Embedding,
}

/// One file that must be present for a catalog entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ModelArtifact {
    /// Path relative to the HuggingFace repository root.
    pub remote_path: &'static str,
    /// Optional lowercase hex SHA-256 digest.
    pub sha256: Option<&'static str>,
    /// Optional per-artifact size cap in bytes.
    pub max_bytes: Option<u64>,
    /// Optional known size used for disk preflight (skips remote HEAD when set).
    pub expected_bytes: Option<u64>,
}

/// A downloadable / discoverable model specification.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ModelSpec {
    /// Stable catalog identifier (for example `tiny-random-llama`).
    pub id: &'static str,
    /// Human-readable label.
    pub display_name: &'static str,
    /// HuggingFace repository id (`org/name`).
    pub repo: &'static str,
    /// Optional pinned revision (tag, branch, or commit).
    pub revision: Option<&'static str>,
    /// Target runtime family.
    pub kind: ModelKind,
    /// Required repository files.
    pub artifacts: &'static [ModelArtifact],
    /// Search / filter tags.
    pub tags: &'static [&'static str],
    /// Optional memory / context hints for capability probes.
    pub resources: ModelResourceHint,
}

/// Options for [`ensure_model`].
#[derive(Debug, Clone)]
pub struct EnsureModelOptions {
    /// Shared download safety options (headers, resume, default max bytes).
    pub download: DownloadOptions,
    /// Prefer `hf_hub` cache + atomic copy when a fetcher is provided.
    pub prefer_hub_cache: bool,
}

impl Default for EnsureModelOptions {
    fn default() -> Self {
        Self {
            download: DownloadOptions::default(),
            prefer_hub_cache: true,
        }
    }
}

/// Callback used to resolve a repository file through `hf_hub` when available.
pub type HubFileFetcher<'a> = dyn Fn(&str) -> Result<PathBuf, DripstoneError> + Send + Sync + 'a;

/// Built-in catalog of small, useful bootstrap models.
#[must_use]
pub fn builtin_catalog() -> &'static [ModelSpec] {
    &BUILTIN_CATALOG
}

/// Find a catalog entry by id (case-sensitive).
#[must_use]
pub fn find_model(id: &str) -> Option<&'static ModelSpec> {
    builtin_catalog().iter().find(|spec| spec.id == id)
}

/// List catalog entries matching all provided tags (case-insensitive).
#[must_use]
pub fn list_models_with_tags(tags: &[&str]) -> Vec<&'static ModelSpec> {
    builtin_catalog()
        .iter()
        .filter(|spec| {
            tags.iter().all(|wanted| {
                let wanted = wanted.to_ascii_lowercase();
                spec.tags
                    .iter()
                    .any(|tag| tag.eq_ignore_ascii_case(&wanted))
                    || matches!(
                        (&wanted[..], spec.kind),
                        ("candle", ModelKind::Candle)
                            | ("ort", ModelKind::Ort)
                            | ("onnx", ModelKind::Ort)
                            | ("gguf", ModelKind::Gguf)
                            | ("llamacpp", ModelKind::Gguf)
                            | ("llama.cpp", ModelKind::Gguf)
                            | ("embedding", ModelKind::Embedding)
                    )
            })
        })
        .collect()
}

/// Ensure every artifact for `spec` exists under `dest_dir`.
///
/// When `hub_fetch` is provided and `prefer_hub_cache` is true, files are pulled
/// through the hub cache then atomically copied. Otherwise files are downloaded
/// directly to `dest_dir` with resume, locking, and integrity checks.
///
/// # Errors
///
/// Returns an error when path sanitization, download, copy, or verification fails.
pub fn ensure_model<'a>(
    spec: &ModelSpec,
    dest_dir: &Path,
    options: &EnsureModelOptions,
    hub_fetch: Option<&'a HubFileFetcher<'a>>,
) -> Result<PathBuf, DripstoneError> {
    fs_create_dir_all(dest_dir)?;

    for artifact in spec.artifacts {
        let dest = sanitize_relative_path(dest_dir, artifact.remote_path)?;
        if dest.is_file() {
            let mut verify_opts = options.download.clone();
            apply_artifact_limits(&mut verify_opts, artifact);
            crate::loader::download::verify_existing_file(&dest, &verify_opts)?;
            continue;
        }

        let mut download_opts = options.download.clone();
        apply_artifact_limits(&mut download_opts, artifact);

        let fetched = if options.prefer_hub_cache {
            hub_fetch.and_then(|fetch| fetch(artifact.remote_path).ok())
        } else {
            None
        };

        if let Some(source) = fetched {
            atomic_copy(&source, &dest, &download_opts)?;
        } else {
            let url = huggingface_resolve_url(spec.repo, spec.revision, artifact.remote_path);
            ensure_downloaded(&url, &dest, &download_opts)?;
        }
    }

    Ok(dest_dir.to_path_buf())
}

fn apply_artifact_limits(options: &mut DownloadOptions, artifact: &ModelArtifact) {
    if let Some(max_bytes) = artifact.max_bytes {
        options.max_bytes = Some(max_bytes);
    }
    if let Some(sha256) = artifact.sha256 {
        options.expected_sha256 = Some(sha256.to_string());
    }
    if let Some(expected_bytes) = artifact.expected_bytes {
        options.expected_bytes = Some(expected_bytes);
        // Catalog size hints make the HEAD probe redundant.
        options.check_remote_size = false;
    }
}

fn fs_create_dir_all(path: &Path) -> Result<(), DripstoneError> {
    std::fs::create_dir_all(path).map_err(|error| {
        DripstoneError::storage_with_source(
            format!("failed to create model directory '{}'", path.display()),
            error,
        )
    })
}

static BUILTIN_CATALOG: [ModelSpec; 4] = [
    ModelSpec {
        id: "tiny-random-llama",
        display_name: "Tiny Random Llama (Candle smoke test)",
        repo: "hf-internal-testing/tiny-random-LlamaForCausalLM",
        revision: None,
        kind: ModelKind::Candle,
        artifacts: &[
            ModelArtifact {
                remote_path: "config.json",
                sha256: None,
                max_bytes: Some(1024 * 1024),
                expected_bytes: None,
            },
            ModelArtifact {
                remote_path: "tokenizer.json",
                sha256: None,
                max_bytes: Some(16 * 1024 * 1024),
                expected_bytes: None,
            },
            ModelArtifact {
                remote_path: "model.safetensors",
                sha256: None,
                max_bytes: Some(64 * 1024 * 1024),
                expected_bytes: None,
            },
        ],
        tags: &["candle", "smoke", "tiny", "llama"],
        resources: ModelResourceHint {
            weight_bytes: Some(2 * 1024 * 1024),
            kv_bytes_per_token: Some(256),
            recommended_n_ctx: Some(512),
            min_ram_bytes: Some(256 * 1024 * 1024),
        },
    },
    ModelSpec {
        id: "tinyllama-1.1b-chat",
        display_name: "TinyLlama 1.1B Chat (Candle)",
        repo: "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        revision: None,
        kind: ModelKind::Candle,
        artifacts: &[
            ModelArtifact {
                remote_path: "config.json",
                sha256: None,
                max_bytes: Some(1024 * 1024),
                expected_bytes: None,
            },
            ModelArtifact {
                remote_path: "tokenizer.json",
                sha256: None,
                max_bytes: Some(16 * 1024 * 1024),
                expected_bytes: None,
            },
            ModelArtifact {
                remote_path: "model.safetensors",
                sha256: None,
                max_bytes: Some(3 * 1024 * 1024 * 1024),
                expected_bytes: Some(2_200_000_000),
            },
        ],
        tags: &["candle", "chat", "tinyllama"],
        resources: ModelResourceHint {
            weight_bytes: Some(2_200_000_000),
            kv_bytes_per_token: Some(4 * 1024),
            recommended_n_ctx: Some(2048),
            min_ram_bytes: Some(3_500_000_000),
        },
    },
    ModelSpec {
        id: "distilgpt2-onnx",
        display_name: "DistilGPT2 ONNX (ORT smoke test)",
        repo: "Xenova/distilgpt2",
        revision: None,
        kind: ModelKind::Ort,
        artifacts: &[
            ModelArtifact {
                remote_path: "tokenizer.json",
                sha256: None,
                max_bytes: Some(16 * 1024 * 1024),
                expected_bytes: None,
            },
            ModelArtifact {
                remote_path: "onnx/model.onnx",
                sha256: None,
                max_bytes: Some(512 * 1024 * 1024),
                expected_bytes: Some(330_000_000),
            },
        ],
        tags: &["ort", "onnx", "smoke", "gpt2"],
        resources: ModelResourceHint {
            weight_bytes: Some(330_000_000),
            kv_bytes_per_token: Some(1024),
            recommended_n_ctx: Some(1024),
            min_ram_bytes: Some(768 * 1024 * 1024),
        },
    },
    ModelSpec {
        id: "stories15m-gguf",
        display_name: "TinyStories 15M GGUF (llama.cpp smoke test)",
        repo: "ggml-org/models-moved",
        revision: None,
        kind: ModelKind::Gguf,
        artifacts: &[ModelArtifact {
            remote_path: "tinyllamas/stories15M-q4_0.gguf",
            sha256: None,
            max_bytes: Some(25 * 1024 * 1024),
            expected_bytes: Some(19_077_344),
        }],
        tags: &["gguf", "llamacpp", "smoke", "tiny"],
        resources: ModelResourceHint {
            weight_bytes: Some(19_077_344),
            kv_bytes_per_token: Some(256),
            recommended_n_ctx: Some(2048),
            min_ram_bytes: Some(256 * 1024 * 1024),
        },
    },
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builtin_catalog_has_stable_ids() {
        let ids: Vec<_> = builtin_catalog().iter().map(|spec| spec.id).collect();
        assert!(ids.contains(&"tiny-random-llama"));
        assert!(ids.contains(&"distilgpt2-onnx"));
        assert!(ids.contains(&"stories15m-gguf"));
        assert_eq!(
            find_model("tiny-random-llama").map(|spec| spec.repo),
            Some("hf-internal-testing/tiny-random-LlamaForCausalLM")
        );
    }

    #[test]
    fn tag_filter_matches_kind_aliases() {
        let ort = list_models_with_tags(&["onnx"]);
        assert!(ort.iter().any(|spec| spec.id == "distilgpt2-onnx"));
        let candle = list_models_with_tags(&["candle", "smoke"]);
        assert!(candle.iter().any(|spec| spec.id == "tiny-random-llama"));
        let gguf = list_models_with_tags(&["gguf", "smoke"]);
        assert!(gguf.iter().any(|spec| spec.id == "stories15m-gguf"));
    }
}
