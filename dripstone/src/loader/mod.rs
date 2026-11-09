//! Shared utilities for loading models from local files or `HuggingFace` Hub.
//!
//! This module provides reusable functionality for:
//! - Safe resumable downloads with size/hash checks
//! - Discoverable model catalog entries
//! - Detecting and handling local vs remote model sources
//! - Loading models from `HuggingFace` Hub
//! - Loading tokenizers
//! - Building ONNX Runtime sessions

pub mod catalog;
pub mod curl;
pub mod disk;
pub mod download;
pub mod fit;
pub mod resources;
pub mod source;

#[cfg(any(
    feature = "embedding-ort",
    feature = "embedding-candle",
    feature = "embedding-mlx",
    feature = "inference-mlx",
    feature = "inference-llamacpp"
))]
pub mod huggingface;
#[cfg(feature = "embedding-ort")]
pub mod ort_session;
#[cfg(any(
    feature = "embedding-ort",
    feature = "embedding-candle",
    feature = "embedding-mlx",
    feature = "inference-mlx"
))]
pub mod tokenizer;

pub use catalog::{
    EnsureModelOptions, HubFileFetcher, ModelArtifact, ModelKind, ModelSpec, builtin_catalog,
    ensure_model, find_model, list_models_with_tags,
};
pub use disk::{
    CACHE_DIR_ENV, DEFAULT_RESERVE_FREE_BYTES, DISK_RESERVE_ENV, available_disk_bytes, cache_root,
    ensure_disk_space, probe_remote_content_length, reserve_free_bytes_from_env,
};
pub use download::{
    DEFAULT_MAX_DOWNLOAD_BYTES, DownloadOptions, atomic_copy, ensure_downloaded,
    huggingface_resolve_url, sanitize_relative_path, sha256_hex_file, verify_existing_file,
};
pub use fit::{
    CapabilityProbe, DefaultCapabilityProbe, FitOptions, FitStatus, HostCapabilities, ModelFit,
    estimate_model_fit, estimate_model_resources, fit_catalog, host_capabilities,
};
#[cfg(any(
    feature = "embedding-ort",
    feature = "embedding-candle",
    feature = "embedding-mlx",
    feature = "inference-mlx",
    feature = "inference-llamacpp"
))]
pub use huggingface::{HuggingFaceLoader, huggingface_token_from_env, normalize_huggingface_token};
#[cfg(feature = "embedding-ort")]
pub use ort_session::{OrtSessionConfig, build_ort_session};
pub use resources::{
    DEFAULT_RESERVE_RAM_BYTES, HostMemory, ModelResourceHint, RESERVE_RAM_ENV, ResourceBudget,
    ResourceEstimate, default_kv_bytes_per_token, ensure_resource_budget, estimate_runtime_memory,
    max_safe_n_ctx, reserve_ram_bytes_from_env,
};
pub use source::ModelSource;
#[cfg(any(
    feature = "embedding-ort",
    feature = "embedding-candle",
    feature = "embedding-mlx",
    feature = "inference-mlx"
))]
pub use tokenizer::{load_tokenizer, load_tokenizer_from_path};
