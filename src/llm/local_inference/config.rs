use super::provider::LocalInferenceProvider;
use crate::llm::config::CommonProviderOptions;
use dripstone::inference::{DeviceConfig, GenerationParams, InferenceBackendType};
use dripstone::loader::{DEFAULT_MAX_DOWNLOAD_BYTES, ModelSource};
use std::path::PathBuf;

/// Bootstrap settings for materializing model artifacts before backend init.
///
/// Prefer [`LocalInferenceProvider::ensure_catalog_model`] and
/// [`LocalInferenceProvider::fit_catalog_model`] when selecting what to download;
/// this config is the env/CLI path that copies/fetches files into place.
#[cfg(feature = "local-inference-base")]
#[derive(Clone)]
#[expect(
    clippy::struct_excessive_bools,
    reason = "public bootstrap configuration fields are an API compatibility surface"
)]
pub struct LocalInferenceBootstrapConfig {
    /// Whether bootstrap should run before backend initialization.
    pub enabled: bool,
    /// Create `HuggingFace` cache directory (`HF_HOME` or ~/.cache/huggingface).
    pub create_huggingface_cache_dir: bool,
    /// Optional explicit `HuggingFace` cache directory.
    pub huggingface_cache_dir: Option<PathBuf>,
    /// Continue provider initialization if bootstrap fails.
    ///
    /// Defaults to `false` so missing/failed downloads surface clearly. Set
    /// `CALICHE_LOCAL_INFERENCE_CONTINUE_ON_ERROR=1` for soft-fail behavior.
    pub continue_on_error: bool,
    /// Allow curl fallback when `hf_hub` file fetch fails.
    pub allow_curl_fallback: bool,
    /// Reject individual artifact downloads larger than this many bytes.
    /// `None` disables the limit. Defaults to 20 GiB.
    pub max_download_bytes: Option<u64>,
    /// Optional explicit `HuggingFace` bearer token for private or gated model downloads.
    pub huggingface_token: Option<String>,
    /// Candle bootstrap config.
    pub candle: Option<CandleBootstrapConfig>,
    /// ORT bootstrap config.
    pub ort: Option<OrtBootstrapConfig>,
    /// GGUF / llama.cpp bootstrap config.
    pub gguf: Option<GgufBootstrapConfig>,
}

#[cfg(feature = "local-inference-base")]
impl std::fmt::Debug for LocalInferenceBootstrapConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LocalInferenceBootstrapConfig")
            .field("enabled", &self.enabled)
            .field(
                "create_huggingface_cache_dir",
                &self.create_huggingface_cache_dir,
            )
            .field("huggingface_cache_dir", &self.huggingface_cache_dir)
            .field("continue_on_error", &self.continue_on_error)
            .field("allow_curl_fallback", &self.allow_curl_fallback)
            .field("max_download_bytes", &self.max_download_bytes)
            .field(
                "huggingface_token",
                &self.huggingface_token.as_ref().map(|_| "<redacted>"),
            )
            .field("candle", &self.candle)
            .field("ort", &self.ort)
            .field("gguf", &self.gguf)
            .finish()
    }
}

/// Candle artifact bootstrap configuration.
#[cfg(feature = "local-inference-base")]
#[derive(Debug, Clone)]
pub struct CandleBootstrapConfig {
    /// Local directory where Candle model artifacts should be stored.
    pub local_dir: PathBuf,
    /// Optional override repository for Candle downloads.
    /// If omitted, `HuggingFace` source from `LocalInferenceConfig` is used.
    pub repo: Option<String>,
    /// Optional override revision for Candle downloads.
    pub revision: Option<String>,
    /// Rewrite `LocalInferenceConfig.source` to local directory after bootstrap.
    pub redirect_source_to_local: bool,
}

/// ORT artifact bootstrap configuration.
#[cfg(feature = "local-inference-base")]
#[derive(Debug, Clone)]
pub struct OrtBootstrapConfig {
    /// Local directory where ORT model artifacts should be stored.
    pub local_dir: PathBuf,
    /// Repository to download ORT artifacts from.
    pub repo: String,
    /// Optional revision for ORT downloads.
    pub revision: Option<String>,
    /// Candidate ONNX files checked in order.
    pub onnx_candidates: Vec<String>,
    /// Rewrite `LocalInferenceConfig.source` to local model.onnx after bootstrap.
    pub redirect_source_to_local: bool,
}

/// GGUF / llama.cpp artifact bootstrap configuration.
#[cfg(feature = "local-inference-base")]
#[derive(Debug, Clone)]
pub struct GgufBootstrapConfig {
    /// Local directory where the `.gguf` file should be stored.
    pub local_dir: PathBuf,
    /// Optional override repository for GGUF downloads.
    pub repo: Option<String>,
    /// Optional override revision for GGUF downloads.
    pub revision: Option<String>,
    /// Remote filename inside the repository (for example `model.Q4_K_M.gguf`).
    pub filename: Option<String>,
    /// Rewrite `LocalInferenceConfig.source` to the local `.gguf` after bootstrap.
    pub redirect_source_to_local: bool,
}

#[cfg(feature = "local-inference-base")]
impl Default for LocalInferenceBootstrapConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            create_huggingface_cache_dir: true,
            huggingface_cache_dir: None,
            continue_on_error: false,
            allow_curl_fallback: true,
            max_download_bytes: Some(DEFAULT_MAX_DOWNLOAD_BYTES),
            huggingface_token: None,
            candle: Some(CandleBootstrapConfig::default()),
            ort: Some(OrtBootstrapConfig::default()),
            gguf: Some(GgufBootstrapConfig::default()),
        }
    }
}

#[cfg(feature = "local-inference-base")]
impl Default for CandleBootstrapConfig {
    fn default() -> Self {
        Self {
            local_dir: PathBuf::from("./models/tiny-random-llama-candle"),
            repo: None,
            revision: None,
            redirect_source_to_local: true,
        }
    }
}

#[cfg(feature = "local-inference-base")]
impl Default for OrtBootstrapConfig {
    fn default() -> Self {
        Self {
            local_dir: PathBuf::from("./models/qwen2.5-1.5b-instruct"),
            repo: "Xenova/distilgpt2".to_string(),
            revision: None,
            onnx_candidates: vec![
                "model.onnx".to_string(),
                "onnx/model.onnx".to_string(),
                "model_optimized.onnx".to_string(),
                "onnx/model_optimized.onnx".to_string(),
                "generation_model.onnx".to_string(),
                "onnx/generation_model.onnx".to_string(),
            ],
            redirect_source_to_local: true,
        }
    }
}

#[cfg(feature = "local-inference-base")]
impl Default for GgufBootstrapConfig {
    fn default() -> Self {
        Self {
            local_dir: PathBuf::from("./models/stories15m-gguf"),
            repo: Some("ggml-org/models-moved".to_string()),
            revision: None,
            filename: Some("tinyllamas/stories15M-q4_0.gguf".to_string()),
            redirect_source_to_local: true,
        }
    }
}

/// Configuration for `LocalInferenceProvider`
#[cfg(feature = "local-inference-base")]
#[derive(Debug, Clone)]
pub struct LocalInferenceConfig {
    /// Model source (local path or `HuggingFace` repo)
    pub source: ModelSource,
    /// Backend type to use
    pub backend: InferenceBackendType,
    /// Device/runtime preference (Auto, Mlx, Cuda, Metal, Cpu)
    pub device: DeviceConfig,
    /// Default generation parameters
    pub generation: GenerationParams,
    /// Common provider options
    pub common: CommonProviderOptions,
    /// Optional bootstrap behavior for provisioning model artifacts.
    pub bootstrap: Option<LocalInferenceBootstrapConfig>,
}

#[cfg(feature = "local-inference-base")]
impl LocalInferenceConfig {
    /// Create a new `LocalInferenceConfig` with default settings
    #[must_use]
    pub fn new(source: ModelSource, backend: InferenceBackendType) -> Self {
        let device = if matches!(backend, InferenceBackendType::Mlx) {
            DeviceConfig::Mlx
        } else {
            DeviceConfig::default()
        };
        Self {
            source,
            backend,
            device,
            generation: GenerationParams::default(),
            common: CommonProviderOptions::default(),
            bootstrap: None,
        }
    }

    /// Set device/runtime preference (Auto, Mlx, Cuda, Metal, Cpu)
    #[must_use]
    pub const fn with_device(mut self, device: DeviceConfig) -> Self {
        self.device = device;
        self
    }

    /// Set the inference backend and keep bootstrap settings aligned.
    #[must_use]
    pub fn with_backend(mut self, backend: InferenceBackendType) -> Self {
        self.backend = backend;
        if matches!(backend, InferenceBackendType::Mlx)
            && matches!(self.device, DeviceConfig::Cpu | DeviceConfig::Auto)
        {
            self.device = DeviceConfig::Mlx;
        }
        if self.bootstrap.is_some() {
            self.bootstrap = Some(LocalInferenceProvider::bootstrap_for_source(
                &self.source,
                self.backend,
            ));
        }
        self
    }

    /// Set generation parameters
    #[must_use]
    pub fn with_generation_params(mut self, params: GenerationParams) -> Self {
        self.generation = params;
        self
    }

    /// Set common provider options
    #[must_use]
    pub fn with_common_options(mut self, options: CommonProviderOptions) -> Self {
        self.common = options;
        self
    }

    /// Set temperature
    #[must_use]
    pub const fn with_temperature(mut self, temperature: f32) -> Self {
        self.generation.temperature = temperature;
        self
    }

    /// Set max tokens
    #[must_use]
    pub const fn with_max_tokens(mut self, max_tokens: usize) -> Self {
        self.generation.max_tokens = max_tokens;
        self
    }

    /// Enable local artifact bootstrap with default settings.
    #[must_use]
    pub fn with_default_bootstrap(mut self) -> Self {
        self.bootstrap = Some(LocalInferenceBootstrapConfig::default());
        self
    }

    /// Set custom bootstrap configuration.
    #[must_use]
    pub fn with_bootstrap(mut self, bootstrap: LocalInferenceBootstrapConfig) -> Self {
        self.bootstrap = Some(bootstrap);
        self
    }

    /// Download artifacts from a specific `HuggingFace` repository/revision.
    ///
    /// This also rewrites the model source to the same repository so the
    /// backend and bootstrap target stay in sync.
    #[must_use]
    pub fn with_download_target(
        mut self,
        repo: impl Into<String>,
        revision: Option<String>,
    ) -> Self {
        self.source = ModelSource::huggingface(repo, revision);
        self.bootstrap = Some(LocalInferenceProvider::bootstrap_for_source(
            &self.source,
            self.backend,
        ));
        self
    }

    /// Store downloaded local inference artifacts in this exact directory.
    #[must_use]
    pub fn with_model_storage_dir(mut self, local_dir: impl Into<PathBuf>) -> Self {
        let mut bootstrap = self.bootstrap.take().unwrap_or_else(|| {
            LocalInferenceProvider::bootstrap_for_source(&self.source, self.backend)
        });
        bootstrap.set_model_storage_dir(local_dir);
        self.bootstrap = Some(bootstrap);
        self
    }

    /// Set `HuggingFace` cache directory used by bootstrap setup.
    #[must_use]
    pub fn with_huggingface_cache_dir(mut self, cache_dir: impl Into<PathBuf>) -> Self {
        let mut bootstrap = self.bootstrap.take().unwrap_or_else(|| {
            LocalInferenceProvider::bootstrap_for_source(&self.source, self.backend)
        });
        bootstrap.huggingface_cache_dir = Some(cache_dir.into());
        self.bootstrap = Some(bootstrap);
        self
    }

    /// Set an explicit `HuggingFace` token for local artifact bootstrap downloads.
    #[must_use]
    pub fn with_huggingface_token(mut self, token: impl Into<String>) -> Self {
        let mut bootstrap = self.bootstrap.take().unwrap_or_default();
        bootstrap = bootstrap.with_huggingface_token(token);
        self.bootstrap = Some(bootstrap);
        self
    }

    /// Disable bootstrap even if previously enabled.
    #[must_use]
    pub fn without_bootstrap(mut self) -> Self {
        self.bootstrap = None;
        self
    }
}

#[cfg(feature = "local-inference-base")]
impl Default for LocalInferenceConfig {
    fn default() -> Self {
        Self::new(
            ModelSource::huggingface("hf-internal-testing/tiny-random-LlamaForCausalLM", None),
            InferenceBackendType::Candle,
        )
    }
}
