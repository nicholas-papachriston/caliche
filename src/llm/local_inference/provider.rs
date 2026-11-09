use crate::llm::provider::Provider;
use crate::messages::{
    CompletionContent, CompletionRequest, CompletionResponse, MessageRole, StreamEvent,
};
use async_stream::stream;
use async_trait::async_trait;
use futures_util::Stream as FuturesStream;
use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};
use std::pin::Pin;
use std::sync::Arc;

use super::config::{LocalInferenceBootstrapConfig, LocalInferenceConfig};
use crate::llm::env_compat::{env_flag, env_nonempty, env_var_nonempty};
use dripstone::inference::{DeviceConfig, InferenceBackend, InferenceBackendType, InferenceConfig};
use dripstone::loader::{
    CapabilityProbe, DefaultCapabilityProbe, DownloadOptions, EnsureModelOptions, FitOptions,
    FitStatus, HostCapabilities, HuggingFaceLoader, ModelFit, ModelKind, ModelSource, ModelSpec,
    builtin_catalog, ensure_model, find_model, fit_catalog, host_capabilities,
    list_models_with_tags,
};

/// Local inference provider using `Dripstone` inference backends.
#[cfg(feature = "local-inference-base")]
pub struct LocalInferenceProvider {
    backend: Arc<dyn InferenceBackend>,
    config: LocalInferenceConfig,
}

/// Local inference provider that defers model loading until the first request.
#[cfg(feature = "local-inference-base")]
pub struct LazyLocalInferenceProvider {
    config: LocalInferenceConfig,
    inner: tokio::sync::OnceCell<Arc<LocalInferenceProvider>>,
}

#[cfg(feature = "local-inference-base")]
impl std::fmt::Debug for LazyLocalInferenceProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LazyLocalInferenceProvider")
            .field("config", &self.config)
            .finish_non_exhaustive()
    }
}

#[cfg(feature = "local-inference-base")]
impl LazyLocalInferenceProvider {
    /// Create a lazy provider from an already-resolved config.
    #[must_use]
    pub fn new(config: LocalInferenceConfig) -> Self {
        Self {
            config,
            inner: tokio::sync::OnceCell::new(),
        }
    }

    /// Create a lazy provider from CLI/environment settings without loading the model.
    /// # Errors
    ///
    /// Returns an error if the language-model operation cannot be completed.
    pub fn from_env(model: Option<String>) -> anyhow::Result<Self> {
        Ok(Self::new(LocalInferenceProvider::config_from_env(model)?))
    }

    async fn provider(&self) -> anyhow::Result<Arc<LocalInferenceProvider>> {
        self.inner
            .get_or_try_init(|| async {
                LocalInferenceProvider::with_config(self.config.clone())
                    .await
                    .map(Arc::new)
            })
            .await
            .map(Arc::clone)
    }
}

#[cfg(feature = "local-inference-base")]
impl std::fmt::Debug for LocalInferenceProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LocalInferenceProvider")
            .field("backend", &"<inference backend>")
            .field("config", &self.config)
            .finish()
    }
}

#[cfg(feature = "local-inference-base")]
/// Every backend name the core parser accepts as a selection.
const KNOWN_BACKEND_NAMES: &[&str] = &["candle", "ort", "mlx", "gguf", "llamacpp"];

impl LocalInferenceProvider {
    #[must_use]
    pub const fn default_model_name() -> &'static str {
        "hf-internal-testing/tiny-random-LlamaForCausalLM"
    }

    /// Discover previously downloaded artifacts under configured model roots.
    ///
    /// Filesystem-only (never contacts a host). Prefer
    /// [`Self::catalog_models`] / [`Self::fit_catalog_models`] for choosing what
    /// to download or load; use this for completion of already-materialized paths.
    #[must_use]
    pub fn discover_downloaded_models() -> Vec<String> {
        let mut models = BTreeSet::new();
        for root in Self::model_discovery_roots() {
            Self::collect_local_model_candidates(&root, &mut models);
        }

        models.into_iter().collect()
    }

    /// Built-in downloadable model catalog (ids, repos, tags).
    #[must_use]
    pub fn catalog_models() -> &'static [ModelSpec] {
        builtin_catalog()
    }

    /// Look up a built-in catalog entry by id.
    #[must_use]
    pub fn catalog_find(id: &str) -> Option<&'static ModelSpec> {
        find_model(id)
    }

    /// Filter the built-in catalog by tags (`candle`, `ort`, `smoke`, …).
    #[must_use]
    pub fn catalog_search(tags: &[&str]) -> Vec<&'static ModelSpec> {
        list_models_with_tags(tags)
    }

    /// Probe host RAM/VRAM (and optional disk) for local inference planning.
    ///
    /// # Errors
    ///
    /// Returns an error when the host cannot be measured.
    pub fn probe_host_capabilities(options: FitOptions) -> anyhow::Result<HostCapabilities> {
        host_capabilities(&options)
            .map_err(|error| anyhow::anyhow!("failed to probe host capabilities: {error}"))
    }

    /// Map Caliche local-inference env onto Dripstone fit options.
    ///
    /// Dripstone reads `DRIPSTONE_RESERVE_RAM_BYTES` on its own. This helper
    /// applies `CALICHE_LOCAL_INFERENCE_RESERVE_RAM_BYTES` when set so Caliche
    /// callers keep one env surface.
    #[must_use]
    pub fn fit_options_from_env() -> FitOptions {
        let mut options = FitOptions::default();
        if let Some(bytes) = env_nonempty("CALICHE_LOCAL_INFERENCE_RESERVE_RAM_BYTES")
            .and_then(|value| value.parse::<u64>().ok())
        {
            options = options.with_reserve_ram_bytes(bytes);
        }
        options
    }

    /// Estimate which catalog models fit the current host memory budget.
    ///
    /// # Errors
    ///
    /// Returns an error when the host cannot be measured.
    pub fn fit_catalog_models(options: FitOptions) -> anyhow::Result<Vec<ModelFit>> {
        fit_catalog(&options)
            .map_err(|error| anyhow::anyhow!("failed to fit local model catalog: {error}"))
    }

    /// Estimate fit for one catalog id.
    ///
    /// # Errors
    ///
    /// Returns an error when the id is unknown or the host cannot be measured.
    pub fn fit_catalog_model(id: &str, options: FitOptions) -> anyhow::Result<ModelFit> {
        let spec = find_model(id).ok_or_else(|| {
            anyhow::anyhow!(
                "unknown local inference catalog id '{id}'; known: {}",
                builtin_catalog()
                    .iter()
                    .map(|spec| spec.id)
                    .collect::<Vec<_>>()
                    .join(", ")
            )
        })?;
        DefaultCapabilityProbe
            .estimate_model(spec, &options)
            .map_err(|error| anyhow::anyhow!("failed to fit catalog model '{id}': {error}"))
    }

    /// Catalog models classified as safe to load on this host.
    ///
    /// # Errors
    ///
    /// Returns an error when the host cannot be measured.
    pub fn safe_catalog_models(options: FitOptions) -> anyhow::Result<Vec<ModelFit>> {
        Ok(safe_fits(Self::fit_catalog_models(options)?))
    }

    /// Download a catalog model into `dest_dir` using safe shared download helpers.
    ///
    /// # Errors
    ///
    /// Returns an error when download, path sanitization, or verification fails.
    pub fn ensure_catalog_model(
        id: &str,
        dest_dir: impl AsRef<Path>,
        token: Option<&str>,
    ) -> anyhow::Result<PathBuf> {
        let spec = find_model(id).ok_or_else(|| {
            anyhow::anyhow!(
                "unknown local inference catalog id '{id}'; known: {}",
                builtin_catalog()
                    .iter()
                    .map(|spec| spec.id)
                    .collect::<Vec<_>>()
                    .join(", ")
            )
        })?;
        let options = EnsureModelOptions {
            download: DownloadOptions::default().with_huggingface_token(token),
            prefer_hub_cache: true,
        };

        if let Ok(loader) =
            HuggingFaceLoader::new_with_token(spec.repo, spec.revision, token.map(str::to_string))
        {
            let fetch = |filename: &str| loader.get_file(filename);
            return ensure_model(spec, dest_dir.as_ref(), &options, Some(&fetch)).map_err(
                |error| anyhow::anyhow!("failed to ensure catalog model '{id}': {error}"),
            );
        }

        ensure_model(spec, dest_dir.as_ref(), &options, None)
            .map_err(|error| anyhow::anyhow!("failed to ensure catalog model '{id}': {error}"))
    }

    #[must_use]
    pub fn preferred_downloaded_model() -> Option<String> {
        let mut models = Self::discover_downloaded_models();
        if models.len() == 1 {
            models.pop()
        } else {
            None
        }
    }

    /// Create a local inference provider from CLI/environment settings.
    ///
    /// `model` may be a `HuggingFace` repo, a local directory, or a local ONNX file.
    /// Set `CALICHE_LOCAL_INFERENCE_BACKEND` to `candle`, `ort`, `mlx`, or `gguf`
    /// (`llamacpp` is an alias). When unset, `.onnx` paths select ORT; otherwise Candle.
    /// # Errors
    ///
    /// Returns an error if the language-model operation cannot be completed.
    pub async fn from_env(model: Option<String>) -> anyhow::Result<Self> {
        Self::with_config(Self::config_from_env(model)?).await
    }

    /// Resolve configuration from environment without loading model artifacts.
    /// # Errors
    ///
    /// Returns an error if the language-model operation cannot be completed.
    pub fn config_from_env(model: Option<String>) -> anyhow::Result<LocalInferenceConfig> {
        let model_name = Self::resolve_model_name(model);
        let backend = Self::resolve_backend_for_model(&model_name)?;
        let mut config = LocalInferenceConfig::new(ModelSource::from(model_name.as_str()), backend);

        if let Some(repo) = env_nonempty("CALICHE_LOCAL_INFERENCE_DOWNLOAD_REPO") {
            let revision = env_nonempty("CALICHE_LOCAL_INFERENCE_DOWNLOAD_REVISION");
            config = config.with_download_target(repo, revision);
        }

        if config.source.is_huggingface() || env_flag("CALICHE_LOCAL_INFERENCE_BOOTSTRAP") {
            let bootstrap = Self::bootstrap_for_source(&config.source, backend);
            config = config.with_bootstrap(bootstrap);
        }

        if let Some(local_dir) = env_nonempty("CALICHE_LOCAL_INFERENCE_MODEL_DIR") {
            config = config.with_model_storage_dir(PathBuf::from(local_dir));
        }

        if let Some(cache_dir) = env_nonempty("CALICHE_LOCAL_INFERENCE_HF_CACHE_DIR") {
            config = config.with_huggingface_cache_dir(PathBuf::from(cache_dir));
        }

        if let Ok(max_tokens) = env_var_nonempty("CALICHE_LOCAL_INFERENCE_MAX_TOKENS") {
            let parsed = max_tokens.parse::<usize>().map_err(|e| {
                anyhow::anyhow!(
                    "invalid CALICHE_LOCAL_INFERENCE_MAX_TOKENS value '{max_tokens}': {e}"
                )
            })?;
            config = config.with_max_tokens(parsed);
        }

        if let Some(device) = env_nonempty("CALICHE_LOCAL_INFERENCE_DEVICE") {
            let parsed = Self::parse_device(&device).ok_or_else(|| {
                anyhow::anyhow!(
                    "invalid CALICHE_LOCAL_INFERENCE_DEVICE '{device}' (expected auto|cpu|metal|cuda|mlx)"
                )
            })?;
            config = config.with_device(parsed);
        }

        Ok(config)
    }

    /// Create a new `LocalInferenceProvider` with configuration
    /// # Errors
    ///
    /// Returns an error if the language-model operation cannot be completed.
    pub async fn with_config(config: LocalInferenceConfig) -> anyhow::Result<Self> {
        // Bootstrap downloads/copies are sync filesystem + network work.
        let config = tokio::task::spawn_blocking(move || Self::prepare_artifacts(config))
            .await
            .map_err(|error| anyhow::anyhow!("local inference bootstrap task failed: {error}"))??;

        let backend: Arc<dyn InferenceBackend> = match config.backend {
            InferenceBackendType::Candle => {
                #[cfg(feature = "local-inference-candle")]
                {
                    let inference_config =
                        InferenceConfig::new(config.source.clone(), config.backend)
                            .with_device(config.device)
                            .with_default_params(config.generation.clone());
                    Arc::new(
                        dripstone::inference::CandleInference::new(inference_config)
                            .await
                            .map_err(|e| {
                                anyhow::anyhow!("Failed to create Candle inference backend: {e}")
                            })?,
                    )
                }
                #[cfg(not(feature = "local-inference-candle"))]
                {
                    return Err(anyhow::anyhow!(
                        "Candle local inference requires feature 'local-inference-candle' (or umbrella 'local-inference')"
                    ));
                }
            }
            InferenceBackendType::Ort => {
                #[cfg(feature = "local-inference-ort")]
                {
                    let inference_config =
                        InferenceConfig::new(config.source.clone(), config.backend)
                            .with_device(config.device)
                            .with_default_params(config.generation.clone());
                    Arc::new(
                        dripstone::inference::OrtInference::new(inference_config)
                            .await
                            .map_err(|e| {
                                anyhow::anyhow!(
                                    "Failed to create ONNX Runtime inference backend: {e}"
                                )
                            })?,
                    )
                }
                #[cfg(not(feature = "local-inference-ort"))]
                {
                    return Err(anyhow::anyhow!(
                        "ORT local inference requires feature 'local-inference-ort' (or umbrella 'local-inference')"
                    ));
                }
            }
            InferenceBackendType::Mlx => {
                #[cfg(feature = "inference-mlx")]
                {
                    let inference_config =
                        InferenceConfig::new(config.source.clone(), config.backend)
                            .with_device(config.device)
                            .with_default_params(config.generation.clone());
                    Arc::new(
                        dripstone::inference::MlxInference::new(inference_config)
                            .await
                            .map_err(|e| {
                                anyhow::anyhow!("Failed to create MLX inference backend: {}", e)
                            })?,
                    )
                }
                #[cfg(not(feature = "inference-mlx"))]
                {
                    return Err(anyhow::anyhow!(
                        "MLX inference requires the 'inference-mlx' feature (dripstone and caliche)"
                    ));
                }
            }
            InferenceBackendType::LlamaCpp => {
                #[cfg(feature = "local-inference-llamacpp")]
                {
                    let inference_config =
                        InferenceConfig::new(config.source.clone(), config.backend)
                            .with_device(config.device)
                            .with_default_params(config.generation.clone());
                    Arc::new(
                        dripstone::inference::LlamaCppInference::new(inference_config)
                            .await
                            .map_err(|e| {
                                anyhow::anyhow!("Failed to create llama.cpp inference backend: {e}")
                            })?,
                    )
                }
                #[cfg(not(feature = "local-inference-llamacpp"))]
                {
                    return Err(anyhow::anyhow!(
                        "GGUF/llama.cpp local inference requires feature 'local-inference-llamacpp'"
                    ));
                }
            }
        };

        Ok(Self { backend, config })
    }

    /// Download/copy configured local inference artifacts without initializing
    /// the inference backend.
    /// # Errors
    ///
    /// Returns an error if the language-model operation cannot be completed.
    pub fn prepare_artifacts(
        mut config: LocalInferenceConfig,
    ) -> anyhow::Result<LocalInferenceConfig> {
        if let Some(bootstrap) = config.bootstrap.clone()
            && bootstrap.enabled
        {
            let bootstrap_result = bootstrap.apply(&mut config);
            if let Err(err) = bootstrap_result {
                if bootstrap.continue_on_error {
                    tracing::warn!(
                        "Local inference bootstrap failed but continue_on_error=true: {}",
                        err
                    );
                } else {
                    return Err(err);
                }
            }
        }

        Ok(config)
    }

    fn resolve_model_name(model: Option<String>) -> String {
        model
            .filter(|model| model != "local-model")
            .or_else(|| env_nonempty("CALICHE_LOCAL_INFERENCE_DOWNLOAD_REPO"))
            .or_else(|| env_nonempty("CALICHE_LOCAL_INFERENCE_MODEL"))
            .or_else(Self::preferred_downloaded_model)
            .unwrap_or_else(|| Self::default_model_name().to_string())
    }

    /// Build bootstrap settings for a source/backend pair.
    ///
    /// Prefer [`Self::ensure_catalog_model`] + [`Self::fit_catalog_model`] when
    /// choosing what to download; bootstrap remains the env/CLI materialization path.
    pub(crate) fn bootstrap_for_source(
        source: &ModelSource,
        backend: InferenceBackendType,
    ) -> LocalInferenceBootstrapConfig {
        let mut bootstrap = LocalInferenceBootstrapConfig::default();
        if env_flag("CALICHE_LOCAL_INFERENCE_CONTINUE_ON_ERROR") {
            bootstrap.continue_on_error = true;
        }
        if env_flag("CALICHE_LOCAL_INFERENCE_DISABLE_CURL_FALLBACK") {
            bootstrap.allow_curl_fallback = false;
        }
        if let Some(max) = env_nonempty("CALICHE_LOCAL_INFERENCE_MAX_DOWNLOAD_BYTES") {
            if max == "0" || max.eq_ignore_ascii_case("unlimited") {
                bootstrap.max_download_bytes = None;
            } else {
                match max.parse::<u64>() {
                    Ok(parsed) => bootstrap.max_download_bytes = Some(parsed),
                    Err(error) => tracing::warn!(
                        "ignoring invalid CALICHE_LOCAL_INFERENCE_MAX_DOWNLOAD_BYTES '{max}': {error}"
                    ),
                }
            }
        }
        if let ModelSource::HuggingFace { repo, revision } = source {
            let model_root = env_nonempty("CALICHE_LOCAL_INFERENCE_MODELS_DIR")
                .map_or_else(|| PathBuf::from("./models"), PathBuf::from);
            match backend {
                InferenceBackendType::Ort => {
                    if let Some(ort) = bootstrap.ort.as_mut() {
                        ort.repo.clone_from(repo);
                        ort.revision.clone_from(revision);
                        ort.local_dir = model_root.join(format!("{}-ort", repo.replace('/', "-")));
                    }
                }
                InferenceBackendType::Candle | InferenceBackendType::Mlx => {
                    if let Some(candle) = bootstrap.candle.as_mut() {
                        candle.repo = Some(repo.clone());
                        candle.revision.clone_from(revision);
                        candle.local_dir =
                            model_root.join(format!("{}-candle", repo.replace('/', "-")));
                    }
                }
                InferenceBackendType::LlamaCpp => {
                    if let Some(gguf) = bootstrap.gguf.as_mut() {
                        gguf.repo = Some(repo.clone());
                        gguf.revision.clone_from(revision);
                        gguf.local_dir =
                            model_root.join(format!("{}-gguf", repo.replace('/', "-")));
                        if gguf.filename.is_none() {
                            gguf.filename = Some(Self::resolve_gguf_filename_for_repo(repo));
                        }
                    }
                }
            }
        }
        bootstrap
    }

    pub(crate) fn resolve_backend_for_model(
        model_name: &str,
    ) -> anyhow::Result<InferenceBackendType> {
        if let Ok(backend) = env_var_nonempty("CALICHE_LOCAL_INFERENCE_BACKEND") {
            return Self::parse_backend(&backend).ok_or_else(|| {
                anyhow::anyhow!(
                    "invalid local inference backend '{backend}'; expected candle, ort, onnx, mlx, gguf, or llamacpp"
                )
            });
        }

        if Self::looks_like_onnx_model(model_name) {
            return Ok(InferenceBackendType::Ort);
        }

        if Self::looks_like_gguf_model(model_name) {
            return Ok(InferenceBackendType::LlamaCpp);
        }

        if Self::looks_like_mlx_model(model_name) {
            return Ok(InferenceBackendType::Mlx);
        }

        Ok(InferenceBackendType::Candle)
    }

    fn looks_like_onnx_model(model_name: &str) -> bool {
        let model_name = model_name.trim().to_ascii_lowercase();
        let repo_name = model_name.rsplit('/').next().unwrap_or(model_name.as_str());

        std::path::Path::new(&model_name)
            .extension()
            .is_some_and(|ext| ext.eq_ignore_ascii_case("onnx"))
            || model_name.starts_with("onnx-community/")
            || repo_name == "onnx"
            || repo_name.ends_with("-onnx")
            || repo_name.ends_with("_onnx")
    }

    fn looks_like_mlx_model(model_name: &str) -> bool {
        let model_name = model_name.trim().to_ascii_lowercase();
        let repo_name = model_name.rsplit('/').next().unwrap_or(model_name.as_str());

        model_name.starts_with("mlx-community/")
            || repo_name.starts_with("mlx-community-")
            || repo_name == "mlx"
            || repo_name.ends_with("-mlx")
            || repo_name.ends_with("_mlx")
    }

    fn looks_like_gguf_model(model_name: &str) -> bool {
        let model_name = model_name.trim().to_ascii_lowercase();
        let repo_name = model_name.rsplit('/').next().unwrap_or(model_name.as_str());

        std::path::Path::new(&model_name)
            .extension()
            .is_some_and(|ext| ext.eq_ignore_ascii_case("gguf"))
            || model_name.contains("-gguf")
            || model_name.contains("_gguf")
            || model_name.ends_with("/gguf")
            || repo_name.ends_with("-gguf")
            || repo_name.ends_with("_gguf")
            || repo_name == "gguf"
    }

    #[must_use]
    pub fn parse_backend(backend: &str) -> Option<InferenceBackendType> {
        match backend.trim().to_ascii_lowercase().as_str() {
            "candle" => Some(InferenceBackendType::Candle),
            "ort" | "onnx" | "onnx-runtime" | "onnxruntime" => Some(InferenceBackendType::Ort),
            "mlx" => Some(InferenceBackendType::Mlx),
            "gguf" | "llamacpp" | "llama.cpp" | "llama-cpp" => Some(InferenceBackendType::LlamaCpp),
            _ => None,
        }
    }

    /// Parse a device preference string (`auto`, `cpu`, `metal`, `cuda`, `mlx`).
    #[must_use]
    pub fn parse_device(device: &str) -> Option<DeviceConfig> {
        match device.trim().to_ascii_lowercase().as_str() {
            "auto" => Some(DeviceConfig::Auto),
            "cpu" => Some(DeviceConfig::Cpu),
            "metal" => Some(DeviceConfig::Metal),
            "cuda" => Some(DeviceConfig::Cuda { device_id: 0 }),
            "mlx" => Some(DeviceConfig::Mlx),
            _ => None,
        }
    }

    /// Resolve a GGUF artifact path for a HuggingFace repo (catalog-aware).
    #[must_use]
    pub fn resolve_gguf_filename_for_repo(repo: &str) -> String {
        if let Some(spec) = builtin_catalog()
            .iter()
            .find(|spec| spec.repo == repo && matches!(spec.kind, ModelKind::Gguf))
            && let Some(artifact) = spec.artifacts.first()
        {
            return artifact.remote_path.to_string();
        }
        let leaf = repo.rsplit('/').next().unwrap_or(repo);
        if leaf.to_ascii_lowercase().ends_with(".gguf") {
            leaf.to_string()
        } else {
            format!("{leaf}.gguf")
        }
    }

    /// The backend names this build can run, for offers such as
    /// `/local-model download --backend` completion.
    #[must_use]
    pub fn backend_names() -> &'static [&'static str] {
        static BUILT: std::sync::LazyLock<Vec<&str>> = std::sync::LazyLock::new(|| {
            KNOWN_BACKEND_NAMES
                .iter()
                .copied()
                .filter(|name| {
                    LocalInferenceProvider::parse_backend(name)
                        .is_some_and(LocalInferenceProvider::backend_in_build)
                })
                .collect()
        });
        &BUILT
    }

    /// True when this build can run `backend`.
    #[must_use]
    pub const fn backend_in_build(backend: InferenceBackendType) -> bool {
        match backend {
            InferenceBackendType::Candle => cfg!(feature = "local-inference-candle"),
            InferenceBackendType::Ort => cfg!(feature = "local-inference-ort"),
            InferenceBackendType::Mlx => cfg!(feature = "inference-mlx"),
            InferenceBackendType::LlamaCpp => cfg!(feature = "local-inference-llamacpp"),
        }
    }

    /// The feature that builds `backend` in.
    #[must_use]
    pub const fn backend_build_feature(backend: InferenceBackendType) -> &'static str {
        match backend {
            InferenceBackendType::Candle | InferenceBackendType::Ort => "local-inference",
            InferenceBackendType::Mlx => "inference-mlx",
            InferenceBackendType::LlamaCpp => "local-inference-llamacpp",
        }
    }

    /// Parse a backend the user selects: like [`Self::parse_backend`], but
    /// a backend this build left out is an error that names the feature.
    ///
    /// # Errors
    ///
    /// Returns the message for an unknown name or a backend not in this
    /// build.
    pub fn parse_backend_in_build(name: &str) -> Result<InferenceBackendType, String> {
        let backend = Self::parse_backend(name).ok_or_else(|| {
            format!(
                "invalid local inference backend '{name}'. Expected one of: {}",
                Self::backend_names().join(", ")
            )
        })?;
        if Self::backend_in_build(backend) {
            Ok(backend)
        } else {
            Err(crate::llm::models::not_in_build_message(
                name.trim(),
                Self::backend_build_feature(backend),
            ))
        }
    }

    fn model_discovery_roots() -> Vec<PathBuf> {
        let mut roots = BTreeSet::new();
        for name in [
            "CALICHE_LOCAL_INFERENCE_MODEL_DIR",
            "CALICHE_LOCAL_INFERENCE_MODELS_DIR",
        ] {
            if let Some(path) = env_nonempty(name).map(PathBuf::from) {
                roots.insert(path);
            }
        }
        roots.insert(PathBuf::from("./models"));
        roots.into_iter().collect()
    }

    fn collect_local_model_candidates(root: &Path, models: &mut BTreeSet<String>) {
        Self::insert_local_model_candidate(root, models);

        let Ok(entries) = fs::read_dir(root) else {
            return;
        };

        for entry in entries.flatten() {
            Self::insert_local_model_candidate(&entry.path(), models);
        }
    }

    fn insert_local_model_candidate(path: &Path, models: &mut BTreeSet<String>) {
        if path.is_file() {
            if Self::is_supported_local_model_file(path) {
                models.insert(path.display().to_string());
            }
            return;
        }

        if !path.is_dir() {
            return;
        }

        let canonical_onnx = path.join("model.onnx");
        if canonical_onnx.is_file() {
            models.insert(canonical_onnx.display().to_string());
            return;
        }

        if Self::looks_like_candle_model_dir(path) {
            models.insert(path.display().to_string());
            return;
        }

        if Self::looks_like_gguf_model_dir(path) {
            models.insert(path.display().to_string());
        }
    }

    fn is_supported_local_model_file(path: &Path) -> bool {
        path.extension()
            .and_then(|extension| extension.to_str())
            .is_some_and(|extension| {
                extension.eq_ignore_ascii_case("onnx") || extension.eq_ignore_ascii_case("gguf")
            })
    }

    fn looks_like_candle_model_dir(path: &Path) -> bool {
        path.join("config.json").is_file()
            && path.join("tokenizer.json").is_file()
            && Self::has_local_candle_weights(path)
    }

    fn looks_like_gguf_model_dir(path: &Path) -> bool {
        let Ok(entries) = fs::read_dir(path) else {
            return false;
        };
        entries.flatten().any(|entry| {
            entry
                .path()
                .extension()
                .is_some_and(|ext| ext.eq_ignore_ascii_case("gguf"))
        })
    }

    fn has_local_candle_weights(model_dir: &Path) -> bool {
        model_dir.join("model.safetensors").is_file()
            || model_dir.join("pytorch_model.safetensors").is_file()
            || model_dir.join("model-00001-of-00001.safetensors").is_file()
            || model_dir.join("consolidated.safetensors").is_file()
            || model_dir.join("model.safetensors.index.json").is_file()
    }
}
#[cfg(feature = "local-inference-base")]
#[async_trait]
impl Provider for LocalInferenceProvider {
    async fn complete(&self, request: CompletionRequest) -> anyhow::Result<CompletionResponse> {
        // Convert messages to a prompt string
        let prompt = Self::convert_messages_to_prompt_static(&request.messages);

        // Build generation parameters from request and config
        let mut params = self.config.generation.clone();
        if let Some(max_tokens) = request.max_tokens {
            params.max_tokens = max_tokens as usize;
        }
        if let Some(temperature) = request.temperature {
            params.temperature = temperature;
        }

        // Generate text
        let text = self
            .backend
            .generate(&prompt, &params)
            .await
            .map_err(|e| anyhow::anyhow!("Inference generation failed: {e}"))?;

        let content = CompletionContent::Text { content: text };

        // Local inference backends don't provide token usage by default
        // Could be added in the future with token counting
        Ok(CompletionResponse {
            content,
            finish_reason: Some(beef::lean::Cow::borrowed("stop")),
            usage: None,
            reasoning_content: None,
        })
    }

    fn stream(
        &self,
        request: CompletionRequest,
    ) -> Pin<Box<dyn FuturesStream<Item = anyhow::Result<StreamEvent>> + Send + '_>> {
        let backend = Arc::clone(&self.backend);
        let config = self.config.clone();
        let messages = request.messages.clone();

        Box::pin(stream! {
            // Convert messages to prompt
            let prompt = Self::convert_messages_to_prompt_static(&messages);

            // Build generation parameters
            let mut params = config.generation.clone();
            if let Some(max_tokens) = request.max_tokens {
                params.max_tokens = max_tokens as usize;
            }
            if let Some(temperature) = request.temperature {
                params.temperature = temperature;
            }

            // Stream tokens
            let mut token_stream = backend.stream(&prompt, &params);
            use futures_util::StreamExt;
            while let Some(token_result) = token_stream.next().await {
                match token_result {
                    Ok(token) => {
                        yield Ok(StreamEvent::TextChunk { content: token });
                    }
                    Err(e) => {
                        yield Err(anyhow::anyhow!("Inference streaming failed: {e}"));
                        return;
                    }
                }
            }

            yield Ok(StreamEvent::Done);
        })
    }

    fn name(&self) -> &'static str {
        "local-inference"
    }
}

#[cfg(feature = "local-inference-base")]
#[async_trait]
impl Provider for LazyLocalInferenceProvider {
    async fn complete(&self, request: CompletionRequest) -> anyhow::Result<CompletionResponse> {
        self.provider().await?.complete(request).await
    }

    fn stream(
        &self,
        request: CompletionRequest,
    ) -> Pin<Box<dyn FuturesStream<Item = anyhow::Result<StreamEvent>> + Send + '_>> {
        Box::pin(stream! {
            let provider = match self.provider().await {
                Ok(provider) => provider,
                Err(error) => {
                    yield Err(error);
                    return;
                }
            };

            let mut stream = provider.stream(request);
            use futures_util::StreamExt;
            while let Some(event) = stream.next().await {
                yield event;
            }
        })
    }

    fn name(&self) -> &'static str {
        "local-inference"
    }
}

#[cfg(feature = "local-inference-base")]
impl LocalInferenceProvider {
    fn convert_messages_to_prompt_static(messages: &Arc<Vec<crate::messages::Message>>) -> String {
        let mut prompt = String::new();

        for message in messages.iter() {
            match message.role {
                MessageRole::System => {
                    prompt.push_str("System: ");
                    prompt.push_str(&message.content.extract_text());
                    prompt.push_str("\n\n");
                }
                MessageRole::User => {
                    prompt.push_str("User: ");
                    prompt.push_str(&message.content.extract_text());
                    prompt.push_str("\n\n");
                }
                MessageRole::Assistant => {
                    prompt.push_str("Assistant: ");
                    prompt.push_str(&message.content.extract_text());
                    prompt.push_str("\n\n");
                }
                MessageRole::Tool => {}
            }
        }

        prompt.push_str("Assistant: ");
        prompt
    }
}

/// Keep the fits that load without exceeding the memory budget.
fn safe_fits(fits: Vec<ModelFit>) -> Vec<ModelFit> {
    fits.into_iter()
        .filter(|fit| {
            matches!(
                fit.status,
                FitStatus::FitsComfortably | FitStatus::FitsTight
            )
        })
        .collect()
}

#[cfg(all(test, feature = "local-inference-base"))]
mod tests {
    use super::super::config::{LocalInferenceBootstrapConfig, LocalInferenceConfig};
    use super::*;
    use dripstone::inference::{DeviceConfig, InferenceBackendType};
    use dripstone::loader::{
        DEFAULT_RESERVE_RAM_BYTES, FitOptions, FitStatus, ModelSource, ResourceBudget,
        estimate_model_fit,
    };

    fn catalog_fits_on(available_ram_bytes: u64) -> Vec<ModelFit> {
        let budget = ResourceBudget {
            reserve_ram_bytes: DEFAULT_RESERVE_RAM_BYTES,
            available_ram_bytes,
            available_vram_bytes: None,
        };
        builtin_catalog()
            .iter()
            .map(|spec| estimate_model_fit(spec, &FitOptions::default(), &budget))
            .collect()
    }

    #[test]
    fn catalog_fit_marks_stories_as_safe_on_generous_host() {
        let fits = catalog_fits_on(16 * 1024 * 1024 * 1024);
        let stories = fits
            .iter()
            .find(|fit| fit.model_id == "stories15m-gguf")
            .expect("stories entry");
        assert_eq!(stories.status, FitStatus::FitsComfortably);
        let safe = safe_fits(fits);
        assert!(safe.iter().any(|fit| fit.model_id == "stories15m-gguf"));
    }

    #[test]
    fn safe_catalog_drops_models_that_exceed_memory() {
        let fits = catalog_fits_on(DEFAULT_RESERVE_RAM_BYTES);
        let stories = fits
            .iter()
            .find(|fit| fit.model_id == "stories15m-gguf")
            .expect("stories entry");
        assert_eq!(stories.status, FitStatus::ExceedsMemory);
        assert_eq!(safe_fits(fits), Vec::<ModelFit>::new());
    }

    #[test]
    #[serial_test::serial(env)]
    fn fit_options_from_env_maps_caliche_reserve() {
        let mut env = crate::test_env::EnvGuard::acquire();
        env.set("CALICHE_LOCAL_INFERENCE_RESERVE_RAM_BYTES", "4242");
        let options = LocalInferenceProvider::fit_options_from_env();
        assert_eq!(options.reserve_ram_bytes, Some(4242));
        env.remove("CALICHE_LOCAL_INFERENCE_RESERVE_RAM_BYTES");
        let options = LocalInferenceProvider::fit_options_from_env();
        assert_eq!(options.reserve_ram_bytes, None);
    }

    #[test]
    fn parse_backend_accepts_ort_aliases() {
        assert_eq!(
            LocalInferenceProvider::parse_backend("ort"),
            Some(InferenceBackendType::Ort)
        );
        assert_eq!(
            LocalInferenceProvider::parse_backend("onnx-runtime"),
            Some(InferenceBackendType::Ort)
        );
        assert_eq!(
            LocalInferenceProvider::parse_backend("candle"),
            Some(InferenceBackendType::Candle)
        );
        assert_eq!(
            LocalInferenceProvider::parse_backend("gguf"),
            Some(InferenceBackendType::LlamaCpp)
        );
        assert_eq!(
            LocalInferenceProvider::parse_backend("llama.cpp"),
            Some(InferenceBackendType::LlamaCpp)
        );
        assert_eq!(LocalInferenceProvider::parse_backend("unknown"), None);
    }

    /// A llama.cpp-only build offers its backend names and names the
    /// feature for the rest.
    #[test]
    #[cfg(all(
        feature = "local-inference-llamacpp",
        not(feature = "local-inference-candle"),
        not(feature = "local-inference-ort"),
        not(feature = "inference-mlx")
    ))]
    fn a_llamacpp_build_offers_only_its_backends() {
        assert_eq!(
            LocalInferenceProvider::backend_names(),
            ["gguf", "llamacpp"]
        );
        assert_eq!(
            LocalInferenceProvider::parse_backend_in_build("candle"),
            Err(
                "candle is not in this build. Rebuild with --features local-inference.".to_string()
            )
        );
        assert_eq!(
            LocalInferenceProvider::parse_backend_in_build("gguf"),
            Ok(InferenceBackendType::LlamaCpp)
        );
    }

    #[test]
    fn parse_backend_trims_and_normalizes_case() {
        for alias in [
            " CANDLE ",
            "Ort",
            "ONNX",
            "OnnxRuntime",
            " MLX ",
            "GGUF",
            "LlamaCpp",
            "llama-cpp",
            "LLAMACPP",
        ] {
            assert!(
                LocalInferenceProvider::parse_backend(alias).is_some(),
                "expected backend parse for {alias:?}"
            );
        }
        assert_eq!(LocalInferenceProvider::parse_backend(""), None);
        assert_eq!(LocalInferenceProvider::parse_backend("   "), None);
        assert_eq!(LocalInferenceProvider::parse_backend("torch"), None);
    }

    #[test]
    fn parse_device_accepts_common_aliases() {
        assert_eq!(
            LocalInferenceProvider::parse_device("auto"),
            Some(DeviceConfig::Auto)
        );
        assert_eq!(
            LocalInferenceProvider::parse_device("cpu"),
            Some(DeviceConfig::Cpu)
        );
        assert_eq!(
            LocalInferenceProvider::parse_device("metal"),
            Some(DeviceConfig::Metal)
        );
        assert_eq!(
            LocalInferenceProvider::parse_device("cuda"),
            Some(DeviceConfig::Cuda { device_id: 0 })
        );
        assert_eq!(LocalInferenceProvider::parse_device("unknown"), None);
    }

    #[test]
    fn parse_device_trims_and_accepts_mlx() {
        assert_eq!(
            LocalInferenceProvider::parse_device(" MLX "),
            Some(DeviceConfig::Mlx)
        );
        assert_eq!(
            LocalInferenceProvider::parse_device("CUDA"),
            Some(DeviceConfig::Cuda { device_id: 0 })
        );
        assert_eq!(LocalInferenceProvider::parse_device(""), None);
        assert_eq!(LocalInferenceProvider::parse_device("gpu"), None);
    }

    #[test]
    fn local_inference_config_defaults_to_auto_device() {
        let config = LocalInferenceConfig::new(
            ModelSource::local("models/model.gguf"),
            InferenceBackendType::LlamaCpp,
        );
        assert_eq!(config.device, DeviceConfig::Auto);
    }

    #[test]
    fn gguf_bootstrap_uses_catalog_nested_filename() {
        assert_eq!(
            LocalInferenceProvider::resolve_gguf_filename_for_repo("ggml-org/models-moved"),
            "tinyllamas/stories15M-q4_0.gguf"
        );
        let source = ModelSource::huggingface("ggml-org/models-moved", None);
        let bootstrap =
            LocalInferenceProvider::bootstrap_for_source(&source, InferenceBackendType::LlamaCpp);
        let gguf = bootstrap.gguf.expect("gguf bootstrap");
        assert_eq!(
            gguf.filename.as_deref(),
            Some("tinyllamas/stories15M-q4_0.gguf")
        );
    }

    #[test]
    fn backend_resolution_infers_gguf_sources() {
        assert_eq!(
            LocalInferenceProvider::resolve_backend_for_model("models/model.Q4_K_M.gguf").unwrap(),
            InferenceBackendType::LlamaCpp
        );
        assert_eq!(
            LocalInferenceProvider::resolve_backend_for_model(
                "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF"
            )
            .unwrap(),
            InferenceBackendType::LlamaCpp
        );
    }

    #[test]
    fn model_name_resolution_preserves_explicit_model() {
        assert_eq!(
            LocalInferenceProvider::resolve_model_name(Some("models/model.onnx".to_string())),
            "models/model.onnx"
        );
    }

    #[test]
    fn backend_resolution_infers_ort_for_onnx_sources() {
        assert_eq!(
            LocalInferenceProvider::resolve_backend_for_model("models/model.onnx").unwrap(),
            InferenceBackendType::Ort
        );
        assert_eq!(
            LocalInferenceProvider::resolve_backend_for_model("onnx-community/tiny2GPT-ONNX")
                .unwrap(),
            InferenceBackendType::Ort
        );
    }

    #[test]
    fn backend_resolution_infers_mlx_for_mlx_sources() {
        assert_eq!(
            LocalInferenceProvider::resolve_backend_for_model("mlx-community/gpt-oss-20b-MXFP4")
                .unwrap(),
            InferenceBackendType::Mlx
        );
        assert_eq!(
            LocalInferenceProvider::resolve_backend_for_model(
                "./models/mlx-community-gpt-oss-20b-MXFP4-Q8-candle"
            )
            .unwrap(),
            InferenceBackendType::Mlx
        );
    }

    #[test]
    fn bootstrap_for_source_uses_selected_ort_repo() {
        let source = ModelSource::huggingface("onnx-community/tiny2GPT-ONNX", Some("main".into()));
        let bootstrap =
            LocalInferenceProvider::bootstrap_for_source(&source, InferenceBackendType::Ort);
        let ort = bootstrap.ort.expect("ort bootstrap config");

        assert_eq!(ort.repo, "onnx-community/tiny2GPT-ONNX");
        assert_eq!(ort.revision.as_deref(), Some("main"));
        assert!(ort.local_dir.ends_with("onnx-community-tiny2GPT-ONNX-ort"));
    }

    #[test]
    fn bootstrap_accepts_programmatic_huggingface_token_without_debug_leak() {
        let bootstrap =
            LocalInferenceBootstrapConfig::default().with_huggingface_token("  hf_programmatic  ");

        assert_eq!(
            bootstrap.huggingface_token.as_deref(),
            Some("hf_programmatic")
        );
        assert!(format!("{bootstrap:?}").contains("<redacted>"));
        assert!(!format!("{bootstrap:?}").contains("hf_programmatic"));
    }

    #[test]
    fn local_inference_config_can_set_programmatic_huggingface_token() {
        let config = LocalInferenceConfig::default().with_huggingface_token("hf_config");

        assert_eq!(
            config
                .bootstrap
                .as_ref()
                .and_then(|bootstrap| bootstrap.huggingface_token.as_deref()),
            Some("hf_config")
        );
    }
}
