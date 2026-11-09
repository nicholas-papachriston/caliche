//! Candle-based inference backend for language models.

#[cfg(feature = "inference-candle")]
use crate::inference::candle_gen;

use crate::errors::DripstoneError;
use crate::inference::{GenerationParams, InferenceBackend, KvCacheSupport, ModelInfo};
use crate::loader::{HuggingFaceLoader, ModelSource, load_tokenizer_from_path};
use async_stream::stream;
use async_trait::async_trait;
use futures_util::Stream;
use std::pin::Pin;
use std::sync::Arc;

#[cfg(feature = "inference-candle")]
use candle_core::Device;
#[cfg(feature = "inference-candle")]
use parking_lot::Mutex;
#[cfg(feature = "inference-candle")]
use tokenizers::Tokenizer;

/// Bound streaming so producers cannot unbounded-buffer token text.
#[cfg(feature = "inference-candle")]
const STREAM_CHANNEL_CAPACITY: usize = 64;

/// Candle-based inference backend.
///
/// Supports loading models from safetensors files (via HuggingFace or local paths).
/// Currently supports decoder-only transformer models like LLaMA, GPT, etc.
#[cfg(feature = "inference-candle")]
pub struct CandleInference {
    model_info: ModelInfo,
    device: Device,
    tokenizer: Arc<Tokenizer>,
    /// Weights loaded once at construction and reused across requests.
    loaded: Arc<Mutex<candle_gen::LoadedLlamaModel>>,
}

#[cfg(feature = "inference-candle")]
impl CandleInference {
    /// Create a new Candle inference backend from configuration.
    pub async fn new(config: crate::inference::InferenceConfig) -> Result<Self, DripstoneError> {
        let device = resolve_candle_device(config.device)?;

        // Resolve model path and determine model info (HF I/O is sync — run off async runtime).
        let source = config.source.clone();
        let architecture_hint = config.architecture_hint.clone();
        let (model_dir, model_name, config_path, architecture) =
            tokio::task::spawn_blocking(move || resolve_model_dir(source, architecture_hint))
                .await
                .map_err(|e| {
                    DripstoneError::storage_with_source("candle resolve task failed", e)
                })??;

        let tokenizer_path = model_dir.join("tokenizer.json");
        let tokenizer = if tokenizer_path.exists() {
            load_tokenizer_from_path(&tokenizer_path)?
        } else {
            return Err(DripstoneError::storage(format!(
                "Tokenizer not found at expected path: {}",
                tokenizer_path.display()
            )));
        };

        let mut model_info = ModelInfo::new(&model_name);
        if let Some(arch) = architecture.clone() {
            model_info = model_info.with_architecture(arch);
        }

        let model_config = if config_path.exists() {
            candle_gen::parse_model_config(&config_path)?
        } else {
            return Err(DripstoneError::storage(format!(
                "config.json not found at {}",
                config_path.display()
            )));
        };

        // Reject unsupported MoE/quant architectures early.
        if matches!(
            model_config.architecture,
            candle_gen::ModelArchitecture::GptOss
        ) {
            return Err(DripstoneError::Misconfiguration(
                "Candle inference does not support gpt_oss models yet".to_string(),
            ));
        }

        let weight_bytes = candle_gen::find_weights_paths(&model_dir)?
            .iter()
            .map(|path| std::fs::metadata(path).map(|meta| meta.len()).unwrap_or(0))
            .sum::<u64>();
        let n_ctx = u32::try_from(model_config.max_position_embeddings).unwrap_or(2048);
        let estimate = crate::loader::estimate_runtime_memory(weight_bytes, n_ctx.min(8192), None);
        let budget = crate::loader::ResourceBudget::from_host()?;
        crate::loader::ensure_resource_budget(&estimate, &budget)?;

        let device_for_load = device.clone();
        let loaded = tokio::task::spawn_blocking(move || {
            candle_gen::load_llama_model(&model_dir, &device_for_load, &model_config)
        })
        .await
        .map_err(|e| DripstoneError::storage_with_source("candle load task failed", e))??;

        Ok(Self {
            model_info,
            device,
            tokenizer: Arc::new(tokenizer),
            loaded: Arc::new(Mutex::new(loaded)),
        })
    }
}

#[cfg(feature = "inference-candle")]
fn resolve_candle_device(device: crate::inference::DeviceConfig) -> Result<Device, DripstoneError> {
    match device {
        crate::inference::DeviceConfig::Cpu => Ok(Device::Cpu),
        crate::inference::DeviceConfig::Cuda { device_id } => {
            #[cfg(feature = "candle-cuda")]
            {
                Device::cuda_if_available(device_id).map_err(|e| {
                    DripstoneError::storage_with_source(
                        format!("Failed to initialize CUDA device {device_id}"),
                        e,
                    )
                })
            }
            #[cfg(not(feature = "candle-cuda"))]
            {
                let _ = device_id;
                Err(DripstoneError::Misconfiguration(
                    "CUDA device requested but dripstone was built without the 'candle-cuda' feature"
                        .to_string(),
                ))
            }
        }
        crate::inference::DeviceConfig::Metal => {
            #[cfg(feature = "candle-metal")]
            {
                Device::new_metal(0).map_err(|e| {
                    DripstoneError::storage_with_source("Failed to initialize Metal device", e)
                })
            }
            #[cfg(not(feature = "candle-metal"))]
            {
                Err(DripstoneError::Misconfiguration(
                    "Metal device requested but dripstone was built without the 'candle-metal' feature"
                        .to_string(),
                ))
            }
        }
        crate::inference::DeviceConfig::Mlx => Err(DripstoneError::Misconfiguration(
            "MLX device requires InferenceBackendType::Mlx; use the MLX backend instead of Candle"
                .to_string(),
        )),
        crate::inference::DeviceConfig::Auto => {
            let resolved = device.resolve();
            if let Some(reason) = &resolved.reason {
                tracing::info!(
                    selected = ?resolved.selected,
                    reason = %reason,
                    "inference-candle device selection"
                );
            }
            #[cfg(feature = "candle-cuda")]
            if let Ok(cuda) = Device::cuda_if_available(0) {
                return Ok(cuda);
            }
            #[cfg(feature = "candle-metal")]
            if let Ok(metal) = Device::new_metal(0) {
                return Ok(metal);
            }
            Ok(Device::Cpu)
        }
    }
}

#[cfg(feature = "inference-candle")]
fn resolve_model_dir(
    source: ModelSource,
    architecture_hint: Option<String>,
) -> Result<
    (
        std::path::PathBuf,
        String,
        std::path::PathBuf,
        Option<String>,
    ),
    DripstoneError,
> {
    match source {
        ModelSource::Local { path } => {
            let local = ModelSource::Local { path: path.clone() };
            local.validate_local()?;
            let name = path
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("local-model")
                .to_string();

            let dir = if path.is_file() {
                path.parent()
                    .unwrap_or_else(|| std::path::Path::new("."))
                    .to_path_buf()
            } else {
                path
            };

            let config_path = dir.join("config.json");
            let architecture = if config_path.exists() {
                candle_gen::detect_architecture(&config_path).ok()
            } else {
                architecture_hint
            };
            Ok((dir, name, config_path, architecture))
        }
        ModelSource::HuggingFace { repo, revision } => {
            let name = repo.split('/').next_back().unwrap_or(&repo).to_string();
            let loader = HuggingFaceLoader::new(&repo, revision.as_deref())?;
            let config_path = loader.get_file("config.json")?;
            let dir = config_path
                .parent()
                .ok_or_else(|| DripstoneError::storage("invalid config path"))?
                .to_path_buf();
            let architecture = candle_gen::detect_architecture(&config_path)
                .ok()
                .or(architecture_hint);
            Ok((dir, name, config_path, architecture))
        }
    }
}

#[cfg(feature = "inference-candle")]
#[async_trait]
impl InferenceBackend for CandleInference {
    async fn generate(
        &self,
        prompt: &str,
        params: &GenerationParams,
    ) -> Result<String, DripstoneError> {
        let prompt = prompt.to_string();
        let params = params.clone();
        let tokenizer = Arc::clone(&self.tokenizer);
        let device = self.device.clone();
        let loaded = Arc::clone(&self.loaded);

        tokio::task::spawn_blocking(move || {
            let guard = loaded.lock();
            candle_gen::generate_llama_loaded(&guard, tokenizer.as_ref(), &device, &prompt, &params)
        })
        .await
        .map_err(|e| DripstoneError::storage_with_source("task join failed", e))?
    }

    fn stream(
        &self,
        prompt: &str,
        params: &GenerationParams,
    ) -> Pin<Box<dyn Stream<Item = Result<String, DripstoneError>> + Send + '_>> {
        let prompt = prompt.to_string();
        let params = params.clone();
        let tokenizer = Arc::clone(&self.tokenizer);
        let device = self.device.clone();
        let loaded = Arc::clone(&self.loaded);

        Box::pin(stream! {
            let (tx, mut rx) = tokio::sync::mpsc::channel(STREAM_CHANNEL_CAPACITY);

            let mut generation_task = tokio::task::spawn_blocking(move || {
                let guard = loaded.lock();
                candle_gen::generate_llama_streaming_loaded(
                    &guard,
                    tokenizer.as_ref(),
                    &device,
                    &prompt,
                    &params,
                    tx,
                )
            });

            loop {
                tokio::select! {
                    token_result = rx.recv() => {
                        match token_result {
                            Some(Ok(token)) => yield Ok(token),
                            Some(Err(e)) => {
                                yield Err(e);
                                break;
                            }
                            None => break,
                        }
                    }
                    result = &mut generation_task => {
                        match result {
                            Ok(Ok(())) => {
                                while let Some(token) = rx.recv().await {
                                    yield token;
                                }
                                break;
                            }
                            Ok(Err(e)) => {
                                yield Err(e);
                                break;
                            }
                            Err(e) => {
                                yield Err(DripstoneError::storage_with_source("task join failed", e));
                                break;
                            }
                        }
                    }
                }
            }
        })
    }

    fn model_info(&self) -> &ModelInfo {
        &self.model_info
    }

    fn supports_kv_cache(&self) -> bool {
        true
    }
}

#[cfg(feature = "inference-candle")]
impl KvCacheSupport for CandleInference {
    fn supports_kv_cache(&self) -> bool {
        true
    }
}

#[cfg(not(feature = "inference-candle"))]
pub struct CandleInference;

#[cfg(not(feature = "inference-candle"))]
impl CandleInference {
    pub async fn new(_config: crate::inference::InferenceConfig) -> Result<Self, DripstoneError> {
        Err(DripstoneError::Misconfiguration(
            "Candle inference requires the 'inference-candle' feature".to_string(),
        ))
    }
}
