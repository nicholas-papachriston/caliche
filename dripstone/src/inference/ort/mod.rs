//! ONNX Runtime-based inference backend for language models.
//!
//! When the model exposes optional past/present key-value inputs and outputs, a KV cache
//! is used: prefill runs once on the full prompt, then decode steps run on one token at a
//! time with the cache. Otherwise behaviour is unchanged (full sequence re-fed each step).

mod generate;
mod kv;

use crate::errors::DripstoneError;
use crate::inference::{GenerationParams, InferenceBackend, KvCacheSupport, ModelInfo};
use crate::loader::huggingface::{onnx_model_candidates, resolve_local_onnx_model_path};
use crate::loader::{
    HuggingFaceLoader, ModelSource, OrtSessionConfig, build_ort_session, load_tokenizer_from_path,
};
use async_stream::stream;
use async_trait::async_trait;
use futures_util::Stream;
use kv::{OrtKvCacheConfig, OrtSimpleInputNames, detect_kv_cache_io};
use std::pin::Pin;
use std::sync::Arc;

#[cfg(feature = "inference-ort")]
use ort::session::Session;
#[cfg(feature = "inference-ort")]
use tokenizers::Tokenizer;

#[cfg(feature = "inference-ort")]
pub struct OrtInference {
    model_info: ModelInfo,
    session: Arc<parking_lot::Mutex<Session>>,
    tokenizer: Arc<Tokenizer>,
    simple_input_names: OrtSimpleInputNames,
    kv_cache_config: Option<OrtKvCacheConfig>,
}

#[cfg(feature = "inference-ort")]
impl OrtInference {
    /// Create a new ONNX Runtime inference backend from configuration.
    pub async fn new(config: crate::inference::InferenceConfig) -> Result<Self, DripstoneError> {
        // HF fetch + session build are sync and can take seconds — keep them off the async runtime.
        tokio::task::spawn_blocking(move || Self::new_blocking(config))
            .await
            .map_err(|e| DripstoneError::storage_with_source("ort load task failed", e))?
    }

    fn new_blocking(config: crate::inference::InferenceConfig) -> Result<Self, DripstoneError> {
        let model_name = match &config.source {
            ModelSource::Local { path } => path
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("local-model")
                .to_string(),
            ModelSource::HuggingFace { repo, .. } => {
                repo.split('/').next_back().unwrap_or(repo).to_string()
            }
        };
        let model_info = ModelInfo::new(model_name);

        // Resolve model path
        let model_path = match &config.source {
            ModelSource::Local { path } => {
                config.source.validate_local()?;
                let candidates =
                    onnx_model_candidates(&["generation_model.onnx", "onnx/generation_model.onnx"]);
                resolve_local_onnx_model_path(path, &candidates)?
            }
            ModelSource::HuggingFace { repo, revision } => {
                let loader = HuggingFaceLoader::new(repo, revision.as_deref())?;
                loader
                    .get_onnx_model_file(&["generation_model.onnx", "onnx/generation_model.onnx"])?
            }
        };

        // Load tokenizer
        let tokenizer_path = match &config.source {
            ModelSource::Local { path } => {
                if path.is_file() {
                    path.parent()
                        .unwrap_or_else(|| std::path::Path::new("."))
                        .join("tokenizer.json")
                } else {
                    path.join("tokenizer.json")
                }
            }
            ModelSource::HuggingFace { repo, revision } => {
                let loader = HuggingFaceLoader::new(repo, revision.as_deref())?;
                loader.get_file("tokenizer.json")?
            }
        };

        let tokenizer = load_tokenizer_from_path(&tokenizer_path)?;

        let weight_bytes = std::fs::metadata(&model_path)
            .map(|meta| meta.len())
            .unwrap_or(0);
        let n_ctx = u32::try_from(config.default_params.max_tokens.saturating_add(1024))
            .unwrap_or(2048)
            .clamp(512, 8192);
        let estimate = crate::loader::estimate_runtime_memory(weight_bytes, n_ctx, None);
        let budget = crate::loader::ResourceBudget::from_host()?;
        crate::loader::ensure_resource_budget(&estimate, &budget)?;

        // Build ORT session with device configuration
        let mut session_config = OrtSessionConfig::default();
        // Convert InferenceConfig device to OrtDevice (MLX not supported by ORT; use Mlx backend)
        let ort_device = match config.device {
            crate::inference::DeviceConfig::Cpu => crate::loader::ort_session::OrtDevice::Cpu,
            crate::inference::DeviceConfig::Cuda { device_id } => {
                crate::loader::ort_session::OrtDevice::Cuda { device_id }
            }
            crate::inference::DeviceConfig::Metal | crate::inference::DeviceConfig::Mlx => {
                crate::loader::ort_session::OrtDevice::Cpu
            }
            crate::inference::DeviceConfig::Auto => crate::loader::ort_session::OrtDevice::Auto,
        };
        session_config = session_config.with_device(ort_device);
        let session = build_ort_session(&model_path, &session_config)?;
        let kv_cache_config = detect_kv_cache_io(&session);
        let simple_input_names =
            OrtSimpleInputNames::from_session(&session, kv_cache_config.as_ref())?;

        Ok(Self {
            model_info,
            session: Arc::new(parking_lot::Mutex::new(session)),
            tokenizer: Arc::new(tokenizer),
            simple_input_names,
            kv_cache_config,
        })
    }
}

#[cfg(feature = "inference-ort")]
#[async_trait]
impl InferenceBackend for OrtInference {
    async fn generate(
        &self,
        prompt: &str,
        params: &GenerationParams,
    ) -> Result<String, DripstoneError> {
        // Clone Arc references before moving into spawn_blocking
        let prompt = prompt.to_string();
        let params = params.clone();
        let session = Arc::clone(&self.session);
        let tokenizer = Arc::clone(&self.tokenizer);
        let simple_input_names = self.simple_input_names.clone();
        let kv_cache_config = self.kv_cache_config.clone();

        tokio::task::spawn_blocking(move || {
            Self::generate_internal(
                tokenizer,
                session,
                simple_input_names,
                kv_cache_config,
                &prompt,
                &params,
            )
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
        let session = Arc::clone(&self.session);
        let tokenizer = Arc::clone(&self.tokenizer);
        let simple_input_names = self.simple_input_names.clone();
        let kv_cache_config = self.kv_cache_config.clone();

        Box::pin(stream! {
            // Bound channel so producers cannot unbounded-buffer token text.
            let (tx, mut rx) = tokio::sync::mpsc::channel(64);

            // Spawn generation task
            let mut generation_task = tokio::task::spawn_blocking({
                let tokenizer = tokenizer.clone();
                let session = session.clone();
                let simple_input_names = simple_input_names.clone();
                let kv_cache_config = kv_cache_config.clone();
                let prompt = prompt.clone();
                let params = params.clone();
                move || {
                    Self::generate_internal_streaming(
                        tokenizer,
                        session,
                        simple_input_names,
                        kv_cache_config,
                        &prompt,
                        &params,
                        tx,
                    )
                }
            });

            // Stream tokens as they arrive
            loop {
                tokio::select! {
                    token_result = rx.recv() => {
                        match token_result {
                            Some(Ok(token)) => yield Ok(token),
                            Some(Err(e)) => {
                                yield Err(e);
                                break;
                            }
                            None => break, // Generation complete
                        }
                    }
                    result = &mut generation_task => {
                        // Generation task completed
                        match result {
                            Ok(Ok(())) => break,
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
        self.kv_cache_config.is_some()
    }
}

#[cfg(feature = "inference-ort")]
impl KvCacheSupport for OrtInference {
    fn supports_kv_cache(&self) -> bool {
        self.kv_cache_config.is_some()
    }
}

#[cfg(not(feature = "inference-ort"))]
pub struct OrtInference;

#[cfg(not(feature = "inference-ort"))]
impl OrtInference {
    pub async fn new(_config: crate::inference::InferenceConfig) -> Result<Self, DripstoneError> {
        Err(DripstoneError::Misconfiguration(
            "ONNX Runtime inference requires the 'inference-ort' feature".to_string(),
        ))
    }
}
