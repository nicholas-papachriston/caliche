//! MLX-based inference backend for language models (Apple Silicon).
//!
//! Feature-gated on `inference-mlx`. When device is `Mlx` or `Auto` on Apple,
//! this backend is used. Generation tries native mlx-rs first (Mistral-format
//! with params.json + weights.safetensors); on missing files or error, falls
//! back to the `mlx_lm` Python CLI when available (`pip install mlx mlx-lm`).

use crate::errors::DripstoneError;
use crate::inference::{GenerationParams, InferenceBackend, KvCacheSupport, ModelInfo};
use crate::loader::{HuggingFaceLoader, ModelSource, load_tokenizer_from_path};
use async_stream::stream;
use async_trait::async_trait;
use futures_util::Stream;
use std::path::Path;
use std::pin::Pin;
use std::process::Command;
use std::sync::Arc;

#[cfg(feature = "inference-mlx")]
use crate::inference::mlx_gen;
#[cfg(feature = "inference-mlx")]
use tokenizers::Tokenizer;

/// MLX-based inference backend (Apple Silicon).
///
/// Uses the `mlx_lm` Python package for generation when available; model path
/// and tokenizer are resolved at construction.
#[cfg(feature = "inference-mlx")]
pub struct MlxInference {
    model_info: ModelInfo,
    #[allow(
        dead_code,
        reason = "kept for native mlx-rs generation or token counting; this build host cannot compile the MLX features, so an expect here could not be checked"
    )]
    tokenizer: Arc<Tokenizer>,
    model_path: std::path::PathBuf,
}

#[cfg(feature = "inference-mlx")]
impl MlxInference {
    /// Create a new MLX inference backend from configuration.
    ///
    /// Requires `DeviceConfig::Mlx` or `DeviceConfig::Auto` (on Apple with MLX).
    /// Resolves model path and loads tokenizer; LLM forward is stubbed.
    pub async fn new(config: crate::inference::InferenceConfig) -> Result<Self, DripstoneError> {
        if matches!(config.device, crate::inference::DeviceConfig::Mlx)
            || matches!(config.device, crate::inference::DeviceConfig::Auto)
        {
            let _ = config.device.resolve();
        } else {
            return Err(DripstoneError::Misconfiguration(
                "MlxInference requires device Mlx or Auto".to_string(),
            ));
        }

        let (model_dir, model_name) = match &config.source {
            ModelSource::Local { path } => {
                config.source.validate_local()?;
                let dir = if path.is_file() {
                    path.parent()
                        .map(std::path::Path::to_path_buf)
                        .unwrap_or_else(|| std::path::Path::new(".").to_path_buf())
                } else {
                    path.clone()
                };
                let name = path
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .unwrap_or("local-model")
                    .to_string();
                (dir, name)
            }
            ModelSource::HuggingFace { repo, revision } => {
                let loader = HuggingFaceLoader::new(repo, revision.as_deref())?;
                let config_path = loader.get_file("config.json")?;
                let dir = config_path
                    .parent()
                    .ok_or_else(|| DripstoneError::storage("invalid config path"))?
                    .to_path_buf();
                let name = repo.split('/').next_back().unwrap_or(repo).to_string();
                (dir, name)
            }
        };

        let tokenizer_path = model_dir.join("tokenizer.json");
        let tokenizer = if tokenizer_path.exists() {
            load_tokenizer_from_path(&tokenizer_path)?
        } else {
            return Err(DripstoneError::storage(format!(
                "Tokenizer not found at {}",
                tokenizer_path.display()
            )));
        };

        let model_info = ModelInfo::new(model_name);

        Ok(Self {
            model_info,
            tokenizer: Arc::new(tokenizer),
            model_path: model_dir,
        })
    }

    /// Run generation via `python3 -m mlx_lm.generate`. Requires `pip install mlx mlx-lm`.
    fn generate_via_mlx_lm_cli(
        model_path: &Path,
        prompt: &str,
        params: &GenerationParams,
    ) -> Result<String, DripstoneError> {
        let output = Command::new("python3")
            .args([
                "-m",
                "mlx_lm.generate",
                "--model",
                model_path
                    .to_str()
                    .ok_or_else(|| DripstoneError::storage("model path not UTF-8"))?,
                "--prompt",
                prompt,
                "--max-tokens",
                &params.max_tokens.to_string(),
                "--temp",
                &params.temperature.to_string(),
            ])
            .output()
            .map_err(|e| {
                DripstoneError::Misconfiguration(format!(
                    "failed to run mlx_lm.generate (is Python with mlx-lm installed?): {}",
                    e
                ))
            })?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(DripstoneError::storage(format!(
                "mlx_lm.generate failed: {}",
                stderr.trim()
            )));
        }

        let out = String::from_utf8_lossy(&output.stdout);
        // mlx_lm may print prompt then completion; we want only the generated continuation.
        // If the output starts with the prompt, strip it; otherwise return full stdout.
        let text = out.trim();
        let generated = if text.starts_with(prompt) {
            text[prompt.len()..].trim_start()
        } else {
            text
        };
        Ok(generated.to_string())
    }
}

#[cfg(feature = "inference-mlx")]
#[async_trait]
impl InferenceBackend for MlxInference {
    async fn generate(
        &self,
        prompt: &str,
        params: &GenerationParams,
    ) -> Result<String, DripstoneError> {
        let model_path = self.model_path.clone();
        let tokenizer = Arc::clone(&self.tokenizer);
        let prompt = prompt.to_string();
        let params = params.clone();
        tokio::task::spawn_blocking(move || {
            match mlx_gen::generate_mlx_rs(&model_path, tokenizer, &prompt, &params) {
                Ok(s) => Ok(s),
                Err(native_err) => Self::generate_via_mlx_lm_cli(&model_path, &prompt, &params)
                    .map_err(|_cli_err| {
                        DripstoneError::storage_with_source(
                            "MLX generation failed (native and Python fallback). Use an MLX-format model (params.json + weights.safetensors) or install: pip install mlx mlx-lm".to_string(),
                            native_err,
                        )
                    }),
            }
        })
        .await
        .map_err(|e| DripstoneError::storage_with_source("MLX generate task join failed", e))?
    }

    fn stream(
        &self,
        prompt: &str,
        params: &GenerationParams,
    ) -> Pin<Box<dyn Stream<Item = Result<String, DripstoneError>> + Send + '_>> {
        let model_path = self.model_path.clone();
        let tokenizer = Arc::clone(&self.tokenizer);
        let prompt = prompt.to_string();
        let params = params.clone();
        Box::pin(stream! {
            // Try mlx-rs first, then CLI; neither streams tokens, yield once
            let result = tokio::task::spawn_blocking({
                let model_path = model_path.clone();
                let prompt = prompt.clone();
                let params = params.clone();
                move || {
                    match mlx_gen::generate_mlx_rs(&model_path, tokenizer, &prompt, &params) {
                        Ok(s) => Ok(s),
                        Err(native_err) => {
                            MlxInference::generate_via_mlx_lm_cli(&model_path, &prompt, &params)
                                .map_err(|_cli_err| {
                                    DripstoneError::storage_with_source(
                                        "MLX generation failed (native and Python fallback). Use an MLX-format model (params.json + weights.safetensors) or install: pip install mlx mlx-lm".to_string(),
                                        native_err,
                                    )
                                })
                        }
                    }
                }
            })
            .await;
            match result {
                Ok(Ok(text)) => yield Ok(text),
                Ok(Err(e)) => yield Err(e),
                Err(e) => yield Err(DripstoneError::storage_with_source(
                    "MLX stream task join failed",
                    e,
                )),
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

#[cfg(feature = "inference-mlx")]
impl KvCacheSupport for MlxInference {
    fn supports_kv_cache(&self) -> bool {
        true
    }
}

#[cfg(not(feature = "inference-mlx"))]
pub struct MlxInference;

#[cfg(not(feature = "inference-mlx"))]
impl MlxInference {
    pub async fn new(_config: crate::inference::InferenceConfig) -> Result<Self, DripstoneError> {
        Err(DripstoneError::Misconfiguration(
            "MLX inference requires the 'inference-mlx' feature".to_string(),
        ))
    }
}
