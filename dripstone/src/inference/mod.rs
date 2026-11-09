//! Local model inference for text generation.
//!
//! This module provides backends for running language models locally with support for
//! multiple inference engines (Candle, ONNX Runtime, MLX, llama.cpp).

use crate::errors::DripstoneError;
use async_trait::async_trait;
use futures_util::Stream;
use std::pin::Pin;

pub mod config;
#[cfg(any(
    feature = "inference-candle",
    feature = "inference-ort",
    feature = "inference-mlx"
))]
mod kv_cache;
#[cfg(any(
    feature = "inference-candle",
    feature = "inference-ort",
    feature = "inference-mlx"
))]
mod sampling;

#[cfg(feature = "inference-candle")]
pub(crate) mod candle_gen;

#[cfg(feature = "inference-mlx")]
pub(crate) mod mlx_gen;

pub use config::{
    DeviceConfig, GenerationParams, InferenceBackendType, InferenceConfig, ModelInfo,
};
#[cfg(any(
    feature = "inference-candle",
    feature = "inference-ort",
    feature = "inference-mlx"
))]
pub use kv_cache::KvCacheSupport;
#[cfg(any(
    feature = "inference-candle",
    feature = "inference-ort",
    feature = "inference-mlx"
))]
pub use sampling::{sample_greedy, sample_token};

/// Trait for inference backends that can generate text from prompts.
#[async_trait]
pub trait InferenceBackend: Send + Sync {
    /// Generate text from a prompt synchronously.
    async fn generate(
        &self,
        prompt: &str,
        params: &GenerationParams,
    ) -> Result<String, DripstoneError>;

    /// Stream text generation tokens as they are produced.
    fn stream(
        &self,
        prompt: &str,
        params: &GenerationParams,
    ) -> Pin<Box<dyn Stream<Item = Result<String, DripstoneError>> + Send + '_>>;

    /// Get information about the loaded model.
    fn model_info(&self) -> &ModelInfo;

    /// Returns whether this backend supports incremental decode with a KV cache.
    /// Default is false; backends that use prefill-then-decode implement this and return true.
    fn supports_kv_cache(&self) -> bool {
        false
    }
}

#[cfg(feature = "inference-candle")]
pub mod candle;

#[cfg(feature = "inference-ort")]
pub mod ort;

#[cfg(feature = "inference-mlx")]
pub mod mlx;

#[cfg(feature = "inference-llamacpp")]
pub mod llamacpp;

#[cfg(feature = "inference-candle")]
pub use candle::CandleInference;

#[cfg(feature = "inference-ort")]
pub use ort::OrtInference;

#[cfg(feature = "inference-mlx")]
pub use mlx::MlxInference;

#[cfg(feature = "inference-llamacpp")]
pub use llamacpp::LlamaCppInference;
