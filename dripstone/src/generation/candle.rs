//! Candle-based embedding generation.

use crate::embedding::EmbeddingVector;
use crate::errors::DripstoneError;
use crate::generation::{EmbeddingConfig, PoolingStrategy, VectorGenerator};
use crate::loader::{HuggingFaceLoader, load_tokenizer_from_path};
use parking_lot::Mutex;
use std::sync::Arc;

#[cfg(feature = "local-candle")]
use candle_core::{Device, Tensor};
#[cfg(feature = "local-candle")]
use candle_nn::VarBuilder;
#[cfg(feature = "local-candle")]
use candle_transformers::models::bert::{BertModel, Config as BertConfig};
#[cfg(feature = "local-candle")]
use tokenizers::Tokenizer;

/// Candle-based embedder for generating embeddings locally.
#[cfg(feature = "local-candle")]
pub struct CandleEmbedder {
    config: EmbeddingConfig,
    model: Arc<Mutex<BertModel>>,
    tokenizer: Arc<Tokenizer>,
    device: Device,
    dimension: usize,
}

#[cfg(feature = "local-candle")]
impl std::fmt::Debug for CandleEmbedder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CandleEmbedder")
            .field("dimension", &self.dimension)
            .field("config", &self.config)
            .finish_non_exhaustive()
    }
}

#[cfg(feature = "local-candle")]
impl CandleEmbedder {
    /// Pick the best available Candle device at runtime.
    ///
    /// Preference order: Metal on macOS (when built with `--features metal`),
    /// CUDA on Linux/Windows (when built with `--features cuda`), then CPU.
    /// Both accelerator probes are gated on the corresponding Candle feature;
    /// on a stock build this collapses to `Device::Cpu`, matching the old
    /// behaviour while letting users opt in without code changes.
    fn preferred_device() -> Device {
        #[cfg(feature = "candle-metal")]
        if let Ok(d) = Device::new_metal(0) {
            tracing::debug!("candle: using Metal device 0");
            return d;
        }
        #[cfg(feature = "candle-cuda")]
        if let Ok(d) = Device::new_cuda(0) {
            tracing::debug!("candle: using CUDA device 0");
            return d;
        }
        tracing::debug!("candle: using CPU device");
        Device::Cpu
    }

    /// Create a new Candle embedder with the given configuration.
    ///
    /// # Errors
    ///
    /// Returns an error when the operation cannot be completed.
    pub fn new(config: EmbeddingConfig) -> Result<Self, DripstoneError> {
        let device = Self::preferred_device();

        // Use shared HuggingFace loader
        let loader = HuggingFaceLoader::new(&config.model_repo, config.revision.as_deref())?;

        // Load tokenizer using shared utility
        let tokenizer_path = loader.get_file("tokenizer.json")?;
        let tokenizer = load_tokenizer_from_path(&tokenizer_path)?;

        // Load model config
        let config_path = loader.get_file("config.json")?;
        let config_str = std::fs::read_to_string(&config_path)
            .map_err(|e| DripstoneError::storage_with_source("failed to read model config", e))?;
        let bert_config: BertConfig = serde_json::from_str(&config_str)
            .map_err(|e| DripstoneError::storage_with_source("failed to parse model config", e))?;

        // Load model weights
        let weights_path = loader.get_file("model.safetensors")?;
        let weights = candle_core::safetensors::load(&weights_path, &device)
            .map_err(|e| DripstoneError::storage_with_source("failed to load weights", e))?;

        let vb = VarBuilder::from_tensors(weights, candle_core::DType::F32, &device);
        let model = BertModel::load(vb, &bert_config)
            .map_err(|e| DripstoneError::storage_with_source("failed to load BERT model", e))?;

        let dimension = bert_config.hidden_size;

        Ok(Self {
            config,
            model: Arc::new(Mutex::new(model)),
            tokenizer: Arc::new(tokenizer),
            device,
            dimension,
        })
    }

    fn apply_pooling(
        embeddings: &Tensor,
        attention_mask: &Tensor,
        pooling: PoolingStrategy,
    ) -> Result<Tensor, DripstoneError> {
        match pooling {
            PoolingStrategy::Mean => {
                // Mean pooling: sum embeddings weighted by attention mask, then divide by sum of mask
                let mask_sum = attention_mask
                    .sum_keepdim(1)
                    .map_err(|e| DripstoneError::storage_with_source("failed to sum mask", e))?;
                // Expand mask from [batch, seq] to [batch, seq, 1] for broadcasting
                let mask_expanded = attention_mask
                    .unsqueeze(2)
                    .map_err(|e| DripstoneError::storage_with_source("failed to expand mask", e))?;
                // Apply mask: embeddings * mask
                let masked_embeddings = embeddings
                    .broadcast_mul(&mask_expanded)
                    .map_err(|e| DripstoneError::storage_with_source("failed to apply mask", e))?;
                // Sum over sequence dimension: [batch, seq, hidden] -> [batch, hidden]
                let summed = masked_embeddings.sum_keepdim(1).map_err(|e| {
                    DripstoneError::storage_with_source("failed to sum embeddings", e)
                })?;
                // Divide by mask sum: [batch, hidden] / [batch, 1]
                summed
                    .broadcast_div(&mask_sum)
                    .map_err(|e| DripstoneError::storage_with_source("failed to divide", e))
            }
            PoolingStrategy::Cls => {
                // CLS token pooling: take the first token embedding
                embeddings.narrow(1, 0, 1).map_err(|e| {
                    DripstoneError::storage_with_source("failed to extract CLS token", e)
                })
            }
            PoolingStrategy::Max => {
                // Max pooling: element-wise maximum
                let mask_expanded = attention_mask
                    .unsqueeze(2)
                    .map_err(|e| DripstoneError::storage_with_source("failed to expand mask", e))?;
                let masked = embeddings
                    .broadcast_mul(&mask_expanded)
                    .map_err(|e| DripstoneError::storage_with_source("failed to apply mask", e))?;
                masked
                    .max_keepdim(1)
                    .map_err(|e| DripstoneError::storage_with_source("failed to compute max", e))
            }
        }
    }

    fn normalize(tensor: &Tensor, should_normalize: bool) -> Result<Tensor, DripstoneError> {
        if !should_normalize {
            return Ok(tensor.clone());
        }

        // Compute L2 norm: sqrt(sum(x^2)) over the last dimension
        let squared = tensor
            .sqr()
            .map_err(|e| DripstoneError::storage_with_source("failed to square", e))?;
        let norm = squared
            .sum_keepdim(2)
            .map_err(|e| DripstoneError::storage_with_source("failed to sum", e))?
            .sqrt()
            .map_err(|e| DripstoneError::storage_with_source("failed to sqrt", e))?;
        // Normalize: x / norm
        tensor
            .broadcast_div(&norm)
            .map_err(|e| DripstoneError::storage_with_source("failed to normalize", e))
    }
}

#[cfg(feature = "local-candle")]
#[async_trait::async_trait]
impl VectorGenerator for CandleEmbedder {
    async fn generate(&self, texts: &[String]) -> Result<Vec<EmbeddingVector>, DripstoneError> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        // Run synchronous Candle operations in a blocking thread pool
        let texts = texts.to_vec();
        let tokenizer = self.tokenizer.clone();
        let model = self.model.clone();
        let device = self.device.clone();
        let config = self.config.clone();

        tokio::task::spawn_blocking(move || {
            // Tokenize all texts
            let encodings: Result<Vec<_>, _> = texts
                .iter()
                .map(|text| {
                    tokenizer
                        .encode(text.as_str(), true)
                        .map_err(|e| DripstoneError::storage(format!("tokenization failed: {e}")))
                })
                .collect();

            let encodings = encodings?;

            // Find max length for padding
            let max_len = encodings
                .iter()
                .map(|e: &tokenizers::Encoding| e.get_ids().len())
                .max()
                .unwrap_or(0)
                .min(config.max_length);

            // Prepare batch tensors
            let batch_size = texts.len();
            let mut input_ids = Vec::with_capacity(batch_size * max_len);
            let mut attention_mask = Vec::with_capacity(batch_size * max_len);

            for encoding in &encodings {
                let ids = encoding.get_ids();
                let len = ids.len().min(max_len);

                // Add input IDs
                input_ids.extend(ids.iter().take(len));
                // Pad
                input_ids.extend(std::iter::repeat_n(0u32, max_len - len));

                // Add attention mask
                attention_mask.extend(std::iter::repeat_n(1u32, len));
                attention_mask.extend(std::iter::repeat_n(0u32, max_len - len));
            }

            // Create tensors
            let input_ids_tensor = Tensor::from_vec(input_ids, (batch_size, max_len), &device)
                .map_err(|e| {
                    DripstoneError::storage_with_source("failed to create input_ids tensor", e)
                })?;

            let attention_mask_tensor =
                Tensor::from_vec(attention_mask, (batch_size, max_len), &device).map_err(|e| {
                    DripstoneError::storage_with_source("failed to create attention_mask tensor", e)
                })?;

            // Run inference
            let model_output = {
                let model = model.lock();
                // Candle 0.9+ requires token_type_ids as third parameter
                let token_type_ids = input_ids_tensor.zeros_like().map_err(|e| {
                    DripstoneError::storage_with_source("failed to create token_type_ids", e)
                })?;
                model
                    .forward(
                        &input_ids_tensor,
                        &token_type_ids,
                        Some(&attention_mask_tensor),
                    )
                    .map_err(|e| DripstoneError::storage_with_source("model forward failed", e))?
            };

            // Apply pooling
            let pooled =
                Self::apply_pooling(&model_output, &attention_mask_tensor, config.pooling)?;

            // Normalize
            let normalized = Self::normalize(&pooled, config.normalize)?;

            // Extract embeddings
            let embeddings_vec = normalized.to_vec2::<f32>().map_err(|e| {
                DripstoneError::storage_with_source("failed to extract embeddings", e)
            })?;

            // Convert Vec<Vec<f32>> to Vec<Arc<[f32]>>
            let embeddings: Vec<EmbeddingVector> = embeddings_vec
                .into_iter()
                .map(std::convert::Into::into)
                .collect();

            Ok(embeddings)
        })
        .await
        .map_err(|e| DripstoneError::storage_with_source("task join failed", e))?
    }

    fn dimension(&self) -> usize {
        self.dimension
    }

    fn name(&self) -> &'static str {
        "candle"
    }
}

#[cfg(not(feature = "local-candle"))]
pub struct CandleEmbedder;

#[cfg(not(feature = "local-candle"))]
impl CandleEmbedder {
    pub fn new(_config: EmbeddingConfig) -> Result<Self, DripstoneError> {
        Err(DripstoneError::Misconfiguration(
            "Candle embedder requires the 'local-candle' feature".to_string(),
        ))
    }
}
