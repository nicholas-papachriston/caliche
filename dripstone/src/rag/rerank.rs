//! Cross-encoder reranking using ONNX Runtime.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use crate::core::SearchResult;
use crate::errors::DripstoneError;

#[cfg(feature = "embedding-ort")]
use crate::loader::{OrtSessionConfig, build_ort_session, load_tokenizer_from_path};
#[cfg(feature = "embedding-ort")]
use ndarray;
#[cfg(feature = "embedding-ort")]
use ort::{session::Session, value::TensorRef};
#[cfg(feature = "embedding-ort")]
use tokenizers::utils::padding::PaddingDirection;
#[cfg(feature = "embedding-ort")]
use tokenizers::utils::truncation::{TruncationDirection, TruncationStrategy};
#[cfg(feature = "embedding-ort")]
use tokenizers::{PaddingParams, PaddingStrategy, Tokenizer, TruncationParams};

/// Configuration for cross-encoder reranking.
#[derive(Debug, Clone)]
pub struct CrossEncoderConfig {
    /// Path to the ONNX model file
    pub model_path: String,
    /// Number of top results to rerank
    pub top_k: usize,
    /// Batch size for inference
    pub batch_size: usize,
    /// Maximum sequence length
    pub max_length: usize,
}

impl CrossEncoderConfig {
    /// Creates a new cross-encoder configuration.
    pub fn new(model_path: impl Into<String>) -> Self {
        Self {
            model_path: model_path.into(),
            top_k: 100,
            batch_size: 32,
            max_length: 512,
        }
    }

    #[must_use]
    pub const fn with_top_k(mut self, top_k: usize) -> Self {
        self.top_k = top_k;
        self
    }

    #[must_use]
    pub const fn with_batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = batch_size;
        self
    }

    #[must_use]
    pub const fn with_max_length(mut self, max_length: usize) -> Self {
        self.max_length = max_length;
        self
    }
}

/// Cross-encoder reranker using ONNX Runtime.
#[cfg(feature = "embedding-ort")]
pub struct CrossEncoderReranker {
    config: CrossEncoderConfig,
    session: Arc<parking_lot::Mutex<Session>>,
    tokenizer: Arc<Tokenizer>,
    needs_token_type_ids: bool,
    pad_id: u32,
    pad_type_id: u32,
}

#[cfg(feature = "embedding-ort")]
impl std::fmt::Debug for CrossEncoderReranker {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CrossEncoderReranker")
            .field("config", &self.config)
            .finish_non_exhaustive()
    }
}

#[cfg(feature = "embedding-ort")]
impl CrossEncoderReranker {
    /// Creates a new cross-encoder reranker.
    ///
    /// # Errors
    ///
    /// Returns an error when the operation cannot be completed.
    pub fn new(config: CrossEncoderConfig) -> Result<Self, DripstoneError> {
        let model_path = PathBuf::from(&config.model_path);

        let session_cfg = OrtSessionConfig::default();
        let session = build_ort_session(&model_path, &session_cfg)?;

        // Load tokenizer from `tokenizer.json` in the same directory as the model file.
        let tokenizer_path = model_path
            .parent()
            .unwrap_or_else(|| Path::new("."))
            .join("tokenizer.json");
        let mut tokenizer = load_tokenizer_from_path(&tokenizer_path)?;

        // Configure truncation + padding to fixed max length.
        tokenizer
            .with_truncation(Some(TruncationParams {
                direction: TruncationDirection::Right,
                max_length: config.max_length,
                strategy: TruncationStrategy::LongestFirst,
                stride: 0,
            }))
            .map_err(|e| {
                DripstoneError::storage(format!("failed to set tokenizer truncation: {e}"))
            })?;

        // Prefer padding params shipped with the tokenizer. When the
        // tokenizer doesn't declare one, probe a broader list of
        // conventions covering BERT (`[PAD]`), T5/XLM-R (`<pad>`), LLaMA 3
        // (`<|finetune_right_pad_id|>`), GPT-2/OpenAI (`<|endoftext|>`),
        // Mistral-style (`</s>`), and generic (`<|pad|>`). If none of those
        // resolve we fall back to token id 0 and emit a warning so users
        // don't silently run with a nonsense pad token on an unknown model.
        let (pad_id, pad_token, pad_resolved) = tokenizer
            .get_padding()
            .map(|p| (p.pad_id, p.pad_token.clone(), true))
            .or_else(|| {
                const CANDIDATES: &[&str] = &[
                    "[PAD]",
                    "<pad>",
                    "<|pad|>",
                    "<|finetune_right_pad_id|>",
                    "</s>",
                    "<|endoftext|>",
                ];
                for tok in CANDIDATES {
                    if let Some(id) = tokenizer.token_to_id(tok) {
                        return Some((id, (*tok).to_string(), true));
                    }
                }
                None
            })
            .unwrap_or_else(|| (0, "[PAD]".to_string(), false));

        if !pad_resolved {
            tracing::warn!(
                "rerank tokenizer has no padding config and no known pad token; falling back to token id 0. Padded sequences may yield incorrect scores."
            );
        }

        let pad_type_id = tokenizer.get_padding().map_or(0, |p| p.pad_type_id);

        tokenizer.with_padding(Some(PaddingParams {
            strategy: PaddingStrategy::Fixed(config.max_length),
            direction: PaddingDirection::Right,
            pad_to_multiple_of: None,
            pad_id,
            pad_type_id,
            pad_token,
        }));

        let needs_token_type_ids = session.inputs().len() >= 3;

        Ok(Self {
            config,
            session: Arc::new(parking_lot::Mutex::new(session)),
            tokenizer: Arc::new(tokenizer),
            needs_token_type_ids,
            pad_id,
            pad_type_id,
        })
    }

    /// Reranks results using the cross-encoder model.
    ///
    /// This method works with the new `SearchResult` type from `core.rs`.
    ///
    /// # Errors
    ///
    /// Returns an error when the operation cannot be completed.
    pub fn rerank(
        &self,
        query: &str,
        results: Vec<SearchResult>,
    ) -> Result<Vec<SearchResult>, DripstoneError> {
        if results.is_empty() {
            return Ok(results);
        }

        // Take top-k candidates for reranking
        let mut candidates = results;
        if candidates.len() > self.config.top_k {
            candidates.truncate(self.config.top_k);
        }

        // Prepare query-document pairs for scoring
        let pairs: Vec<(String, String)> = candidates
            .iter()
            .map(|result| {
                // Extract text from chunk content
                (query.to_string(), result.chunk.content.clone())
            })
            .collect();

        // Score in batches
        let scores = self.score_batch(&pairs)?;

        // Update search results with new scores
        let mut reranked: Vec<SearchResult> = candidates
            .into_iter()
            .zip(scores.iter())
            .map(|(mut result, &score)| {
                result.score = score;
                result
            })
            .collect();

        // Sort by new scores (descending)
        reranked.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Update ranks
        for (rank, result) in reranked.iter_mut().enumerate() {
            result.rank = rank;
        }

        Ok(reranked)
    }

    fn score_batch(&self, pairs: &[(String, String)]) -> Result<Vec<f32>, DripstoneError> {
        if pairs.is_empty() {
            return Ok(Vec::new());
        }

        let mut all_scores = Vec::with_capacity(pairs.len());
        let max_len = self.config.max_length;

        for chunk in pairs.chunks(self.config.batch_size.max(1)) {
            // Tokenize batch of pairs
            let inputs: Vec<(&str, &str)> = chunk
                .iter()
                .map(|(q, d)| (q.as_str(), d.as_str()))
                .collect();

            let encodings = self
                .tokenizer
                .encode_batch(inputs, true)
                .map_err(|e| DripstoneError::storage(format!("tokenization failed: {e}")))?;

            let batch_size = encodings.len();
            let mut input_ids: Vec<i64> = Vec::with_capacity(batch_size * max_len);
            let mut attention_mask: Vec<i64> = Vec::with_capacity(batch_size * max_len);
            let mut token_type_ids: Vec<i64> = Vec::with_capacity(batch_size * max_len);

            for enc in &encodings {
                let ids = enc.get_ids();
                let mask = enc.get_attention_mask();
                let types = enc.get_type_ids();

                for i in 0..max_len {
                    if i < ids.len() {
                        input_ids.push(i64::from(ids[i]));
                        attention_mask.push(i64::from(mask.get(i).copied().unwrap_or(1)));
                        token_type_ids.push(i64::from(types.get(i).copied().unwrap_or(0)));
                    } else {
                        input_ids.push(i64::from(self.pad_id));
                        attention_mask.push(0);
                        token_type_ids.push(i64::from(self.pad_type_id));
                    }
                }
            }

            let ids_array = ndarray::Array::from_shape_vec((batch_size, max_len), input_ids)
                .map_err(|e| {
                    DripstoneError::storage_with_source("failed to build input_ids array", e)
                })?;
            let mask_array = ndarray::Array::from_shape_vec((batch_size, max_len), attention_mask)
                .map_err(|e| {
                    DripstoneError::storage_with_source("failed to build attention_mask array", e)
                })?;
            let type_array = ndarray::Array::from_shape_vec((batch_size, max_len), token_type_ids)
                .map_err(|e| {
                    DripstoneError::storage_with_source("failed to build token_type_ids array", e)
                })?;

            let ids_tensor = TensorRef::<i64>::from_array_view(&ids_array).map_err(|e| {
                DripstoneError::storage_with_source("failed to create input_ids tensor", e)
            })?;
            let mask_tensor = TensorRef::<i64>::from_array_view(&mask_array).map_err(|e| {
                DripstoneError::storage_with_source("failed to create attention_mask tensor", e)
            })?;
            let type_tensor = TensorRef::<i64>::from_array_view(&type_array).map_err(|e| {
                DripstoneError::storage_with_source("failed to create token_type_ids tensor", e)
            })?;

            // Run inference (session needs &mut)
            let mut session = self.session.lock();
            let outputs = if self.needs_token_type_ids {
                session
                    .run(ort::inputs![ids_tensor, mask_tensor, type_tensor])
                    .map_err(|e| {
                        DripstoneError::storage_with_source("ONNX session run failed", e)
                    })?
            } else {
                session
                    .run(ort::inputs![ids_tensor, mask_tensor])
                    .map_err(|e| {
                        DripstoneError::storage_with_source("ONNX session run failed", e)
                    })?
            };

            let (shape, data) = outputs[0]
                .try_extract_tensor::<f32>()
                .map_err(|e| DripstoneError::storage_with_source("failed to extract logits", e))?;

            let batch_scores = Self::decode_batch_scores(shape, data, batch_size)?;

            if batch_scores.len() != batch_size {
                return Err(DripstoneError::storage(
                    "reranker output size mismatch with batch size",
                ));
            }

            all_scores.extend(batch_scores);
        }

        Ok(all_scores)
    }

    fn decode_batch_scores(
        shape: &[i64],
        data: &[f32],
        batch_size: usize,
    ) -> Result<Vec<f32>, DripstoneError> {
        if shape.len() == 1 {
            return Ok(data.to_vec());
        }
        if shape.len() != 2 {
            return Err(DripstoneError::storage(
                "unexpected reranker output rank (expected 1D or 2D tensor)",
            ));
        }
        let cols = usize::try_from(shape[1]).map_err(|_| {
            DripstoneError::storage("reranker output width is negative or out of range")
        })?;
        match cols {
            1 => Ok(data.to_vec()),
            2 => Ok((0..batch_size)
                .map(|row| *data.get(row * cols + 1).unwrap_or(&0.0))
                .collect()),
            _ => Err(DripstoneError::storage(
                "unexpected reranker output shape (expected [batch], [batch,1], or [batch,2])",
            )),
        }
    }
}

/// Stub implementation when ort feature is not enabled.
#[cfg(not(feature = "embedding-ort"))]
pub struct CrossEncoderReranker {
    config: CrossEncoderConfig,
}

#[cfg(not(feature = "embedding-ort"))]
impl std::fmt::Debug for CrossEncoderReranker {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CrossEncoderReranker")
            .field("config", &self.config)
            .finish()
    }
}

#[cfg(not(feature = "embedding-ort"))]
impl CrossEncoderReranker {
    pub fn new(config: CrossEncoderConfig) -> Result<Self, DripstoneError> {
        Err(DripstoneError::Misconfiguration(
            "Cross-encoder reranking requires the 'embedding-ort' feature".to_string(),
        ))
    }

    pub fn rerank(
        &self,
        _query: &str,
        results: Vec<SearchResult>,
    ) -> Result<Vec<SearchResult>, DripstoneError> {
        Ok(results)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cross_encoder_config() {
        let config = CrossEncoderConfig::new("model.onnx")
            .with_top_k(50)
            .with_batch_size(16)
            .with_max_length(256);

        assert_eq!(config.top_k, 50);
        assert_eq!(config.batch_size, 16);
        assert_eq!(config.max_length, 256);
    }

    #[test]
    #[cfg(not(feature = "embedding-ort"))]
    fn test_reranker_without_feature() {
        let config = CrossEncoderConfig::new("model.onnx");
        let result = CrossEncoderReranker::new(config);
        assert!(result.is_err());
    }
}
