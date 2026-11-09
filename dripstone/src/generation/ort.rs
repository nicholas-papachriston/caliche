//! ONNX Runtime-based embedding generation.

use crate::embedding::EmbeddingVector;
use crate::errors::DripstoneError;
use crate::generation::{EmbeddingConfig, PoolingStrategy, VectorGenerator};
use crate::loader::ModelSource;
use crate::loader::huggingface::{onnx_model_candidates, resolve_local_onnx_model_path};
use crate::loader::{
    HuggingFaceLoader, OrtSessionConfig, build_ort_session, load_tokenizer_from_path,
};
use crate::utils::numeric::u32_to_f32;
use std::sync::Arc;

#[cfg(feature = "local-ort")]
use ort::session::Session;
#[cfg(feature = "local-ort")]
use ort::value::TensorRef;
#[cfg(feature = "local-ort")]
use tokenizers::Tokenizer;

/// Inspect an ORT session's first output tensor shape and return the trailing
/// dimension (the embedding size) when it is statically known. Most encoder
/// exports produce `(batch, seq, hidden)` or `(batch, hidden)`; the last dim
/// is the embedding dim in both cases. Returns `None` for fully dynamic
/// shapes so callers can pick a sensible default.
#[cfg(feature = "local-ort")]
fn infer_ort_text_embedding_dim(session: &Session) -> Option<usize> {
    let outputs = session.outputs();
    let first = outputs.first()?;
    let ort::value::ValueType::Tensor { shape, .. } = first.dtype() else {
        return None;
    };
    let last = shape.iter().last().copied()?;
    usize::try_from(last)
        .ok()
        .filter(|dimension| *dimension > 0)
}

/// ONNX Runtime-based embedder for generating embeddings locally.
#[cfg(feature = "local-ort")]
pub struct OrtEmbedder {
    config: EmbeddingConfig,
    session: Arc<parking_lot::Mutex<Session>>,
    tokenizer: Arc<Tokenizer>,
    input_names: OrtTextInputNames,
    dimension: usize,
}

#[cfg(feature = "local-ort")]
impl std::fmt::Debug for OrtEmbedder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OrtEmbedder")
            .field("dimension", &self.dimension)
            .field("config", &self.config)
            .finish_non_exhaustive()
    }
}

#[cfg(feature = "local-ort")]
#[derive(Debug, Clone)]
struct OrtTextInputNames {
    input_ids: String,
    attention_mask: String,
    token_type_ids: Option<String>,
}

#[cfg(feature = "local-ort")]
struct OrtTokenBatch {
    batch_size: usize,
    max_len: usize,
    input_ids: Vec<i64>,
    attention_mask: Vec<u32>,
    token_type_ids: Vec<i64>,
}

#[cfg(feature = "local-ort")]
struct OrtInputArrays {
    input_ids: ndarray::Array2<i64>,
    attention_mask: ndarray::Array2<i64>,
}

#[cfg(feature = "local-ort")]
impl OrtTextInputNames {
    fn from_session(session: &Session) -> Result<Self, DripstoneError> {
        let inputs = session.inputs();
        let input_ids = inputs
            .iter()
            .find(|input| input.name() == "input_ids")
            .or_else(|| inputs.first())
            .map(|input| input.name().to_string())
            .ok_or_else(|| DripstoneError::storage("ONNX embedding model has no inputs"))?;
        let attention_mask = inputs
            .iter()
            .find(|input| input.name() == "attention_mask")
            .or_else(|| inputs.get(1))
            .map(|input| input.name().to_string())
            .ok_or_else(|| {
                DripstoneError::configuration(
                    "ONNX embedding model must expose an attention_mask input or at least two inputs",
                )
            })?;
        let token_type_ids = inputs
            .iter()
            .find(|input| input.name() == "token_type_ids")
            .map(|input| input.name().to_string());

        let unsupported: Vec<String> = inputs
            .iter()
            .map(ort::value::Outlet::name)
            .filter(|name| {
                *name != input_ids
                    && *name != attention_mask
                    && token_type_ids.as_deref() != Some(name)
            })
            .map(ToString::to_string)
            .collect();
        if !unsupported.is_empty() {
            return Err(DripstoneError::configuration(format!(
                "ONNX embedding model requires unsupported inputs: {}",
                unsupported.join(", ")
            )));
        }

        Ok(Self {
            input_ids,
            attention_mask,
            token_type_ids,
        })
    }
}

#[cfg(feature = "local-ort")]
impl OrtEmbedder {
    fn prepare_token_batch(
        texts: &[String],
        tokenizer: &Tokenizer,
        max_length: usize,
    ) -> Result<OrtTokenBatch, DripstoneError> {
        let encodings: Vec<_> = texts
            .iter()
            .map(|text| {
                tokenizer.encode(text.as_str(), true).map_err(|error| {
                    DripstoneError::storage(format!("tokenization failed: {error}"))
                })
            })
            .collect::<Result<_, _>>()?;
        let max_len = encodings
            .iter()
            .map(|encoding| encoding.get_ids().len())
            .max()
            .unwrap_or(0)
            .min(max_length);
        let batch_size = texts.len();
        let mut input_ids = Vec::with_capacity(batch_size * max_len);
        let mut attention_mask = Vec::with_capacity(batch_size * max_len);
        let mut token_type_ids = Vec::with_capacity(batch_size * max_len);
        for encoding in &encodings {
            let ids = encoding.get_ids();
            let len = ids.len().min(max_len);
            input_ids.extend(ids.iter().take(len).map(|&id| i64::from(id)));
            input_ids.extend(std::iter::repeat_n(0, max_len - len));
            attention_mask.extend(std::iter::repeat_n(1, len));
            attention_mask.extend(std::iter::repeat_n(0, max_len - len));
            let types = encoding.get_type_ids();
            token_type_ids
                .extend((0..len).map(|index| i64::from(types.get(index).copied().unwrap_or(0))));
            token_type_ids.extend(std::iter::repeat_n(0, max_len - len));
        }
        Ok(OrtTokenBatch {
            batch_size,
            max_len,
            input_ids,
            attention_mask,
            token_type_ids,
        })
    }

    fn build_input_arrays(
        input_ids: Vec<i64>,
        attention_mask: &[u32],
        batch_size: usize,
        max_len: usize,
    ) -> Result<OrtInputArrays, DripstoneError> {
        let input_ids =
            ndarray::Array::from_shape_vec((batch_size, max_len), input_ids).map_err(|error| {
                DripstoneError::storage_with_source("failed to create input_ids array", error)
            })?;
        let attention_values = attention_mask
            .iter()
            .map(|&value| i64::from(value))
            .collect();
        let attention_mask = ndarray::Array::from_shape_vec(
            (batch_size, max_len),
            attention_values,
        )
        .map_err(|error| {
            DripstoneError::storage_with_source("failed to create attention_mask array", error)
        })?;
        Ok(OrtInputArrays {
            input_ids,
            attention_mask,
        })
    }

    /// Create a new ORT embedder with the given configuration.
    ///
    /// # Errors
    ///
    /// Returns an error when the operation cannot be completed.
    pub fn new(config: EmbeddingConfig) -> Result<Self, DripstoneError> {
        let source = ModelSource::from_str(&config.model_repo);

        let (model_path, tokenizer_path) = match &source {
            ModelSource::Local { path } => {
                source.validate_local()?;
                let candidates = onnx_model_candidates(&[]);
                let model_path = resolve_local_onnx_model_path(path, &candidates)?;
                let tokenizer_base = if path.is_file() {
                    path.parent().unwrap_or_else(|| std::path::Path::new("."))
                } else {
                    path.as_path()
                };
                (model_path, tokenizer_base.join("tokenizer.json"))
            }
            ModelSource::HuggingFace { repo, revision } => {
                let loader = HuggingFaceLoader::new(repo, revision.as_deref())?;
                (
                    loader.get_onnx_model_file(&[])?,
                    loader.get_file("tokenizer.json")?,
                )
            }
        };

        let tokenizer = load_tokenizer_from_path(&tokenizer_path)?;

        // Build ORT session using shared utility
        let session_config = OrtSessionConfig::default();
        let session = build_ort_session(&model_path, &session_config)?;
        let input_names = OrtTextInputNames::from_session(&session)?;

        // Prefer the model's declared output shape. Most HuggingFace text
        // encoders export `last_hidden_state: (batch, seq, hidden)`; the
        // trailing dim is the embedding dim. We fall back to 384 (MiniLM-
        // family default) only when the shape is fully dynamic.
        let dimension = infer_ort_text_embedding_dim(&session).unwrap_or(384);

        Ok(Self {
            config,
            session: Arc::new(parking_lot::Mutex::new(session)),
            tokenizer: Arc::new(tokenizer),
            input_names,
            dimension,
        })
    }

    fn apply_pooling(
        embeddings: &[f32],
        attention_mask: &[u32],
        batch_size: usize,
        seq_len: usize,
        hidden_size: usize,
        pooling: PoolingStrategy,
    ) -> Vec<f32> {
        let mut output = Vec::with_capacity(batch_size * hidden_size);

        match pooling {
            PoolingStrategy::Mean => {
                for b in 0..batch_size {
                    let mut pooled = vec![0.0f32; hidden_size];
                    let mut mask_sum = 0.0f32;

                    for s in 0..seq_len {
                        let mask_val = u32_to_f32(attention_mask[b * seq_len + s]);
                        mask_sum += mask_val;

                        for (h, pooled_val) in pooled.iter_mut().enumerate().take(hidden_size) {
                            let idx = (b * seq_len + s) * hidden_size + h;
                            *pooled_val += embeddings[idx] * mask_val;
                        }
                    }

                    if mask_sum > 0.0 {
                        for pooled_val in pooled.iter_mut().take(hidden_size) {
                            *pooled_val /= mask_sum;
                        }
                    }

                    output.extend(pooled);
                }
            }
            PoolingStrategy::Cls => {
                // CLS token is typically at index 0
                for b in 0..batch_size {
                    let start_idx = b * seq_len * hidden_size;
                    for h in 0..hidden_size {
                        output.push(embeddings[start_idx + h]);
                    }
                }
            }
            PoolingStrategy::Max => {
                for b in 0..batch_size {
                    let mut pooled = vec![f32::NEG_INFINITY; hidden_size];

                    for s in 0..seq_len {
                        let mask_val = attention_mask[b * seq_len + s];
                        if mask_val > 0 {
                            for (h, pooled_val) in pooled.iter_mut().enumerate().take(hidden_size) {
                                let idx = (b * seq_len + s) * hidden_size + h;
                                *pooled_val = (*pooled_val).max(embeddings[idx]);
                            }
                        }
                    }

                    output.extend(pooled);
                }
            }
        }

        output
    }

    fn normalize(embeddings: &mut [f32], hidden_size: usize) {
        let num_embeddings = embeddings.len() / hidden_size;
        for i in 0..num_embeddings {
            let start = i * hidden_size;
            let end = start + hidden_size;
            let slice = &mut embeddings[start..end];

            let norm: f32 = slice.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 0.0 {
                for val in slice.iter_mut() {
                    *val /= norm;
                }
            }
        }
    }

    fn finalize_output(
        shape: &[i64],
        data: &[f32],
        attention_mask: &[u32],
        batch_size: usize,
        pooling: PoolingStrategy,
        normalize: bool,
    ) -> Result<Vec<EmbeddingVector>, DripstoneError> {
        if shape.len() < 3 {
            return Err(DripstoneError::storage(
                "unexpected output shape from ONNX model",
            ));
        }
        let seq_len = usize::try_from(shape[1]).map_err(|_| {
            DripstoneError::storage("ONNX sequence dimension is negative or out of range")
        })?;
        let hidden_size = usize::try_from(shape[2]).map_err(|_| {
            DripstoneError::storage("ONNX hidden dimension is negative or out of range")
        })?;
        let mut pooled = Self::apply_pooling(
            data,
            attention_mask,
            batch_size,
            seq_len,
            hidden_size,
            pooling,
        );
        if normalize {
            Self::normalize(&mut pooled, hidden_size);
        }
        Ok((0..batch_size)
            .map(|index| {
                let start = index * hidden_size;
                Vec::from(&pooled[start..start + hidden_size]).into()
            })
            .collect())
    }
}

#[cfg(feature = "local-ort")]
#[async_trait::async_trait]
impl VectorGenerator for OrtEmbedder {
    async fn generate(&self, texts: &[String]) -> Result<Vec<EmbeddingVector>, DripstoneError> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        // Run synchronous ORT operations in a blocking thread pool
        let texts = texts.to_vec();
        let tokenizer = self.tokenizer.clone();
        let session = self.session.clone();
        let config = self.config.clone();
        let input_names = self.input_names.clone();

        tokio::task::spawn_blocking(move || {
            let OrtTokenBatch {
                batch_size,
                max_len,
                input_ids,
                attention_mask,
                token_type_ids,
            } = Self::prepare_token_batch(&texts, &tokenizer, config.max_length)?;

            let arrays = Self::build_input_arrays(input_ids, &attention_mask, batch_size, max_len)?;
            let input_ids_value =
                TensorRef::<i64>::from_array_view(&arrays.input_ids).map_err(|e| {
                    DripstoneError::storage_with_source("failed to create input_ids value", e)
                })?;
            let attention_mask_value = TensorRef::<i64>::from_array_view(&arrays.attention_mask)
                .map_err(|e| {
                    DripstoneError::storage_with_source("failed to create attention_mask value", e)
                })?;

            // Run inference - lock the session mutex
            let mut session_guard = session.lock();
            let outputs = if let Some(token_type_name) = input_names.token_type_ids.as_ref() {
                let token_type_array = ndarray::Array::from_shape_vec(
                    (batch_size, max_len),
                    token_type_ids,
                )
                .map_err(|e| {
                    DripstoneError::storage_with_source("failed to create token_type_ids array", e)
                })?;
                let token_type_value = TensorRef::<i64>::from_array_view(&token_type_array)
                    .map_err(|e| {
                        DripstoneError::storage_with_source(
                            "failed to create token_type_ids value",
                            e,
                        )
                    })?;
                let inputs: std::collections::HashMap<String, ort::session::SessionInputValue<'_>> =
                    [
                        (input_names.input_ids.clone(), input_ids_value.into()),
                        (
                            input_names.attention_mask.clone(),
                            attention_mask_value.into(),
                        ),
                        (token_type_name.clone(), token_type_value.into()),
                    ]
                    .into_iter()
                    .collect();
                session_guard
                    .run(inputs)
                    .map_err(|e| DripstoneError::storage_with_source("ONNX inference failed", e))?
            } else {
                let inputs: std::collections::HashMap<String, ort::session::SessionInputValue<'_>> =
                    [
                        (input_names.input_ids.clone(), input_ids_value.into()),
                        (
                            input_names.attention_mask.clone(),
                            attention_mask_value.into(),
                        ),
                    ]
                    .into_iter()
                    .collect();
                session_guard
                    .run(inputs)
                    .map_err(|e| DripstoneError::storage_with_source("ONNX inference failed", e))?
            };

            // Extract output tensor (assuming first output is the embeddings)
            // Extract embeddings as ndarray view
            let (output_shape, output_data) =
                outputs[0].try_extract_tensor::<f32>().map_err(|e| {
                    DripstoneError::storage_with_source("failed to extract output tensor", e)
                })?;
            Self::finalize_output(
                output_shape,
                output_data,
                &attention_mask,
                batch_size,
                config.pooling,
                config.normalize,
            )
        })
        .await
        .map_err(|e| DripstoneError::storage_with_source("task join failed", e))?
    }

    fn dimension(&self) -> usize {
        self.dimension
    }

    fn name(&self) -> &'static str {
        "ort"
    }
}

#[cfg(not(feature = "local-ort"))]
pub struct OrtEmbedder;

#[cfg(not(feature = "local-ort"))]
impl OrtEmbedder {
    pub fn new(_config: EmbeddingConfig) -> Result<Self, DripstoneError> {
        Err(DripstoneError::Misconfiguration(
            "ORT embedder requires the 'local-ort' feature".to_string(),
        ))
    }
}
