//! Native MLX (mlx-rs) text generation.
//!
//! Supports MLX-format models: `params.json` + `weights.safetensors` in the model directory.
//! Convert HuggingFace models with `mlx_lm.convert` or use pre-converted MLX weights.

mod model;

use crate::errors::DripstoneError;
use crate::inference::{GenerationParams, sampling};
use mlx_rs::ops::indexing::IndexOp;
use mlx_rs::{
    Array,
    module::{Module, ModuleParametersExt},
    random::seed,
    transforms::eval,
};
use model::{Mistral, MistralInput, MistralOutput, ModelArgs};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokenizers::Tokenizer;

fn to_embedding_error(e: impl std::fmt::Display) -> DripstoneError {
    DripstoneError::storage(e.to_string())
}

fn resolve_mlx_weights_path(model_path: &Path) -> Result<PathBuf, DripstoneError> {
    let weights = model_path.join("weights.safetensors");
    if weights.exists() {
        return Ok(weights);
    }
    let model = model_path.join("model.safetensors");
    if model.exists() {
        return Ok(model);
    }
    Err(DripstoneError::storage(
        "weights.safetensors or model.safetensors not found in model path",
    ))
}

fn sample_from_mlx_logits(
    logits: &Array,
    params: &GenerationParams,
    previous_tokens: Option<&[usize]>,
) -> Result<u32, DripstoneError> {
    eval([logits]).map_err(to_embedding_error)?;
    let logits_slice = logits.as_slice::<f32>();
    sampling::sample_token(
        logits_slice,
        params.temperature,
        params.top_k,
        params.top_p,
        params.repetition_penalty,
        previous_tokens,
    )
    .map(|id| id as u32)
}

/// Run text generation using mlx-rs with a Mistral-format model.
///
/// Requires `params.json` and `weights.safetensors` (or `model.safetensors`) in `model_path`.
/// Returns an error if files are missing or generation fails, so the caller can fall back to the Python CLI.
pub fn generate_mlx_rs(
    model_path: &Path,
    tokenizer: Arc<Tokenizer>,
    prompt: &str,
    params: &GenerationParams,
) -> Result<String, DripstoneError> {
    let params_path = model_path.join("params.json");
    let params_str = std::fs::read_to_string(&params_path)
        .map_err(|e| DripstoneError::storage_with_source("failed to read params.json", e))?;
    let model_args: ModelArgs = serde_json::from_str(&params_str)
        .map_err(|e| DripstoneError::storage_with_source("failed to parse params.json", e))?;

    let weights_path = resolve_mlx_weights_path(model_path)?;

    let mut model = Mistral::new(&model_args).map_err(to_embedding_error)?;
    model
        .load_safetensors(&weights_path)
        .map_err(|e| DripstoneError::storage_with_source("failed to load weights", e))?;

    let encoding = tokenizer
        .encode(prompt, true)
        .map_err(|e| DripstoneError::storage(format!("tokenization failed: {}", e)))?;
    let ids: Vec<i32> = encoding.get_ids().iter().map(|&x| x as i32).collect();
    if ids.is_empty() {
        return Err(DripstoneError::storage(
            "empty prompt: tokenization produced no token IDs",
        ));
    }

    seed(0).map_err(to_embedding_error)?;

    let seq_len = ids.len() as i32;
    let prompt_tokens = Array::from_slice(&ids, &[1, seq_len]);
    let mut cache: Vec<Option<(Array, Array)>> = Vec::with_capacity(model_args.n_layers as usize);

    let mut generated: Vec<u32> = Vec::with_capacity(params.max_tokens);
    let eos_token = 2u32; // common EOS; could read from config

    let input = MistralInput {
        inputs: &prompt_tokens,
        cache: &cache,
    };
    let MistralOutput {
        logits,
        cache: out_cache,
    } = model.forward(input).map_err(to_embedding_error)?;
    cache = out_cache;

    let last_logits = logits.index((.., -1, ..));
    let next_id = sample_from_mlx_logits(&last_logits, params, None)?;
    if next_id == eos_token {
        return Ok(String::new());
    }
    generated.push(next_id);

    for _ in 1..params.max_tokens {
        let next_arr = Array::from_slice(&[next_id as i32], &[1, 1]);
        let input = MistralInput {
            inputs: &next_arr,
            cache: &cache,
        };
        let MistralOutput {
            logits,
            cache: new_cache,
        } = model.forward(input).map_err(to_embedding_error)?;
        cache = new_cache;

        let logits_1 = logits.squeeze_axes(&[1]).map_err(to_embedding_error)?;
        let prev: Vec<usize> = generated.iter().map(|&x| x as usize).collect();
        let next_id = sample_from_mlx_logits(&logits_1, params, Some(&prev))?;
        if next_id == eos_token {
            break;
        }
        generated.push(next_id);
    }

    let decoded = tokenizer
        .decode(&generated, true)
        .map_err(|e| DripstoneError::storage(format!("decode failed: {}", e)))?;
    Ok(decoded)
}
