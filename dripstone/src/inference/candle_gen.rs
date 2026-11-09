//! Candle generation utilities and architecture-specific model implementations.

use crate::errors::DripstoneError;
use crate::inference::{GenerationParams, sampling};
#[cfg(feature = "inference-candle")]
use crate::loader::sanitize_relative_path;
use std::path::{Path, PathBuf};
use std::sync::Arc;

#[cfg(feature = "inference-candle")]
use candle_core::{DType, Device, Tensor};
#[cfg(feature = "inference-candle")]
use candle_nn::VarBuilder;
#[cfg(feature = "inference-candle")]
use candle_transformers::models::llama::Cache;
#[cfg(feature = "inference-candle")]
use tokenizers::Tokenizer;

/// Supported model architectures for Candle inference.
///
/// Phi and Gemma used to have dedicated variants, but were routed to a dead
/// "not yet implemented" error arm in practice. They have been intentionally
/// dropped from this enum: unknown architectures (including phi / phi2 /
/// phi3 / gemma / gemma2) now map to `Unknown(name)`, which is routed
/// through the Llama-compatible loader with a tracing warning. This is
/// honest about the feature set while still letting users attempt those
/// models rather than hard-failing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ModelArchitecture {
    Llama,
    Mistral,
    Qwen2,
    GptOss,
    Unknown(String),
}

impl ModelArchitecture {
    pub fn from_str(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "llama" | "llama2" | "llama3" => ModelArchitecture::Llama,
            "mistral" => ModelArchitecture::Mistral,
            "qwen2" => ModelArchitecture::Qwen2,
            "gpt_oss" | "gpt-oss" | "gptoss" => ModelArchitecture::GptOss,
            other => ModelArchitecture::Unknown(other.to_string()),
        }
    }
}

#[cfg(feature = "inference-candle")]
fn unsupported_candle_architecture(architecture: &str) -> DripstoneError {
    DripstoneError::Misconfiguration(format!(
        "Candle inference does not support {architecture} models yet. \
         GPT-OSS checkpoints use MoE layers and MXFP4 quantized weights that are not \
         compatible with the current Llama/Mistral/Qwen2 Candle loader. \
         Use an ONNX model, a supported Candle safetensors model, or build the CLI \
         with MLX support and select the MLX backend."
    ))
}

/// Model configuration parsed from config.json
#[cfg(feature = "inference-candle")]
#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub architecture: ModelArchitecture,
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub intermediate_size: usize,
    pub max_position_embeddings: usize,
    pub rms_norm_eps: f64,
    pub num_key_value_heads: Option<usize>,
    pub rope_theta: Option<f64>,
    pub bos_token_id: Option<u32>,
    pub eos_token_id: Option<u32>,
}

/// Detect model architecture from config.json
#[cfg(feature = "inference-candle")]
pub fn detect_architecture(config_path: &Path) -> Result<String, DripstoneError> {
    let config_str = std::fs::read_to_string(config_path)
        .map_err(|e| DripstoneError::storage_with_source("failed to read config.json", e))?;

    let config_json: serde_json::Value = serde_json::from_str(&config_str)
        .map_err(|e| DripstoneError::storage_with_source("failed to parse config.json", e))?;

    // Try to detect architecture from model_type field
    let arch = config_json
        .get("model_type")
        .and_then(|v| v.as_str())
        .ok_or_else(|| DripstoneError::storage("config.json missing 'model_type' field"))?;

    Ok(arch.to_string())
}

/// Parse full model configuration from config.json
#[cfg(feature = "inference-candle")]
pub fn parse_model_config(config_path: &Path) -> Result<ModelConfig, DripstoneError> {
    let config_str = std::fs::read_to_string(config_path)
        .map_err(|e| DripstoneError::storage_with_source("failed to read config.json", e))?;

    let json: serde_json::Value = serde_json::from_str(&config_str)
        .map_err(|e| DripstoneError::storage_with_source("failed to parse config.json", e))?;

    let model_type = json
        .get("model_type")
        .and_then(|v| v.as_str())
        .unwrap_or("unknown");

    let architecture = ModelArchitecture::from_str(model_type);

    let vocab_size = json
        .get("vocab_size")
        .and_then(|v| v.as_u64())
        .unwrap_or(32000) as usize;

    let hidden_size = json
        .get("hidden_size")
        .and_then(|v| v.as_u64())
        .unwrap_or(4096) as usize;

    let num_hidden_layers = json
        .get("num_hidden_layers")
        .and_then(|v| v.as_u64())
        .unwrap_or(32) as usize;

    let num_attention_heads = json
        .get("num_attention_heads")
        .and_then(|v| v.as_u64())
        .unwrap_or(32) as usize;

    let intermediate_size = json
        .get("intermediate_size")
        .and_then(|v| v.as_u64())
        .unwrap_or(11008) as usize;

    let max_position_embeddings = json
        .get("max_position_embeddings")
        .and_then(|v| v.as_u64())
        .unwrap_or(2048) as usize;

    let rms_norm_eps = json
        .get("rms_norm_eps")
        .and_then(|v| v.as_f64())
        .unwrap_or(1e-6);

    let num_key_value_heads = json
        .get("num_key_value_heads")
        .and_then(|v| v.as_u64())
        .map(|v| v as usize);

    let rope_theta = json.get("rope_theta").and_then(|v| v.as_f64());

    let bos_token_id = json
        .get("bos_token_id")
        .and_then(|v| v.as_u64())
        .map(|v| v as u32);

    let eos_token_id = json
        .get("eos_token_id")
        .and_then(|v| v.as_u64())
        .map(|v| v as u32);

    Ok(ModelConfig {
        architecture,
        vocab_size,
        hidden_size,
        num_hidden_layers,
        num_attention_heads,
        intermediate_size,
        max_position_embeddings,
        rms_norm_eps,
        num_key_value_heads,
        rope_theta,
        bos_token_id,
        eos_token_id,
    })
}

/// Find model weights in the model directory. Returns one path for single-file
/// models, or multiple paths (in deterministic order) for sharded models.
#[cfg(feature = "inference-candle")]
pub fn find_weights_paths(model_dir: &Path) -> Result<Vec<PathBuf>, DripstoneError> {
    let candidates = [
        "model.safetensors",
        "pytorch_model.safetensors",
        "model-00001-of-00001.safetensors",
        "consolidated.safetensors",
    ];

    for candidate in &candidates {
        let path = model_dir.join(candidate);
        if path.exists() {
            return Ok(vec![path]);
        }
    }

    // Sharded models: parse index and collect shard paths
    let index_path = model_dir.join("model.safetensors.index.json");
    if index_path.exists() {
        let index_str = std::fs::read_to_string(&index_path).map_err(|e| {
            DripstoneError::storage_with_source("failed to read safetensors index", e)
        })?;
        let index_json: serde_json::Value = serde_json::from_str(&index_str).map_err(|e| {
            DripstoneError::storage_with_source("failed to parse safetensors index", e)
        })?;
        let weight_map = index_json
            .get("weight_map")
            .and_then(|v| v.as_object())
            .ok_or_else(|| DripstoneError::storage("safetensors index missing 'weight_map'"))?;
        let mut shards: Vec<PathBuf> = Vec::new();
        for value in weight_map.values() {
            let Some(relative) = value.as_str() else {
                continue;
            };
            // Reject path traversal before mmap — index files are untrusted input.
            let path = sanitize_relative_path(model_dir, relative)?;
            shards.push(path);
        }
        shards.sort();
        shards.dedup();
        if shards.is_empty() {
            return Err(DripstoneError::storage(
                "safetensors index weight_map empty",
            ));
        }
        return Ok(shards);
    }

    Err(DripstoneError::storage(format!(
        "No model weights found in {}. Expected one of: {:?} or model.safetensors.index.json",
        model_dir.display(),
        candidates
    )))
}

/// Preferred compute dtype: F16 on accelerators, F32 on CPU.
#[cfg(feature = "inference-candle")]
pub fn inference_dtype(device: &Device) -> DType {
    if device.is_cpu() {
        DType::F32
    } else {
        DType::F16
    }
}

/// Llama / Mistral / Qwen2 weights loaded once and reused across requests.
#[cfg(feature = "inference-candle")]
pub struct LoadedLlamaModel {
    model: candle_transformers::models::llama::Llama,
    llama_config: candle_transformers::models::llama::Config,
    dtype: DType,
    eos_token_id: Option<u32>,
}

/// Load Llama-compatible weights once for reuse across generate/stream calls.
#[cfg(feature = "inference-candle")]
pub fn load_llama_model(
    model_dir: &Path,
    device: &Device,
    config: &ModelConfig,
) -> Result<LoadedLlamaModel, DripstoneError> {
    use candle_transformers::models::llama::{Config, Llama, LlamaEosToks};

    let weights_paths = find_weights_paths(model_dir)?;
    let dtype = inference_dtype(device);
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&weights_paths, dtype, device) }
        .map_err(|e| DripstoneError::storage_with_source("failed to load weights", e))?;

    let llama_config = Config {
        hidden_size: config.hidden_size,
        intermediate_size: config.intermediate_size,
        vocab_size: config.vocab_size,
        num_hidden_layers: config.num_hidden_layers,
        num_attention_heads: config.num_attention_heads,
        num_key_value_heads: config
            .num_key_value_heads
            .unwrap_or(config.num_attention_heads),
        rms_norm_eps: config.rms_norm_eps,
        rope_theta: config.rope_theta.unwrap_or(10000.0) as f32,
        bos_token_id: config.bos_token_id,
        eos_token_id: config.eos_token_id.map(LlamaEosToks::Single),
        rope_scaling: None,
        max_position_embeddings: config.max_position_embeddings,
        tie_word_embeddings: false,
        use_flash_attn: false,
    };

    let model = Llama::load(vb, &llama_config)
        .map_err(|e| DripstoneError::storage_with_source("failed to load Llama model", e))?;

    Ok(LoadedLlamaModel {
        model,
        llama_config,
        dtype,
        eos_token_id: config.eos_token_id,
    })
}

#[cfg(feature = "inference-candle")]
fn llama_eos_token(loaded: &LoadedLlamaModel, tokenizer: &Tokenizer) -> u32 {
    loaded
        .eos_token_id
        .or_else(|| tokenizer.token_to_id("</s>"))
        .or_else(|| tokenizer.token_to_id("<|endoftext|>"))
        .unwrap_or(2)
}

#[cfg(feature = "inference-candle")]
fn llama_forward_logits(
    loaded: &LoadedLlamaModel,
    device: &Device,
    tokens: &[u32],
    pos: &mut usize,
    cache: &mut Cache,
) -> Result<Vec<f32>, DripstoneError> {
    let input_len = if *pos == 0 { tokens.len() } else { 1 };
    let input_tokens: &[u32] = if *pos == 0 {
        tokens
    } else {
        let last = tokens
            .last()
            .ok_or_else(|| DripstoneError::storage("empty tokens"))?;
        std::slice::from_ref(last)
    };

    let input = Tensor::new(input_tokens, device)
        .map_err(|e| DripstoneError::storage_with_source("failed to create input tensor", e))?
        .unsqueeze(0)
        .map_err(|e| DripstoneError::storage_with_source("failed to unsqueeze", e))?;

    let logits = loaded
        .model
        .forward(&input, *pos, cache)
        .map_err(|e| DripstoneError::storage_with_source("forward pass failed", e))?;

    *pos += input_len;

    let logits = logits
        .squeeze(0)
        .map_err(|e| DripstoneError::storage_with_source("squeeze failed", e))?;
    let logits = if logits.dims().len() == 2 {
        logits
            .get(logits.dim(0).unwrap_or(1) - 1)
            .map_err(|e| DripstoneError::storage_with_source("get last logits failed", e))?
    } else {
        logits
    };

    logits
        .to_vec1()
        .map_err(|e| DripstoneError::storage_with_source("failed to extract logits", e))
}

#[cfg(feature = "inference-candle")]
fn first_stop_trim<'a>(text: &'a str, stop_sequences: &[String]) -> Option<&'a str> {
    for stop_seq in stop_sequences {
        if text.contains(stop_seq) {
            return Some(text.split(stop_seq).next().unwrap_or(text));
        }
    }
    None
}

/// Returns `true` when generation should stop (stop sequence hit or channel closed).
#[cfg(feature = "inference-candle")]
fn send_streaming_stop_piece(
    tokenizer: &Tokenizer,
    generated_tokens: &[u32],
    next_token: u32,
    stop_sequences: &[String],
    token_tx: &tokio::sync::mpsc::Sender<Result<String, DripstoneError>>,
) -> Result<bool, DripstoneError> {
    if stop_sequences.is_empty() {
        return Ok(false);
    }
    let text_so_far = tokenizer
        .decode(generated_tokens, true)
        .map_err(|e| DripstoneError::storage(format!("decode failed: {}", e)))?;
    let mut with_next = generated_tokens.to_vec();
    with_next.push(next_token);
    let text_with_next = tokenizer
        .decode(&with_next, true)
        .map_err(|e| DripstoneError::storage(format!("decode failed: {}", e)))?;

    let Some(trimmed) = first_stop_trim(&text_with_next, stop_sequences) else {
        return Ok(false);
    };
    let to_send = trimmed
        .strip_prefix(text_so_far.as_str())
        .unwrap_or(trimmed);
    if !to_send.is_empty() && token_tx.blocking_send(Ok(to_send.to_string())).is_err() {
        return Ok(true);
    }
    Ok(true)
}

/// Generate text using an already-loaded Llama model.
#[cfg(feature = "inference-candle")]
pub fn generate_llama_loaded(
    loaded: &LoadedLlamaModel,
    tokenizer: &Tokenizer,
    device: &Device,
    prompt: &str,
    params: &GenerationParams,
) -> Result<String, DripstoneError> {
    let encoding = tokenizer
        .encode(prompt, true)
        .map_err(|e| DripstoneError::storage(format!("tokenization failed: {}", e)))?;

    let mut tokens: Vec<u32> = encoding.get_ids().to_vec();
    if tokens.is_empty() {
        return Err(DripstoneError::storage(
            "empty prompt: tokenization produced no token IDs",
        ));
    }
    let mut generated_tokens = Vec::new();
    let mut previous_token_ids: Vec<usize> = Vec::new();
    let eos_token = llama_eos_token(loaded, tokenizer);

    let mut cache = Cache::new(true, loaded.dtype, &loaded.llama_config, device)
        .map_err(|e| DripstoneError::storage_with_source("failed to create KV cache", e))?;

    let mut pos = 0usize;
    for _step in 0..params.max_tokens {
        let logits_vec = llama_forward_logits(loaded, device, &tokens, &mut pos, &mut cache)?;
        let next_token = sampling::sample_token(
            &logits_vec,
            params.temperature,
            params.top_k,
            params.top_p,
            params.repetition_penalty,
            Some(&previous_token_ids),
        )? as u32;

        if next_token == eos_token {
            break;
        }

        generated_tokens.push(next_token);
        previous_token_ids.push(next_token as usize);
        tokens.push(next_token);

        if !params.stop_sequences.is_empty() {
            let current_text = tokenizer
                .decode(&generated_tokens, true)
                .map_err(|e| DripstoneError::storage(format!("decode failed: {}", e)))?;
            if let Some(trimmed) = first_stop_trim(&current_text, &params.stop_sequences) {
                return Ok(trimmed.to_string());
            }
        }
    }

    let output = tokenizer
        .decode(&generated_tokens, true)
        .map_err(|e| DripstoneError::storage(format!("decode failed: {}", e)))?;

    Ok(output)
}

/// Generate text using Llama architecture (loads weights each call — prefer loaded path).
#[cfg(feature = "inference-candle")]
pub fn generate_llama(
    model_dir: &Path,
    tokenizer: Arc<Tokenizer>,
    device: &Device,
    prompt: &str,
    params: &GenerationParams,
    config: &ModelConfig,
) -> Result<String, DripstoneError> {
    let loaded = load_llama_model(model_dir, device, config)?;
    generate_llama_loaded(&loaded, tokenizer.as_ref(), device, prompt, params)
}

/// Generate text using Qwen2 architecture.
///
/// Qwen2 is decoder-only and compatible with the Llama execution path used here.
#[cfg(feature = "inference-candle")]
pub fn generate_qwen2(
    model_dir: &Path,
    tokenizer: Arc<Tokenizer>,
    device: &Device,
    prompt: &str,
    params: &GenerationParams,
    config: &ModelConfig,
) -> Result<String, DripstoneError> {
    generate_llama(model_dir, tokenizer, device, prompt, params, config)
}

/// Generate text with streaming - yields tokens as they're generated via channel
#[cfg(feature = "inference-candle")]
#[expect(dead_code, reason = "nothing calls it yet")]
pub fn generate_streaming(
    model_dir: &Path,
    tokenizer: Arc<Tokenizer>,
    device: &Device,
    prompt: &str,
    params: &GenerationParams,
    token_tx: tokio::sync::mpsc::Sender<Result<String, DripstoneError>>,
) -> Result<(), DripstoneError> {
    // Parse config
    let config_path = model_dir.join("config.json");
    let config = parse_model_config(&config_path)?;

    match config.architecture {
        ModelArchitecture::Llama | ModelArchitecture::Mistral => generate_llama_streaming(
            model_dir, tokenizer, device, prompt, params, &config, token_tx,
        ),
        ModelArchitecture::Qwen2 => generate_qwen2_streaming(
            model_dir, tokenizer, device, prompt, params, &config, token_tx,
        ),
        ModelArchitecture::GptOss => Err(unsupported_candle_architecture("gpt_oss")),
        ModelArchitecture::Unknown(ref arch) => {
            tracing::warn!(
                "Unknown architecture '{}', attempting Llama-compatible loading",
                arch
            );
            generate_llama_streaming(
                model_dir, tokenizer, device, prompt, params, &config, token_tx,
            )
        }
    }
}

/// Streaming generation using an already-loaded Llama model.
#[cfg(feature = "inference-candle")]
pub fn generate_llama_streaming_loaded(
    loaded: &LoadedLlamaModel,
    tokenizer: &Tokenizer,
    device: &Device,
    prompt: &str,
    params: &GenerationParams,
    token_tx: tokio::sync::mpsc::Sender<Result<String, DripstoneError>>,
) -> Result<(), DripstoneError> {
    let encoding = tokenizer
        .encode(prompt, true)
        .map_err(|e| DripstoneError::storage(format!("tokenization failed: {}", e)))?;

    let mut tokens: Vec<u32> = encoding.get_ids().to_vec();
    if tokens.is_empty() {
        return Err(DripstoneError::storage(
            "empty prompt: tokenization produced no token IDs",
        ));
    }
    let mut generated_tokens = Vec::new();
    let mut previous_token_ids: Vec<usize> = Vec::new();
    let eos_token = llama_eos_token(loaded, tokenizer);

    let mut cache = Cache::new(true, loaded.dtype, &loaded.llama_config, device)
        .map_err(|e| DripstoneError::storage_with_source("failed to create KV cache", e))?;

    let mut pos = 0usize;
    for _step in 0..params.max_tokens {
        let logits_vec = llama_forward_logits(loaded, device, &tokens, &mut pos, &mut cache)?;
        let next_token = sampling::sample_token(
            &logits_vec,
            params.temperature,
            params.top_k,
            params.top_p,
            params.repetition_penalty,
            Some(&previous_token_ids),
        )? as u32;

        if next_token == eos_token {
            break;
        }

        if send_streaming_stop_piece(
            tokenizer,
            &generated_tokens,
            next_token,
            &params.stop_sequences,
            &token_tx,
        )? {
            return Ok(());
        }

        generated_tokens.push(next_token);
        previous_token_ids.push(next_token as usize);
        tokens.push(next_token);

        let token_text = tokenizer
            .decode(&[next_token], false)
            .map_err(|e| DripstoneError::storage(format!("decode failed: {}", e)))?;

        if token_tx.blocking_send(Ok(token_text)).is_err() {
            return Ok(());
        }
    }

    Ok(())
}

/// Streaming version of generate_llama - yields tokens as generated
#[cfg(feature = "inference-candle")]
fn generate_llama_streaming(
    model_dir: &Path,
    tokenizer: Arc<Tokenizer>,
    device: &Device,
    prompt: &str,
    params: &GenerationParams,
    config: &ModelConfig,
    token_tx: tokio::sync::mpsc::Sender<Result<String, DripstoneError>>,
) -> Result<(), DripstoneError> {
    let loaded = load_llama_model(model_dir, device, config)?;
    generate_llama_streaming_loaded(
        &loaded,
        tokenizer.as_ref(),
        device,
        prompt,
        params,
        token_tx,
    )
}

/// Streaming generation for Qwen2 architecture.
///
/// Qwen2 currently reuses the Llama-compatible streaming path.
#[cfg(feature = "inference-candle")]
fn generate_qwen2_streaming(
    model_dir: &Path,
    tokenizer: Arc<Tokenizer>,
    device: &Device,
    prompt: &str,
    params: &GenerationParams,
    config: &ModelConfig,
    token_tx: tokio::sync::mpsc::Sender<Result<String, DripstoneError>>,
) -> Result<(), DripstoneError> {
    generate_llama_streaming(
        model_dir, tokenizer, device, prompt, params, config, token_tx,
    )
}

/// Main generation entry point - routes to architecture-specific implementation
#[cfg(feature = "inference-candle")]
#[expect(dead_code, reason = "nothing calls it yet")]
pub fn generate(
    model_dir: &Path,
    tokenizer: Arc<Tokenizer>,
    device: &Device,
    prompt: &str,
    params: &GenerationParams,
) -> Result<String, DripstoneError> {
    // Parse config
    let config_path = model_dir.join("config.json");
    let config = parse_model_config(&config_path)?;

    match config.architecture {
        ModelArchitecture::Llama | ModelArchitecture::Mistral => {
            // Mistral is a minor variation of Llama; the Llama loader works
            // for most published Mistral checkpoints.
            generate_llama(model_dir, tokenizer, device, prompt, params, &config)
        }
        ModelArchitecture::Qwen2 => {
            generate_qwen2(model_dir, tokenizer, device, prompt, params, &config)
        }
        ModelArchitecture::GptOss => Err(unsupported_candle_architecture("gpt_oss")),
        ModelArchitecture::Unknown(ref arch) => {
            tracing::warn!(
                "Unknown architecture '{}', attempting Llama-compatible loading",
                arch
            );
            generate_llama(model_dir, tokenizer, device, prompt, params, &config)
        }
    }
}

#[cfg(all(test, feature = "inference-candle"))]
mod tests {
    use super::*;

    #[test]
    fn find_weights_paths_rejects_traversal_in_index() {
        let temp = tempfile::tempdir().expect("tempdir");
        let index = temp.path().join("model.safetensors.index.json");
        std::fs::write(&index, r#"{"weight_map":{"a":"../escape.safetensors"}}"#)
            .expect("write index");
        let err = find_weights_paths(temp.path()).expect_err("traversal must fail");
        assert!(err.to_string().contains("path traversal"));
    }
}
