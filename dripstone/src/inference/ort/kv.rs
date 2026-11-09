//! KV-cache detection and tensor helpers for ORT generation.

use crate::errors::DripstoneError;
#[cfg(feature = "inference-ort")]
use ort::session::Session;
#[cfg(feature = "inference-ort")]
use ort::value::Tensor;

/// Names of inputs/outputs for KV cache (Optimum/transformers-style naming).
#[cfg(feature = "inference-ort")]
#[derive(Debug, Clone)]
pub(crate) struct OrtKvCacheConfig {
    pub(crate) input_ids_name: String,
    pub(crate) attention_mask_name: String,
    pub(crate) position_ids_name: Option<String>,
    /// (past_key_name, past_value_name) per layer, in order.
    pub(crate) past_names: Vec<(String, String)>,
    /// Empty prefill shapes for required past inputs when no cache exists yet.
    pub(crate) past_empty_shapes: Vec<(Vec<i64>, Vec<i64>)>,
    /// (present_key_name, present_value_name) per layer, in order.
    pub(crate) present_names: Vec<(String, String)>,
    pub(crate) logits_output_name: String,
}

#[cfg(feature = "inference-ort")]
#[derive(Debug, Clone)]
pub(crate) struct OrtSimpleInputNames {
    pub(crate) input_ids: String,
    pub(crate) attention_mask: String,
    pub(crate) position_ids: Option<String>,
    pub(crate) token_type_ids: Option<String>,
}

#[cfg(feature = "inference-ort")]
impl OrtSimpleInputNames {
    pub(crate) fn from_session(
        session: &Session,
        kv_cache_config: Option<&OrtKvCacheConfig>,
    ) -> Result<Self, DripstoneError> {
        let inputs = session.inputs();
        let input_ids = inputs
            .iter()
            .find(|input| input.name() == "input_ids")
            .or_else(|| inputs.first())
            .map(|input| input.name().to_string())
            .ok_or_else(|| DripstoneError::storage("ONNX inference model has no inputs"))?;
        let attention_mask = inputs
            .iter()
            .find(|input| input.name() == "attention_mask")
            .or_else(|| inputs.get(1))
            .map(|input| input.name().to_string())
            .ok_or_else(|| {
                DripstoneError::configuration(
                    "ONNX inference model must expose an attention_mask input or at least two inputs",
                )
            })?;
        let token_type_ids = inputs
            .iter()
            .find(|input| input.name() == "token_type_ids")
            .map(|input| input.name().to_string());
        let position_ids = inputs
            .iter()
            .find(|input| input.name() == "position_ids")
            .map(|input| input.name().to_string());

        let unsupported: Vec<String> = inputs
            .iter()
            .map(|input| input.name())
            .filter(|name| {
                *name != input_ids
                    && *name != attention_mask
                    && position_ids.as_deref() != Some(name)
                    && token_type_ids.as_deref() != Some(name)
                    && !kv_cache_config.is_some_and(|config| {
                        config
                            .past_names
                            .iter()
                            .any(|(k, v)| name == k || name == v)
                    })
            })
            .map(ToString::to_string)
            .collect();
        if !unsupported.is_empty() {
            return Err(DripstoneError::configuration(format!(
                "ONNX inference model requires unsupported inputs: {}",
                unsupported.join(", ")
            )));
        }

        Ok(Self {
            input_ids,
            attention_mask,
            position_ids,
            token_type_ids,
        })
    }
}

/// Detects whether the session has optional past/present KV inputs and outputs.
#[cfg(feature = "inference-ort")]
pub(crate) fn detect_kv_cache_io(session: &Session) -> Option<OrtKvCacheConfig> {
    let inputs = session.inputs();
    let outputs = session.outputs();
    if inputs.len() < 2 || outputs.is_empty() {
        return None;
    }
    let (input_ids_name, attention_mask_name, position_ids_name, past_names, past_empty_shapes) =
        collect_past_kv_inputs(inputs)?;
    let (logits_output_name, present_names) =
        collect_present_kv_outputs(outputs, past_names.len())?;
    Some(OrtKvCacheConfig {
        input_ids_name,
        attention_mask_name,
        position_ids_name,
        past_names,
        past_empty_shapes,
        present_names,
        logits_output_name,
    })
}

#[cfg(feature = "inference-ort")]
pub(crate) fn collect_past_kv_inputs(
    inputs: &[ort::value::Outlet],
) -> Option<(
    String,
    String,
    Option<String>,
    Vec<(String, String)>,
    Vec<(Vec<i64>, Vec<i64>)>,
)> {
    let mut input_ids_name: Option<String> = None;
    let mut attention_mask_name: Option<String> = None;
    let mut position_ids_name: Option<String> = None;
    let mut past_key_names: std::collections::BTreeMap<usize, String> =
        std::collections::BTreeMap::new();
    let mut past_value_names: std::collections::BTreeMap<usize, String> =
        std::collections::BTreeMap::new();
    let mut past_key_shapes: std::collections::BTreeMap<usize, Vec<i64>> =
        std::collections::BTreeMap::new();
    let mut past_value_shapes: std::collections::BTreeMap<usize, Vec<i64>> =
        std::collections::BTreeMap::new();
    for inp in inputs {
        record_past_kv_input(
            inp,
            &mut input_ids_name,
            &mut attention_mask_name,
            &mut position_ids_name,
            &mut past_key_names,
            &mut past_value_names,
            &mut past_key_shapes,
            &mut past_value_shapes,
        );
    }
    if input_ids_name.is_none() || attention_mask_name.is_none() {
        input_ids_name = Some(inputs[0].name().to_string());
        attention_mask_name = Some(inputs[1].name().to_string());
    }
    let input_ids_name = input_ids_name?;
    let attention_mask_name = attention_mask_name?;
    let num_layers = past_key_names.len().max(past_value_names.len());
    if num_layers == 0 {
        return None;
    }
    let past_names = pair_layer_names(&past_key_names, &past_value_names, num_layers)?;
    let past_empty_shapes = pair_layer_shapes(&past_key_shapes, &past_value_shapes, num_layers)?;
    Some((
        input_ids_name,
        attention_mask_name,
        position_ids_name,
        past_names,
        past_empty_shapes,
    ))
}

#[cfg(feature = "inference-ort")]
pub(crate) fn record_past_kv_input(
    inp: &ort::value::Outlet,
    input_ids_name: &mut Option<String>,
    attention_mask_name: &mut Option<String>,
    position_ids_name: &mut Option<String>,
    past_key_names: &mut std::collections::BTreeMap<usize, String>,
    past_value_names: &mut std::collections::BTreeMap<usize, String>,
    past_key_shapes: &mut std::collections::BTreeMap<usize, Vec<i64>>,
    past_value_shapes: &mut std::collections::BTreeMap<usize, Vec<i64>>,
) {
    let name = inp.name();
    if name == "input_ids" {
        *input_ids_name = Some(name.to_string());
    } else if name == "attention_mask" {
        *attention_mask_name = Some(name.to_string());
    } else if name == "position_ids" {
        *position_ids_name = Some(name.to_string());
    } else if name.contains("past_key_values")
        && name.contains(".key")
        && let Some(n) = parse_layer_index(name, "past_key_values", ".key")
    {
        past_key_names.insert(n, name.to_string());
        past_key_shapes.insert(n, empty_past_shape(inp));
    } else if name.contains("past_key_values")
        && name.contains(".value")
        && let Some(n) = parse_layer_index(name, "past_key_values", ".value")
    {
        past_value_names.insert(n, name.to_string());
        past_value_shapes.insert(n, empty_past_shape(inp));
    }
}

#[cfg(feature = "inference-ort")]
pub(crate) fn collect_present_kv_outputs(
    outputs: &[ort::value::Outlet],
    num_layers: usize,
) -> Option<(String, Vec<(String, String)>)> {
    let mut present_key_names: std::collections::BTreeMap<usize, String> =
        std::collections::BTreeMap::new();
    let mut present_value_names: std::collections::BTreeMap<usize, String> =
        std::collections::BTreeMap::new();
    let mut logits_output_name: Option<String> = None;
    for out in outputs {
        let name = out.name();
        if logits_output_name.is_none() && !is_kv_output_name(name) {
            logits_output_name = Some(name.to_string());
        }
        record_present_kv_output(name, &mut present_key_names, &mut present_value_names);
    }
    let logits_output_name = logits_output_name.unwrap_or_else(|| outputs[0].name().to_string());
    let present_num = present_key_names.len().max(present_value_names.len());
    if present_num != num_layers {
        return None;
    }
    let present_names = pair_layer_names(&present_key_names, &present_value_names, num_layers)?;
    Some((logits_output_name, present_names))
}

#[cfg(feature = "inference-ort")]
pub(crate) fn record_present_kv_output(
    name: &str,
    present_key_names: &mut std::collections::BTreeMap<usize, String>,
    present_value_names: &mut std::collections::BTreeMap<usize, String>,
) {
    if name.contains("present_key_values") && name.contains(".key") {
        if let Some(n) = parse_layer_index(name, "present_key_values", ".key") {
            present_key_names.insert(n, name.to_string());
        }
    } else if name.contains("present_key_values") && name.contains(".value") {
        if let Some(n) = parse_layer_index(name, "present_key_values", ".value") {
            present_value_names.insert(n, name.to_string());
        }
    } else if name.contains("past_key_values") && name.contains(".key") {
        if let Some(n) = parse_layer_index(name, "past_key_values", ".key") {
            present_key_names.insert(n, name.to_string());
        }
    } else if name.contains("past_key_values") && name.contains(".value") {
        if let Some(n) = parse_layer_index(name, "past_key_values", ".value") {
            present_value_names.insert(n, name.to_string());
        }
    } else if name.contains("present.")
        && name.contains(".key")
        && let Some(n) = parse_layer_index(name, "present", ".key")
    {
        present_key_names.insert(n, name.to_string());
    } else if name.contains("present.")
        && name.contains(".value")
        && let Some(n) = parse_layer_index(name, "present", ".value")
    {
        present_value_names.insert(n, name.to_string());
    }
}

#[cfg(feature = "inference-ort")]
pub(crate) fn pair_layer_names(
    keys: &std::collections::BTreeMap<usize, String>,
    values: &std::collections::BTreeMap<usize, String>,
    num_layers: usize,
) -> Option<Vec<(String, String)>> {
    let pairs: Vec<(String, String)> = (0..num_layers)
        .filter_map(|i| {
            let k = keys.get(&i)?.clone();
            let v = values.get(&i)?.clone();
            Some((k, v))
        })
        .collect();
    (pairs.len() == num_layers).then_some(pairs)
}

#[cfg(feature = "inference-ort")]
pub(crate) fn pair_layer_shapes(
    keys: &std::collections::BTreeMap<usize, Vec<i64>>,
    values: &std::collections::BTreeMap<usize, Vec<i64>>,
    num_layers: usize,
) -> Option<Vec<(Vec<i64>, Vec<i64>)>> {
    let pairs: Vec<(Vec<i64>, Vec<i64>)> = (0..num_layers)
        .filter_map(|i| {
            let k = keys.get(&i)?.clone();
            let v = values.get(&i)?.clone();
            Some((k, v))
        })
        .collect();
    (pairs.len() == num_layers).then_some(pairs)
}

#[cfg(feature = "inference-ort")]
pub(crate) fn is_kv_output_name(name: &str) -> bool {
    (name.contains("present_key_values")
        || name.contains("past_key_values")
        || name.contains("present."))
        && (name.contains(".key") || name.contains(".value"))
}

#[cfg(feature = "inference-ort")]
pub(crate) fn empty_past_shape(input: &ort::value::Outlet) -> Vec<i64> {
    match input.dtype() {
        ort::value::ValueType::Tensor { shape, .. } => shape
            .iter()
            .copied()
            .enumerate()
            .map(|(idx, dim)| match (idx, dim) {
                (0, _) => 1,
                (2, _) => 0,
                (_, dim) if dim > 0 => dim,
                (1, _) => 12,
                (3, _) => 64,
                _ => 0,
            })
            .collect(),
        _ => vec![1, 12, 0, 64],
    }
}

#[cfg(feature = "inference-ort")]
pub(crate) fn parse_layer_index(name: &str, prefix: &str, suffix: &str) -> Option<usize> {
    let name = name.strip_prefix(prefix)?.strip_suffix(suffix)?;
    let name = name.strip_prefix('.').unwrap_or(name);
    name.parse::<usize>().ok()
}

#[cfg(feature = "inference-ort")]
pub(crate) fn past_shape_with_sequence_len(shape: &[i64], sequence_len: i64) -> Vec<i64> {
    shape
        .iter()
        .copied()
        .enumerate()
        .map(|(idx, dim)| if idx == 2 { sequence_len } else { dim })
        .collect()
}

#[cfg(feature = "inference-ort")]
pub(crate) fn truncate_prompt_tokens(tokens: &mut Vec<i64>) {
    const DEFAULT_MAX_POSITION_EMBEDDINGS: usize = 512;
    if tokens.len() > DEFAULT_MAX_POSITION_EMBEDDINGS {
        let keep_from = tokens.len() - DEFAULT_MAX_POSITION_EMBEDDINGS;
        tokens.drain(..keep_from);
    }
}

/// Per-layer cache: (key shape, key data), (value shape, value data). F32 tensors.
#[cfg(feature = "inference-ort")]
pub(crate) struct OrtKvCacheLayer {
    pub(crate) key_shape: Vec<i64>,
    pub(crate) key_data: Vec<f32>,
    pub(crate) value_shape: Vec<i64>,
    pub(crate) value_data: Vec<f32>,
}

#[cfg(feature = "inference-ort")]
impl OrtKvCacheLayer {
    fn from_outputs(
        key_out: &ort::value::DynValue,
        value_out: &ort::value::DynValue,
    ) -> Result<Self, DripstoneError> {
        let (key_shape, key_data) = key_out
            .try_extract_tensor::<f32>()
            .map_err(|e| DripstoneError::storage_with_source("extract cache key tensor", e))?;
        let (value_shape, value_data) = value_out
            .try_extract_tensor::<f32>()
            .map_err(|e| DripstoneError::storage_with_source("extract cache value tensor", e))?;
        Ok(Self {
            key_shape: key_shape.iter().copied().collect(),
            key_data: key_data.to_vec(),
            value_shape: value_shape.iter().copied().collect(),
            value_data: value_data.to_vec(),
        })
    }

    /// Best-effort estimate of cached sequence length from key tensor shape.
    /// Most decoder exports use [batch, heads, seq_len, head_dim].
    fn sequence_len(&self) -> usize {
        if self.key_shape.len() >= 3 {
            self.key_shape[self.key_shape.len() - 2].max(0) as usize
        } else {
            0
        }
    }
}

#[cfg(feature = "inference-ort")]
pub(crate) fn step_seq_lens(
    input_ids: &[i64],
    cache: Option<&[OrtKvCacheLayer]>,
) -> (usize, usize, usize) {
    let seq_len = input_ids.len();
    let cached_seq_len = cache
        .and_then(|layers| layers.first())
        .map_or(0, OrtKvCacheLayer::sequence_len);
    (seq_len, cached_seq_len, seq_len + cached_seq_len)
}

#[cfg(feature = "inference-ort")]
pub(crate) fn i64_row_tensor(
    values: Vec<i64>,
    cols: usize,
    label: &'static str,
) -> Result<Tensor<i64>, DripstoneError> {
    Tensor::<i64>::from_array((vec![1i64, cols as i64], values))
        .map_err(|e| DripstoneError::storage_with_source(label, e))
}

#[cfg(feature = "inference-ort")]
pub(crate) fn position_ids_tensor(
    wants_position_ids: bool,
    cached_seq_len: usize,
    total_seq_len: usize,
    seq_len: usize,
) -> Result<Option<Tensor<i64>>, DripstoneError> {
    if !wants_position_ids {
        return Ok(None);
    }
    let position_ids: Vec<i64> = (cached_seq_len..total_seq_len)
        .map(|position| position as i64)
        .collect();
    Ok(Some(i64_row_tensor(
        position_ids,
        seq_len,
        "create position_ids tensor",
    )?))
}

#[cfg(feature = "inference-ort")]
pub(crate) fn past_tensors_from_layers(
    cache: &[OrtKvCacheLayer],
) -> Result<(Vec<Tensor<f32>>, Vec<Tensor<f32>>), DripstoneError> {
    let mut past_key_tensors = Vec::with_capacity(cache.len());
    let mut past_value_tensors = Vec::with_capacity(cache.len());
    for layer in cache {
        let key = Tensor::<f32>::from_array((layer.key_shape.clone(), layer.key_data.clone()))
            .map_err(|e| DripstoneError::storage_with_source("create past key tensor", e))?;
        let value =
            Tensor::<f32>::from_array((layer.value_shape.clone(), layer.value_data.clone()))
                .map_err(|e| DripstoneError::storage_with_source("create past value tensor", e))?;
        past_key_tensors.push(key);
        past_value_tensors.push(value);
    }
    Ok((past_key_tensors, past_value_tensors))
}

#[cfg(feature = "inference-ort")]
pub(crate) fn empty_past_tensors(
    past_empty_shapes: &[(Vec<i64>, Vec<i64>)],
) -> Result<(Vec<Tensor<f32>>, Vec<Tensor<f32>>), DripstoneError> {
    let allocator = ort::memory::Allocator::default();
    let mut past_key_tensors = Vec::with_capacity(past_empty_shapes.len());
    let mut past_value_tensors = Vec::with_capacity(past_empty_shapes.len());
    for (key_shape, value_shape) in past_empty_shapes {
        let key_shape = past_shape_with_sequence_len(key_shape, 0);
        let value_shape = past_shape_with_sequence_len(value_shape, 0);
        let key = Tensor::<f32>::new(&allocator, key_shape).map_err(|e| {
            DripstoneError::storage_with_source(format!("create empty past key tensor: {e}"), e)
        })?;
        let value = Tensor::<f32>::new(&allocator, value_shape).map_err(|e| {
            DripstoneError::storage_with_source(format!("create empty past value tensor: {e}"), e)
        })?;
        past_key_tensors.push(key);
        past_value_tensors.push(value);
    }
    Ok((past_key_tensors, past_value_tensors))
}

#[cfg(feature = "inference-ort")]
pub(crate) fn past_layer_tensors(
    cache: Option<&[OrtKvCacheLayer]>,
    past_empty_shapes: &[(Vec<i64>, Vec<i64>)],
) -> Result<(Vec<Tensor<f32>>, Vec<Tensor<f32>>), DripstoneError> {
    match cache {
        Some(layers) => past_tensors_from_layers(layers),
        None => empty_past_tensors(past_empty_shapes),
    }
}

#[cfg(feature = "inference-ort")]
pub(crate) fn collect_step_inputs<'a>(
    config: &'a OrtKvCacheConfig,
    input_tensor: &'a Tensor<i64>,
    mask_tensor: &'a Tensor<i64>,
    position_tensor: &'a Option<Tensor<i64>>,
    past_key_tensors: &'a [Tensor<f32>],
    past_value_tensors: &'a [Tensor<f32>],
) -> Vec<(String, ort::session::SessionInputValue<'a>)> {
    let mut inputs: Vec<(String, ort::session::SessionInputValue<'_>)> = vec![
        (config.input_ids_name.clone(), input_tensor.view().into()),
        (
            config.attention_mask_name.clone(),
            mask_tensor.view().into(),
        ),
    ];
    if let (Some(name), Some(tensor)) =
        (config.position_ids_name.as_ref(), position_tensor.as_ref())
    {
        inputs.push((name.clone(), tensor.view().into()));
    }
    for ((pk, pv), (k, v)) in config
        .past_names
        .iter()
        .zip(past_key_tensors.iter().zip(past_value_tensors.iter()))
    {
        inputs.push((pk.clone(), k.view().into()));
        inputs.push((pv.clone(), v.view().into()));
    }
    inputs
}

#[cfg(feature = "inference-ort")]
#[cfg(feature = "inference-ort")]
pub(crate) fn logits_last_position<'a>(
    logits_shape: &[i64],
    logits_data: &'a [f32],
    seq_len: usize,
) -> Result<&'a [f32], DripstoneError> {
    if logits_shape.len() == 3 {
        let vocab_size = logits_shape[2] as usize;
        let last_pos = (logits_shape[1] as usize - 1) * vocab_size;
        Ok(&logits_data[last_pos..last_pos + vocab_size])
    } else if logits_shape.len() == 2 {
        let vocab_size = logits_shape[1] as usize;
        let last_pos = (seq_len - 1) * vocab_size;
        Ok(&logits_data[last_pos..last_pos + vocab_size])
    } else {
        Err(DripstoneError::storage(
            "unexpected logits shape from ONNX model",
        ))
    }
}

pub(crate) fn logits_vec_from_outputs(
    outputs: &ort::session::SessionOutputs<'_>,
    logits_output_name: &str,
    seq_len: usize,
) -> Result<Vec<f32>, DripstoneError> {
    let logits_out = outputs
        .get(logits_output_name)
        .ok_or_else(|| DripstoneError::storage("logits output not found"))?;
    let (logits_shape, logits_data) = logits_out
        .try_extract_tensor::<f32>()
        .map_err(|e| DripstoneError::storage_with_source("failed to extract logits", e))?;
    let logits_slice = logits_last_position(logits_shape, logits_data, seq_len)?;
    Ok(logits_slice.to_vec())
}

#[cfg(feature = "inference-ort")]
pub(crate) fn present_layers_from_outputs(
    outputs: &ort::session::SessionOutputs<'_>,
    present_names: &[(String, String)],
) -> Result<Option<Vec<OrtKvCacheLayer>>, DripstoneError> {
    if present_names.is_empty() {
        return Ok(None);
    }
    let mut layers = Vec::with_capacity(present_names.len());
    for (pk, pv) in present_names {
        let key_out = outputs
            .get(pk.as_str())
            .ok_or_else(|| DripstoneError::storage("present key output not found"))?;
        let value_out = outputs
            .get(pv.as_str())
            .ok_or_else(|| DripstoneError::storage("present value output not found"))?;
        layers.push(OrtKvCacheLayer::from_outputs(key_out, value_out)?);
    }
    Ok(Some(layers))
}
