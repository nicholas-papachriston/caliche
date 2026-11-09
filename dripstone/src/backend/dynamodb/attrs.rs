//! DynamoDB attribute encode/decode helpers for vector items.

use serde_json::{Map, Value, json};
use std::collections::HashMap;
use std::sync::Arc;

use crate::backend::chunk_envelope::{self, decode_chunk_from_envelope, encode_chunk_metadata};
use crate::core::{Chunk, Metric};
use crate::embedding::DEFAULT_NAMESPACE;
use crate::errors::DripstoneError;
use crate::search::{MultiVectors, multi_vectors_from_owned};

pub(crate) const ATTR_PK: &str = "pk";
pub(crate) const ATTR_SK: &str = "sk";
pub(crate) const ATTR_EMBEDDING: &str = "embedding";
pub(crate) const ATTR_ENVELOPE: &str = "envelope";
pub(crate) const ATTR_DOC_ID: &str = "doc_id";
pub(crate) const ATTR_MULTI_VECTORS: &str = "multi_vectors";

/// DynamoDB wire attribute map (`{"attr": {"S"|"N"|"L": ...}}`).
pub(crate) type AttributeMap = HashMap<String, Value>;

/// Resolve the partition-key namespace string written to DynamoDB.
#[must_use]
pub(crate) fn namespace_key(namespace: Option<&str>) -> &str {
    namespace.unwrap_or(DEFAULT_NAMESPACE)
}

fn format_f32(value: f32) -> String {
    if value.fract() == 0.0 && value.abs() < 1e15 {
        format!("{value:.1}")
    } else {
        format!("{value}")
    }
}

/// Encode an f32 slice as DynamoDB `L` of `N`.
#[must_use]
pub(crate) fn encode_vector_attr(vector: &[f32]) -> Value {
    json!({
        "L": vector.iter().map(|v| json!({ "N": format_f32(*v) })).collect::<Vec<_>>()
    })
}

/// Encode an f32 slice as a DynamoDB `SearchVector` list of `N` values.
#[must_use]
pub(crate) fn encode_search_vector(vector: &[f32]) -> Vec<Value> {
    vector
        .iter()
        .map(|v| json!({ "N": format_f32(*v) }))
        .collect()
}

/// Decode a DynamoDB list-of-numbers attribute into `Vec<f32>`.
pub(crate) fn decode_vector_attr(value: &Value) -> Result<Vec<f32>, DripstoneError> {
    let items = value
        .get("L")
        .and_then(Value::as_array)
        .ok_or_else(|| DripstoneError::storage("DynamoDB embedding attribute must be a List"))?;
    let mut out = Vec::with_capacity(items.len());
    for item in items {
        let text = item
            .get("N")
            .and_then(Value::as_str)
            .ok_or_else(|| DripstoneError::storage("DynamoDB embedding list element must be N"))?;
        let parsed = text.parse::<f32>().map_err(|e| {
            DripstoneError::storage(format!("invalid DynamoDB vector number '{text}': {e}"))
        })?;
        out.push(parsed);
    }
    Ok(out)
}

/// Map a DynamoDB `SearchVectors` raw score onto Dripstone metric semantics.
#[must_use]
pub(crate) fn map_dynamodb_score(metric: Metric, raw_score: f64) -> f32 {
    let raw = raw_score as f32;
    match metric {
        Metric::Cosine => (1.0 - raw).clamp(-1.0, 1.0),
        Metric::Euclidean | Metric::Manhattan => raw,
        Metric::DotProduct => raw,
    }
}

/// Convert a Dripstone [`Metric`] into the DynamoDB distance-function name.
pub(crate) fn distance_function_name(metric: Metric) -> Result<&'static str, DripstoneError> {
    match metric {
        Metric::Cosine => Ok("COSINE"),
        Metric::Euclidean => Ok("EUCLIDEAN"),
        Metric::DotProduct => Ok("DOT_PRODUCT"),
        Metric::Manhattan => Err(DripstoneError::storage(
            "DynamoDB vector indexes do not support Manhattan distance",
        )),
    }
}

fn attr_s(value: impl Into<String>) -> Value {
    json!({ "S": value.into() })
}

/// Build a PutItem attribute map for a chunk.
pub(crate) fn chunk_to_item(
    chunk: &Chunk,
    expected_dimension: usize,
) -> Result<AttributeMap, DripstoneError> {
    if chunk.vector.len() != expected_dimension {
        return Err(DripstoneError::storage(format!(
            "Vector dimension mismatch: expected {expected_dimension}, got {}",
            chunk.vector.len()
        )));
    }

    let ns = namespace_key(chunk.namespace.as_deref());
    let envelope = encode_chunk_metadata(chunk);
    let envelope_json = serde_json::to_string(&envelope)
        .map_err(|e| DripstoneError::storage(format!("failed to serialize chunk envelope: {e}")))?;

    let mut item = AttributeMap::new();
    item.insert(ATTR_PK.to_string(), attr_s(ns));
    item.insert(ATTR_SK.to_string(), attr_s(chunk.id.clone()));
    item.insert(
        ATTR_EMBEDDING.to_string(),
        encode_vector_attr(&chunk.vector),
    );
    item.insert(ATTR_ENVELOPE.to_string(), attr_s(envelope_json));
    item.insert(ATTR_DOC_ID.to_string(), attr_s(chunk.doc_id.clone()));

    if let Some(ref multi) = chunk.multi_vectors {
        let owned: Vec<Vec<f32>> = multi.iter().map(|token| token.as_ref().to_vec()).collect();
        let json = serde_json::to_string(&owned).map_err(|e| {
            DripstoneError::storage(format!("failed to serialize multi_vectors: {e}"))
        })?;
        item.insert(ATTR_MULTI_VECTORS.to_string(), attr_s(json));
    }

    Ok(item)
}

fn attr_string<'a>(item: &'a AttributeMap, key: &str) -> Result<&'a str, DripstoneError> {
    item.get(key)
        .and_then(|v| v.get("S"))
        .and_then(Value::as_str)
        .ok_or_else(|| DripstoneError::storage(format!("DynamoDB attribute '{key}' must be S")))
}

fn decode_multi_vectors(item: &AttributeMap) -> Option<MultiVectors> {
    let json = item.get(ATTR_MULTI_VECTORS)?.get("S")?.as_str()?;
    let parsed: Vec<Vec<f32>> = serde_json::from_str(json).ok()?;
    if parsed.is_empty() {
        return None;
    }
    Some(multi_vectors_from_owned(parsed))
}

/// Rehydrate a [`Chunk`] from a DynamoDB item.
pub(crate) fn item_to_chunk(item: &AttributeMap) -> Result<Chunk, DripstoneError> {
    let id = attr_string(item, ATTR_SK)?.to_string();
    let namespace = attr_string(item, ATTR_PK)?.to_string();
    let embedding = item
        .get(ATTR_EMBEDDING)
        .ok_or_else(|| DripstoneError::storage("DynamoDB item missing embedding attribute"))?;
    let vector = Arc::<[f32]>::from(decode_vector_attr(embedding)?);
    let envelope_json = attr_string(item, ATTR_ENVELOPE)?;
    let envelope: Value = serde_json::from_str(envelope_json)
        .map_err(|e| DripstoneError::storage(format!("failed to parse chunk envelope: {e}")))?;

    let mut chunk = decode_chunk_from_envelope(id, vector, &envelope, Some(namespace));
    chunk.multi_vectors = decode_multi_vectors(item);
    if let Ok(doc_id) = attr_string(item, ATTR_DOC_ID)
        && !doc_id.is_empty()
    {
        chunk.doc_id = doc_id.to_string();
    }
    Ok(chunk)
}

/// Extract document metadata used by [`crate::search::Filter::matches`].
#[must_use]
pub(crate) fn item_filter_metadata(item: &AttributeMap) -> Value {
    match item
        .get(ATTR_ENVELOPE)
        .and_then(|v| v.get("S"))
        .and_then(Value::as_str)
        .and_then(|s| serde_json::from_str::<Value>(s).ok())
    {
        Some(envelope) => chunk_envelope::filter_document_metadata(&envelope).clone(),
        None => Value::Null,
    }
}

/// Convert a JSON object of DynamoDB attributes into [`AttributeMap`].
pub(crate) fn attribute_map_from_value(value: &Value) -> Result<AttributeMap, DripstoneError> {
    let obj = value
        .as_object()
        .ok_or_else(|| DripstoneError::storage("DynamoDB item must be a JSON object"))?;
    Ok(obj.iter().map(|(k, v)| (k.clone(), v.clone())).collect())
}

/// Convert [`AttributeMap`] into a JSON object (for request bodies).
#[must_use]
pub(crate) fn attribute_map_to_value(map: &AttributeMap) -> Value {
    let mut obj = Map::new();
    for (k, v) in map {
        obj.insert(k.clone(), v.clone());
    }
    Value::Object(obj)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_chunk() -> Chunk {
        Chunk {
            id: "doc1#0".to_string(),
            doc_id: "doc1".to_string(),
            content: "hello".to_string(),
            vector: Arc::from([0.1_f32, -0.2, 0.5]),
            start_offset: 0,
            end_offset: 5,
            chunk_index: 0,
            total_chunks: 1,
            metadata: json!({"tag": "a"}),
            namespace: Some("ns-a".to_string()),
            multi_vectors: Some(multi_vectors_from_owned(vec![
                vec![0.1, 0.2],
                vec![0.3, 0.4],
            ])),
        }
    }

    #[test]
    fn round_trip_chunk_item() {
        let chunk = sample_chunk();
        let item = chunk_to_item(&chunk, 3).expect("encode");
        let decoded = item_to_chunk(&item).expect("decode");
        assert_eq!(decoded.id, chunk.id);
        assert_eq!(decoded.doc_id, chunk.doc_id);
        assert_eq!(decoded.content, chunk.content);
        assert_eq!(decoded.namespace.as_deref(), Some("ns-a"));
        assert_eq!(decoded.vector.as_ref(), chunk.vector.as_ref());
        assert_eq!(decoded.metadata, chunk.metadata);
        assert_eq!(
            decoded.multi_vectors.as_ref().map(|m| m.as_ref()),
            chunk.multi_vectors.as_ref().map(|m| m.as_ref())
        );
    }

    #[test]
    fn default_namespace_when_missing() {
        let mut chunk = sample_chunk();
        chunk.namespace = None;
        let item = chunk_to_item(&chunk, 3).expect("encode");
        assert_eq!(item.get(ATTR_PK), Some(&attr_s(DEFAULT_NAMESPACE)));
    }

    #[test]
    fn dimension_mismatch_errors() {
        let chunk = sample_chunk();
        let err = chunk_to_item(&chunk, 8).expect_err("dim");
        assert!(err.to_string().contains("dimension"));
    }

    #[test]
    fn map_cosine_identical_is_one() {
        assert!((map_dynamodb_score(Metric::Cosine, 0.0) - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn map_cosine_opposite_is_neg_one() {
        assert!((map_dynamodb_score(Metric::Cosine, 2.0) - (-1.0)).abs() < f32::EPSILON);
    }

    #[test]
    fn map_euclidean_passthrough() {
        assert!((map_dynamodb_score(Metric::Euclidean, 0.25) - 0.25).abs() < f32::EPSILON);
    }

    #[test]
    fn map_dot_passthrough() {
        assert!((map_dynamodb_score(Metric::DotProduct, 3.5) - 3.5).abs() < f32::EPSILON);
    }

    #[test]
    fn search_vector_len_matches() {
        let v = [1.0_f32, 2.0, 3.0];
        assert_eq!(encode_search_vector(&v).len(), 3);
    }

    #[test]
    fn filter_metadata_from_envelope() {
        let chunk = sample_chunk();
        let item = chunk_to_item(&chunk, 3).expect("encode");
        let meta = item_filter_metadata(&item);
        assert_eq!(meta.get("tag").and_then(Value::as_str), Some("a"));
    }
}
