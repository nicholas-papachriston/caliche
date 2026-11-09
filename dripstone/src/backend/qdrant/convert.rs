use super::client::QdrantEmbeddingBackend;
use super::helpers::{ORIGINAL_ID_KEY, ORIGINAL_NORM_KEY, parse_timestamp_field, parse_vector};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use time::OffsetDateTime;

use crate::embedding::{DEFAULT_NAMESPACE, EmbeddingRecord, SimilarityMetric};
use crate::errors::DripstoneError;
use crate::utils::numeric::checked_f64_to_f32;

pub struct EncodedPointId {
    transport_id: Value,
    #[expect(dead_code, reason = "stored with the encoded id but not read yet")]
    path_fragment: String,
    original_id: Option<String>,
}

pub fn encode_point_id(id: &str, namespace: &str) -> EncodedPointId {
    use uuid::Uuid;

    if let Ok(numeric) = id.parse::<u64>() {
        let fragment = numeric.to_string();
        return EncodedPointId {
            transport_id: Value::from(numeric),
            path_fragment: fragment,
            original_id: None,
        };
    }
    if Uuid::parse_str(id).is_ok() {
        return EncodedPointId {
            transport_id: Value::String(id.to_owned()),
            path_fragment: id.to_owned(),
            original_id: None,
        };
    }
    // Stable per (namespace, logical id) so HTTP upserts replace the same Qdrant point instead of
    // accumulating duplicates that share `ORIGINAL_ID_KEY` (which breaks scroll-by-original-id).
    let material = format!("dripstone|qdrant|{namespace}|{id}");
    let generated = Uuid::new_v5(&Uuid::NAMESPACE_URL, material.as_bytes()).to_string();
    EncodedPointId {
        transport_id: Value::String(generated.clone()),
        path_fragment: generated,
        original_id: Some(id.to_string()),
    }
}

pub fn extract_point_id(point: &Value) -> Result<String, DripstoneError> {
    match point.get("id") {
        Some(Value::String(value)) => Ok(value.clone()),
        Some(Value::Number(number)) => number.as_u64().map_or_else(
            || {
                number.as_i64().map_or_else(
                    || {
                        number.as_f64().map_or_else(
                            || {
                                Err(DripstoneError::storage(
                                    "unsupported numeric identifier returned by qdrant",
                                ))
                            },
                            |value| {
                                if value.is_sign_negative() {
                                    Err(DripstoneError::storage(
                                        "qdrant returned negative floating point identifier",
                                    ))
                                } else {
                                    Ok(value.trunc().to_string())
                                }
                            },
                        )
                    },
                    |value| {
                        if value >= 0 {
                            Ok(value.to_string())
                        } else {
                            Err(DripstoneError::storage(
                                "qdrant returned negative point identifier",
                            ))
                        }
                    },
                )
            },
            |value| Ok(value.to_string()),
        ),
        Some(other) => Ok(other.to_string()),
        None => Err(DripstoneError::storage("point missing id")),
    }
}

#[derive(Clone, Serialize, Deserialize)]
pub struct PointPayload {
    id: Value,
    vector: Vec<f32>,
    payload: Value,
}

#[derive(Serialize)]
pub struct UpsertRequest {
    pub(crate) points: Vec<PointPayload>,
}

impl QdrantEmbeddingBackend {
    pub(crate) fn point_from_record(
        &self,
        record: EmbeddingRecord,
    ) -> Result<(String, PointPayload), DripstoneError> {
        let EmbeddingRecord {
            id,
            vector,
            namespace,
            metadata,
            sparse_vector: _,
            multi_vectors: _,
            created_at,
            updated_at: _,
        } = record;
        let updated_at = OffsetDateTime::now_utc();
        if vector.len() != self.vector_dimension {
            return Err(DripstoneError::InvalidVectorLength {
                expected: self.vector_dimension,
                actual: vector.len(),
            });
        }
        let namespace = namespace.unwrap_or_else(|| std::sync::Arc::from(DEFAULT_NAMESPACE));
        let created_at = EmbeddingRecord::format_timestamp(&created_at)?;
        let updated_at_str = EmbeddingRecord::format_timestamp(&updated_at)?;
        let EncodedPointId {
            transport_id,
            original_id,
            path_fragment: _,
        } = encode_point_id(id.as_ref(), namespace.as_ref());
        let original_norm = if self.distance == SimilarityMetric::Cosine {
            let squared_sum: f64 = vector
                .iter()
                .map(|value| {
                    let value = f64::from(*value);
                    value * value
                })
                .sum();
            checked_f64_to_f32(squared_sum.sqrt())
        } else {
            None
        };
        let mut payload_map = Map::new();
        payload_map.insert("created_at".to_string(), Value::String(created_at));
        payload_map.insert("updated_at".to_string(), Value::String(updated_at_str));
        if let Some(meta) = metadata {
            payload_map.insert("metadata".to_string(), meta);
        }
        if let Some(original) = original_id {
            payload_map.insert(ORIGINAL_ID_KEY.to_string(), Value::String(original));
        }
        if let Some(norm) = original_norm {
            payload_map.insert(ORIGINAL_NORM_KEY.to_string(), Value::from(f64::from(norm)));
        }
        Ok((
            namespace.to_string(),
            PointPayload {
                id: transport_id,
                // Convert Arc<[f32]> to Vec<f32> for serialization
                vector: vector.to_vec(),
                payload: Value::Object(payload_map),
            },
        ))
    }

    pub(crate) fn record_from_point(
        &self,
        namespace: &str,
        point: &Value,
    ) -> Result<EmbeddingRecord, DripstoneError> {
        let transport_id = extract_point_id(point)?;
        let vector_value = point
            .get("vector")
            .ok_or_else(|| DripstoneError::storage("point missing vector"))?;
        let mut vector = parse_vector(vector_value)?;

        let mut payload = point
            .get("payload")
            .and_then(Value::as_object)
            .cloned()
            .unwrap_or_default();
        let metadata = payload.get("metadata").cloned();
        let created_at = parse_timestamp_field(payload.get("created_at"))?;
        let updated_at = parse_timestamp_field(payload.get("updated_at"))?;
        let original_id = payload
            .remove(ORIGINAL_ID_KEY)
            .and_then(|value| value.as_str().map(std::string::ToString::to_string));
        let original_norm = payload
            .remove(ORIGINAL_NORM_KEY)
            .and_then(|value| value.as_f64().and_then(checked_f64_to_f32));
        if self.distance == SimilarityMetric::Cosine
            && let Some(norm) = original_norm
            && norm.is_finite()
        {
            if norm > 0.0 {
                for value in &mut vector {
                    *value *= norm;
                }
            } else {
                vector.fill(0.0);
            }
        }
        let id = original_id.unwrap_or(transport_id);

        Ok(EmbeddingRecord {
            id: std::sync::Arc::from(id),
            vector: vector.into(),
            namespace: Some(std::sync::Arc::from(namespace)),
            metadata,
            sparse_vector: None,
            multi_vectors: None,
            created_at,
            updated_at,
        })
    }
}
