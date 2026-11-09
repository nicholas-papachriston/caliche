use reqwest::header::{HeaderMap, HeaderValue};

pub const DEFAULT_CLASS_PREFIX: &str = "Dripstone_";
pub const DEFAULT_BATCH_SIZE: usize = 100;
pub const CREATED_KEY: &str = "_dripstone_created_at";
pub const UPDATED_KEY: &str = "_dripstone_updated_at";

use serde_json::Value;
use time::OffsetDateTime;
use uuid::Uuid;

use crate::embedding::{EmbeddingRecord, SimilarityMetric};
use crate::errors::DripstoneError;
use crate::utils::numeric::checked_f64_to_f32;

pub fn encode_metadata(
    metadata: Option<Value>,
    created_at: OffsetDateTime,
    updated_at: OffsetDateTime,
) -> (Option<Value>, OffsetDateTime, OffsetDateTime) {
    let metadata = match metadata {
        Some(Value::Object(map)) if !map.is_empty() => Some(Value::Object(map)),
        Some(other) => Some(other),
        None => None,
    };
    (metadata, created_at, updated_at)
}

pub fn extract_record(
    namespace: &str,
    body: &Value,
) -> Result<Option<EmbeddingRecord>, DripstoneError> {
    // Handle both REST API response and GraphQL response formats
    let properties = body
        .get("properties")
        .or_else(|| body.get("result"))
        .and_then(Value::as_object);

    // Get the original record ID from the record_id property (not the UUID)
    // Try top level first (GraphQL), then nested (REST API)
    let id = body
        .get("record_id")
        .and_then(Value::as_str)
        .or_else(|| {
            properties
                .and_then(|p| p.get("record_id"))
                .and_then(Value::as_str)
        })
        .ok_or_else(|| DripstoneError::storage("weaviate object missing record_id"))?;

    // Vector can be at top level (REST API) or in _additional.vector (GraphQL)
    let vector = body
        .get("vector")
        .or_else(|| body.get("_additional").and_then(|a| a.get("vector")))
        .or_else(|| body.get("result").and_then(|r| r.get("vector")))
        .map(parse_vector_value)
        .transpose()?
        .ok_or_else(|| DripstoneError::storage("weaviate object missing vector"))?;

    // Metadata is stored as a JSON string, so we need to parse it
    // Try top level first (GraphQL), then nested (REST API)
    let metadata_node = body
        .get("metadata")
        .or_else(|| properties.and_then(|p| p.get("metadata")))
        .and_then(|v| {
            v.as_str().map_or_else(
                || Some(v.clone()),
                |s| serde_json::from_str::<Value>(s).ok(),
            )
        })
        .unwrap_or(Value::Null);

    // Timestamps: try top level first (GraphQL), then nested (REST API)
    let created_at_str = body.get(CREATED_KEY).and_then(Value::as_str).or_else(|| {
        properties
            .and_then(|p| p.get(CREATED_KEY))
            .and_then(Value::as_str)
    });
    let updated_at_str = body.get(UPDATED_KEY).and_then(Value::as_str).or_else(|| {
        properties
            .and_then(|p| p.get(UPDATED_KEY))
            .and_then(Value::as_str)
    });

    let created_at = created_at_str
        .map(EmbeddingRecord::parse_timestamp)
        .transpose()?
        .unwrap_or_else(OffsetDateTime::now_utc);
    let updated_at = updated_at_str
        .map(EmbeddingRecord::parse_timestamp)
        .transpose()?
        .unwrap_or(created_at);

    let metadata = if metadata_node.is_null() {
        None
    } else if let Some(obj) = metadata_node.as_object() {
        if obj.is_empty() {
            None
        } else {
            Some(metadata_node)
        }
    } else {
        Some(metadata_node)
    };

    Ok(Some(EmbeddingRecord {
        id: std::sync::Arc::from(id),
        vector: vector.into(),
        namespace: Some(std::sync::Arc::from(namespace)),
        metadata,
        sparse_vector: None,
        multi_vectors: None,
        created_at,
        updated_at,
    }))
}

pub fn parse_vector_value(value: &Value) -> Result<Vec<f32>, DripstoneError> {
    if let Some(array) = value.as_array() {
        let mut vector = Vec::with_capacity(array.len());
        for item in array {
            let number = item
                .as_f64()
                .ok_or_else(|| DripstoneError::storage("vector element was not numeric"))?;
            vector.push(checked_f64_to_f32(number).ok_or_else(|| {
                DripstoneError::storage("weaviate vector element exceeds finite f32 range")
            })?);
        }
        return Ok(vector);
    }
    Err(DripstoneError::storage(
        "expected array of vector components from weaviate",
    ))
}

pub const fn similarity_metric_to_weaviate(metric: SimilarityMetric) -> &'static str {
    match metric {
        SimilarityMetric::Cosine => "cosine",
        SimilarityMetric::Dot => "dot",
        SimilarityMetric::Euclidean => "l2-squared",
        SimilarityMetric::Manhattan => "manhattan",
        SimilarityMetric::Chebyshev => "chebyshev",
    }
}

pub fn build_weaviate_headers(api_key: &str) -> Result<HeaderMap, DripstoneError> {
    let trimmed = api_key.trim();
    if trimmed.is_empty() {
        return Err(DripstoneError::Misconfiguration(
            "weaviate api key cannot be empty".to_string(),
        ));
    }
    let mut headers = HeaderMap::new();
    headers.insert(
        "Authorization",
        HeaderValue::from_str(&format!("Bearer {trimmed}")).map_err(|_| {
            DripstoneError::Misconfiguration("invalid characters in weaviate api key".to_string())
        })?,
    );
    Ok(headers)
}

/// Fixed namespace UUID for dripstone's Weaviate UUID v5 generation.
///
/// This is the well-known DNS namespace UUID (RFC 4122 §Appendix C).
/// Using the `uuid!()` macro makes the literal a compile-time constant,
/// so this cannot panic at runtime and no lazy initialisation is needed.
pub const DRIPSTONE_WEAVIATE_NAMESPACE: Uuid = uuid::uuid!("6ba7b810-9dad-11d1-80b4-00c04fd430c8");

/// Generate a deterministic UUID v5 for Weaviate object IDs.
/// Uses namespace + `record_id` as the input to ensure consistency.
pub fn generate_weaviate_uuid(namespace: &str, record_id: &str) -> Uuid {
    let name = format!("{namespace}:{record_id}");
    Uuid::new_v5(&DRIPSTONE_WEAVIATE_NAMESPACE, name.as_bytes())
}

// =============================================================================
