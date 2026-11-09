#![cfg(any(feature = "qdrant-backend", feature = "qdrant-grpc-backend"))]

use serde_json::{Value, json};
use time::OffsetDateTime;

#[cfg(feature = "qdrant-backend")]
use crate::backend::http::HttpClient;
use crate::embedding::{EmbeddingRecord, SimilarityMetric};
use crate::errors::DripstoneError;
use crate::utils::numeric::checked_f64_to_f32;

pub const DEFAULT_COLLECTION_PREFIX: &str = "embeddings_";
pub const SCROLL_PAGE_SIZE: usize = 64;
pub const ORIGINAL_ID_KEY: &str = "__dripstone_original_id";
pub const ORIGINAL_NORM_KEY: &str = "__dripstone_vector_norm";

pub fn normalize_prefix(prefix: Option<String>) -> String {
    match prefix {
        Some(value) if !value.is_empty() => value,
        Some(_) | None => DEFAULT_COLLECTION_PREFIX.to_string(),
    }
}

pub const fn qdrant_distance(metric: SimilarityMetric) -> &'static str {
    match metric {
        SimilarityMetric::Cosine => "Cosine",
        SimilarityMetric::Dot => "Dot",
        // Qdrant doesn't support Manhattan/Chebyshev natively, use Euclidean as fallback.
        SimilarityMetric::Euclidean | SimilarityMetric::Manhattan | SimilarityMetric::Chebyshev => {
            "Euclid"
        }
    }
}

pub fn parse_vector(value: &Value) -> Result<Vec<f32>, DripstoneError> {
    if let Some(array) = value.as_array() {
        let mut output = Vec::with_capacity(array.len());
        for item in array {
            let number = item
                .as_f64()
                .ok_or_else(|| DripstoneError::storage("vector element was not numeric"))?;
            output.push(checked_f64_to_f32(number).ok_or_else(|| {
                DripstoneError::storage("vector element exceeds finite f32 range")
            })?);
        }
        return Ok(output);
    }

    if let Some(object) = value.as_object()
        && let Some(first) = object.values().next()
    {
        return parse_vector(first);
    }

    Err(DripstoneError::storage(
        "unexpected vector structure returned by qdrant",
    ))
}

pub fn parse_timestamp_field(value: Option<&Value>) -> Result<OffsetDateTime, DripstoneError> {
    match value {
        Some(Value::String(string)) => EmbeddingRecord::parse_timestamp(string.as_str()),
        Some(other) => EmbeddingRecord::parse_timestamp(other.to_string().as_str()),
        None => Ok(OffsetDateTime::now_utc()),
    }
}

#[cfg(feature = "qdrant-backend")]
pub async fn count_namespace(http: &HttpClient, collection: &str) -> Result<usize, DripstoneError> {
    let response = match http
        .post_json(
            &format!("/collections/{collection}/points/count"),
            &json!({ "exact": true }),
        )
        .await
    {
        Ok(response) => {
            if response.status() == reqwest::StatusCode::NOT_FOUND {
                return Ok(0);
            }
            response
        }
        Err(e) => {
            // Check if the error is a 404 (collection doesn't exist)
            // The HTTP client wraps 404s as permanent errors
            let error_msg = format!("{e}");
            if error_msg.contains("404")
                || error_msg.contains("Not found")
                || error_msg.contains("doesn't exist")
            {
                return Ok(0);
            }
            return Err(e);
        }
    };
    let response = response.error_for_status().map_err(DripstoneError::from)?;
    let body: Value = response.json().await.map_err(DripstoneError::from)?;
    let status = body
        .get("status")
        .and_then(Value::as_str)
        .unwrap_or("error");
    if status != "ok" {
        return Err(DripstoneError::storage(format!(
            "qdrant count failed: {body}"
        )));
    }
    let count = body
        .get("result")
        .and_then(Value::as_object)
        .and_then(|result| result.get("count"))
        .and_then(Value::as_u64)
        .unwrap_or(0);
    usize::try_from(count)
        .map_err(|_| DripstoneError::storage("qdrant count exceeds addressable size"))
}
