//! Shared helpers for memory persistence (e.g. `DateTime` serialization in snapshot metadata).

use chrono::{DateTime, Utc};
use serde_json::{Map, Value};

const LAST_UPDATED_KEY: &str = "last_updated";

/// Insert `last_updated` into metadata map when present.
pub fn serialize_last_updated(
    metadata: &mut Map<String, Value>,
    last_updated: Option<DateTime<Utc>>,
) {
    if let Some(updated) = last_updated {
        metadata.insert(
            LAST_UPDATED_KEY.to_string(),
            Value::String(updated.to_rfc3339()),
        );
    }
}

/// Read `last_updated` from snapshot metadata if present.
pub fn deserialize_last_updated(metadata: &Map<String, Value>) -> Option<DateTime<Utc>> {
    metadata
        .get(LAST_UPDATED_KEY)
        .and_then(|v| v.as_str())
        .and_then(|s| DateTime::parse_from_rfc3339(s).ok())
        .map(|dt| dt.with_timezone(&Utc))
}
