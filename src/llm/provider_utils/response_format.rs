//! Helpers for converting response format to provider-specific JSON.

use crate::output::ResponseFormat;

/// Converts a [`ResponseFormat`] to the JSON value expected by LLM provider APIs.
#[must_use]
pub fn response_format_to_json_value(rf: &ResponseFormat) -> serde_json::Value {
    rf.to_provider_json()
}
