//! Minimal JSON Schema checks used by tools and input validators.
//!
//! Supported keywords: top-level and nested `type`, `required`, and `properties`.

use anyhow::{Context, Result, bail};
use serde_json::Value;

/// Validate `data` against a small JSON Schema subset.
pub fn validate_basic_schema(data: &Value, schema: &Value) -> Result<()> {
    if let Some(r#type) = schema.get("type").and_then(Value::as_str) {
        match r#type {
            "object" if !data.is_object() => {
                bail!("Expected object, got {}", value_type_name(data));
            }
            "array" if !data.is_array() => {
                bail!("Expected array, got {}", value_type_name(data));
            }
            "string" if !data.is_string() => {
                bail!("Expected string, got {}", value_type_name(data));
            }
            "number" if !data.is_number() => {
                bail!("Expected number, got {}", value_type_name(data));
            }
            "boolean" if !data.is_boolean() => {
                bail!("Expected boolean, got {}", value_type_name(data));
            }
            _ => {}
        }
    }

    if let Some(required) = schema.get("required").and_then(Value::as_array)
        && let Some(obj) = data.as_object()
    {
        for prop in required {
            if let Some(prop_name) = prop.as_str()
                && !obj.contains_key(prop_name)
            {
                bail!("Missing required property: {prop_name}");
            }
        }
    }

    if let Some(properties) = schema.get("properties").and_then(Value::as_object)
        && let Some(obj) = data.as_object()
    {
        for (key, prop_schema) in properties {
            if let Some(value) = obj.get(key) {
                validate_basic_schema(value, prop_schema)
                    .with_context(|| format!("Property '{key}' validation failed"))?;
            }
        }
    }

    Ok(())
}

/// Human-readable JSON value kind for error messages.
#[must_use]
pub const fn value_type_name(value: &Value) -> &'static str {
    match value {
        Value::Object(_) => "object",
        Value::Array(_) => "array",
        Value::String(_) => "string",
        Value::Number(_) => "number",
        Value::Bool(_) => "boolean",
        Value::Null => "null",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn validates_type_required_and_nested() {
        let schema = json!({
            "type": "object",
            "required": ["name"],
            "properties": {
                "name": { "type": "string" },
                "age": { "type": "number" }
            }
        });
        assert!(validate_basic_schema(&json!({"name": "a", "age": 1}), &schema).is_ok());
        assert!(validate_basic_schema(&json!({"age": 1}), &schema).is_err());
        assert!(validate_basic_schema(&json!({"name": 1}), &schema).is_err());
    }
}
