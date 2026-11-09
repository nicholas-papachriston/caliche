//! Schema definitions for structured output support.
//!
//! This module provides types for configuring response formats including
//! JSON object mode and JSON schema mode with validation.

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Response format configuration for structured outputs
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
#[derive(Default)]
pub enum ResponseFormat {
    /// Plain text response (default)
    #[serde(rename = "text")]
    #[default]
    Text,

    /// JSON object response (any valid JSON)
    #[serde(rename = "json_object")]
    JsonObject,

    /// JSON response conforming to a specific schema
    #[serde(rename = "json_schema")]
    JsonSchema { json_schema: JsonSchemaDefinition },
}

impl ResponseFormat {
    /// Convert to the JSON format expected by providers
    pub fn to_provider_json(&self) -> serde_json::Value {
        match self {
            ResponseFormat::Text => serde_json::json!({"type": "text"}),
            ResponseFormat::JsonObject => serde_json::json!({"type": "json_object"}),
            ResponseFormat::JsonSchema { json_schema } => serde_json::json!({
                "type": "json_schema",
                "json_schema": {
                    "name": json_schema.name,
                    "description": json_schema.description,
                    "schema": json_schema.schema,
                    "strict": json_schema.strict
                }
            }),
        }
    }
}

/// JSON Schema definition for structured output validation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonSchemaDefinition {
    /// Name of the schema (used by LLM providers)
    pub name: String,
    /// Optional description of the schema
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// JSON Schema object (must be valid JSON Schema)
    pub schema: Value,
    /// Whether to enforce strict validation (default: true)
    #[serde(default = "default_strict")]
    pub strict: bool,
}

impl JsonSchemaDefinition {
    /// Normalize the schema to ensure it's compatible with OpenAI's requirements
    /// Specifically, adds `additionalProperties: false` to object schemas if not present
    pub fn normalize_schema(mut schema: Value) -> Value {
        if let Some(obj) = schema.as_object_mut() {
            // If it's an object schema, ensure additionalProperties is set
            if obj.get("type") == Some(&Value::String("object".to_string()))
                && !obj.contains_key("additionalProperties")
            {
                obj.insert("additionalProperties".to_string(), Value::Bool(false));
            }

            // Recursively normalize nested schemas in properties
            if let Some(properties) = obj.get_mut("properties")
                && let Some(props_obj) = properties.as_object_mut()
            {
                for prop_value in props_obj.values_mut() {
                    *prop_value = Self::normalize_schema(prop_value.clone());
                }
            }

            // Normalize items in arrays
            if let Some(items) = obj.get_mut("items") {
                *items = Self::normalize_schema(items.clone());
            }

            // Normalize anyOf, allOf, oneOf arrays
            for key in &["anyOf", "allOf", "oneOf"] {
                if let Some(Value::Array(arr)) = obj.get_mut(*key) {
                    for item in arr.iter_mut() {
                        *item = Self::normalize_schema(item.clone());
                    }
                }
            }
        }
        schema
    }
}

fn default_strict() -> bool {
    true
}

/// Configuration for structured output with retry
#[derive(Debug, Clone)]
pub struct StructuredOutputConfig {
    /// Response format to enforce
    pub format: ResponseFormat,
    /// Maximum number of retries on validation failure
    pub max_retries: u32,
    /// Custom retry prompt (if None, uses default)
    pub retry_prompt: Option<String>,
}

impl StructuredOutputConfig {
    /// Create a new configuration with default retry settings
    pub fn new(format: ResponseFormat) -> Self {
        Self {
            format,
            max_retries: 2,
            retry_prompt: None,
        }
    }

    /// Set maximum retries
    pub fn with_max_retries(mut self, max_retries: u32) -> Self {
        self.max_retries = max_retries;
        self
    }

    /// Set custom retry prompt
    pub fn with_retry_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.retry_prompt = Some(prompt.into());
        self
    }
}
