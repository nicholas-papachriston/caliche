//! JSON output parser with schema validation.

use super::parser::{OutputParser, ParseError};
use async_trait::async_trait;
use serde_json::Value;

#[cfg(feature = "output-parsing")]
use jsonschema::Validator;

/// Parser for JSON outputs.
#[derive(Debug)]
pub struct JsonOutputParser {
    #[cfg(feature = "output-parsing")]
    schema: Option<Validator>,
    pretty_print: bool,
}

impl Clone for JsonOutputParser {
    fn clone(&self) -> Self {
        // JSONSchema doesn't implement Clone, so we recreate without schema
        Self {
            #[cfg(feature = "output-parsing")]
            schema: None,
            pretty_print: self.pretty_print,
        }
    }
}

impl JsonOutputParser {
    /// Create a new JSON parser without schema validation.
    pub fn new() -> Self {
        Self {
            #[cfg(feature = "output-parsing")]
            schema: None,
            pretty_print: false,
        }
    }

    /// Create a JSON parser with schema validation (requires `output-parsing` feature).
    #[cfg(feature = "output-parsing")]
    pub fn with_schema(schema: Value) -> Result<Self, ParseError> {
        let validator = jsonschema::validator_for(&schema)
            .map_err(|e| ParseError::SchemaValidation(e.to_string()))?;

        Ok(Self {
            schema: Some(validator),
            pretty_print: false,
        })
    }

    /// Enable pretty printing in format instructions.
    pub fn with_pretty_print(mut self, pretty: bool) -> Self {
        self.pretty_print = pretty;
        self
    }
}

impl Default for JsonOutputParser {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl OutputParser for JsonOutputParser {
    type Output = Value;

    async fn parse(&self, text: &str) -> Result<Self::Output, ParseError> {
        let text = caliche_contracts::text::extract_json(text).unwrap_or(text);

        let value: Value = serde_json::from_str(text)
            .map_err(|e| ParseError::ParsingFailed(format!("Invalid JSON: {}", e)))?;

        // Validate against schema if provided
        #[cfg(feature = "output-parsing")]
        if let Some(validator) = &self.schema
            && !validator.is_valid(&value)
        {
            let error_msgs: Vec<String> = validator
                .iter_errors(&value)
                .map(|e| e.to_string())
                .collect();
            return Err(ParseError::SchemaValidation(error_msgs.join(", ")));
        }

        Ok(value)
    }

    fn get_format_instructions(&self) -> String {
        let mut instructions = String::from("Your response should be valid JSON");

        #[cfg(feature = "output-parsing")]
        if let Some(_schema) = &self.schema {
            instructions.push_str(" that conforms to the provided schema");
        }

        if self.pretty_print {
            instructions.push_str(", formatted with proper indentation");
        }

        instructions.push('.');
        instructions
    }

    fn name(&self) -> &str {
        "json"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[tokio::test]
    async fn fence_without_a_newline_is_a_parse_error() {
        let parser = JsonOutputParser::new();
        for text in ["```jsoné```", "```json"] {
            assert_eq!(
                parser.parse(text).await.map_err(|e| e.to_string()),
                Err(
                    "Failed to parse output: Invalid JSON: expected value at line 1 column 1"
                        .to_string()
                ),
                "{text}"
            );
        }
    }

    #[tokio::test]
    async fn fence_after_text_parses() {
        let parser = JsonOutputParser::new();
        let text = "Here it is:\n```json\n{\"a\": 1}\n```\nDone.";
        assert_eq!(parser.parse(text).await.ok(), Some(json!({"a": 1})));
    }

    #[tokio::test]
    async fn test_json_parser_basic() {
        let parser = JsonOutputParser::new();
        let result = parser
            .parse(r#"{"name": "Alice", "age": 30}"#)
            .await
            .expect("Should parse valid JSON");

        assert_eq!(result["name"], "Alice");
        assert_eq!(result["age"], 30);
    }

    #[tokio::test]
    async fn test_json_parser_markdown() {
        let parser = JsonOutputParser::new();
        let text = r#"```json
{
  "name": "Bob",
  "age": 25
}
```"#;

        let result = parser
            .parse(text)
            .await
            .expect("Should parse JSON from markdown");
        assert_eq!(result["name"], "Bob");
    }

    #[tokio::test]
    async fn test_json_parser_invalid() {
        let parser = JsonOutputParser::new();
        let result = parser.parse("not json").await;
        assert!(result.is_err());
    }

    #[cfg(feature = "output-parsing")]
    #[tokio::test]
    async fn test_json_parser_with_schema() {
        let schema = json!({
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "number"}
            },
            "required": ["name"]
        });

        let parser =
            JsonOutputParser::with_schema(schema).expect("Should create parser with schema");

        // Valid
        let result = parser.parse(r#"{"name": "Alice", "age": 30}"#).await;
        assert!(result.is_ok());

        // Invalid - missing required field
        let result = parser.parse(r#"{"age": 30}"#).await;
        assert!(result.is_err());
    }
}
