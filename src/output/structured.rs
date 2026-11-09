//! Structured output parser for typed deserialization.

use super::parser::{OutputParser, ParseError};
use async_trait::async_trait;
use serde::de::DeserializeOwned;
use serde_json::Value;
use std::marker::PhantomData;

/// Parser for typed structured outputs.
///
/// Deserializes JSON output into a specific Rust type.
///
/// # Examples
///
/// ```rust
/// use caliche::output::StructuredOutputParser;
/// use serde::Deserialize;
///
/// #[derive(Deserialize)]
/// struct Person {
///     name: String,
///     age: u32,
/// }
///
/// let parser = StructuredOutputParser::<Person>::new();
/// ```
#[derive(Debug)]
pub struct StructuredOutputParser<T> {
    _phantom: PhantomData<T>,
    include_instructions: bool,
    example: Option<String>,
}

impl<T> StructuredOutputParser<T>
where
    T: DeserializeOwned + Send + Sync,
{
    /// Create a new structured output parser.
    pub fn new() -> Self {
        Self {
            _phantom: PhantomData,
            include_instructions: true,
            example: None,
        }
    }

    /// Set whether to include format instructions in the prompt.
    pub fn with_instructions(mut self, include: bool) -> Self {
        self.include_instructions = include;
        self
    }

    /// Set an example JSON output to show in instructions.
    pub fn with_example(mut self, example: impl Into<String>) -> Self {
        self.example = Some(example.into());
        self
    }
}

impl<T> Default for StructuredOutputParser<T>
where
    T: DeserializeOwned + Send + Sync,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<T> Clone for StructuredOutputParser<T> {
    fn clone(&self) -> Self {
        Self {
            _phantom: PhantomData,
            include_instructions: self.include_instructions,
            example: self.example.clone(),
        }
    }
}

#[async_trait]
impl<T> OutputParser for StructuredOutputParser<T>
where
    T: DeserializeOwned + Send + Sync,
{
    type Output = T;

    async fn parse(&self, text: &str) -> Result<Self::Output, ParseError> {
        // Extract JSON from markdown if present
        let text = text.trim();
        let json_text = caliche_contracts::text::extract_json(text).unwrap_or(text);

        // First parse as JSON Value
        let value: Value = serde_json::from_str(json_text)
            .map_err(|e| ParseError::ParsingFailed(format!("Invalid JSON: {}", e)))?;

        // Then deserialize into target type
        serde_json::from_value(value).map_err(|e| ParseError::DeserializationError(e.to_string()))
    }

    fn get_format_instructions(&self) -> String {
        if !self.include_instructions {
            return String::new();
        }

        let mut instructions = String::from(
            "Your response should be valid JSON that can be parsed into the required structure.",
        );

        if let Some(example) = &self.example {
            instructions.push_str("\n\nExample:\n");
            instructions.push_str(example);
        }

        instructions
    }

    fn name(&self) -> &str {
        "structured"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::Deserialize;

    #[derive(Debug, Deserialize, PartialEq)]
    struct Person {
        name: String,
        age: u32,
    }

    #[tokio::test]
    async fn test_structured_parser() {
        let parser = StructuredOutputParser::<Person>::new();

        let json = r#"{"name": "Alice", "age": 30}"#;
        let result = parser
            .parse(json)
            .await
            .expect("Should parse structured output");

        assert_eq!(result.name, "Alice");
        assert_eq!(result.age, 30);
    }

    #[tokio::test]
    async fn test_structured_parser_markdown() {
        let parser = StructuredOutputParser::<Person>::new();

        let json = r#"```json
{"name": "Bob", "age": 25}
```"#;

        let result = parser
            .parse(json)
            .await
            .expect("Should parse structured output");
        assert_eq!(result.name, "Bob");
    }

    /// The closing `]` comes before the opening `{`.
    #[tokio::test]
    async fn closer_before_opener_is_a_parse_error() {
        let parser = StructuredOutputParser::<Value>::new();
        let error = parser
            .parse("```] then {")
            .await
            .err()
            .map(|e| e.to_string());
        assert_eq!(
            error.as_deref(),
            Some("Failed to parse output: Invalid JSON: expected value at line 1 column 1")
        );
    }

    /// The array opens first, so the array is the JSON, not the object inside it.
    #[tokio::test]
    async fn fenced_array_of_objects_parses_as_array() {
        let parser = StructuredOutputParser::<Value>::new();
        let value = parser
            .parse("```json\n[{\"a\": 1}]\n```")
            .await
            .map_err(|e| e.to_string());
        assert_eq!(value, Ok(serde_json::json!([{"a": 1}])));
    }

    #[tokio::test]
    async fn fence_after_text_parses() {
        let parser = StructuredOutputParser::<Value>::new();
        let value = parser
            .parse("Here it is:\n```json\n{\"a\": 1}\n```\nDone.")
            .await
            .map_err(|e| e.to_string());
        assert_eq!(value, Ok(serde_json::json!({"a": 1})));
    }

    #[tokio::test]
    async fn test_structured_parser_missing_field() {
        let parser = StructuredOutputParser::<Person>::new();

        let json = r#"{"name": "Charlie"}"#;
        let result = parser.parse(json).await;
        assert!(result.is_err());
    }

    #[derive(Debug, Deserialize)]
    struct OptionalAge {
        name: String,
        age: Option<u32>,
    }

    #[tokio::test]
    async fn test_structured_parser_optional() {
        let parser = StructuredOutputParser::<OptionalAge>::new();

        let json = r#"{"name": "David"}"#;
        let result = parser
            .parse(json)
            .await
            .expect("Should parse structured output");

        assert_eq!(result.name, "David");
        assert_eq!(result.age, None);
    }
}
