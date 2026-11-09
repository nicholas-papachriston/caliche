//! Core output parser trait and error types.

use async_trait::async_trait;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum ParseError {
    #[error("Failed to parse output: {0}")]
    ParsingFailed(String),

    #[error("Invalid format: {0}")]
    InvalidFormat(String),

    #[error("Schema validation failed: {0}")]
    SchemaValidation(String),

    #[error("Deserialization error: {0}")]
    DeserializationError(String),

    #[error("Missing required field: {0}")]
    MissingField(String),
}

/// Trait for parsing LLM outputs into structured data.
///
/// Output parsers can extract structured information from LLM responses,
/// validate against schemas, and retry on failures.
#[async_trait]
pub trait OutputParser: Send + Sync {
    /// The type this parser produces.
    type Output;

    /// Parse the LLM output into the desired type.
    async fn parse(&self, text: &str) -> Result<Self::Output, ParseError>;

    /// Get instructions to include in the prompt.
    ///
    /// These instructions help the LLM generate output in the correct format.
    fn get_format_instructions(&self) -> String;

    /// Get the name of this parser.
    fn name(&self) -> &str {
        "output_parser"
    }
}

/// A simple output parser that returns the text as-is.
#[derive(Debug, Clone)]
pub struct PassthroughParser;

#[async_trait]
impl OutputParser for PassthroughParser {
    type Output = String;

    async fn parse(&self, text: &str) -> Result<Self::Output, ParseError> {
        Ok(text.to_string())
    }

    fn get_format_instructions(&self) -> String {
        String::new()
    }

    fn name(&self) -> &str {
        "passthrough"
    }
}

/// Parser for comma-separated lists.
#[derive(Debug, Clone)]
pub struct ListOutputParser {
    separator: String,
    trim_items: bool,
}

impl ListOutputParser {
    pub fn new() -> Self {
        Self {
            separator: ",".to_string(),
            trim_items: true,
        }
    }

    pub fn with_separator(mut self, separator: impl Into<String>) -> Self {
        self.separator = separator.into();
        self
    }

    pub fn with_trim(mut self, trim: bool) -> Self {
        self.trim_items = trim;
        self
    }
}

impl Default for ListOutputParser {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl OutputParser for ListOutputParser {
    type Output = Vec<String>;

    async fn parse(&self, text: &str) -> Result<Self::Output, ParseError> {
        let items: Vec<String> = text
            .split(&self.separator)
            .map(|s| if self.trim_items { s.trim() } else { s })
            .filter(|s| !s.is_empty())
            .map(String::from)
            .collect();

        Ok(items)
    }

    fn get_format_instructions(&self) -> String {
        format!(
            "Your response should be a list of items separated by '{}'",
            self.separator
        )
    }

    fn name(&self) -> &str {
        "list"
    }
}

/// Parser for enum/choice outputs.
#[derive(Debug, Clone)]
pub struct EnumOutputParser {
    choices: Vec<String>,
    case_sensitive: bool,
}

impl EnumOutputParser {
    pub fn new(choices: Vec<String>) -> Self {
        Self {
            choices,
            case_sensitive: false,
        }
    }

    pub fn with_case_sensitive(mut self, sensitive: bool) -> Self {
        self.case_sensitive = sensitive;
        self
    }
}

#[async_trait]
impl OutputParser for EnumOutputParser {
    type Output = String;

    async fn parse(&self, text: &str) -> Result<Self::Output, ParseError> {
        let text = text.trim();

        for choice in &self.choices {
            if self.case_sensitive {
                if text == choice {
                    return Ok(choice.clone());
                }
            } else if text.eq_ignore_ascii_case(choice) {
                return Ok(choice.clone());
            }
        }

        Err(ParseError::InvalidFormat(format!(
            "Output '{}' is not one of the allowed choices: {:?}",
            text, self.choices
        )))
    }

    fn get_format_instructions(&self) -> String {
        format!(
            "Your response must be exactly one of the following choices: {}",
            self.choices.join(", ")
        )
    }

    fn name(&self) -> &str {
        "enum"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_passthrough_parser() {
        let parser = PassthroughParser;
        let result = parser
            .parse("hello world")
            .await
            .expect("Should parse passthrough");
        assert_eq!(result, "hello world");
    }

    #[tokio::test]
    async fn test_list_parser() {
        let parser = ListOutputParser::new();
        let result = parser
            .parse("apple, banana, cherry")
            .await
            .expect("Should parse list");
        assert_eq!(result, vec!["apple", "banana", "cherry"]);
    }

    #[tokio::test]
    async fn test_list_parser_custom_separator() {
        let parser = ListOutputParser::new().with_separator("|");
        let result = parser
            .parse("apple|banana|cherry")
            .await
            .expect("Should parse list with custom separator");
        assert_eq!(result, vec!["apple", "banana", "cherry"]);
    }

    #[tokio::test]
    async fn test_enum_parser() {
        let parser = EnumOutputParser::new(vec![
            "positive".to_string(),
            "negative".to_string(),
            "neutral".to_string(),
        ]);

        let result = parser
            .parse("positive")
            .await
            .expect("Should parse enum value");
        assert_eq!(result, "positive");

        let result = parser
            .parse("NEGATIVE")
            .await
            .expect("Should parse enum value (case insensitive)");
        assert_eq!(result, "negative");

        let err = parser.parse("invalid").await;
        assert!(err.is_err());
    }
}
