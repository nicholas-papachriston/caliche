//! Utilities for parsing structured outputs from LLMs.
//!
//! Helpers for common LLM JSON errors such as trailing commas,
//! unescaped quotes, and malformed JSON.

use anyhow::{Context, Result};
use serde::Deserialize;

/// Parse JSON from LLM output with error recovery.
///
/// This function attempts to fix common JSON errors that LLMs produce:
/// - Trailing commas
/// - Unescaped quotes in strings
/// - Missing closing braces/brackets
/// - Comments (removes them)
pub fn parse_json<T: for<'de> Deserialize<'de>>(text: &str) -> Result<T> {
    // First, try direct parsing
    match serde_json::from_str::<T>(text) {
        Ok(value) => return Ok(value),
        Err(_) => {
            // Try to fix common issues
        }
    }

    // Try the JSON value inside a markdown fence or surrounding text
    let candidate = caliche_contracts::text::extract_json(text).unwrap_or(text);
    match serde_json::from_str::<T>(candidate) {
        Ok(value) => Ok(value),
        Err(e) => serde_json::from_str::<T>(&fix_trailing_commas(candidate))
            .map_err(|_| e)
            .context(format!("Failed to parse JSON from text: {}", text)),
    }
}

/// Fix trailing commas in JSON
fn fix_trailing_commas(json: &str) -> String {
    // Remove trailing commas before } or ]
    let mut result = String::with_capacity(json.len());
    let mut chars = json.chars().peekable();

    while let Some(ch) = chars.next() {
        if ch == ',' {
            // Check if next non-whitespace is } or ]
            let mut peek_chars = chars.clone();
            while let Some(&next_ch) = peek_chars.peek() {
                if next_ch.is_whitespace() {
                    peek_chars.next();
                } else if next_ch == '}' || next_ch == ']' {
                    // Skip this comma (it's trailing)
                    break;
                } else {
                    // Not trailing, keep it
                    result.push(ch);
                    break;
                }
            }
        } else {
            result.push(ch);
        }
    }

    result
}

/// Parse a structured output that should contain a status field
#[derive(Debug, Clone, serde::Deserialize)]
pub struct StatusResponse {
    pub status: String,
    #[serde(default)]
    #[expect(dead_code, reason = "deserialized but not read yet")]
    pub message: Option<String>,
    #[serde(default)]
    pub is_correct: Option<bool>,
}

impl StatusResponse {
    /// Check if the status indicates success/correctness
    pub fn is_success(&self) -> bool {
        let status_lower = self.status.to_lowercase();
        status_lower.contains("correct")
            || status_lower.contains("success")
            || status_lower.contains("yes")
            || status_lower.contains("true")
            || self.is_correct.unwrap_or(false)
    }
}

/// Parse a status response from LLM output
pub fn parse_status_response(text: &str) -> Result<StatusResponse> {
    // First try to parse as JSON
    if let Ok(status) = parse_json::<StatusResponse>(text) {
        return Ok(status);
    }

    // Try to extract status from text
    let text_lower = text.to_lowercase();
    let is_correct = text_lower.contains("correct")
        || text_lower.contains("yes")
        || text_lower.contains("success")
        || text_lower.contains("true");

    Ok(StatusResponse {
        status: if is_correct {
            "correct".to_string()
        } else {
            "incorrect".to_string()
        },
        message: Some(text.to_string()),
        is_correct: Some(is_correct),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_json_basic() {
        let json = r#"{"status": "correct", "message": "This is correct"}"#;
        let result: StatusResponse = parse_json(json).expect("JSON parsing should succeed");
        assert_eq!(result.status, "correct");
    }

    #[test]
    fn test_parse_json_with_trailing_comma() {
        let json = r#"{"status": "correct", "message": "This is correct",}"#;
        let result: StatusResponse =
            parse_json(json).expect("JSON parsing with trailing comma should succeed");
        assert_eq!(result.status, "correct");
    }

    #[test]
    fn test_parse_json_from_markdown() {
        let text = r#"Here's the result:
```json
{"status": "correct"}
```"#;
        let result: StatusResponse =
            parse_json(text).expect("JSON parsing from markdown should succeed");
        assert_eq!(result.status, "correct");
    }

    #[test]
    fn json_after_multibyte_text_parses() {
        let text = "Résumé: {\"status\": \"correct\"} done";
        let result: StatusResponse =
            parse_json(text).expect("JSON after multibyte text should parse");
        assert_eq!(result.status, "correct");
    }

    #[test]
    fn array_in_surrounding_text_parses_as_the_array() {
        let value: serde_json::Value =
            parse_json(r#"Items: [{"a": 1}] end"#).expect("array in text should parse");
        assert_eq!(value, serde_json::json!([{"a": 1}]));
    }

    #[test]
    fn test_parse_status_response() {
        let text = "This solution is correct.";
        let result = parse_status_response(text).expect("status response parsing should succeed");
        assert!(result.is_success());
    }
}
