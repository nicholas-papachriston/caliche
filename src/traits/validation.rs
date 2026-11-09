//! Validation strategy abstraction for input and output validation.
//!
//! This module provides a unified interface for validating inputs and outputs
//! at various stages of the agent pipeline. Implement the `InputValidator` and
//! `OutputValidator` traits to add custom validation logic.

use crate::messages::CompletionResponse;
use anyhow::Result;
use async_trait::async_trait;
use std::sync::Arc;

/// Validation result
#[derive(Debug, Clone)]
pub struct ValidationResult {
    /// Whether validation passed
    pub is_valid: bool,
    /// Error message if validation failed
    pub error: Option<String>,
}

impl ValidationResult {
    /// Create a valid result
    pub fn valid() -> Self {
        Self {
            is_valid: true,
            error: None,
        }
    }

    /// Create an invalid result with error message
    pub fn invalid(error: impl Into<String>) -> Self {
        Self {
            is_valid: false,
            error: Some(error.into()),
        }
    }
}

/// Trait for validating input strings
#[async_trait]
pub trait InputValidator: Send + Sync + std::fmt::Debug {
    /// Validate an input string
    ///
    /// # Arguments
    /// * `input` - Input string to validate
    ///
    /// # Returns
    /// * `Ok(ValidationResult)` - Validation result
    /// * `Err` - Error during validation
    async fn validate_input(&self, input: &str) -> Result<ValidationResult>;

    /// Get the name of this validator (for debugging/logging)
    fn validator_name(&self) -> &str;
}

/// Trait for validating output responses
#[async_trait]
pub trait OutputValidator: Send + Sync + std::fmt::Debug {
    /// Validate an output response
    ///
    /// # Arguments
    /// * `output` - Output response to validate
    ///
    /// # Returns
    /// * `Ok(ValidationResult)` - Validation result
    /// * `Err` - Error during validation
    async fn validate_output(&self, output: &CompletionResponse) -> Result<ValidationResult>;

    /// Get the name of this validator (for debugging/logging)
    fn validator_name(&self) -> &str;
}

/// Input validator that checks length constraints
#[derive(Debug, Clone)]
pub struct LengthValidator {
    min_length: Option<usize>,
    max_length: Option<usize>,
}

impl LengthValidator {
    /// Create a new length validator
    ///
    /// # Arguments
    /// * `min_length` - Minimum allowed length (None means no minimum)
    /// * `max_length` - Maximum allowed length (None means no maximum)
    pub fn new(min_length: Option<usize>, max_length: Option<usize>) -> Self {
        Self {
            min_length,
            max_length,
        }
    }

    /// Create a validator that only checks maximum length
    pub fn max_length(max_length: usize) -> Self {
        Self::new(None, Some(max_length))
    }

    /// Create a validator that only checks minimum length
    pub fn min_length(min_length: usize) -> Self {
        Self::new(Some(min_length), None)
    }
}

#[async_trait]
impl InputValidator for LengthValidator {
    async fn validate_input(&self, input: &str) -> Result<ValidationResult> {
        let len = input.len();

        if let Some(min) = self.min_length
            && len < min
        {
            return Ok(ValidationResult::invalid(format!(
                "Input length {} is less than minimum {}",
                len, min
            )));
        }

        if let Some(max) = self.max_length
            && len > max
        {
            return Ok(ValidationResult::invalid(format!(
                "Input length {} exceeds maximum {}",
                len, max
            )));
        }

        Ok(ValidationResult::valid())
    }

    fn validator_name(&self) -> &str {
        "length"
    }
}

/// Input validator that checks for unsafe content
#[derive(Debug, Clone)]
pub struct ContentSafetyValidator {
    blocked_patterns: Vec<String>,
    case_sensitive: bool,
}

impl ContentSafetyValidator {
    /// Create a new content safety validator
    ///
    /// # Arguments
    /// * `blocked_patterns` - List of patterns to block
    /// * `case_sensitive` - Whether pattern matching is case sensitive
    pub fn new(blocked_patterns: Vec<String>, case_sensitive: bool) -> Self {
        Self {
            blocked_patterns,
            case_sensitive,
        }
    }

    /// Create a validator with common unsafe patterns
    pub fn default_unsafe_patterns() -> Self {
        Self::new(
            vec![
                "password".to_string(),
                "secret".to_string(),
                "api_key".to_string(),
                "token".to_string(),
            ],
            false,
        )
    }
}

#[async_trait]
impl InputValidator for ContentSafetyValidator {
    async fn validate_input(&self, input: &str) -> Result<ValidationResult> {
        let check_input = if self.case_sensitive {
            input.to_string()
        } else {
            input.to_lowercase()
        };

        for pattern in &self.blocked_patterns {
            let check_pattern = if self.case_sensitive {
                pattern.clone()
            } else {
                pattern.to_lowercase()
            };

            if check_input.contains(&check_pattern) {
                return Ok(ValidationResult::invalid(format!(
                    "Input contains blocked pattern: {}",
                    pattern
                )));
            }
        }

        Ok(ValidationResult::valid())
    }

    fn validator_name(&self) -> &str {
        "content-safety"
    }
}

/// Input validator that validates JSON structure against a schema
#[derive(Debug, Clone)]
pub struct JsonSchemaValidator {
    schema: serde_json::Value,
}

impl JsonSchemaValidator {
    /// Create a new JSON schema validator
    ///
    /// # Arguments
    /// * `schema` - JSON schema to validate against
    pub fn new(schema: serde_json::Value) -> Self {
        Self { schema }
    }
}

#[async_trait]
impl InputValidator for JsonSchemaValidator {
    async fn validate_input(&self, input: &str) -> Result<ValidationResult> {
        let value: serde_json::Value = match serde_json::from_str(input) {
            Ok(v) => v,
            Err(e) => return Ok(ValidationResult::invalid(format!("Invalid JSON: {e}"))),
        };

        match crate::utils::json_schema_lite::validate_basic_schema(&value, &self.schema) {
            Ok(()) => Ok(ValidationResult::valid()),
            Err(e) => Ok(ValidationResult::invalid(e.to_string())),
        }
    }

    fn validator_name(&self) -> &str {
        "json-schema"
    }
}

/// Input validator that chains multiple validators
#[derive(Debug)]
pub struct ChainedInputValidator {
    validators: Vec<Arc<dyn InputValidator>>,
}

impl ChainedInputValidator {
    /// Create a new chained input validator
    pub fn new(validators: Vec<Arc<dyn InputValidator>>) -> Self {
        Self { validators }
    }
}

#[async_trait]
impl InputValidator for ChainedInputValidator {
    async fn validate_input(&self, input: &str) -> Result<ValidationResult> {
        for validator in &self.validators {
            let result = validator.validate_input(input).await?;
            if !result.is_valid {
                return Ok(result);
            }
        }
        Ok(ValidationResult::valid())
    }

    fn validator_name(&self) -> &str {
        "chained-input"
    }
}

/// Output validator that chains multiple validators
#[derive(Debug)]
pub struct ChainedOutputValidator {
    validators: Vec<Arc<dyn OutputValidator>>,
}

impl ChainedOutputValidator {
    /// Create a new chained output validator
    pub fn new(validators: Vec<Arc<dyn OutputValidator>>) -> Self {
        Self { validators }
    }
}

#[async_trait]
impl OutputValidator for ChainedOutputValidator {
    async fn validate_output(&self, output: &CompletionResponse) -> Result<ValidationResult> {
        for validator in &self.validators {
            let result = validator.validate_output(output).await?;
            if !result.is_valid {
                return Ok(result);
            }
        }
        Ok(ValidationResult::valid())
    }

    fn validator_name(&self) -> &str {
        "chained-output"
    }
}

/// Output validator that checks response content length
#[derive(Debug, Clone)]
pub struct OutputLengthValidator {
    max_length: Option<usize>,
}

impl OutputLengthValidator {
    /// Create a new output length validator
    ///
    /// # Arguments
    /// * `max_length` - Maximum allowed content length
    pub fn new(max_length: Option<usize>) -> Self {
        Self { max_length }
    }
}

#[async_trait]
impl OutputValidator for OutputLengthValidator {
    async fn validate_output(&self, output: &CompletionResponse) -> Result<ValidationResult> {
        let content_length = output.content.as_text().len();

        if let Some(max) = self.max_length
            && content_length > max
        {
            return Ok(ValidationResult::invalid(format!(
                "Output length {} exceeds maximum {}",
                content_length, max
            )));
        }

        Ok(ValidationResult::valid())
    }

    fn validator_name(&self) -> &str {
        "output-length"
    }
}

/// No-op input validator that always passes
#[derive(Debug, Clone)]
pub struct NoOpInputValidator;

impl NoOpInputValidator {
    /// Create a new no-op input validator
    pub fn new() -> Self {
        Self
    }
}

impl Default for NoOpInputValidator {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl InputValidator for NoOpInputValidator {
    async fn validate_input(&self, _input: &str) -> Result<ValidationResult> {
        Ok(ValidationResult::valid())
    }

    fn validator_name(&self) -> &str {
        "no-op-input"
    }
}

/// No-op output validator that always passes
#[derive(Debug, Clone)]
pub struct NoOpOutputValidator;

impl NoOpOutputValidator {
    /// Create a new no-op output validator
    pub fn new() -> Self {
        Self
    }
}

impl Default for NoOpOutputValidator {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl OutputValidator for NoOpOutputValidator {
    async fn validate_output(&self, _output: &CompletionResponse) -> Result<ValidationResult> {
        Ok(ValidationResult::valid())
    }

    fn validator_name(&self) -> &str {
        "no-op-output"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_length_validator() {
        let validator = LengthValidator::new(Some(5), Some(10));

        assert!(
            validator
                .validate_input("short")
                .await
                .expect("validation should succeed")
                .is_valid
        );
        assert!(
            validator
                .validate_input("toolongtext")
                .await
                .expect("validation should succeed")
                .is_valid
                == false
        );
        assert!(
            validator
                .validate_input("hi")
                .await
                .expect("validation should succeed")
                .is_valid
                == false
        );
    }

    #[tokio::test]
    async fn test_content_safety_validator() {
        let validator = ContentSafetyValidator::default_unsafe_patterns();

        assert!(
            validator
                .validate_input("hello world")
                .await
                .expect("validation should succeed")
                .is_valid
        );
        assert!(
            validator
                .validate_input("my password is secret")
                .await
                .expect("validation should succeed")
                .is_valid
                == false
        );
    }

    #[tokio::test]
    async fn test_chained_validator() {
        let length_validator = Arc::new(LengthValidator::new(Some(5), Some(10)));
        let safety_validator = Arc::new(ContentSafetyValidator::default_unsafe_patterns());

        let chain = ChainedInputValidator::new(vec![length_validator, safety_validator]);

        // "hello" is 5 characters, within the 5-10 range
        assert!(
            chain
                .validate_input("hello")
                .await
                .expect("validation should succeed")
                .is_valid
        );
        // "hello world" is 11 characters, exceeds max of 10
        assert!(
            chain
                .validate_input("hello world")
                .await
                .expect("validation should succeed")
                .is_valid
                == false
        );
        // "hi" is 2 characters, below min of 5
        assert!(
            chain
                .validate_input("hi")
                .await
                .expect("validation should succeed")
                .is_valid
                == false
        );
    }

    #[tokio::test]
    async fn test_no_op_validator() {
        let validator = NoOpInputValidator::new();
        assert!(
            validator
                .validate_input("anything")
                .await
                .expect("validation should succeed")
                .is_valid
        );
    }
}
