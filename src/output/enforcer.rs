//! Schema enforcement with automatic retry for structured outputs.
//!
//! This module provides validation and retry logic for ensuring LLM responses
//! conform to specified JSON schemas.

use crate::llm::Provider;
use crate::messages::{CompletionRequest, CompletionResponse, Message};
#[cfg(feature = "jsonschema")]
use crate::output::schema::ResponseFormat;
use crate::output::schema::StructuredOutputConfig;
#[cfg(feature = "jsonschema")]
use anyhow::bail;
use anyhow::{Context, Result};
#[cfg(feature = "jsonschema")]
use jsonschema::Validator;

/// Enforces structured output schemas with automatic retry on validation failure
pub struct StructuredOutputEnforcer {
    config: StructuredOutputConfig,
    #[cfg(feature = "jsonschema")]
    compiled_schema: Option<Validator>,
}

impl StructuredOutputEnforcer {
    /// Create a new enforcer with the given configuration
    pub fn new(config: StructuredOutputConfig) -> anyhow::Result<Self> {
        #[cfg(feature = "jsonschema")]
        let compiled_schema = if let ResponseFormat::JsonSchema { ref json_schema } = config.format
        {
            Some(
                jsonschema::validator_for(&json_schema.schema)
                    .map_err(|e| anyhow::anyhow!("Invalid JSON schema: {}", e))?,
            )
        } else {
            None
        };

        Ok(Self {
            config,
            #[cfg(feature = "jsonschema")]
            compiled_schema,
        })
    }

    /// Execute request with schema enforcement and automatic retry
    pub async fn complete_with_schema(
        &self,
        provider: &dyn Provider,
        mut request: CompletionRequest,
    ) -> anyhow::Result<CompletionResponse> {
        // Set response format on request
        request.response_format = Some(self.config.format.clone());

        let mut attempts = 0;
        let mut last_error: Option<String> = None;

        while attempts <= self.config.max_retries {
            let response = provider.complete(request.clone()).await?;

            // Validate response against schema
            match self.validate_response(&response) {
                Ok(()) => return Ok(response),
                Err(validation_error) => {
                    attempts += 1;
                    let validation_msg = validation_error.to_string();
                    last_error = Some(validation_msg.clone());

                    if attempts <= self.config.max_retries {
                        // Add retry message with error context
                        let retry_message = self
                            .config
                            .retry_prompt
                            .clone()
                            .unwrap_or_else(|| {
                                format!(
                                    "Your response did not match the required schema. Error: {validation_msg}. Please try again."
                                )
                            });
                        request.messages = Arc::new({
                            let mut msgs = (*request.messages).clone();
                            msgs.push(Message::user(retry_message));
                            msgs
                        });
                    }
                }
            }
        }

        Err(anyhow::anyhow!(
            "Failed to get valid structured output after {} attempts. Last error: {:?}",
            self.config.max_retries + 1,
            last_error
        ))
    }

    fn validate_response(&self, response: &CompletionResponse) -> Result<()> {
        let content = response.content.as_text();

        // Parse JSON
        let json: serde_json::Value = serde_json::from_str(content).context("Invalid JSON")?;
        #[cfg(not(feature = "jsonschema"))]
        let _ = &json;

        // Validate against schema if we have one
        #[cfg(feature = "jsonschema")]
        {
            if let Some(ref validator) = self.compiled_schema
                && !validator.is_valid(&json)
            {
                let error_msgs: Vec<String> = validator
                    .iter_errors(&json)
                    .map(|e| e.to_string())
                    .collect();
                bail!("{}", error_msgs.join("; "));
            }
        }

        Ok(())
    }
}

use std::sync::Arc;
