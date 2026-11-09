//! Test scenario engine for declarative testing

use crate::agent::builder::AgentBuilder;
use crate::agent::service::sandbox::AgentSandbox;
use anyhow::Result;
use serde_json::Value;

/// Test scenario for declarative agent testing
pub struct TestScenario {
    tools: Vec<String>,
    expected_tool_calls: Vec<ExpectedToolCall>,
    expected_response_contains: Vec<String>,
    expected_response_matches: Option<String>, // Regex pattern
    sandbox: AgentSandbox,
}

/// Expected tool call
pub struct ExpectedToolCall {
    pub tool_name: String,
    pub expected_arguments: Value,
}

impl TestScenario {
    pub fn new() -> Self {
        Self {
            tools: Vec::new(),
            expected_tool_calls: Vec::new(),
            expected_response_contains: Vec::new(),
            expected_response_matches: None,
            sandbox: AgentSandbox::new(),
        }
    }

    pub fn with_tool(mut self, tool: &str) -> Self {
        self.tools.push(tool.to_string());
        self
    }

    pub fn expect_tool_call(mut self, tool_name: &str, expected_arguments: Value) -> Self {
        self.expected_tool_calls.push(ExpectedToolCall {
            tool_name: tool_name.to_string(),
            expected_arguments,
        });
        self
    }

    pub fn expect_response_contains(mut self, text: &str) -> Self {
        self.expected_response_contains.push(text.to_string());
        self
    }

    pub fn expect_response_matches(mut self, pattern: &str) -> Self {
        self.expected_response_matches = Some(pattern.to_string());
        self
    }

    pub fn with_sandbox(mut self, sandbox: AgentSandbox) -> Self {
        self.sandbox = sandbox;
        self
    }

    pub fn build(self) -> Self {
        self
    }

    pub async fn create_agent(&self) -> Result<crate::AgentService> {
        let mut builder = AgentBuilder::new();
        for tool in &self.tools {
            builder = builder.enable_tool(tool);
        }
        Box::pin(builder.build()).await
    }

    pub fn assert_expectations(&self, response: &str) -> Result<()> {
        // Check response contains expected text
        for expected in &self.expected_response_contains {
            if !response.contains(expected) {
                return Err(anyhow::anyhow!(
                    "Response does not contain expected text: {}",
                    expected
                ));
            }
        }

        // Check regex pattern if provided
        if let Some(ref pattern) = self.expected_response_matches {
            use regex::Regex;
            let re = Regex::new(pattern)?;
            if !re.is_match(response) {
                return Err(anyhow::anyhow!(
                    "Response does not match pattern: {}",
                    pattern
                ));
            }
        }

        Ok(())
    }
}

impl Default for TestScenario {
    fn default() -> Self {
        Self::new()
    }
}
