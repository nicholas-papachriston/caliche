//! Chain builder for fluent API construction of chains

use super::chain::{Chain, ChainError, ChainResult};
use super::llm_chain::LLMChain;
use crate::agent::service::AgentService;
use crate::prompts::PromptTemplate;
use serde_json::Value;
use std::sync::Arc;
use tokio::sync::Mutex;

/// Builder for creating chains with a fluent API
///
/// # Examples
///
/// ```rust,no_run
/// use caliche::{AgentService, ChainBuilder};
///
/// # async fn example() -> anyhow::Result<()> {
/// let service = AgentService::new().await?;
///
/// // Simple chain with auto-detected single variable
/// let result = ChainBuilder::new(service)
///     .with_template("Translate to French: {text}")
///     .run_with("Hello, world!")
///     .await?;
///
/// println!("Result: {}", result.output["output"]);
/// # Ok(())
/// # }
/// ```
pub struct ChainBuilder {
    agent: Arc<Mutex<AgentService>>,
    template: Option<String>,
    variables: Option<Vec<String>>,
    output_key: Option<String>,
}

impl ChainBuilder {
    /// Create a new chain builder with an agent service
    pub fn new(agent: AgentService) -> Self {
        Self {
            agent: Arc::new(Mutex::new(agent)),
            template: None,
            variables: None,
            output_key: None,
        }
    }

    /// Create a new chain builder from an Arc<Mutex<AgentService>>
    pub fn from_arc(agent: Arc<Mutex<AgentService>>) -> Self {
        Self {
            agent,
            template: None,
            variables: None,
            output_key: None,
        }
    }

    /// Set the prompt template
    ///
    /// The template can use {variable_name} syntax for variables.
    pub fn with_template(mut self, template: impl Into<String>) -> Self {
        self.template = Some(template.into());
        self
    }

    /// Explicitly set template variables (optional - auto-detected by default)
    pub fn with_variables(mut self, variables: Vec<String>) -> Self {
        self.variables = Some(variables);
        self
    }

    /// Set the output key name (defaults to "output")
    pub fn with_output_key(mut self, key: impl Into<String>) -> Self {
        self.output_key = Some(key.into());
        self
    }

    /// Build the chain
    pub fn build(self) -> Result<Box<dyn Chain>, ChainError> {
        let template_str = self
            .template
            .ok_or_else(|| ChainError::ExecutionFailed("Template is required".to_string()))?;

        // PromptTemplate automatically extracts variables from the template
        let prompt = PromptTemplate::new(template_str);

        let mut chain = LLMChain::new(self.agent, prompt);

        if let Some(key) = self.output_key {
            chain = chain.with_output_key(key);
        }

        Ok(Box::new(chain))
    }

    /// Build and run the chain with a single string input
    ///
    /// This is a convenience method for simple templates with a single variable.
    /// It will automatically detect the variable name and create the input JSON.
    pub async fn run_with(self, input: impl Into<String>) -> Result<ChainResult, ChainError> {
        let template_str = self
            .template
            .as_ref()
            .ok_or_else(|| ChainError::ExecutionFailed("Template is required".to_string()))?;

        let variables = extract_variables(template_str);

        if variables.len() != 1 {
            return Err(ChainError::ExecutionFailed(format!(
                "run_with() requires exactly one variable in template, found {}",
                variables.len()
            )));
        }

        let var_name = &variables[0];
        let input_json = serde_json::json!({ var_name: input.into() });

        let chain = self.build()?;
        chain.run(&input_json).await
    }

    /// Build and run the chain with JSON input
    pub async fn run(self, input: &Value) -> Result<ChainResult, ChainError> {
        let chain = self.build()?;
        chain.run(input).await
    }
}

/// Extract variables from a template string
///
/// Looks for {variable_name} patterns
fn extract_variables(template: &str) -> Vec<String> {
    let mut variables = Vec::new();
    let mut chars = template.chars().peekable();
    let mut current_var = String::new();
    let mut in_var = false;

    while let Some(ch) = chars.next() {
        if ch == '{' && chars.peek().is_some_and(|&next| next != '{') {
            in_var = true;
            current_var.clear();
        } else if ch == '}' && in_var {
            if !current_var.is_empty() && !variables.contains(&current_var) {
                variables.push(current_var.clone());
            }
            in_var = false;
        } else if in_var {
            current_var.push(ch);
        }
    }

    variables
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_variables() {
        let template = "Hello {name}, how are you {time}?";
        let vars = extract_variables(template);
        assert_eq!(vars, vec!["name", "time"]);
    }

    #[test]
    fn test_extract_variables_single() {
        let template = "Translate: {text}";
        let vars = extract_variables(template);
        assert_eq!(vars, vec!["text"]);
    }

    #[test]
    fn test_extract_variables_none() {
        let template = "No variables here";
        let vars = extract_variables(template);
        assert!(vars.is_empty());
    }

    #[test]
    fn test_extract_variables_duplicate() {
        let template = "Hello {name}, goodbye {name}";
        let vars = extract_variables(template);
        assert_eq!(vars, vec!["name"]); // Should only appear once
    }
}
