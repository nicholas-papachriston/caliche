//! LLM chain - basic chain combining prompt template, LLM call, and output parser.

use super::chain::{Chain, ChainError, ChainResult};
use crate::agent::service::AgentService;
use crate::prompts::PromptTemplate;
use async_trait::async_trait;
use serde_json::Value;
use std::sync::Arc;
use tokio::sync::Mutex;

/// Basic LLM chain that combines a prompt template with an LLM call.
///
/// This is the foundational chain pattern that formats a prompt and calls the LLM.
pub struct LLMChain {
    agent: Arc<Mutex<AgentService>>,
    prompt: PromptTemplate,
    output_key: String,
}

impl std::fmt::Debug for LLMChain {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LLMChain")
            .field("agent", &"Arc<Mutex<AgentService>>")
            .field("prompt", &self.prompt)
            .field("output_key", &self.output_key)
            .finish()
    }
}

impl LLMChain {
    /// Create a new LLM chain.
    pub fn new(agent: Arc<Mutex<AgentService>>, prompt: PromptTemplate) -> Self {
        Self {
            agent,
            prompt,
            output_key: "output".to_owned(),
        }
    }

    /// Set the output key name.
    pub fn with_output_key(mut self, key: impl Into<String>) -> Self {
        self.output_key = key.into();
        self
    }
}

#[async_trait]
impl Chain for LLMChain {
    async fn run(&self, input: &Value) -> Result<ChainResult, ChainError> {
        // Format the prompt with input variables
        let prompt_text = self
            .prompt
            .format(input)
            .map_err(|e| ChainError::ExecutionFailed(format!("Prompt formatting failed: {}", e)))?;

        // Call the LLM
        let mut agent = self.agent.lock().await;
        let chat_response = agent
            .chat(&prompt_text)
            .await
            .map_err(ChainError::AgentError)?;

        // Return the result (use reference to avoid clone)
        let output = serde_json::json!({
            &self.output_key: chat_response.text()
        });

        Ok(ChainResult::new(output))
    }

    fn name(&self) -> &str {
        "llm_chain"
    }

    fn input_keys(&self) -> Vec<String> {
        self.prompt.variables().iter().cloned().collect()
    }

    fn output_keys(&self) -> Vec<String> {
        vec![self.output_key.clone()]
    }
}
