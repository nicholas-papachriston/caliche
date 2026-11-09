//! Reasoning patterns for complex agent behavior.
//!
//! This module provides various reasoning patterns that allow agents to handle
//! complex multi-step tasks through structured thought processes:
//!
//! - **ReAct**: Interleaves reasoning (Thought) and acting (Action/Observation)
//! - **Plan-and-Execute**: Separates planning from execution for complex workflows
//! - **Self-Ask**: Decomposes questions into sub-questions recursively
//! - **Reflexion**: Self-critique and error correction through reflection
//! - **Tree of Thoughts**: Explores multiple reasoning paths in parallel
//!
//! # Examples
//!
//! ```rust,no_run
//! use caliche::{AgentBuilder, ProviderType};
//! use caliche::reasoning::{react::ReActPattern, ReasoningPattern};
//!
//! # async fn example() -> anyhow::Result<()> {
//! let agent = AgentBuilder::new()
//!     .provider(ProviderType::Bedrock)
//!     .build()
//!     .await?;
//!
//! let react = ReActPattern::new(agent, 5); // Max 5 iterations
//!
//! let result = react.execute(
//!     "What is the current temperature in San Francisco and how does it compare to New York?"
//! ).await?;
//!
//! println!("Final answer: {}", result.final_answer);
//! println!("Steps taken: {}", result.steps.len());
//! # Ok(())
//! # }
//! ```

pub mod plan_execute;
pub mod react;
pub mod reflexion;
pub mod self_ask;
pub mod tree_of_thoughts;

use async_trait::async_trait;

use crate::messages::TokenUsage;
use serde::{Deserialize, Serialize};

/// A step in a reasoning process
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReasoningStep {
    /// Type of step: thought, action, observation, plan, reflection, etc.
    pub step_type: StepType,
    /// Content of the step
    pub content: String,
    /// Optional metadata (tool used, error info, etc.)
    pub metadata: Option<serde_json::Value>,
}

/// Types of reasoning steps
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum StepType {
    /// A thought or reasoning step
    Thought,
    /// An action taken (tool call)
    Action,
    /// Observation from action result
    Observation,
    /// A plan or sub-goal
    Plan,
    /// Self-reflection or critique
    Reflection,
    /// Final answer
    Answer,
}

/// Result of a reasoning process
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReasoningResult {
    /// The final answer
    pub final_answer: String,
    /// All reasoning steps taken
    pub steps: Vec<ReasoningStep>,
    /// Whether the reasoning succeeded
    pub success: bool,
    /// Optional error message if reasoning failed
    pub error: Option<String>,
    /// Token usage across all steps
    pub token_usage: Option<TokenUsage>,
}

impl ReasoningResult {
    pub fn success(answer: impl Into<String>, steps: Vec<ReasoningStep>) -> Self {
        Self {
            final_answer: answer.into(),
            steps,
            success: true,
            error: None,
            token_usage: None,
        }
    }

    pub fn failure(error: impl Into<String>, steps: Vec<ReasoningStep>) -> Self {
        Self {
            final_answer: String::new(),
            steps,
            success: false,
            error: Some(error.into()),
            token_usage: None,
        }
    }

    pub fn with_usage(mut self, usage: TokenUsage) -> Self {
        self.token_usage = Some(usage);
        self
    }
}

/// Trait for reasoning patterns
#[async_trait]
pub trait ReasoningPattern: Send + Sync + std::fmt::Debug {
    /// Execute the reasoning pattern on a query
    async fn execute(&self, query: &str) -> anyhow::Result<ReasoningResult>;

    /// Get the name of this reasoning pattern
    fn name(&self) -> &str;

    /// Get a description of this reasoning pattern
    fn description(&self) -> &str;
}
