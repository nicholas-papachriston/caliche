//! Self-Ask reasoning pattern.
//!
//! This pattern recursively decomposes complex questions into simpler sub-questions.

use async_trait::async_trait;

use crate::AgentService;
use crate::messages::TokenUsage;
use crate::reasoning::{ReasoningPattern, ReasoningResult, ReasoningStep, StepType};
use serde_json::json;
use std::fmt::Write;
use std::sync::Arc;

/// Self-Ask reasoning pattern
pub struct SelfAskPattern {
    agent: Arc<tokio::sync::Mutex<AgentService>>,
    _max_depth: usize,
}

impl std::fmt::Debug for SelfAskPattern {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SelfAskPattern")
            .field("agent", &"Arc<Mutex<AgentService>>")
            .field("_max_depth", &self._max_depth)
            .finish()
    }
}

impl SelfAskPattern {
    pub fn new(agent: AgentService, max_depth: usize) -> Self {
        Self {
            agent: Arc::new(tokio::sync::Mutex::new(agent)),
            _max_depth: max_depth,
        }
    }
}

#[async_trait]
impl ReasoningPattern for SelfAskPattern {
    async fn execute(&self, query: &str) -> anyhow::Result<ReasoningResult> {
        let mut steps = Vec::new();
        let mut total_usage = TokenUsage::default();

        // Decompose question with pre-allocated buffer
        let mut decompose_prompt = String::with_capacity(60 + query.len());
        write!(
            decompose_prompt,
            "Break down this question into simpler sub-questions:\n\n{}",
            query
        )
        .map_err(|e| anyhow::anyhow!("Failed to write decompose prompt: {}", e))?;
        let sub_questions_chat_response = self.agent.lock().await.chat(&decompose_prompt).await?;
        total_usage = total_usage + *sub_questions_chat_response.usage();
        let sub_questions = sub_questions_chat_response.text().to_string();

        steps.push(ReasoningStep {
            step_type: StepType::Thought,
            content: sub_questions.clone(),
            metadata: Some(json!({"decomposition": true})),
        });

        // Answer with context - pre-allocate buffer
        let mut answer_prompt = String::with_capacity(80 + query.len() + sub_questions.len());
        write!(answer_prompt,
            "Answer this question, using the sub-questions as guidance:\n\nQuestion: {}\n\nSub-questions: {}",
            query, sub_questions
        ).map_err(|e| anyhow::anyhow!("Failed to write answer prompt: {}", e))?;
        let answer_chat_response = self.agent.lock().await.chat(&answer_prompt).await?;
        total_usage = total_usage + *answer_chat_response.usage();

        let final_answer = answer_chat_response.text().to_string();
        steps.push(ReasoningStep {
            step_type: StepType::Answer,
            content: final_answer.clone(),
            metadata: Some(json!({"synthesis": true})),
        });

        Ok(ReasoningResult::success(final_answer, steps).with_usage(total_usage))
    }

    fn name(&self) -> &str {
        "Self-Ask"
    }

    fn description(&self) -> &str {
        "Recursively decomposes questions into simpler sub-questions"
    }
}
