//! Reflexion reasoning pattern.
//!
//! This pattern implements self-reflection and error correction.

use async_trait::async_trait;

use crate::AgentService;
use crate::messages::TokenUsage;
use crate::reasoning::{ReasoningPattern, ReasoningResult, ReasoningStep, StepType};
use crate::utils::parsing::parse_status_response;
use serde_json::json;
use std::fmt::Write;
use std::sync::Arc;

/// Reflexion reasoning pattern
pub struct ReflexionPattern {
    agent: Arc<tokio::sync::Mutex<AgentService>>,
    max_iterations: usize,
}

impl std::fmt::Debug for ReflexionPattern {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ReflexionPattern")
            .field("agent", &"Arc<Mutex<AgentService>>")
            .field("max_iterations", &self.max_iterations)
            .finish()
    }
}

impl ReflexionPattern {
    pub fn new(agent: AgentService, max_iterations: usize) -> Self {
        Self {
            agent: Arc::new(tokio::sync::Mutex::new(agent)),
            max_iterations,
        }
    }
}

#[async_trait]
impl ReasoningPattern for ReflexionPattern {
    async fn execute(&self, query: &str) -> anyhow::Result<ReasoningResult> {
        let mut steps = Vec::new();
        let mut total_usage = TokenUsage::default();
        let mut current_solution = String::new();

        for iteration in 0..self.max_iterations {
            // Attempt solution with pre-allocated buffer
            let solve_prompt = if iteration == 0 {
                let mut prompt = String::with_capacity(30 + query.len());
                write!(prompt, "Solve this problem:\n\n{}", query)
                    .map_err(|e| anyhow::anyhow!("Failed to write solve prompt: {}", e))?;
                prompt
            } else {
                let mut prompt = String::with_capacity(60 + query.len() + current_solution.len());
                write!(
                    prompt,
                    "Improve your previous solution:\n\nProblem: {}\n\nPrevious attempt: {}",
                    query, current_solution
                )
                .map_err(|e| anyhow::anyhow!("Failed to write solve prompt: {}", e))?;
                prompt
            };

            let solution_chat_response = self.agent.lock().await.chat(&solve_prompt).await?;
            total_usage = total_usage + *solution_chat_response.usage();
            current_solution = solution_chat_response.text().to_string();

            steps.push(ReasoningStep {
                step_type: StepType::Action,
                content: current_solution.clone(),
                metadata: Some(json!({"iteration": iteration, "attempt": true})),
            });

            // Reflect with pre-allocated buffer
            let mut reflect_prompt =
                String::with_capacity(60 + query.len() + current_solution.len());
            write!(
                reflect_prompt,
                "Reflect on this solution. Is it correct?\n\nProblem: {}\n\nSolution: {}",
                query, current_solution
            )
            .map_err(|e| anyhow::anyhow!("Failed to write reflect prompt: {}", e))?;
            let reflection_chat_response = self.agent.lock().await.chat(&reflect_prompt).await?;
            total_usage = total_usage + *reflection_chat_response.usage();
            let reflection = reflection_chat_response.text();

            // Use structured parsing to determine correctness
            let status_response = parse_status_response(reflection).unwrap_or_else(|_| {
                // Fallback to simple text parsing if structured parsing fails
                let text_lower = reflection.to_lowercase();
                crate::utils::parsing::StatusResponse {
                    status: if text_lower.contains("correct") || text_lower.contains("yes") {
                        "correct".to_owned()
                    } else {
                        "incorrect".to_owned()
                    },
                    message: Some(reflection.to_string()),
                    is_correct: Some(text_lower.contains("correct") || text_lower.contains("yes")),
                }
            });
            let is_correct = status_response.is_success();

            steps.push(ReasoningStep {
                step_type: StepType::Reflection,
                content: reflection.to_string(),
                metadata: Some(json!({"iteration": iteration, "is_correct": is_correct})),
            });

            if is_correct {
                break;
            }
        }

        let answer = current_solution.clone();
        steps.push(ReasoningStep {
            step_type: StepType::Answer,
            content: current_solution,
            metadata: Some(json!({"final": true})),
        });

        Ok(ReasoningResult::success(answer, steps).with_usage(total_usage))
    }

    fn name(&self) -> &str {
        "Reflexion"
    }

    fn description(&self) -> &str {
        "Self-reflects on solutions and iteratively improves them"
    }
}
