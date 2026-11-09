//! Plan-and-Execute reasoning pattern.
//!
//! This pattern separates planning from execution:
//! 1. **Plan**: Create a step-by-step plan to solve the problem
//! 2. **Execute**: Execute each step of the plan sequentially
//! 3. **Synthesize**: Generate final answer from working memory

use async_trait::async_trait;

use crate::AgentService;
use crate::messages::{Message, MessageContent, MessageRole, TokenUsage};
use crate::reasoning::{ReasoningPattern, ReasoningResult, ReasoningStep, StepType};
use crate::utils::parsing::parse_json;
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::sync::Arc;

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Plan {
    steps: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct PlanResponse {
    plan: Plan,
}

/// Plan-and-Execute reasoning pattern
pub struct PlanExecutePattern {
    agent: Arc<tokio::sync::Mutex<AgentService>>,
    max_steps: usize,
}

impl std::fmt::Debug for PlanExecutePattern {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PlanExecutePattern")
            .field("agent", &"Arc<Mutex<AgentService>>")
            .field("max_steps", &self.max_steps)
            .finish()
    }
}

impl PlanExecutePattern {
    pub fn new(agent: AgentService, max_steps: usize) -> Self {
        Self {
            agent: Arc::new(tokio::sync::Mutex::new(agent)),
            max_steps,
        }
    }

    /// Parse plan steps from LLM response
    fn parse_plan_steps(plan_text: &str) -> Vec<String> {
        // Try to parse as structured JSON first
        if let Ok(plan_resp) = parse_json::<PlanResponse>(plan_text) {
            return plan_resp.plan.steps;
        }
        if let Ok(plan) = parse_json::<Plan>(plan_text) {
            return plan.steps;
        }

        // Fallback: extract numbered steps from text
        let mut steps = Vec::new();
        for line in plan_text.lines() {
            let line = line.trim();
            // Look for numbered steps: "1. ...", "Step 1: ...", etc.
            if let Some(step_start) = line.find(|c: char| c.is_ascii_digit()) {
                if let Some(colon_pos) = line[step_start..].find(':') {
                    let step_content = line[step_start + colon_pos + 1..].trim();
                    if !step_content.is_empty() {
                        steps.push(step_content.to_string());
                    }
                } else if let Some(dot_pos) = line[step_start..].find('.') {
                    let step_content = line[step_start + dot_pos + 1..].trim();
                    if !step_content.is_empty() {
                        steps.push(step_content.to_string());
                    }
                }
            }
        }

        // If we found steps, return them; otherwise return the whole text as one step
        if steps.is_empty() {
            vec![plan_text.to_string()]
        } else {
            steps
        }
    }
}

fn build_synthesis_prompt(
    query: &str,
    working_memory: &[String],
    planned_steps: usize,
    truncated: bool,
) -> String {
    let mut prompt = String::with_capacity(
        220 + query.len() + working_memory.iter().map(String::len).sum::<usize>(),
    );
    prompt.push_str("Based on the execution results, provide a final answer to: ");
    prompt.push_str(query);
    prompt.push_str("\n\nGive the complete final answer now. Do not describe remaining steps or say you will continue.");
    prompt.push_str("\nIf the answer is a number, output the number.\n");
    if truncated {
        prompt.push_str("\nThe plan had ");
        prompt.push_str(&planned_steps.to_string());
        prompt.push_str(" steps; only the first ");
        prompt.push_str(&working_memory.len().to_string());
        prompt.push_str(" were executed. Finish from the results you have.\n");
    }
    prompt.push_str("\nExecution results:\n");
    if working_memory.is_empty() {
        prompt.push_str("None");
        return prompt;
    }
    for (index, result) in working_memory.iter().enumerate() {
        if index > 0 {
            prompt.push_str("\n\n");
        }
        prompt.push_str("Step ");
        prompt.push_str(&(index + 1).to_string());
        prompt.push_str(": ");
        prompt.push_str(result);
    }
    prompt
}

#[async_trait]
impl ReasoningPattern for PlanExecutePattern {
    async fn execute(&self, query: &str) -> anyhow::Result<ReasoningResult> {
        let mut reasoning_steps = Vec::new();
        let mut total_usage = TokenUsage::default();

        // Step 1: Create plan
        use std::fmt::Write;
        let mut plan_prompt = String::with_capacity(200 + query.len());
        write!(
            plan_prompt,
            r#"Create a step-by-step plan to answer: {}

Return the plan as JSON in this format:
{{
  "plan": {{
    "steps": [
      "Step 1 description",
      "Step 2 description",
      ...
    ]
  }}
}}

Or provide numbered steps in plain text (1. Step one, 2. Step two, etc.)."#,
            query
        )
        .map_err(|e| anyhow::anyhow!("Failed to write plan prompt: {}", e))?;

        let plan_chat_response = self.agent.lock().await.chat(&plan_prompt).await?;
        total_usage = total_usage + *plan_chat_response.usage();

        reasoning_steps.push(ReasoningStep {
            step_type: StepType::Plan,
            content: plan_chat_response.text().to_string(),
            metadata: Some(json!({"planning": true})),
        });

        // Parse plan into individual steps
        let plan_steps = Self::parse_plan_steps(plan_chat_response.text());

        if plan_steps.is_empty() {
            return Ok(ReasoningResult::failure(
                "Failed to parse plan steps".to_owned(),
                reasoning_steps,
            )
            .with_usage(total_usage));
        }

        // Step 2: Execute each plan step sequentially
        let mut working_memory = Vec::new();
        let mut message_history = Arc::new(vec![Message {
            role: MessageRole::System,
            content: MessageContent::text(format!(
                "You are executing a plan to answer: {}\n\nPlan:\n{}",
                query,
                plan_steps
                    .iter()
                    .enumerate()
                    .map(|(i, s)| format!("{}. {}", i + 1, s))
                    .collect::<Vec<_>>()
                    .join("\n")
            )),
            metadata: None,
        }]);

        for (step_idx, step_description) in plan_steps.iter().enumerate() {
            if step_idx >= self.max_steps {
                break;
            }

            // Build prompt for this step — write directly into the buffer to avoid
            // an intermediate Vec<String> from collect+join.
            let mut step_prompt = String::with_capacity(
                60 + step_description.len()
                    + working_memory
                        .iter()
                        .map(|s: &String| s.len())
                        .sum::<usize>(),
            );
            write!(
                step_prompt,
                "Execute step {}: {step_description}\n\nPrevious results:\n",
                step_idx + 1
            )
            .map_err(|e| anyhow::anyhow!("Failed to write step prompt: {e}"))?;
            if working_memory.is_empty() {
                step_prompt.push_str("None");
            } else {
                for (i, r) in working_memory.iter().enumerate() {
                    if i > 0 {
                        step_prompt.push('\n');
                    }
                    write!(step_prompt, "Step {} result: {r}", i + 1)
                        .map_err(|e| anyhow::anyhow!("Failed to write step result: {e}"))?;
                }
            }

            let step_messages = {
                let mut msgs = (*message_history).clone();
                msgs.push(Message {
                    role: MessageRole::User,
                    content: MessageContent::text(step_prompt),
                    metadata: None,
                });
                Arc::new(msgs)
            };

            // Execute step using step() method
            let step_result = {
                let agent_service = self.agent.lock().await;
                agent_service.step(step_messages.clone()).await?
            };

            total_usage = total_usage + step_result.usage;
            message_history = step_result.updated_history.clone();

            // Store result in working memory - dereference Arc for owned String
            working_memory.push((*step_result.response_text).clone());

            reasoning_steps.push(ReasoningStep {
                step_type: StepType::Action,
                content: format!(
                    "Step {}: {}\nResult: {}",
                    step_idx + 1,
                    step_description,
                    step_result.response_text
                ),
                metadata: Some(json!({"step": step_idx + 1, "description": step_description})),
            });
        }

        // Step 3: Synthesize final answer from working memory on a fresh
        // conversation. Reusing the execute-step history caused models to
        // continue the plan ("I'll now complete step 3") instead of answering.
        let truncated = working_memory.len() < plan_steps.len();
        let synthesis_prompt =
            build_synthesis_prompt(query, &working_memory, plan_steps.len(), truncated);
        let synthesis_history = vec![Message::system(
            "You synthesize a complete final answer from completed plan steps. \
             Output the answer itself. Do not describe remaining work or next steps. \
             If the answer is a number, output only that number.",
        )];

        let synthesis_response = {
            let mut agent_service = self.agent.lock().await;
            agent_service
                .chat_with_history(&synthesis_prompt, synthesis_history)
                .await?
        };

        total_usage = total_usage + *synthesis_response.usage();
        let final_answer = synthesis_response.text().to_string();

        reasoning_steps.push(ReasoningStep {
            step_type: StepType::Answer,
            content: final_answer.clone(),
            metadata: Some(json!({"synthesis": true})),
        });

        Ok(ReasoningResult::success(final_answer, reasoning_steps).with_usage(total_usage))
    }

    fn name(&self) -> &str {
        "Plan-and-Execute"
    }

    fn description(&self) -> &str {
        "Creates a plan first, then executes each step sequentially"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "testing")]
    use crate::Agent as CoreAgent;
    #[cfg(feature = "testing")]
    use crate::AgentService;
    #[cfg(feature = "testing")]
    use crate::agent::tools::{CalculatorTool, ToolRegistry};
    #[cfg(feature = "testing")]
    use crate::messages::{CompletionContent, CompletionResponse, TokenUsage};
    #[cfg(feature = "testing")]
    use crate::prelude::ProviderType;
    #[cfg(feature = "testing")]
    use crate::testing::MockProvider;
    #[cfg(feature = "testing")]
    use std::sync::Arc;

    #[test]
    fn parse_plan_steps_reads_json_and_numbered_lists() {
        let json_steps = PlanExecutePattern::parse_plan_steps(
            r#"{"plan":{"steps":["Multiply 17 and 19","Report the product"]}}"#,
        );
        assert_eq!(
            json_steps,
            vec![
                "Multiply 17 and 19".to_string(),
                "Report the product".to_string()
            ]
        );

        let numbered = PlanExecutePattern::parse_plan_steps(
            "1. Multiply 17 and 19\n2. Add nothing\n3. Report 323",
        );
        assert_eq!(numbered.len(), 3);
        assert_eq!(numbered[2], "Report 323");
    }

    #[test]
    fn synthesis_prompt_asks_for_the_answer_when_steps_are_truncated() {
        let prompt = build_synthesis_prompt(
            "What is 17*19?",
            &["17".to_string(), "19".to_string()],
            3,
            true,
        );
        assert!(prompt.contains("What is 17*19?"));
        assert!(prompt.contains("complete final answer"));
        assert!(prompt.contains("Do not describe remaining steps"));
        assert!(prompt.contains("If the answer is a number, output the number"));
        assert!(prompt.contains("The plan had 3 steps; only the first 2 were executed"));
        assert!(prompt.contains("Step 1: 17"));
        assert!(prompt.contains("Step 2: 19"));
    }

    #[cfg(feature = "testing")]
    #[tokio::test]
    async fn truncated_plan_execute_synthesizes_the_numeric_answer() {
        let plan = CompletionResponse {
            content: CompletionContent::Text {
                content: r#"{"plan":{"steps":["Note 17","Note 19","Multiply them"]}}"#.to_string(),
            },
            finish_reason: Some(beef::lean::Cow::borrowed("end_turn")),
            usage: Some(TokenUsage::default()),
            reasoning_content: None,
        };
        let step1 = CompletionResponse {
            content: CompletionContent::Text {
                content: "17".to_string(),
            },
            finish_reason: Some(beef::lean::Cow::borrowed("end_turn")),
            usage: Some(TokenUsage::default()),
            reasoning_content: None,
        };
        let step2 = CompletionResponse {
            content: CompletionContent::Text {
                content: "19".to_string(),
            },
            finish_reason: Some(beef::lean::Cow::borrowed("end_turn")),
            usage: Some(TokenUsage::default()),
            reasoning_content: None,
        };
        let synthesis = CompletionResponse {
            content: CompletionContent::Text {
                content: "323".to_string(),
            },
            finish_reason: Some(beef::lean::Cow::borrowed("end_turn")),
            usage: Some(TokenUsage::default()),
            reasoning_content: None,
        };

        let provider = MockProvider::with_responses(vec![plan, step1, step2, synthesis]);
        let tools = ToolRegistry::new();
        tools.register(Arc::new(CalculatorTool));
        let agent_core = CoreAgent::new(Arc::new(provider), Arc::new(tools));
        let agent = AgentService::new_with_agent(agent_core, ProviderType::OpenAI)
            .expect("failed to create agent service");
        let planner = PlanExecutePattern::new(agent, 2);

        let result = planner
            .execute("What is 17*19?")
            .await
            .expect("plan-execute should succeed");

        assert!(result.success);
        assert_eq!(result.final_answer.trim(), "323");
        assert!(
            !result
                .final_answer
                .to_ascii_lowercase()
                .contains("i'll now"),
            "synthesis must not continue leftover plan steps: {}",
            result.final_answer
        );
    }
}
