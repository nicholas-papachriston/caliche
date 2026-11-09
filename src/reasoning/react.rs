//! ReAct (Reasoning + Acting) pattern implementation.
//!
//! ReAct interleaves reasoning (thinking about what to do) with acting
//! (using tools to gather information). This pattern is effective for:
//! - Multi-step question answering
//! - Information gathering tasks
//! - Tasks requiring tool usage
//!
//! # Pattern Flow
//!
//! 1. **Thought**: Agent thinks about what to do next
//! 2. **Action**: Agent decides to use a tool
//! 3. **Observation**: Agent observes the tool's result
//! 4. **Repeat** until the agent has enough information
//! 5. **Answer**: Agent provides final answer
//!
//! # Example
//!
//! ```rust,no_run
//! use caliche::{AgentBuilder, ProviderType};
//! use caliche::reasoning::{react::ReActPattern, ReasoningPattern, StepType};
//!
//! # async fn example() -> anyhow::Result<()> {
//! let agent = AgentBuilder::new()
//!     .provider(ProviderType::Bedrock)
//!     .build()
//!     .await?;
//!
//! let react = ReActPattern::new(agent, 5); // Max 5 thought-action cycles
//!
//! let result = react.execute(
//!     "Search for information about Rust programming and summarize the key features"
//! ).await?;
//!
//! for step in result.steps {
//!     match step.step_type {
//!         StepType::Thought => println!("[THOUGHT] {}", step.content),
//!         StepType::Action => println!("[ACTION] {}", step.content),
//!         StepType::Observation => println!("[OBSERVATION] {}", step.content),
//!         StepType::Answer => println!("[ANSWER] {}", step.content),
//!         _ => {}
//!     }
//! }
//! # Ok(())
//! # }
//! ```

use async_trait::async_trait;

use crate::AgentService;
use crate::messages::{Message, MessageContent, MessageRole, TokenUsage};
use crate::reasoning::{ReasoningPattern, ReasoningResult, ReasoningStep, StepType};
use serde_json::json;
use std::sync::Arc;

/// ReAct reasoning pattern
pub struct ReActPattern {
    agent: Arc<tokio::sync::Mutex<AgentService>>,
    max_iterations: usize,
}

impl std::fmt::Debug for ReActPattern {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ReActPattern")
            .field("agent", &"Arc<Mutex<AgentService>>")
            .field("max_iterations", &self.max_iterations)
            .finish()
    }
}

impl ReActPattern {
    /// Create a new ReAct pattern with a maximum number of thought-action iterations
    pub fn new(agent: AgentService, max_iterations: usize) -> Self {
        Self {
            agent: Arc::new(tokio::sync::Mutex::new(agent)),
            max_iterations,
        }
    }

    /// Create a new ReAct pattern from an Arc<tokio::sync::Mutex<AgentService>>
    /// This is useful for bindings and when you already have the agent wrapped in a Mutex
    pub fn from_arc(agent: Arc<tokio::sync::Mutex<AgentService>>, max_iterations: usize) -> Self {
        Self {
            agent,
            max_iterations,
        }
    }

    /// Parse the agent's response into thought, action, or answer
    fn parse_response(&self, response: &str) -> ParsedResponse {
        let response_lower = response.to_ascii_lowercase();

        // Look for "final answer:" first (more specific), then "answer:"
        if let Some(answer_start) = response_lower.find("final answer:") {
            let answer = response[answer_start + 13..].trim().to_string();
            if !answer.is_empty() {
                return ParsedResponse::Answer(answer);
            }
        }

        // Check for "answer:" pattern (but not as part of "final answer:")
        if let Some(answer_start) = response_lower.find("answer:") {
            // Make sure it's not part of "final answer:"
            if answer_start == 0
                || !response_lower.as_bytes()[answer_start.saturating_sub(6)..answer_start]
                    .windows(5)
                    .any(|window| window == b"final")
            {
                let answer = response[answer_start + 7..].trim().to_string();
                if !answer.is_empty() {
                    return ParsedResponse::Answer(answer);
                }
            }
        }

        // Look for "thought:" pattern
        if response_lower.contains("thought:")
            && let Some(thought_start) = response_lower.find("thought:")
        {
            let thought = response[thought_start + 8..].trim();
            // Extract just the thought part (until action or end)
            let thought_end = thought
                .to_ascii_lowercase()
                .find("action:")
                .unwrap_or(thought.len());
            let thought_text = thought[..thought_end].trim().to_string();

            return ParsedResponse::Thought(thought_text);
        }

        // If no explicit markers, treat as a thought
        ParsedResponse::Thought(response.trim().to_string())
    }

    /// Build the initial ReAct system prompt
    fn build_system_prompt(&self) -> String {
        r#"You are an AI assistant using the ReAct (Reasoning + Acting) framework to answer questions.

Use this format:
Thought: [Your reasoning about what to do next]
Action: [The action you want to take - use available tools]
Observation: [Result from the action]
... (Repeat Thought/Action/Observation as needed)
Thought: I now have enough information to answer
Answer: [Your final answer to the original question]

IMPORTANT INSTRUCTIONS:
- Tools are automatically called by the system when you need them. Simply think about what information you need, and the system will execute the appropriate tools.
- When you have gathered enough information to answer the question, you MUST provide your final answer using the format "Answer: [your answer]" or "Final Answer: [your answer]".
- Always end with a clear "Answer:" statement - do not leave the question unanswered.
- Think step-by-step and use tools when you need information."#.to_string()
    }

    /// Build message history from steps for the step() method
    async fn build_message_history(
        &self,
        query: &str,
        steps: &[ReasoningStep],
    ) -> anyhow::Result<Vec<Message>> {
        // Build messages with system prompt
        let mut messages = vec![Message {
            role: MessageRole::System,
            content: MessageContent::text(self.build_system_prompt()),
            metadata: None,
        }];

        // Add tools info if available
        // We'll build the tools description inside the lock scope
        {
            let agent = self.agent.lock().await;
            let tools_list = agent.list_tools();
            if !tools_list.is_empty() {
                let tools_desc: String = tools_list
                    .iter()
                    .map(|(name, desc, params)| {
                        format!(
                            "- {}: {}\n  Parameters: {}",
                            name,
                            desc,
                            serde_json::to_string_pretty(params).unwrap_or_default()
                        )
                    })
                    .collect::<Vec<_>>()
                    .join("\n");
                messages.push(Message {
                    role: MessageRole::System,
                    content: MessageContent::text(format!("Available tools:\n{}", tools_desc)),
                    metadata: None,
                });
            }
        }

        messages.push(Message {
            role: MessageRole::User,
            content: MessageContent::text(format!("Question: {}", query)),
            metadata: None,
        });

        // Add step history as assistant/user messages
        for step in steps {
            match step.step_type {
                StepType::Thought => {
                    messages.push(Message {
                        role: MessageRole::Assistant,
                        content: MessageContent::text(format!("Thought: {}", step.content)),
                        metadata: None,
                    });
                }
                StepType::Action => {
                    // Action is typically embedded in the assistant message with tool calls
                    // We'll add it as part of the thought/action sequence
                    if let Some(last_msg) = messages.last_mut()
                        && let MessageContent::Text(arc_text) = &mut last_msg.content
                    {
                        std::sync::Arc::make_mut(arc_text)
                            .push_str(&format!("\nAction: {}", step.content));
                    }
                }
                StepType::Observation => {
                    messages.push(Message {
                        role: MessageRole::User,
                        content: MessageContent::text(format!("Observation: {}", step.content)),
                        metadata: None,
                    });
                }
                _ => {}
            }
        }

        if !steps.is_empty() {
            messages.push(Message {
                role: MessageRole::User,
                content: MessageContent::text(
                    "Continue reasoning. If you have enough information, provide your final answer using 'Answer:' or 'Final Answer:'.".to_string()
                ),
                metadata: None,
            });
        }

        Ok(messages)
    }
}

#[async_trait]
impl ReasoningPattern for ReActPattern {
    async fn execute(&self, query: &str) -> anyhow::Result<ReasoningResult> {
        let mut steps = Vec::new();
        let mut total_usage = TokenUsage::default();
        let mut message_history = Arc::new(self.build_message_history(query, &steps).await?);

        for iteration in 0..self.max_iterations {
            // Execute one step using the step() method
            let step_result = {
                let agent_service = self.agent.lock().await;
                agent_service.step(message_history.clone()).await?
            };

            // Track token usage
            total_usage = total_usage + step_result.usage;

            // Update message history for next iteration
            message_history = step_result.updated_history.clone();

            // Extract thought from response
            let response_text = step_result.response_text.clone();
            let parsed = self.parse_response(&response_text);

            match parsed {
                ParsedResponse::Thought(thought) => {
                    steps.push(ReasoningStep {
                        step_type: StepType::Thought,
                        content: thought,
                        metadata: Some(json!({"iteration": iteration})),
                    });

                    // Record tool calls as Actions
                    if !step_result.tool_calls.is_empty() {
                        for tool_call in &step_result.tool_calls {
                            steps.push(ReasoningStep {
                                step_type: StepType::Action,
                                content: format!(
                                    "{} with arguments: {}",
                                    tool_call.name, tool_call.arguments
                                ),
                                metadata: Some(
                                    json!({"tool": tool_call.name, "iteration": iteration}),
                                ),
                            });
                        }
                    }

                    // Record tool results as Observations
                    for (tool_name, tool_result) in &step_result.tool_results {
                        let observation_content = if tool_result.success {
                            format!(
                                "Tool {} executed successfully. Result: {}",
                                tool_name, tool_result.content
                            )
                        } else {
                            format!(
                                "Tool {} execution failed. Error: {}",
                                tool_name,
                                tool_result.error.as_deref().unwrap_or("Unknown error")
                            )
                        };
                        steps.push(ReasoningStep {
                            step_type: StepType::Observation,
                            content: observation_content,
                            metadata: Some(json!({"tool": tool_name, "iteration": iteration})),
                        });
                    }

                    // If there are no pending tool calls and we have a thought, check if it contains an answer
                    if !step_result.has_pending_tool_calls {
                        // Re-parse to check if the response actually contains an answer
                        let re_parsed = self.parse_response(&response_text);
                        if let ParsedResponse::Answer(answer) = re_parsed {
                            steps.push(ReasoningStep {
                                step_type: StepType::Answer,
                                content: answer.clone(),
                                metadata: Some(json!({"iteration": iteration})),
                            });
                            return Ok(
                                ReasoningResult::success(answer, steps).with_usage(total_usage)
                            );
                        }
                    }
                }
                ParsedResponse::Answer(answer) => {
                    // Found final answer
                    steps.push(ReasoningStep {
                        step_type: StepType::Answer,
                        content: answer.clone(),
                        metadata: Some(json!({"iteration": iteration})),
                    });

                    return Ok(ReasoningResult::success(answer, steps).with_usage(total_usage));
                }
            }

            // If no more tool calls pending and we didn't get an answer, check if we should continue
            if !step_result.has_pending_tool_calls {
                // Check if the response suggests we're done
                let response_lower = response_text.to_lowercase();
                if response_lower.contains("final answer") || response_lower.contains("answer:") {
                    // Try to extract answer one more time
                    let final_parsed = self.parse_response(&response_text);
                    if let ParsedResponse::Answer(answer) = final_parsed {
                        steps.push(ReasoningStep {
                            step_type: StepType::Answer,
                            content: answer.clone(),
                            metadata: Some(json!({"iteration": iteration})),
                        });
                        return Ok(ReasoningResult::success(answer, steps).with_usage(total_usage));
                    }
                }
            }
        }

        // Max iterations reached without finding answer
        let error_msg = format!(
            "ReAct pattern reached maximum iterations ({}) without finding an answer",
            self.max_iterations
        );

        Ok(ReasoningResult::failure(error_msg, steps).with_usage(total_usage))
    }

    fn name(&self) -> &str {
        "ReAct"
    }

    fn description(&self) -> &str {
        "Reasoning + Acting pattern that interleaves thinking and tool usage"
    }
}

/// Parsed response from the agent
enum ParsedResponse {
    Thought(String),
    Answer(String),
}
