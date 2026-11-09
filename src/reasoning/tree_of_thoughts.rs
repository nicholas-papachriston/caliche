//! Tree of Thoughts reasoning pattern.
//!
//! Explores multiple reasoning paths in parallel using BFS/DFS search.
//! Each node represents a reasoning state that can be expanded and evaluated.

use async_trait::async_trait;

use crate::AgentService;
use crate::messages::{Message, MessageContent, MessageRole, TokenUsage};
use crate::reasoning::{ReasoningPattern, ReasoningResult, ReasoningStep, StepType};
use serde_json::json;
use std::collections::VecDeque;
use std::sync::Arc;

/// A node in the search tree representing a reasoning state
#[derive(Debug, Clone)]
struct SearchNode {
    /// Message history leading to this state
    history: Arc<Vec<Message>>,
    /// Evaluation score (higher is better)
    score: f32,
    /// Depth in the search tree
    depth: usize,
    /// The response text at this node
    response: String,
}

impl SearchNode {
    fn new(history: Arc<Vec<Message>>, response: String, depth: usize) -> Self {
        Self {
            history,
            score: 0.0,
            depth,
            response,
        }
    }
}

/// Tree of Thoughts reasoning pattern
pub struct TreeOfThoughtsPattern {
    agent: Arc<tokio::sync::Mutex<AgentService>>,
    max_depth: usize,
    branching_factor: usize,
    beam_width: usize, // Number of top nodes to keep at each level
}

impl std::fmt::Debug for TreeOfThoughtsPattern {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TreeOfThoughtsPattern")
            .field("agent", &"Arc<Mutex<AgentService>>")
            .field("max_depth", &self.max_depth)
            .field("branching_factor", &self.branching_factor)
            .field("beam_width", &self.beam_width)
            .finish()
    }
}

impl TreeOfThoughtsPattern {
    pub fn new(agent: AgentService, max_depth: usize, branching_factor: usize) -> Self {
        Self {
            agent: Arc::new(tokio::sync::Mutex::new(agent)),
            max_depth,
            branching_factor,
            beam_width: branching_factor, // Default to same as branching factor
        }
    }

    /// Set the beam width (number of top nodes to keep at each level)
    pub fn with_beam_width(mut self, beam_width: usize) -> Self {
        self.beam_width = beam_width;
        self
    }

    /// Expand a node by generating k continuations
    async fn expand_node(
        &self,
        node: &SearchNode,
        _query: &str,
    ) -> anyhow::Result<Vec<SearchNode>> {
        let mut expanded_nodes = Vec::new();

        // Generate k different continuations
        for branch_idx in 0..self.branching_factor {
            let expansion_prompt = format!(
                "Given this reasoning so far:\n{}\n\nGenerate a different approach or continuation (approach {} of {}):",
                node.response,
                branch_idx + 1,
                self.branching_factor
            );

            let mut messages = (*node.history).clone();
            messages.push(Message {
                role: MessageRole::User,
                content: MessageContent::text(expansion_prompt),
                metadata: None,
            });

            let step_result = {
                let agent_service = self.agent.lock().await;
                agent_service.step(Arc::new(messages)).await?
            };

            let new_node = SearchNode::new(
                step_result.updated_history.clone(),
                (*step_result.response_text).clone(),
                node.depth + 1,
            );

            expanded_nodes.push(new_node);
        }

        Ok(expanded_nodes)
    }

    /// Evaluate a node's quality using a "judge" prompt
    async fn evaluate_node(&self, node: &SearchNode, query: &str) -> anyhow::Result<f32> {
        let judge_prompt = format!(
            "Evaluate the quality of this reasoning approach on a scale of 0.0 to 1.0:\n\nQuestion: {}\n\nReasoning:\n{}\n\nRate this approach (0.0 = poor, 1.0 = excellent). Respond with just a number.",
            query, node.response
        );

        let judge_messages = vec![
            Message {
                role: MessageRole::System,
                content: MessageContent::text(
                    "You are an evaluator. Rate reasoning quality from 0.0 to 1.0.".to_string(),
                ),
                metadata: None,
            },
            Message {
                role: MessageRole::User,
                content: MessageContent::text(judge_prompt),
                metadata: None,
            },
        ];

        let step_result = {
            let agent_service = self.agent.lock().await;
            agent_service.step(Arc::new(judge_messages)).await?
        };

        // Parse score from response
        let score = step_result
            .response_text
            .trim()
            .chars()
            .take_while(|c: &char| c.is_ascii_digit() || *c == '.')
            .collect::<String>()
            .parse::<f32>()
            .unwrap_or(0.5); // Default to 0.5 if parsing fails

        Ok(score.clamp(0.0, 1.0)) // Clamp to [0.0, 1.0]
    }
}

#[async_trait]
impl ReasoningPattern for TreeOfThoughtsPattern {
    async fn execute(&self, query: &str) -> anyhow::Result<ReasoningResult> {
        let mut reasoning_steps = Vec::new();
        let mut total_usage = TokenUsage::default();

        // Initialize root node
        let root_messages = Arc::new(vec![
            Message {
                role: MessageRole::System,
                content: MessageContent::text(format!(
                    "You are exploring different reasoning approaches to solve: {}",
                    query
                )),
                metadata: None,
            },
            Message {
                role: MessageRole::User,
                content: MessageContent::text(format!("Problem: {}", query)),
                metadata: None,
            },
        ]);

        // Get initial response for root
        let root_step = {
            let agent_service = self.agent.lock().await;
            agent_service.step(root_messages.clone()).await?
        };
        total_usage = total_usage + root_step.usage;

        let mut root_node = SearchNode::new(
            root_step.updated_history.clone(),
            (*root_step.response_text).clone(),
            0,
        );
        root_node.score = self.evaluate_node(&root_node, query).await?;

        reasoning_steps.push(ReasoningStep {
            step_type: StepType::Thought,
            content: format!("Initial approach: {}", root_node.response),
            metadata: Some(json!({"depth": 0, "score": root_node.score})),
        });

        // BFS search: maintain a queue of nodes to explore
        let mut queue = VecDeque::new();
        queue.push_back(root_node);
        let mut best_node: Option<SearchNode> = None;

        // Search up to max_depth
        for depth in 0..self.max_depth {
            if queue.is_empty() {
                break;
            }

            // Expand all nodes at current depth
            let mut next_level = Vec::new();
            while let Some(node) = queue.pop_front() {
                if node.depth >= self.max_depth {
                    // Update best node if this is better
                    if best_node
                        .as_ref()
                        .map(|b| b.score < node.score)
                        .unwrap_or(true)
                    {
                        best_node = Some(node.clone());
                    }
                    continue;
                }

                // Expand this node
                let expanded = self.expand_node(&node, query).await?;

                // Evaluate each expanded node
                for mut child in expanded {
                    child.score = self.evaluate_node(&child, query).await?;
                    next_level.push(child);
                }

                reasoning_steps.push(ReasoningStep {
                    step_type: StepType::Thought,
                    content: format!(
                        "Expanded node at depth {} with {} branches",
                        depth, self.branching_factor
                    ),
                    metadata: Some(json!({"depth": depth, "branches": self.branching_factor})),
                });
            }

            // Select top beam_width nodes for next level
            next_level.sort_by(|a, b| {
                b.score
                    .partial_cmp(&a.score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            next_level.truncate(self.beam_width);

            // Update best node
            if let Some(top_node) = next_level.first()
                && best_node
                    .as_ref()
                    .map(|b| b.score < top_node.score)
                    .unwrap_or(true)
            {
                best_node = Some(top_node.clone());
            }

            // Add selected nodes to queue for next iteration
            for node in next_level {
                queue.push_back(node);
            }
        }

        // Use best node to generate final answer
        let best = best_node.unwrap_or_else(|| {
            // Fallback: use root node
            SearchNode::new(
                root_step.updated_history.clone(),
                (*root_step.response_text).clone(),
                0,
            )
        });

        // Generate final answer from best path
        let final_prompt = format!(
            "Based on the best reasoning approach:\n\n{}\n\nProvide the final answer to: {}",
            best.response, query
        );

        let final_messages = {
            let mut msgs = (*best.history).clone();
            msgs.push(Message {
                role: MessageRole::User,
                content: MessageContent::text(final_prompt),
                metadata: None,
            });
            Arc::new(msgs)
        };

        let final_result = {
            let agent_service = self.agent.lock().await;
            agent_service.step(final_messages).await?
        };
        total_usage = total_usage + final_result.usage;

        // Convert Arc<String> to String for final answer
        let final_answer = (*final_result.response_text).clone();

        reasoning_steps.push(ReasoningStep {
            step_type: StepType::Answer,
            content: final_answer.clone(),
            metadata: Some(json!({"best_score": best.score, "depth": best.depth})),
        });

        Ok(ReasoningResult::success(final_answer, reasoning_steps).with_usage(total_usage))
    }

    fn name(&self) -> &str {
        "Tree of Thoughts"
    }

    fn description(&self) -> &str {
        "Explores multiple reasoning paths and selects the most promising"
    }
}
