//! Pluggable context engine: controls how conversation context is assembled,
//! compacted, and maintained across the agent's lifecycle.
//!
//! The [`ContextEngine`] trait provides four lifecycle hooks that the agent step
//! loop calls at well-defined points. The [`DefaultContextEngine`] wraps the
//! existing [`CompactionStrategy`] + [`ContextWindowGuard`] so that behaviour is
//! unchanged for users who do not provide a custom engine.
//!
//! Custom engines can implement entirely different strategies: DAG-based
//! summarisation, RAG-driven assembly, vector retrieval instead of sliding
//! window, etc.

use crate::agent::memory::TokenCounter;
use crate::messages::Message;
use anyhow::Result;
use async_trait::async_trait;
use std::sync::Arc;

#[cfg(feature = "compaction")]
use super::context_window_guard::ContextWindowGuard;

/// Token budget handed to the engine before each LLM call.
#[derive(Debug, Clone, Copy)]
pub struct TokenBudget {
    /// Hard ceiling from the model's context window.
    pub max_tokens: usize,
    /// Tokens reserved for the expected response.
    pub reserved_for_response: usize,
    /// Tokens consumed by tool schemas.
    pub tool_schema_tokens: usize,
    /// Effective tokens available for message content.
    pub available: usize,
}

impl TokenBudget {
    #[must_use]
    pub const fn new(
        max_tokens: usize,
        reserved_for_response: usize,
        tool_schema_tokens: usize,
    ) -> Self {
        let available = max_tokens
            .saturating_sub(reserved_for_response)
            .saturating_sub(tool_schema_tokens);
        Self {
            max_tokens,
            reserved_for_response,
            tool_schema_tokens,
            available,
        }
    }
}

/// The output of [`ContextEngine::assemble`]: the messages to send and metadata.
#[derive(Debug, Clone)]
pub struct AssembledContext {
    /// Ordered messages to include in the LLM call.
    pub messages: Vec<Message>,
    /// Engine's estimate of total tokens in the assembled messages.
    pub estimated_tokens: usize,
    /// Optional text to prepend/append to the system prompt.
    pub system_prompt_addition: Option<String>,
    /// When `true`, the engine owns compaction; the step loop will not apply
    /// its own pruning/compaction on top.
    pub owns_compaction: bool,
}

/// Lifecycle trait for pluggable context management.
///
/// Engines are exclusive: one engine at a time controls context assembly for a
/// given agent. The default engine (`DefaultContextEngine`) preserves the
/// existing compaction + pruning behaviour.
#[async_trait]
pub trait ContextEngine: Send + Sync + std::fmt::Debug {
    /// Called when a new message enters the session.
    ///
    /// Engines can index, persist, or transform the message here.
    /// The default implementation is a no-op.
    async fn ingest(&self, _session_id: &str, _message: &Message) -> Result<()> {
        Ok(())
    }

    /// Build the context for the next LLM call within the given budget.
    ///
    /// Returns an [`AssembledContext`] whose `messages` fit within
    /// `budget.available` tokens.
    async fn assemble(
        &self,
        session_id: &str,
        messages: &[Message],
        budget: &TokenBudget,
        token_counter: &dyn TokenCounter,
    ) -> Result<AssembledContext>;

    /// Explicit compaction request (e.g. when over budget after assembly).
    ///
    /// The default implementation returns the messages unchanged.
    async fn compact(
        &self,
        _session_id: &str,
        messages: &[Message],
        _budget: &TokenBudget,
        _token_counter: &dyn TokenCounter,
    ) -> Result<AssembledContext> {
        let estimated = messages.len() * 4; // rough fallback
        Ok(AssembledContext {
            messages: messages.to_vec(),
            estimated_tokens: estimated,
            system_prompt_addition: None,
            owns_compaction: false,
        })
    }

    /// Called after each complete turn (assistant response consumed).
    ///
    /// Engines can persist summaries, update indexes, etc.
    async fn after_turn(&self, _session_id: &str) -> Result<()> {
        Ok(())
    }

    /// Human-readable name of this engine.
    fn name(&self) -> &str;
}

// ---------------------------------------------------------------------------
// DefaultContextEngine
// ---------------------------------------------------------------------------

/// Wraps the existing `ContextWindowGuard` + `CompactionStrategy` behind the
/// `ContextEngine` trait. Behaviour is identical to pre-context-engine builds.
#[cfg(feature = "compaction")]
#[derive(Debug)]
pub struct DefaultContextEngine {
    guard: Arc<ContextWindowGuard>,
}

#[cfg(feature = "compaction")]
impl DefaultContextEngine {
    #[must_use]
    pub const fn new(guard: Arc<ContextWindowGuard>) -> Self {
        Self { guard }
    }
}

#[cfg(feature = "compaction")]
#[async_trait]
impl ContextEngine for DefaultContextEngine {
    async fn assemble(
        &self,
        _session_id: &str,
        messages: &[Message],
        budget: &TokenBudget,
        token_counter: &dyn TokenCounter,
    ) -> Result<AssembledContext> {
        let context_window = budget.max_tokens;
        let tool_schema_tokens = budget.tool_schema_tokens;

        match self
            .guard
            .check_and_compact(messages, context_window, tool_schema_tokens, token_counter)
            .await?
        {
            super::context_window_guard::GuardResult::WithinBudget => {
                let estimated = token_counter.estimate_messages_tokens(messages);
                Ok(AssembledContext {
                    messages: messages.to_vec(),
                    estimated_tokens: estimated,
                    system_prompt_addition: None,
                    owns_compaction: true,
                })
            }
            super::context_window_guard::GuardResult::Compacted(result) => Ok(AssembledContext {
                messages: result.messages,
                estimated_tokens: result.tokens_after,
                system_prompt_addition: None,
                owns_compaction: true,
            }),
            super::context_window_guard::GuardResult::OverBudget {
                estimated_tokens,
                budget: budget_val,
            } => {
                tracing::warn!(
                    estimated_tokens,
                    budget_val,
                    "DefaultContextEngine: over budget after compaction, falling back to prune"
                );
                let mut pruned = messages.to_vec();
                super::Agent::prune_history_for_budget(token_counter, &mut pruned, budget_val);
                let after = token_counter.estimate_messages_tokens(&pruned);
                if after > budget_val {
                    anyhow::bail!(
                        "Context window exceeded: request needs {after} message tokens after compaction/pruning but the effective message budget is {budget_val}. Reduce the input/history/tool schemas or configure a larger context window."
                    );
                }
                Ok(AssembledContext {
                    messages: pruned,
                    estimated_tokens: after,
                    system_prompt_addition: None,
                    owns_compaction: true,
                })
            }
        }
    }

    async fn compact(
        &self,
        session_id: &str,
        messages: &[Message],
        budget: &TokenBudget,
        token_counter: &dyn TokenCounter,
    ) -> Result<AssembledContext> {
        self.assemble(session_id, messages, budget, token_counter)
            .await
    }

    fn name(&self) -> &'static str {
        "default"
    }
}

// ---------------------------------------------------------------------------
// PassthroughContextEngine (used when compaction is disabled)
// ---------------------------------------------------------------------------

/// Minimal engine that prunes history via binary search but does not run any
/// LLM-based compaction. Used when the `compaction` feature is disabled.
#[derive(Debug)]
pub struct PassthroughContextEngine;

#[async_trait]
impl ContextEngine for PassthroughContextEngine {
    async fn assemble(
        &self,
        _session_id: &str,
        messages: &[Message],
        budget: &TokenBudget,
        token_counter: &dyn TokenCounter,
    ) -> Result<AssembledContext> {
        let estimated = token_counter.estimate_messages_tokens(messages);
        if estimated <= budget.available {
            return Ok(AssembledContext {
                messages: messages.to_vec(),
                estimated_tokens: estimated,
                system_prompt_addition: None,
                owns_compaction: false,
            });
        }
        let mut pruned = messages.to_vec();
        super::Agent::prune_history_for_budget(token_counter, &mut pruned, budget.available);
        let after = token_counter.estimate_messages_tokens(&pruned);
        Ok(AssembledContext {
            messages: pruned,
            estimated_tokens: after,
            system_prompt_addition: None,
            owns_compaction: false,
        })
    }

    fn name(&self) -> &'static str {
        "passthrough"
    }
}
