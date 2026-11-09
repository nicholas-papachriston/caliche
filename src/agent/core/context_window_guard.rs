//! Context window guard: triggers compaction when history exceeds a threshold.

use super::compaction::{CompactionResult, CompactionStrategy};
use crate::agent::memory::TokenCounter;
use crate::messages::Message;
use anyhow::Result;
use std::sync::Arc;

/// Result of checking whether history fits within the context budget.
#[derive(Debug)]
pub enum GuardResult {
    /// History fits within budget, no changes needed.
    WithinBudget,
    /// History was compacted to fit.
    Compacted(CompactionResult),
    /// Could not compact enough to fit (caller should error or degrade).
    OverBudget {
        estimated_tokens: usize,
        budget: usize,
    },
}

/// Pre-flight check before an LLM call: compact history if over threshold.
#[derive(Debug)]
pub struct ContextWindowGuard {
    /// Strategy to use when compaction is needed.
    strategy: Arc<dyn CompactionStrategy>,
    /// Fraction of context window that triggers compaction (default: 0.90).
    trigger_threshold: f64,
    /// Target fraction after compaction (default: 0.70).
    target_threshold: f64,
    /// Maximum number of compaction attempts per chat turn (default: 2).
    max_compaction_attempts: u32,
}

impl ContextWindowGuard {
    /// Create a guard with the given strategy and default thresholds.
    pub fn new(strategy: Arc<dyn CompactionStrategy>) -> Self {
        Self {
            strategy,
            trigger_threshold: 0.90,
            target_threshold: 0.70,
            max_compaction_attempts: 2,
        }
    }

    /// Set the fraction of context window that triggers compaction (e.g. 0.90 = 90%).
    #[must_use]
    pub const fn trigger_threshold(mut self, f: f64) -> Self {
        self.trigger_threshold = f;
        self
    }

    /// Set the target fraction after compaction (e.g. 0.70 = 70%).
    #[must_use]
    pub const fn target_threshold(mut self, f: f64) -> Self {
        self.target_threshold = f;
        self
    }

    /// Set max compaction attempts per turn.
    #[must_use]
    pub const fn max_compaction_attempts(mut self, n: u32) -> Self {
        self.max_compaction_attempts = n;
        self
    }

    /// Pre-flight check before an LLM call. Returns compacted history if needed.
    #[tracing::instrument(
        name = "compaction.guard",
        skip_all,
        fields(
            compaction.strategy = self.strategy.name(),
            compaction.context_window = context_window,
            compaction.estimated_tokens = tracing::field::Empty,
            compaction.tokens_after = tracing::field::Empty,
        )
    )]
    pub async fn check_and_compact(
        &self,
        messages: &[Message],
        context_window: usize,
        tool_schema_tokens: usize,
        token_counter: &dyn TokenCounter,
    ) -> Result<GuardResult> {
        let budget = context_window.saturating_sub(tool_schema_tokens);
        let estimated_tokens = token_counter.estimate_messages_tokens(messages);
        tracing::Span::current().record("compaction.estimated_tokens", estimated_tokens);

        let trigger_limit = (budget as f64 * self.trigger_threshold) as usize;
        if estimated_tokens <= trigger_limit {
            return Ok(GuardResult::WithinBudget);
        }

        tracing::debug!(
            estimated_tokens,
            budget,
            trigger_limit,
            "context window guard triggered compaction"
        );

        let target_tokens = (budget as f64 * self.target_threshold) as usize;
        let mut current = messages.to_vec();
        let mut attempts = 0u32;

        while attempts < self.max_compaction_attempts {
            let result = self
                .strategy
                .compact(&current, target_tokens, token_counter)
                .await?;

            current = result.messages;
            let after_tokens = token_counter.estimate_messages_tokens(&current);

            if after_tokens <= budget {
                tracing::Span::current().record("compaction.tokens_after", after_tokens);
                return Ok(GuardResult::Compacted(CompactionResult {
                    messages: current,
                    summary: result.summary,
                    tokens_before: result.tokens_before,
                    tokens_after: after_tokens,
                    messages_removed: result.messages_removed,
                    compaction_type: result.compaction_type,
                    report: result.report,
                }));
            }

            attempts += 1;
        }

        // Final fallback: try one more time with a more aggressive target so the agent loop can continue
        let aggressive_target = (budget as f64 * 0.5) as usize;
        if aggressive_target < target_tokens {
            let result = self
                .strategy
                .compact(&current, aggressive_target, token_counter)
                .await?;
            current = result.messages;
            let after_tokens = token_counter.estimate_messages_tokens(&current);
            if after_tokens <= budget {
                return Ok(GuardResult::Compacted(CompactionResult {
                    messages: current,
                    summary: result.summary,
                    tokens_before: result.tokens_before,
                    tokens_after: after_tokens,
                    messages_removed: result.messages_removed,
                    compaction_type: result.compaction_type,
                    report: result.report,
                }));
            }
        }

        let final_tokens = token_counter.estimate_messages_tokens(&current);
        Ok(GuardResult::OverBudget {
            estimated_tokens: final_tokens,
            budget,
        })
    }
}
