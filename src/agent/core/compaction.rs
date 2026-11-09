//! Compaction strategies for conversation history when context window is exceeded.

use crate::agent::memory::TokenCounter;
use crate::messages::{CompactionChangeKind, CompactionReport, Message, MessageRole};
use anyhow::Result;
use async_trait::async_trait;
use std::sync::Arc;

/// Type of compaction that was applied.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompactionType {
    /// LLM summarization of older turns
    LlmSummarization,
    /// Sliding window: drop oldest messages
    SlidingWindow,
    /// Importance-based: drop low-scoring messages first
    ImportanceScoring,
    /// Simple pruning (binary search removal)
    Pruning,
}

/// Result of a compaction operation.
#[derive(Debug, Clone)]
pub struct CompactionResult {
    /// Compacted message list
    pub messages: Vec<Message>,
    /// Optional summary text (for `LlmSummarization`)
    pub summary: Option<String>,
    /// Token count before compaction
    pub tokens_before: usize,
    /// Token count after compaction
    pub tokens_after: usize,
    /// Number of messages removed or merged
    pub messages_removed: usize,
    /// Which compaction strategy was used
    pub compaction_type: CompactionType,
    /// Machine-readable omitted / truncated / compacted changes.
    pub report: CompactionReport,
}

impl CompactionResult {
    /// Result for history that already fits the budget: returned unchanged.
    fn unchanged(messages: &[Message], tokens: usize, compaction_type: CompactionType) -> Self {
        Self {
            messages: messages.to_vec(),
            summary: None,
            tokens_before: tokens,
            tokens_after: tokens,
            messages_removed: 0,
            compaction_type,
            report: CompactionReport::new(),
        }
    }
}

/// Record messages present in `before` but absent from `after`.
fn report_removed_messages(
    before: &[Message],
    after: &[Message],
    source: &'static str,
    kind: CompactionChangeKind,
) -> CompactionReport {
    let mut remaining: std::collections::HashMap<(MessageRole, String), usize> =
        std::collections::HashMap::new();
    for message in after {
        *remaining
            .entry((message.role.clone(), message.extract_text()))
            .or_insert(0) += 1;
    }
    let mut report = CompactionReport::new();
    for message in before {
        let key = (message.role.clone(), message.extract_text());
        match remaining.get_mut(&key) {
            Some(count) if *count > 0 => {
                *count -= 1;
            }
            _ => {
                report.record(
                    kind,
                    source,
                    format!(
                        "dropped {} message ({} chars)",
                        role_label(message.role.clone()),
                        message.extract_text().len()
                    ),
                );
            }
        }
    }
    report
}

const fn role_label(role: MessageRole) -> &'static str {
    match role {
        MessageRole::System => "system",
        MessageRole::User => "user",
        MessageRole::Assistant => "assistant",
        MessageRole::Tool => "tool",
    }
}

/// Strategy for compacting conversation history when context window fills up.
#[async_trait]
pub trait CompactionStrategy: Send + Sync + std::fmt::Debug {
    /// Compact the given message history to fit within `target_tokens`.
    /// Returns the compacted messages and metadata.
    async fn compact(
        &self,
        messages: &[Message],
        target_tokens: usize,
        token_counter: &dyn TokenCounter,
    ) -> Result<CompactionResult>;

    /// Human-readable name of the strategy.
    fn name(&self) -> &str;
}

/// Sliding window: drop oldest non-system messages until within budget.
/// Keeps all system messages and the last N messages (default 4).
#[derive(Debug, Clone)]
pub struct SlidingWindowCompaction {
    /// Number of recent messages to preserve (default: 4)
    pub preserve_recent: usize,
}

impl Default for SlidingWindowCompaction {
    fn default() -> Self {
        Self { preserve_recent: 4 }
    }
}

/// Builder for `SlidingWindowCompaction`.
#[derive(Debug, Default)]
pub struct SlidingWindowCompactionBuilder {
    preserve_recent: Option<usize>,
}

impl SlidingWindowCompactionBuilder {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    #[must_use]
    pub const fn preserve_recent(mut self, n: usize) -> Self {
        self.preserve_recent = Some(n);
        self
    }

    #[must_use]
    pub fn build(self) -> SlidingWindowCompaction {
        SlidingWindowCompaction {
            preserve_recent: self.preserve_recent.unwrap_or(4),
        }
    }
}

impl SlidingWindowCompaction {
    /// Fluent builder for sliding window compaction.
    #[must_use]
    pub fn builder() -> SlidingWindowCompactionBuilder {
        SlidingWindowCompactionBuilder::new()
    }
}

/// Drop oldest non-system messages before the recent window until under budget.
fn drop_oldest_until_budget(
    messages: &[Message],
    target_tokens: usize,
    token_counter: &dyn TokenCounter,
    preserve_recent: usize,
) -> Vec<Message> {
    let tokens_before = token_counter.estimate_messages_tokens(messages);
    if tokens_before <= target_tokens {
        return messages.to_vec();
    }

    let n = messages.len();
    let preserve_recent = preserve_recent.min(n);
    if n <= preserve_recent {
        return messages.to_vec();
    }

    let recent_start = n.saturating_sub(preserve_recent);
    // Ascending index order, so the oldest droppable message comes first.
    let droppable: Vec<usize> = messages
        .iter()
        .enumerate()
        .take(recent_start)
        .filter(|(_, message)| !matches!(message.role, MessageRole::System))
        .map(|(index, _)| index)
        .collect();
    let mut current_tokens = tokens_before;
    let mut drop_indices = std::collections::BTreeSet::new();
    for index in droppable {
        if current_tokens <= target_tokens {
            break;
        }
        current_tokens =
            current_tokens.saturating_sub(token_counter.estimate_message_tokens(&messages[index]));
        drop_indices.insert(index);
    }

    messages
        .iter()
        .enumerate()
        .filter(|(index, _)| !drop_indices.contains(index))
        .map(|(_, message)| message.clone())
        .collect()
}

#[async_trait]
impl CompactionStrategy for SlidingWindowCompaction {
    async fn compact(
        &self,
        messages: &[Message],
        target_tokens: usize,
        token_counter: &dyn TokenCounter,
    ) -> Result<CompactionResult> {
        let tokens_before = token_counter.estimate_messages_tokens(messages);
        if tokens_before <= target_tokens {
            return Ok(CompactionResult::unchanged(
                messages,
                tokens_before,
                CompactionType::SlidingWindow,
            ));
        }

        let pruned =
            drop_oldest_until_budget(messages, target_tokens, token_counter, self.preserve_recent);
        let tokens_after = token_counter.estimate_messages_tokens(&pruned);
        let messages_removed = messages.len().saturating_sub(pruned.len());
        let report = report_removed_messages(
            messages,
            &pruned,
            "sliding_window",
            CompactionChangeKind::Omitted,
        );

        Ok(CompactionResult {
            messages: pruned,
            summary: None,
            tokens_before,
            tokens_after,
            messages_removed,
            compaction_type: CompactionType::SlidingWindow,
            report,
        })
    }

    fn name(&self) -> &'static str {
        "sliding_window"
    }
}

/// Heuristic importance score for a message (higher = more important to keep).
const fn message_importance(msg: &Message) -> u32 {
    match msg.role {
        MessageRole::System => 100,
        MessageRole::User => 10,
        MessageRole::Assistant => 8,
        MessageRole::Tool => 3,
    }
}

/// Importance scoring: drop lowest-importance messages first until within budget.
#[derive(Debug, Clone)]
pub struct ImportanceScoringCompaction {
    /// Number of recent messages to always keep regardless of score (default: 4)
    pub preserve_recent: usize,
}

impl Default for ImportanceScoringCompaction {
    fn default() -> Self {
        Self { preserve_recent: 4 }
    }
}

#[async_trait]
impl CompactionStrategy for ImportanceScoringCompaction {
    async fn compact(
        &self,
        messages: &[Message],
        target_tokens: usize,
        token_counter: &dyn TokenCounter,
    ) -> Result<CompactionResult> {
        let tokens_before = token_counter.estimate_messages_tokens(messages);
        if tokens_before <= target_tokens {
            return Ok(CompactionResult::unchanged(
                messages,
                tokens_before,
                CompactionType::ImportanceScoring,
            ));
        }

        let n = messages.len();
        let preserve_recent = self.preserve_recent.min(n);
        let recent_start = n.saturating_sub(preserve_recent);

        // Index, importance, token count for each message (excluding always-kept recent)
        let mut candidates: Vec<(usize, u32, usize)> = messages
            .iter()
            .enumerate()
            .take(recent_start)
            .map(|(i, m)| {
                let imp = message_importance(m);
                let tok = token_counter.estimate_message_tokens(m);
                (i, imp, tok)
            })
            .collect();

        // Sort by importance ascending (drop low importance first), then by index ascending
        candidates.sort_by(|a, b| a.1.cmp(&b.1).then_with(|| a.0.cmp(&b.0)));

        let mut drop_indices: Vec<usize> = Vec::new();
        let mut current_tokens = tokens_before;
        for (idx, _imp, tok) in candidates {
            if current_tokens <= target_tokens {
                break;
            }
            drop_indices.push(idx);
            current_tokens = current_tokens.saturating_sub(tok);
        }

        drop_indices.sort_unstable();
        let kept: Vec<Message> = messages
            .iter()
            .enumerate()
            .filter(|(i, _)| drop_indices.binary_search(i).is_err())
            .map(|(_, m)| m.clone())
            .collect();

        let tokens_after = token_counter.estimate_messages_tokens(&kept);
        let messages_removed = messages.len().saturating_sub(kept.len());
        let report = report_removed_messages(
            messages,
            &kept,
            "importance_scoring",
            CompactionChangeKind::Omitted,
        );

        Ok(CompactionResult {
            messages: kept,
            summary: None,
            tokens_before,
            tokens_after,
            messages_removed,
            compaction_type: CompactionType::ImportanceScoring,
            report,
        })
    }

    fn name(&self) -> &'static str {
        "importance_scoring"
    }
}

/// LLM-based summarization: summarize older turns into one system message, keep recent N verbatim.
#[derive(Debug)]
pub struct LlmSummarizationCompaction {
    /// Provider used for summarization (can be cheaper/smaller model).
    pub summarization_provider: Arc<dyn crate::llm::Provider>,
    /// Number of recent messages to keep verbatim (default: 6)
    pub preserve_recent: usize,
    /// Max tokens for the summary output (default: 500)
    pub max_summary_tokens: usize,
    /// Prompt template: {history} is replaced with the conversation to summarize.
    pub summarization_prompt_template: String,
    /// Optional Jev asker (TypeSafe System One) for tool call/result pre-filtering.
    pub jev_asker: Option<Arc<dyn crate::agent::core::compaction_jev::JevAsker>>,
    /// Options for Jev pre-filtering.
    pub jev_options: crate::agent::core::compaction_jev::JevCompactOptions,
}

impl LlmSummarizationCompaction {
    pub const DEFAULT_PROMPT: &'static str = "Summarize the following conversation concisely, preserving key facts, decisions, and context. Output only the summary, no preamble.\n\n---\n{history}\n---";

    pub fn new(
        summarization_provider: Arc<dyn crate::llm::Provider>,
        preserve_recent: usize,
        max_summary_tokens: usize,
    ) -> Self {
        Self {
            summarization_provider,
            preserve_recent,
            max_summary_tokens,
            summarization_prompt_template: Self::DEFAULT_PROMPT.to_string(),
            jev_asker: None,
            jev_options: crate::agent::core::compaction_jev::JevCompactOptions::default(),
        }
    }

    /// Fluent builder for LLM summarization compaction.
    #[must_use]
    pub fn builder() -> LlmSummarizationCompactionBuilder {
        LlmSummarizationCompactionBuilder::new()
    }
}

/// Builder for `LlmSummarizationCompaction`.
#[derive(Debug)]
pub struct LlmSummarizationCompactionBuilder {
    summarization_provider: Option<Arc<dyn crate::llm::Provider>>,
    preserve_recent: Option<usize>,
    max_summary_tokens: Option<usize>,
    summarization_prompt_template: Option<String>,
    jev_asker: Option<Arc<dyn crate::agent::core::compaction_jev::JevAsker>>,
    jev_options: Option<crate::agent::core::compaction_jev::JevCompactOptions>,
}

impl LlmSummarizationCompactionBuilder {
    #[must_use]
    pub fn new() -> Self {
        Self {
            summarization_provider: None,
            preserve_recent: None,
            max_summary_tokens: None,
            summarization_prompt_template: None,
            jev_asker: None,
            jev_options: None,
        }
    }

    pub fn provider(mut self, p: Arc<dyn crate::llm::Provider>) -> Self {
        self.summarization_provider = Some(p);
        self
    }

    #[must_use]
    pub const fn preserve_recent(mut self, n: usize) -> Self {
        self.preserve_recent = Some(n);
        self
    }

    #[must_use]
    pub const fn max_summary_tokens(mut self, n: usize) -> Self {
        self.max_summary_tokens = Some(n);
        self
    }

    pub fn summarization_prompt_template(mut self, t: impl Into<String>) -> Self {
        self.summarization_prompt_template = Some(t.into());
        self
    }

    pub fn jev_asker(
        mut self,
        asker: Arc<dyn crate::agent::core::compaction_jev::JevAsker>,
    ) -> Self {
        self.jev_asker = Some(asker);
        self
    }

    #[must_use]
    pub fn jev_options(
        mut self,
        options: crate::agent::core::compaction_jev::JevCompactOptions,
    ) -> Self {
        self.jev_options = Some(options);
        self
    }

    pub fn build(self) -> Result<LlmSummarizationCompaction> {
        let summarization_provider = self
            .summarization_provider
            .ok_or_else(|| anyhow::anyhow!("provider is required"))?;
        Ok(LlmSummarizationCompaction {
            summarization_provider,
            preserve_recent: self.preserve_recent.unwrap_or(6),
            max_summary_tokens: self.max_summary_tokens.unwrap_or(500),
            summarization_prompt_template: self
                .summarization_prompt_template
                .unwrap_or_else(|| LlmSummarizationCompaction::DEFAULT_PROMPT.to_string()),
            jev_asker: self.jev_asker,
            jev_options: self.jev_options.unwrap_or_default(),
        })
    }
}

impl Default for LlmSummarizationCompactionBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl CompactionStrategy for LlmSummarizationCompaction {
    async fn compact(
        &self,
        messages: &[Message],
        target_tokens: usize,
        token_counter: &dyn TokenCounter,
    ) -> Result<CompactionResult> {
        let tokens_before = token_counter.estimate_messages_tokens(messages);
        if tokens_before <= target_tokens {
            return Ok(CompactionResult::unchanged(
                messages,
                tokens_before,
                CompactionType::LlmSummarization,
            ));
        }

        let n = messages.len();
        let preserve_recent = self.preserve_recent.min(n);
        if n <= preserve_recent {
            return Ok(CompactionResult::unchanged(
                messages,
                tokens_before,
                CompactionType::LlmSummarization,
            ));
        }

        let to_summarize = &messages[0..n.saturating_sub(preserve_recent)];
        let recent = &messages[n.saturating_sub(preserve_recent)..];

        let mut filtered_messages: Option<Vec<Message>> = None;
        let mut report = CompactionReport::new();

        if let Some(ref jev_asker) = self.jev_asker {
            let calls = crate::agent::core::compaction_jev::collect_tool_calls(
                messages,
                self.preserve_recent,
            );
            if !calls.is_empty() {
                match crate::agent::core::compaction_jev::evaluate_tool_decisions(
                    jev_asker.as_ref(),
                    messages,
                    &calls,
                    &self.jev_options,
                )
                .await
                {
                    Ok(decisions) => {
                        let candidate_calls = calls.iter().filter(|c| !c.pinned).count();
                        let dropped_or_truncated = decisions
                            .iter()
                            .filter(|d| {
                                d.action
                                    != crate::agent::core::compaction_jev::JevCallAction::KeepBoth
                            })
                            .count();
                        let reduction_ratio = if candidate_calls == 0 {
                            0.0
                        } else {
                            dropped_or_truncated as f64 / candidate_calls as f64
                        };

                        if reduction_ratio >= self.jev_options.min_reduction_ratio {
                            report.extend(
                                crate::agent::core::compaction_jev::report_from_jev_decisions(
                                    &decisions,
                                ),
                            );
                            let filtered =
                                crate::agent::core::compaction_jev::filter_messages_for_summarizer(
                                    to_summarize,
                                    &decisions,
                                    self.jev_options.truncate_head_chars,
                                );
                            filtered_messages = Some(filtered);
                        } else {
                            tracing::debug!(
                                reduction_ratio,
                                min_required = self.jev_options.min_reduction_ratio,
                                "Jev compaction pre-filter yielded insufficient reduction; falling back to unfiltered summarizer"
                            );
                        }
                    }
                    Err(err) => {
                        tracing::warn!(
                            error = %err,
                            "Jev compaction evaluation failed; falling back to unfiltered summarizer"
                        );
                    }
                }
            }
        }

        let slice_for_summary = filtered_messages.as_deref().unwrap_or(to_summarize);

        let history_text: String = slice_for_summary
            .iter()
            .map(|m| {
                let role = match m.role {
                    MessageRole::System => "System",
                    MessageRole::User => "User",
                    MessageRole::Assistant => "Assistant",
                    MessageRole::Tool => "Tool",
                };
                format!("[{}] {}", role, m.extract_text())
            })
            .collect::<Vec<_>>()
            .join("\n\n");

        let prompt = self
            .summarization_prompt_template
            .replace("{history}", &history_text);

        let request = crate::messages::CompletionRequest {
            messages: Arc::new(vec![
                Message::system("You are a concise summarizer. Output only the summary."),
                Message::user(prompt),
            ]),
            max_tokens: Some(self.max_summary_tokens as u32),
            temperature: Some(0.3),
            reasoning_effort: None,
            tools: None,
            tool_choice: None,
            response_format: None,
        };

        let response = self.summarization_provider.complete(request).await?;
        let summary = response.content.extract_text();

        let summary_msg = Message::system(format!(
            "[Previous conversation summary]\n{}",
            summary.trim()
        ));
        let mut compacted = vec![summary_msg];
        compacted.extend_from_slice(recent);

        // If the summary + recent window is still oversized, drop oldest non-system
        // messages while preserving the recent window.
        if token_counter.estimate_messages_tokens(&compacted) > target_tokens {
            compacted =
                drop_oldest_until_budget(&compacted, target_tokens, token_counter, preserve_recent);
        }

        let tokens_after = token_counter.estimate_messages_tokens(&compacted);
        let messages_removed = messages.len().saturating_sub(compacted.len());
        report.record(
            CompactionChangeKind::Compacted,
            "llm_summarization",
            format!(
                "replaced {} older messages with a summary ({} chars)",
                to_summarize.len(),
                summary.len()
            ),
        );
        report.extend(report_removed_messages(
            messages,
            &compacted,
            "llm_summarization",
            CompactionChangeKind::Omitted,
        ));

        Ok(CompactionResult {
            messages: compacted,
            summary: Some(summary),
            tokens_before,
            tokens_after,
            messages_removed,
            compaction_type: CompactionType::LlmSummarization,
            report,
        })
    }

    fn name(&self) -> &'static str {
        "llm_summarization"
    }
}

/// Composite: try strategies in order until one succeeds and fits.
#[derive(Debug)]
pub struct CompositeCompaction {
    /// Strategies to try in order (e.g. [`ImportanceScoring`, `SlidingWindow`] or [`ImportanceScoring`, `LlmSummarization`])
    pub strategies: Vec<Arc<dyn CompactionStrategy>>,
}

#[async_trait]
impl CompactionStrategy for CompositeCompaction {
    async fn compact(
        &self,
        messages: &[Message],
        target_tokens: usize,
        token_counter: &dyn TokenCounter,
    ) -> Result<CompactionResult> {
        let mut last_error: Option<anyhow::Error> = None;
        for strategy in &self.strategies {
            match strategy
                .compact(messages, target_tokens, token_counter)
                .await
            {
                Ok(result) if result.tokens_after <= target_tokens => return Ok(result),
                Ok(result) => {
                    last_error = Some(anyhow::anyhow!(
                        "Strategy {} did not reduce enough ({} > {} tokens)",
                        strategy.name(),
                        result.tokens_after,
                        target_tokens
                    ));
                }
                Err(e) => {
                    last_error = Some(e.context(format!("Strategy {}", strategy.name())));
                }
            }
        }
        Err(last_error.unwrap_or_else(|| anyhow::anyhow!("No compaction strategy succeeded")))
    }

    fn name(&self) -> &'static str {
        "composite"
    }
}

/// Configuration for context window compaction (strategy + guard thresholds).
/// Use with `AgentBuilder::compaction_config()` to wire compaction from a single config.
#[derive(Debug, Clone)]
pub struct CompactionConfig {
    /// Compaction strategy (e.g. sliding window or LLM summarization).
    pub strategy: Arc<dyn CompactionStrategy>,
    /// Fraction of context window that triggers compaction (e.g. 0.90).
    pub trigger_threshold: f64,
    /// Target fraction after compaction (e.g. 0.70).
    pub target_threshold: f64,
    /// Max compaction attempts per chat turn.
    pub max_attempts: u32,
}

/// Builder for `CompactionConfig`.
#[derive(Debug)]
pub struct CompactionConfigBuilder {
    strategy: Option<Arc<dyn CompactionStrategy>>,
    trigger_threshold: Option<f64>,
    target_threshold: Option<f64>,
    max_attempts: Option<u32>,
}

impl CompactionConfigBuilder {
    #[must_use]
    pub fn new() -> Self {
        Self {
            strategy: None,
            trigger_threshold: None,
            target_threshold: None,
            max_attempts: None,
        }
    }

    pub fn strategy(mut self, s: Arc<dyn CompactionStrategy>) -> Self {
        self.strategy = Some(s);
        self
    }

    #[must_use]
    pub const fn trigger_threshold(mut self, f: f64) -> Self {
        self.trigger_threshold = Some(f);
        self
    }

    #[must_use]
    pub const fn target_threshold(mut self, f: f64) -> Self {
        self.target_threshold = Some(f);
        self
    }

    #[must_use]
    pub const fn max_attempts(mut self, n: u32) -> Self {
        self.max_attempts = Some(n);
        self
    }

    pub fn build(self) -> Result<CompactionConfig> {
        let strategy = self
            .strategy
            .ok_or_else(|| anyhow::anyhow!("strategy is required"))?;
        Ok(CompactionConfig {
            strategy,
            trigger_threshold: self.trigger_threshold.unwrap_or(0.90),
            target_threshold: self.target_threshold.unwrap_or(0.70),
            max_attempts: self.max_attempts.unwrap_or(2),
        })
    }
}

impl Default for CompactionConfigBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl CompactionConfig {
    /// Fluent builder for compaction config (strategy + guard thresholds).
    #[must_use]
    pub fn builder() -> CompactionConfigBuilder {
        CompactionConfigBuilder::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::memory::token_counter::ApproximateTokenCounter;
    use crate::testing::mock_provider::MockProvider;

    #[test]
    fn report_removed_messages_counts_duplicate_texts() {
        let before = vec![
            Message::user("hi"),
            Message::user("hi"),
            Message::assistant("ok"),
        ];
        let after = vec![Message::user("hi"), Message::assistant("ok")];
        let report = report_removed_messages(
            &before,
            &after,
            "sliding_window",
            CompactionChangeKind::Omitted,
        );
        assert_eq!(
            report.changes.len(),
            1,
            "one of two identical user messages was dropped: {:?}",
            report.changes
        );
    }

    fn long_history(n: usize) -> Vec<Message> {
        let mut messages = vec![Message::system("system preamble")];
        for i in 0..n {
            messages.push(Message::user(format!("user turn {i} {}", "x".repeat(80))));
            messages.push(Message::assistant(format!(
                "assistant turn {i} {}",
                "y".repeat(80)
            )));
        }
        messages
    }

    #[tokio::test]
    async fn sliding_window_honors_preserve_recent() {
        let counter = ApproximateTokenCounter::default();
        let messages = long_history(10);
        let strategy = SlidingWindowCompaction { preserve_recent: 2 };
        let before = counter.estimate_messages_tokens(&messages);
        let result = strategy
            .compact(&messages, before / 3, &counter)
            .await
            .expect("compact");
        assert!(result.messages_removed > 0);
        assert!(result.tokens_after <= before / 3 || result.messages.len() <= 3);
        // System message retained; last two messages retained.
        assert!(matches!(result.messages[0].role, MessageRole::System));
        assert_eq!(
            result.messages[result.messages.len() - 1].extract_text(),
            messages[messages.len() - 1].extract_text()
        );
        assert_eq!(
            result.messages[result.messages.len() - 2].extract_text(),
            messages[messages.len() - 2].extract_text()
        );
    }

    #[tokio::test]
    async fn sliding_window_noop_when_under_budget() {
        let counter = ApproximateTokenCounter::default();
        let messages = vec![Message::user("hi"), Message::assistant("hello")];
        let strategy = SlidingWindowCompaction::default();
        let result = strategy
            .compact(&messages, 10_000, &counter)
            .await
            .expect("compact");
        assert_eq!(result.messages_removed, 0);
        assert_eq!(result.messages.len(), 2);
    }

    #[tokio::test]
    async fn importance_scoring_prefers_dropping_tool_messages() {
        let counter = ApproximateTokenCounter::default();
        let messages = vec![
            Message::system("sys"),
            Message::user("u1"),
            Message {
                role: MessageRole::Tool,
                content: crate::messages::MessageContent::text("tool output ".repeat(40)),
                metadata: None,
            },
            Message::assistant("a1"),
            Message::user("u2"),
            Message::assistant("a2"),
        ];
        let before = counter.estimate_messages_tokens(&messages);
        let strategy = ImportanceScoringCompaction { preserve_recent: 2 };
        let result = strategy
            .compact(&messages, before.saturating_sub(20), &counter)
            .await
            .expect("compact");
        assert!(result.messages_removed >= 1);
        assert!(
            result
                .messages
                .iter()
                .all(|m| !matches!(m.role, MessageRole::Tool))
                || result.messages_removed > 0
        );
        // Recent window kept.
        assert_eq!(
            result.messages[result.messages.len() - 1].extract_text(),
            "a2"
        );
    }

    #[tokio::test]
    async fn composite_falls_through_when_first_strategy_insufficient() {
        let counter = ApproximateTokenCounter::default();
        let messages = long_history(8);
        let before = counter.estimate_messages_tokens(&messages);
        // First strategy preserves almost everything (won't fit aggressive budget).
        let weak = Arc::new(SlidingWindowCompaction {
            preserve_recent: messages.len(),
        }) as Arc<dyn CompactionStrategy>;
        let strong =
            Arc::new(SlidingWindowCompaction { preserve_recent: 2 }) as Arc<dyn CompactionStrategy>;
        let composite = CompositeCompaction {
            strategies: vec![weak, strong],
        };
        let result = composite
            .compact(&messages, before / 4, &counter)
            .await
            .expect("composite");
        assert_eq!(result.compaction_type, CompactionType::SlidingWindow);
        assert!(result.tokens_after <= before / 4 || result.messages.len() <= 3);
    }

    #[tokio::test]
    async fn llm_summarization_keeps_recent_and_inserts_summary() {
        let counter = ApproximateTokenCounter::default();
        let messages = long_history(6);
        let before = counter.estimate_messages_tokens(&messages);
        let provider = Arc::new(MockProvider::new(vec![
            "Summary of earlier turns.".to_string(),
        ]));
        let strategy = LlmSummarizationCompaction::new(provider, 2, 128);
        let result = strategy
            .compact(&messages, before / 2, &counter)
            .await
            .expect("summarize");
        assert_eq!(result.compaction_type, CompactionType::LlmSummarization);
        assert!(result.summary.is_some());
        assert!(
            result.messages[0]
                .extract_text()
                .contains("Previous conversation summary")
        );
        assert_eq!(
            result.messages[result.messages.len() - 1].extract_text(),
            messages[messages.len() - 1].extract_text()
        );
    }

    #[test]
    fn compaction_config_builder_requires_strategy() {
        let err = CompactionConfig::builder().build().unwrap_err();
        assert!(err.to_string().contains("strategy"));
    }

    #[tokio::test]
    async fn llm_summarization_with_mock_jev_prefilter_drops_and_truncates() {
        use crate::agent::core::compaction_jev::{
            JevAnswer, JevAsker, JevCompactOptions, JevQuestion, JevResponse,
        };
        use std::collections::HashMap;

        #[derive(Debug)]
        struct FakeJev;

        #[async_trait::async_trait]
        impl JevAsker for FakeJev {
            async fn ask(
                &self,
                _state: &serde_json::Value,
                questions: &HashMap<String, JevQuestion>,
            ) -> anyhow::Result<JevResponse> {
                let mut answers = HashMap::new();
                for q_key in questions.keys() {
                    // Drop call_1 completely (both call and result < 0.5)
                    if q_key.contains("t1") {
                        answers.insert(q_key.clone(), JevAnswer { noul: Some(0.1) });
                    }
                    // Keep call but truncate result for call_2 (call >= 0.5, result < 0.5)
                    else if q_key.contains("t2") {
                        if q_key.starts_with("call_") {
                            answers.insert(q_key.clone(), JevAnswer { noul: Some(0.9) });
                        } else {
                            answers.insert(q_key.clone(), JevAnswer { noul: Some(0.1) });
                        }
                    }
                    // Keep both for other calls
                    else {
                        answers.insert(q_key.clone(), JevAnswer { noul: Some(0.9) });
                    }
                }
                Ok(JevResponse {
                    model: Some("jev-latest".to_string()),
                    answers,
                    usage: None,
                })
            }
        }

        let counter = ApproximateTokenCounter::default();
        let messages = vec![
            Message::system("System instructions"),
            Message::user("Task 1"),
            // Tool call 1 (will be dropped)
            Message {
                role: MessageRole::Assistant,
                content: crate::messages::MessageContent::Structured {
                    text: None,
                    tool_calls: Some(vec![crate::messages::ToolCallInfo::from_json(
                        "c1",
                        "read_file",
                        r#"{"path":"old.txt"}"#,
                    )]),
                    tool_call_id: None,
                },
                metadata: None,
            },
            Message {
                role: MessageRole::Tool,
                content: crate::messages::MessageContent::Structured {
                    text: Some(Arc::new("contents of old.txt ".repeat(30))),
                    tool_calls: None,
                    tool_call_id: Some(beef::lean::Cow::owned("c1".to_string())),
                },
                metadata: None,
            },
            Message::user("Task 2"),
            // Tool call 2 (result will be truncated)
            Message {
                role: MessageRole::Assistant,
                content: crate::messages::MessageContent::Structured {
                    text: Some(Arc::new("Checking file".to_string())),
                    tool_calls: Some(vec![crate::messages::ToolCallInfo::from_json(
                        "c2",
                        "grep_search",
                        r#"{"query":"important"}"#,
                    )]),
                    tool_call_id: None,
                },
                metadata: None,
            },
            Message {
                role: MessageRole::Tool,
                content: crate::messages::MessageContent::Structured {
                    text: Some(Arc::new("very long grep results ".repeat(40))),
                    tool_calls: None,
                    tool_call_id: Some(beef::lean::Cow::owned("c2".to_string())),
                },
                metadata: None,
            },
            // Recent messages (preserved)
            Message::user("Current question"),
            Message::assistant("Current reply"),
        ];

        let before = counter.estimate_messages_tokens(&messages);
        let provider = Arc::new(MockProvider::new(vec![
            "Filtered summary of prior actions.".to_string(),
        ]));

        let strategy = LlmSummarizationCompaction::builder()
            .provider(provider)
            .preserve_recent(2)
            .max_summary_tokens(128)
            .jev_asker(Arc::new(FakeJev))
            .jev_options(JevCompactOptions {
                truncate_head_chars: 50,
                ..Default::default()
            })
            .build()
            .expect("build strategy");

        let result = strategy
            .compact(&messages, before / 2, &counter)
            .await
            .expect("compact");

        assert_eq!(result.compaction_type, CompactionType::LlmSummarization);
        assert!(result.summary.is_some());
        assert_eq!(result.messages.len(), 3); // 1 summary + 2 recent messages
        assert!(
            result
                .report
                .changes
                .iter()
                .any(|c| c.source == "jev" && c.kind == CompactionChangeKind::Omitted),
            "Jev drop should appear in the compaction report: {:?}",
            result.report
        );
        assert!(
            result
                .report
                .changes
                .iter()
                .any(|c| c.source == "jev" && c.kind == CompactionChangeKind::Truncated),
            "Jev truncate should appear in the compaction report: {:?}",
            result.report
        );
        assert!(
            result
                .report
                .changes
                .iter()
                .any(|c| c.kind == CompactionChangeKind::Compacted),
            "summary should be recorded as compacted: {:?}",
            result.report
        );
    }

    #[tokio::test]
    async fn llm_summarization_with_jev_network_error_falls_back_gracefully() {
        use crate::agent::core::compaction_jev::{
            JevAsker, JevCompactOptions, JevQuestion, JevResponse,
        };
        use std::collections::HashMap;

        #[derive(Debug)]
        struct FailingJev;

        #[async_trait::async_trait]
        impl JevAsker for FailingJev {
            async fn ask(
                &self,
                _state: &serde_json::Value,
                _questions: &HashMap<String, JevQuestion>,
            ) -> anyhow::Result<JevResponse> {
                Err(anyhow::anyhow!("simulated connection timeout"))
            }
        }

        let counter = ApproximateTokenCounter::default();
        let messages = long_history(6);
        let before = counter.estimate_messages_tokens(&messages);
        let provider = Arc::new(MockProvider::new(vec![
            "Fallback summary after Jev error.".to_string(),
        ]));

        let strategy = LlmSummarizationCompaction::builder()
            .provider(provider)
            .preserve_recent(2)
            .max_summary_tokens(128)
            .jev_asker(Arc::new(FailingJev))
            .jev_options(JevCompactOptions::default())
            .build()
            .expect("build strategy");

        let result = strategy
            .compact(&messages, before / 2, &counter)
            .await
            .expect("compact fallback succeeded");

        assert_eq!(result.compaction_type, CompactionType::LlmSummarization);
        assert!(result.summary.is_some());
    }
}
