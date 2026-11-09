use std::collections::VecDeque;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use crate::agent::memory::{
    base::{MemoryInspection, MemorySnapshot, MemoryStats, MemoryStrategy, render_conversation},
    config::SummaryMemoryConfig,
    token_counter::{TokenCounter, estimate_messages_tokens_iter},
};
use crate::llm::Provider;
use crate::messages::{CompletionRequest, Message, MessageRole};
use anyhow::Result;
use async_trait::async_trait;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Service responsible for turning a batch of messages into a concise summary.
#[async_trait]
pub trait ConversationSummarizer: Send + Sync {
    /// Generate a summary representing the provided message history.
    async fn summarize(&self, messages: &[Message]) -> Result<String>;
}

#[derive(Clone, Serialize, Deserialize)]
pub(crate) struct SummaryEntry {
    message: Message,
    token_estimate: usize,
    created_at: DateTime<Utc>,
    source_messages: usize,
}

impl SummaryEntry {
    const fn new(
        message: Message,
        token_estimate: usize,
        created_at: DateTime<Utc>,
        source_messages: usize,
    ) -> Self {
        Self {
            message,
            token_estimate,
            created_at,
            source_messages,
        }
    }

    fn as_message(&self) -> Message {
        self.message.clone()
    }
}

/// Result payload from a background summarization task. The `generation` is
/// captured at spawn time so that completions triggered before a `clear()` can
/// be discarded instead of leaking back into state.
type PendingSummaryResult = Result<(u64, Vec<SummaryEntry>)>;

/// Handle for a background summarization task.
type PendingSummaryHandle =
    Arc<tokio::sync::Mutex<Option<tokio::task::JoinHandle<PendingSummaryResult>>>>;

/// Memory strategy that progressively summarizes earlier conversation history to stay within token limits.
///
/// Summarization is intentionally single-flight: only one background summarization run is active at a time to avoid races.
pub struct SummaryMemory {
    config: SummaryMemoryConfig,
    token_counter: Arc<dyn TokenCounter>,
    summarizer: Arc<dyn ConversationSummarizer>,
    messages: VecDeque<Message>,
    summaries: Arc<std::sync::RwLock<VecDeque<SummaryEntry>>>,
    last_updated: Option<DateTime<Utc>>,
    /// Background summarization in progress (collect in `get_context` / `maybe_summarize`).
    pending_summary: PendingSummaryHandle,
    /// Monotonic counter bumped on `clear()`. Spawned summarization tasks capture
    /// the value at spawn and any result whose generation no longer matches is
    /// discarded so stale summaries can't resurrect cleared state.
    summary_generation: Arc<AtomicU64>,
}

impl SummaryMemory {
    /// Create a new summary memory strategy using the provided configuration, token counter, and summarizer.
    pub fn new(
        config: SummaryMemoryConfig,
        token_counter: Arc<dyn TokenCounter>,
        summarizer: Arc<dyn ConversationSummarizer>,
    ) -> Self {
        Self {
            config,
            token_counter,
            summarizer,
            messages: VecDeque::new(),
            summaries: Arc::new(std::sync::RwLock::new(VecDeque::new())),
            last_updated: None,
            pending_summary: Arc::new(tokio::sync::Mutex::new(None)),
            summary_generation: Arc::new(AtomicU64::new(0)),
        }
    }

    /// Create a summary memory from forked state (same config, cloned messages and summaries).
    pub(crate) fn from_fork(
        config: SummaryMemoryConfig,
        token_counter: Arc<dyn TokenCounter>,
        summarizer: Arc<dyn ConversationSummarizer>,
        messages: VecDeque<Message>,
        summaries: VecDeque<SummaryEntry>,
        last_updated: Option<DateTime<Utc>>,
    ) -> Self {
        Self {
            config,
            token_counter,
            summarizer,
            messages,
            summaries: Arc::new(std::sync::RwLock::new(summaries)),
            last_updated,
            pending_summary: Arc::new(tokio::sync::Mutex::new(None)),
            summary_generation: Arc::new(AtomicU64::new(0)),
        }
    }

    /// Collect any completed background summary into summaries (non-blocking).
    async fn collect_completed_summaries(&self) -> Result<()> {
        let pending = {
            let mut pending = self.pending_summary.lock().await;
            pending.take()
        };
        let mut handle = match pending {
            Some(h) => h,
            None => return Ok(()),
        };
        let result = tokio::select! {
            biased;
            res = &mut handle => Some(res),
            () = tokio::time::sleep(Duration::ZERO) => None
        };
        if let Some(join_result) = result {
            match join_result {
                Ok(Ok((task_generation, entries))) => {
                    // Discard results that were produced before a clear() reset the memory.
                    if task_generation != self.summary_generation.load(Ordering::Acquire) {
                        return Ok(());
                    }
                    let mut sum = self
                        .summaries
                        .write()
                        .map_err(|e| anyhow::anyhow!("summaries lock: {e}"))?;
                    sum.extend(entries);
                    if sum.len() > self.config.max_summaries {
                        let excess = sum.len() - self.config.max_summaries;
                        for _ in 0..excess {
                            sum.pop_front();
                        }
                    }
                }
                Ok(Err(e)) => {
                    return Err(anyhow::anyhow!("Background summarization failed: {e}"));
                }
                Err(join_err) => {
                    return Err(anyhow::anyhow!("Background summarization join: {join_err}"));
                }
            }
        } else {
            *self.pending_summary.lock().await = Some(handle);
        }
        Ok(())
    }

    fn threshold_tokens(&self) -> usize {
        if self.config.max_tokens == 0 {
            return 0;
        }

        let fraction = self.config.summarize_threshold_pct.clamp(0.1, 1.0).max(0.1);
        ((self.config.max_tokens as f32) * fraction).floor() as usize
    }

    fn total_token_estimate(&self) -> usize {
        let mut total =
            estimate_messages_tokens_iter(self.token_counter.as_ref(), self.messages.iter());
        if let Ok(sum) = self.summaries.read() {
            total = sum
                .iter()
                .fold(total, |acc, entry| acc.saturating_add(entry.token_estimate));
        }
        total
    }

    async fn maybe_summarize(&mut self) -> Result<()> {
        self.collect_completed_summaries().await?;

        let threshold = self.threshold_tokens();
        if threshold == 0 || self.config.chunk_size == 0 {
            return Ok(());
        }

        let total = self.total_token_estimate();
        if total <= threshold {
            return Ok(());
        }

        let start_index = match self
            .messages
            .iter()
            .position(|message| !matches!(message.role, MessageRole::System))
        {
            Some(index) => index,
            None => return Ok(()),
        };

        let available = self.messages.len().saturating_sub(start_index);
        if available < self.config.chunk_size {
            return Ok(());
        }

        // Only one summarization run at a time; skip spawning if one is already in progress.
        let summary_pending = self.pending_summary.lock().await.is_some();
        if summary_pending {
            return Ok(());
        }

        let end_index = start_index + self.config.chunk_size;
        let chunk: Vec<Message> = self.messages.drain(start_index..end_index).collect();
        let summarizer = Arc::clone(&self.summarizer);
        let token_counter = Arc::clone(&self.token_counter);
        let max_summary_tokens = self
            .config
            .max_summary_tokens
            .unwrap_or(self.config.max_tokens / 4)
            .max(1);
        let task_generation = self.summary_generation.load(Ordering::Acquire);

        let handle = tokio::spawn(async move {
            let summary_text = summarizer.summarize(&chunk).await?;
            let mut formatted = format!(
                "Summary of earlier conversation ({} messages):\n{}",
                chunk.len(),
                summary_text.trim()
            );
            let mut summary_message = Message::new(MessageRole::System, formatted.clone());
            let mut token_estimate = token_counter.estimate_message_tokens(&summary_message);
            if token_estimate > max_summary_tokens {
                const CHARS_PER_TOKEN: usize = 4;
                const PREFIX_TOKENS_APPROX: usize = 10;
                let content_budget = max_summary_tokens.saturating_sub(PREFIX_TOKENS_APPROX);
                let max_content_chars = content_budget
                    .saturating_mul(CHARS_PER_TOKEN)
                    .saturating_sub(3);
                if let Some(nl) = formatted.find('\n') {
                    let content = formatted[nl + 1..].trim();
                    let truncated_content =
                        if content.len() > max_content_chars && max_content_chars > 0 {
                            format!(
                                "{}...",
                                caliche_contracts::text::truncate_utf8(content, max_content_chars)
                                    .trim_end()
                            )
                        } else {
                            content.to_string()
                        };
                    formatted = format!(
                        "Summary of earlier conversation ({} messages):\n{}",
                        chunk.len(),
                        truncated_content
                    );
                    summary_message = Message::new(MessageRole::System, formatted);
                    token_estimate = token_counter.estimate_message_tokens(&summary_message);
                }
            }
            Ok((
                task_generation,
                vec![SummaryEntry::new(
                    summary_message,
                    token_estimate,
                    Utc::now(),
                    chunk.len(),
                )],
            ))
        });

        *self.pending_summary.lock().await = Some(handle);
        Ok(())
    }

    fn enforce_limits(&mut self) {
        while self.total_token_estimate() > self.config.max_tokens {
            if self.remove_first_non_system_message() {
                continue;
            }
            if let Ok(mut sum) = self.summaries.write()
                && sum.len() > 1
            {
                sum.pop_front();
                continue;
            }
            break;
        }
    }

    fn remove_first_non_system_message(&mut self) -> bool {
        // VecDeque::remove is O(min(i, n-i)), which is O(1) for index 0
        // For better performance when removing from front, we can use pop_front after finding
        if let Some(index) = self
            .messages
            .iter()
            .position(|message| !matches!(message.role, MessageRole::System))
        {
            // If removing from front (index 0), use pop_front for O(1)
            if index == 0 {
                self.messages.pop_front();
            } else {
                // Otherwise use remove (O(min(i, n-i)))
                self.messages.remove(index);
            }
            return true;
        }

        false
    }
}

#[async_trait]
impl MemoryStrategy for SummaryMemory {
    async fn add_messages(&mut self, messages: Vec<Message>) -> Result<()> {
        if messages.is_empty() {
            return Ok(());
        }

        // Move messages instead of cloning
        self.messages.extend(messages);
        self.last_updated = Some(Utc::now());

        self.maybe_summarize().await?;
        self.enforce_limits();

        Ok(())
    }

    async fn get_context(&self) -> Result<Vec<Message>> {
        self.collect_completed_summaries().await?;
        let sum = self
            .summaries
            .read()
            .map_err(|e| anyhow::anyhow!("summaries read: {e}"))?;
        let mut context = Vec::with_capacity(sum.len() + self.messages.len());
        context.extend(sum.iter().map(SummaryEntry::as_message));
        context.extend(self.messages.iter().cloned());
        Ok(context)
    }

    async fn clear(&mut self) -> Result<()> {
        self.messages.clear();
        if let Ok(mut sum) = self.summaries.write() {
            sum.clear();
        }
        // Bump the generation so any in-flight summarization result is rejected
        // when it completes, then abort the handle to stop the task eagerly.
        self.summary_generation.fetch_add(1, Ordering::AcqRel);
        let pending = {
            let mut pending = self.pending_summary.lock().await;
            pending.take()
        };
        if let Some(handle) = pending {
            handle.abort();
        }
        self.last_updated = Some(Utc::now());
        Ok(())
    }

    fn stats(&self) -> MemoryStats {
        let summary_len = self.summaries.read().map(|s| s.len()).unwrap_or(0);
        MemoryStats {
            total_messages: self.messages.len() + summary_len,
            total_tokens_estimated: self.total_token_estimate(),
            total_summaries: summary_len,
            total_entities: 0,
            last_updated: self.last_updated,
        }
    }

    async fn save(
        &self,
        persistence: Option<&dyn crate::agent::memory::MemoryPersistence>,
        conversation_id: &str,
    ) -> Result<()> {
        if let Some(store) = persistence {
            let snapshot = self.save_state().await?;
            store.save_snapshot(conversation_id, &snapshot).await?;
        }

        Ok(())
    }

    async fn load(
        &mut self,
        persistence: Option<&dyn crate::agent::memory::MemoryPersistence>,
        conversation_id: &str,
    ) -> Result<()> {
        if let Some(store) = persistence
            && let Some(snapshot) = store.load_snapshot(conversation_id).await?
        {
            self.load_state(snapshot).await?;
        }

        Ok(())
    }

    async fn save_state(&self) -> Result<MemorySnapshot> {
        let summaries_snap: Vec<SummaryEntry> = self
            .summaries
            .read()
            .map_err(|e| anyhow::anyhow!("summaries read: {e}"))?
            .iter()
            .cloned()
            .collect();
        let mut metadata = serde_json::Map::new();
        metadata.insert(
            "summaries".to_string(),
            serde_json::to_value(&summaries_snap)?,
        );
        if let Some(updated) = self.last_updated {
            metadata.insert(
                "last_updated".to_string(),
                Value::String(updated.to_rfc3339()),
            );
        }
        let messages_vec: Vec<Message> = self.messages.iter().cloned().collect();
        Ok(MemorySnapshot {
            messages: messages_vec,
            metadata: Value::Object(metadata),
        })
    }

    async fn load_state(&mut self, snapshot: MemorySnapshot) -> Result<()> {
        // Convert Vec to VecDeque for efficient front removal.
        self.messages = snapshot.messages.into_iter().collect();

        if let Some(summaries_value) = snapshot.metadata.get("summaries") {
            match serde_json::from_value::<Vec<SummaryEntry>>(summaries_value.clone()) {
                Ok(summaries_vec) => {
                    if let Ok(mut sum) = self.summaries.write() {
                        *sum = summaries_vec.into_iter().collect();
                    }
                }
                Err(err) => {
                    tracing::warn!(
                        error = %err,
                        "summary memory load_state: failed to deserialize summaries; starting empty"
                    );
                }
            }
        }

        if let Some(updated) = snapshot
            .metadata
            .get("last_updated")
            .and_then(|value| value.as_str())
            .and_then(|value| DateTime::parse_from_rfc3339(value).ok())
        {
            self.last_updated = Some(updated.with_timezone(&Utc));
        }

        self.enforce_limits();
        Ok(())
    }

    async fn inspect(&self) -> Result<MemoryInspection> {
        let summaries: Vec<String> = {
            let sum = self
                .summaries
                .read()
                .map_err(|e| anyhow::anyhow!("summaries read: {e}"))?;
            sum.iter()
                .map(|e| e.message.content.extract_text())
                .collect()
        };
        Ok(MemoryInspection {
            strategy_name: "summary".to_string(),
            messages: self.get_context().await?,
            summaries,
            entities: vec![],
            knowledge: vec![],
            stats: self.stats(),
        })
    }

    async fn fork(&self) -> Result<Box<dyn MemoryStrategy>> {
        let sum = self
            .summaries
            .read()
            .map_err(|e| anyhow::anyhow!("summaries read: {e}"))?;
        let summaries_clone = sum.clone();
        let forked = Self::from_fork(
            self.config.clone(),
            Arc::clone(&self.token_counter),
            Arc::clone(&self.summarizer),
            self.messages.clone(),
            summaries_clone,
            self.last_updated,
        );
        Ok(Box::new(forked))
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

const SUMMARY_SYSTEM_PROMPT: &str = "You are a meticulous assistant that produces concise, factual summaries of prior conversation turns. Capture key facts, decisions, open questions, and commitments without adding new information.";

/// LLM-backed summarizer that reuses the agent's provider.
pub struct ProviderSummarizer {
    provider: Arc<dyn Provider>,
    max_tokens: u32,
    temperature: f32,
}

impl ProviderSummarizer {
    pub fn new(provider: Arc<dyn Provider>, max_tokens: u32) -> Self {
        Self {
            provider,
            max_tokens: max_tokens.clamp(128, 1024),
            temperature: 0.2,
        }
    }
}

#[async_trait]
impl ConversationSummarizer for ProviderSummarizer {
    async fn summarize(&self, messages: &[Message]) -> Result<String> {
        if messages.is_empty() {
            return Ok(String::new());
        }

        let mut prompt_messages = Vec::with_capacity(messages.len() + 2);
        prompt_messages.push(Message::system(SUMMARY_SYSTEM_PROMPT));
        prompt_messages.push(Message::user(render_conversation(messages)));

        let request = CompletionRequest {
            messages: Arc::new(prompt_messages),
            max_tokens: Some(self.max_tokens),
            temperature: Some(self.temperature),
            reasoning_effort: None,
            tools: None,
            tool_choice: None,
            response_format: None,
        };

        let response = self.provider.complete(request).await?;
        // Use as_text() to avoid unnecessary clone, only clone when trimming
        Ok(response.content.as_text().trim().to_string())
    }
}
