//! Composite memory strategy that combines a primary strategy with one or more secondaries.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use anyhow::Result;
use async_trait::async_trait;

use crate::agent::memory::base::{MemoryInspection, MemorySnapshot, MemoryStats, MemoryStrategy};
use crate::agent::memory::token_counter::TokenCounter;
use crate::messages::Message;

/// Content fingerprint for deduplication (role + text). Optional metadata id is included when
/// present so clients can distinguish same (role, text) at different times.
fn message_fingerprint(msg: &Message) -> String {
    let text = msg.content.as_text().unwrap_or_default();
    let meta_id = msg
        .metadata
        .as_ref()
        .and_then(|m| m.get("id").and_then(|v| v.as_str()))
        .unwrap_or("");
    format!("{:?}:{}:{}", msg.role, text, meta_id)
}

/// Base fingerprint (role + text only) for occurrence counting.
fn base_fingerprint(msg: &Message) -> String {
    let text = msg.content.as_text().unwrap_or_default();
    format!("{:?}:{}", msg.role, text)
}

/// Composite memory: primary strategy for recent context plus secondaries for supplemental retrieval.
pub struct CompositeMemory {
    /// Primary strategy (e.g. `BufferMemory`) for recent context.
    primary: Box<dyn MemoryStrategy>,
    /// Secondary strategies (e.g. `VectorMemory`) for supplemental context.
    secondaries: Vec<Box<dyn MemoryStrategy>>,
    /// Token counter for budget trimming.
    token_counter: Arc<dyn TokenCounter>,
    /// Max tokens for the combined context.
    max_tokens: usize,
    /// Fraction of budget (0.0–1.0) allocated to primary; rest goes to secondaries.
    primary_budget_pct: f32,
}

impl CompositeMemory {
    /// Create a new composite memory.
    pub fn new(
        primary: Box<dyn MemoryStrategy>,
        secondaries: Vec<Box<dyn MemoryStrategy>>,
        token_counter: Arc<dyn TokenCounter>,
        max_tokens: usize,
        primary_budget_pct: f32,
    ) -> Self {
        Self {
            primary,
            secondaries,
            token_counter,
            max_tokens,
            primary_budget_pct: primary_budget_pct.clamp(0.0, 1.0),
        }
    }

    /// Trim messages to fit a token budget (by dropping oldest).
    fn trim_to_budget(&self, messages: &[Message], budget: usize) -> Vec<Message> {
        if budget == 0 {
            return vec![];
        }
        let mut total = 0usize;
        let mut out = Vec::with_capacity(messages.len());
        for msg in messages.iter().rev() {
            let t = self.token_counter.estimate_message_tokens(msg);
            if total.saturating_add(t) <= budget {
                total = total.saturating_add(t);
                out.push(msg.clone());
            } else {
                break;
            }
        }
        out.reverse();
        out
    }
}

#[async_trait]
impl MemoryStrategy for CompositeMemory {
    async fn add_messages(&mut self, messages: Vec<Message>) -> Result<()> {
        if messages.is_empty() {
            return Ok(());
        }
        // Clone only for secondaries; move owned vec on the last consumer.
        if self.secondaries.is_empty() {
            return self.primary.add_messages(messages).await;
        }
        self.primary.add_messages(messages.clone()).await?;
        let last_idx = self.secondaries.len() - 1;
        for (i, s) in self.secondaries.iter_mut().enumerate() {
            if i == last_idx {
                s.add_messages(messages).await?;
                break;
            }
            s.add_messages(messages.clone()).await?;
        }
        Ok(())
    }

    async fn get_context(&self) -> Result<Vec<Message>> {
        let primary_budget = (self.max_tokens as f32 * self.primary_budget_pct).round() as usize;
        let mut primary_msgs = self.primary.get_context().await?;
        primary_msgs = self.trim_to_budget(&primary_msgs, primary_budget);

        // Count occurrences of (role, text) in primary so we only treat secondary as duplicate
        // when the same message appears exactly once in primary (preserves temporal context).
        let mut primary_base_count: HashMap<String, usize> = HashMap::new();
        for msg in &primary_msgs {
            *primary_base_count.entry(base_fingerprint(msg)).or_insert(0) += 1;
        }
        let mut seen_fp = HashSet::new();
        for msg in &primary_msgs {
            seen_fp.insert(message_fingerprint(msg));
        }
        let mut total_tokens: usize = self
            .token_counter
            .estimate_messages_tokens(primary_msgs.as_slice());
        let mut out = primary_msgs;
        for secondary in &self.secondaries {
            if total_tokens >= self.max_tokens {
                break;
            }
            let sec_budget = self.max_tokens.saturating_sub(total_tokens);
            let mut sec_msgs = secondary.get_context().await?;
            sec_msgs = self.trim_to_budget(&sec_msgs, sec_budget);
            for msg in sec_msgs {
                let fp = message_fingerprint(&msg);
                let base = base_fingerprint(&msg);
                let count = primary_base_count.get(&base).copied().unwrap_or(0);
                // Skip only when (role, text) appears exactly once in primary (likely same message).
                // If 0 or 2+ times, add to preserve temporal context.
                let skip = count == 1 && seen_fp.contains(&fp);
                if !skip && seen_fp.insert(fp) {
                    let t = self.token_counter.estimate_message_tokens(&msg);
                    if total_tokens.saturating_add(t) <= self.max_tokens {
                        total_tokens = total_tokens.saturating_add(t);
                        out.push(msg);
                    }
                }
            }
        }

        Ok(out)
    }

    async fn clear(&mut self) -> Result<()> {
        self.primary.clear().await?;
        for s in &mut self.secondaries {
            s.clear().await?;
        }
        Ok(())
    }

    fn stats(&self) -> MemoryStats {
        let primary_s = self.primary.stats();
        let mut total_messages = primary_s.total_messages;
        let mut total_tokens = primary_s.total_tokens_estimated;
        let mut total_summaries = primary_s.total_summaries;
        let mut total_entities = primary_s.total_entities;
        let mut last_updated = primary_s.last_updated;
        for s in &self.secondaries {
            let st = s.stats();
            total_messages = total_messages.saturating_add(st.total_messages);
            total_tokens = total_tokens.saturating_add(st.total_tokens_estimated);
            total_summaries = total_summaries.saturating_add(st.total_summaries);
            total_entities = total_entities.saturating_add(st.total_entities);
            if st.last_updated > last_updated {
                last_updated = st.last_updated;
            }
        }
        MemoryStats {
            total_messages,
            total_tokens_estimated: total_tokens,
            total_summaries,
            total_entities,
            last_updated,
        }
    }

    async fn inspect(&self) -> Result<MemoryInspection> {
        let primary_in = self.primary.inspect().await?;
        let mut messages = primary_in.messages;
        let mut summaries = primary_in.summaries;
        let mut entities = primary_in.entities;
        let mut seen = HashSet::new();
        for msg in &messages {
            seen.insert(message_fingerprint(msg));
        }
        for s in &self.secondaries {
            let in_ = s.inspect().await?;
            for msg in in_.messages {
                if seen.insert(message_fingerprint(&msg)) {
                    messages.push(msg);
                }
            }
            summaries.extend(in_.summaries);
            entities.extend(in_.entities);
        }
        Ok(MemoryInspection {
            strategy_name: "composite".to_string(),
            messages,
            summaries,
            entities,
            knowledge: vec![],
            stats: self.stats(),
        })
    }

    async fn search(&self, query: &str) -> Result<Vec<(Message, f32)>> {
        let mut primary_results = self.primary.search(query).await?;
        let mut seen = HashSet::new();
        for (msg, _) in &primary_results {
            seen.insert(message_fingerprint(msg));
        }
        for s in &self.secondaries {
            let sec = s.search(query).await?;
            for (msg, score) in sec {
                if seen.insert(message_fingerprint(&msg)) {
                    primary_results.push((msg, score));
                }
            }
        }
        Ok(primary_results)
    }

    async fn fork(&self) -> Result<Box<dyn MemoryStrategy>> {
        let primary_fork = self.primary.fork().await?;
        let mut secondaries_fork = Vec::with_capacity(self.secondaries.len());
        for s in &self.secondaries {
            secondaries_fork.push(s.fork().await?);
        }
        Ok(Box::new(Self::new(
            primary_fork,
            secondaries_fork,
            Arc::clone(&self.token_counter),
            self.max_tokens,
            self.primary_budget_pct,
        )))
    }

    async fn save_state(&self) -> Result<MemorySnapshot> {
        let primary_snap = self.primary.save_state().await?;
        let mut secondary_snaps = Vec::with_capacity(self.secondaries.len());
        for s in &self.secondaries {
            secondary_snaps.push(s.save_state().await?);
        }
        let metadata = serde_json::json!({
            "primary": primary_snap,
            "secondaries": secondary_snaps,
        });
        // Use the combined context as `messages` for diagnostics; authoritative
        // state lives inside `metadata.primary` / `metadata.secondaries`.
        let messages = self.get_context().await.unwrap_or_default();
        Ok(MemorySnapshot { messages, metadata })
    }

    async fn load_state(&mut self, snapshot: MemorySnapshot) -> Result<()> {
        let primary_snap: MemorySnapshot = snapshot
            .metadata
            .get("primary")
            .cloned()
            .and_then(|v| serde_json::from_value(v).ok())
            .unwrap_or_default();
        self.primary.load_state(primary_snap).await?;

        let secondary_snaps: Vec<MemorySnapshot> = snapshot
            .metadata
            .get("secondaries")
            .cloned()
            .and_then(|v| serde_json::from_value(v).ok())
            .unwrap_or_default();
        for (idx, s) in self.secondaries.iter_mut().enumerate() {
            let snap = secondary_snaps.get(idx).cloned().unwrap_or_default();
            s.load_state(snap).await?;
        }
        Ok(())
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}
