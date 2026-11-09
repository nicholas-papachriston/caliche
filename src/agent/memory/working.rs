//! Working memory abstraction for current task-specific scratchpad.

use crate::agent::memory::token_counter::TokenCounter;
use crate::messages::Message;
use anyhow::Result;
use chrono::{DateTime, Utc};
use std::sync::Arc;

/// Working memory for current task context.
/// This is separate from long-term conversation history and is cleared between tasks.
pub struct WorkingMemory {
    messages: Vec<Message>,
    token_counter: Arc<dyn TokenCounter>,
    max_tokens: usize,
    task_id: Option<String>,
    created_at: DateTime<Utc>,
    last_updated: DateTime<Utc>,
}

impl Clone for WorkingMemory {
    fn clone(&self) -> Self {
        Self {
            messages: self.messages.clone(),
            token_counter: Arc::clone(&self.token_counter),
            max_tokens: self.max_tokens,
            task_id: self.task_id.clone(),
            created_at: self.created_at,
            last_updated: self.last_updated,
        }
    }
}

impl WorkingMemory {
    /// Create a new working memory instance.
    pub fn new(token_counter: Arc<dyn TokenCounter>, max_tokens: usize) -> Self {
        let now = Utc::now();
        Self {
            messages: Vec::new(),
            token_counter,
            max_tokens,
            task_id: None,
            created_at: now,
            last_updated: now,
        }
    }

    /// Create working memory for a specific task.
    pub fn for_task(
        token_counter: Arc<dyn TokenCounter>,
        max_tokens: usize,
        task_id: String,
    ) -> Self {
        let now = Utc::now();
        Self {
            messages: Vec::new(),
            token_counter,
            max_tokens,
            task_id: Some(task_id),
            created_at: now,
            last_updated: now,
        }
    }

    /// Add messages to working memory.
    pub fn add_messages(&mut self, messages: Vec<Message>) -> Result<()> {
        if messages.is_empty() {
            return Ok(());
        }

        self.messages.extend(messages);
        self.enforce_limits();
        self.last_updated = Utc::now();

        Ok(())
    }

    /// Get all messages in working memory.
    #[must_use]
    pub fn get_messages(&self) -> &[Message] {
        &self.messages
    }

    /// Get messages as owned vector.
    #[must_use]
    pub fn into_messages(self) -> Vec<Message> {
        self.messages
    }

    /// Clear working memory.
    pub fn clear(&mut self) {
        self.messages.clear();
        self.task_id = None;
        self.last_updated = Utc::now();
    }

    /// Check if working memory is empty.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.messages.is_empty()
    }

    /// Get current token count.
    #[must_use]
    pub fn token_count(&self) -> usize {
        self.token_counter.estimate_messages_tokens(&self.messages)
    }

    /// Get remaining token capacity.
    #[must_use]
    pub fn remaining_capacity(&self) -> usize {
        self.max_tokens.saturating_sub(self.token_count())
    }

    /// Get task ID if set.
    #[must_use]
    pub fn task_id(&self) -> Option<&str> {
        self.task_id.as_deref()
    }

    /// Set task ID.
    pub fn set_task_id(&mut self, task_id: impl Into<String>) {
        self.task_id = Some(task_id.into());
        self.last_updated = Utc::now();
    }

    /// Get creation timestamp.
    #[must_use]
    pub const fn created_at(&self) -> DateTime<Utc> {
        self.created_at
    }

    /// Get last update timestamp.
    #[must_use]
    pub const fn last_updated(&self) -> DateTime<Utc> {
        self.last_updated
    }

    /// Enforce token limits by removing oldest messages.
    fn enforce_limits(&mut self) {
        let mut total_tokens = self.token_count();
        if total_tokens <= self.max_tokens {
            return;
        }

        let mut drop_count = 0usize;
        while total_tokens > self.max_tokens && drop_count < self.messages.len() {
            total_tokens = total_tokens.saturating_sub(
                self.token_counter
                    .estimate_message_tokens(&self.messages[drop_count]),
            );
            drop_count += 1;
        }

        if drop_count > 0 {
            self.messages.drain(..drop_count);
        }
    }
}

/// Working memory manager for coordinating multiple working memories.
#[derive(Clone)]
pub struct WorkingMemoryManager {
    memories: std::collections::HashMap<String, WorkingMemory>,
    default_max_tokens: usize,
    token_counter: Arc<dyn TokenCounter>,
}

impl WorkingMemoryManager {
    /// Create a new working memory manager.
    pub fn new(token_counter: Arc<dyn TokenCounter>, default_max_tokens: usize) -> Self {
        Self {
            memories: std::collections::HashMap::new(),
            default_max_tokens,
            token_counter,
        }
    }

    /// Get or create working memory for a task.
    pub fn get_or_create(&mut self, task_id: impl Into<String>) -> &mut WorkingMemory {
        let task_id = task_id.into();
        self.memories.entry(task_id.clone()).or_insert_with(|| {
            WorkingMemory::for_task(
                Arc::clone(&self.token_counter),
                self.default_max_tokens,
                task_id,
            )
        })
    }

    /// Get working memory for a task.
    #[must_use]
    pub fn get(&self, task_id: &str) -> Option<&WorkingMemory> {
        self.memories.get(task_id)
    }

    /// Remove working memory for a task.
    pub fn remove(&mut self, task_id: &str) -> Option<WorkingMemory> {
        self.memories.remove(task_id)
    }

    /// Clear all working memories.
    pub fn clear_all(&mut self) {
        self.memories.clear();
    }

    /// Get all task IDs.
    #[must_use]
    pub fn task_ids(&self) -> Vec<String> {
        self.memories.keys().cloned().collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::memory::token_counter::ApproximateTokenCounter;

    #[test]
    fn test_working_memory() {
        let counter = Arc::new(ApproximateTokenCounter::default());
        let wm = WorkingMemory::new(counter, 1000);
        assert!(wm.is_empty());
        assert_eq!(wm.token_count(), 0);
    }

    #[test]
    fn test_working_memory_manager() {
        let counter = Arc::new(ApproximateTokenCounter::default());
        let mut manager = WorkingMemoryManager::new(counter, 1000);
        let wm = manager.get_or_create("task1".to_string());
        assert_eq!(wm.task_id(), Some("task1"));
    }
}
