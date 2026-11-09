use crate::agent::memory::TokenCounter;
use crate::errors::AgentError;
use crate::messages::MessageRole;

/// Threshold to start warning about context size (80% of max).
const CONTEXT_WARNING_THRESHOLD: f32 = 0.8;

impl super::Agent {
    /// Workspace root used to resolve relative tool paths.
    ///
    /// Returns `None` when the working directory could not be determined, so tools fall back
    /// to the process working directory.
    pub(crate) fn workspace_root(&self) -> Option<&std::path::Path> {
        if self.current_dir.is_empty() || self.current_dir == "unknown" {
            None
        } else {
            Some(std::path::Path::new(&self.current_dir))
        }
    }

    /// Owned form of [`Self::workspace_root`] for the tool execution engine.
    pub(crate) fn workspace_root_buf(&self) -> Option<std::path::PathBuf> {
        self.workspace_root().map(std::path::Path::to_path_buf)
    }

    /// Full model context window used for reporting and hard-limit checks.
    pub(crate) fn effective_context_window(&self) -> usize {
        self.cached_context_window
            .unwrap_or(self.max_context_tokens)
    }

    /// Tokens reserved for model output or provider-specific overhead.
    pub(crate) fn reserved_context_tokens(&self) -> usize {
        self.effective_context_window()
            .saturating_sub(self.max_context_tokens)
    }

    /// Ensure the current prompt, tool schemas, and reserved output fit in the
    /// configured context window.
    pub(crate) fn ensure_context_window_available(
        &self,
        messages: &[crate::messages::Message],
        tool_schema_tokens: usize,
    ) -> Result<(), AgentError> {
        let message_tokens = Self::estimate_context_tokens(self.token_counter.as_ref(), messages);
        let current = message_tokens
            .saturating_add(tool_schema_tokens)
            .saturating_add(self.reserved_context_tokens());
        let limit = self.effective_context_window();
        if current > limit {
            return Err(AgentError::ContextWindowExceeded { current, limit });
        }
        Ok(())
    }

    /// Estimate total tokens for a message list using the provided counter.
    pub(crate) fn estimate_context_tokens(
        counter: &dyn TokenCounter,
        messages: &[crate::messages::Message],
    ) -> usize {
        counter.estimate_messages_tokens(messages)
    }

    /// Prune message history to stay within token budget.
    ///
    /// Removes older non-system messages in one bulk drain (after finding cut index via binary
    /// search). Always preserves at least the 4 most recent messages.
    pub(crate) fn prune_history_for_budget(
        counter: &dyn TokenCounter,
        messages: &mut Vec<crate::messages::Message>,
        max_tokens: usize,
    ) {
        let current_tokens = Self::estimate_context_tokens(counter, messages);

        if current_tokens <= max_tokens {
            return;
        }

        let warning_threshold = (max_tokens as f32 * CONTEXT_WARNING_THRESHOLD) as usize;
        if current_tokens > warning_threshold {
            tracing::warn!(
                "Context approaching limit: {} tokens (max: {}). Pruning history.",
                current_tokens,
                max_tokens
            );
        }

        let min_recent_messages = 4;
        let n = messages.len();
        if n <= min_recent_messages {
            return;
        }

        let first_non_system = messages
            .iter()
            .position(|m| !matches!(m.role, MessageRole::System));
        let first_remove = match first_non_system {
            Some(i) => i,
            None => return,
        };
        let last_keep_start = n.saturating_sub(min_recent_messages);
        if last_keep_start <= first_remove {
            return;
        }
        let max_remove = last_keep_start.saturating_sub(first_remove);

        let tokens_without_range = |k: usize| {
            let end = first_remove + k;
            let left = counter.estimate_messages_tokens(&messages[0..first_remove]);
            let right = counter.estimate_messages_tokens(&messages[end..n]);
            left.saturating_add(right)
        };

        let mut low = 0usize;
        let mut high = max_remove;
        while low < high {
            let mid = low + (high - low) / 2;
            if tokens_without_range(mid) > max_tokens {
                low = mid + 1;
            } else {
                high = mid;
            }
        }
        let to_remove = high;
        if to_remove == 0 {
            return;
        }

        let drain_end = first_remove + to_remove;
        messages.drain(first_remove..drain_end);

        let final_tokens = Self::estimate_context_tokens(counter, messages);
        tracing::info!(
            "History pruned: {} -> {} tokens ({} messages remaining)",
            current_tokens,
            final_tokens,
            messages.len()
        );
    }
}
