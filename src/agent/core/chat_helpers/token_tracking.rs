use crate::messages::TokenUsage;

#[derive(Debug, Default, Clone)]
pub struct TokenUsageAccumulator {
    prompt_tokens: u32,
    completion_tokens: u32,
    embedding_tokens: u32,
    total_tokens: u32,
    context_window: Option<u32>,
    cache_creation_tokens: u32,
    cache_read_tokens: u32,
    last_prompt_tokens: Option<u32>,
}

impl TokenUsageAccumulator {
    pub(crate) fn add_usage(&mut self, usage: &TokenUsage) {
        self.prompt_tokens = self
            .prompt_tokens
            .saturating_add(usage.prompt_tokens.unwrap_or(0));
        self.completion_tokens = self
            .completion_tokens
            .saturating_add(usage.completion_tokens.unwrap_or(0));
        self.embedding_tokens = self
            .embedding_tokens
            .saturating_add(usage.embedding_tokens.unwrap_or(0));
        self.total_tokens = self
            .total_tokens
            .saturating_add(usage.total_tokens.unwrap_or(0));
        self.cache_creation_tokens = self
            .cache_creation_tokens
            .saturating_add(usage.cache_creation_tokens.unwrap_or(0));
        self.cache_read_tokens = self
            .cache_read_tokens
            .saturating_add(usage.cache_read_tokens.unwrap_or(0));
        if let Some(prompt) = usage.last_prompt_tokens.or(usage.prompt_tokens) {
            self.last_prompt_tokens = Some(prompt);
        }
    }

    pub(crate) const fn total_tokens(&self) -> u32 {
        self.total_tokens
    }

    /// Set the model context window size (for display as utilization percentage).
    pub(crate) fn set_context_window(&mut self, window: Option<usize>) {
        self.context_window = window.and_then(|w| u32::try_from(w).ok());
    }

    /// Convert to a `TokenUsage` snapshot by reference, leaving the accumulator intact.
    ///
    /// Prefer this over `clone().into_token_usage()` when you need the accumulator to
    /// keep accumulating after the snapshot is taken.
    pub(crate) fn to_token_usage(&self) -> TokenUsage {
        TokenUsage {
            prompt_tokens: if self.prompt_tokens > 0 {
                Some(self.prompt_tokens)
            } else {
                None
            },
            completion_tokens: if self.completion_tokens > 0 {
                Some(self.completion_tokens)
            } else {
                None
            },
            embedding_tokens: if self.embedding_tokens > 0 {
                Some(self.embedding_tokens)
            } else {
                None
            },
            total_tokens: if self.total_tokens > 0 {
                Some(self.total_tokens)
            } else {
                None
            },
            context_window: self.context_window,
            cache_creation_tokens: if self.cache_creation_tokens > 0 {
                Some(self.cache_creation_tokens)
            } else {
                None
            },
            cache_read_tokens: if self.cache_read_tokens > 0 {
                Some(self.cache_read_tokens)
            } else {
                None
            },
            last_prompt_tokens: self.last_prompt_tokens,
        }
        .ensure_total_tokens()
    }

    pub(crate) fn into_token_usage(self) -> TokenUsage {
        self.to_token_usage()
    }
}

#[cfg(test)]
mod tests {
    use super::TokenUsageAccumulator;
    use crate::messages::TokenUsage;

    #[test]
    fn last_prompt_tokens_follows_the_latest_request() {
        let mut acc = TokenUsageAccumulator::default();
        acc.add_usage(&TokenUsage {
            prompt_tokens: Some(100_000),
            completion_tokens: Some(10),
            ..TokenUsage::default()
        });
        acc.add_usage(&TokenUsage {
            prompt_tokens: Some(22_848),
            completion_tokens: Some(20),
            ..TokenUsage::default()
        });
        let usage = acc.to_token_usage();
        assert_eq!(usage.prompt_tokens, Some(122_848));
        assert_eq!(usage.last_prompt_tokens, Some(22_848));
    }
}
