//! Logging utilities for LLM providers.

use crate::messages::TokenUsage;

/// Log token usage in a standardized format across all providers.
///
/// This function logs token usage information including prompt tokens,
/// completion tokens, prompt-cache counters, and total tokens. If usage is None, it logs a warning.
pub fn log_token_usage(usage: Option<&TokenUsage>, provider_name: &str) {
    log_prompt_cache_usage(usage, provider_name, None);
}

/// Log token usage and attach a provider prompt-cache key when one exists.
pub fn log_prompt_cache_usage(
    usage: Option<&TokenUsage>,
    provider_name: &str,
    cache_key: Option<&str>,
) {
    if let Some(usage) = usage {
        crate::llm::prompt_cache::PromptCacheReport::from_usage(usage)
            .log(provider_name, cache_key);
    } else {
        tracing::warn!("No token usage found in {} response", provider_name);
    }
}
