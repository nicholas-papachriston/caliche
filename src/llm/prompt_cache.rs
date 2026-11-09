//! Provider prompt-cache helpers (Anthropic `cache_control`, OpenAI `prompt_cache_key`).
//!
//! This is distinct from the local exact-match [`crate::llm::caching_provider::CachingProvider`].
//! Prompt cache breakpoints keep a stable system+tools prefix so multi-turn
//! REPL and harness requests can reuse provider inference cache.

use crate::messages::{Message, MessageRole, TokenUsage, ToolSchema};
use std::fmt;
use std::hash::{Hash, Hasher};

/// What a provider reported for prompt-cache reuse on one completion.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PromptCacheOutcome {
    /// No prompt tokens were reported.
    None,
    /// Prompt tokens were billed with no cache read or write.
    Miss,
    /// Tokens were written into the provider cache.
    Write,
    /// Tokens were read from the provider cache.
    Hit,
    /// The response both wrote and read cache tokens.
    Mixed,
}

impl PromptCacheOutcome {
    /// Stable lowercase label for logs and REPL output.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Miss => "miss",
            Self::Write => "write",
            Self::Hit => "hit",
            Self::Mixed => "mixed",
        }
    }
}

impl fmt::Display for PromptCacheOutcome {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Prompt-cache counters derived from [`TokenUsage`] for logs and REPL.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PromptCacheReport {
    /// Prompt / input tokens including cache read and write when the provider reports them that way.
    pub prompt_tokens: u32,
    /// Completion tokens.
    pub completion_tokens: u32,
    /// Tokens read from the provider prompt cache.
    pub cache_read: u32,
    /// Tokens written to the provider prompt cache.
    pub cache_write: u32,
}

impl PromptCacheReport {
    /// Build a report from provider usage counters.
    #[must_use]
    pub fn from_usage(usage: &TokenUsage) -> Self {
        Self {
            prompt_tokens: usage.prompt_tokens.unwrap_or(0),
            completion_tokens: usage.completion_tokens.unwrap_or(0),
            cache_read: usage.cache_read_tokens.unwrap_or(0),
            cache_write: usage.cache_creation_tokens.unwrap_or(0),
        }
    }

    /// Prompt tokens that were not served from cache and not written this turn.
    #[must_use]
    pub const fn uncached_tokens(self) -> u32 {
        self.prompt_tokens
            .saturating_sub(self.cache_read)
            .saturating_sub(self.cache_write)
    }

    /// Cache-read share of prompt tokens, when a prompt size is present.
    #[must_use]
    pub fn hit_ratio(self) -> Option<f64> {
        if self.prompt_tokens == 0 {
            None
        } else {
            Some(f64::from(self.cache_read) / f64::from(self.prompt_tokens))
        }
    }

    /// Hit ratio as a percentage rounded to one decimal place.
    #[must_use]
    pub fn hit_pct(self) -> Option<f64> {
        self.hit_ratio()
            .map(|ratio| (ratio * 10_000.0).round() / 100.0)
    }

    /// Classify cache reuse for this completion.
    #[must_use]
    pub const fn outcome(self) -> PromptCacheOutcome {
        if self.prompt_tokens == 0 && self.cache_read == 0 && self.cache_write == 0 {
            return PromptCacheOutcome::None;
        }
        match (self.cache_read > 0, self.cache_write > 0) {
            (true, true) => PromptCacheOutcome::Mixed,
            (true, false) => PromptCacheOutcome::Hit,
            (false, true) => PromptCacheOutcome::Write,
            (false, false) => PromptCacheOutcome::Miss,
        }
    }

    /// True when the provider reported cache read or write tokens.
    #[must_use]
    pub const fn saw_cache(self) -> bool {
        self.cache_read > 0 || self.cache_write > 0
    }

    /// Emit a structured log line so operators can judge cache effectiveness.
    pub fn log(self, provider: &str, cache_key: Option<&str>) {
        let hit_pct = self.hit_pct();
        let outcome = self.outcome().as_str();
        tracing::info!(
            target: "caliche::prompt_cache",
            provider,
            prompt_tokens = self.prompt_tokens,
            completion_tokens = self.completion_tokens,
            cache_read = self.cache_read,
            cache_write = self.cache_write,
            uncached = self.uncached_tokens(),
            hit_pct,
            outcome,
            prompt_cache_key = cache_key,
            "provider prompt cache"
        );
    }
}

/// Anthropic uses ephemeral `cache_control` breakpoints rather than a string key.
pub const ANTHROPIC_PROMPT_CACHE_KEY: &str = "anthropic-ephemeral";

/// Anthropic / compatible ephemeral cache breakpoint.
#[must_use]
pub fn ephemeral_cache_control() -> serde_json::Value {
    serde_json::json!({"type": "ephemeral"})
}

/// Build an OpenAI `prompt_cache_key` from the stable request prefix.
///
/// System text and tool names stay fixed across a session while chat history
/// grows, so the same key routes related turns onto one cache partition.
#[must_use]
pub fn openai_prompt_cache_key(messages: &[Message], tools: Option<&[ToolSchema]>) -> String {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    "caliche-prompt-cache-v1".hash(&mut hasher);
    for message in messages {
        if message.role == MessageRole::System {
            "system".hash(&mut hasher);
            message.content.as_text().unwrap_or("").hash(&mut hasher);
        }
    }
    if let Some(tools) = tools {
        for tool in tools {
            tool.name.as_ref().hash(&mut hasher);
        }
    }
    format!("caliche-{:016x}", hasher.finish())
}

/// Map Anthropic usage, including prompt-cache counters.
///
/// Anthropic `input_tokens` are uncached. Prompt size for context display is
/// uncached + cache write + cache read.
#[must_use]
pub fn token_usage_from_anthropic(
    input_tokens: u32,
    output_tokens: u32,
    cache_creation: Option<u32>,
    cache_read: Option<u32>,
) -> TokenUsage {
    let creation = cache_creation.unwrap_or(0);
    let read = cache_read.unwrap_or(0);
    let prompt = input_tokens.saturating_add(creation).saturating_add(read);
    TokenUsage::from_counts(
        Some(prompt),
        Some(output_tokens),
        Some(prompt.saturating_add(output_tokens)),
    )
    .with_cache_tokens(cache_creation, cache_read)
}

/// Attach OpenAI `prompt_tokens_details.cached_tokens` onto usage.
#[must_use]
pub fn token_usage_with_openai_cached(usage: TokenUsage, cached_tokens: Option<u32>) -> TokenUsage {
    usage.with_cache_tokens(None, cached_tokens)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::messages::{Message, ParametersSchema};
    use beef::lean::Cow;
    use std::sync::Arc;

    fn tool(name: &str) -> ToolSchema {
        ToolSchema {
            name: Cow::owned(name.to_string()),
            description: Cow::borrowed("d"),
            parameters: ParametersSchema {
                schema_type: Cow::borrowed("object"),
                properties: serde_json::json!({}),
                required: vec![],
            },
        }
    }

    #[test]
    fn openai_cache_key_is_stable_for_same_prefix() {
        let messages = vec![Message::system("sys"), Message::user("turn-1")];
        let tools = vec![tool("fs"), tool("bash")];
        let a = openai_prompt_cache_key(&messages, Some(&tools));
        let b = openai_prompt_cache_key(
            &[Message::system("sys"), Message::user("turn-2")],
            Some(&tools),
        );
        assert_eq!(a, b);
        assert!(a.starts_with("caliche-"));
        assert_eq!(a.len(), 24);
    }

    #[test]
    fn openai_cache_key_changes_when_tools_change() {
        let messages = [Message::system("sys")];
        let a = openai_prompt_cache_key(&messages, Some(&[tool("fs")]));
        let b = openai_prompt_cache_key(&messages, Some(&[tool("bash")]));
        assert_ne!(a, b);
    }

    #[test]
    fn anthropic_usage_sums_cache_into_prompt_tokens() {
        let usage = token_usage_from_anthropic(10, 4, Some(20), Some(100));
        assert_eq!(usage.prompt_tokens, Some(130));
        assert_eq!(usage.completion_tokens, Some(4));
        assert_eq!(usage.cache_creation_tokens, Some(20));
        assert_eq!(usage.cache_read_tokens, Some(100));
        assert_eq!(usage.total_tokens, Some(134));
    }

    #[test]
    fn openai_cached_tokens_map_to_cache_read() {
        let usage = token_usage_with_openai_cached(
            TokenUsage::from_counts(Some(50), Some(8), Some(58)),
            Some(40),
        );
        assert_eq!(usage.prompt_tokens, Some(50));
        assert_eq!(usage.cache_read_tokens, Some(40));
        assert_eq!(usage.cache_creation_tokens, None);
    }

    #[test]
    fn ephemeral_cache_control_is_anthropic_shape() {
        let value = ephemeral_cache_control();
        assert_eq!(value["type"], "ephemeral");
        let _unused = Arc::new(value);
    }

    #[test]
    fn prompt_cache_report_classifies_hit_and_uncached() {
        let usage = token_usage_from_anthropic(10, 4, Some(20), Some(100));
        let report = PromptCacheReport::from_usage(&usage);
        assert_eq!(report.prompt_tokens, 130);
        assert_eq!(report.cache_read, 100);
        assert_eq!(report.cache_write, 20);
        assert_eq!(report.uncached_tokens(), 10);
        assert_eq!(report.outcome(), PromptCacheOutcome::Mixed);
        let hit = report.hit_ratio().expect("prompt tokens");
        assert!((hit - 100.0 / 130.0).abs() < 1e-9);
        assert!(report.saw_cache());
    }

    #[test]
    fn prompt_cache_report_miss_when_no_cache_counters() {
        let usage = TokenUsage::from_counts(Some(40), Some(8), Some(48));
        let report = PromptCacheReport::from_usage(&usage);
        assert_eq!(report.outcome(), PromptCacheOutcome::Miss);
        assert_eq!(report.uncached_tokens(), 40);
        assert!(!report.saw_cache());
        assert_eq!(report.hit_pct(), Some(0.0));
    }
}
