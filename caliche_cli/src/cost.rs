//! Cost estimation for LLM API usage.
//!
//! Provides pricing data and cost calculation for different providers and models.

pub use caliche::cost::{ModelPricing, get_pricing};

/// Format token count with K/M suffix
pub fn format_tokens(tokens: u64) -> String {
    if tokens >= 1_000_000 {
        format!("{:.1}M", tokens as f64 / 1_000_000.0)
    } else if tokens >= 1_000 {
        format!("{:.1}K", tokens as f64 / 1_000.0)
    } else {
        format!("{}", tokens)
    }
}

/// Get context window size for a model (in tokens).
/// This is a hardcoded fallback table; prefer `Provider::context_window()` when available.
pub fn get_context_window(provider: &str, model: &str) -> u64 {
    let provider = provider.to_lowercase();
    let model = model.to_lowercase();
    match provider.as_str() {
        "openai" => openai_context_window(&model),
        "anthropic" | "google" | "bedrock" => family_context_window(&provider, &model),
        _ => other_context_window(&provider, &model),
    }
}

fn openai_context_window(model: &str) -> u64 {
    if model.contains("gpt-4.1") {
        1_000_000 // GPT-4.1 / 4.1-mini / 4.1-nano: 1M context
    } else if model.starts_with("o3") || model.starts_with("o4") {
        200_000 // o3 / o4-mini: 200K context
    } else if model.starts_with("o1") {
        200_000
    } else if model.contains("gpt-4o") {
        128_000
    } else if model.contains("gpt-4.5") {
        128_000
    } else if model.contains("gpt-4-turbo") {
        128_000
    } else if model.contains("gpt-4") {
        8_192
    } else if model.contains("gpt-3.5-turbo-16k") {
        16_384
    } else if model.contains("gpt-3.5") {
        16_385
    } else {
        128_000
    }
}

fn family_context_window(provider: &str, model: &str) -> u64 {
    match provider {
        "anthropic" => {
            if model.contains("claude-2") {
                100_000
            } else {
                // Claude 3.x, 4.x, Sonnet 4, Opus 4, Haiku 4.5: 200K standard
                200_000
            }
        }
        "google" => {
            if model.contains("gemini-1.0") {
                32_000
            } else {
                // Gemini 1.5, 2.0, 2.5 (Pro & Flash): 1M context
                1_000_000
            }
        }
        "bedrock" => bedrock_context_window(model),
        _ => 128_000,
    }
}

fn bedrock_context_window(model: &str) -> u64 {
    if model.contains("claude") {
        200_000
    } else if model.contains("llama") {
        128_000
    } else if model.contains("nova") {
        300_000
    } else {
        128_000
    }
}

fn other_context_window(provider: &str, model: &str) -> u64 {
    match provider {
        "openrouter" => openrouter_context_window(model),
        "perplexity" => 128_000,
        "lmstudio" => 32_000,
        "ollama" => 32_000,
        "deepseek" => 128_000,
        "mistral" => {
            if model.contains("large") {
                128_000
            } else {
                32_000
            }
        }
        "cohere" => 128_000,
        _ => 128_000,
    }
}

fn openrouter_context_window(model: &str) -> u64 {
    if model.contains("claude") {
        200_000
    } else if model.contains("gemini") {
        1_000_000
    } else if model.contains("gpt-4.1") {
        1_000_000
    } else if model.contains("gpt-4o") {
        128_000
    } else if model.contains("/o3") || model.contains("/o4") {
        200_000
    } else if model.contains("deepseek") {
        128_000
    } else if model.contains("llama") || model.contains("qwen") {
        128_000
    } else {
        200_000
    }
}

/// Warning level for token budget
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BudgetWarningLevel {
    /// Usage is normal (<70%)
    Normal,
    /// Usage is high (70-85%)
    Warning,
    /// Usage is critical (>85%)
    Critical,
}

/// Token budget tracker for interactive sessions
#[derive(Debug, Clone)]
pub struct TokenBudget {
    /// Maximum context window
    pub max_tokens: u64,
    /// Current used tokens
    pub used_tokens: u64,
    /// Provider name
    pub provider: String,
    /// Model name
    pub model: String,
}

impl TokenBudget {
    /// Create a new token budget tracker
    pub fn new(provider: &str, model: &str) -> Self {
        let max_tokens = get_context_window(provider, model);
        Self {
            max_tokens,
            used_tokens: 0,
            provider: provider.to_string(),
            model: model.to_string(),
        }
    }

    /// Override displayed/tracked context window size.
    pub fn with_max_tokens(mut self, max_tokens: u64) -> Self {
        self.max_tokens = max_tokens;
        self
    }

    /// Add token usage
    pub fn add_usage(&mut self, tokens: u64) {
        self.used_tokens += tokens;
    }

    /// Set used tokens directly
    pub fn set_used(&mut self, tokens: u64) {
        self.used_tokens = tokens;
    }

    /// Reset usage
    pub fn reset(&mut self) {
        self.used_tokens = 0;
    }

    /// Get usage percentage
    pub fn usage_percent(&self) -> f64 {
        if self.max_tokens == 0 {
            return 0.0;
        }
        (self.used_tokens as f64 / self.max_tokens as f64) * 100.0
    }

    /// Get warning level based on usage
    pub fn warning_level(&self) -> BudgetWarningLevel {
        let percent = self.usage_percent();
        if percent >= 85.0 {
            BudgetWarningLevel::Critical
        } else if percent >= 70.0 {
            BudgetWarningLevel::Warning
        } else {
            BudgetWarningLevel::Normal
        }
    }

    /// Check if approaching limit (>70%)
    pub fn is_warning(&self) -> bool {
        self.usage_percent() >= 70.0
    }

    /// Check if at critical level (>85%)
    pub fn is_critical(&self) -> bool {
        self.usage_percent() >= 85.0
    }

    /// Check if compression should be suggested
    pub fn should_suggest_compress(&self) -> bool {
        self.usage_percent() >= 70.0 && self.used_tokens > 5000
    }

    /// Get remaining tokens
    pub fn remaining(&self) -> u64 {
        self.max_tokens.saturating_sub(self.used_tokens)
    }

    /// Format as a display string for prompts (context tokens only)
    pub fn format_short(&self) -> String {
        let context = format_tokens(self.used_tokens);
        format!("{} context", context)
    }

    /// Format with session tokens for display
    pub fn format_with_session(&self, session_tokens: u64) -> String {
        let session = format_tokens(session_tokens);
        let context = format_tokens(self.used_tokens);
        format!("{} session | {} context", session, context)
    }

    /// Format warning message if appropriate
    /// Note: Compression is now automatic, so no manual action is suggested
    pub fn warning_message(&self) -> Option<String> {
        // Compression is automatic - no warnings needed
        None
    }

    /// Format as a progress bar
    pub fn format_bar(&self, width: usize) -> String {
        let filled = ((self.usage_percent() / 100.0) * width as f64) as usize;
        let empty = width.saturating_sub(filled);

        let color = if self.is_critical() {
            "\x1b[31m"
        } else if self.is_warning() {
            "\x1b[33m"
        } else {
            "\x1b[32m"
        };

        format!(
            "{}[{}{}]\x1b[0m",
            color,
            "=".repeat(filled),
            " ".repeat(empty)
        )
    }
}

impl std::fmt::Display for TokenBudget {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.format_short())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pricing_calculation() {
        let pricing = ModelPricing::new(3.0, 15.0); // Claude 3.5 Sonnet pricing

        // 1000 input + 500 output tokens
        let cost = pricing.calculate(1000, 500);
        assert!((cost - 0.0105).abs() < 0.0001);
    }

    #[test]
    fn test_format_tokens() {
        assert_eq!(format_tokens(500), "500");
        assert_eq!(format_tokens(1500), "1.5K");
        assert_eq!(format_tokens(1_500_000), "1.5M");
    }

    #[test]
    fn test_context_window() {
        assert_eq!(
            get_context_window("anthropic", "claude-3-5-sonnet"),
            200_000
        );
        assert_eq!(get_context_window("openai", "gpt-4o"), 128_000);
    }

    #[test]
    fn test_token_budget() {
        let mut budget = TokenBudget::new("anthropic", "claude-3-5-sonnet");
        assert_eq!(budget.max_tokens, 200_000);

        budget.add_usage(50_000);
        assert!((budget.usage_percent() - 25.0).abs() < 0.1);
        assert!(!budget.is_warning());
        assert_eq!(budget.warning_level(), BudgetWarningLevel::Normal);

        // At 70%, should be warning
        budget.set_used(140_000);
        assert!(budget.is_warning());
        assert!(!budget.is_critical());
        assert_eq!(budget.warning_level(), BudgetWarningLevel::Warning);

        // At 85%, should be critical
        budget.set_used(170_000);
        assert!(budget.is_critical());
        assert_eq!(budget.warning_level(), BudgetWarningLevel::Critical);
    }

    #[test]
    fn test_warning_messages() {
        let mut budget = TokenBudget::new("anthropic", "claude-3-5-sonnet");

        // Warnings are intentionally suppressed because compression is automatic.
        budget.set_used(50_000);
        assert!(budget.warning_message().is_none());
        budget.set_used(150_000);
        assert!(budget.warning_message().is_none());
        budget.set_used(180_000);
        assert!(budget.warning_message().is_none());
    }
}
