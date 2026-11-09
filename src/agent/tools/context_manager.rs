//! Context window management for RAG operations.

use crate::llm::ProviderType;
#[cfg(feature = "tools-rag")]
use dripstone::SearchResult;
use serde::{Deserialize, Serialize};

/// Context window statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextStats {
    pub total_chunks: usize,
    pub included_chunks: usize,
    pub omitted_chunks: usize,
    pub total_tokens: usize,
    pub available_tokens: usize,
    pub truncated: bool,
}

/// Context manager for fitting chunks within model context limits
pub struct ContextManager {
    max_tokens: usize,
    reserve_for_completion: usize,
}

impl ContextManager {
    /// Create a new context manager
    #[must_use]
    pub const fn new(max_tokens: usize, reserve_for_completion: usize) -> Self {
        Self {
            max_tokens,
            reserve_for_completion,
        }
    }

    /// Create from provider type (auto-detects context limits)
    #[must_use]
    pub fn from_provider(provider: ProviderType, model: Option<&str>) -> Self {
        let (max_tokens, reserve) = get_provider_context_limits(provider, model);
        Self::new(max_tokens, reserve)
    }

    /// Fit search results within the available context window
    /// Returns (`fitted_results`, stats)
    #[cfg(feature = "tools-rag")]
    #[must_use]
    pub fn fit_search_results(
        &self,
        results: Vec<SearchResult>,
        system_prompt_tokens: usize,
        query_tokens: usize,
    ) -> (Vec<SearchResult>, ContextStats) {
        let available_tokens = self
            .max_tokens
            .saturating_sub(system_prompt_tokens)
            .saturating_sub(query_tokens)
            .saturating_sub(self.reserve_for_completion);

        if available_tokens == 0 {
            return (
                Vec::new(),
                ContextStats {
                    total_chunks: results.len(),
                    included_chunks: 0,
                    omitted_chunks: results.len(),
                    total_tokens: 0,
                    available_tokens: 0,
                    truncated: true,
                },
            );
        }

        // Sort by relevance (highest score first)
        let total_chunks = results.len();
        let mut sorted_results = results;
        sorted_results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let mut fitted_results = Vec::new();
        let mut total_tokens = 0;

        for mut result in sorted_results {
            let chunk_tokens = estimate_search_result_tokens(&result);

            if total_tokens + chunk_tokens <= available_tokens {
                fitted_results.push(result);
                total_tokens += chunk_tokens;
            } else {
                // Truncate the content of the first chunk that did not fit so we
                // still surface partial context instead of silently dropping it.
                if fitted_results.is_empty() && chunk_tokens > 0 {
                    let remaining_tokens = available_tokens.saturating_sub(total_tokens);
                    if remaining_tokens > 100 {
                        // Approximate byte budget via 4 chars/token (matches the
                        // estimator) and cut on a char boundary to avoid splitting
                        // UTF-8 code points.
                        const CHARS_PER_TOKEN: usize = 4;
                        let char_budget = remaining_tokens.saturating_mul(CHARS_PER_TOKEN);
                        let content = &result.chunk.content;
                        if content.len() > char_budget {
                            let mut boundary = char_budget.min(content.len());
                            while boundary > 0 && !content.is_char_boundary(boundary) {
                                boundary -= 1;
                            }
                            let truncated = format!(
                                "{}...[truncated for context]",
                                &content[..boundary].trim_end()
                            );
                            result.chunk.content = truncated;
                        }
                        total_tokens = total_tokens.saturating_add(remaining_tokens);
                        fitted_results.push(result);
                    }
                }
                break;
            }
        }

        let omitted = total_chunks - fitted_results.len();
        let truncated = omitted > 0;
        let included_count = fitted_results.len();

        (
            fitted_results,
            ContextStats {
                total_chunks,
                included_chunks: included_count,
                omitted_chunks: omitted,
                total_tokens,
                available_tokens,
                truncated,
            },
        )
    }

    /// Get available tokens after accounting for system prompt and query
    #[must_use]
    pub const fn available_tokens(
        &self,
        system_prompt_tokens: usize,
        query_tokens: usize,
    ) -> usize {
        self.max_tokens
            .saturating_sub(system_prompt_tokens)
            .saturating_sub(query_tokens)
            .saturating_sub(self.reserve_for_completion)
    }
}

/// Estimate tokens in a search result chunk
#[cfg(feature = "tools-rag")]
fn estimate_search_result_tokens(result: &SearchResult) -> usize {
    // Use the chunk content directly
    estimate_tokens(&result.chunk.content)
}

/// Rough token estimation (1 token ~ 4 characters for English)
#[cfg(feature = "tools-rag")]
const fn estimate_tokens(text: &str) -> usize {
    (text.len().saturating_add(3)) / 4
}

/// Get context limits for a provider/model combination.
/// These are hardcoded fallbacks; prefer `Provider::context_window()` when available.
#[must_use]
pub fn get_provider_context_limits(provider: ProviderType, model: Option<&str>) -> (usize, usize) {
    match provider {
        ProviderType::OpenAI => openai_context_limits(model.unwrap_or("gpt-4o")),
        ProviderType::Anthropic => anthropic_context_limits(model.unwrap_or("claude-sonnet-4")),
        ProviderType::Bedrock => bedrock_context_limits(model.unwrap_or("")),
        ProviderType::OpenRouter => openrouter_context_limits(model.unwrap_or("")),
        ProviderType::Perplexity => (128_000, 4_000),
        ProviderType::Google => google_context_limits(model.unwrap_or("")),
        ProviderType::LMStudio => (32_768, 2_000),
        ProviderType::Ollama => (32_768, 2_000),
        ProviderType::Cohere => (128_000, 4_000),
        #[cfg(feature = "azure-openai")]
        ProviderType::AzureOpenAI => azure_openai_context_limits(model.unwrap_or("gpt-4o")),
        #[cfg(not(feature = "azure-openai"))]
        ProviderType::AzureOpenAI => (128_000, 16_000),
        #[cfg(feature = "local-inference-base")]
        ProviderType::LocalInference => (8_192, 1_000),
        ProviderType::External => (128_000, 4_000),
        ProviderType::CloudflareAIGateway => (128_000, 4_000),
    }
}

fn openai_context_limits(model: &str) -> (usize, usize) {
    if model.contains("gpt-4.1") {
        (1_000_000, 32_000)
    } else if model.contains("gpt-4o") || model.contains("gpt-4.5") {
        (128_000, 16_000)
    } else if model.starts_with("o3") || model.starts_with("o4") || model.starts_with("o1") {
        (200_000, 100_000)
    } else if model.contains("gpt-4-turbo") {
        (128_000, 4_000)
    } else if model.contains("gpt-4") {
        (8_192, 1_000)
    } else if model.contains("gpt-3.5") {
        (16_385, 1_000)
    } else {
        (128_000, 16_000)
    }
}

fn anthropic_context_limits(model: &str) -> (usize, usize) {
    if model.contains("claude-2") {
        (100_000, 4_000)
    } else {
        (200_000, 8_000)
    }
}

fn bedrock_context_limits(model: &str) -> (usize, usize) {
    if model.contains("claude") {
        (200_000, 8_000)
    } else if model.contains("llama") {
        (128_000, 4_000)
    } else if model.contains("nova") {
        (300_000, 5_000)
    } else {
        (128_000, 4_000)
    }
}

fn openrouter_context_limits(model: &str) -> (usize, usize) {
    if model.contains("claude") {
        (200_000, 8_000)
    } else if model.contains("gemini") {
        (1_000_000, 8_000)
    } else if model.contains("gpt-4.1") {
        (1_000_000, 32_000)
    } else if model.contains("gpt-4o") {
        (128_000, 16_000)
    } else if is_openrouter_o_series(model) {
        (200_000, 100_000)
    } else if model.contains("deepseek") {
        (128_000, 8_000)
    } else if model.contains("llama") || model.contains("qwen") {
        (128_000, 4_000)
    } else {
        (200_000, 8_000)
    }
}

fn is_openrouter_o_series(model: &str) -> bool {
    model.contains("openai/o3")
        || model.contains("openai/o4")
        || model.starts_with("o3")
        || model.starts_with("o4")
}

fn google_context_limits(model: &str) -> (usize, usize) {
    if model.contains("gemini-1.0") {
        (32_000, 2_000)
    } else {
        (1_000_000, 8_000)
    }
}

#[cfg(feature = "azure-openai")]
fn azure_openai_context_limits(model: &str) -> (usize, usize) {
    if model.contains("gpt-4.1") {
        (1_000_000, 32_000)
    } else if model.contains("gpt-4o") {
        (128_000, 16_000)
    } else if model.starts_with("o3") || model.starts_with("o4") || model.starts_with("o1") {
        (200_000, 100_000)
    } else if model.contains("gpt-4") {
        (8_192, 1_000)
    } else if model.contains("gpt-3.5") {
        (16_385, 1_000)
    } else {
        (128_000, 16_000)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg(feature = "tools-rag")]
    fn test_context_manager() {
        use dripstone::{Chunk, SearchResult};

        let manager = ContextManager::new(4_000, 500);

        // Create mock search results
        let results: Vec<SearchResult> = (0..10)
            .map(|i| {
                let text = "x".repeat(200); // ~50 tokens each
                let chunk = Chunk::new(
                    format!("doc_{}", i),
                    i,
                    text,
                    vec![0.1; 128],
                    0,
                    200,
                    10,
                    serde_json::json!({}),
                    None,
                );
                SearchResult {
                    chunk,
                    score: 1.0 - (i as f32 * 0.1),
                    rank: i,
                    score_is_authoritative: true,
                }
            })
            .collect();

        let (fitted, stats) = manager.fit_search_results(results, 100, 50);

        // Should fit multiple chunks within 4000 - 100 - 50 - 500 = 3350 tokens
        assert!(!fitted.is_empty());
        assert!(stats.truncated || fitted.len() == 10);
    }

    #[test]
    fn test_provider_limits() {
        let (max, reserve) = get_provider_context_limits(ProviderType::OpenAI, Some("gpt-4o"));
        assert_eq!(max, 128_000);
        assert_eq!(reserve, 16_000);

        let (max, _) = get_provider_context_limits(ProviderType::OpenAI, Some("gpt-4.1"));
        assert_eq!(max, 1_000_000);

        let (max, _) = get_provider_context_limits(ProviderType::OpenAI, Some("o3-mini"));
        assert_eq!(max, 200_000);

        let (max, _) =
            get_provider_context_limits(ProviderType::Anthropic, Some("claude-sonnet-4"));
        assert_eq!(max, 200_000);

        let (max, _) =
            get_provider_context_limits(ProviderType::Google, Some("models/gemini-2.5-pro"));
        assert_eq!(max, 1_000_000);
    }

    #[test]
    fn provider_limits_cover_openrouter_bedrock_and_local_defaults() {
        let (max, reserve) =
            get_provider_context_limits(ProviderType::OpenRouter, Some("openai/o3-mini"));
        assert_eq!((max, reserve), (200_000, 100_000));

        let (max, reserve) = get_provider_context_limits(ProviderType::OpenRouter, Some("o3-mini"));
        assert_eq!((max, reserve), (200_000, 100_000));

        let (max, _) = get_provider_context_limits(ProviderType::Bedrock, Some("amazon.nova-pro"));
        assert_eq!(max, 300_000);

        let (max, _) = get_provider_context_limits(ProviderType::Bedrock, Some(""));
        assert_eq!(max, 128_000);

        let (max, reserve) =
            get_provider_context_limits(ProviderType::CloudflareAIGateway, Some("openai/gpt-4o"));
        assert_eq!((max, reserve), (128_000, 4_000));

        #[cfg(feature = "local-inference-base")]
        {
            let (max, reserve) =
                get_provider_context_limits(ProviderType::LocalInference, Some("tinyllama"));
            assert_eq!((max, reserve), (8_192, 1_000));
        }

        let (max, _) = get_provider_context_limits(ProviderType::OpenAI, Some("chatgpt-4o-latest"));
        assert_eq!(max, 128_000);

        let (max, _) = get_provider_context_limits(ProviderType::OpenAI, Some("gpt-4"));
        assert_eq!(max, 8_192);
    }
}
