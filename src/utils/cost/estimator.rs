//! Cost estimation engine for predicting LLM and tool execution costs.

use crate::llm::ProviderType;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Provider-specific pricing per 1K tokens
#[derive(Debug, Clone)]
struct Pricing {
    prompt_price_per_1k: f64,
    completion_price_per_1k: f64,
    embedding_price_per_1k: f64,
}

impl Pricing {
    fn new(prompt: f64, completion: f64, embedding: f64) -> Self {
        Self {
            prompt_price_per_1k: prompt,
            completion_price_per_1k: completion,
            embedding_price_per_1k: embedding,
        }
    }
}

/// Cost estimation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CostEstimate {
    pub estimated_cost_usd: f64,
    pub breakdown: CostBreakdown,
    pub confidence: ConfidenceLevel,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CostBreakdown {
    pub prompt_tokens: u64,
    pub estimated_completion_tokens: u64,
    pub embedding_calls: u64,
    pub tool_executions: HashMap<String, u64>,
    pub prompt_cost_usd: f64,
    pub completion_cost_usd: f64,
    pub embedding_cost_usd: f64,
    pub tool_cost_usd: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ConfidenceLevel {
    Low,
    Medium,
    High,
}

/// Cost estimator with provider pricing tables
pub struct CostEstimator {
    provider_pricing: HashMap<ProviderType, Pricing>,
}

impl CostEstimator {
    /// Create a new cost estimator with default pricing tables
    pub fn new() -> Self {
        let mut pricing = HashMap::new();

        // OpenAI pricing (as of 2024, approximate)
        pricing.insert(
            ProviderType::OpenAI,
            Pricing::new(0.005, 0.015, 0.0001), // gpt-4o: $5/$15 per 1M tokens, embeddings: $0.10 per 1M
        );

        // Anthropic pricing
        pricing.insert(
            ProviderType::Anthropic,
            Pricing::new(0.003, 0.015, 0.0001), // claude-3-5-sonnet: $3/$15 per 1M tokens
        );

        // AWS Bedrock pricing (Claude models)
        pricing.insert(
            ProviderType::Bedrock,
            Pricing::new(0.003, 0.015, 0.0001), // Similar to Anthropic
        );

        // OpenRouter pricing (varies by model, using average)
        pricing.insert(
            ProviderType::OpenRouter,
            Pricing::new(0.002, 0.006, 0.0001), // Average across models
        );

        // Perplexity pricing
        pricing.insert(
            ProviderType::Perplexity,
            Pricing::new(0.0007, 0.0007, 0.0001), // Sonar models
        );

        // Google Gemini pricing
        pricing.insert(
            ProviderType::Google,
            Pricing::new(0.000125, 0.0005, 0.0001), // gemini-2.5-flash: $0.125/$0.50 per 1M tokens
        );

        // Local providers (no cloud billing).
        pricing.insert(ProviderType::LMStudio, Pricing::new(0.0, 0.0, 0.0));
        pricing.insert(ProviderType::Ollama, Pricing::new(0.0, 0.0, 0.0));
        #[cfg(feature = "local-inference-base")]
        pricing.insert(ProviderType::LocalInference, Pricing::new(0.0, 0.0, 0.0));

        // Cloudflare AI Gateway (proxies upstream providers; varies by model)
        pricing.insert(
            ProviderType::CloudflareAIGateway,
            Pricing::new(0.002, 0.006, 0.0001),
        );

        Self {
            provider_pricing: pricing,
        }
    }

    /// Estimate cost for a chat request
    pub fn estimate_chat(
        &self,
        message: &str,
        provider: ProviderType,
        _model: Option<&str>,
        context_size: usize,
        tools: &[String],
    ) -> CostEstimate {
        // Estimate token counts (rough approximation: 1 token ≈ 4 characters)
        let message_tokens = estimate_tokens(message);
        let system_prompt_tokens = 100; // Typical system prompt
        let tools_tokens = estimate_tool_tokens(tools);

        let prompt_tokens =
            (message_tokens + system_prompt_tokens + tools_tokens + context_size) as u64;

        // Estimate completion tokens (typically 1-2x prompt tokens for responses)
        let estimated_completion_tokens = (prompt_tokens as f64 * 1.5) as u64;

        // Estimate tool usage
        let mut tool_executions = HashMap::new();
        let mut embedding_calls = 0u64;

        for tool in tools {
            if tool == "rag" {
                // RAG typically uses 1-2 embedding calls per query
                embedding_calls += 2;
                tool_executions.insert("rag".to_string(), 1);
            } else {
                tool_executions.insert(tool.clone(), 1);
            }
        }

        // Get pricing
        let pricing = self
            .provider_pricing
            .get(&provider)
            .cloned()
            .unwrap_or_else(|| Pricing::new(0.001, 0.002, 0.0001)); // Default fallback

        // Calculate costs
        let prompt_cost = (prompt_tokens as f64 / 1000.0) * pricing.prompt_price_per_1k;
        let completion_cost =
            (estimated_completion_tokens as f64 / 1000.0) * pricing.completion_price_per_1k;
        let embedding_cost = (embedding_calls as f64 / 1000.0) * pricing.embedding_price_per_1k;
        let tool_cost = 0.0; // Tools are typically free (API calls, etc.)

        let total_cost = prompt_cost + completion_cost + embedding_cost + tool_cost;

        // Determine confidence level
        let confidence = if tools.is_empty() && context_size == 0 {
            ConfidenceLevel::High
        } else if tools.len() <= 2 {
            ConfidenceLevel::Medium
        } else {
            ConfidenceLevel::Low
        };

        CostEstimate {
            estimated_cost_usd: total_cost,
            breakdown: CostBreakdown {
                prompt_tokens,
                estimated_completion_tokens,
                embedding_calls,
                tool_executions,
                prompt_cost_usd: prompt_cost,
                completion_cost_usd: completion_cost,
                embedding_cost_usd: embedding_cost,
                tool_cost_usd: tool_cost,
            },
            confidence,
        }
    }

    /// Estimate cost for embedding generation
    pub fn estimate_embeddings(&self, provider: ProviderType, text_count: usize) -> f64 {
        let pricing = self
            .provider_pricing
            .get(&provider)
            .cloned()
            .unwrap_or_else(|| Pricing::new(0.0, 0.0, 0.0001));

        // Estimate tokens: ~75 tokens per 100 characters average
        let estimated_tokens = text_count * 75;
        (estimated_tokens as f64 / 1000.0) * pricing.embedding_price_per_1k
    }
}

impl Default for CostEstimator {
    fn default() -> Self {
        Self::new()
    }
}

/// Rough token estimation (1 token ≈ 4 characters for English)
fn estimate_tokens(text: &str) -> usize {
    (text.len().saturating_add(3)) / 4
}

/// Estimate tokens for tool schemas
fn estimate_tool_tokens(tools: &[String]) -> usize {
    // Rough estimate: ~200 tokens per tool schema
    tools.len() * 200
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cost_estimation() {
        let estimator = CostEstimator::new();

        let estimate = estimator.estimate_chat(
            "What is the weather today?",
            ProviderType::OpenAI,
            None,
            0,
            &[],
        );

        assert!(estimate.estimated_cost_usd > 0.0);
        assert!(estimate.breakdown.prompt_tokens > 0);
        assert!(estimate.breakdown.estimated_completion_tokens > 0);
    }

    #[test]
    fn test_embedding_cost() {
        let estimator = CostEstimator::new();
        let cost = estimator.estimate_embeddings(ProviderType::OpenAI, 10);
        assert!(cost >= 0.0);
    }

    #[test]
    fn local_providers_estimate_zero_cost() {
        let estimator = CostEstimator::new();
        for provider in [
            ProviderType::Ollama,
            ProviderType::LMStudio,
            #[cfg(feature = "local-inference-base")]
            ProviderType::LocalInference,
        ] {
            let estimate = estimator.estimate_chat("hello local model", provider, None, 128, &[]);
            assert_eq!(
                estimate.estimated_cost_usd, 0.0,
                "expected zero cost for {provider}"
            );
        }
    }
}
