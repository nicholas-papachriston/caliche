//! OpenAI-compatible provider abstraction.
//!
//! This module provides a generic implementation for all providers that follow
//! the `OpenAI` chat completions API format (`OpenAI`, `OpenRouter`, `LMStudio`, Ollama, Perplexity).
//!
//! Instead of duplicating ~800 lines of code per provider, we use a single generic
//! `OpenAICompatibleProvider<C>` that's configured via the `OpenAICompatConfig` trait.

mod config;
mod provider;
mod types;

pub mod providers;

pub use config::{AuthHeadersFuture, OpenAICompatConfig};
pub use provider::OpenAICompatibleProvider;
pub use types::*;

// Re-export provider configs and types for convenience
pub use providers::{
    CloudflareAIGatewayConfig, CloudflareAIGatewayConfigBuilder, CloudflareAIGatewayProvider,
    LMStudioConfig, LMStudioConfigBuilder, LMStudioProvider, OllamaConfig, OllamaConfigBuilder,
    OllamaProvider, OpenAIConfig, OpenAIConfigBuilder, OpenAIProvider, OpenRouterConfig,
    OpenRouterConfigBuilder, OpenRouterProvider, PerplexityConfig, PerplexityConfigBuilder,
    PerplexityProvider, normalize_cloudflare_ai_gateway_model,
    validate_cloudflare_ai_gateway_model,
};
