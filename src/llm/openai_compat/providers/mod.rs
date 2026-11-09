//! Provider-specific configurations for OpenAI-compatible APIs.

pub mod cloudflare_ai_gateway;
pub mod lmstudio;
pub mod ollama;
pub mod openai;
pub mod openrouter;
pub mod perplexity;

pub use cloudflare_ai_gateway::{
    CloudflareAIGatewayConfig, CloudflareAIGatewayConfigBuilder, CloudflareAIGatewayProvider,
    normalize_cloudflare_ai_gateway_model, validate_cloudflare_ai_gateway_model,
};
pub use lmstudio::{LMStudioConfig, LMStudioConfigBuilder, LMStudioProvider};
pub use ollama::{OllamaConfig, OllamaConfigBuilder, OllamaProvider};
pub use openai::{OpenAIConfig, OpenAIConfigBuilder, OpenAIProvider};
pub use openrouter::{OpenRouterConfig, OpenRouterConfigBuilder, OpenRouterProvider};
pub use perplexity::{PerplexityConfig, PerplexityConfigBuilder, PerplexityProvider};
