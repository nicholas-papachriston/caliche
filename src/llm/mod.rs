pub mod anthropic;
#[cfg(feature = "failover")]
pub mod auth_profiles;
#[cfg(feature = "azure-openai")]
pub mod azure_openai;
#[cfg(feature = "bedrock")]
pub mod bedrock;
pub mod cache;
pub mod caching_provider;
pub mod catalog;
pub mod client;
#[cfg(feature = "cohere")]
pub mod cohere;
pub mod config;
pub mod embedding;
pub mod env_compat;
pub mod external;
#[cfg(feature = "failover")]
pub mod failover;
pub mod google;
pub mod image_generation;
#[cfg(any(feature = "embedding-candle", feature = "embedding-ort"))]
pub mod local;
#[cfg(feature = "local-inference-base")]
pub mod local_inference;
pub mod middleware;
pub mod models;
pub mod provider;
pub mod provider_factory;
pub mod provider_utils;
pub mod retry;
pub mod streaming;
pub mod unconfigured;
#[cfg(any(feature = "embedding-candle", feature = "embedding-ort"))]
pub mod vision;

pub mod openai_compat;
pub mod prompt_cache;

/// `HTTP-Referer` value sent on OpenRouter requests.
pub const OPENROUTER_HTTP_REFERER: &str = "https://github.com/nicholas-papachriston/caliche";

/// `X-Title` value sent on OpenRouter requests.
pub const OPENROUTER_X_TITLE: &str = "Caliche";
#[cfg(feature = "router")]
pub mod router;

pub use anthropic::AnthropicProvider;
#[cfg(feature = "azure-openai")]
pub use azure_openai::AzureOpenAIProvider;
#[cfg(feature = "bedrock")]
pub use bedrock::BedrockProvider;
pub use caching_provider::CachingProvider;
pub use catalog::{
    ModelCatalogEntry, ModelCatalogOptions, ProviderCatalogEntry, compiled_provider_types,
    discover_provider_catalog, discover_provider_models, flatten_catalog_models,
};
#[cfg(feature = "cohere")]
pub use cohere::CohereProvider;
#[cfg(feature = "local-inference-base")]
pub use dripstone::inference::{GenerationParams, InferenceBackendType};
#[cfg(feature = "local-inference-base")]
pub use dripstone::loader::{FitOptions, FitStatus, HostCapabilities, ModelFit, ModelResourceHint};
pub use embedding::{
    EmbeddingProvider, EmbeddingRequest, EmbeddingResponse, EmbeddingStopped, StopOnPermanentError,
    is_permanent_embedding_error,
};
pub use env_compat::{env_flag, env_nonempty, env_var_nonempty};
pub use external::{
    ExternalCompletionCallback, ExternalEmbeddingCallback, ExternalEmbeddingProvider,
    ExternalProvider,
};
pub use google::GoogleProvider;
pub use image_generation::{
    GeneratedImage, GoogleImageGenerationProvider, ImageGenerationProvider, ImageGenerationRequest,
    ImageGenerationResponse, ImageOutputFormat, OpenAIChatImageGenerationProvider,
    OpenAIImageGenerationProvider, default_image_generation_model,
    discover_image_generation_models, fallback_image_generation_models,
    resolve_image_generation_model,
};
#[cfg(feature = "embedding-candle")]
pub use local::{LocalCandleProvider, LocalCandleProviderConfig};
#[cfg(feature = "embedding-ort")]
pub use local::{LocalOrtProvider, LocalOrtProviderConfig};
#[cfg(feature = "local-inference-base")]
pub use local_inference::{
    CandleBootstrapConfig, GgufBootstrapConfig, LazyLocalInferenceProvider,
    LocalInferenceBootstrapConfig, LocalInferenceConfig, LocalInferenceProvider,
    OrtBootstrapConfig,
};
pub use middleware::{MiddlewareProvider, ProviderMiddleware};
pub use models::{
    AnthropicModel, AzureOpenAIModel, BedrockModel, CloudflareAIGatewayModel, CohereModel,
    GoogleModel, LMStudioModel, ModelProvider, OllamaModel, OpenAIModel, OpenRouterModel,
    PerplexityModel, ProviderType, defaults,
};
pub use provider::{Provider, UsageMergePolicy};
pub use provider_factory::{ProviderFactoryFn, default_provider_factory};
pub use retry::RetryError;
#[cfg(feature = "embedding-candle")]
pub use vision::local::{LocalVisionCandleProvider, LocalVisionCandleProviderConfig};
#[cfg(feature = "embedding-ort")]
pub use vision::local::{LocalVisionOrtProvider, LocalVisionOrtProviderConfig};
#[cfg(any(feature = "embedding-candle", feature = "embedding-ort"))]
pub use vision::{
    VideoEmbeddingRequest, VisionEmbeddingProvider, VisionEmbeddingRequest, VisionEmbeddingResponse,
};

pub use openai_compat::providers::{
    CloudflareAIGatewayProvider, LMStudioProvider, OllamaProvider, OpenAIProvider,
    OpenRouterProvider, PerplexityProvider,
};

pub use openai_compat::{
    CloudflareAIGatewayConfig, CloudflareAIGatewayConfigBuilder, LMStudioConfig,
    LMStudioConfigBuilder, OllamaConfig, OllamaConfigBuilder, OpenAICompatConfig,
    OpenAICompatibleProvider, OpenAIConfig, OpenAIConfigBuilder, OpenRouterConfig,
    OpenRouterConfigBuilder, PerplexityConfig, PerplexityConfigBuilder,
};

#[cfg(feature = "failover")]
pub use auth_profiles::{AuthProfile, AuthProfileManager, RotationStrategy};
pub use client::{LlmClient, LlmClientBuilder};
#[cfg(feature = "failover")]
pub use failover::{
    FailoverConfig, FailoverConfigBuilder, FailoverEntry, FailoverEvent, FailoverProvider,
    FailoverReason, ProviderModelPair,
};
