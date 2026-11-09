//! Configuration trait for OpenAI-compatible providers.
//!
//! This trait defines the customization points for providers that follow
//! the `OpenAI` chat completions API format.

use beef::Cow;
use std::future::Future;
use std::pin::Pin;
use std::time::Duration;

use super::types::OpenAICompatRequest;
use crate::messages::ToolChoice;

/// Future resolving HTTP authorization header key/value pairs.
pub type AuthHeadersFuture<'a> =
    Pin<Box<dyn Future<Output = Vec<(&'static str, String)>> + Send + 'a>>;

/// Trait for customizing OpenAI-compatible provider behavior.
///
/// Implement this trait to create a new provider that uses the `OpenAI`
/// chat completions API format. The generic `OpenAICompatibleProvider<C>`
/// will handle all the common logic.
pub trait OpenAICompatConfig: Send + Sync + Clone + std::fmt::Debug + 'static {
    /// Provider name for logging and identification (e.g., "openai", "openrouter")
    fn provider_name(&self) -> &'static str;

    /// Base URL for the API (e.g., "<https://api.openai.com/v1>")
    fn base_url(&self) -> &str;

    /// Optional override for the chat completions endpoint URL.
    /// When `None`, `{base_url}/chat/completions` is used.
    fn chat_completions_url(&self) -> Option<&str> {
        None
    }

    /// Optional override for the embeddings endpoint URL.
    /// When `None`, `{base_url}/embeddings` is used.
    fn embeddings_url(&self) -> Option<&str> {
        None
    }

    /// Build authorization headers.
    /// Returns a vector of (`header_name`, `header_value`) pairs.
    fn auth_headers(&self) -> Vec<(&'static str, String)>;

    /// Resolve authorization headers asynchronously for providers with expiring tokens.
    fn resolve_auth_headers(&self) -> AuthHeadersFuture<'_> {
        let headers = self.auth_headers();
        Box::pin(async move { headers })
    }

    /// Model identifier to use for requests
    fn model(&self) -> &str;

    /// Whether this provider uses `max_completion_tokens` instead of `max_tokens`.
    /// This is required for newer `OpenAI` models (gpt-4o, o1, o3, gpt-5).
    fn uses_max_completion_tokens(&self) -> bool {
        false
    }

    /// Whether temperature must be left at default (1.0).
    /// This is required for o1, o3, and gpt-5 models.
    fn requires_default_temperature(&self) -> bool {
        false
    }

    /// Provider-specific request modifications.
    /// Override this to add custom fields or modify the request before sending.
    fn customize_request(&self, _request: &mut OpenAICompatRequest) {}

    /// Provider-specific output post-processing.
    /// Override this to clean up model outputs (e.g., `LMStudio` special tokens).
    /// Return `Cow::borrowed(content)` when no change is needed to avoid allocation.
    fn post_process_output<'a>(&self, content: &'a str) -> Cow<'a, str> {
        Cow::borrowed(content)
    }

    /// Whether this provider supports native tool/function calling.
    fn supports_native_tools(&self) -> bool {
        true
    }

    /// Whether this provider emits tool calls inside the response `content`
    /// (text-based JSON) instead of native `tool_calls`. Default: `false`.
    fn supports_text_based_tools(&self) -> bool {
        false
    }

    /// Whether this provider supports vision/multimodal image inputs.
    fn supports_vision(&self) -> bool {
        false
    }

    /// Whether this provider supports native audio input.
    fn supports_audio_input(&self) -> bool {
        false
    }

    /// Whether this provider supports native document input.
    fn supports_document_input(&self) -> bool {
        false
    }

    /// Return media types supported by this provider.
    fn supported_media_types(&self) -> Vec<&str> {
        vec![]
    }

    /// Optionally provide provider-specific error enhancement.
    fn enhance_error(&self, _status: u16, _body: &str) -> Option<String> {
        None
    }

    /// Additional headers for requests (e.g., `OpenRouter`'s X-Title).
    /// Returns a vector of (`header_name`, `header_value`) pairs.
    fn extra_headers(&self) -> Vec<(&'static str, &'static str)> {
        vec![]
    }

    /// Whether to include usage tracking in streaming requests via `stream_options`.
    /// `OpenAI` uses `stream_options`: { `include_usage`: true } for this.
    fn include_usage_in_stream(&self) -> bool {
        false
    }

    /// Whether this provider needs the 'usage' parameter in streaming requests.
    /// Only `OpenRouter` requires this parameter. `OpenAI` uses `stream_options` instead.
    fn needs_usage_parameter(&self) -> bool {
        false
    }

    /// Default embedding model for this provider.
    /// Override if the provider supports embeddings with a specific default model.
    fn default_embedding_model(&self) -> Option<String> {
        None
    }

    /// Whether embeddings are supported by this provider.
    fn supports_embeddings(&self) -> bool {
        false
    }

    /// Convert `tool_choice` to provider-specific format.
    /// Override this if the provider has special requirements (e.g., LM Studio only supports string values).
    /// Default implementation uses standard `OpenAI` format.
    fn convert_tool_choice(&self, tool_choice: &ToolChoice) -> Option<serde_json::Value> {
        crate::llm::provider_utils::convert_tool_choice_openai(tool_choice)
    }

    /// Whether this provider supports `response_format` parameter.
    /// Some providers (like Perplexity) don't support `response_format` and will reject requests with it.
    fn supports_response_format(&self) -> bool {
        true
    }

    /// Wait allowed for the first byte of a streamed reply. `None` (the
    /// default, for cloud providers) applies [`Self::stall_timeout`] to it.
    /// A local server returns a long wait: it may load the model and read
    /// the prompt before its first token. The wait also bounds the request
    /// itself, which then has no whole-request timeout of its own.
    fn first_token_timeout(&self) -> Option<Duration> {
        None
    }

    /// Longest gap between chunks of a streamed reply once bytes flow.
    fn stall_timeout(&self) -> Duration {
        crate::llm::provider_utils::SseLimits::DEFAULT_STALL
    }

    /// How to ask the server whether the model is loaded, for the status
    /// line while the first token is awaited.
    fn model_load_probe(&self) -> Option<crate::llm::provider_utils::first_token::ModelLoadProbe> {
        None
    }

    /// How to load a model the server has not loaded, before a request.
    /// `None` (the default) sends the request as it is.
    fn model_loader(&self) -> Option<crate::llm::provider_utils::model_load::ModelLoader> {
        None
    }
}
