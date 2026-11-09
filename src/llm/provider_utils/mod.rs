//! Utilities shared across LLM providers.
pub mod anthropic_stream;
pub mod auth;
pub mod cohere_stream;
pub mod content_fallback;
pub mod content_part;
pub mod first_token;
pub mod google_stream;
pub mod logging;
pub mod macros;
pub mod model_discovery;
pub mod model_load;
pub mod openai_compat;
pub mod openai_stream;
pub mod response_format;
pub mod response_parser;
pub mod serde_helpers;
pub mod setup;
pub mod streaming;
pub mod streaming_request;
pub mod tools;

pub use auth::{
    looks_like_openrouter_api_key, read_api_key_from_env, read_base_url_from_env,
    read_model_from_env, resolve_api_key, resolve_model,
};
pub use logging::{log_prompt_cache_usage, log_token_usage};
pub use model_discovery::{
    // Builder and config
    AuthMethod,
    GoogleModelInfo,
    GoogleModelsResponse,
    // Response types
    ModelData,
    ModelDiscoveryBuilder,
    ModelsResponse,
    OllamaModelInfo,
    OllamaTagsResponse,
    check_local_provider_availability,
    fetch_anthropic_models,
    fetch_google_models,
    fetch_ollama_models,
    // Convenience functions
    fetch_openai_compatible_models,
};

pub use anthropic_stream::{AnthropicStreamFeed, AnthropicStreamParser};
pub use cohere_stream::CohereStreamParser;
pub use content_fallback::prepare_content_for_provider;
pub use content_part::{ContentPartJsonFormat, content_parts_to_json_array};
pub use google_stream::GoogleStreamAccumulator;
pub use openai_compat::{
    convert_tool_choice_anthropic, convert_tool_choice_cohere, convert_tool_choice_openai,
    role_to_str,
};
pub use openai_stream::{
    openai_stream_channel, openai_usage_to_token_usage, parse_sse_stream_to_channel,
};
pub use response_format::response_format_to_json_value;
pub use response_parser::{
    check_response_status, check_response_status_with_retry, classify_error, extract_message_text,
    retry_llm_request, role_to_string,
};
pub use setup::{create_http_client, create_retry_strategy};
pub use streaming::{
    FirstByteDeadline, SseDataLine, SseLimits, StreamingToolCallAccumulator, is_stream_done,
    no_first_token_error, parse_sse_data_line, sse_data_lines, sse_data_lines_within,
};
pub use streaming_request::retry_stream_establishment;
pub use tools::{cached_tool_mapping, tool_schema_to_json_value};
