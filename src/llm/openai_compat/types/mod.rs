//! Shared types for OpenAI-compatible providers.
//!
//! These types are used by all providers that follow the `OpenAI` chat completions API format.

mod embeddings;
mod messages;
mod requests;
mod responses;
mod streaming;
mod tools;

pub use embeddings::{
    OpenAICompatEmbeddingData, OpenAICompatEmbeddingRequest, OpenAICompatEmbeddingResponse,
};
pub use messages::{
    OpenAICompatContent, OpenAICompatFunctionCall, OpenAICompatMessage, OpenAICompatToolCall,
};
pub use requests::OpenAICompatRequest;
pub use responses::{OpenAICompatChoice, OpenAICompatResponse, OpenAICompatUsage};
pub use streaming::{
    OpenAICompatDelta, OpenAICompatStreamChoice, OpenAICompatStreamChunk,
    OpenAICompatStreamFunction, OpenAICompatStreamToolCall,
};
pub use tools::{OpenAICompatFunctionSchema, OpenAICompatTool};

/// Check if a model is thinking-enabled and requires `reasoning_content` for tool calls
#[must_use]
pub fn is_thinking_enabled_model(model: &str) -> bool {
    let model_lower = model.to_lowercase();
    model_lower.contains("thinking")
        || model_lower.contains("kimi-k2")
        || model_lower.contains("moonshot")
        || model_lower.ends_with(":thinking")
}
