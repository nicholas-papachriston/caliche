//! Optional LLM response cache wrapper.
//!
//! When enabled, completion requests are checked against a cache before calling
//! the inner provider. Exact-match cache keys are built from messages and tools
//! hashes. Streaming is not cached.

use crate::llm::Provider;
use crate::llm::cache::conversion::{hash_messages_for_cache, hash_tools_for_cache};
use crate::llm::cache::{CachedResponse, LlmCache};
use crate::messages::{CompletionContent, CompletionRequest, CompletionResponse};
use async_trait::async_trait;
use futures_util::Stream;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

/// Build a cache key string from a completion request for exact-match lookup.
fn completion_request_cache_key(request: &CompletionRequest) -> String {
    let msg_hash = hash_messages_for_cache(request.messages.as_ref());
    let tool_hash = request
        .tools
        .as_ref()
        .map_or(0, |t| hash_tools_for_cache(t.as_ref()));
    let max_tokens = request.max_tokens.unwrap_or(0);
    let temp_bits = request.temperature.map_or(0, f32::to_bits);
    let choice_str = request
        .tool_choice
        .as_ref()
        .and_then(|c| serde_json::to_string(c).ok())
        .unwrap_or_default();
    let fmt_str = request
        .response_format
        .as_ref()
        .and_then(|f| serde_json::to_value(f).ok())
        .and_then(|v| serde_json::to_string(&v).ok())
        .unwrap_or_default();
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    choice_str.as_str().hash(&mut hasher);
    fmt_str.as_str().hash(&mut hasher);
    let extra_hash = hasher.finish();
    format!("{msg_hash}:{tool_hash}:{max_tokens}:{temp_bits}:{extra_hash}")
}

fn completion_response_to_cached(response: &CompletionResponse) -> CachedResponse {
    let content = response.content.as_text().to_string();
    let token_usage = response.usage.unwrap_or_default();
    let tool_calls = response.content.as_tool_calls();
    let metadata = serde_json::json!({
        "finish_reason": response.finish_reason.as_ref().map(std::convert::AsRef::as_ref),
        "tool_calls": if tool_calls.is_empty() {
            serde_json::Value::Null
        } else {
            serde_json::to_value(tool_calls).unwrap_or(serde_json::Value::Null)
        }
    });
    CachedResponse {
        content,
        token_usage,
        metadata,
    }
}

fn cached_to_completion_response(cached: &CachedResponse) -> anyhow::Result<CompletionResponse> {
    let finish_reason = cached
        .metadata
        .get("finish_reason")
        .and_then(|v| v.as_str())
        .map(|s| beef::lean::Cow::owned(s.to_string()));
    let tool_calls_value = cached.metadata.get("tool_calls");
    let content = if let Some(arr) = tool_calls_value.and_then(|v| v.as_array())
        && !arr.is_empty()
    {
        let tool_calls: Vec<crate::messages::ToolCallInfo> =
            serde_json::from_value(serde_json::Value::Array(arr.clone()))
                .map_err(|e| anyhow::anyhow!("Failed to deserialize cached tool_calls: {e}"))?;
        CompletionContent::WithToolCalls {
            content: cached.content.clone(),
            tool_calls,
        }
    } else {
        CompletionContent::Text {
            content: cached.content.clone(),
        }
    };
    Ok(CompletionResponse {
        content,
        finish_reason,
        usage: Some(cached.token_usage),
        reasoning_content: None,
    })
}

/// Provider wrapper that caches completion responses by exact request match.
/// Streaming is delegated to the inner provider without caching.
#[derive(Debug)]
pub struct CachingProvider {
    inner: Arc<dyn Provider>,
    cache: Arc<LlmCache>,
    model: String,
}

impl CachingProvider {
    /// Create a new caching provider wrapping `inner`.
    /// `model` is used as the cache key component alongside the request hash.
    pub fn new(inner: Arc<dyn Provider>, cache: Arc<LlmCache>, model: impl Into<String>) -> Self {
        Self {
            inner,
            cache,
            model: model.into(),
        }
    }
}

#[async_trait]
impl Provider for CachingProvider {
    async fn complete(&self, request: CompletionRequest) -> anyhow::Result<CompletionResponse> {
        let key = completion_request_cache_key(&request);
        let cached = match self.cache.get(&key, &self.model).await {
            Ok(hit) => hit,
            Err(error) => {
                tracing::warn!(
                    model = %self.model,
                    error = %error,
                    "LLM cache read error, treating as miss"
                );
                None
            }
        };

        if let Some(cached) = cached {
            tracing::debug!("LLM cache hit for model {}", self.model);
            return cached_to_completion_response(&cached);
        }
        let response = self.inner.complete(request).await?;
        let cached = completion_response_to_cached(&response);
        if let Err(e) = self.cache.put(&key, &self.model, cached).await {
            tracing::warn!("Failed to write LLM response to cache: {}", e);
        }
        Ok(response)
    }

    fn stream(
        &self,
        request: CompletionRequest,
    ) -> std::pin::Pin<
        Box<dyn Stream<Item = anyhow::Result<crate::messages::StreamEvent>> + Send + '_>,
    > {
        self.inner.stream(request)
    }

    fn name(&self) -> &str {
        self.inner.name()
    }

    crate::delegate_provider_capabilities!(inner);
}
