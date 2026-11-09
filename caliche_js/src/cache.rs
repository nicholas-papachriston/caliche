//! JavaScript bindings for Caliche Caching

use crate::TokenUsageJs;
use caliche::llm::cache::{
    CachePolicy as RustCachePolicy, CacheStats as RustCacheStats,
    CachedResponse as RustCachedResponse, LlmCache as RustLlmCache,
};
use caliche::messages::TokenUsage;
use napi_derive::napi;
use serde_json::json;
use std::sync::Arc;

/// JavaScript wrapper for `CachePolicy`
#[napi]
pub struct CachePolicy {
    inner: RustCachePolicy,
}

#[napi]
#[expect(
    clippy::needless_pass_by_value,
    reason = "NAPI converts and transfers ownership of JavaScript string arguments"
)]
impl CachePolicy {
    /// Create a new `CachePolicy`
    #[napi(factory, catch_unwind)]
    ///
    /// # Errors
    ///
    /// Returns a NAPI error when validation, synchronization, serialization,
    /// or an underlying runtime operation fails.
    pub fn new(policy_type: String, threshold: Option<f64>) -> napi::Result<Self> {
        let policy = match policy_type.to_lowercase().as_str() {
            "exact_match" | "exact" => RustCachePolicy::ExactMatch,
            "semantic_similarity" | "semantic" => RustCachePolicy::SemanticSimilarity {
                threshold: threshold.unwrap_or(0.95) as f32,
            },
            "disabled" => RustCachePolicy::Disabled,
            _ => {
                return Err(napi::Error::from_reason(
                    "Invalid policy type: must be 'exact_match', 'semantic_similarity', or 'disabled'",
                ));
            }
        };
        Ok(Self { inner: policy })
    }

    /// Convert `CachePolicy` to string
    #[napi(catch_unwind)]
    pub fn to_string(&self) -> String {
        match &self.inner {
            RustCachePolicy::ExactMatch => "exact_match".to_string(),
            RustCachePolicy::SemanticSimilarity { .. } => "semantic_similarity".to_string(),
            RustCachePolicy::Disabled => "disabled".to_string(),
        }
    }
}

/// JavaScript wrapper for `CachedResponse`
#[napi]
pub struct CachedResponse {
    content: String,
    token_usage: TokenUsageJs,
    metadata: serde_json::Value,
}

#[napi]
impl CachedResponse {
    /// Create a new `CachedResponse`
    #[napi(constructor, catch_unwind)]
    pub fn new(
        content: String,
        prompt_tokens: Option<u32>,
        completion_tokens: Option<u32>,
    ) -> Self {
        Self {
            content,
            token_usage: TokenUsageJs {
                prompt_tokens,
                completion_tokens,
                total_tokens: prompt_tokens.zip(completion_tokens).map(|(p, c)| p + c),
                embedding_tokens: None,
                cache_creation_tokens: None,
                cache_read_tokens: None,
            },
            metadata: json!({}),
        }
    }

    /// Get content
    #[napi(catch_unwind)]
    pub fn get_content(&self) -> String {
        self.content.clone()
    }

    /// Get token usage
    #[napi(catch_unwind)]
    pub const fn get_token_usage(&self) -> TokenUsageJs {
        TokenUsageJs {
            prompt_tokens: self.token_usage.prompt_tokens,
            completion_tokens: self.token_usage.completion_tokens,
            total_tokens: self.token_usage.total_tokens,
            embedding_tokens: self.token_usage.embedding_tokens,
            cache_creation_tokens: self.token_usage.cache_creation_tokens,
            cache_read_tokens: self.token_usage.cache_read_tokens,
        }
    }

    /// Get metadata
    #[napi(catch_unwind)]
    pub fn get_metadata(&self) -> serde_json::Value {
        self.metadata.clone()
    }
}

/// JavaScript wrapper for `CacheStats`
#[napi(object)]
pub struct CacheStats {
    pub hits: f64,
    pub misses: f64,
    pub size: u32,
    pub hit_rate: f64,
}

impl TryFrom<RustCacheStats> for CacheStats {
    type Error = napi::Error;

    fn try_from(stats: RustCacheStats) -> napi::Result<Self> {
        const MAX_SAFE_INTEGER: u64 = (1_u64 << 53) - 1;

        let hits = u64_to_js_number(stats.hits, "cache hits", MAX_SAFE_INTEGER)?;
        let misses = u64_to_js_number(stats.misses, "cache misses", MAX_SAFE_INTEGER)?;
        let size = u32::try_from(stats.size)
            .map_err(|_| napi::Error::from_reason("cache size exceeds the JavaScript API limit"))?;
        Ok(Self {
            hits,
            misses,
            size,
            hit_rate: stats.hit_rate,
        })
    }
}

fn u64_to_js_number(value: u64, field: &str, max_safe_integer: u64) -> napi::Result<f64> {
    if value > max_safe_integer {
        return Err(napi::Error::from_reason(format!(
            "{field} exceeds JavaScript's maximum safe integer"
        )));
    }
    #[expect(
        clippy::cast_precision_loss,
        reason = "the preceding JavaScript safe-integer bound makes this conversion exact"
    )]
    Ok(value as f64)
}

/// JavaScript wrapper for `LlmCache`
#[napi]
pub struct LlmCache {
    inner: Arc<caliche::llm::cache::LlmCache>,
}

#[napi]
#[expect(
    clippy::needless_pass_by_value,
    reason = "NAPI converts and transfers ownership of JavaScript string arguments"
)]
impl LlmCache {
    /// Create a new `LlmCache`
    #[napi(constructor, catch_unwind)]
    ///
    /// # Errors
    ///
    /// Returns a NAPI error when validation, synchronization, serialization,
    /// or an underlying runtime operation fails.
    pub fn new(
        max_size: u32,
        ttl_secs: u32,
        policy_type: String,
        threshold: Option<f64>,
    ) -> napi::Result<Self> {
        let _ = threshold;
        let policy = match policy_type.to_lowercase().as_str() {
            "exact_match" | "exact" | "semantic_similarity" | "semantic" => {
                RustCachePolicy::ExactMatch
            }
            "disabled" => RustCachePolicy::Disabled,
            _ => {
                return Err(napi::Error::from_reason(format!(
                    "Unknown cache policy type: {policy_type}"
                )));
            }
        };
        Ok(Self {
            inner: Arc::new(RustLlmCache::new(
                max_size as usize,
                u64::from(ttl_secs),
                policy,
            )),
        })
    }

    /// Get cached response
    #[napi]
    ///
    /// # Errors
    ///
    /// Returns a NAPI error when validation, synchronization, serialization,
    /// or an underlying runtime operation fails.
    pub async fn get(&self, prompt: String, model: String) -> napi::Result<Option<CachedResponse>> {
        let result = self.inner.get(&prompt, &model).await;
        match result {
            Ok(Some(response)) => Ok(Some(CachedResponse {
                content: response.content,
                token_usage: response.token_usage.into(),
                metadata: response.metadata,
            })),
            Ok(None) => Ok(None),
            Err(e) => Err(napi::Error::from_reason(format!("Cache get failed: {e}"))),
        }
    }

    /// Put response in cache
    #[napi]
    ///
    /// # Errors
    ///
    /// Returns a NAPI error when validation, synchronization, serialization,
    /// or an underlying runtime operation fails.
    pub async fn put(
        &self,
        prompt: String,
        model: String,
        content: String,
        prompt_tokens: Option<u32>,
        completion_tokens: Option<u32>,
        total_tokens: Option<u32>,
        embedding_tokens: Option<u32>,
        metadata: Option<serde_json::Value>,
    ) -> napi::Result<()> {
        let cached_response = RustCachedResponse {
            content,
            token_usage: TokenUsage {
                prompt_tokens,
                completion_tokens,
                total_tokens,
                embedding_tokens,
                context_window: None,
                cache_creation_tokens: None,
                cache_read_tokens: None,
                last_prompt_tokens: None,
            },
            metadata: metadata.unwrap_or_else(|| serde_json::json!({})),
        };
        self.inner
            .put(&prompt, &model, cached_response)
            .await
            .map_err(|e| napi::Error::from_reason(format!("Cache put failed: {e}")))
    }

    /// Clear cache
    #[napi]
    ///
    /// # Errors
    ///
    /// Returns a NAPI error when validation, synchronization, serialization,
    /// or an underlying runtime operation fails.
    pub async fn clear(&self) -> napi::Result<()> {
        self.inner
            .clear()
            .await
            .map_err(|e| napi::Error::from_reason(format!("Cache clear failed: {e}")))
    }

    /// Get cache statistics
    #[napi]
    pub async fn get_stats(&self) -> napi::Result<CacheStats> {
        self.inner
            .get_stats()
            .await
            .map_err(|e| napi::Error::from_reason(format!("Failed to get cache stats: {e}")))
            .and_then(TryInto::try_into)
    }
}
