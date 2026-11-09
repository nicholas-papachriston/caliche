use crate::llm::embedding::{EmbeddingProvider, EmbeddingRequest};
use crate::traits::cache::CacheBackend;
use crate::utils::math::cosine_similarity;
use serde::{Deserialize, Serialize};
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::RwLock;

fn u64_to_f64(value: u64) -> f64 {
    const TWO_TO_32: f64 = 4_294_967_296.0;
    let high = u32::try_from(value >> 32).unwrap_or(u32::MAX);
    let low = u32::try_from(value & u64::from(u32::MAX)).unwrap_or(u32::MAX);
    f64::from(high).mul_add(TWO_TO_32, f64::from(low))
}

#[derive(Debug, Clone)]
pub enum CachePolicy {
    ExactMatch,
    /// Match cached prompts by embedding cosine similarity.
    ///
    /// Requires an [`EmbeddingProvider`] via [`LlmCache::with_embedding`].
    /// Without a provider, behaves as [`CachePolicy::ExactMatch`].
    SemanticSimilarity {
        threshold: f32,
    },
    Disabled,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CachedResponse {
    pub content: String,
    pub token_usage: crate::messages::TokenUsage,
    pub metadata: serde_json::Value,
}

#[derive(Debug, Clone)]
struct SemanticIndexEntry {
    model: String,
    embedding: Vec<f32>,
    key: String,
}

#[derive(Debug)]
pub struct LlmCache {
    cache: Arc<dyn CacheBackend>,
    policy: CachePolicy,
    embedding_provider: Option<Arc<dyn EmbeddingProvider>>,
    /// Side index for semantic lookup (model-scoped).
    semantic_index: RwLock<Vec<SemanticIndexEntry>>,
    /// Soft cap on side-index rows (aligned with typical backend capacity).
    semantic_index_limit: usize,
    hits: std::sync::atomic::AtomicU64,
    misses: std::sync::atomic::AtomicU64,
}

impl LlmCache {
    /// Create a new LLM cache with a custom cache backend
    pub fn with_backend(backend: Arc<dyn CacheBackend>, policy: CachePolicy) -> Self {
        Self {
            cache: backend,
            policy,
            embedding_provider: None,
            semantic_index: RwLock::new(Vec::new()),
            semantic_index_limit: 1_024,
            hits: std::sync::atomic::AtomicU64::new(0),
            misses: std::sync::atomic::AtomicU64::new(0),
        }
    }

    /// Create a new LLM cache with default in-memory backend
    #[must_use]
    pub fn new(max_size: usize, ttl_secs: u64, policy: CachePolicy) -> Self {
        use crate::traits::cache::InMemoryCacheBackend;
        let backend = Arc::new(InMemoryCacheBackend::new(
            max_size,
            Duration::from_secs(ttl_secs),
        )) as Arc<dyn CacheBackend>;
        let mut cache = Self::with_backend(backend, policy);
        cache.semantic_index_limit = max_size.max(1);
        cache
    }

    /// Attach an embedding provider for [`CachePolicy::SemanticSimilarity`].
    #[must_use]
    pub fn with_embedding(mut self, provider: Arc<dyn EmbeddingProvider>) -> Self {
        self.embedding_provider = Some(provider);
        self
    }

    /// # Errors
    ///
    /// Returns an error if the language-model operation cannot be completed.
    pub async fn get(&self, prompt: &str, model: &str) -> anyhow::Result<Option<CachedResponse>> {
        match &self.policy {
            CachePolicy::Disabled => Ok(None),
            CachePolicy::ExactMatch => self.get_exact(prompt, model).await,
            CachePolicy::SemanticSimilarity { threshold } => {
                if self.embedding_provider.is_none() {
                    return self.get_exact(prompt, model).await;
                }
                self.get_semantic(prompt, model, *threshold).await
            }
        }
    }

    /// # Errors
    ///
    /// Returns an error if the language-model operation cannot be completed.
    pub async fn put(
        &self,
        prompt: &str,
        model: &str,
        response: CachedResponse,
    ) -> anyhow::Result<()> {
        if matches!(self.policy, CachePolicy::Disabled) {
            return Ok(());
        }

        let key = Self::cache_key(prompt, model);
        let bytes = super::codec::encode_cached_response(&response)
            .map_err(|e| anyhow::anyhow!("Failed to serialize response: {e}"))?;
        self.cache.set(&key, &bytes, None).await?;

        if let CachePolicy::SemanticSimilarity { .. } = self.policy
            && let Some(provider) = &self.embedding_provider
        {
            let embedding = embed_text(provider.as_ref(), prompt).await?;
            let mut index = self.semantic_index.write().await;
            index.retain(|entry| entry.key != key);
            index.push(SemanticIndexEntry {
                model: model.to_string(),
                embedding,
                key,
            });
            let overflow = index.len().saturating_sub(self.semantic_index_limit);
            if overflow > 0 {
                index.drain(0..overflow);
            }
        }

        Ok(())
    }

    /// # Errors
    ///
    /// Returns an error if the language-model operation cannot be completed.
    pub async fn clear(&self) -> anyhow::Result<()> {
        self.semantic_index.write().await.clear();
        self.cache.clear().await
    }

    /// # Errors
    ///
    /// Returns an error if the language-model operation cannot be completed.
    pub async fn get_stats(&self) -> anyhow::Result<CacheStats> {
        let hits = self.hits.load(std::sync::atomic::Ordering::Relaxed);
        let misses = self.misses.load(std::sync::atomic::Ordering::Relaxed);
        let total = hits + misses;
        let hit_rate = if total > 0 {
            u64_to_f64(hits) / u64_to_f64(total)
        } else {
            0.0
        };

        let backend_stats = self.cache.stats().await?;

        Ok(CacheStats {
            hits,
            misses,
            size: backend_stats.size,
            hit_rate,
        })
    }

    async fn get_exact(&self, prompt: &str, model: &str) -> anyhow::Result<Option<CachedResponse>> {
        let key = Self::cache_key(prompt, model);
        self.load_by_key(&key).await
    }

    async fn get_semantic(
        &self,
        prompt: &str,
        model: &str,
        threshold: f32,
    ) -> anyhow::Result<Option<CachedResponse>> {
        let Some(provider) = &self.embedding_provider else {
            return self.get_exact(prompt, model).await;
        };

        let query_embedding = embed_text(provider.as_ref(), prompt).await?;
        let best_key = {
            let index = self.semantic_index.read().await;
            let mut best: Option<(f32, String)> = None;
            for entry in index.iter() {
                if entry.model != model {
                    continue;
                }
                let score = cosine_similarity(&query_embedding, &entry.embedding);
                if score >= threshold {
                    let better = best.as_ref().map_or(true, |(s, _)| score > *s);
                    if better {
                        best = Some((score, entry.key.clone()));
                    }
                }
            }
            drop(index);
            best.map(|(_, key)| key)
        };

        if let Some(key) = best_key {
            if let Some(response) = self.load_by_key(&key).await? {
                return Ok(Some(response));
            }
            // Backend evicted/expired the row; drop the stale index entry.
            self.semantic_index
                .write()
                .await
                .retain(|entry| entry.key != key);
            return Ok(None);
        }

        self.misses
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Ok(None)
    }

    async fn load_by_key(&self, key: &str) -> anyhow::Result<Option<CachedResponse>> {
        match self.cache.get(key).await {
            Ok(Some(bytes)) => match super::codec::decode_cached_response(&bytes) {
                Ok(response) => {
                    self.hits.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    Ok(Some(response))
                }
                Err(e) => Err(anyhow::anyhow!(
                    "Failed to deserialize cached response: {e}"
                )),
            },
            Ok(None) => {
                self.misses
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                Ok(None)
            }
            Err(e) => Err(e),
        }
    }

    fn cache_key(prompt: &str, model: &str) -> String {
        let mut hasher = DefaultHasher::new();
        prompt.hash(&mut hasher);
        model.hash(&mut hasher);
        format!("llm_cache:{}", hasher.finish())
    }
}

async fn embed_text(provider: &dyn EmbeddingProvider, text: &str) -> anyhow::Result<Vec<f32>> {
    let response = provider
        .embed(EmbeddingRequest {
            texts: vec![text.to_string()],
            model: None,
        })
        .await?;
    response
        .embeddings
        .into_iter()
        .next()
        .ok_or_else(|| anyhow::anyhow!("Embedding provider returned no vectors"))
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheStats {
    pub hits: u64,
    pub misses: u64,
    pub size: usize,
    pub hit_rate: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::embedding::EmbeddingResponse;
    use async_trait::async_trait;
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    const DIM: usize = 16;

    #[derive(Debug)]
    struct FakeEmbeddingProvider;

    #[async_trait]
    impl EmbeddingProvider for FakeEmbeddingProvider {
        async fn embed(&self, request: EmbeddingRequest) -> anyhow::Result<EmbeddingResponse> {
            let embeddings = request
                .texts
                .iter()
                .map(|text| hash_embedding(text))
                .collect();
            Ok(EmbeddingResponse {
                embeddings,
                usage: None,
                model: "fake".to_string(),
            })
        }

        fn name(&self) -> &str {
            "fake"
        }
    }

    fn hash_embedding(text: &str) -> Vec<f32> {
        let mut hasher = DefaultHasher::new();
        text.hash(&mut hasher);
        let seed = hasher.finish();
        let mut v = Vec::with_capacity(DIM);
        for i in 0..DIM {
            let mut h = DefaultHasher::new();
            (seed.wrapping_add(i as u64)).hash(&mut h);
            let bits = h.finish();
            // Map to [-1, 1]
            let f = ((bits % 1000) as f32 / 500.0) - 1.0;
            v.push(f);
        }
        // L2 normalize for stable cosine
        let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for x in &mut v {
                *x /= norm;
            }
        }
        v
    }

    fn sample_response(content: &str) -> CachedResponse {
        CachedResponse {
            content: content.to_string(),
            token_usage: crate::messages::TokenUsage::default(),
            metadata: serde_json::json!({}),
        }
    }

    #[test]
    #[expect(
        clippy::lossy_float_literal,
        reason = "the literals spell the integer bounds under test"
    )]
    fn u64_to_f64_handles_precision_boundaries() {
        assert_eq!(u64_to_f64(0), 0.0);
        assert_eq!(u64_to_f64(1_u64 << 53), 9_007_199_254_740_992.0);
        assert_eq!(u64_to_f64(u64::MAX), 18_446_744_073_709_551_615.0);
    }

    #[tokio::test]
    async fn semantic_hit_on_identical_prompt() {
        let cache = LlmCache::new(32, 60, CachePolicy::SemanticSimilarity { threshold: 0.99 })
            .with_embedding(Arc::new(FakeEmbeddingProvider));

        cache
            .put("hello world", "gpt-test", sample_response("cached"))
            .await
            .unwrap();

        let hit = cache.get("hello world", "gpt-test").await.unwrap();
        assert_eq!(hit.map(|r| r.content), Some("cached".to_string()));
    }

    #[tokio::test]
    async fn semantic_miss_below_threshold() {
        let cache = LlmCache::new(32, 60, CachePolicy::SemanticSimilarity { threshold: 0.999 })
            .with_embedding(Arc::new(FakeEmbeddingProvider));

        cache
            .put("alpha unique prompt", "gpt-test", sample_response("a"))
            .await
            .unwrap();

        let miss = cache
            .get("completely different omega text", "gpt-test")
            .await
            .unwrap();
        assert!(miss.is_none());
    }

    #[tokio::test]
    async fn semantic_without_provider_falls_back_to_exact() {
        let cache = LlmCache::new(32, 60, CachePolicy::SemanticSimilarity { threshold: 0.5 });

        cache
            .put("exact key", "gpt-test", sample_response("ok"))
            .await
            .unwrap();

        let hit = cache.get("exact key", "gpt-test").await.unwrap();
        assert_eq!(hit.map(|r| r.content), Some("ok".to_string()));

        let miss = cache.get("other key", "gpt-test").await.unwrap();
        assert!(miss.is_none());
    }

    #[tokio::test]
    async fn exact_match_still_works() {
        let cache = LlmCache::new(32, 60, CachePolicy::ExactMatch);
        cache.put("p", "m", sample_response("v")).await.unwrap();
        assert_eq!(
            cache.get("p", "m").await.unwrap().map(|r| r.content),
            Some("v".to_string())
        );
        assert!(cache.get("other", "m").await.unwrap().is_none());
    }

    #[tokio::test]
    async fn semantic_lookup_is_model_scoped() {
        let cache = LlmCache::new(32, 60, CachePolicy::SemanticSimilarity { threshold: 0.99 })
            .with_embedding(Arc::new(FakeEmbeddingProvider));

        cache
            .put("shared prompt", "model-a", sample_response("from-a"))
            .await
            .unwrap();

        let miss = cache.get("shared prompt", "model-b").await.unwrap();
        assert!(miss.is_none());

        let hit = cache.get("shared prompt", "model-a").await.unwrap();
        assert_eq!(hit.map(|r| r.content), Some("from-a".to_string()));
    }

    #[tokio::test]
    async fn semantic_index_respects_capacity_cap() {
        let cache = LlmCache::new(2, 60, CachePolicy::SemanticSimilarity { threshold: 0.99 })
            .with_embedding(Arc::new(FakeEmbeddingProvider));

        cache
            .put("prompt-one", "m", sample_response("1"))
            .await
            .unwrap();
        cache
            .put("prompt-two", "m", sample_response("2"))
            .await
            .unwrap();
        cache
            .put("prompt-three", "m", sample_response("3"))
            .await
            .unwrap();

        let index_len = cache.semantic_index.read().await.len();
        assert_eq!(index_len, 2);

        // Oldest index entry should have been drained; newest prompts remain hittable.
        let miss = cache.get("prompt-one", "m").await.unwrap();
        assert!(miss.is_none());
        let hit = cache.get("prompt-three", "m").await.unwrap();
        assert_eq!(hit.map(|r| r.content), Some("3".to_string()));
    }

    #[tokio::test]
    async fn semantic_put_replaces_same_key_index_entry() {
        let cache = LlmCache::new(8, 60, CachePolicy::SemanticSimilarity { threshold: 0.99 })
            .with_embedding(Arc::new(FakeEmbeddingProvider));

        cache
            .put("same prompt", "m", sample_response("first"))
            .await
            .unwrap();
        cache
            .put("same prompt", "m", sample_response("second"))
            .await
            .unwrap();

        let index_len = cache.semantic_index.read().await.len();
        assert_eq!(index_len, 1);
        let hit = cache.get("same prompt", "m").await.unwrap();
        assert_eq!(hit.map(|r| r.content), Some("second".to_string()));
    }

    #[tokio::test]
    async fn disabled_policy_skips_get_and_put() {
        let cache = LlmCache::new(8, 60, CachePolicy::Disabled);
        cache.put("p", "m", sample_response("v")).await.unwrap();
        assert!(cache.get("p", "m").await.unwrap().is_none());
    }

    #[tokio::test]
    async fn clear_wipes_semantic_index() {
        let cache = LlmCache::new(8, 60, CachePolicy::SemanticSimilarity { threshold: 0.99 })
            .with_embedding(Arc::new(FakeEmbeddingProvider));
        cache.put("p", "m", sample_response("v")).await.unwrap();
        assert_eq!(cache.semantic_index.read().await.len(), 1);
        cache.clear().await.unwrap();
        assert!(cache.semantic_index.read().await.is_empty());
        assert!(cache.get("p", "m").await.unwrap().is_none());
    }

    #[tokio::test]
    async fn semantic_stale_index_entry_dropped_when_backend_misses() {
        use crate::traits::cache::{CacheBackend, InMemoryCacheBackend};
        use std::time::Duration;

        let backend = Arc::new(InMemoryCacheBackend::new(32, Duration::from_secs(60)));
        let cache = LlmCache::with_backend(
            backend.clone() as Arc<dyn CacheBackend>,
            CachePolicy::SemanticSimilarity { threshold: 0.99 },
        )
        .with_embedding(Arc::new(FakeEmbeddingProvider));

        cache
            .put("hello world", "m", sample_response("cached"))
            .await
            .unwrap();
        assert_eq!(cache.semantic_index.read().await.len(), 1);

        backend.clear().await.unwrap();
        let miss = cache.get("hello world", "m").await.unwrap();
        assert!(miss.is_none());
        assert!(cache.semantic_index.read().await.is_empty());
    }
}
