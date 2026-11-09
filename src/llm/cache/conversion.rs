//! Generic conversion cache with automatic invalidation
//!
//! Caches the result of expensive transformations based on input hash.
//! Thread-safe and suitable for use in provider structs.
//! Uses `arc_swap` for lock-free reads on the hot path.

use arc_swap::ArcSwap;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

/// Hash messages for conversion cache key.
/// Includes role and full content shape (text/structured/multimodal).
pub fn hash_messages_for_cache(msgs: &[crate::messages::Message]) -> u64 {
    let mut hasher = DefaultHasher::new();
    for msg in msgs {
        msg.role.hash(&mut hasher);
        match &msg.content {
            crate::messages::MessageContent::Text(text) => {
                0u8.hash(&mut hasher);
                text.hash(&mut hasher);
            }
            crate::messages::MessageContent::Structured {
                text,
                tool_calls,
                tool_call_id,
            } => {
                1u8.hash(&mut hasher);
                text.hash(&mut hasher);
                tool_call_id.hash(&mut hasher);
                match serde_json::to_string(tool_calls) {
                    Ok(serialized) => serialized.hash(&mut hasher),
                    Err(error) => {
                        tracing::warn!(
                            error = %error,
                            "Failed to hash structured tool calls, using fallback marker"
                        );
                        u64::MAX.hash(&mut hasher);
                    }
                }
            }
            crate::messages::MessageContent::Multimodal(parts) => {
                2u8.hash(&mut hasher);
                parts.len().hash(&mut hasher);
                for part in parts {
                    match part {
                        crate::messages::ContentPart::Text { text } => {
                            0u8.hash(&mut hasher);
                            text.hash(&mut hasher);
                        }
                        crate::messages::ContentPart::Image { data, media_type } => {
                            1u8.hash(&mut hasher);
                            data.len().hash(&mut hasher);
                            media_type.hash(&mut hasher);
                        }
                        crate::messages::ContentPart::ImageUrl { url } => {
                            2u8.hash(&mut hasher);
                            url.hash(&mut hasher);
                        }
                        crate::messages::ContentPart::Video { data, media_type } => {
                            3u8.hash(&mut hasher);
                            data.len().hash(&mut hasher);
                            media_type.hash(&mut hasher);
                        }
                        crate::messages::ContentPart::VideoUrl { url } => {
                            4u8.hash(&mut hasher);
                            url.hash(&mut hasher);
                        }
                        crate::messages::ContentPart::Audio { data, media_type } => {
                            5u8.hash(&mut hasher);
                            data.len().hash(&mut hasher);
                            media_type.hash(&mut hasher);
                        }
                        crate::messages::ContentPart::AudioUrl { url } => {
                            6u8.hash(&mut hasher);
                            url.hash(&mut hasher);
                        }
                        crate::messages::ContentPart::Document {
                            data,
                            media_type,
                            filename,
                        } => {
                            7u8.hash(&mut hasher);
                            data.len().hash(&mut hasher);
                            media_type.hash(&mut hasher);
                            filename.hash(&mut hasher);
                        }
                        crate::messages::ContentPart::DocumentUrl { url, filename } => {
                            8u8.hash(&mut hasher);
                            url.hash(&mut hasher);
                            filename.hash(&mut hasher);
                        }
                    }
                }
            }
        }
    }
    hasher.finish()
}

/// Hash messages for conversion cache key, skipping system messages.
/// Use when system messages contain timestamps that change frequently
/// but do not affect message conversion logic.
#[must_use]
pub fn hash_messages_for_cache_skip_system(msgs: &[crate::messages::Message]) -> u64 {
    let mut hasher = DefaultHasher::new();
    for msg in msgs {
        if matches!(msg.role, crate::messages::MessageRole::System) {
            continue;
        }
        msg.role.hash(&mut hasher);
        hash_messages_for_cache(std::slice::from_ref(msg)).hash(&mut hasher);
    }
    hasher.finish()
}

/// Hash tools for conversion cache key.
/// Includes name, description, and JSON-serialized parameters.
pub fn hash_tools_for_cache(tools: &[crate::messages::ToolSchema]) -> u64 {
    let mut hasher = DefaultHasher::new();
    for tool in tools {
        tool.name.hash(&mut hasher);
        tool.description.hash(&mut hasher);
        match serde_json::to_string(&tool.parameters) {
            Ok(params_json) => params_json.hash(&mut hasher),
            Err(error) => {
                tracing::warn!(
                    tool = %tool.name,
                    error = %error,
                    "Failed to serialize tool parameters, using fallback marker"
                );
                u64::MAX.hash(&mut hasher);
            }
        }
    }
    hasher.finish()
}

/// Thread-safe conversion cache for provider transformations.
/// Returns `Arc<T>` on cache hits to avoid cloning the cached value.
///
/// # Type Parameters
/// - `T`: The cached result type (must be `Send + Sync`)
///
/// # Example
/// ```
/// use caliche::llm::cache::ConversionCache;
/// use std::sync::Arc;
///
/// // Example: Cache conversion of integers to strings
/// struct Converter {
///     cache: ConversionCache<Vec<String>>,
/// }
///
/// impl Converter {
///     fn convert_numbers(&self, numbers: &[i32]) -> Arc<Vec<String>> {
///         self.cache.get_or_compute(numbers, |nums| {
///             nums.iter().map(|n| n.to_string()).collect()
///         })
///     }
/// }
///
/// let converter = Converter {
///     cache: ConversionCache::new(),
/// };
///
/// let result1 = converter.convert_numbers(&[1, 2, 3]);
/// let result2 = converter.convert_numbers(&[1, 2, 3]); // Uses cache
/// assert_eq!(*result1, *result2);
/// ```
pub struct ConversionCache<T> {
    inner: Arc<ArcSwap<Option<CacheEntry<T>>>>,
}

struct CacheEntry<T> {
    hash: u64,
    value: Arc<T>,
}

impl<T: Send + Sync> ConversionCache<T> {
    /// Create new empty cache
    #[must_use]
    pub fn new() -> Self {
        Self {
            inner: Arc::new(ArcSwap::from_pointee(None)),
        }
    }

    /// Get cached value or compute and cache it
    ///
    /// # Arguments
    /// - `input`: The input to hash for cache key
    /// - `compute`: Function to compute value if cache miss
    ///
    /// # Returns
    /// `Arc` to cached value or newly computed value (zero-copy on cache hit)
    pub fn get_or_compute<F, I>(&self, input: &I, compute: F) -> Arc<T>
    where
        F: FnOnce(&I) -> T,
        I: Hash + ?Sized,
    {
        let guard = self.inner.as_ref().load();
        if let Some(ref entry) = **guard {
            let hash = Self::hash_input(input);
            if entry.hash == hash {
                return Arc::clone(&entry.value);
            }
            drop(guard);
            let value = Arc::new(compute(input));
            self.inner.as_ref().store(Arc::new(Some(CacheEntry {
                hash,
                value: Arc::clone(&value),
            })));
            return value;
        }
        drop(guard);
        // Cache empty - compute value first, then hash
        let value = Arc::new(compute(input));
        let hash = Self::hash_input(input);
        self.inner.as_ref().store(Arc::new(Some(CacheEntry {
            hash,
            value: Arc::clone(&value),
        })));
        value
    }

    /// Get cached value or compute and cache it with custom hash function
    ///
    /// Use this when the input type doesn't implement Hash.
    /// When the cache is empty, skips hash computation until after computing the value.
    ///
    /// # Arguments
    /// - `input`: The input to cache
    /// - `hash_fn`: Function to compute hash of input
    /// - `compute`: Function to compute value if cache miss
    ///
    /// # Returns
    /// `Arc` to cached value or newly computed value (zero-copy on cache hit)
    pub fn get_or_compute_with_hash<F, H, I: ?Sized>(
        &self,
        input: &I,
        hash_fn: H,
        compute: F,
    ) -> Arc<T>
    where
        F: FnOnce(&I) -> T,
        H: FnOnce(&I) -> u64,
    {
        let guard = self.inner.as_ref().load();
        if let Some(ref entry) = **guard {
            let hash = hash_fn(input);
            if entry.hash == hash {
                return Arc::clone(&entry.value);
            }
            drop(guard);
            let value = Arc::new(compute(input));
            self.inner.as_ref().store(Arc::new(Some(CacheEntry {
                hash,
                value: Arc::clone(&value),
            })));
            return value;
        }
        drop(guard);
        // Cache empty - compute value first, then hash (avoids expensive hash on first call)
        let value = Arc::new(compute(input));
        let hash = hash_fn(input);
        self.inner.as_ref().store(Arc::new(Some(CacheEntry {
            hash,
            value: Arc::clone(&value),
        })));
        value
    }

    /// Clear cache, forcing next access to recompute
    pub fn invalidate(&self) {
        self.inner.as_ref().store(Arc::new(None));
    }

    #[inline]
    fn hash_input<H: Hash + ?Sized>(input: &H) -> u64 {
        let mut hasher = DefaultHasher::new();
        input.hash(&mut hasher);
        hasher.finish()
    }
}

impl<T: Send + Sync> Default for ConversionCache<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Send + Sync + std::fmt::Debug> std::fmt::Debug for ConversionCache<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let has_cache = self.inner.as_ref().load().as_ref().is_some();
        f.debug_struct("ConversionCache")
            .field("has_cache", &has_cache)
            .finish()
    }
}

impl<T> Clone for ConversionCache<T> {
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cache_hit() {
        let cache = ConversionCache::new();
        let input = vec![1, 2, 3];

        let result1 = cache.get_or_compute(&input, |input| {
            input.iter().map(|x| x * 2).collect::<Vec<_>>()
        });

        let mut called = false;
        let result2 = cache.get_or_compute(&input, |_v| {
            called = true;
            vec![]
        });

        assert_eq!(*result1, *result2);
        assert!(
            !called,
            "Compute function should not be called on cache hit"
        );
    }

    #[test]
    fn test_cache_miss() {
        let cache = ConversionCache::new();

        let input1 = vec![1, 2, 3];
        let result1 = cache.get_or_compute(&input1, |v| v.iter().sum::<i32>());

        let input2 = vec![4, 5, 6];
        let result2 = cache.get_or_compute(&input2, |v| v.iter().sum::<i32>());

        assert_eq!(*result1, 6);
        assert_eq!(*result2, 15);
    }

    #[test]
    fn test_cache_invalidate() {
        let cache = ConversionCache::new();
        let input = vec![1, 2, 3];

        let result1 = cache.get_or_compute(&input, |v| v.iter().sum::<i32>());

        cache.invalidate();

        let mut call_count = 0;
        let result2 = cache.get_or_compute(&input, |v| {
            call_count += 1;
            v.iter().sum::<i32>()
        });

        assert_eq!(*result1, *result2);
        assert_eq!(call_count, 1); // Should recompute after invalidation
    }

    #[test]
    fn test_cache_clone() {
        let cache1 = ConversionCache::new();
        let input = vec![1, 2, 3];

        cache1.get_or_compute(&input, |input| input.iter().sum::<i32>());

        #[expect(
            clippy::redundant_clone,
            reason = "the test checks that a clone shares the cache"
        )]
        let cache2 = cache1.clone();

        // Clone should share the same cache
        let mut called = false;
        let result = cache2.get_or_compute(&input, |_v| {
            called = true;
            0
        });

        assert_eq!(*result, 6);
        assert!(
            !called,
            "Compute function should not be called when cache is shared"
        );
    }
}
