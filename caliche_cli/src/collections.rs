//! Bounded collections for memory-efficient data structures.
//!
//! This module provides collections with built-in capacity limits to prevent
//! unbounded memory growth in long-running CLI sessions.

use lru::LruCache;
use std::collections::VecDeque;
use std::hash::Hash;
use std::num::NonZeroUsize;
use std::sync::Arc;

/// A vector with a maximum capacity that drops oldest items when full.
///
/// Uses a `VecDeque` internally for O(1) push/pop at both ends.
#[derive(Debug, Clone)]
pub struct BoundedVec<T> {
    inner: VecDeque<T>,
    capacity: usize,
}

impl<T> BoundedVec<T> {
    /// Create a new bounded vector with the specified capacity.
    ///
    /// # Panics
    /// Panics if capacity is 0.
    pub fn with_capacity(capacity: usize) -> Self {
        assert!(capacity > 0, "BoundedVec capacity must be > 0");
        Self {
            inner: VecDeque::with_capacity(capacity),
            capacity,
        }
    }

    /// Push an item to the back, dropping the oldest item if at capacity.
    pub fn push(&mut self, item: T) {
        if self.inner.len() >= self.capacity {
            self.inner.pop_front();
        }
        self.inner.push_back(item);
    }

    /// Push an item to the front, dropping the newest item if at capacity.
    pub fn push_front(&mut self, item: T) {
        if self.inner.len() >= self.capacity {
            self.inner.pop_back();
        }
        self.inner.push_front(item);
    }

    /// Pop an item from the back.
    pub fn pop(&mut self) -> Option<T> {
        self.inner.pop_back()
    }

    /// Pop an item from the front.
    pub fn pop_front(&mut self) -> Option<T> {
        self.inner.pop_front()
    }

    /// Get the number of items.
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Get the maximum capacity.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Check if at capacity.
    pub fn is_full(&self) -> bool {
        self.inner.len() >= self.capacity
    }

    /// Clear all items.
    pub fn clear(&mut self) {
        self.inner.clear();
    }

    /// Get an item by index.
    pub fn get(&self, index: usize) -> Option<&T> {
        self.inner.get(index)
    }

    /// Get a mutable item by index.
    pub fn get_mut(&mut self, index: usize) -> Option<&mut T> {
        self.inner.get_mut(index)
    }

    /// Get the last item.
    pub fn last(&self) -> Option<&T> {
        self.inner.back()
    }

    /// Get the first item.
    pub fn first(&self) -> Option<&T> {
        self.inner.front()
    }

    /// Iterate over items.
    pub fn iter(&self) -> impl Iterator<Item = &T> + DoubleEndedIterator {
        self.inner.iter()
    }

    /// Iterate over items mutably.
    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut T> + DoubleEndedIterator {
        self.inner.iter_mut()
    }

    /// Retain only items matching the predicate.
    pub fn retain<F>(&mut self, f: F)
    where
        F: FnMut(&T) -> bool,
    {
        self.inner.retain(f);
    }

    /// Convert to a Vec.
    pub fn to_vec(&self) -> Vec<T>
    where
        T: Clone,
    {
        self.inner.iter().cloned().collect()
    }

    /// Get as a slice (may not be contiguous).
    pub fn as_slices(&self) -> (&[T], &[T]) {
        self.inner.as_slices()
    }

    /// Make contiguous and return as slice.
    pub fn make_contiguous(&mut self) -> &mut [T] {
        self.inner.make_contiguous()
    }
}

impl<T> Default for BoundedVec<T> {
    fn default() -> Self {
        Self::with_capacity(1000) // Default capacity
    }
}

impl<T> FromIterator<T> for BoundedVec<T> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        let mut vec = Self::default();
        for item in iter {
            vec.push(item);
        }
        vec
    }
}

impl<T> IntoIterator for BoundedVec<T> {
    type Item = T;
    type IntoIter = std::collections::vec_deque::IntoIter<T>;

    fn into_iter(self) -> Self::IntoIter {
        self.inner.into_iter()
    }
}

impl<'a, T> IntoIterator for &'a BoundedVec<T> {
    type Item = &'a T;
    type IntoIter = std::collections::vec_deque::Iter<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.inner.iter()
    }
}

/// An LRU cache for interned strings.
///
/// Provides string deduplication with bounded memory usage.
pub struct LruStringCache {
    cache: LruCache<String, Arc<str>>,
}

impl LruStringCache {
    /// Create a new cache with the specified capacity.
    pub fn new(capacity: usize) -> Self {
        Self {
            cache: LruCache::new(NonZeroUsize::new(capacity).unwrap_or(NonZeroUsize::MIN)),
        }
    }

    /// Get or intern a string.
    ///
    /// Returns an `Arc<str>` that can be cheaply cloned.
    pub fn get_or_intern(&mut self, s: &str) -> Arc<str> {
        if let Some(cached) = self.cache.get(s) {
            Arc::clone(cached)
        } else {
            let arc: Arc<str> = Arc::from(s);
            self.cache.put(s.to_string(), Arc::clone(&arc));
            arc
        }
    }

    /// Get a cached string if it exists.
    pub fn get(&mut self, s: &str) -> Option<Arc<str>> {
        self.cache.get(s).cloned()
    }

    /// Check if a string is cached.
    pub fn contains(&self, s: &str) -> bool {
        self.cache.contains(s)
    }

    /// Get the number of cached strings.
    pub fn len(&self) -> usize {
        self.cache.len()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.cache.is_empty()
    }

    /// Clear the cache.
    pub fn clear(&mut self) {
        self.cache.clear();
    }
}

impl Default for LruStringCache {
    fn default() -> Self {
        Self::new(10000) // Default 10k entries
    }
}

/// A generic LRU cache wrapper with convenient methods.
pub struct Cache<K, V> {
    inner: LruCache<K, V>,
}

impl<K: Hash + Eq, V> Cache<K, V> {
    /// Create a new cache with the specified capacity.
    pub fn new(capacity: usize) -> Self {
        Self {
            inner: LruCache::new(NonZeroUsize::new(capacity).unwrap_or(NonZeroUsize::MIN)),
        }
    }

    /// Get a value by key.
    pub fn get(&mut self, key: &K) -> Option<&V> {
        self.inner.get(key)
    }

    /// Get a mutable value by key.
    pub fn get_mut(&mut self, key: &K) -> Option<&mut V> {
        self.inner.get_mut(key)
    }

    /// Insert a value.
    pub fn insert(&mut self, key: K, value: V) -> Option<V> {
        self.inner.put(key, value)
    }

    /// Get or insert with a function.
    ///
    /// Returns [`None`] only if the key could not be read back immediately after insert
    /// (should not occur for a non-zero capacity `LruCache`).
    pub fn get_or_insert_with<F>(&mut self, key: K, f: F) -> Option<&V>
    where
        F: FnOnce() -> V,
        K: Clone,
    {
        if !self.inner.contains(&key) {
            self.inner.put(key.clone(), f());
        }
        self.inner.get(&key)
    }

    /// Remove a value.
    pub fn remove(&mut self, key: &K) -> Option<V> {
        self.inner.pop(key)
    }

    /// Check if a key exists.
    pub fn contains(&self, key: &K) -> bool {
        self.inner.contains(key)
    }

    /// Get the number of entries.
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Clear the cache.
    pub fn clear(&mut self) {
        self.inner.clear();
    }

    /// Peek at a value without updating LRU order.
    pub fn peek(&self, key: &K) -> Option<&V> {
        self.inner.peek(key)
    }
}

impl<K: Hash + Eq, V> Default for Cache<K, V> {
    fn default() -> Self {
        Self::new(1000)
    }
}

/// A bounded map that evicts oldest entries when full.
///
/// Unlike LRU, this uses insertion order (FIFO) for eviction.
pub struct BoundedMap<K, V> {
    keys: VecDeque<K>,
    values: std::collections::HashMap<K, V>,
    capacity: usize,
}

impl<K: Hash + Eq + Clone, V> BoundedMap<K, V> {
    /// Create a new bounded map with the specified capacity.
    pub fn with_capacity(capacity: usize) -> Self {
        assert!(capacity > 0, "BoundedMap capacity must be > 0");
        Self {
            keys: VecDeque::with_capacity(capacity),
            values: std::collections::HashMap::with_capacity(capacity),
            capacity,
        }
    }

    /// Insert a key-value pair.
    pub fn insert(&mut self, key: K, value: V) -> Option<V> {
        // If key exists, just update the value
        if self.values.contains_key(&key) {
            return self.values.insert(key, value);
        }

        // Evict oldest if at capacity
        if self.keys.len() >= self.capacity {
            if let Some(oldest_key) = self.keys.pop_front() {
                self.values.remove(&oldest_key);
            }
        }

        self.keys.push_back(key.clone());
        self.values.insert(key, value)
    }

    /// Get a value by key.
    pub fn get(&self, key: &K) -> Option<&V> {
        self.values.get(key)
    }

    /// Get a mutable value by key.
    pub fn get_mut(&mut self, key: &K) -> Option<&mut V> {
        self.values.get_mut(key)
    }

    /// Remove a value by key.
    pub fn remove(&mut self, key: &K) -> Option<V> {
        if let Some(value) = self.values.remove(key) {
            self.keys.retain(|k| k != key);
            Some(value)
        } else {
            None
        }
    }

    /// Check if a key exists.
    pub fn contains_key(&self, key: &K) -> bool {
        self.values.contains_key(key)
    }

    /// Get the number of entries.
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// Clear all entries.
    pub fn clear(&mut self) {
        self.keys.clear();
        self.values.clear();
    }

    /// Iterate over key-value pairs.
    pub fn iter(&self) -> impl Iterator<Item = (&K, &V)> {
        self.values.iter()
    }

    /// Get keys in insertion order.
    pub fn keys(&self) -> impl Iterator<Item = &K> {
        self.keys.iter()
    }
}

impl<K: Hash + Eq + Clone, V> Default for BoundedMap<K, V> {
    fn default() -> Self {
        Self::with_capacity(1000)
    }
}

/// A sorted collection for efficient prefix-based completion filtering.
///
/// Uses binary search for O(log n) prefix lookups instead of O(n) linear search.
#[derive(Debug, Clone)]
pub struct SortedCompletions {
    /// Items stored in sorted order.
    items: Arc<[Arc<str>]>,
}

impl SortedCompletions {
    /// Create from an iterator of strings, sorting them.
    pub fn from_iter<I, S>(iter: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let mut items: Vec<Arc<str>> = iter.into_iter().map(|s| Arc::from(s.as_ref())).collect();
        items.sort_by(|a, b| a.as_ref().cmp(b.as_ref()));
        Self {
            items: Arc::from(items),
        }
    }

    /// Create from a pre-sorted slice (assumes already sorted).
    pub fn from_sorted(items: Vec<Arc<str>>) -> Self {
        Self {
            items: Arc::from(items),
        }
    }

    /// Create an empty collection.
    pub fn empty() -> Self {
        Self {
            items: Arc::from([]),
        }
    }

    /// Filter items by prefix using binary search.
    ///
    /// Returns a Vec of items that start with the given prefix.
    /// Uses binary search to find the start position, then iterates until
    /// items no longer match the prefix.
    pub fn filter_prefix(&self, prefix: &str) -> Vec<&Arc<str>> {
        if prefix.is_empty() {
            return self.items.iter().collect();
        }

        // Find the first item >= prefix using binary search
        let start = self.items.partition_point(|item| item.as_ref() < prefix);

        // Iterate from start while items still have the prefix
        self.items[start..]
            .iter()
            .take_while(|item| item.starts_with(prefix))
            .collect()
    }

    /// Get all items.
    pub fn all(&self) -> &[Arc<str>] {
        &self.items
    }

    /// Get the number of items.
    pub fn len(&self) -> usize {
        self.items.len()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    /// Check if an item exists (exact match).
    pub fn contains(&self, item: &str) -> bool {
        self.items
            .binary_search_by(|probe| probe.as_ref().cmp(item))
            .is_ok()
    }

    /// Get the underlying Arc for cheap cloning.
    pub fn as_arc(&self) -> &Arc<[Arc<str>]> {
        &self.items
    }
}

impl Default for SortedCompletions {
    fn default() -> Self {
        Self::empty()
    }
}

/// A bounded conversation history that limits both message count and token usage.
///
/// Automatically evicts oldest messages when limits are exceeded, ensuring
/// memory usage stays bounded in long-running sessions.
#[derive(Debug, Clone)]
pub struct BoundedConversation<T> {
    /// Messages stored in order (oldest first).
    messages: VecDeque<T>,

    /// Maximum number of messages to retain.
    max_messages: usize,

    /// Maximum total tokens to retain.
    max_tokens: usize,

    /// Current total token count.
    current_tokens: usize,

    /// Function to extract token count from a message.
    /// Stored as a function pointer for simplicity.
    token_counter: fn(&T) -> usize,
}

impl<T> BoundedConversation<T> {
    /// Create a new bounded conversation.
    ///
    /// # Arguments
    /// * `max_messages` - Maximum number of messages to retain
    /// * `max_tokens` - Maximum total tokens to retain
    /// * `token_counter` - Function to count tokens in a message
    pub fn new(max_messages: usize, max_tokens: usize, token_counter: fn(&T) -> usize) -> Self {
        Self {
            messages: VecDeque::with_capacity(max_messages.min(1000)),
            max_messages,
            max_tokens,
            current_tokens: 0,
            token_counter,
        }
    }

    /// Push a message, evicting old messages if necessary.
    pub fn push(&mut self, message: T) {
        let msg_tokens = (self.token_counter)(&message);

        // Add the message
        self.messages.push_back(message);
        self.current_tokens += msg_tokens;

        // Evict old messages if over message limit
        while self.messages.len() > self.max_messages {
            if let Some(old) = self.messages.pop_front() {
                self.current_tokens = self
                    .current_tokens
                    .saturating_sub((self.token_counter)(&old));
            }
        }

        // Evict old messages if over token limit (keep at least 2 messages)
        while self.current_tokens > self.max_tokens && self.messages.len() > 2 {
            if let Some(old) = self.messages.pop_front() {
                self.current_tokens = self
                    .current_tokens
                    .saturating_sub((self.token_counter)(&old));
            }
        }
    }

    /// Get the number of messages.
    pub fn len(&self) -> usize {
        self.messages.len()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.messages.is_empty()
    }

    /// Get the current token count.
    pub fn token_count(&self) -> usize {
        self.current_tokens
    }

    /// Get remaining token budget.
    pub fn remaining_tokens(&self) -> usize {
        self.max_tokens.saturating_sub(self.current_tokens)
    }

    /// Get remaining message capacity.
    pub fn remaining_messages(&self) -> usize {
        self.max_messages.saturating_sub(self.messages.len())
    }

    /// Check if at message capacity.
    pub fn is_message_full(&self) -> bool {
        self.messages.len() >= self.max_messages
    }

    /// Check if at token capacity.
    pub fn is_token_full(&self) -> bool {
        self.current_tokens >= self.max_tokens
    }

    /// Clear all messages.
    pub fn clear(&mut self) {
        self.messages.clear();
        self.current_tokens = 0;
    }

    /// Iterate over messages (oldest first).
    pub fn iter(&self) -> impl Iterator<Item = &T> + DoubleEndedIterator {
        self.messages.iter()
    }

    /// Get the last message.
    pub fn last(&self) -> Option<&T> {
        self.messages.back()
    }

    /// Get the first message.
    pub fn first(&self) -> Option<&T> {
        self.messages.front()
    }

    /// Get a message by index.
    pub fn get(&self, index: usize) -> Option<&T> {
        self.messages.get(index)
    }

    /// Convert to a Vec (for serialization).
    pub fn to_vec(&self) -> Vec<T>
    where
        T: Clone,
    {
        self.messages.iter().cloned().collect()
    }

    /// Get statistics about the conversation.
    pub fn stats(&self) -> ConversationStats {
        ConversationStats {
            message_count: self.messages.len(),
            max_messages: self.max_messages,
            token_count: self.current_tokens,
            max_tokens: self.max_tokens,
            utilization_messages: self.messages.len() as f64 / self.max_messages as f64,
            utilization_tokens: self.current_tokens as f64 / self.max_tokens as f64,
        }
    }
}

/// Statistics about a bounded conversation.
#[derive(Debug, Clone)]
pub struct ConversationStats {
    pub message_count: usize,
    pub max_messages: usize,
    pub token_count: usize,
    pub max_tokens: usize,
    pub utilization_messages: f64,
    pub utilization_tokens: f64,
}

impl ConversationStats {
    /// Get a human-readable summary.
    pub fn summary(&self) -> String {
        format!(
            "{}/{} messages ({:.1}%), {}/{} tokens ({:.1}%)",
            self.message_count,
            self.max_messages,
            self.utilization_messages * 100.0,
            self.token_count,
            self.max_tokens,
            self.utilization_tokens * 100.0
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bounded_vec_push() {
        let mut vec = BoundedVec::with_capacity(3);
        vec.push(1);
        vec.push(2);
        vec.push(3);
        assert_eq!(vec.len(), 3);

        vec.push(4);
        assert_eq!(vec.len(), 3);
        assert_eq!(vec.first(), Some(&2)); // 1 was dropped
        assert_eq!(vec.last(), Some(&4));
    }

    #[test]
    fn test_bounded_vec_iter() {
        let mut vec = BoundedVec::with_capacity(5);
        for i in 0..5 {
            vec.push(i);
        }

        let collected: Vec<_> = vec.iter().copied().collect();
        assert_eq!(collected, vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn test_lru_string_cache() {
        let mut cache = LruStringCache::new(3);

        let s1 = cache.get_or_intern("hello");
        let _s2 = cache.get_or_intern("world");
        let s3 = cache.get_or_intern("hello"); // Should return same Arc

        assert!(Arc::ptr_eq(&s1, &s3));
        assert_eq!(cache.len(), 2);

        // Fill cache
        cache.get_or_intern("foo");
        cache.get_or_intern("bar"); // Should evict "world"

        assert!(cache.contains("hello"));
        assert!(!cache.contains("world"));
    }

    #[test]
    fn test_cache_get_or_insert() {
        let mut cache: Cache<String, i32> = Cache::new(10);

        let Some(val) = cache.get_or_insert_with("key".to_string(), || 42) else {
            panic!("inserted key should be readable");
        };
        assert_eq!(*val, 42);

        // Should return cached value
        let Some(val2) = cache.get_or_insert_with("key".to_string(), || 100) else {
            panic!("cached key should be readable");
        };
        assert_eq!(*val2, 42);
    }

    #[test]
    fn test_bounded_map() {
        let mut map = BoundedMap::with_capacity(3);
        map.insert("a", 1);
        map.insert("b", 2);
        map.insert("c", 3);

        assert_eq!(map.len(), 3);

        map.insert("d", 4); // Should evict "a"
        assert!(!map.contains_key(&"a"));
        assert!(map.contains_key(&"d"));
    }

    #[test]
    fn test_bounded_conversation_message_limit() {
        // Simple message type for testing
        #[derive(Clone)]
        struct TestMsg {
            tokens: usize,
        }

        fn count_tokens(msg: &TestMsg) -> usize {
            msg.tokens
        }

        let mut conv = BoundedConversation::new(3, 1000, count_tokens);

        conv.push(TestMsg { tokens: 10 });
        conv.push(TestMsg { tokens: 20 });
        conv.push(TestMsg { tokens: 30 });

        assert_eq!(conv.len(), 3);
        assert_eq!(conv.token_count(), 60);

        // Push one more - should evict oldest
        conv.push(TestMsg { tokens: 40 });

        assert_eq!(conv.len(), 3);
        assert_eq!(conv.token_count(), 90); // 20 + 30 + 40
    }

    #[test]
    fn test_bounded_conversation_token_limit() {
        #[derive(Clone)]
        struct TestMsg {
            tokens: usize,
        }

        fn count_tokens(msg: &TestMsg) -> usize {
            msg.tokens
        }

        let mut conv = BoundedConversation::new(100, 50, count_tokens);

        conv.push(TestMsg { tokens: 10 });
        conv.push(TestMsg { tokens: 20 });
        conv.push(TestMsg { tokens: 30 });

        // Total is 60, over limit of 50
        // Should evict until under limit (keeping at least 2)
        assert!(conv.len() >= 2);
        assert!(conv.token_count() <= 50 || conv.len() == 2);
    }

    #[test]
    fn test_sorted_completions_filter_prefix() {
        let completions = SortedCompletions::from_iter(vec![
            "apple",
            "application",
            "banana",
            "bandana",
            "cherry",
            "apply",
        ]);

        // Filter by "app"
        let filtered = completions.filter_prefix("app");
        let matches: Vec<&str> = filtered.iter().map(|s| s.as_ref()).collect();
        assert_eq!(matches, vec!["apple", "application", "apply"]);

        // Filter by "ban"
        let filtered = completions.filter_prefix("ban");
        let matches: Vec<&str> = filtered.iter().map(|s| s.as_ref()).collect();
        assert_eq!(matches, vec!["banana", "bandana"]);

        // Filter by non-existent prefix
        let matches = completions.filter_prefix("xyz");
        assert!(matches.is_empty());

        // Empty prefix returns all
        let matches = completions.filter_prefix("");
        assert_eq!(matches.len(), 6);
    }

    #[test]
    fn test_sorted_completions_contains() {
        let completions = SortedCompletions::from_iter(vec!["alpha", "beta", "gamma"]);

        assert!(completions.contains("alpha"));
        assert!(completions.contains("beta"));
        assert!(!completions.contains("delta"));
        assert!(!completions.contains("alph")); // Prefix, not exact match
    }

    #[test]
    fn test_bounded_conversation_stats() {
        #[derive(Clone)]
        struct TestMsg {
            tokens: usize,
        }

        fn count_tokens(msg: &TestMsg) -> usize {
            msg.tokens
        }

        let mut conv = BoundedConversation::new(10, 100, count_tokens);

        conv.push(TestMsg { tokens: 25 });
        conv.push(TestMsg { tokens: 25 });

        let stats = conv.stats();
        assert_eq!(stats.message_count, 2);
        assert_eq!(stats.token_count, 50);
        assert!((stats.utilization_messages - 0.2).abs() < 0.01);
        assert!((stats.utilization_tokens - 0.5).abs() < 0.01);
    }
}
