use parking_lot::RwLock;
use siphasher::sip::SipHasher13;
use std::collections::BTreeMap;
use std::hash::Hasher;
use std::sync::Arc;

/// Default number of virtual nodes per physical shard on the hash ring.
///
/// 150 is a common setting (used by e.g. Riak / Cassandra style rings) and
/// provides a reasonable distribution without exploding the ring size.
pub const DEFAULT_VIRTUAL_REPLICAS: usize = 150;

/// Consistent hash ring for distributing embeddings across shards.
///
/// Uses [`SipHasher13`] with fixed keys so that the mapping from keys to shards
/// is stable across processes, machines, and Rust versions -- a requirement
/// for any real distributed deployment. The standard library's
/// `DefaultHasher` is deliberately *not* stable across builds and therefore
/// cannot be used when multiple participants must agree on shard placement.
pub struct ConsistentHashRing {
    ring: Arc<RwLock<BTreeMap<u64, String>>>,
    replicas: usize,
}

impl ConsistentHashRing {
    /// Create a new ring with the given number of virtual replicas per node.
    #[must_use]
    pub fn new(replicas: u32) -> Self {
        Self {
            ring: Arc::new(RwLock::new(BTreeMap::new())),
            replicas: usize::try_from(replicas).unwrap_or(usize::MAX),
        }
    }

    /// Stable SipHash of `key` (fixed keys; identical across processes/builds).
    #[must_use]
    pub fn hash_key(key: &str) -> u64 {
        // Fixed keys so every process computes the same hash for the same input.
        let mut hasher = SipHasher13::new_with_keys(0, 0);
        hasher.write(key.as_bytes());
        hasher.finish()
    }

    pub fn add_node(&self, node: String) {
        let mut ring = self.ring.write();
        for i in 0..self.replicas {
            let virtual_key = format!("{node}:{i}");
            let hash = Self::hash_key(&virtual_key);
            if i + 1 == self.replicas {
                ring.insert(hash, node);
                break;
            }
            ring.insert(hash, node.clone());
        }
    }

    pub fn remove_node(&self, node: &str) {
        let mut ring = self.ring.write();
        for i in 0..self.replicas {
            let virtual_key = format!("{node}:{i}");
            let hash = Self::hash_key(&virtual_key);
            ring.remove(&hash);
        }
    }

    #[must_use]
    pub fn get_node(&self, key: &str) -> Option<String> {
        let ring = self.ring.read();
        if ring.is_empty() {
            return None;
        }

        let hash = Self::hash_key(key);

        if let Some((_hash, node)) = ring.range(hash..).next() {
            return Some(node.clone());
        }

        ring.values().next().cloned()
    }

    #[must_use]
    pub fn get_nodes(&self, key: &str, count: usize) -> Vec<String> {
        if count == 0 {
            return Vec::new();
        }
        let ring = self.ring.read();
        if ring.is_empty() {
            return Vec::new();
        }

        let hash = Self::hash_key(key);
        let mut result = Vec::new();
        let mut seen = std::collections::HashSet::new();

        for (_hash, node) in ring.range(hash..) {
            if seen.insert(node.clone()) {
                result.push(node.clone());
                if result.len() >= count {
                    break;
                }
            }
        }

        if result.len() < count {
            for (_hash, node) in ring.iter() {
                if seen.insert(node.clone()) {
                    result.push(node.clone());
                    if result.len() >= count {
                        break;
                    }
                }
            }
        }

        drop(ring);
        result
    }
}

/// Manages distribution of data across shards using [`ConsistentHashRing`].
pub struct ShardManager {
    hash_ring: ConsistentHashRing,
    shard_count: usize,
}

impl ShardManager {
    /// Create a new shard manager with [`DEFAULT_VIRTUAL_REPLICAS`] replicas per shard.
    #[must_use]
    pub fn new(shard_count: usize) -> Self {
        Self::with_replicas(shard_count, DEFAULT_VIRTUAL_REPLICAS)
    }

    /// Create a new shard manager with a custom number of virtual replicas per
    /// physical shard. Higher values smooth out shard size variance at the cost
    /// of a larger ring; 100-500 is typical.
    #[must_use]
    pub fn with_replicas(shard_count: usize, replicas: usize) -> Self {
        let replicas = replicas.max(1);
        let replicas = u32::try_from(replicas).unwrap_or(u32::MAX);
        let hash_ring = ConsistentHashRing::new(replicas);

        for i in 0..shard_count {
            hash_ring.add_node(format!("shard-{i}"));
        }

        Self {
            hash_ring,
            shard_count,
        }
    }

    #[must_use]
    pub fn get_shard_for_key(&self, key: &str) -> Option<String> {
        self.hash_ring.get_node(key)
    }

    /// Turbopuffer-style shard index: `hash(key) % shard_count`.
    ///
    /// Prefer this for single-WAL multi-shard namespaces where every consumer
    /// must agree on ownership without consulting a dynamic ring view.
    #[must_use]
    pub fn shard_index_for_key(&self, key: &str) -> Option<usize> {
        if self.shard_count == 0 {
            return None;
        }
        let hash = ConsistentHashRing::hash_key(key);
        Some((hash as usize) % self.shard_count)
    }

    #[must_use]
    pub fn get_replicas_for_key(&self, key: &str, replica_count: usize) -> Vec<String> {
        self.hash_ring.get_nodes(key, replica_count)
    }

    pub fn add_shard(&mut self, shard_id: String) {
        self.hash_ring.add_node(shard_id);
        self.shard_count += 1;
    }

    pub fn remove_shard(&mut self, shard_id: &str) {
        self.hash_ring.remove_node(shard_id);
        self.shard_count = self.shard_count.saturating_sub(1);
    }

    #[must_use]
    pub const fn shard_count(&self) -> usize {
        self.shard_count
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hash_is_stable_across_calls() {
        let a = ConsistentHashRing::hash_key("foo");
        let b = ConsistentHashRing::hash_key("foo");
        assert_eq!(a, b);
        let c = ConsistentHashRing::hash_key("bar");
        assert_ne!(a, c);
    }

    #[test]
    fn shard_manager_distributes_keys() {
        let manager = ShardManager::new(4);
        let mut counts = std::collections::HashMap::new();
        for i in 0..1000 {
            if let Some(shard) = manager.get_shard_for_key(&format!("key-{i}")) {
                *counts.entry(shard).or_insert(0) += 1;
            }
        }
        assert_eq!(counts.len(), 4);
    }

    #[test]
    fn shard_index_for_key_is_modulo_stable() {
        let manager = ShardManager::new(7);
        for i in 0..200 {
            let key = format!("doc-{i}");
            let a = manager.shard_index_for_key(&key);
            let b = manager.shard_index_for_key(&key);
            assert_eq!(a, b);
            assert!(a.expect("index") < 7);
        }
    }
}
