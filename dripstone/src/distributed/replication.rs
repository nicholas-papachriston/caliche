//! Replication policy helpers.
//!
//! `ReplicationManager` is a pure policy/metadata helper: it tracks which
//! nodes participate in each replica set and exposes lookup utilities. It
//! does **not** perform I/O, wire replication, or drive failover on its own.
//! Callers (higher-level coordinators, shard routers, etc.) are responsible
//! for turning these policy decisions into actual network operations.

use crate::errors::DripstoneError;
use parking_lot::RwLock;
use std::sync::Arc;

#[derive(Debug, Clone)]
pub struct ReplicaSet {
    pub primary: String,
    pub secondaries: Vec<String>,
}

impl ReplicaSet {
    /// Total replica count (primary + secondaries).
    #[must_use]
    pub const fn size(&self) -> usize {
        1 + self.secondaries.len()
    }

    /// Returns true if `node` is the primary or a secondary in this set.
    #[must_use]
    pub fn contains(&self, node: &str) -> bool {
        self.primary == node || self.secondaries.iter().any(|s| s == node)
    }
}

/// Manages replication policy for embedding data across nodes.
///
/// This is an in-memory view of the current replica layout. It does not
/// perform writes, reads, or health checks; integrate it with a transport
/// layer to drive real replication.
pub struct ReplicationManager {
    replica_sets: Arc<RwLock<Vec<ReplicaSet>>>,
    replication_factor: usize,
}

impl ReplicationManager {
    /// Construct a new manager.
    ///
    /// `replication_factor` is the desired number of replicas per shard
    /// (primary + secondaries). A value of 0 is rejected.
    ///
    /// # Errors
    ///
    /// Returns an error when the operation cannot be completed.
    pub fn new(replication_factor: usize) -> Result<Self, DripstoneError> {
        if replication_factor == 0 {
            return Err(DripstoneError::Misconfiguration(
                "replication_factor must be >= 1".to_string(),
            ));
        }
        Ok(Self {
            replica_sets: Arc::new(RwLock::new(Vec::new())),
            replication_factor,
        })
    }

    /// Register a replica set.
    ///
    /// The set is validated against the configured `replication_factor`:
    /// its size (primary + secondaries) must match exactly.
    ///
    /// # Errors
    ///
    /// Returns an error when the operation cannot be completed.
    pub fn add_replica_set(&self, replica_set: ReplicaSet) -> Result<(), DripstoneError> {
        if replica_set.size() != self.replication_factor {
            return Err(DripstoneError::Misconfiguration(format!(
                "replica set size {} does not match configured replication_factor {}",
                replica_set.size(),
                self.replication_factor
            )));
        }
        if replica_set.primary.is_empty() {
            return Err(DripstoneError::Misconfiguration(
                "replica set primary cannot be empty".to_string(),
            ));
        }
        if replica_set.secondaries.iter().any(String::is_empty) {
            return Err(DripstoneError::Misconfiguration(
                "replica set secondaries cannot contain empty node ids".to_string(),
            ));
        }
        let mut sets = self.replica_sets.write();
        sets.push(replica_set);
        drop(sets);
        Ok(())
    }

    /// Find a replica set by primary id.
    #[must_use]
    pub fn get_replica_set(&self, shard_id: &str) -> Option<ReplicaSet> {
        let sets = self.replica_sets.read();
        sets.iter().find(|set| set.primary == shard_id).cloned()
    }

    /// Find the replica set containing `node`, whether as primary or secondary.
    #[must_use]
    pub fn find_replica_set_containing(&self, node: &str) -> Option<ReplicaSet> {
        let sets = self.replica_sets.read();
        sets.iter().find(|set| set.contains(node)).cloned()
    }

    /// Promote `new_primary` (currently a secondary) to primary of the set
    /// whose current primary is `old_primary`.
    ///
    /// Returns an error if the set is not found or `new_primary` is not
    /// currently a secondary of that set.
    ///
    /// # Errors
    ///
    /// Returns an error when the operation cannot be completed.
    pub fn promote_secondary(
        &self,
        old_primary: &str,
        new_primary: &str,
    ) -> Result<(), DripstoneError> {
        let mut sets = self.replica_sets.write();
        let Some(set) = sets.iter_mut().find(|s| s.primary == old_primary) else {
            return Err(DripstoneError::Misconfiguration(format!(
                "no replica set with primary '{old_primary}'"
            )));
        };
        if !set.secondaries.iter().any(|s| s == new_primary) {
            return Err(DripstoneError::Misconfiguration(format!(
                "'{new_primary}' is not a secondary of '{old_primary}'"
            )));
        }
        set.secondaries.retain(|s| s != new_primary);
        set.secondaries.push(set.primary.clone());
        set.primary = new_primary.to_string();
        drop(sets);
        Ok(())
    }

    #[must_use]
    pub const fn replication_factor(&self) -> usize {
        self.replication_factor
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_zero_rf() {
        assert!(ReplicationManager::new(0).is_err());
    }

    #[test]
    fn add_requires_matching_size() {
        let mgr = ReplicationManager::new(3).unwrap();
        let set = ReplicaSet {
            primary: "a".to_string(),
            secondaries: vec!["b".to_string()],
        };
        assert!(mgr.add_replica_set(set).is_err());
    }

    #[test]
    fn find_containing_matches_secondary() {
        let mgr = ReplicationManager::new(2).unwrap();
        mgr.add_replica_set(ReplicaSet {
            primary: "a".to_string(),
            secondaries: vec!["b".to_string()],
        })
        .unwrap();
        let found = mgr.find_replica_set_containing("b").unwrap();
        assert_eq!(found.primary, "a");
    }

    #[test]
    fn promote_checks_membership() {
        let mgr = ReplicationManager::new(2).unwrap();
        mgr.add_replica_set(ReplicaSet {
            primary: "a".to_string(),
            secondaries: vec!["b".to_string()],
        })
        .unwrap();
        assert!(mgr.promote_secondary("a", "c").is_err());
        mgr.promote_secondary("a", "b").unwrap();
        let set = mgr.get_replica_set("b").unwrap();
        assert_eq!(set.secondaries, vec!["a".to_string()]);
    }
}
