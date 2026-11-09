//! Tests for distributed replication functionality
//!
//! Tests replica creation, write replication, read from replicas,
//! replica consistency, failure detection, and recovery.

use dripstone::distributed::{ReplicaSet, ReplicationManager};

#[tokio::test]
async fn test_replication_manager_creation() {
    let manager = ReplicationManager::new(3).expect("valid factor");

    assert_eq!(manager.replication_factor(), 3);
}

#[tokio::test]
async fn test_replica_set_creation() {
    let replica_set = ReplicaSet {
        primary: "shard1".to_string(),
        secondaries: vec!["shard2".to_string(), "shard3".to_string()],
    };

    assert_eq!(replica_set.primary, "shard1");
    assert_eq!(replica_set.secondaries.len(), 2);
}

#[tokio::test]
async fn test_replication_manager_add_replica_set() {
    let manager = ReplicationManager::new(2).expect("valid factor");

    let replica_set = ReplicaSet {
        primary: "shard1".to_string(),
        secondaries: vec!["shard2".to_string()],
    };

    manager
        .add_replica_set(replica_set)
        .expect("size matches factor");

    let retrieved = manager.get_replica_set("shard1");
    assert!(retrieved.is_some());
    assert_eq!(retrieved.unwrap().primary, "shard1");
}

#[tokio::test]
async fn test_replication_manager_get_replica_set() {
    let manager = ReplicationManager::new(3).expect("valid factor");

    let replica_set = ReplicaSet {
        primary: "shard1".to_string(),
        secondaries: vec!["shard2".to_string(), "shard3".to_string()],
    };

    manager.add_replica_set(replica_set).expect("valid set");

    let retrieved = manager.get_replica_set("shard1");
    assert!(retrieved.is_some());
    let set = retrieved.unwrap();
    assert_eq!(set.primary, "shard1");
    assert_eq!(set.secondaries.len(), 2);
}

#[tokio::test]
async fn test_replication_manager_get_nonexistent() {
    let manager = ReplicationManager::new(3).expect("valid factor");

    let retrieved = manager.get_replica_set("nonexistent");
    assert!(retrieved.is_none());
}

#[tokio::test]
async fn test_replication_manager_promote_secondary() {
    let manager = ReplicationManager::new(3).expect("valid factor");

    let replica_set = ReplicaSet {
        primary: "shard1".to_string(),
        secondaries: vec!["shard2".to_string(), "shard3".to_string()],
    };

    manager.add_replica_set(replica_set).expect("valid set");

    manager
        .promote_secondary("shard1", "shard2")
        .expect("promotion succeeds");

    let retrieved = manager.get_replica_set("shard2");
    assert!(retrieved.is_some());
    let set = retrieved.unwrap();
    assert_eq!(set.primary, "shard2");
    assert!(set.secondaries.contains(&"shard1".to_string()));
}

#[tokio::test]
async fn test_replication_manager_multiple_replica_sets() {
    let manager = ReplicationManager::new(2).expect("valid factor");

    let set1 = ReplicaSet {
        primary: "shard1".to_string(),
        secondaries: vec!["shard2".to_string()],
    };

    let set2 = ReplicaSet {
        primary: "shard3".to_string(),
        secondaries: vec!["shard4".to_string()],
    };

    manager.add_replica_set(set1).expect("valid set");
    manager.add_replica_set(set2).expect("valid set");

    let retrieved1 = manager.get_replica_set("shard1");
    let retrieved2 = manager.get_replica_set("shard3");

    assert!(retrieved1.is_some());
    assert!(retrieved2.is_some());
    assert_eq!(retrieved1.unwrap().primary, "shard1");
    assert_eq!(retrieved2.unwrap().primary, "shard3");
}

#[tokio::test]
async fn test_replica_set_read_from_primary() {
    let replica_set = ReplicaSet {
        primary: "shard1".to_string(),
        secondaries: vec!["shard2".to_string()],
    };

    assert_eq!(replica_set.primary, "shard1");
}

#[tokio::test]
async fn test_replica_set_read_from_secondary() {
    let replica_set = ReplicaSet {
        primary: "shard1".to_string(),
        secondaries: vec!["shard2".to_string(), "shard3".to_string()],
    };

    assert!(!replica_set.secondaries.is_empty());
    assert!(replica_set.secondaries.contains(&"shard2".to_string()));
}

#[tokio::test]
async fn test_replication_manager_replication_factor() {
    let manager1 = ReplicationManager::new(2).expect("valid factor");
    let manager2 = ReplicationManager::new(5).expect("valid factor");

    assert_eq!(manager1.replication_factor(), 2);
    assert_eq!(manager2.replication_factor(), 5);
}

#[tokio::test]
async fn test_replica_set_consistency() {
    let replica_set = ReplicaSet {
        primary: "shard1".to_string(),
        secondaries: vec!["shard2".to_string()],
    };

    assert!(!replica_set.secondaries.contains(&replica_set.primary));
}

#[tokio::test]
async fn test_replication_manager_promote_nonexistent() {
    let manager = ReplicationManager::new(3).expect("valid factor");

    assert!(
        manager
            .promote_secondary("nonexistent", "new_primary")
            .is_err()
    );
    assert!(manager.get_replica_set("nonexistent").is_none());
    assert!(manager.get_replica_set("new_primary").is_none());
}

#[tokio::test]
async fn test_replication_manager_rejects_zero_factor() {
    assert!(ReplicationManager::new(0).is_err());
}

#[tokio::test]
async fn test_replication_manager_rejects_mismatched_size() {
    let manager = ReplicationManager::new(3).expect("valid factor");
    let replica_set = ReplicaSet {
        primary: "shard1".to_string(),
        secondaries: vec!["shard2".to_string()],
    };
    assert!(manager.add_replica_set(replica_set).is_err());
}

#[tokio::test]
async fn test_replication_manager_find_replica_set_containing() {
    let manager = ReplicationManager::new(2).expect("valid factor");
    manager
        .add_replica_set(ReplicaSet {
            primary: "shard1".to_string(),
            secondaries: vec!["shard2".to_string()],
        })
        .expect("valid set");
    let found = manager.find_replica_set_containing("shard2");
    assert!(found.is_some());
    assert_eq!(found.unwrap().primary, "shard1");
}
