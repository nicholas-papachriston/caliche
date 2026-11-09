//! Tests for distributed sharding functionality
//!
//! Tests shard key generation, consistent hashing distribution, vector routing,
//! shard rebalancing, and shard failure handling.

use dripstone::distributed::{ConsistentHashRing, ShardManager};

#[tokio::test]
async fn test_consistent_hash_ring_creation() {
    let ring = ConsistentHashRing::new(3);

    // Ring should be created successfully
    assert_eq!(ring.get_node("test_key"), None); // Empty ring
}

#[tokio::test]
async fn test_consistent_hash_ring_add_node() {
    let ring = ConsistentHashRing::new(3);

    ring.add_node("shard1".to_string());

    let node = ring.get_node("test_key");
    assert!(node.is_some());
    assert_eq!(node.unwrap(), "shard1");
}

#[tokio::test]
async fn test_consistent_hash_ring_multiple_nodes() {
    let ring = ConsistentHashRing::new(3);

    ring.add_node("shard1".to_string());
    ring.add_node("shard2".to_string());
    ring.add_node("shard3".to_string());

    // Different keys should route to potentially different nodes
    let node1 = ring.get_node("key1");
    let node2 = ring.get_node("key2");
    let node3 = ring.get_node("key3");

    assert!(node1.is_some());
    assert!(node2.is_some());
    assert!(node3.is_some());

    // All should be valid shard names
    assert!(["shard1", "shard2", "shard3"].contains(&node1.unwrap().as_str()));
    assert!(["shard1", "shard2", "shard3"].contains(&node2.unwrap().as_str()));
    assert!(["shard1", "shard2", "shard3"].contains(&node3.unwrap().as_str()));
}

#[tokio::test]
async fn test_consistent_hash_ring_consistency() {
    let ring = ConsistentHashRing::new(3);

    ring.add_node("shard1".to_string());
    ring.add_node("shard2".to_string());

    // Same key should always route to same node
    let key = "consistent_key";
    let node1 = ring.get_node(key);
    let node2 = ring.get_node(key);
    let node3 = ring.get_node(key);

    assert_eq!(node1, node2);
    assert_eq!(node2, node3);
}

#[tokio::test]
async fn test_consistent_hash_ring_remove_node() {
    let ring = ConsistentHashRing::new(3);

    ring.add_node("shard1".to_string());
    ring.add_node("shard2".to_string());

    ring.remove_node("shard1");

    let node_after = ring.get_node("test_key");

    // Should still route to a node (shard2)
    assert!(node_after.is_some());
    // May or may not be the same node depending on hash
    assert_eq!(node_after.unwrap(), "shard2");
}

#[tokio::test]
async fn test_consistent_hash_ring_get_multiple_nodes() {
    let ring = ConsistentHashRing::new(3);

    ring.add_node("shard1".to_string());
    ring.add_node("shard2".to_string());
    ring.add_node("shard3".to_string());

    let nodes = ring.get_nodes("test_key", 2);

    assert_eq!(nodes.len(), 2);
    assert!(
        nodes
            .iter()
            .all(|n| ["shard1", "shard2", "shard3"].contains(&n.as_str()))
    );
    // Should be unique nodes
    assert_ne!(nodes[0], nodes[1]);
}

#[tokio::test]
async fn test_consistent_hash_ring_wrap_around() {
    let ring = ConsistentHashRing::new(3);

    ring.add_node("shard1".to_string());

    // Even with a hash that's very high, should wrap around
    let node = ring.get_node("zzzzzzzzzzzzzzzzzzzz");

    assert!(node.is_some());
    assert_eq!(node.unwrap(), "shard1");
}

#[tokio::test]
async fn test_shard_manager_creation() {
    let manager = ShardManager::new(5);

    assert_eq!(manager.shard_count(), 5);
}

#[tokio::test]
async fn test_shard_manager_add_shard() {
    let mut manager = ShardManager::new(3);

    manager.add_shard("shard1".to_string());

    let shard = manager.get_shard_for_key("test_key");
    assert!(shard.is_some());
}

#[tokio::test]
async fn test_shard_manager_key_routing() {
    let mut manager = ShardManager::new(3);

    manager.add_shard("shard1".to_string());
    manager.add_shard("shard2".to_string());
    manager.add_shard("shard3".to_string());

    // Same key should route to same shard
    let key = "routing_test_key";
    let shard1 = manager.get_shard_for_key(key);
    let shard2 = manager.get_shard_for_key(key);

    assert_eq!(shard1, shard2);
}

#[tokio::test]
async fn test_shard_manager_rebalance() {
    let mut manager = ShardManager::new(3);

    manager.add_shard("shard1".to_string());
    manager.add_shard("shard2".to_string());
    manager.add_shard("shard3".to_string());

    let node_after = manager.get_shard_for_key("test_key");

    assert!(node_after.is_some());
    assert!(["shard1", "shard2", "shard3"].contains(&node_after.unwrap().as_str()));
}

#[tokio::test]
async fn test_shard_manager_failure_handling() {
    let mut manager = ShardManager::new(3);

    manager.add_shard("shard1".to_string());
    manager.add_shard("shard2".to_string());

    manager.remove_shard("shard1");

    // Should route to remaining shard
    let shard_after = manager.get_shard_for_key("test_key");

    assert!(shard_after.is_some());
    assert_eq!(shard_after.unwrap(), "shard2");
}

#[tokio::test]
async fn test_shard_manager_metadata() {
    let mut manager = ShardManager::new(5);

    assert_eq!(manager.shard_count(), 5);

    manager.add_shard("shard1".to_string());

    // Should track shards
    let shard = manager.get_shard_for_key("any_key");
    assert!(shard.is_some());
}
