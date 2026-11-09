use caliche::distributed::coordinator::{NodeInfo, NodeStatus, Task};
use caliche::distributed::load_balancer::{LoadBalancer, RoutingStrategy};
use chrono::Utc;
use serde_json::json;

fn create_test_node(id: &str, load: usize, capacity: usize, status: NodeStatus) -> NodeInfo {
    NodeInfo {
        node_id: id.to_string(),
        address: format!("127.0.0.1:{}", 8000 + id.parse::<u16>().unwrap_or(0)),
        region: Some("us-east-1".to_string()),
        capabilities: vec!["chat".to_string()],
        current_load: load,
        max_capacity: capacity,
        last_heartbeat: Utc::now(),
        status,
    }
}

#[test]
fn test_node_info_serialization() {
    let node = create_test_node("1", 10, 100, NodeStatus::Healthy);

    let serialized = serde_json::to_string(&node).expect("Node serialization should succeed");
    let deserialized: NodeInfo =
        serde_json::from_str(&serialized).expect("Node deserialization should succeed");

    assert_eq!(node.node_id, deserialized.node_id);
    assert_eq!(node.current_load, deserialized.current_load);
    assert_eq!(node.status, deserialized.status);
}

#[test]
fn test_task_serialization() {
    let task = Task::new("chat_completion".to_string(), json!({ "message": "hello" }))
        .with_priority(1)
        .with_session_id("session_123".to_string());

    let serialized = serde_json::to_string(&task).expect("Task serialization should succeed");
    let deserialized: Task =
        serde_json::from_str(&serialized).expect("Task deserialization should succeed");

    assert_eq!(task.task_id, deserialized.task_id);
    assert_eq!(task.task_type, deserialized.task_type);
    assert_eq!(task.session_id, deserialized.session_id);
}

#[tokio::test]
async fn test_load_balancer_round_robin() {
    let lb = LoadBalancer::new(RoutingStrategy::RoundRobin);

    let nodes = vec![
        create_test_node("1", 10, 100, NodeStatus::Healthy),
        create_test_node("2", 20, 100, NodeStatus::Healthy),
        create_test_node("3", 30, 100, NodeStatus::Healthy),
    ];

    // First call should return first node (or start of sequence)
    let first = lb
        .select_node(&nodes)
        .await
        .expect("Node selection should succeed");

    // Second call should return next node
    let second = lb
        .select_node(&nodes)
        .await
        .expect("Node selection should succeed");

    // Third call should return next node
    let third = lb
        .select_node(&nodes)
        .await
        .expect("Node selection should succeed");

    // Fourth call should wrap around
    let fourth = lb
        .select_node(&nodes)
        .await
        .expect("Node selection should succeed");

    // Logic check: Round Robin should visit all nodes
    let selected = vec![first, second, third];
    assert!(selected.contains(&"1".to_string()));
    assert!(selected.contains(&"2".to_string()));
    assert!(selected.contains(&"3".to_string()));

    // Sequence should repeat
    assert_eq!(fourth, selected[0]);
}

#[tokio::test]
async fn test_load_balancer_least_connections() {
    let lb = LoadBalancer::new(RoutingStrategy::LeastConnections);

    let nodes = vec![
        create_test_node("1", 50, 100, NodeStatus::Healthy),
        create_test_node("2", 10, 100, NodeStatus::Healthy), // Lowest load
        create_test_node("3", 80, 100, NodeStatus::Healthy),
    ];

    let selected = lb
        .select_node(&nodes)
        .await
        .expect("Node selection should succeed");
    assert_eq!(selected, "2");
}

#[tokio::test]
async fn test_load_balancer_weighted_round_robin() {
    let lb = LoadBalancer::new(RoutingStrategy::WeightedRoundRobin);

    let nodes = vec![
        create_test_node("1", 90, 100, NodeStatus::Healthy), // 10 capacity left
        create_test_node("2", 50, 100, NodeStatus::Healthy), // 50 capacity left
        create_test_node("3", 10, 20, NodeStatus::Healthy),  // 10 capacity left
    ];

    // Weighted RR in this impl selects max remaining capacity
    let selected = lb
        .select_node(&nodes)
        .await
        .expect("Node selection should succeed");
    assert_eq!(selected, "2");
}

#[tokio::test]
async fn test_load_balancer_ignore_unhealthy() {
    let lb = LoadBalancer::new(RoutingStrategy::RoundRobin);

    let nodes = vec![
        create_test_node("1", 10, 100, NodeStatus::Unhealthy),
        create_test_node("2", 10, 100, NodeStatus::Offline),
        create_test_node("3", 10, 100, NodeStatus::Healthy),
    ];

    let selected = lb
        .select_node(&nodes)
        .await
        .expect("Node selection should succeed");
    assert_eq!(selected, "3");
}

#[tokio::test]
async fn test_load_balancer_no_healthy_nodes() {
    let lb = LoadBalancer::new(RoutingStrategy::RoundRobin);

    let nodes = vec![
        create_test_node("1", 10, 100, NodeStatus::Unhealthy),
        create_test_node("2", 10, 100, NodeStatus::Offline),
    ];

    let selected = lb.select_node(&nodes).await;
    assert!(selected.is_none());
}
