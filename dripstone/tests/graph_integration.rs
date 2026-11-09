//! Integration tests for graph backend

use dripstone::graph::memory::InMemoryGraphBackend;
use dripstone::graph::{Edge, GraphBackend, Node};
use serde_json::json;

#[tokio::test]
async fn test_graph_operations() {
    let backend = InMemoryGraphBackend::new();

    // Create nodes
    let alice = Node::new("alice".to_string(), "Person".to_string())
        .with_property("name".to_string(), json!("Alice"))
        .with_property("age".to_string(), json!(30));

    let bob = Node::new("bob".to_string(), "Person".to_string())
        .with_property("name".to_string(), json!("Bob"))
        .with_property("age".to_string(), json!(25));

    let acme = Node::new("acme".to_string(), "Company".to_string())
        .with_property("name".to_string(), json!("Acme Corp"));

    // Add nodes
    backend.add_node(alice.clone()).await.unwrap();
    backend.add_node(bob.clone()).await.unwrap();
    backend.add_node(acme.clone()).await.unwrap();

    // Create edges
    let edge1 = Edge::new(
        "alice".to_string(),
        "acme".to_string(),
        "WORKS_FOR".to_string(),
    )
    .with_property("since".to_string(), json!("2020"));

    let edge2 = Edge::new(
        "bob".to_string(),
        "acme".to_string(),
        "WORKS_FOR".to_string(),
    )
    .with_property("since".to_string(), json!("2021"));

    let edge3 =
        Edge::new("alice".to_string(), "bob".to_string(), "KNOWS".to_string()).with_weight(0.8);

    // Add edges
    backend.add_edge(edge1).await.unwrap();
    backend.add_edge(edge2).await.unwrap();
    backend.add_edge(edge3).await.unwrap();

    // Test get_node
    let retrieved_alice = backend.get_node("alice").await.unwrap().unwrap();
    assert_eq!(retrieved_alice.id, "alice");
    assert_eq!(
        retrieved_alice.properties.get("name"),
        Some(&json!("Alice"))
    );

    // Test get_neighbors
    let alice_neighbors = backend.get_neighbors("alice", None).await.unwrap();
    assert_eq!(alice_neighbors.len(), 2); // acme and bob

    // Test filtered neighbors
    let work_relations = backend
        .get_neighbors("alice", Some(&["WORKS_FOR"]))
        .await
        .unwrap();
    assert_eq!(work_relations.len(), 1);
    assert_eq!(work_relations[0].node.id, "acme");

    // Test find_path
    let path = backend.find_path("bob", "acme", 2).await.unwrap();
    assert!(path.is_some());
    let path = path.unwrap();
    assert_eq!(path.nodes.len(), 2);
    assert_eq!(path.nodes[0].id, "bob");
    assert_eq!(path.nodes[1].id, "acme");

    // Test get_nodes_by_type
    let people = backend.get_nodes_by_type("Person").await.unwrap();
    assert_eq!(people.len(), 2);

    // Test get_edges_by_type
    let work_edges = backend.get_edges_by_type("WORKS_FOR").await.unwrap();
    assert_eq!(work_edges.len(), 2);

    // Test counts
    assert_eq!(backend.count_nodes().await.unwrap(), 3);
    assert_eq!(backend.count_edges().await.unwrap(), 3);

    // Test delete_edge
    backend.delete_edge("alice", "bob", "KNOWS").await.unwrap();
    assert_eq!(backend.count_edges().await.unwrap(), 2);

    // Test delete_node
    backend.delete_node("bob").await.unwrap();
    assert_eq!(backend.count_nodes().await.unwrap(), 2);
    assert_eq!(backend.count_edges().await.unwrap(), 1); // Only alice->acme remains
}

#[tokio::test]
async fn test_graph_with_embeddings() {
    let backend = InMemoryGraphBackend::new();

    let embedding = vec![0.1, 0.2, 0.3, 0.4];
    let node =
        Node::new("entity1".to_string(), "Concept".to_string()).with_embedding(embedding.clone());

    backend.add_node(node).await.unwrap();

    let retrieved = backend.get_node("entity1").await.unwrap().unwrap();
    assert_eq!(retrieved.embedding, Some(embedding));
}

#[tokio::test]
async fn test_path_finding_complex() {
    let backend = InMemoryGraphBackend::new();

    // Create a more complex graph: A -> B -> C -> D, and A -> D (shortcut)
    for id in ["a", "b", "c", "d"] {
        backend
            .add_node(Node::new(id.to_string(), "Node".to_string()))
            .await
            .unwrap();
    }

    backend
        .add_edge(Edge::new(
            "a".to_string(),
            "b".to_string(),
            "CONNECTS".to_string(),
        ))
        .await
        .unwrap();
    backend
        .add_edge(Edge::new(
            "b".to_string(),
            "c".to_string(),
            "CONNECTS".to_string(),
        ))
        .await
        .unwrap();
    backend
        .add_edge(Edge::new(
            "c".to_string(),
            "d".to_string(),
            "CONNECTS".to_string(),
        ))
        .await
        .unwrap();
    backend
        .add_edge(Edge::new(
            "a".to_string(),
            "d".to_string(),
            "CONNECTS".to_string(),
        ))
        .await
        .unwrap();

    // Should find the direct path (a -> d) even though longer path exists
    let path = backend.find_path("a", "d", 3).await.unwrap();
    assert!(path.is_some());
    let path = path.unwrap();
    // Should find shortest path (direct connection)
    assert_eq!(path.nodes.len(), 2);
    assert_eq!(path.nodes[0].id, "a");
    assert_eq!(path.nodes[1].id, "d");
}

#[tokio::test]
async fn test_error_handling() {
    let backend = InMemoryGraphBackend::new();

    // Try to add edge with non-existent node
    let edge = Edge::new(
        "nonexistent".to_string(),
        "target".to_string(),
        "RELATES".to_string(),
    );
    let result = backend.add_edge(edge).await;
    assert!(result.is_err());

    // Try to get non-existent node
    let result = backend.get_node("nonexistent").await.unwrap();
    assert!(result.is_none());

    // Try to get neighbors of non-existent node
    let result = backend.get_neighbors("nonexistent", None).await;
    assert!(result.is_err());
}
