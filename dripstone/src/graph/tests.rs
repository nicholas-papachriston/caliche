#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::memory::InMemoryGraphBackend;
    use serde_json::json;

    #[tokio::test]
    async fn test_add_and_get_node() {
        let backend = InMemoryGraphBackend::new();
        let node = Node::new("person1".to_string(), "Person".to_string())
            .with_property("name".to_string(), json!("Alice"));

        backend.add_node(node.clone()).await.expect("add_node should succeed");
        let retrieved = backend.get_node("person1").await.expect("get_node should succeed");
        assert!(retrieved.is_some());
        assert_eq!(retrieved.expect("node should exist").id, "person1");
    }

    #[tokio::test]
    async fn test_add_edge() {
        let backend = InMemoryGraphBackend::new();
        let alice = Node::new("alice".to_string(), "Person".to_string());
        let company = Node::new("acme".to_string(), "Company".to_string());

        backend.add_node(alice).await.expect("add_node should succeed");
        backend.add_node(company).await.expect("add_node should succeed");

        let edge = Edge::new(
            "alice".to_string(),
            "acme".to_string(),
            "WORKS_FOR".to_string(),
        );
        backend.add_edge(edge).await.expect("add_edge should succeed");

        let neighbors = backend.get_neighbors("alice", None).await.expect("get_neighbors should succeed");
        assert_eq!(neighbors.len(), 1);
        assert_eq!(neighbors[0].node.id, "acme");
        assert_eq!(neighbors[0].edge.edge_type, "WORKS_FOR");
    }

    #[tokio::test]
    async fn test_find_path() {
        let backend = InMemoryGraphBackend::new();
        
        // Create a chain: A -> B -> C
        backend.add_node(Node::new("a".to_string(), "Node".to_string())).await.expect("add_node should succeed");
        backend.add_node(Node::new("b".to_string(), "Node".to_string())).await.expect("add_node should succeed");
        backend.add_node(Node::new("c".to_string(), "Node".to_string())).await.expect("add_node should succeed");

        backend.add_edge(Edge::new("a".to_string(), "b".to_string(), "CONNECTS".to_string())).await.expect("add_edge should succeed");
        backend.add_edge(Edge::new("b".to_string(), "c".to_string(), "CONNECTS".to_string())).await.expect("add_edge should succeed");

        let path = backend.find_path("a", "c", 3).await.expect("find_path should succeed");
        assert!(path.is_some());
        let path = path.expect("path should exist");
        assert_eq!(path.nodes.len(), 3);
        assert_eq!(path.edges.len(), 2);
    }

    #[tokio::test]
    async fn test_get_nodes_by_type() {
        let backend = InMemoryGraphBackend::new();
        backend.add_node(Node::new("p1".to_string(), "Person".to_string())).await.expect("add_node should succeed");
        backend.add_node(Node::new("p2".to_string(), "Person".to_string())).await.expect("add_node should succeed");
        backend.add_node(Node::new("c1".to_string(), "Company".to_string())).await.expect("add_node should succeed");

        let people = backend.get_nodes_by_type("Person").await.expect("get_nodes_by_type should succeed");
        assert_eq!(people.len(), 2);
    }

    #[tokio::test]
    async fn test_delete_node() {
        let backend = InMemoryGraphBackend::new();
        backend.add_node(Node::new("a".to_string(), "Node".to_string())).await.expect("add_node should succeed");
        backend.add_node(Node::new("b".to_string(), "Node".to_string())).await.expect("add_node should succeed");
        backend.add_edge(Edge::new("a".to_string(), "b".to_string(), "CONNECTS".to_string())).await.expect("add_edge should succeed");

        backend.delete_node("a").await.expect("delete_node should succeed");
        
        assert!(backend.get_node("a").await.expect("get_node should succeed").is_none());
        let neighbors = backend.get_neighbors("a", None).await;
        assert!(neighbors.is_err()); // Should fail because node doesn't exist
    }

    #[tokio::test]
    async fn test_count_nodes_and_edges() {
        let backend = InMemoryGraphBackend::new();
        backend.add_node(Node::new("a".to_string(), "Node".to_string())).await.expect("add_node should succeed");
        backend.add_node(Node::new("b".to_string(), "Node".to_string())).await.expect("add_node should succeed");
        backend.add_edge(Edge::new("a".to_string(), "b".to_string(), "CONNECTS".to_string())).await.expect("add_edge should succeed");

        assert_eq!(backend.count_nodes().await.expect("count_nodes should succeed"), 2);
        assert_eq!(backend.count_edges().await.expect("count_edges should succeed"), 1);
    }
}

