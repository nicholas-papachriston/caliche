//! In-memory graph backend implementation
//!
//! A simple, thread-safe graph storage using HashMap-based adjacency lists.

use crate::graph::{Edge, FindPathOptions, GraphBackend, GraphError, Neighbor, Node, Path};
use async_trait::async_trait;
use parking_lot::RwLock;
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, HashSet};

/// In-memory graph backend using adjacency lists
pub struct InMemoryGraphBackend {
    /// Map from node ID to node
    nodes: RwLock<HashMap<String, Node>>,
    /// Map from node ID to outgoing edges (source -> `Vec<Edge>`)
    outgoing_edges: RwLock<HashMap<String, Vec<Edge>>>,
    /// Map from node ID to incoming edges (target -> `Vec<Edge>`)
    incoming_edges: RwLock<HashMap<String, Vec<Edge>>>,
}

impl InMemoryGraphBackend {
    /// Create a new in-memory graph backend
    #[must_use]
    pub fn new() -> Self {
        Self {
            nodes: RwLock::new(HashMap::new()),
            outgoing_edges: RwLock::new(HashMap::new()),
            incoming_edges: RwLock::new(HashMap::new()),
        }
    }

    fn find_bfs_path(
        source_id: &str,
        target_id: &str,
        max_hops: usize,
        outgoing: &HashMap<String, Vec<Edge>>,
        edge_types: Option<&HashSet<&str>>,
    ) -> Option<Vec<String>> {
        let mut queue = std::collections::VecDeque::new();
        queue.push_back(vec![source_id.to_string()]);

        let mut visited = HashSet::new();
        visited.insert(source_id.to_string());

        while let Some(path) = queue.pop_front() {
            let current_id = match path.last() {
                Some(id) => id.clone(),
                None => continue,
            };
            if current_id == target_id {
                return Some(path);
            }
            if path.len() > max_hops {
                continue;
            }
            let Some(edges) = outgoing.get(&current_id) else {
                continue;
            };
            for edge in edges {
                if let Some(types) = edge_types
                    && !types.contains(edge.edge_type.as_str())
                {
                    continue;
                }
                if visited.insert(edge.target_id.clone()) {
                    let mut new_path = path.clone();
                    new_path.push(edge.target_id.clone());
                    queue.push_back(new_path);
                }
            }
        }
        None
    }

    fn find_weighted_path(
        source_id: &str,
        target_id: &str,
        max_hops: usize,
        outgoing: &HashMap<String, Vec<Edge>>,
        edge_types: Option<&HashSet<&str>>,
    ) -> Option<Vec<String>> {
        /// A heap entry: (cost, hops, `node_id`). `Reverse`-style ordering so the
        /// smallest cost pops first; ties broken by fewer hops.
        #[derive(Debug)]
        struct Entry {
            cost: f32,
            hops: usize,
            node: String,
        }

        impl PartialEq for Entry {
            fn eq(&self, other: &Self) -> bool {
                self.cost.total_cmp(&other.cost).is_eq() && self.hops == other.hops
            }
        }
        impl Eq for Entry {}
        impl Ord for Entry {
            fn cmp(&self, other: &Self) -> Ordering {
                other
                    .cost
                    .partial_cmp(&self.cost)
                    .unwrap_or(Ordering::Equal)
                    .then_with(|| other.hops.cmp(&self.hops))
            }
        }
        impl PartialOrd for Entry {
            fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
                Some(self.cmp(other))
            }
        }

        let mut best_cost: HashMap<String, f32> = HashMap::new();
        let mut prev: HashMap<String, String> = HashMap::new();
        let mut heap: BinaryHeap<Entry> = BinaryHeap::new();

        best_cost.insert(source_id.to_string(), 0.0);
        heap.push(Entry {
            cost: 0.0,
            hops: 0,
            node: source_id.to_string(),
        });

        while let Some(Entry { cost, hops, node }) = heap.pop() {
            if node == target_id {
                let mut path = vec![node.clone()];
                let mut current = node;
                while let Some(parent) = prev.get(&current).cloned() {
                    path.push(parent.clone());
                    current = parent;
                }
                path.reverse();
                return Some(path);
            }
            if let Some(&existing) = best_cost.get(&node)
                && existing < cost
            {
                continue;
            }
            if hops >= max_hops {
                continue;
            }
            let Some(edges) = outgoing.get(&node) else {
                continue;
            };
            for edge in edges {
                if let Some(types) = edge_types
                    && !types.contains(edge.edge_type.as_str())
                {
                    continue;
                }
                let edge_weight = edge.weight.unwrap_or(1.0).max(0.0);
                let next_cost = cost + edge_weight;
                let improves = match best_cost.get(&edge.target_id) {
                    Some(&existing) => next_cost < existing,
                    None => true,
                };
                if improves {
                    best_cost.insert(edge.target_id.clone(), next_cost);
                    prev.insert(edge.target_id.clone(), node.clone());
                    heap.push(Entry {
                        cost: next_cost,
                        hops: hops + 1,
                        node: edge.target_id.clone(),
                    });
                }
            }
        }
        None
    }
}

impl Default for InMemoryGraphBackend {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl GraphBackend for InMemoryGraphBackend {
    async fn add_node(&self, node: Node) -> Result<(), GraphError> {
        let mut nodes = self.nodes.write();
        nodes.insert(node.id.clone(), node);
        drop(nodes);
        Ok(())
    }

    async fn get_node(&self, node_id: &str) -> Result<Option<Node>, GraphError> {
        let nodes = self.nodes.read();
        Ok(nodes.get(node_id).cloned())
    }

    async fn add_edge(&self, edge: Edge) -> Result<(), GraphError> {
        // Verify both nodes exist
        let nodes = self.nodes.read();
        if !nodes.contains_key(&edge.source_id) {
            return Err(GraphError::NodeNotFound(edge.source_id));
        }
        if !nodes.contains_key(&edge.target_id) {
            return Err(GraphError::NodeNotFound(edge.target_id));
        }
        drop(nodes);

        // Add to outgoing edges
        let mut outgoing = self.outgoing_edges.write();
        outgoing
            .entry(edge.source_id.clone())
            .or_default()
            .push(edge.clone());
        drop(outgoing);

        // Add to incoming edges
        let mut incoming = self.incoming_edges.write();
        incoming
            .entry(edge.target_id.clone())
            .or_default()
            .push(edge);
        drop(incoming);

        Ok(())
    }

    async fn get_neighbors(
        &self,
        node_id: &str,
        edge_types: Option<&[&str]>,
    ) -> Result<Vec<Neighbor>, GraphError> {
        let nodes = self.nodes.read();
        let outgoing = self.outgoing_edges.read();

        if !nodes.contains_key(node_id) {
            return Err(GraphError::NodeNotFound(node_id.to_string()));
        }

        let edges = outgoing.get(node_id).cloned().unwrap_or_default();
        drop(nodes);
        drop(outgoing);

        let edge_type_set: Option<HashSet<&str>> =
            edge_types.map(|types| types.iter().copied().collect());

        let mut neighbors = Vec::new();
        for edge in edges {
            // Filter by edge type if specified
            if let Some(ref type_set) = edge_type_set
                && !type_set.contains(edge.edge_type.as_str())
            {
                continue;
            }

            let nodes = self.nodes.read();
            if let Some(target_node) = nodes.get(&edge.target_id).cloned() {
                neighbors.push(Neighbor {
                    node: target_node,
                    edge,
                });
            }
        }

        Ok(neighbors)
    }

    async fn find_path(
        &self,
        source_id: &str,
        target_id: &str,
        max_hops: usize,
    ) -> Result<Option<Path>, GraphError> {
        self.find_path_ext(source_id, target_id, max_hops, &FindPathOptions::default())
            .await
    }

    async fn find_path_ext(
        &self,
        source_id: &str,
        target_id: &str,
        max_hops: usize,
        options: &FindPathOptions,
    ) -> Result<Option<Path>, GraphError> {
        let nodes = self.nodes.read();
        if !nodes.contains_key(source_id) {
            return Err(GraphError::NodeNotFound(source_id.to_string()));
        }
        if !nodes.contains_key(target_id) {
            return Err(GraphError::NodeNotFound(target_id.to_string()));
        }
        drop(nodes);

        let edge_type_set: Option<HashSet<&str>> = options
            .edge_types
            .as_ref()
            .map(|types| types.iter().map(String::as_str).collect());

        let outgoing = self.outgoing_edges.read();
        let nodes = self.nodes.read();

        let path_node_ids = if options.weighted {
            Self::find_weighted_path(
                source_id,
                target_id,
                max_hops,
                &outgoing,
                edge_type_set.as_ref(),
            )
        } else {
            Self::find_bfs_path(
                source_id,
                target_id,
                max_hops,
                &outgoing,
                edge_type_set.as_ref(),
            )
        };

        let Some(path) = path_node_ids else {
            return Ok(None);
        };

        let mut path_nodes = Vec::with_capacity(path.len());
        let mut path_edges = Vec::with_capacity(path.len().saturating_sub(1));
        let mut total_weight = 0.0_f32;

        for (i, node_id) in path.iter().enumerate() {
            if let Some(node) = nodes.get(node_id).cloned() {
                path_nodes.push(node);
            }
            if i + 1 < path.len()
                && let Some(edges) = outgoing.get(node_id)
            {
                let selected = edges
                    .iter()
                    .filter(|e| e.target_id == path[i + 1])
                    .filter(|e| {
                        edge_type_set
                            .as_ref()
                            .is_none_or(|set| set.contains(e.edge_type.as_str()))
                    })
                    .min_by(|a, b| {
                        let wa = a.weight.unwrap_or(1.0);
                        let wb = b.weight.unwrap_or(1.0);
                        wa.partial_cmp(&wb).unwrap_or(Ordering::Equal)
                    })
                    .cloned();
                if let Some(edge) = selected {
                    total_weight += edge.weight.unwrap_or(1.0);
                    path_edges.push(edge);
                }
            }
        }

        drop(nodes);
        drop(outgoing);
        Ok(Some(Path {
            nodes: path_nodes,
            edges: path_edges,
            total_weight: Some(total_weight),
        }))
    }

    async fn get_nodes_by_type(&self, node_type: &str) -> Result<Vec<Node>, GraphError> {
        let nodes = self.nodes.read();
        let result = nodes
            .values()
            .filter(|n| n.node_type == node_type)
            .cloned()
            .collect();
        drop(nodes);
        Ok(result)
    }

    async fn get_edges_by_type(&self, edge_type: &str) -> Result<Vec<Edge>, GraphError> {
        let outgoing = self.outgoing_edges.read();
        let mut edges = Vec::new();
        for edge_list in outgoing.values() {
            for edge in edge_list {
                if edge.edge_type == edge_type {
                    edges.push(edge.clone());
                }
            }
        }
        drop(outgoing);
        Ok(edges)
    }

    async fn delete_node(&self, node_id: &str) -> Result<(), GraphError> {
        let mut nodes = self.nodes.write();
        if !nodes.contains_key(node_id) {
            return Err(GraphError::NodeNotFound(node_id.to_string()));
        }
        nodes.remove(node_id);
        drop(nodes);

        // Remove all edges connected to this node
        let mut outgoing = self.outgoing_edges.write();

        outgoing.remove(node_id);

        // Remove edges from other nodes that point to this node
        for edges in outgoing.values_mut() {
            edges.retain(|e| e.target_id != node_id);
        }
        drop(outgoing);

        let mut incoming = self.incoming_edges.write();
        incoming.remove(node_id);
        for edges in incoming.values_mut() {
            edges.retain(|e| e.source_id != node_id);
        }
        drop(incoming);

        Ok(())
    }

    async fn delete_edge(
        &self,
        source_id: &str,
        target_id: &str,
        edge_type: &str,
    ) -> Result<(), GraphError> {
        // Remove from outgoing
        let mut outgoing = self.outgoing_edges.write();
        if let Some(edges) = outgoing.get_mut(source_id) {
            let initial_len = edges.len();
            edges.retain(|e| !(e.target_id == target_id && e.edge_type == edge_type));
            if edges.is_empty() && initial_len > 0 {
                outgoing.remove(source_id);
            }
        }
        drop(outgoing);

        // Remove from incoming
        let mut incoming = self.incoming_edges.write();
        if let Some(edges) = incoming.get_mut(target_id) {
            let initial_len = edges.len();
            edges.retain(|e| !(e.source_id == source_id && e.edge_type == edge_type));
            if edges.is_empty() && initial_len > 0 {
                incoming.remove(target_id);
            }
        }
        drop(incoming);

        Ok(())
    }

    async fn count_nodes(&self) -> Result<usize, GraphError> {
        let nodes = self.nodes.read();
        Ok(nodes.len())
    }

    async fn count_edges(&self) -> Result<usize, GraphError> {
        let outgoing = self.outgoing_edges.read();
        Ok(outgoing.values().map(std::vec::Vec::len).sum())
    }
}
