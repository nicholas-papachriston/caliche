//! Knowledge Graph primitives for Graph RAG
//!
//! This module provides data structures and traits for storing and traversing
//! knowledge graphs, enabling multi-hop reasoning over relationships.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A node in the knowledge graph representing an entity
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Node {
    /// Unique identifier for the node
    pub id: String,
    /// Type of entity (e.g., "Person", "Company", "Concept")
    pub node_type: String,
    /// Properties/metadata associated with the node
    pub properties: HashMap<String, serde_json::Value>,
    /// Optional embedding vector for semantic search
    pub embedding: Option<Vec<f32>>,
}

impl Node {
    /// Create a new node
    #[must_use]
    pub fn new(id: String, node_type: String) -> Self {
        Self {
            id,
            node_type,
            properties: HashMap::new(),
            embedding: None,
        }
    }

    /// Add a property to the node
    #[must_use]
    pub fn with_property(mut self, key: String, value: serde_json::Value) -> Self {
        self.properties.insert(key, value);
        self
    }

    /// Set the embedding vector for semantic search
    #[must_use]
    pub fn with_embedding(mut self, embedding: Vec<f32>) -> Self {
        self.embedding = Some(embedding);
        self
    }
}

/// An edge in the knowledge graph representing a relationship
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Edge {
    /// Source node ID
    pub source_id: String,
    /// Target node ID
    pub target_id: String,
    /// Type of relationship (e.g., "`WORKS_FOR`", "`LOCATED_IN`", "`RELATED_TO`")
    pub edge_type: String,
    /// Properties/metadata associated with the edge
    pub properties: HashMap<String, serde_json::Value>,
    /// Optional weight/strength of the relationship
    pub weight: Option<f32>,
}

impl Edge {
    /// Create a new edge
    #[must_use]
    pub fn new(source_id: String, target_id: String, edge_type: String) -> Self {
        Self {
            source_id,
            target_id,
            edge_type,
            properties: HashMap::new(),
            weight: None,
        }
    }

    /// Add a property to the edge
    #[must_use]
    pub fn with_property(mut self, key: String, value: serde_json::Value) -> Self {
        self.properties.insert(key, value);
        self
    }

    /// Set the weight of the relationship
    #[must_use]
    pub const fn with_weight(mut self, weight: f32) -> Self {
        self.weight = Some(weight);
        self
    }
}

/// A neighbor node with its connecting edge
#[derive(Debug, Clone)]
pub struct Neighbor {
    /// The neighbor node
    pub node: Node,
    /// The edge connecting to this neighbor
    pub edge: Edge,
}

/// Path through the graph from source to target
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Path {
    /// Sequence of nodes in the path
    pub nodes: Vec<Node>,
    /// Sequence of edges connecting the nodes
    pub edges: Vec<Edge>,
    /// Total path weight (sum of edge weights if available)
    pub total_weight: Option<f32>,
}

/// Trait for graph storage and traversal backends
#[async_trait]
pub trait GraphBackend: Send + Sync {
    /// Add a node to the graph
    async fn add_node(&self, node: Node) -> Result<(), GraphError>;

    /// Get a node by ID
    async fn get_node(&self, node_id: &str) -> Result<Option<Node>, GraphError>;

    /// Add an edge to the graph
    async fn add_edge(&self, edge: Edge) -> Result<(), GraphError>;

    /// Get neighbors of a node (1-hop)
    async fn get_neighbors(
        &self,
        node_id: &str,
        edge_types: Option<&[&str]>,
    ) -> Result<Vec<Neighbor>, GraphError>;

    /// Find a path between two nodes (up to `max_hops`).
    ///
    /// Default implementation delegates to [`Self::find_path_ext`] with no
    /// edge-type filter and unweighted semantics (shortest in number of
    /// hops). Backends can override this directly for performance.
    async fn find_path(
        &self,
        source_id: &str,
        target_id: &str,
        max_hops: usize,
    ) -> Result<Option<Path>, GraphError> {
        self.find_path_ext(source_id, target_id, max_hops, &FindPathOptions::default())
            .await
    }

    /// Find a path between two nodes with edge-type filtering and optional
    /// weighted-path semantics.
    ///
    /// The default implementation ignores `options` and falls back to plain
    /// BFS via [`Self::find_path`]; backends should override this to honour
    /// filters and weights when supported.
    async fn find_path_ext(
        &self,
        source_id: &str,
        target_id: &str,
        max_hops: usize,
        _options: &FindPathOptions,
    ) -> Result<Option<Path>, GraphError> {
        self.find_path(source_id, target_id, max_hops).await
    }

    /// Get all nodes of a specific type
    async fn get_nodes_by_type(&self, node_type: &str) -> Result<Vec<Node>, GraphError>;

    /// Get all edges of a specific type
    async fn get_edges_by_type(&self, edge_type: &str) -> Result<Vec<Edge>, GraphError>;

    /// Delete a node and all its edges
    async fn delete_node(&self, node_id: &str) -> Result<(), GraphError>;

    /// Delete an edge
    async fn delete_edge(
        &self,
        source_id: &str,
        target_id: &str,
        edge_type: &str,
    ) -> Result<(), GraphError>;

    /// Count nodes in the graph
    async fn count_nodes(&self) -> Result<usize, GraphError>;

    /// Count edges in the graph
    async fn count_edges(&self) -> Result<usize, GraphError>;
}

/// Options for graph pathfinding queries.
///
/// - `edge_types`: when `Some`, only traverse edges whose `edge_type` is in
///   this set. When `None`, any edge type is allowed.
/// - `weighted`: when `true`, return a path that minimizes the sum of
///   `Edge::weight` (unweighted edges are treated as weight `1.0`). When
///   `false`, the backend returns the shortest path in number of hops.
#[derive(Debug, Clone, Default)]
pub struct FindPathOptions {
    pub edge_types: Option<Vec<String>>,
    pub weighted: bool,
}

impl FindPathOptions {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    #[must_use]
    pub fn with_edge_types<I, S>(mut self, edge_types: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.edge_types = Some(edge_types.into_iter().map(Into::into).collect());
        self
    }

    #[must_use]
    pub const fn with_weighted(mut self, weighted: bool) -> Self {
        self.weighted = weighted;
        self
    }
}

/// Error type for graph operations
#[derive(Debug, thiserror::Error)]
pub enum GraphError {
    #[error("Node not found: {0}")]
    NodeNotFound(String),
    #[error("Edge not found: {0} -> {1} ({2})")]
    EdgeNotFound(String, String, String),
    #[error("Invalid operation: {0}")]
    InvalidOperation(String),
    #[error("Storage error: {0}")]
    Storage(String),
    #[error("Query execution error: {0}")]
    QueryError(String),
    #[error("Schema introspection error: {0}")]
    SchemaError(String),
    #[error("Connection error: {0}")]
    Connection(String),
    #[error("Unsupported operation: {0}")]
    Unsupported(String),
}

#[cfg(feature = "graph-age")]
pub mod age;
#[cfg(feature = "graph-neo4j")]
pub mod neo4j;
pub mod query;
pub mod schema;

#[cfg(feature = "graph-age")]
pub use age::AgeBackend;
#[cfg(feature = "graph-neo4j")]
pub use neo4j::Neo4jBackend;
pub use query::*;
pub use schema::*;

/// Capabilities of a queryable graph backend
#[derive(Debug, Clone)]
#[expect(
    clippy::struct_excessive_bools,
    reason = "public capability fields preserve the existing API contract"
)]
pub struct QueryCapabilities {
    pub supports_cypher: bool,
    pub supports_gremlin: bool,
    pub supports_vector_index: bool,
    pub supports_full_text: bool,
    pub max_query_depth: Option<usize>,
    /// Whether this backend can execute unified CTE queries
    /// combining vector search and graph traversal in one statement
    pub supports_unified_hybrid: bool,
}

/// Extended trait for graph databases that support query languages
/// and schema introspection. Supersets `GraphBackend`.
#[async_trait]
pub trait QueryableGraphBackend: GraphBackend {
    async fn execute_query(
        &self,
        query: &str,
        params: &QueryParams,
    ) -> Result<QueryResult, GraphError>;

    async fn get_schema(&self) -> Result<GraphSchema, GraphError>;

    async fn refresh_schema(&self) -> Result<GraphSchema, GraphError>;

    async fn search_nodes_by_embedding(
        &self,
        embedding: &[f32],
        top_k: usize,
        node_type_filter: Option<&str>,
    ) -> Result<Vec<(Node, f32)>, GraphError>;

    fn query_capabilities(&self) -> QueryCapabilities;
}

pub mod memory;

pub use memory::InMemoryGraphBackend;
