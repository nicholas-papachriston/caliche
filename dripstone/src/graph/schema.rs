//! Graph schema types for introspection and LLM context injection.

use serde::{Deserialize, Serialize};

/// Complete schema of a graph database
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphSchema {
    pub vertex_labels: Vec<VertexLabel>,
    pub edge_labels: Vec<EdgeLabel>,
    pub indexes: Vec<GraphIndex>,
    pub constraints: Vec<GraphConstraint>,
    pub estimated_node_count: Option<u64>,
    pub estimated_edge_count: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VertexLabel {
    pub name: String,
    pub properties: Vec<PropertyDefinition>,
    pub estimated_count: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EdgeLabel {
    pub name: String,
    pub source_labels: Vec<String>,
    pub target_labels: Vec<String>,
    pub properties: Vec<PropertyDefinition>,
    pub estimated_count: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PropertyDefinition {
    pub name: String,
    pub value_type: PropertyType,
    pub indexed: bool,
    pub required: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PropertyType {
    String,
    Integer,
    Float,
    Boolean,
    DateTime,
    List,
    Map,
    Unknown,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphIndex {
    pub name: String,
    pub label: String,
    pub properties: Vec<String>,
    pub index_type: IndexType,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum IndexType {
    BTree,
    FullText,
    Vector,
    Composite,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphConstraint {
    pub name: String,
    pub label: String,
    pub properties: Vec<String>,
    pub constraint_type: ConstraintType,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConstraintType {
    Unique,
    Required,
    NodeKey,
}

impl GraphSchema {
    /// Generate a compact summary for LLM context injection
    #[must_use]
    pub fn to_summary(&self) -> String {
        let mut parts = Vec::new();
        parts.push("Graph Schema:".to_string());
        parts.push("  Vertex types:".to_string());
        for vl in &self.vertex_labels {
            let props: Vec<&str> = vl.properties.iter().map(|p| p.name.as_str()).collect();
            let count_str = vl
                .estimated_count
                .map(|c| format!(" (~{c} nodes)"))
                .unwrap_or_default();
            parts.push(format!(
                "    :{} [{}]{}",
                vl.name,
                props.join(", "),
                count_str
            ));
        }
        parts.push("  Edge types:".to_string());
        for el in &self.edge_labels {
            let direction = format!(
                "(:{}) -> (:{})",
                el.source_labels.join("|"),
                el.target_labels.join("|")
            );
            parts.push(format!("    :{} {}", el.name, direction));
        }
        parts.join("\n")
    }
}
