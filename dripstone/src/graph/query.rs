//! Query result types for graph database operations.

use super::{Edge, Node, Path};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Parameters for a graph query
pub type QueryParams = HashMap<String, serde_json::Value>;

/// Result of a graph query execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryResult {
    pub columns: Vec<String>,
    pub records: Vec<Record>,
    pub execution_time_ms: Option<u64>,
    pub rows_affected: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Record {
    pub fields: HashMap<String, GraphValue>,
}

/// A value returned from a graph query
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GraphValue {
    Null,
    Bool(bool),
    Int(i64),
    Float(f64),
    String(String),
    List(Vec<Self>),
    Map(HashMap<String, Self>),
    Node(Node),
    Edge(Edge),
    Path(Path),
}

impl QueryResult {
    /// Create an empty result
    #[must_use]
    pub const fn empty() -> Self {
        Self {
            columns: Vec::new(),
            records: Vec::new(),
            execution_time_ms: None,
            rows_affected: None,
        }
    }

    /// Format results as compact string for LLM consumption (`max_rows` caps output)
    #[must_use]
    pub fn to_display_string(&self, max_rows: usize) -> String {
        if self.records.is_empty() {
            return "No results returned.".to_string();
        }

        let mut lines = Vec::new();
        lines.push(format!("Columns: {}", self.columns.join(", ")));
        lines.push(format!("Rows: {} total", self.records.len()));

        let display_count = self.records.len().min(max_rows);
        for (i, record) in self.records.iter().take(display_count).enumerate() {
            let fields: Vec<String> = self
                .columns
                .iter()
                .filter_map(|col| {
                    record
                        .fields
                        .get(col)
                        .map(|v| format!("{}: {}", col, v.to_compact_string()))
                })
                .collect();
            lines.push(format!("  [{}] {}", i, fields.join(" | ")));
        }

        if self.records.len() > max_rows {
            lines.push(format!(
                "  ... ({} more rows)",
                self.records.len() - max_rows
            ));
        }

        lines.join("\n")
    }
}

/// Convert `serde_json::Value` to `GraphValue` (generic; backends may extend for Node/Edge)
pub fn json_to_graph_value(v: &serde_json::Value) -> GraphValue {
    match v {
        serde_json::Value::Null => GraphValue::Null,
        serde_json::Value::Bool(b) => GraphValue::Bool(*b),
        serde_json::Value::Number(n) => n.as_i64().map_or_else(
            || n.as_f64().map_or(GraphValue::Null, GraphValue::Float),
            GraphValue::Int,
        ),
        serde_json::Value::String(s) => GraphValue::String(s.clone()),
        serde_json::Value::Array(arr) => {
            GraphValue::List(arr.iter().map(json_to_graph_value).collect())
        }
        serde_json::Value::Object(obj) => {
            let map: HashMap<String, GraphValue> = obj
                .iter()
                .map(|(k, v)| (k.clone(), json_to_graph_value(v)))
                .collect();
            GraphValue::Map(map)
        }
    }
}

impl GraphValue {
    /// Compact string representation for LLM output
    #[must_use]
    pub fn to_compact_string(&self) -> String {
        match self {
            Self::Null => "null".to_string(),
            Self::Bool(b) => b.to_string(),
            Self::Int(i) => i.to_string(),
            Self::Float(f) => format!("{f:.4}"),
            Self::String(s) => {
                if s.len() > 100 {
                    format!("\"{}...\"", &s[..s.floor_char_boundary(97)])
                } else {
                    format!("\"{s}\"")
                }
            }
            Self::List(l) => format!("[{} items]", l.len()),
            Self::Map(m) => format!("{{{} keys}}", m.len()),
            Self::Node(n) => {
                let mut parts = vec![format!("id: {}", n.id)];
                let mut rest: Vec<_> = n
                    .properties
                    .iter()
                    .filter(|(k, _)| *k != "id")
                    .take(4)
                    .map(|(k, v)| {
                        let v_str = match v {
                            serde_json::Value::String(s) => format!("\"{s}\""),
                            _ => v.to_string(),
                        };
                        format!("{k}: {v_str}")
                    })
                    .collect();
                parts.append(&mut rest);
                if n.properties.len() > 5 {
                    parts.push("...".to_string());
                }
                let props = parts.join(", ");
                format!("(:{} {{{}}})", n.node_type, props)
            }
            Self::Edge(e) => format!("-[:{}]->", e.edge_type),
            Self::Path(p) => {
                format!("Path({} nodes, {} edges)", p.nodes.len(), p.edges.len())
            }
        }
    }
}
