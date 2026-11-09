//! Graph query tool (Level 1+2): schema introspection and Cypher execution.

use crate::agent::tools::common::params;
use crate::agent::tools::{Tool, ToolMeta, ToolResult};
use async_trait::async_trait;
use dripstone::graph::Record;
use dripstone::graph::{GraphSchema, GraphValue, QueryParams, QueryResult, QueryableGraphBackend};
use parking_lot::RwLock;
use serde_json::json;
use std::collections::HashMap;
use std::sync::Arc;

/// Tool for querying and traversing graph databases (schema, Cypher, find nodes, traverse, paths).
pub struct GraphQueryTool {
    backend: Arc<dyn QueryableGraphBackend>,
    cached_schema: RwLock<Option<GraphSchema>>,
}

impl std::fmt::Debug for GraphQueryTool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GraphQueryTool")
            .field("cached_schema", &self.cached_schema.read().is_some())
            .finish()
    }
}

impl GraphQueryTool {
    /// Create a new graph query tool with the given queryable backend.
    pub fn new(backend: Arc<dyn QueryableGraphBackend>) -> Self {
        Self {
            backend,
            cached_schema: RwLock::new(None),
        }
    }

    fn build_find_nodes_cypher(
        node_type: &str,
        filters: &serde_json::Value,
        limit: usize,
    ) -> (String, QueryParams) {
        let mut where_clauses = Vec::new();
        let mut params = QueryParams::new();
        if let Some(obj) = filters.as_object() {
            for (key, value) in obj {
                where_clauses.push(format!("n.{key} = ${key}"));
                params.insert(key.clone(), value.clone());
            }
        }
        let where_str = if where_clauses.is_empty() {
            String::new()
        } else {
            format!(" WHERE {}", where_clauses.join(" AND "))
        };
        let type_escaped = node_type.replace('\'', "''");
        (
            format!("MATCH (n:{type_escaped}){where_str} RETURN n LIMIT {limit}"),
            params,
        )
    }

    fn build_traverse_cypher(
        node_id: &str,
        edge_types: Option<&[String]>,
        max_hops: usize,
        direction: &str,
    ) -> String {
        let id_escaped = node_id.replace('\'', "''");
        let edge_filter = edge_types
            .map(|t| {
                t.iter()
                    .map(|s| format!(":{}", s.replace('\'', "''")))
                    .collect::<Vec<_>>()
                    .join("|")
            })
            .unwrap_or_default();
        let pattern = match direction {
            "incoming" => format!("(s {{id: '{id_escaped}'}})<-[r{edge_filter}*1..{max_hops}]-(e)"),
            "both" => format!("(s {{id: '{id_escaped}'}})-[r{edge_filter}*1..{max_hops}]-(e)"),
            _ => format!("(s {{id: '{id_escaped}'}})-[r{edge_filter}*1..{max_hops}]->(e)"),
        };
        format!("MATCH p = {pattern} RETURN p")
    }

    async fn schema_result(&self) -> ToolResult {
        let schema = match self.backend.get_schema().await {
            Ok(schema) => schema,
            Err(error) => return ToolResult::error(error.to_string()),
        };
        {
            let mut guard = self.cached_schema.write();
            *guard = Some(schema.clone());
        }
        let summary = schema.to_summary();
        let structured = json!({
            "vertex_labels": schema.vertex_labels,
            "edge_labels": schema.edge_labels
        });
        let content = format!(
            "{}\n\nStructured schema:\n{}",
            summary,
            serde_json::to_string_pretty(&structured).unwrap_or_default()
        );
        ToolResult::success(content)
    }

    async fn query_result(&self, arguments: &serde_json::Value) -> ToolResult {
        let Some(cypher) = arguments["query"].as_str() else {
            return ToolResult::error("Missing 'query' parameter for 'query' operation".to_owned());
        };
        let params = arguments["params"]
            .as_object()
            .map(|object| {
                object
                    .iter()
                    .map(|(key, value)| (key.clone(), value.clone()))
                    .collect::<QueryParams>()
            })
            .unwrap_or_default();
        let result = match self.backend.execute_query(cypher, &params).await {
            Ok(result) => result,
            Err(error) => return ToolResult::error(error.to_string()),
        };
        ToolResult::success(result.to_display_string(25))
    }

    async fn find_nodes_result(&self, arguments: &serde_json::Value) -> ToolResult {
        let Some(node_type) = arguments["node_type"].as_str() else {
            return ToolResult::error("Missing 'node_type' for 'find_nodes'".to_owned());
        };
        let empty = json!({});
        let filters = arguments.get("filters").unwrap_or(&empty);
        let limit = arguments["limit"].as_u64().unwrap_or(25) as usize;
        let limit = limit.min(100);
        let filters_empty = filters.as_object().map_or(true, serde_json::Map::is_empty);
        if filters_empty {
            return self.nodes_by_type(node_type, limit).await;
        }
        let (cypher, params) = Self::build_find_nodes_cypher(node_type, filters, limit);
        let result = match self.backend.execute_query(&cypher, &params).await {
            Ok(result) => result,
            Err(error) => return ToolResult::error(error.to_string()),
        };
        ToolResult::success(result.to_display_string(25))
    }

    async fn nodes_by_type(&self, node_type: &str, limit: usize) -> ToolResult {
        let nodes = match self.backend.get_nodes_by_type(node_type).await {
            Ok(nodes) => nodes,
            Err(error) => return ToolResult::error(error.to_string()),
        };
        let records = nodes
            .into_iter()
            .take(limit)
            .map(|node| {
                let mut fields = HashMap::new();
                fields.insert("n".to_owned(), GraphValue::Node(node));
                Record { fields }
            })
            .collect();
        let result = QueryResult {
            columns: vec!["n".to_owned()],
            records,
            execution_time_ms: None,
            rows_affected: None,
        };
        ToolResult::success(result.to_display_string(25))
    }

    async fn traverse_result(&self, arguments: &serde_json::Value) -> ToolResult {
        let Some(node_id) = arguments["node_id"].as_str() else {
            return ToolResult::error("Missing 'node_id' for 'traverse'");
        };
        let edge_types: Option<Vec<String>> = arguments["edge_types"].as_array().map(|values| {
            values
                .iter()
                .filter_map(|value| value.as_str().map(str::to_owned))
                .collect()
        });
        let max_hops = arguments["max_hops"].as_u64().unwrap_or(3) as usize;
        let max_hops = max_hops.clamp(1, 10);
        let direction = arguments["direction"].as_str().unwrap_or("outgoing");
        let cypher =
            Self::build_traverse_cypher(node_id, edge_types.as_deref(), max_hops, direction);
        let params = QueryParams::new();
        let result = match self.backend.execute_query(&cypher, &params).await {
            Ok(result) => result,
            Err(error) => return ToolResult::error(error.to_string()),
        };
        ToolResult::success(result.to_display_string(25))
    }

    async fn find_paths_result(&self, arguments: &serde_json::Value) -> ToolResult {
        let Some(node_id) = arguments["node_id"].as_str() else {
            return ToolResult::error("Missing 'node_id' for 'find_paths'");
        };
        let Some(target_node_id) = arguments["target_node_id"].as_str() else {
            return ToolResult::error("Missing 'target_node_id' for 'find_paths'".to_owned());
        };
        let max_hops = arguments["max_hops"].as_u64().unwrap_or(10) as usize;
        let max_hops = max_hops.clamp(1, 10);
        let path = match self
            .backend
            .find_path(node_id, target_node_id, max_hops)
            .await
        {
            Ok(path) => path,
            Err(error) => return ToolResult::error(error.to_string()),
        };
        let content = match path {
            Some(path) => format!(
                "Path ({} nodes, {} edges): {:?}",
                path.nodes.len(),
                path.edges.len(),
                path.nodes.iter().map(|node| &node.id).collect::<Vec<_>>()
            ),
            None => "No path found.".to_owned(),
        };
        ToolResult::success(content)
    }

    async fn neighbors_result(&self, arguments: &serde_json::Value) -> ToolResult {
        let Some(node_id) = arguments["node_id"].as_str() else {
            return ToolResult::error("Missing 'node_id' for 'get_neighbors'".to_owned());
        };
        let edge_types: Option<Vec<&str>> = arguments["edge_types"].as_array().map(|values| {
            values
                .iter()
                .filter_map(serde_json::Value::as_str)
                .collect()
        });
        let neighbors = match self
            .backend
            .get_neighbors(node_id, edge_types.as_deref())
            .await
        {
            Ok(neighbors) => neighbors,
            Err(error) => return ToolResult::error(error.to_string()),
        };
        let list: Vec<serde_json::Value> = neighbors
            .iter()
            .map(|neighbor| {
                json!({
                    "node_id": neighbor.node.id,
                    "node_type": neighbor.node.node_type,
                    "edge_type": neighbor.edge.edge_type
                })
            })
            .collect();
        let content = serde_json::to_string_pretty(&list).unwrap_or_else(|_| "[]".to_owned());
        ToolResult::success(content)
    }
}

const NAME: &str = "graph_query";
const DESCRIPTION: &str = "Query and traverse a graph database. Operations: 'get_schema' (discover vertex types, edge types, properties), 'query' (execute raw Cypher), 'find_nodes' (find nodes by type and property filters), 'traverse' (traverse from a node following edge types), 'find_paths' (shortest path between two nodes), 'get_neighbors' (1-hop neighbors with edge type filter). Always call get_schema first to understand the graph structure before writing queries.";

impl ToolMeta for GraphQueryTool {
    fn description(&self) -> &str {
        DESCRIPTION
    }

    fn description_as_cow(&self) -> beef::lean::Cow<'static, str> {
        beef::lean::Cow::borrowed(DESCRIPTION)
    }

    fn name_as_cow(&self) -> beef::lean::Cow<'static, str> {
        beef::lean::Cow::borrowed(NAME)
    }

    fn name(&self) -> &str {
        NAME
    }

    fn parameters(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": ["get_schema", "query", "find_nodes", "traverse", "find_paths", "get_neighbors"],
                    "description": "Operation to perform on the graph database"
                },
                "query": {
                    "type": "string",
                    "description": "Raw Cypher query string (for 'query' operation). Use get_schema first to understand available types."
                },
                "params": {
                    "type": "object",
                    "description": "Query parameters as key-value pairs, substituted for $name in Cypher",
                    "additionalProperties": true
                },
                "node_type": {
                    "type": "string",
                    "description": "Vertex label/type (for 'find_nodes', 'get_neighbors'). Use get_schema to see available types."
                },
                "node_id": {
                    "type": "string",
                    "description": "Node ID to start from (for 'traverse', 'find_paths', 'get_neighbors')"
                },
                "target_node_id": {
                    "type": "string",
                    "description": "Target node ID (for 'find_paths')"
                },
                "filters": {
                    "type": "object",
                    "description": "Property filters as key-value pairs (for 'find_nodes')",
                    "additionalProperties": true
                },
                "edge_types": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Filter to these edge/relationship types only"
                },
                "direction": {
                    "type": "string",
                    "enum": ["outgoing", "incoming", "both"],
                    "description": "Traversal direction (default: 'outgoing')",
                    "default": "outgoing"
                },
                "max_hops": {
                    "type": "integer",
                    "default": 3,
                    "minimum": 1,
                    "maximum": 10,
                    "description": "Maximum traversal depth"
                },
                "limit": {
                    "type": "integer",
                    "default": 25,
                    "minimum": 1,
                    "maximum": 100,
                    "description": "Maximum results to return"
                }
            },
            "required": ["operation"]
        })
    }
}

#[async_trait]
impl Tool for GraphQueryTool {
    async fn execute(&self, arguments: serde_json::Value) -> ToolResult {
        let operation = match params::extract_operation(&arguments) {
            Ok(op) => op,
            Err(e) => return ToolResult::error(e),
        };

        match operation.as_str() {
            "get_schema" => self.schema_result().await,
            "query" => self.query_result(&arguments).await,
            "find_nodes" => self.find_nodes_result(&arguments).await,
            "traverse" => self.traverse_result(&arguments).await,
            "find_paths" => self.find_paths_result(&arguments).await,
            "get_neighbors" => self.neighbors_result(&arguments).await,
            _ => ToolResult::error(format!("Unknown operation: '{operation}'")),
        }
    }
}
