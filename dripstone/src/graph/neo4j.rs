//! Neo4j graph backend using neo4rs (Bolt protocol).

use crate::graph::query::{Record, json_to_graph_value};
use crate::graph::{
    Edge, EdgeLabel, GraphBackend, GraphError, GraphSchema, GraphValue, Neighbor, Node, Path,
    PropertyDefinition, PropertyType, QueryCapabilities, QueryParams, QueryResult,
    QueryableGraphBackend, VertexLabel,
};
use crate::simd;
use async_trait::async_trait;
use neo4rs::{ConfigBuilder, Graph, Node as Neo4jNode, Relation, UnboundedRelation, query};
use parking_lot::RwLock;
use std::collections::HashMap;
use std::time::Instant;

const CLIENT_SIDE_NODE_LIMIT: usize = 5_000;

/// Add only params that neo4rs accepts (string, number, bool) to the query.
///
/// Returns an error if a parameter is of an unsupported JSON type
/// (`Null`, `Array`, `Object`). Silently dropping these would lead to
/// query failures or incorrect results at the Neo4j layer; surfacing the
/// error here lets callers remap their parameters explicitly.
fn add_params_to_query(
    mut q: neo4rs::Query,
    params: &QueryParams,
) -> Result<neo4rs::Query, GraphError> {
    for (k, v) in params {
        match v {
            serde_json::Value::String(s) => {
                q = q.param(k.as_str(), s.as_str());
            }
            serde_json::Value::Number(n) => {
                if let Some(i) = n.as_i64() {
                    q = q.param(k.as_str(), i);
                } else if let Some(f) = n.as_f64() {
                    q = q.param(k.as_str(), f);
                } else {
                    return Err(GraphError::QueryError(format!(
                        "unsupported numeric parameter `{k}`: value does not fit in i64 or f64",
                    )));
                }
            }
            serde_json::Value::Bool(b) => {
                q = q.param(k.as_str(), *b);
            }
            serde_json::Value::Null => {
                return Err(GraphError::QueryError(format!(
                    "null parameter `{k}` is not supported by the Neo4j bolt bridge"
                )));
            }
            serde_json::Value::Array(_) | serde_json::Value::Object(_) => {
                return Err(GraphError::QueryError(format!(
                    "parameter `{k}` has unsupported type (array/object); \
                     use a scalar (string, integer, float, bool)"
                )));
            }
        }
    }
    Ok(q)
}

/// Configuration for the Neo4j graph backend
#[derive(Debug, Clone)]
pub struct Neo4jConfig {
    /// Bolt URI (e.g. bolt://localhost:7687)
    pub uri: String,
    /// Username
    pub username: String,
    /// Password
    pub password: String,
    /// Database name (default "neo4j")
    pub database: Option<String>,
    /// Connection pool size (default 16)
    pub max_connections: usize,
    /// Schema cache TTL in seconds (default 300)
    pub schema_cache_ttl_secs: u64,
}

impl Default for Neo4jConfig {
    fn default() -> Self {
        Self {
            uri: "bolt://127.0.0.1:7687".to_string(),
            username: "neo4j".to_string(),
            password: "neo4j".to_string(),
            database: None,
            max_connections: 16,
            schema_cache_ttl_secs: 300,
        }
    }
}

/// Neo4j backend using neo4rs (Bolt).
pub struct Neo4jBackend {
    graph: Graph,
    #[expect(dead_code, reason = "stored from the config but not read yet")]
    database: String,
    cached_schema: RwLock<Option<(GraphSchema, Instant)>>,
    schema_cache_ttl: std::time::Duration,
}

fn neo4j_node_to_node(n: &Neo4jNode) -> Node {
    let id = n.get::<String>("id").unwrap_or_else(|_| n.id().to_string());
    let node_type = n
        .labels()
        .first()
        .map(|s| s.to_string())
        .unwrap_or_else(|| String::new());
    let mut properties: HashMap<String, serde_json::Value> = HashMap::new();
    let mut embedding: Option<Vec<f32>> = None;
    for key in n.keys() {
        if let Ok(val) = n.get::<serde_json::Value>(&key) {
            if key == "embedding" {
                if let Some(arr) = val.as_array() {
                    let vec_f32: Option<Vec<f32>> =
                        arr.iter().map(|v| v.as_f64().map(|f| f as f32)).collect();
                    if let Some(v) = vec_f32 {
                        embedding = Some(v);
                    }
                }
            }
            properties.insert(key.to_string(), val);
        }
    }
    Node {
        id,
        node_type,
        properties,
        embedding,
    }
}

fn neo4j_relation_to_edge(r: &Relation, source_id: String, target_id: String) -> Edge {
    let edge_type = r.typ().to_string();
    let mut properties: HashMap<String, serde_json::Value> = HashMap::new();
    for key in r.keys() {
        if let Ok(val) = r.get::<serde_json::Value>(&key) {
            properties.insert(key.to_string(), val);
        }
    }
    Edge {
        source_id,
        target_id,
        edge_type,
        properties,
        weight: None,
    }
}

fn neo4j_unbounded_relation_to_edge(
    r: &UnboundedRelation,
    source_id: String,
    target_id: String,
) -> Edge {
    let edge_type = r.typ().to_string();
    let mut properties: HashMap<String, serde_json::Value> = HashMap::new();
    for key in r.keys() {
        if let Ok(val) = r.get::<serde_json::Value>(&key) {
            properties.insert(key.to_string(), val);
        }
    }
    Edge {
        source_id,
        target_id,
        edge_type,
        properties,
        weight: None,
    }
}

/// Extract column names from a Cypher RETURN clause (best-effort).
fn parse_return_columns(cypher: &str) -> Vec<String> {
    let cypher_lower = cypher.to_ascii_lowercase();
    let return_pos = match cypher_lower.find(" return ") {
        Some(p) => p + 8,
        None => return vec!["result".to_string()],
    };
    let rest = &cypher[return_pos..];
    let end = rest
        .to_ascii_lowercase()
        .find(" order ")
        .or_else(|| rest.to_ascii_lowercase().find(" limit "))
        .or_else(|| rest.to_ascii_lowercase().find(" skip "))
        .unwrap_or(rest.len());
    let return_part = rest[..end].trim();
    if return_part.is_empty() {
        return vec!["result".to_string()];
    }
    let mut columns = Vec::new();
    for part in split_cypher_list(return_part) {
        let part = part.trim();
        if let Some(as_pos) = part.to_ascii_lowercase().find(" as ") {
            let alias = part[as_pos + 4..].trim();
            let alias = alias.split_whitespace().next().unwrap_or(alias);
            columns.push(alias.to_string());
        } else {
            let last = part
                .split_whitespace()
                .last()
                .unwrap_or(part)
                .trim_end_matches(',');
            columns.push(last.to_string());
        }
    }
    if columns.is_empty() {
        columns.push("result".to_string());
    }
    columns
}

fn split_cypher_list(s: &str) -> Vec<&str> {
    let mut parts = Vec::new();
    let mut depth: i32 = 0;
    let mut start = 0;
    for (i, c) in s.char_indices() {
        match c {
            '(' | '[' | '{' => depth += 1,
            ')' | ']' | '}' => depth = depth.saturating_sub(1),
            ',' if depth == 0 => {
                parts.push(s[start..i].trim());
                start = i + 1;
            }
            _ => {}
        }
    }
    if start < s.len() {
        parts.push(s[start..].trim());
    }
    parts
}

impl Neo4jBackend {
    /// Create a new Neo4j backend from config.
    pub async fn new(config: Neo4jConfig) -> Result<Self, GraphError> {
        let neo_config = ConfigBuilder::default()
            .uri(&config.uri)
            .user(&config.username)
            .password(&config.password)
            .db(config.database.as_deref().unwrap_or("neo4j"))
            .max_connections(config.max_connections)
            .build()
            .map_err(|e| GraphError::Connection(e.to_string()))?;

        let graph = Graph::connect(neo_config)
            .await
            .map_err(|e| GraphError::Connection(e.to_string()))?;

        let database = config.database.unwrap_or_else(|| "neo4j".to_string());

        Ok(Self {
            graph,
            database,
            cached_schema: RwLock::new(None),
            schema_cache_ttl: std::time::Duration::from_secs(config.schema_cache_ttl_secs),
        })
    }

    fn row_to_record(row: &neo4rs::Row, columns: &[String]) -> Result<Record, GraphError> {
        let mut fields = HashMap::new();
        for col in columns {
            let gv = row_to_graph_value(row, col).unwrap_or(GraphValue::Null);
            fields.insert(col.clone(), gv);
        }
        Ok(Record { fields })
    }
}

fn row_to_graph_value(row: &neo4rs::Row, key: &str) -> Result<GraphValue, GraphError> {
    if let Ok(n) = row.get::<Neo4jNode>(key) {
        return Ok(GraphValue::Node(neo4j_node_to_node(&n)));
    }
    if let Ok(r) = row.get::<Relation>(key) {
        let src = r.start_node_id().to_string();
        let tgt = r.end_node_id().to_string();
        return Ok(GraphValue::Edge(neo4j_relation_to_edge(&r, src, tgt)));
    }
    if let Ok(p) = row.get::<neo4rs::Path>(key) {
        let nodes: Vec<Node> = p.nodes().iter().map(neo4j_node_to_node).collect();
        let mut edges = Vec::new();
        for (i, rel) in p.rels().iter().enumerate() {
            let src = nodes.get(i).map(|n| n.id.clone()).unwrap_or_default();
            let tgt = nodes.get(i + 1).map(|n| n.id.clone()).unwrap_or_default();
            edges.push(neo4j_unbounded_relation_to_edge(rel, src, tgt));
        }
        return Ok(GraphValue::Path(Path {
            nodes,
            edges,
            total_weight: None,
        }));
    }
    if let Ok(v) = row.get::<serde_json::Value>(key) {
        return Ok(json_to_graph_value(&v));
    }
    if let Ok(i) = row.get::<i64>(key) {
        return Ok(GraphValue::Int(i));
    }
    if let Ok(f) = row.get::<f64>(key) {
        return Ok(GraphValue::Float(f));
    }
    if let Ok(b) = row.get::<bool>(key) {
        return Ok(GraphValue::Bool(b));
    }
    if let Ok(s) = row.get::<String>(key) {
        return Ok(GraphValue::String(s));
    }
    Ok(GraphValue::Null)
}

#[async_trait]
impl GraphBackend for Neo4jBackend {
    async fn add_node(&self, node: Node) -> Result<(), GraphError> {
        let type_escaped = sanitize_label(&node.node_type);
        let mut props = node.properties;
        props.insert("id".to_string(), serde_json::Value::String(node.id.clone()));
        let q = build_create_node_query(&type_escaped, &props);
        let mut query_builder = query(&q);
        for (k, v) in &props {
            match v {
                serde_json::Value::String(s) => {
                    query_builder = query_builder.param(k.as_str(), s.as_str());
                }
                serde_json::Value::Number(n) => {
                    if let Some(i) = n.as_i64() {
                        query_builder = query_builder.param(k.as_str(), i);
                    } else if let Some(f) = n.as_f64() {
                        query_builder = query_builder.param(k.as_str(), f);
                    }
                }
                serde_json::Value::Bool(b) => {
                    query_builder = query_builder.param(k.as_str(), *b);
                }
                _ => {}
            }
        }
        self.graph
            .run(query_builder)
            .await
            .map_err(|e| GraphError::QueryError(e.to_string()))?;
        Ok(())
    }

    async fn get_node(&self, node_id: &str) -> Result<Option<Node>, GraphError> {
        let mut result = self
            .graph
            .execute(query("MATCH (n {id: $id}) RETURN n").param("id", node_id))
            .await
            .map_err(|e| GraphError::QueryError(e.to_string()))?;
        let row = result
            .next()
            .await
            .map_err(|e| GraphError::QueryError(e.to_string()))?;
        Ok(row
            .and_then(|r| r.get::<Neo4jNode>("n").ok())
            .map(|n| neo4j_node_to_node(&n)))
    }

    async fn add_edge(&self, edge: Edge) -> Result<(), GraphError> {
        let type_escaped = sanitize_label(&edge.edge_type);
        let set_parts: Vec<String> = edge
            .properties
            .keys()
            .map(|k| format!("r.{} = ${}", k, k))
            .collect();
        let q = if set_parts.is_empty() {
            format!(
                "MATCH (a {{id: $src}}), (b {{id: $tgt}}) CREATE (a)-[r:{}]->(b) RETURN r",
                type_escaped
            )
        } else {
            format!(
                "MATCH (a {{id: $src}}), (b {{id: $tgt}}) CREATE (a)-[r:{}]->(b) SET {} RETURN r",
                type_escaped,
                set_parts.join(", ")
            )
        };
        let mut qb = query(&q)
            .param("src", edge.source_id.as_str())
            .param("tgt", edge.target_id.as_str());
        for (k, v) in &edge.properties {
            match v {
                serde_json::Value::String(s) => {
                    qb = qb.param(k.as_str(), s.as_str());
                }
                serde_json::Value::Number(n) => {
                    if let Some(i) = n.as_i64() {
                        qb = qb.param(k.as_str(), i);
                    } else if let Some(f) = n.as_f64() {
                        qb = qb.param(k.as_str(), f);
                    }
                }
                serde_json::Value::Bool(b) => {
                    qb = qb.param(k.as_str(), *b);
                }
                _ => {}
            }
        }
        self.graph
            .run(qb)
            .await
            .map_err(|e| GraphError::QueryError(e.to_string()))?;
        Ok(())
    }

    async fn get_neighbors(
        &self,
        node_id: &str,
        edge_types: Option<&[&str]>,
    ) -> Result<Vec<Neighbor>, GraphError> {
        let q = match edge_types {
            None => query("MATCH (n {id: $id})-[r]->(m) RETURN r, m").param("id", node_id),
            Some(types) => {
                let list: Vec<String> = types.iter().map(|s| (*s).to_string()).collect();
                query("MATCH (n {id: $id})-[r]->(m) WHERE type(r) IN $types RETURN r, m")
                    .param("id", node_id)
                    .param("types", list)
            }
        };
        let mut result = self
            .graph
            .execute(q)
            .await
            .map_err(|e| GraphError::QueryError(e.to_string()))?;
        let mut neighbors = Vec::new();
        let source_id = node_id.to_string();
        while let Ok(Some(row)) = result.next().await {
            if let (Ok(r), Ok(m)) = (row.get::<Relation>("r"), row.get::<Neo4jNode>("m")) {
                let target_id = m.get::<String>("id").unwrap_or_else(|_| m.id().to_string());
                let edge = neo4j_relation_to_edge(&r, source_id.clone(), target_id);
                let node = neo4j_node_to_node(&m);
                neighbors.push(Neighbor { node, edge });
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
        let cypher = format!(
            "MATCH p = shortestPath((a {{id: $src}})-[*1..{}]->(b {{id: $tgt}})) RETURN p",
            max_hops
        );
        let mut result = self
            .graph
            .execute(
                query(&cypher)
                    .param("src", source_id)
                    .param("tgt", target_id),
            )
            .await
            .map_err(|e| GraphError::QueryError(e.to_string()))?;
        let row = result
            .next()
            .await
            .map_err(|e| GraphError::QueryError(e.to_string()))?;
        let path = match row.and_then(|r| r.get::<neo4rs::Path>("p").ok()) {
            Some(p) => {
                let nodes: Vec<Node> = p.nodes().iter().map(neo4j_node_to_node).collect();
                let mut edges = Vec::new();
                for (i, rel) in p.rels().iter().enumerate() {
                    let src = nodes.get(i).map(|n| n.id.clone()).unwrap_or_default();
                    let tgt = nodes.get(i + 1).map(|n| n.id.clone()).unwrap_or_default();
                    edges.push(neo4j_unbounded_relation_to_edge(rel, src, tgt));
                }
                Path {
                    nodes,
                    edges,
                    total_weight: None,
                }
            }
            None => return Ok(None),
        };
        Ok(Some(path))
    }

    async fn get_nodes_by_type(&self, node_type: &str) -> Result<Vec<Node>, GraphError> {
        let type_escaped = sanitize_label(node_type);
        let cypher = format!("MATCH (n:{}) RETURN n", type_escaped);
        let mut result = self
            .graph
            .execute(query(&cypher))
            .await
            .map_err(|e| GraphError::QueryError(e.to_string()))?;
        let mut nodes = Vec::new();
        while let Ok(Some(row)) = result.next().await {
            if let Ok(n) = row.get::<Neo4jNode>("n") {
                nodes.push(neo4j_node_to_node(&n));
            }
        }
        Ok(nodes)
    }

    async fn get_edges_by_type(&self, edge_type: &str) -> Result<Vec<Edge>, GraphError> {
        let type_escaped = sanitize_label(edge_type);
        let cypher = format!(
            "MATCH (a)-[r:{}]->(b) RETURN r, a.id AS aid, b.id AS bid",
            type_escaped
        );
        let mut result = self
            .graph
            .execute(query(&cypher))
            .await
            .map_err(|e| GraphError::QueryError(e.to_string()))?;
        let mut edges = Vec::new();
        while let Ok(Some(row)) = result.next().await {
            let aid: String = row.get("aid").unwrap_or_else(|_| String::new());
            let bid: String = row.get("bid").unwrap_or_else(|_| String::new());
            if let Ok(r) = row.get::<Relation>("r") {
                edges.push(neo4j_relation_to_edge(&r, aid, bid));
            } else if let Ok(r) = row.get::<UnboundedRelation>("r") {
                edges.push(neo4j_unbounded_relation_to_edge(&r, aid, bid));
            }
        }
        Ok(edges)
    }

    async fn delete_node(&self, node_id: &str) -> Result<(), GraphError> {
        self.graph
            .run(query("MATCH (n {id: $id}) DETACH DELETE n").param("id", node_id))
            .await
            .map_err(|e| GraphError::QueryError(e.to_string()))?;
        Ok(())
    }

    async fn delete_edge(
        &self,
        source_id: &str,
        target_id: &str,
        edge_type: &str,
    ) -> Result<(), GraphError> {
        let type_escaped = sanitize_label(edge_type);
        let cypher = format!(
            "MATCH (a {{id: $src}})-[r:{}]->(b {{id: $tgt}}) DELETE r",
            type_escaped
        );
        self.graph
            .run(
                query(&cypher)
                    .param("src", source_id)
                    .param("tgt", target_id),
            )
            .await
            .map_err(|e| GraphError::QueryError(e.to_string()))?;
        Ok(())
    }

    async fn count_nodes(&self) -> Result<usize, GraphError> {
        let mut result = self
            .graph
            .execute(query("MATCH (n) RETURN count(n) AS c"))
            .await
            .map_err(|e| GraphError::QueryError(e.to_string()))?;
        let row = result
            .next()
            .await
            .map_err(|e| GraphError::QueryError(e.to_string()))?
            .ok_or_else(|| GraphError::QueryError("no count result".to_string()))?;
        let c: i64 = row
            .get("c")
            .map_err(|e| GraphError::QueryError(e.to_string()))?;
        Ok(c as usize)
    }

    async fn count_edges(&self) -> Result<usize, GraphError> {
        let mut result = self
            .graph
            .execute(query("MATCH ()-[r]->() RETURN count(r) AS c"))
            .await
            .map_err(|e| GraphError::QueryError(e.to_string()))?;
        let row = result
            .next()
            .await
            .map_err(|e| GraphError::QueryError(e.to_string()))?
            .ok_or_else(|| GraphError::QueryError("no count result".to_string()))?;
        let c: i64 = row
            .get("c")
            .map_err(|e| GraphError::QueryError(e.to_string()))?;
        Ok(c as usize)
    }
}

fn sanitize_label(s: &str) -> String {
    s.chars()
        .filter(|c| c.is_ascii_alphanumeric() || *c == '_')
        .collect::<String>()
}

fn build_create_node_query(label: &str, props: &HashMap<String, serde_json::Value>) -> String {
    let keys: Vec<String> = props.keys().cloned().collect();
    let set_parts: Vec<String> = keys.iter().map(|k| format!("n.{} = ${}", k, k)).collect();
    format!("CREATE (n:{}) SET {} RETURN n", label, set_parts.join(", "))
}

#[async_trait]
impl QueryableGraphBackend for Neo4jBackend {
    async fn execute_query(
        &self,
        query_str: &str,
        params: &QueryParams,
    ) -> Result<QueryResult, GraphError> {
        let columns = parse_return_columns(query_str);
        let mut q = query(query_str);
        q = add_params_to_query(q, params)?;
        let mut result = self
            .graph
            .execute(q)
            .await
            .map_err(|e| GraphError::QueryError(e.to_string()))?;
        let start = Instant::now();
        let mut records = Vec::new();
        while let Ok(Some(row)) = result.next().await {
            match Self::row_to_record(&row, &columns) {
                Ok(rec) => records.push(rec),
                Err(_) => {}
            }
        }
        let execution_time_ms = start.elapsed().as_millis() as u64;
        Ok(QueryResult {
            columns,
            records,
            execution_time_ms: Some(execution_time_ms),
            rows_affected: None,
        })
    }

    async fn get_schema(&self) -> Result<GraphSchema, GraphError> {
        {
            let guard = self.cached_schema.read();
            if let Some((schema, cached_at)) = guard.as_ref() {
                if cached_at.elapsed() < self.schema_cache_ttl {
                    return Ok(schema.clone());
                }
            }
        }
        self.refresh_schema().await
    }

    async fn refresh_schema(&self) -> Result<GraphSchema, GraphError> {
        let mut result = self
            .graph
            .execute(query("CALL db.labels()"))
            .await
            .map_err(|e| GraphError::SchemaError(e.to_string()))?;
        let mut vertex_labels = Vec::new();
        while let Ok(Some(row)) = result.next().await {
            if let Ok(label) = row.get::<String>("label") {
                vertex_labels.push(VertexLabel {
                    name: label,
                    properties: vec![],
                    estimated_count: None,
                });
            }
        }

        let mut result = self
            .graph
            .execute(query("CALL db.relationshipTypes()"))
            .await
            .map_err(|e| GraphError::SchemaError(e.to_string()))?;
        let mut edge_labels = Vec::new();
        while let Ok(Some(row)) = result.next().await {
            if let Ok(rel_type) = row.get::<String>("relationshipType") {
                edge_labels.push(EdgeLabel {
                    name: rel_type,
                    source_labels: vec![],
                    target_labels: vec![],
                    properties: vec![],
                    estimated_count: None,
                });
            }
        }

        for vl in &mut vertex_labels {
            let cypher = format!(
                "MATCH (n:{}) RETURN keys(n) AS k LIMIT 1",
                sanitize_label(&vl.name)
            );
            if let Ok(mut res) = self.graph.execute(query(&cypher)).await {
                if let Ok(Some(row)) = res.next().await {
                    if let Ok(keys) = row.get::<Vec<String>>("k") {
                        vl.properties = keys
                            .into_iter()
                            .map(|name| PropertyDefinition {
                                name,
                                value_type: PropertyType::Unknown,
                                indexed: false,
                                required: false,
                            })
                            .collect();
                    }
                }
            }
        }

        let schema = GraphSchema {
            vertex_labels,
            edge_labels,
            indexes: vec![],
            constraints: vec![],
            estimated_node_count: None,
            estimated_edge_count: None,
        };
        {
            let mut guard = self.cached_schema.write();
            *guard = Some((schema.clone(), Instant::now()));
        }
        Ok(schema)
    }

    async fn search_nodes_by_embedding(
        &self,
        embedding: &[f32],
        top_k: usize,
        node_type_filter: Option<&str>,
    ) -> Result<Vec<(Node, f32)>, GraphError> {
        let count = self.count_nodes().await?;
        if count > CLIENT_SIDE_NODE_LIMIT {
            tracing::warn!(
                "Client-side embedding search over {} nodes. Consider creating a vector index in Neo4j.",
                count
            );
        }
        let cypher = match node_type_filter {
            None => "MATCH (n) WHERE n.embedding IS NOT NULL RETURN n".to_string(),
            Some(t) => format!(
                "MATCH (n:{}) WHERE n.embedding IS NOT NULL RETURN n",
                sanitize_label(t)
            ),
        };
        let mut result = self
            .graph
            .execute(query(&cypher))
            .await
            .map_err(|e| GraphError::QueryError(e.to_string()))?;
        let mut nodes_with_embeddings: Vec<(Node, Vec<f32>)> = Vec::new();
        while let Ok(Some(row)) = result.next().await {
            if let Ok(neo_node) = row.get::<Neo4jNode>("n") {
                let node = neo4j_node_to_node(&neo_node);
                if let Some(emb) = node.embedding.clone() {
                    nodes_with_embeddings.push((node, emb));
                }
            }
        }
        let mut scored: Vec<(Node, f32)> = nodes_with_embeddings
            .into_iter()
            .filter_map(|(n, emb)| {
                if emb.len() == embedding.len() {
                    let sim = simd::cosine_similarity(embedding, &emb).unwrap_or(0.0);
                    Some((n, sim))
                } else {
                    None
                }
            })
            .collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(top_k);
        Ok(scored)
    }

    fn query_capabilities(&self) -> QueryCapabilities {
        QueryCapabilities {
            supports_cypher: true,
            supports_gremlin: false,
            supports_vector_index: false,
            supports_full_text: false,
            max_query_depth: None,
            supports_unified_hybrid: false,
        }
    }
}

#[cfg(test)]
mod return_column_tests {
    use super::parse_return_columns;

    /// "İ" lower-cases to two characters, and the columns hold non-ASCII text.
    #[test]
    fn return_columns_read_non_ascii_queries() {
        assert_eq!(
            parse_return_columns("MATCH (n {k:'İ'}) RETURN é, n.name AS naïve"),
            vec!["é".to_string(), "naïve".to_string()]
        );
    }
}
