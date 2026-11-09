//! AGE backend configuration and core helpers.

use super::parse::{parse_age_internal_value, split_age_map_parts};
use crate::graph::query::json_to_graph_value;
use crate::graph::{Edge, GraphError, GraphSchema, GraphValue, Node, Path, QueryParams};
use parking_lot::RwLock;
use sqlx::{Pool, Postgres, Row};
use std::collections::HashMap;
use std::time::Instant;

pub(crate) const CLIENT_SIDE_NODE_LIMIT: usize = 5_000;

/// Configuration for the AGE graph backend
#[derive(Debug, Clone)]
pub struct AgeConfig {
    /// AGE graph name
    pub graph_name: String,
    /// Schema cache TTL in seconds (default 300)
    pub schema_cache_ttl_secs: u64,
    /// Create graph if it does not exist
    pub auto_create_graph: bool,
}

impl Default for AgeConfig {
    fn default() -> Self {
        Self {
            graph_name: "graph".to_string(),
            schema_cache_ttl_secs: 300,
            auto_create_graph: true,
        }
    }
}

/// Apache AGE backend using sqlx (shared pool with PostgresBackend/pgvector).
pub struct AgeBackend {
    pub(crate) pool: Pool<Postgres>,
    pub(crate) graph_name: String,
    pub(crate) cached_schema: RwLock<Option<(GraphSchema, Instant)>>,
    pub(crate) schema_cache_ttl: std::time::Duration,
}

impl AgeBackend {
    /// Create with a new connection pool
    pub async fn new(connection_string: &str, config: AgeConfig) -> Result<Self, GraphError> {
        let pool = sqlx::postgres::PgPoolOptions::new()
            .max_connections(5)
            .connect(connection_string)
            .await
            .map_err(|e| GraphError::Connection(e.to_string()))?;
        Self::from_pool(pool, config).await
    }

    /// Create from an existing sqlx pool (shared with PostgresBackend/pgvector)
    pub async fn from_pool(pool: Pool<Postgres>, config: AgeConfig) -> Result<Self, GraphError> {
        let backend = Self {
            pool: pool.clone(),
            graph_name: config.graph_name.clone(),
            cached_schema: RwLock::new(None),
            schema_cache_ttl: std::time::Duration::from_secs(config.schema_cache_ttl_secs),
        };
        backend.initialize_age().await?;
        if config.auto_create_graph {
            backend.ensure_graph_exists().await?;
        }
        Ok(backend)
    }

    pub(crate) async fn initialize_age(&self) -> Result<(), GraphError> {
        sqlx::query("CREATE EXTENSION IF NOT EXISTS age")
            .execute(&self.pool)
            .await
            .map_err(|e| GraphError::Storage(format!("AGE extension: {}", e)))?;
        Ok(())
    }

    /// Set search_path on the given connection so ag_catalog types (e.g. graphid_ops) are visible.
    pub(crate) async fn set_search_path_on(
        conn: &mut sqlx::pool::PoolConnection<sqlx::Postgres>,
    ) -> Result<(), GraphError> {
        sqlx::query(r#"SET search_path = ag_catalog, "$user", public"#)
            .execute(conn.as_mut())
            .await
            .map_err(|e| GraphError::Storage(format!("search_path: {}", e)))?;
        Ok(())
    }

    pub(crate) async fn ensure_graph_exists(&self) -> Result<(), GraphError> {
        let mut conn = self
            .pool
            .acquire()
            .await
            .map_err(|e| GraphError::Connection(e.to_string()))?;
        Self::set_search_path_on(&mut conn).await?;
        let exists: bool =
            sqlx::query_scalar("SELECT EXISTS(SELECT 1 FROM ag_catalog.ag_graph WHERE name = $1)")
                .bind(&self.graph_name)
                .fetch_one(conn.as_mut())
                .await
                .map_err(|e| GraphError::Storage(e.to_string()))?;

        if !exists {
            let create_sql = format!(
                "SELECT ag_catalog.create_graph('{}')",
                self.graph_name.replace('\'', "''")
            );
            sqlx::query(&create_sql)
                .execute(conn.as_mut())
                .await
                .map_err(|e| GraphError::Storage(format!("create_graph: {}", e)))?;
        }
        Ok(())
    }

    /// Parse a single agtype cell string into JSON.
    ///
    /// `ag_catalog.agtype_out` appends `::vertex` / `::edge` / `::path` on composites,
    /// including nested values inside a path array. Those tokens are removed before JSON
    /// parsing. Scalar `::text` casts are not used: AGE's `agtype_to_text` errors with
    /// `agtype argument must resolve to a scalar value` for maps, vertices, edges, and paths.
    pub(crate) fn parse_agtype_cell(s: &str) -> serde_json::Value {
        let stripped = strip_agtype_annotations(s);
        let normalized = stripped.trim();
        if let Ok(v) = serde_json::from_str(normalized) {
            return v;
        }
        if let Some(v) = Self::parse_age_internal_format(normalized) {
            return v;
        }
        serde_json::Value::Null
    }

    /// SQL projection that turns every agtype column into text via `agtype_out`.
    ///
    /// `column::text` calls `agtype_to_text`, which rejects non-scalars. `agtype_out`
    /// returns the cstring form for vertices, edges, paths, maps, and scalars.
    pub(crate) fn agtype_out_projection(result_columns: &str) -> Result<String, GraphError> {
        let names = Self::parse_result_column_names(result_columns);
        if names.is_empty() {
            return Err(GraphError::QueryError(
                "AGE result column list is empty".to_string(),
            ));
        }
        let mut parts = Vec::with_capacity(names.len());
        for name in names {
            if !is_sql_identifier(&name) {
                return Err(GraphError::QueryError(format!(
                    "invalid AGE result column name {name:?}"
                )));
            }
            parts.push(format!("ag_catalog.agtype_out({name})::text"));
        }
        Ok(parts.join(", "))
    }

    /// Rebuild a [`Path`] from an AGE path array (`[vertex, edge, vertex, ...]`)
    /// or from an object that already has `nodes` and `edges` arrays.
    pub(crate) fn path_from_agtype(val: &serde_json::Value) -> Option<Path> {
        if let Some(arr) = val.as_array() {
            let mut nodes = Vec::new();
            let mut edges = Vec::new();
            for (index, item) in arr.iter().enumerate() {
                if index % 2 == 0 {
                    if let Some(node) = Self::parse_age_vertex(item) {
                        nodes.push(node);
                    }
                } else if let Some(edge) = Self::parse_age_edge(item) {
                    edges.push(edge);
                }
            }
            if nodes.is_empty() {
                return None;
            }
            return Some(Path {
                nodes,
                edges,
                total_weight: None,
            });
        }
        let obj = val.as_object()?;
        let nodes_arr = obj.get("nodes")?.as_array()?;
        let edges_arr = obj.get("edges").and_then(|v| v.as_array());
        let mut nodes = Vec::new();
        let mut edges = Vec::new();
        for n in nodes_arr {
            if let Some(node) = Self::parse_age_vertex(n) {
                nodes.push(node);
            }
        }
        if let Some(edges_arr) = edges_arr {
            for e in edges_arr {
                if let Some(edge) = Self::parse_age_edge(e) {
                    edges.push(edge);
                }
            }
        }
        if nodes.is_empty() {
            return None;
        }
        Some(Path {
            nodes,
            edges,
            total_weight: None,
        })
    }

    /// Parse AGE internal format: `{id: 0; label: 'Person'; properties: {id: 'person-0'}}` (semicolons, unquoted keys) into JSON.
    pub(crate) fn parse_age_internal_format(s: &str) -> Option<serde_json::Value> {
        let s = s.trim();
        if !s.starts_with('{') || !s.ends_with('}') {
            return None;
        }
        let inner = &s[1..s.len() - 1];
        let mut obj = serde_json::Map::new();
        for part in split_age_map_parts(inner) {
            let (key, val_str) = part.split_once(':').map(|(k, v)| (k.trim(), v.trim()))?;
            let key = key.trim_matches(|c| c == '"' || c == '\'');
            let value = parse_age_internal_value(val_str)?;
            obj.insert(key.to_string(), value);
        }
        Some(serde_json::Value::Object(obj))
    }

    /// Execute Cypher returning only scalar/map columns, with each column cast to text in SQL
    /// so the driver returns String. Use for get_neighbors/get_nodes_by_type when vertex/edge
    /// agtype cannot be decoded by the driver. result_columns e.g. "(type_r agtype, props_r agtype)".
    pub(crate) async fn execute_cypher_raw_text(
        &self,
        cypher: &str,
        result_columns: &str,
    ) -> Result<Vec<Vec<serde_json::Value>>, GraphError> {
        // Same text projection as [`Self::execute_cypher_raw`]. The previous
        // `column::text` cast called `agtype_to_text` and failed on `properties()`
        // maps with "agtype argument must resolve to a scalar value".
        self.execute_cypher_raw(cypher, result_columns).await
    }

    /// Parse result_columns e.g. "(r agtype, m agtype)" into column names ["r", "m"].
    pub(crate) fn parse_result_column_names(result_columns: &str) -> Vec<String> {
        let inner = result_columns
            .trim()
            .trim_start_matches('(')
            .trim_end_matches(')');
        inner
            .split(',')
            .map(|s| s.trim().split_whitespace().next().unwrap_or("").to_string())
            .filter(|s| !s.is_empty())
            .collect()
    }

    /// Execute Cypher through AGE's SQL wrapper, return parsed JSON results.
    /// result_columns e.g. "(v agtype)" or "(r agtype, m agtype)"
    /// Columns are projected with `ag_catalog.agtype_out(...)::text` so maps, vertices,
    /// edges, and paths are text. A direct `agtype::text` cast errors with
    /// "agtype argument must resolve to a scalar value".
    pub(crate) async fn execute_cypher_raw(
        &self,
        cypher: &str,
        result_columns: &str,
    ) -> Result<Vec<Vec<serde_json::Value>>, GraphError> {
        let mut conn = self
            .pool
            .acquire()
            .await
            .map_err(|e| GraphError::Connection(e.to_string()))?;
        Self::set_search_path_on(&mut conn).await?;
        let cypher_escaped = cypher.replace("$cypher$", "$_cypher_$");
        let projection = Self::agtype_out_projection(result_columns)?;
        let sql = format!(
            "SELECT {projection} FROM ag_catalog.cypher('{}', $cypher$ {} $cypher$) AS {result_columns}",
            self.graph_name.replace('\'', "''"),
            cypher_escaped,
        );

        let rows = sqlx::query(&sql)
            .fetch_all(conn.as_mut())
            .await
            .map_err(|e| GraphError::QueryError(e.to_string()))?;

        let num_cols = rows.first().map(|r| r.len()).unwrap_or(0);
        let mut results = Vec::with_capacity(rows.len());

        for row in &rows {
            let mut cols = Vec::with_capacity(num_cols);
            for i in 0..num_cols {
                let raw: Option<String> = row.try_get::<String, _>(i).ok().or_else(|| {
                    row.try_get::<Vec<u8>, _>(i)
                        .ok()
                        .and_then(|b| String::from_utf8(b).ok())
                });
                let json = raw
                    .map(|s| Self::parse_agtype_cell(&s))
                    .unwrap_or(serde_json::Value::Null);
                cols.push(json);
            }
            results.push(cols);
        }
        Ok(results)
    }

    pub(crate) fn parse_age_vertex(val: &serde_json::Value) -> Option<Node> {
        let label = val.get("label")?.as_str()?;
        let props: HashMap<String, serde_json::Value> =
            serde_json::from_value(val.get("properties")?.clone()).ok()?;
        let id = props
            .get("id")
            .and_then(|v| v.as_str())
            .map(String::from)
            .unwrap_or_else(|| val.get("id").map(|v| v.to_string()).unwrap_or_default());
        Some(Node {
            id,
            node_type: label.to_string(),
            properties: props,
            embedding: None,
        })
    }

    pub(crate) fn props_to_cypher_map(props: &HashMap<String, serde_json::Value>) -> String {
        let parts: Vec<String> = props
            .iter()
            .map(|(k, v)| {
                let key_escaped = k.replace('\\', "\\\\").replace('"', "\\\"");
                let val_str = match v {
                    serde_json::Value::Null => "null".to_string(),
                    serde_json::Value::Bool(b) => b.to_string(),
                    serde_json::Value::Number(n) => {
                        if n.is_i64() {
                            n.as_i64().unwrap_or(0).to_string()
                        } else {
                            n.as_f64().unwrap_or(0.0).to_string()
                        }
                    }
                    serde_json::Value::String(s) => format!("'{}'", s.replace('\'', "''")),
                    _ => format!("'{}'", v.to_string().replace('\'', "''")),
                };
                format!("{}: {}", key_escaped, val_str)
            })
            .collect();
        format!("{{{}}}", parts.join(", "))
    }

    pub(crate) fn parse_age_edge(val: &serde_json::Value) -> Option<Edge> {
        let label = val.get("label")?.as_str()?;
        let start_id = val
            .get("start_id")
            .or_else(|| val.get("startid"))
            .map(|v| v.to_string())
            .unwrap_or_default()
            .trim_matches('"')
            .to_string();
        let end_id = val
            .get("end_id")
            .or_else(|| val.get("endid"))
            .map(|v| v.to_string())
            .unwrap_or_default()
            .trim_matches('"')
            .to_string();
        let props = val
            .get("properties")
            .and_then(|v| serde_json::from_value(v.clone()).ok())
            .unwrap_or_default();
        Some(Edge {
            source_id: start_id,
            target_id: end_id,
            edge_type: label.to_string(),
            properties: props,
            weight: None,
        })
    }

    pub(crate) fn value_to_graph_value(val: &serde_json::Value) -> GraphValue {
        if let Some(obj) = val.as_object() {
            if obj.contains_key("label") && obj.contains_key("properties") {
                let is_edge = obj.contains_key("start_id")
                    || obj.contains_key("end_id")
                    || obj.contains_key("startid")
                    || obj.contains_key("endid");
                if is_edge {
                    if let Some(edge) = Self::parse_age_edge(val) {
                        return GraphValue::Edge(edge);
                    }
                }
                if let Some(node) = Self::parse_age_vertex(val) {
                    return GraphValue::Node(node);
                }
            }
        }
        json_to_graph_value(val)
    }

    pub(crate) fn has_pgvector_table(&self) -> bool {
        false
    }

    pub(crate) fn search_via_pgvector_cte(
        &self,
        _embedding: &[f32],
        _top_k: usize,
        _node_type_filter: Option<&str>,
        _vector_table: &str,
    ) -> Result<Vec<(Node, f32)>, GraphError> {
        Err(GraphError::Unsupported(
            "pgvector CTE not configured for this backend".to_string(),
        ))
    }

    #[expect(dead_code, reason = "nothing calls it yet")]
    pub(crate) fn agtype_to_i64(v: &serde_json::Value) -> i64 {
        if let Some(n) = v.as_i64() {
            return n;
        }
        if let Some(n) = v.as_f64() {
            return n as i64;
        }
        if let Some(s) = v.as_str() {
            if let Ok(n) = s.parse::<i64>() {
                return n;
            }
            if let Ok(n) = s.parse::<f64>() {
                return n as i64;
            }
        }
        if let Some(obj) = v.as_object() {
            if let Some(n) = obj.get("value").and_then(|v| v.as_i64()) {
                return n;
            }
            if let Some(n) = obj.get("value").and_then(|v| v.as_f64()) {
                return n as i64;
            }
        }
        0
    }

    /// Run Cypher that returns a single scalar column; cast to text so we get a reliable string (avoids agtype_value_to_text unsupported for vertex/edge/path).
    pub(crate) async fn execute_cypher_scalar_text(
        &self,
        cypher: &str,
        column_name: &str,
    ) -> Result<Vec<String>, GraphError> {
        let mut conn = self
            .pool
            .acquire()
            .await
            .map_err(|e| GraphError::Connection(e.to_string()))?;
        Self::set_search_path_on(&mut conn).await?;
        if !is_sql_identifier(column_name) {
            return Err(GraphError::QueryError(format!(
                "invalid AGE scalar column name {column_name:?}"
            )));
        }
        let cypher_escaped = cypher.replace("$cypher$", "$_cypher_$");
        let sql = format!(
            "SELECT ag_catalog.agtype_out({column_name})::text FROM ag_catalog.cypher('{}', $cypher$ {} $cypher$) AS ({column_name} agtype)",
            self.graph_name.replace('\'', "''"),
            cypher_escaped,
        );
        let rows = sqlx::query(&sql)
            .fetch_all(conn.as_mut())
            .await
            .map_err(|e| GraphError::QueryError(e.to_string()))?;
        let mut out = Vec::with_capacity(rows.len());
        for row in &rows {
            let s: Option<String> = row.try_get(0).ok();
            if let Some(s) = s {
                out.push(s);
            }
        }
        Ok(out)
    }

    /// SQL expression for the third argument to `ag_catalog.cypher` from scalar [`QueryParams`]
    /// (null, bool, number, string only).
    ///
    /// Keys must be valid Cypher parameter names (`$key` in the query): ASCII letters, digits, and `_`.
    pub(crate) fn query_params_to_agtype_map_sql(
        params: &QueryParams,
    ) -> Result<String, GraphError> {
        let mut chunks = Vec::with_capacity(params.len().saturating_mul(2));
        for (k, v) in params {
            if k.is_empty() || !k.chars().all(|c| c.is_ascii_alphanumeric() || c == '_') {
                return Err(GraphError::QueryError(format!(
                    "invalid AGE Cypher parameter name {k:?}: use letters, digits, and underscore only"
                )));
            }
            let key_sql = k.replace('\'', "''");
            let value_sql = match v {
                serde_json::Value::Null => "ag_catalog.agtype_in('null'::cstring)".to_string(),
                serde_json::Value::Bool(true) => {
                    "ag_catalog.agtype_in('true'::cstring)".to_string()
                }
                serde_json::Value::Bool(false) => {
                    "ag_catalog.agtype_in('false'::cstring)".to_string()
                }
                serde_json::Value::Number(n) => {
                    let lit = if let Some(i) = n.as_i64() {
                        i.to_string()
                    } else if let Some(u) = n.as_u64() {
                        u.to_string()
                    } else if let Some(f) = n.as_f64() {
                        if !f.is_finite() {
                            return Err(GraphError::QueryError(
                                "non-finite number not allowed in AGE query parameters".into(),
                            ));
                        }
                        format!("{f}")
                    } else {
                        return Err(GraphError::QueryError(
                            "invalid number in AGE query parameters".into(),
                        ));
                    };
                    let lit_esc = lit.replace('\'', "''");
                    format!("ag_catalog.agtype_in('{lit_esc}'::cstring)")
                }
                serde_json::Value::String(s) => {
                    let as_json_string = serde_json::to_string(s)
                        .map_err(|e| GraphError::QueryError(e.to_string()))?;
                    let cy_esc = as_json_string.replace('\'', "''");
                    format!("ag_catalog.agtype_in('{cy_esc}'::cstring)")
                }
                serde_json::Value::Array(_) | serde_json::Value::Object(_) => {
                    return Err(GraphError::Unsupported(
                        "AGE Cypher parameters support only null, bool, number, and string JSON values"
                            .to_string(),
                    ));
                }
            };
            chunks.push(format!("'{key_sql}', {value_sql}"));
        }
        Ok(format!(
            "ag_catalog.agtype_build_map({})",
            chunks.join(", ")
        ))
    }
}

fn is_sql_identifier(name: &str) -> bool {
    let mut chars = name.chars();
    match chars.next() {
        Some(c) if c.is_ascii_alphabetic() || c == '_' => {}
        _ => return false,
    }
    chars.all(|c| c.is_ascii_alphanumeric() || c == '_')
}

/// Drop AGE type tags so path arrays and vertices parse as JSON.
fn strip_agtype_annotations(s: &str) -> String {
    s.trim()
        .replace("::vertex", "")
        .replace("::edge", "")
        .replace("::path", "")
}

/// Parse a count cell from `agtype_out` (`45`, `"45"`, or `45::numeric`).
pub(crate) fn parse_age_count_text(raw: &str) -> Option<i64> {
    let trimmed = raw.trim();
    let trimmed = trimmed
        .strip_suffix("::numeric")
        .or_else(|| trimmed.strip_suffix("::integer"))
        .or_else(|| trimmed.strip_suffix("::int"))
        .unwrap_or(trimmed)
        .trim()
        .trim_matches('"')
        .trim();
    trimmed.parse::<i64>().ok().or_else(|| {
        let number = trimmed.parse::<f64>().ok()?;
        if number.is_finite() {
            Some(number as i64)
        } else {
            None
        }
    })
}
