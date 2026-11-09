//! [`QueryableGraphBackend`] implementation for [`super::AgeBackend`].

use super::AgeBackend;
use super::backend::CLIENT_SIDE_NODE_LIMIT;
use crate::graph::query::Record;
use crate::graph::{
    EdgeLabel, GraphBackend, GraphError, GraphSchema, Node, PropertyDefinition, PropertyType,
    QueryCapabilities, QueryParams, QueryResult, QueryableGraphBackend, VertexLabel,
};
use crate::simd;
use async_trait::async_trait;
use sqlx::Row;
use std::collections::HashMap;
use std::time::Instant;

#[async_trait]
impl QueryableGraphBackend for AgeBackend {
    async fn execute_query(
        &self,
        query: &str,
        params: &QueryParams,
    ) -> Result<QueryResult, GraphError> {
        let mut conn = self
            .pool
            .acquire()
            .await
            .map_err(|e| GraphError::Connection(e.to_string()))?;
        Self::set_search_path_on(&mut conn).await?;
        let cypher_escaped = query.replace("$cypher$", "$_cypher_$");
        let sql = if params.is_empty() {
            format!(
                "SELECT ag_catalog.agtype_out(result)::text FROM ag_catalog.cypher('{}', $cypher$ {} $cypher$) AS (result agtype)",
                self.graph_name.replace('\'', "''"),
                cypher_escaped
            )
        } else {
            let map_sql = Self::query_params_to_agtype_map_sql(params)?;
            format!(
                "SELECT ag_catalog.agtype_out(result)::text FROM ag_catalog.cypher('{}', $cypher$ {} $cypher$, {}) AS (result agtype)",
                self.graph_name.replace('\'', "''"),
                cypher_escaped,
                map_sql
            )
        };
        let rows = sqlx::query(&sql)
            .fetch_all(conn.as_mut())
            .await
            .map_err(|e| GraphError::QueryError(e.to_string()))?;

        let columns = vec!["result".to_string()];
        let mut records = Vec::new();
        for row in &rows {
            let mut fields = HashMap::new();
            if let Ok(text) = row.try_get::<String, _>(0) {
                let val = Self::parse_agtype_cell(&text);
                let gv = Self::value_to_graph_value(&val);
                fields.insert("result".to_string(), gv);
            }
            records.push(Record { fields });
        }
        Ok(QueryResult {
            columns,
            records,
            execution_time_ms: None,
            rows_affected: None,
        })
    }

    async fn get_schema(&self) -> Result<GraphSchema, GraphError> {
        {
            let guard = self.cached_schema.read();
            if let Some((ref schema, instant)) = *guard {
                if instant.elapsed() < self.schema_cache_ttl {
                    return Ok(schema.clone());
                }
            }
        }

        let schema = self.refresh_schema().await?;
        {
            let mut guard = self.cached_schema.write();
            *guard = Some((schema.clone(), Instant::now()));
        }
        Ok(schema)
    }

    async fn refresh_schema(&self) -> Result<GraphSchema, GraphError> {
        let mut conn = self
            .pool
            .acquire()
            .await
            .map_err(|e| GraphError::Connection(e.to_string()))?;
        Self::set_search_path_on(&mut conn).await?;

        let vertex_labels: Vec<String> = sqlx::query_scalar(
            "SELECT name FROM ag_catalog.ag_label WHERE graph = (SELECT graphid FROM ag_catalog.ag_graph WHERE name = $1) AND kind = 'v'",
        )
        .bind(&self.graph_name)
        .fetch_all(conn.as_mut())
        .await
        .map_err(|e| GraphError::SchemaError(e.to_string()))?;

        let edge_labels: Vec<String> = sqlx::query_scalar(
            "SELECT name FROM ag_catalog.ag_label WHERE graph = (SELECT graphid FROM ag_catalog.ag_graph WHERE name = $1) AND kind = 'e'",
        )
        .bind(&self.graph_name)
        .fetch_all(conn.as_mut())
        .await
        .map_err(|e| GraphError::SchemaError(e.to_string()))?;

        let mut vertex_label_infos = Vec::new();
        for name in &vertex_labels {
            let properties = vec![PropertyDefinition {
                name: "id".to_string(),
                value_type: PropertyType::String,
                indexed: false,
                required: false,
            }];
            vertex_label_infos.push(VertexLabel {
                name: name.clone(),
                properties,
                estimated_count: None,
            });
        }

        let mut edge_label_infos = Vec::new();
        for name in &edge_labels {
            edge_label_infos.push(EdgeLabel {
                name: name.clone(),
                source_labels: vec![],
                target_labels: vec![],
                properties: vec![],
                estimated_count: None,
            });
        }

        Ok(GraphSchema {
            vertex_labels: vertex_label_infos,
            edge_labels: edge_label_infos,
            indexes: vec![],
            constraints: vec![],
            estimated_node_count: None,
            estimated_edge_count: None,
        })
    }

    async fn search_nodes_by_embedding(
        &self,
        embedding: &[f32],
        top_k: usize,
        node_type_filter: Option<&str>,
    ) -> Result<Vec<(Node, f32)>, GraphError> {
        if self.has_pgvector_table() {
            return self.search_via_pgvector_cte(embedding, top_k, node_type_filter, "embeddings");
        }

        let node_count = self.count_nodes().await?;
        if node_count > CLIENT_SIDE_NODE_LIMIT {
            tracing::warn!(
                "Client-side embedding search over {} nodes. Performance will degrade. \
                 Configure pgvector co-location for production use.",
                node_count
            );
        }

        let candidates: Vec<Node> = if let Some(ty) = node_type_filter {
            self.get_nodes_by_type(ty).await?
        } else {
            let cypher = "MATCH (n) RETURN n";
            let rows = self.execute_cypher_raw(cypher, "(n agtype)").await?;
            rows.iter()
                .filter_map(|r| r.first().and_then(Self::parse_age_vertex))
                .collect()
        };

        let mut with_scores: Vec<(Node, f32)> = Vec::new();
        for n in candidates {
            let score = match n.embedding.as_ref() {
                Some(e) => simd::cosine_similarity(embedding, e).unwrap_or(0.0),
                None => continue,
            };
            with_scores.push((n, score));
        }
        with_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        with_scores.truncate(top_k);
        Ok(with_scores)
    }

    fn query_capabilities(&self) -> QueryCapabilities {
        QueryCapabilities {
            supports_cypher: true,
            supports_gremlin: false,
            supports_vector_index: false,
            supports_full_text: false,
            max_query_depth: None,
            supports_unified_hybrid: self.has_pgvector_table(),
        }
    }
}
