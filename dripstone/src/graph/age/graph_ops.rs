//! [`GraphBackend`] implementation for [`super::AgeBackend`].

use super::AgeBackend;
use crate::graph::{Edge, GraphBackend, GraphError, Neighbor, Node, Path};
use async_trait::async_trait;
use std::collections::HashMap;

#[async_trait]
impl GraphBackend for AgeBackend {
    async fn add_node(&self, node: Node) -> Result<(), GraphError> {
        let type_escaped = node.node_type.replace('\'', "''");
        let mut props = node.properties;
        props.insert("id".to_string(), serde_json::Value::String(node.id));
        let map_str = Self::props_to_cypher_map(&props);
        let cypher = format!("CREATE (n:{} {}) RETURN n", type_escaped, map_str);
        self.execute_cypher_raw(&cypher, "(n agtype)").await?;
        Ok(())
    }

    async fn get_node(&self, node_id: &str) -> Result<Option<Node>, GraphError> {
        let id_escaped = node_id.replace('\'', "''");
        let cypher = format!(
            "MATCH (n {{id: '{}'}}) RETURN id(n), head(labels(n)), properties(n)",
            id_escaped
        );
        let result_columns = "(graph_id agtype, label agtype, props agtype)";
        let rows = self
            .execute_cypher_raw_text(&cypher, result_columns)
            .await?;
        Ok(rows.first().and_then(|r| {
            if r.len() >= 3 {
                let props = r.get(2).and_then(|v| v.as_object())?;
                let props_map: HashMap<String, serde_json::Value> =
                    props.iter().map(|(k, v)| (k.clone(), v.clone())).collect();
                let id = props_map
                    .get("id")
                    .and_then(|v| v.as_str())
                    .map(String::from)
                    .unwrap_or_else(|| r.get(0).map(|v| v.to_string()).unwrap_or_default());
                let node_type_str = r
                    .get(1)
                    .and_then(|v| v.as_str())
                    .map(String::from)
                    .unwrap_or_default();
                Some(Node {
                    id,
                    node_type: node_type_str,
                    properties: props_map,
                    embedding: None,
                })
            } else {
                None
            }
        }))
    }

    async fn add_edge(&self, edge: Edge) -> Result<(), GraphError> {
        let src = edge.source_id.replace('\'', "''");
        let tgt = edge.target_id.replace('\'', "''");
        let type_escaped = edge.edge_type.replace('\'', "''");
        let props_part = if edge.properties.is_empty() {
            String::new()
        } else {
            format!(" {}", Self::props_to_cypher_map(&edge.properties))
        };
        let cypher = format!(
            "MATCH (a {{id: '{}'}}), (b {{id: '{}'}}) CREATE (a)-[r:{}{}]->(b) RETURN r",
            src, tgt, type_escaped, props_part
        );
        self.execute_cypher_raw(&cypher, "(r agtype)").await?;
        Ok(())
    }

    async fn get_neighbors(
        &self,
        node_id: &str,
        edge_types: Option<&[&str]>,
    ) -> Result<Vec<Neighbor>, GraphError> {
        let id_escaped = node_id.replace('\'', "''");
        let cypher = match edge_types {
            None => format!(
                "MATCH (n {{id: '{}'}})-[r]->(m) RETURN type(r), properties(r), properties(startNode(r)), properties(endNode(r)), head(labels(endNode(r)))",
                id_escaped
            ),
            Some(types) => {
                let list: Vec<String> = types
                    .iter()
                    .map(|t| format!("'{}'", t.replace('\'', "''")))
                    .collect();
                format!(
                    "MATCH (n {{id: '{}'}})-[r]->(m) WHERE type(r) IN [{}] RETURN type(r), properties(r), properties(startNode(r)), properties(endNode(r)), head(labels(endNode(r)))",
                    id_escaped,
                    list.join(", ")
                )
            }
        };
        let result_columns = "(type_r agtype, props_r agtype, start_props agtype, end_props agtype, end_label agtype)";
        let rows = self
            .execute_cypher_raw_text(&cypher, result_columns)
            .await?;
        let mut neighbors = Vec::new();
        for row in rows {
            if row.len() >= 5 {
                let (type_r, props_r, start_props, end_props, end_label) =
                    (row.get(0), row.get(1), row.get(2), row.get(3), row.get(4));
                let edge_type = type_r.and_then(|v| v.as_str()).unwrap_or("").to_string();
                let props_r_map: HashMap<String, serde_json::Value> = props_r
                    .and_then(|v| v.as_object())
                    .map(|o| o.iter().map(|(k, v)| (k.clone(), v.clone())).collect())
                    .unwrap_or_default();
                let start_props_map: HashMap<String, serde_json::Value> = start_props
                    .and_then(|v| v.as_object())
                    .map(|o| o.iter().map(|(k, v)| (k.clone(), v.clone())).collect())
                    .unwrap_or_default();
                let end_props_map: HashMap<String, serde_json::Value> = end_props
                    .and_then(|v| v.as_object())
                    .map(|o| o.iter().map(|(k, v)| (k.clone(), v.clone())).collect())
                    .unwrap_or_default();
                let source_id = start_props_map
                    .get("id")
                    .and_then(|v| v.as_str())
                    .map(String::from)
                    .unwrap_or_else(|| {
                        start_props_map
                            .get("id")
                            .map(|v| v.to_string())
                            .unwrap_or_default()
                    });
                let target_id = end_props_map
                    .get("id")
                    .and_then(|v| v.as_str())
                    .map(String::from)
                    .unwrap_or_else(|| {
                        end_props_map
                            .get("id")
                            .map(|v| v.to_string())
                            .unwrap_or_default()
                    });
                let node_id_str = target_id.clone();
                let node_type = end_label
                    .and_then(|v| v.as_str())
                    .map(String::from)
                    .unwrap_or_else(|| "".to_string());
                let node = Node {
                    id: node_id_str,
                    node_type,
                    properties: end_props_map,
                    embedding: None,
                };
                let edge = Edge {
                    source_id,
                    target_id,
                    edge_type,
                    properties: props_r_map,
                    weight: None,
                };
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
        let src = source_id.replace('\'', "''");
        let tgt = target_id.replace('\'', "''");
        // AGE does not support shortestPath(); use variable-length path and take first result
        let cypher = format!(
            "MATCH p = (a {{id: '{}'}})-[*1..{}]->(b {{id: '{}'}}) RETURN p LIMIT 1",
            src, max_hops, tgt
        );
        let rows = self.execute_cypher_raw(&cypher, "(p agtype)").await?;
        let first = rows.first().and_then(|r| r.first());
        Ok(first.and_then(Self::path_from_agtype))
    }

    async fn get_nodes_by_type(&self, node_type: &str) -> Result<Vec<Node>, GraphError> {
        let type_escaped = node_type.replace('\'', "''");
        let cypher = format!(
            "MATCH (n:{}) RETURN id(n), head(labels(n)), properties(n)",
            type_escaped
        );
        let result_columns = "(graph_id agtype, label agtype, props agtype)";
        let rows = self
            .execute_cypher_raw_text(&cypher, result_columns)
            .await?;
        let nodes: Vec<Node> = rows
            .iter()
            .filter_map(|r| {
                if r.len() >= 3 {
                    let props = r.get(2).and_then(|v| v.as_object())?;
                    let props_map: HashMap<String, serde_json::Value> =
                        props.iter().map(|(k, v)| (k.clone(), v.clone())).collect();
                    let id = props_map
                        .get("id")
                        .and_then(|v| v.as_str())
                        .map(String::from)
                        .unwrap_or_else(|| r.get(0).map(|v| v.to_string()).unwrap_or_default());
                    let node_type_str = r
                        .get(1)
                        .and_then(|v| v.as_str())
                        .map(String::from)
                        .unwrap_or_else(|| node_type.to_string());
                    Some(Node {
                        id,
                        node_type: node_type_str,
                        properties: props_map,
                        embedding: None,
                    })
                } else {
                    None
                }
            })
            .collect();
        Ok(nodes)
    }

    async fn get_edges_by_type(&self, edge_type: &str) -> Result<Vec<Edge>, GraphError> {
        let type_escaped = edge_type.replace('\'', "''");
        let cypher = format!("MATCH (a)-[r:{}]->(b) RETURN r", type_escaped);
        let rows = self.execute_cypher_raw(&cypher, "(r agtype)").await?;
        let edges = rows
            .iter()
            .filter_map(|r| r.first().and_then(Self::parse_age_edge))
            .collect();
        Ok(edges)
    }

    async fn delete_node(&self, node_id: &str) -> Result<(), GraphError> {
        let id_escaped = node_id.replace('\'', "''");
        let cypher = format!("MATCH (n {{id: '{}'}}) DETACH DELETE n", id_escaped);
        self.execute_cypher_raw(&cypher, "(dummy agtype)").await?;
        Ok(())
    }

    async fn delete_edge(
        &self,
        source_id: &str,
        target_id: &str,
        edge_type: &str,
    ) -> Result<(), GraphError> {
        let src = source_id.replace('\'', "''");
        let tgt = target_id.replace('\'', "''");
        let type_escaped = edge_type.replace('\'', "''");
        let cypher = format!(
            "MATCH (a {{id: '{}'}})-[r:{}]->(b {{id: '{}'}}) DELETE r",
            src, type_escaped, tgt
        );
        self.execute_cypher_raw(&cypher, "(dummy agtype)").await?;
        Ok(())
    }

    async fn count_nodes(&self) -> Result<usize, GraphError> {
        let cypher = "MATCH (n) RETURN count(n) AS c";
        let values = self.execute_cypher_scalar_text(cypher, "c").await?;
        let count = values
            .first()
            .and_then(|s| super::backend::parse_age_count_text(s))
            .unwrap_or(0) as usize;
        Ok(count)
    }

    async fn count_edges(&self) -> Result<usize, GraphError> {
        let cypher = "MATCH ()-[r]->() RETURN count(r) AS c";
        let values = self.execute_cypher_scalar_text(cypher, "c").await?;
        let count = values
            .first()
            .and_then(|s| super::backend::parse_age_count_text(s))
            .unwrap_or(0) as usize;
        Ok(count)
    }
}
