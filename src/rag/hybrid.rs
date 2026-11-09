//! Hybrid graph retrieval for RAG using graph-native embedding search plus traversal.

use crate::llm::EmbeddingRequest;
use crate::llm::embedding::EmbeddingProvider;
use anyhow::Result;
use dripstone::Metric;
use dripstone::graph::{Node, QueryableGraphBackend};
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;

#[derive(Debug, Clone)]
pub enum HybridStrategy {
    Auto,
    Intersection,
    VectorThenGraph,
    GraphThenVector,
    Union {
        vector_weight: f32,
        graph_weight: f32,
    },
}

#[derive(Debug, Clone)]
pub struct HybridQuery {
    pub text_query: String,
    pub anchor_node_id: Option<String>,
    pub max_hops: usize,
    pub edge_type_filter: Option<Vec<String>>,
    pub node_type_filter: Option<Vec<String>>,
    pub vector_top_k: usize,
    pub graph_top_k: usize,
    pub strategy: HybridStrategy,
}

#[derive(Debug, Clone)]
pub struct HybridResult {
    pub node: Node,
    pub vector_score: Option<f32>,
    pub graph_distance: Option<usize>,
    pub combined_score: f32,
    pub provenance: ResultProvenance,
}

#[derive(Debug, Clone)]
pub enum ResultProvenance {
    VectorOnly,
    GraphOnly,
    Both,
}

fn resolve_auto_strategy(query: &HybridQuery) -> HybridStrategy {
    let has_anchor = query.anchor_node_id.is_some();
    let has_text = !query.text_query.is_empty();

    match (has_anchor, has_text) {
        (true, true) => HybridStrategy::Intersection,
        (true, false) => HybridStrategy::GraphThenVector,
        (false, true) => HybridStrategy::VectorThenGraph,
        (false, false) => HybridStrategy::VectorThenGraph,
    }
}

pub struct HybridGraphVectorSearch {
    graph_backend: Arc<dyn QueryableGraphBackend>,
    embedding_provider: Arc<dyn EmbeddingProvider>,
}

impl HybridGraphVectorSearch {
    pub fn new(
        graph_backend: Arc<dyn QueryableGraphBackend>,
        embedding_provider: Arc<dyn EmbeddingProvider>,
    ) -> Self {
        Self {
            graph_backend,
            embedding_provider,
        }
    }

    async fn embed_query(&self, text: &str) -> Result<Vec<f32>> {
        let request = EmbeddingRequest {
            texts: vec![text.to_string()],
            model: None,
        };
        let response = self.embedding_provider.embed(request).await?;
        let embedding = response
            .embeddings
            .into_iter()
            .next()
            .ok_or_else(|| anyhow::anyhow!("empty embedding response"))?;
        Ok(embedding)
    }

    async fn collect_reachable_nodes(
        &self,
        anchor_id: &str,
        max_hops: usize,
        edge_types: Option<&[String]>,
    ) -> Result<HashMap<String, usize>> {
        let mut reachable = HashMap::new();
        let mut queue = VecDeque::new();
        queue.push_back((anchor_id.to_string(), 0usize));
        reachable.insert(anchor_id.to_string(), 0);
        let filter: Option<Vec<&str>> =
            edge_types.map(|et| et.iter().map(|s| s.as_str()).collect());
        while let Some((current, depth)) = queue.pop_front() {
            if depth >= max_hops {
                continue;
            }
            let neighbors = self
                .graph_backend
                .get_neighbors(&current, filter.as_deref())
                .await
                .unwrap_or_default();
            for n in neighbors {
                if !reachable.contains_key(&n.node.id) {
                    reachable.insert(n.node.id.clone(), depth + 1);
                    queue.push_back((n.node.id, depth + 1));
                }
            }
        }
        Ok(reachable)
    }

    pub async fn search(&self, query: HybridQuery) -> Result<Vec<HybridResult>> {
        let strategy = match &query.strategy {
            HybridStrategy::Auto => resolve_auto_strategy(&query),
            s => s.clone(),
        };

        match strategy {
            HybridStrategy::Intersection => self.search_intersection(&query).await,
            HybridStrategy::VectorThenGraph => self.search_vector_then_graph(&query).await,
            HybridStrategy::GraphThenVector => self.search_graph_then_vector(&query).await,
            HybridStrategy::Union {
                vector_weight,
                graph_weight,
            } => self.search_union(&query, vector_weight, graph_weight).await,
            HybridStrategy::Auto => self.search_vector_then_graph(&query).await,
        }
    }

    async fn search_intersection(&self, query: &HybridQuery) -> Result<Vec<HybridResult>> {
        let anchor_id = query
            .anchor_node_id
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Intersection requires anchor_node_id"))?;
        let embedding = self.embed_query(&query.text_query).await?;
        let node_type_filter = query
            .node_type_filter
            .as_deref()
            .and_then(|v| v.first())
            .map(|s| s.as_str());
        let similar = self
            .graph_backend
            .search_nodes_by_embedding(&embedding, query.vector_top_k, node_type_filter)
            .await
            .map_err(|e| anyhow::anyhow!("search_nodes_by_embedding: {}", e))?;
        let reachable = self
            .collect_reachable_nodes(anchor_id, query.max_hops, query.edge_type_filter.as_deref())
            .await?;
        let mut results = Vec::new();
        for (node, score) in similar {
            if let Some(&dist) = reachable.get(&node.id) {
                results.push(HybridResult {
                    node,
                    vector_score: Some(score),
                    graph_distance: Some(dist),
                    combined_score: score,
                    provenance: ResultProvenance::Both,
                });
            }
        }
        results.sort_by(|a, b| {
            b.combined_score
                .partial_cmp(&a.combined_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        Ok(results)
    }

    async fn search_vector_then_graph(&self, query: &HybridQuery) -> Result<Vec<HybridResult>> {
        let embedding = self.embed_query(&query.text_query).await?;
        let node_type_filter = query
            .node_type_filter
            .as_deref()
            .and_then(|v| v.first())
            .map(|s| s.as_str());
        let similar = self
            .graph_backend
            .search_nodes_by_embedding(&embedding, query.vector_top_k, node_type_filter)
            .await
            .map_err(|e| anyhow::anyhow!("search_nodes_by_embedding: {}", e))?;
        let mut results = Vec::new();
        for (node, score) in similar {
            let neighbors = self
                .graph_backend
                .get_neighbors(&node.id, None)
                .await
                .unwrap_or_default();
            let _ = neighbors;
            results.push(HybridResult {
                node,
                vector_score: Some(score),
                graph_distance: None,
                combined_score: score,
                provenance: ResultProvenance::VectorOnly,
            });
        }
        Ok(results)
    }

    async fn search_graph_then_vector(&self, query: &HybridQuery) -> Result<Vec<HybridResult>> {
        let anchor_id = query
            .anchor_node_id
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("GraphThenVector requires anchor_node_id"))?;
        let reachable = self
            .collect_reachable_nodes(anchor_id, query.max_hops, query.edge_type_filter.as_deref())
            .await?;
        let embedding = self.embed_query(&query.text_query).await?;
        let mut candidates = Vec::new();
        for (node_id, dist) in &reachable {
            let node = match self.graph_backend.get_node(node_id).await {
                Ok(Some(n)) => n,
                _ => continue,
            };
            let score = node
                .embedding
                .as_ref()
                .map(|node_embedding| {
                    Metric::Cosine
                        .compute(&embedding, node_embedding)
                        .unwrap_or(0.0)
                })
                .unwrap_or(f32::NEG_INFINITY);
            candidates.push((node, *dist, score));
        }
        candidates.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
        let results: Vec<HybridResult> = candidates
            .into_iter()
            .take(query.graph_top_k)
            .map(|(node, dist, score)| HybridResult {
                node,
                vector_score: Some(score),
                graph_distance: Some(dist),
                combined_score: score,
                provenance: ResultProvenance::Both,
            })
            .collect();
        Ok(results)
    }

    async fn search_union(
        &self,
        query: &HybridQuery,
        vector_weight: f32,
        graph_weight: f32,
    ) -> Result<Vec<HybridResult>> {
        let mut by_id: HashMap<String, HybridResult> = HashMap::new();
        let embedding = self.embed_query(&query.text_query).await?;
        let node_type_filter = query
            .node_type_filter
            .as_deref()
            .and_then(|v| v.first())
            .map(|s| s.as_str());
        let similar = self
            .graph_backend
            .search_nodes_by_embedding(&embedding, query.vector_top_k, node_type_filter)
            .await
            .map_err(|e| anyhow::anyhow!("search_nodes_by_embedding: {}", e))?;
        for (node, score) in similar {
            by_id.insert(
                node.id.clone(),
                HybridResult {
                    node,
                    vector_score: Some(score),
                    graph_distance: None,
                    combined_score: score * vector_weight,
                    provenance: ResultProvenance::VectorOnly,
                },
            );
        }
        if let Some(anchor_id) = &query.anchor_node_id {
            let reachable = self
                .collect_reachable_nodes(
                    anchor_id,
                    query.max_hops,
                    query.edge_type_filter.as_deref(),
                )
                .await?;
            for (node_id, dist) in reachable {
                let node = match self.graph_backend.get_node(&node_id).await {
                    Ok(Some(n)) => n,
                    _ => continue,
                };
                let graph_score = 1.0 / (1.0 + dist as f32);
                if let Some(entry) = by_id.get_mut(&node_id) {
                    let vec_score = entry.vector_score.unwrap_or(0.0);
                    entry.combined_score = vec_score * vector_weight + graph_score * graph_weight;
                    entry.graph_distance = Some(dist);
                    entry.provenance = ResultProvenance::Both;
                } else {
                    by_id.insert(
                        node_id,
                        HybridResult {
                            node,
                            vector_score: None,
                            graph_distance: Some(dist),
                            combined_score: graph_score * graph_weight,
                            provenance: ResultProvenance::GraphOnly,
                        },
                    );
                }
            }
        }
        let mut results: Vec<HybridResult> = by_id.into_values().collect();
        results.sort_by(|a, b| {
            b.combined_score
                .partial_cmp(&a.combined_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        Ok(results)
    }
}
