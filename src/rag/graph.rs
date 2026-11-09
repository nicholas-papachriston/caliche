//! Graph RAG implementation for multi-hop reasoning
//!
//! This module provides functionality to extract entities and relationships from text,
//! build knowledge graphs, and perform graph-enhanced retrieval.

use crate::llm::Provider;
use crate::llm::embedding::{EmbeddingProvider, EmbeddingRequest};
use crate::messages::{CompletionRequest, Message, MessageContent, MessageRole};
use crate::prompts::PromptTemplate;
use crate::utils::math::cosine_similarity;
use dripstone::graph::{Edge, GraphBackend, Node};
use serde_json::json;
use std::collections::HashMap;
use std::sync::Arc;

/// Extracted entity-relationship triplets from text
#[derive(Debug, Clone)]
pub struct ExtractedTriplet {
    pub subject: String,
    pub relation: String,
    pub object: String,
    pub subject_type: Option<String>,
    pub object_type: Option<String>,
}

/// Graph indexer that extracts entities and relationships from documents
pub struct GraphIndexer {
    embedding_provider: Arc<dyn EmbeddingProvider>,
    graph_backend: Arc<dyn GraphBackend>,
    llm_provider: Option<Arc<dyn Provider>>,
}

impl GraphIndexer {
    /// Create a new graph indexer
    pub fn new(
        embedding_provider: Arc<dyn EmbeddingProvider>,
        graph_backend: Arc<dyn GraphBackend>,
        llm_provider: Option<Arc<dyn Provider>>,
    ) -> Self {
        Self {
            embedding_provider,
            graph_backend,
            llm_provider,
        }
    }

    /// Extract entities and relationships from a text chunk
    pub async fn extract_triplets(
        &self,
        text: &str,
        _chunk_id: &str,
    ) -> anyhow::Result<Vec<ExtractedTriplet>> {
        // If we have an LLM provider, use it for extraction
        if let Some(ref provider) = self.llm_provider {
            self.extract_with_llm(provider, text).await
        } else {
            // Fallback: simple pattern-based extraction (very basic)
            Ok(self.extract_simple(text))
        }
    }

    /// Extract triplets using LLM
    async fn extract_with_llm(
        &self,
        provider: &Arc<dyn Provider>,
        text: &str,
    ) -> anyhow::Result<Vec<ExtractedTriplet>> {
        let template = PromptTemplate::new(
            r#"Extract all entity-relationship triplets from the following text.
Return ONLY a JSON array of objects with this exact structure:
[
  {
    "subject": "entity name",
    "relation": "relationship type",
    "object": "related entity name",
    "subject_type": "Person|Company|Location|Concept|Other",
    "object_type": "Person|Company|Location|Concept|Other"
  }
]

Text to analyze:
{text}

JSON array:"#,
        );

        let prompt = template.format(&json!({"text": text}))?;

        let messages = Arc::new(vec![Message {
            role: MessageRole::User,
            content: MessageContent::text(prompt),
            metadata: None,
        }]);

        let mut request = CompletionRequest::new(messages);
        request.temperature = Some(0.0); // Deterministic extraction
        request.max_tokens = Some(2000);

        let response = provider.complete(request).await?;
        let content = response.content.extract_text();

        // Try to parse JSON from the response
        // LLM might wrap it in markdown code blocks
        let json_str = caliche_contracts::text::extract_json(&content).unwrap_or(&content);

        let triplets: Vec<serde_json::Value> = serde_json::from_str(json_str)
            .map_err(|e| anyhow::anyhow!("Failed to parse extraction JSON: {}", e))?;

        let mut extracted = Vec::new();
        for triplet in triplets {
            if let (Some(subject), Some(relation), Some(object)) = (
                triplet.get("subject").and_then(|v| v.as_str()),
                triplet.get("relation").and_then(|v| v.as_str()),
                triplet.get("object").and_then(|v| v.as_str()),
            ) {
                extracted.push(ExtractedTriplet {
                    subject: subject.to_string(),
                    relation: relation.to_string(),
                    object: object.to_string(),
                    subject_type: triplet
                        .get("subject_type")
                        .and_then(|v| v.as_str())
                        .map(|s| s.to_string()),
                    object_type: triplet
                        .get("object_type")
                        .and_then(|v| v.as_str())
                        .map(|s| s.to_string()),
                });
            }
        }

        Ok(extracted)
    }

    /// Simple pattern-based extraction using heuristic verb-phrase patterns.
    ///
    /// Splits on sentence boundaries and detects "Subject Verb Object" triples
    /// where the subject and object are capitalized noun phrases. No NER library
    /// is required; this is intentionally low-precision but always produces
    /// some signal when an LLM provider is not available.
    fn extract_simple(&self, text: &str) -> Vec<ExtractedTriplet> {
        // Verbs that indicate a relationship worth indexing
        const RELATION_VERBS: &[&str] = &[
            "is",
            "are",
            "was",
            "were",
            "has",
            "have",
            "had",
            "uses",
            "used",
            "implements",
            "extends",
            "contains",
            "provides",
            "requires",
            "depends on",
            "belongs to",
            "created by",
            "owned by",
        ];

        let mut results = Vec::new();

        for sentence in text.split(['.', '\n', ';']) {
            let sentence = sentence.trim();
            if sentence.len() < 6 {
                continue;
            }

            let words: Vec<&str> = sentence.split_whitespace().collect();
            if words.len() < 3 {
                continue;
            }

            // Find a relation verb in the sentence
            for verb in RELATION_VERBS {
                let verb_words: Vec<&str> = verb.split_whitespace().collect();
                let positions: Vec<usize> = words
                    .windows(verb_words.len())
                    .enumerate()
                    .filter(|(_, w)| {
                        w.iter()
                            .zip(verb_words.iter())
                            .all(|(a, b)| a.to_lowercase() == *b)
                    })
                    .map(|(i, _)| i)
                    .collect();

                for verb_pos in positions {
                    if verb_pos == 0 {
                        continue;
                    }
                    let verb_end = verb_pos + verb_words.len();
                    if verb_end >= words.len() {
                        continue;
                    }

                    let subject_words: Vec<&str> = words[..verb_pos]
                        .iter()
                        .rev()
                        .take(3)
                        .copied()
                        .collect::<Vec<_>>()
                        .into_iter()
                        .rev()
                        .collect();
                    let object_words: Vec<&str> =
                        words[verb_end..].iter().take(3).copied().collect();

                    let subject = subject_words.join(" ");
                    let object = object_words.join(" ");

                    if subject.is_empty() || object.is_empty() {
                        continue;
                    }

                    // Require at least one capitalized word as a rough noun-phrase filter
                    let subject_has_cap = subject_words
                        .iter()
                        .any(|w| w.chars().next().map(|c| c.is_uppercase()).unwrap_or(false));
                    let object_has_cap = object_words
                        .iter()
                        .any(|w| w.chars().next().map(|c| c.is_uppercase()).unwrap_or(false));
                    if !subject_has_cap && !object_has_cap {
                        continue;
                    }

                    results.push(ExtractedTriplet {
                        subject,
                        relation: verb.to_string(),
                        object,
                        subject_type: None,
                        object_type: None,
                    });
                }
            }
        }

        results
    }

    /// Index a document chunk into the graph
    pub async fn index_chunk(&self, text: &str, chunk_id: &str) -> anyhow::Result<usize> {
        let triplets = self.extract_triplets(text, chunk_id).await?;

        let mut node_ids = HashMap::new();
        let mut indexed_count = 0;

        for triplet in triplets {
            // Create or get subject node
            let subject_id = format!("entity:{}", triplet.subject);
            if !node_ids.contains_key(&subject_id) {
                let mut node = Node::new(
                    subject_id.clone(),
                    triplet.subject_type.unwrap_or_else(|| "Entity".to_owned()),
                )
                .with_property("name".to_owned(), json!(triplet.subject))
                .with_property("source_chunk".to_owned(), json!(chunk_id));

                // Generate embedding for the entity name
                match self
                    .embedding_provider
                    .embed(EmbeddingRequest {
                        texts: vec![triplet.subject.clone()],
                        model: None,
                    })
                    .await
                {
                    Ok(response) => {
                        if let Some(embedding) = response.embeddings.first() {
                            node = node.with_embedding(embedding.clone());
                        }
                    }
                    Err(e) => {
                        tracing::warn!(
                            entity = %triplet.subject,
                            "Failed to generate embedding for subject entity: {e}"
                        );
                    }
                }

                self.graph_backend.add_node(node.clone()).await?;
                node_ids.insert(subject_id.clone(), node);
                indexed_count += 1;
            }

            // Create or get object node
            let object_id = format!("entity:{}", triplet.object);
            if !node_ids.contains_key(&object_id) {
                let mut node = Node::new(
                    object_id.clone(),
                    triplet.object_type.unwrap_or_else(|| "Entity".to_owned()),
                )
                .with_property("name".to_owned(), json!(triplet.object))
                .with_property("source_chunk".to_owned(), json!(chunk_id));

                // Generate embedding for the entity name
                match self
                    .embedding_provider
                    .embed(EmbeddingRequest {
                        texts: vec![triplet.object.clone()],
                        model: None,
                    })
                    .await
                {
                    Ok(response) => {
                        if let Some(embedding) = response.embeddings.first() {
                            node = node.with_embedding(embedding.clone());
                        }
                    }
                    Err(e) => {
                        tracing::warn!(
                            entity = %triplet.object,
                            "Failed to generate embedding for object entity: {e}"
                        );
                    }
                }

                self.graph_backend.add_node(node.clone()).await?;
                node_ids.insert(object_id.clone(), node);
                indexed_count += 1;
            }

            // Create edge
            let edge = Edge::new(subject_id.clone(), object_id.clone(), triplet.relation)
                .with_property("source_chunk".to_owned(), json!(chunk_id));

            self.graph_backend.add_edge(edge).await?;
            indexed_count += 1;
        }

        Ok(indexed_count)
    }
}

/// Graph search strategy that combines vector search with graph traversal
pub struct GraphSearch {
    embedding_provider: Arc<dyn EmbeddingProvider>,
    graph_backend: Arc<dyn GraphBackend>,
}

impl GraphSearch {
    /// Create a new graph search instance
    pub fn new(
        embedding_provider: Arc<dyn EmbeddingProvider>,
        graph_backend: Arc<dyn GraphBackend>,
    ) -> Self {
        Self {
            embedding_provider,
            graph_backend,
        }
    }

    /// Perform graph-enhanced search
    ///
    /// Strategy:
    /// 1. Embed query and find top-k relevant nodes using vector similarity
    /// 2. Expand graph from these nodes (1-hop neighbors)
    /// 3. Collect context from expanded subgraph
    pub async fn search(
        &self,
        query: &str,
        top_k: usize,
        _max_hops: usize,
    ) -> anyhow::Result<GraphSearchResult> {
        // Step 1: Embed query
        let query_response = self
            .embedding_provider
            .embed(EmbeddingRequest {
                texts: vec![query.to_string()],
                model: None,
            })
            .await?;

        let query_embedding = query_response
            .embeddings
            .first()
            .ok_or_else(|| anyhow::anyhow!("No embedding generated for query"))?;

        // Step 2: Find relevant nodes using vector similarity
        // We need to search nodes by their embeddings
        // For now, we'll get all nodes and compute similarity manually
        // (In production, you'd want nodes indexed in the vector backend)
        let all_nodes = self.get_all_nodes_with_embeddings().await?;

        // Compute similarities — NLL lets us borrow `node.embedding` for similarity
        // and then move `node` into the tuple once the inner closure returns.
        let mut scored_nodes: Vec<(Node, f32)> = all_nodes
            .into_iter()
            .filter_map(|node| {
                let similarity = node
                    .embedding
                    .as_ref()
                    .map(|emb| cosine_similarity(query_embedding, emb))?;
                Some((node, similarity))
            })
            .collect();

        // Sort by similarity and take top-k
        scored_nodes.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored_nodes.truncate(top_k);

        // Collect entry node IDs before consuming scored_nodes so we can
        // preserve rank order in the result without re-cloning full nodes.
        let entry_node_ids: Vec<String> = scored_nodes.iter().map(|(n, _)| n.id.clone()).collect();

        // Step 3: Expand graph from top nodes — consume scored_nodes to move
        // each Node into the map without an extra clone.
        let mut expanded_nodes: HashMap<String, Node> = HashMap::new();
        let mut expanded_edges = Vec::new();

        for (node, _score) in scored_nodes {
            let node_id = node.id.clone();

            // Get neighbors (1-hop)
            match self.graph_backend.get_neighbors(&node_id, None).await {
                Ok(neighbors) => {
                    for neighbor in neighbors {
                        let neighbor_id = neighbor.node.id.clone();
                        expanded_nodes.insert(neighbor_id, neighbor.node);
                        expanded_edges.push(neighbor.edge);
                    }
                }
                Err(e) => {
                    tracing::warn!(
                        node_id = %node_id,
                        "Failed to get graph neighbors during search expansion: {e}"
                    );
                }
            }

            // Insert after neighbors so neighbor lookup can't return the same node
            expanded_nodes.entry(node_id).or_insert(node);
        }

        // Step 4: Build context description
        let context = self.build_context(&expanded_nodes, &expanded_edges)?;

        // Rebuild entry_nodes in rank order from expanded_nodes (cheap clone —
        // only top_k nodes, already present in the map).
        let entry_nodes = entry_node_ids
            .iter()
            .filter_map(|id| expanded_nodes.get(id).cloned())
            .collect();

        Ok(GraphSearchResult {
            entry_nodes,
            expanded_nodes: expanded_nodes.into_values().collect(),
            expanded_edges,
            context,
        })
    }

    /// Get all nodes that have embeddings by searching the vector backend
    /// Nodes are stored with a special prefix in the vector backend
    async fn get_all_nodes_with_embeddings(&self) -> anyhow::Result<Vec<Node>> {
        // Search for all nodes in the "graph_nodes" namespace
        // In a full implementation, nodes would be indexed in the vector backend
        // For now, we'll get nodes from common types
        let mut nodes = Vec::new();

        // Try to get nodes by common types
        for node_type in ["Person", "Company", "Location", "Concept", "Entity"] {
            match self.graph_backend.get_nodes_by_type(node_type).await {
                Ok(typed_nodes) => {
                    for node in typed_nodes {
                        if node.embedding.is_some() {
                            nodes.push(node);
                        }
                    }
                }
                Err(e) => {
                    tracing::warn!(node_type, "Failed to retrieve graph nodes by type: {e}");
                }
            }
        }

        Ok(nodes)
    }

    /// Build a text context description from the expanded subgraph
    fn build_context(
        &self,
        nodes: &HashMap<String, Node>,
        edges: &[Edge],
    ) -> anyhow::Result<String> {
        let mut context_parts = Vec::new();

        // Add node descriptions
        for node in nodes.values() {
            let name = node
                .properties
                .get("name")
                .and_then(|v| v.as_str())
                .unwrap_or(&node.id);
            context_parts.push(format!("{} ({})", name, node.node_type));
        }

        // Add edge descriptions
        for edge in edges {
            let source_name = nodes
                .get(&edge.source_id)
                .and_then(|n| n.properties.get("name"))
                .and_then(|v| v.as_str())
                .unwrap_or(&edge.source_id);
            let target_name = nodes
                .get(&edge.target_id)
                .and_then(|n| n.properties.get("name"))
                .and_then(|v| v.as_str())
                .unwrap_or(&edge.target_id);

            context_parts.push(format!(
                "{} {} {}",
                source_name, edge.edge_type, target_name
            ));
        }

        Ok(context_parts.join(". "))
    }
}

/// Result of a graph search operation
#[derive(Debug, Clone)]
pub struct GraphSearchResult {
    /// Entry nodes found via vector similarity
    pub entry_nodes: Vec<Node>,
    /// All nodes in the expanded subgraph
    pub expanded_nodes: Vec<Node>,
    /// All edges in the expanded subgraph
    pub expanded_edges: Vec<Edge>,
    /// Textual context description of the subgraph
    pub context: String,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::embedding::EmbeddingResponse;
    use crate::testing::MockProvider;
    use async_trait::async_trait;
    use dripstone::graph::memory::InMemoryGraphBackend;

    #[derive(Debug)]
    struct NoEmbeddings;

    #[async_trait]
    impl EmbeddingProvider for NoEmbeddings {
        async fn embed(&self, _request: EmbeddingRequest) -> anyhow::Result<EmbeddingResponse> {
            anyhow::bail!("extraction does not embed")
        }

        fn name(&self) -> &str {
            "none"
        }
    }

    async fn extract(reply: &str) -> Result<usize, String> {
        let provider: Arc<dyn Provider> = Arc::new(MockProvider::new(vec![reply.to_string()]));
        let indexer = GraphIndexer::new(
            Arc::new(NoEmbeddings),
            Arc::new(InMemoryGraphBackend::new()),
            Some(provider),
        );
        indexer
            .extract_triplets("Ada works at Acme.", "chunk-1")
            .await
            .map(|triplets| triplets.len())
            .map_err(|e| e.to_string())
    }

    /// The closing `]` comes before the opening `[`.
    #[tokio::test]
    async fn closer_before_opener_is_a_parse_error() {
        assert_eq!(
            extract("```] then [").await,
            Err("Failed to parse extraction JSON: expected value at line 1 column 1".to_string())
        );
    }

    #[tokio::test]
    async fn fenced_array_is_extracted() {
        let reply = "```json\n[{\"subject\": \"Ada\", \"relation\": \"works_at\", \"object\": \"Acme\"}]\n```";
        assert_eq!(extract(reply).await, Ok(1));
    }

    /// An object is not read as the array inside it.
    #[tokio::test]
    async fn object_holding_the_array_is_a_parse_error() {
        let reply = "```json\n{\"triplets\": [{\"subject\": \"Ada\", \"relation\": \"works_at\", \"object\": \"Acme\"}]}\n```";
        assert_eq!(
            extract(reply).await,
            Err(
                "Failed to parse extraction JSON: invalid type: map, expected a sequence at line 1 column 0"
                    .to_string()
            )
        );
    }

    #[tokio::test]
    async fn fence_after_text_is_extracted() {
        let reply = "Triplets:\n```json\n[{\"subject\": \"Ada\", \"relation\": \"works_at\", \"object\": \"Acme\"}]\n```";
        assert_eq!(extract(reply).await, Ok(1));
    }
}
