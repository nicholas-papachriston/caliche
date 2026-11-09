use crate::chains::{Chain, ChainContext, ChainError, ChainResult};
use crate::llm::embedding::{EmbeddingProvider, EmbeddingRequest};
use crate::utils::math::cosine_similarity;
use async_trait::async_trait;
use serde_json::{Value, json};
use std::collections::HashMap;
use std::fmt::Write as _;
use std::sync::Arc;

/// Chunking strategy for documents
#[derive(Debug, Clone)]
pub enum ChunkingStrategy {
    /// Fixed character size chunks with overlap
    FixedSize { size: usize, overlap: usize },
    /// Split on sentence boundaries (approximate)
    Sentence { max_size: usize },
    /// Split on paragraph boundaries
    Paragraph { max_size: usize },
    /// Semantic chunking using embedding similarity
    Semantic { threshold: f32, max_size: usize },
}

/// A document chunk with metadata
/// Uses Arc<String> for text to enable cheap sharing during grouping operations
#[derive(Debug, Clone)]
pub struct DocumentChunk {
    pub id: String,
    pub text: Arc<String>,
    pub index: usize,
}

/// A chain that processes large documents using map-reduce with optional embedding-based operations
///
/// This chain:
/// 1. Chunks the input document using the specified strategy
/// 2. Optionally embeds chunks for semantic grouping
/// 3. Maps a processing chain over each chunk (or semantic group)
/// 4. Reduces the results using a final chain
///
/// # Example
///
/// ```rust,no_run
/// use caliche::{AgentBuilder, ProviderType};
/// use caliche::chains::{Chain, DocumentMapReduceChain, LLMChain, ChunkingStrategy};
/// use caliche::prompts::PromptTemplate;
/// use caliche::OpenAIProvider;
/// use serde_json::json;
/// use std::sync::Arc;
/// use tokio::sync::Mutex;
///
/// # async fn example() -> anyhow::Result<()> {
/// let agent = AgentBuilder::new()
///     .provider(ProviderType::Bedrock)
///     .build()
///     .await?;
/// let agent = Arc::new(Mutex::new(agent));
///
/// let embedding_provider = Arc::new(OpenAIProvider::try_new(
///     "key".to_string(),
///     "text-embedding-3-small".to_string()
/// )?);
///
/// let map_template = PromptTemplate::new(
///     "Summarize this section:\n\n{text}\n\nSummary:"
/// );
/// let map_chain = Arc::new(LLMChain::new(agent.clone(), map_template));
///
/// let reduce_template = PromptTemplate::new(
///     "Combine these summaries into one:\n\n{summaries}\n\nFinal summary:"
/// );
/// let reduce_chain = Arc::new(LLMChain::new(agent, reduce_template));
///
/// let doc_chain = DocumentMapReduceChain::new(
///     map_chain,
///     reduce_chain,
/// )
/// .with_chunking(ChunkingStrategy::FixedSize {
///     size: 1000,
///     overlap: 200,
/// })
/// .with_embedding_provider(embedding_provider)
/// .with_semantic_grouping(0.8);
///
/// let result = doc_chain.run(&json!({
///     "document": "Very long document text..."
/// })).await?;
/// # Ok(())
/// # }
/// ```
pub struct DocumentMapReduceChain {
    map_chain: Arc<dyn Chain>,
    reduce_chain: Arc<dyn Chain>,
    chunking_strategy: ChunkingStrategy,
    embedding_provider: Option<Arc<dyn EmbeddingProvider>>,
    semantic_grouping_threshold: Option<f32>,
    batch_size: usize,
    document_key: String,
    text_key: String,
}

impl std::fmt::Debug for DocumentMapReduceChain {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DocumentMapReduceChain")
            .field("map_chain", &"Arc<dyn Chain>")
            .field("reduce_chain", &"Arc<dyn Chain>")
            .field("chunking_strategy", &self.chunking_strategy)
            .field(
                "embedding_provider",
                &self
                    .embedding_provider
                    .as_ref()
                    .map(|_| "Some(Arc<dyn EmbeddingProvider>)"),
            )
            .field(
                "semantic_grouping_threshold",
                &self.semantic_grouping_threshold,
            )
            .field("batch_size", &self.batch_size)
            .field("document_key", &self.document_key)
            .field("text_key", &self.text_key)
            .finish()
    }
}

impl DocumentMapReduceChain {
    /// Create a new document map-reduce chain
    pub fn new(map_chain: Arc<dyn Chain>, reduce_chain: Arc<dyn Chain>) -> Self {
        Self {
            map_chain,
            reduce_chain,
            chunking_strategy: ChunkingStrategy::FixedSize {
                size: 1000,
                overlap: 100,
            },
            embedding_provider: None,
            semantic_grouping_threshold: None,
            batch_size: 5,
            document_key: "document".to_owned(),
            text_key: "text".to_owned(),
        }
    }

    /// Set the chunking strategy
    pub fn with_chunking(mut self, strategy: ChunkingStrategy) -> Self {
        self.chunking_strategy = strategy;
        self
    }

    /// Set embedding provider for semantic operations
    pub fn with_embedding_provider(mut self, provider: Arc<dyn EmbeddingProvider>) -> Self {
        self.embedding_provider = Some(provider);
        self
    }

    /// Enable semantic grouping of similar chunks
    pub fn with_semantic_grouping(mut self, threshold: f32) -> Self {
        self.semantic_grouping_threshold = Some(threshold);
        self
    }

    /// Set batch size for parallel map operations
    pub fn with_batch_size(mut self, size: usize) -> Self {
        self.batch_size = size;
        self
    }

    /// Set the input document key
    pub fn with_document_key(mut self, key: impl Into<String>) -> Self {
        self.document_key = key.into();
        self
    }

    /// Set the key for text in map operations
    pub fn with_text_key(mut self, key: impl Into<String>) -> Self {
        self.text_key = key.into();
        self
    }

    /// Chunk a document according to the strategy
    fn chunk_document(&self, text: &str) -> Vec<DocumentChunk> {
        match &self.chunking_strategy {
            ChunkingStrategy::FixedSize { size, overlap } => {
                self.chunk_fixed_size(text, *size, *overlap)
            }
            ChunkingStrategy::Sentence { max_size } => self.chunk_sentences(text, *max_size),
            ChunkingStrategy::Paragraph { max_size } => self.chunk_paragraphs(text, *max_size),
            ChunkingStrategy::Semantic {
                threshold: _,
                max_size,
            } => {
                // Fall back to fixed size for initial chunking
                self.chunk_fixed_size(text, *max_size, max_size / 10)
            }
        }
    }

    /// Fixed-size chunking with overlap
    fn chunk_fixed_size(&self, text: &str, size: usize, overlap: usize) -> Vec<DocumentChunk> {
        let mut chunks = Vec::new();
        let chars: Vec<char> = text.chars().collect();
        let mut start = 0;
        let mut index = 0;

        while start < chars.len() {
            let end = (start + size).min(chars.len());
            let chunk_text: String = chars[start..end].iter().collect();

            chunks.push(DocumentChunk {
                id: format!("chunk_{}", index),
                text: Arc::new(chunk_text),
                index,
            });

            start += size.saturating_sub(overlap);
            index += 1;

            if start + overlap >= chars.len() && end == chars.len() {
                break;
            }
        }

        chunks
    }

    /// Sentence-based chunking
    fn chunk_sentences(&self, text: &str, max_size: usize) -> Vec<DocumentChunk> {
        let mut chunks = Vec::new();
        let mut current_chunk = String::new();
        let mut index = 0;

        // Simple sentence splitting (. ! ?)
        let sentences: Vec<&str> = text
            .split(['.', '!', '?'])
            .filter(|s| !s.trim().is_empty())
            .collect();

        for sentence in sentences {
            let sentence = sentence.trim();
            if current_chunk.len() + sentence.len() > max_size && !current_chunk.is_empty() {
                chunks.push(DocumentChunk {
                    id: format!("chunk_{}", index),
                    text: Arc::new(current_chunk),
                    index,
                });
                current_chunk = String::new();
                index += 1;
            }
            if !current_chunk.is_empty() {
                current_chunk.push(' ');
            }
            current_chunk.push_str(sentence);
        }

        if !current_chunk.is_empty() {
            chunks.push(DocumentChunk {
                id: format!("chunk_{}", index),
                text: Arc::new(current_chunk),
                index,
            });
        }

        chunks
    }

    /// Paragraph-based chunking
    fn chunk_paragraphs(&self, text: &str, max_size: usize) -> Vec<DocumentChunk> {
        let mut chunks = Vec::new();
        let mut current_chunk = String::new();
        let mut index = 0;

        let paragraphs: Vec<&str> = text
            .split("\n\n")
            .filter(|s| !s.trim().is_empty())
            .collect();

        for para in paragraphs {
            let para = para.trim();
            if current_chunk.len() + para.len() > max_size && !current_chunk.is_empty() {
                chunks.push(DocumentChunk {
                    id: format!("chunk_{}", index),
                    text: Arc::new(current_chunk),
                    index,
                });
                current_chunk = String::new();
                index += 1;
            }
            if !current_chunk.is_empty() {
                current_chunk.push_str("\n\n");
            }
            current_chunk.push_str(para);
        }

        if !current_chunk.is_empty() {
            chunks.push(DocumentChunk {
                id: format!("chunk_{}", index),
                text: Arc::new(current_chunk),
                index,
            });
        }

        chunks
    }

    /// Embed all chunks and return embeddings map keyed by chunk ID
    async fn embed_chunks(
        &self,
        chunks: &[DocumentChunk],
    ) -> Result<HashMap<String, Vec<f32>>, ChainError> {
        let provider = self
            .embedding_provider
            .as_ref()
            .ok_or_else(|| ChainError::ExecutionFailed("No embedding provider set".to_string()))?;

        // Extract texts (Arc<String> can be dereferenced cheaply)
        let texts: Vec<String> = chunks.iter().map(|c| c.text.as_str().to_owned()).collect();

        let request = EmbeddingRequest { texts, model: None };

        let response = provider
            .embed(request)
            .await
            .map_err(|e| ChainError::ExecutionFailed(format!("Embedding failed: {}", e)))?;

        // Build embeddings map
        let mut embeddings = HashMap::with_capacity(chunks.len());
        for (chunk, embedding) in chunks.iter().zip(response.embeddings) {
            embeddings.insert(chunk.id.clone(), embedding);
        }

        Ok(embeddings)
    }

    /// Group chunks by semantic similarity using embeddings map
    fn group_chunks(
        &self,
        chunks: Vec<DocumentChunk>,
        embeddings: &HashMap<String, Vec<f32>>,
        threshold: f32,
    ) -> Vec<Vec<DocumentChunk>> {
        let mut groups: Vec<Vec<DocumentChunk>> = Vec::new();

        for chunk in chunks {
            let mut added = false;
            let chunk_emb = embeddings.get(&chunk.id);

            // Try to add to an existing group
            for group in &mut groups {
                if let (Some(chunk_emb), Some(first_chunk_emb)) =
                    (chunk_emb, embeddings.get(&group[0].id))
                {
                    let similarity = cosine_similarity(chunk_emb, first_chunk_emb);
                    if similarity >= threshold {
                        // Clone chunk (text is Arc<String>, so cheap)
                        group.push(chunk.clone());
                        added = true;
                        break;
                    }
                }
            }

            // Create new group if not added
            if !added {
                groups.push(vec![chunk]);
            }
        }

        groups
    }
}

#[async_trait]
impl Chain for DocumentMapReduceChain {
    async fn run(&self, input: &Value) -> Result<ChainResult, ChainError> {
        let mut context = ChainContext::new();

        // Extract document text
        let document = input
            .get(&self.document_key)
            .and_then(|v| v.as_str())
            .ok_or_else(|| {
                ChainError::InvalidInput(format!("Missing '{}' in input", self.document_key))
            })?;

        // Chunk the document
        let chunks = self.chunk_document(document);
        context.add_step(
            "chunking",
            json!({
                "strategy": format!("{:?}", self.chunking_strategy),
                "num_chunks": chunks.len(),
            }),
        );

        // Optionally embed chunks and store in separate map
        let embeddings = if self.embedding_provider.is_some()
            && (self.semantic_grouping_threshold.is_some()
                || matches!(self.chunking_strategy, ChunkingStrategy::Semantic { .. }))
        {
            let emb = self.embed_chunks(&chunks).await?;
            context.add_step("embedding", json!({"num_embedded": chunks.len()}));
            Some(emb)
        } else {
            None
        };

        // Optionally group by semantic similarity
        let chunk_groups = if let Some(threshold) = self.semantic_grouping_threshold {
            let emb_map = embeddings.as_ref().ok_or_else(|| {
                ChainError::ExecutionFailed(
                    "Semantic grouping requires embedding provider".to_string(),
                )
            })?;
            let groups = self.group_chunks(chunks, emb_map, threshold);
            context.add_step("grouping", json!({"num_groups": groups.len()}));
            groups
        } else {
            chunks.into_iter().map(|c| vec![c]).collect()
        };

        // Map phase: process each chunk/group
        let mut map_results = Vec::new();
        for group in chunk_groups {
            // Combine chunks in group (optimized: pre-allocate buffer)
            let mut combined_text = String::with_capacity(group.len() * 200);
            for (i, c) in group.iter().enumerate() {
                if i > 0 {
                    combined_text.push_str("\n\n");
                }
                combined_text.push_str(c.text.as_str());
            }

            let map_input = json!({
                self.text_key.clone(): combined_text,
            });

            let result = self.map_chain.run(&map_input).await?;
            map_results.push(result.output);
        }

        context.add_step("map", json!({"num_results": map_results.len()}));

        // Reduce phase: combine all results (optimized: pre-allocate buffer)
        let mut combined_results = String::with_capacity(map_results.len() * 150);
        for (i, r) in map_results.iter().enumerate() {
            if i > 0 {
                combined_results.push_str("\n\n");
            }
            writeln!(combined_results, "[Section {}]", i + 1).ok();
            combined_results.push_str(r.as_str().unwrap_or("[No output]"));
        }

        let reduce_input = json!({
            "summaries": combined_results,
        });

        let final_result = self.reduce_chain.run(&reduce_input).await?;

        // Merge contexts efficiently (move instead of clone)
        context.append_steps(final_result.context);

        Ok(ChainResult {
            output: final_result.output,
            context,
        })
    }
}
