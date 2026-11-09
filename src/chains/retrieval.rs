use crate::chains::{Chain, ChainContext, ChainError, ChainResult};
use crate::llm::embedding::{EmbeddingProvider, EmbeddingRequest};
use async_trait::async_trait;
use dripstone::core::{Metric, SearchQuery};
use dripstone::traits::Backend;
use serde_json::{Value, json};
use std::fmt::Write as _;
use std::sync::Arc;

/// A chain that performs retrieval-augmented generation (RAG)
///
/// This chain:
/// 1. Embeds the input query
/// 2. Searches a vector store for relevant documents
/// 3. Augments the prompt with retrieved context
/// 4. Passes to an LLM chain for final response
///
/// # Example
///
/// ```rust,no_run
/// use caliche::{AgentBuilder, ProviderType};
/// use caliche::chains::{Chain, RetrievalChain, LLMChain};
/// use caliche::prompts::PromptTemplate;
/// use caliche::OpenAIProvider;
/// use dripstone::MemoryBackend;
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
/// let vector_backend: Arc<dyn dripstone::Backend> = Arc::new(MemoryBackend::new());
///
/// let template = PromptTemplate::new(
///     "Answer the question based on this context:\n\n{context}\n\nQuestion: {query}\n\nAnswer:"
/// );
///
/// let llm_chain = Arc::new(LLMChain::new(agent, template));
///
/// let retrieval_chain = RetrievalChain::new(
///     embedding_provider,
///     vector_backend,
///     llm_chain,
/// )
/// .with_top_k(5)
/// .with_min_similarity(0.7);
///
/// let result = retrieval_chain.run(&json!({
///     "query": "What is Rust's ownership model?"
/// })).await?;
/// # Ok(())
/// # }
/// ```
pub struct RetrievalChain {
    embedding_provider: Arc<dyn EmbeddingProvider>,
    vector_backend: Arc<dyn Backend>,
    llm_chain: Arc<dyn Chain>,
    top_k: usize,
    min_similarity: f32,
    namespace: Option<String>,
    context_key: String,
    query_key: String,
}

impl std::fmt::Debug for RetrievalChain {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RetrievalChain")
            .field("embedding_provider", &"Arc<dyn EmbeddingProvider>")
            .field("vector_backend", &"Arc<dyn Backend>")
            .field(
                "llm_chain",
                &format!("Arc<dyn Chain: {}>", self.llm_chain.name()),
            )
            .field("top_k", &self.top_k)
            .field("min_similarity", &self.min_similarity)
            .field("namespace", &self.namespace)
            .field("context_key", &self.context_key)
            .field("query_key", &self.query_key)
            .finish()
    }
}

impl RetrievalChain {
    /// Create a new retrieval chain
    pub fn new(
        embedding_provider: Arc<dyn EmbeddingProvider>,
        vector_backend: Arc<dyn Backend>,
        llm_chain: Arc<dyn Chain>,
    ) -> Self {
        Self {
            embedding_provider,
            vector_backend,
            llm_chain,
            top_k: 5,
            min_similarity: 0.0,
            namespace: None,
            context_key: "context".to_owned(),
            query_key: "query".to_owned(),
        }
    }

    /// Set the number of documents to retrieve
    pub fn with_top_k(mut self, top_k: usize) -> Self {
        self.top_k = top_k;
        self
    }

    /// Set minimum similarity threshold (0.0 to 1.0)
    pub fn with_min_similarity(mut self, min_similarity: f32) -> Self {
        self.min_similarity = min_similarity;
        self
    }

    /// Set the namespace to search in
    pub fn with_namespace(mut self, namespace: impl Into<String>) -> Self {
        self.namespace = Some(namespace.into());
        self
    }

    /// Set the key name for retrieved context in the prompt
    pub fn with_context_key(mut self, key: impl Into<String>) -> Self {
        self.context_key = key.into();
        self
    }

    /// Set the key name for the query in the input
    pub fn with_query_key(mut self, key: impl Into<String>) -> Self {
        self.query_key = key.into();
        self
    }

    /// Retrieve relevant documents for a query
    async fn retrieve(&self, query: &str) -> Result<Vec<String>, ChainError> {
        // Embed the query
        let request = EmbeddingRequest {
            texts: vec![query.to_string()],
            model: None,
        };

        let response = self
            .embedding_provider
            .embed(request)
            .await
            .map_err(|e| ChainError::ExecutionFailed(format!("Embedding failed: {}", e)))?;

        let query_vector = &response.embeddings[0];

        // Build search query
        let mut search_query = SearchQuery::new(query_vector.clone())
            .limit(self.top_k)
            .metric(Metric::Cosine);

        if let Some(ref ns) = self.namespace {
            search_query = search_query.namespace(ns);
        }

        // Search vector store
        let results = self
            .vector_backend
            .search(&search_query)
            .await
            .map_err(|e| ChainError::ExecutionFailed(format!("Vector search failed: {}", e)))?;

        // Filter by similarity threshold and extract text from chunk content
        let documents: Vec<String> = results
            .into_iter()
            .filter(|r| r.score >= self.min_similarity)
            .map(|r| r.chunk.content)
            .collect();

        Ok(documents)
    }
}

#[async_trait]
impl Chain for RetrievalChain {
    async fn run(&self, input: &Value) -> Result<ChainResult, ChainError> {
        let mut context = ChainContext::new();

        // Extract query from input
        let query = input
            .get(&self.query_key)
            .and_then(|v| v.as_str())
            .ok_or_else(|| {
                ChainError::InvalidInput(format!("Missing '{}' in input", self.query_key))
            })?;

        // Retrieve relevant documents
        let documents = self.retrieve(query).await?;
        context.add_step(
            "retrieval",
            json!({
                "query": query,
                "num_results": documents.len(),
            }),
        );

        // Format context
        let context_text = if documents.is_empty() {
            "No relevant context found.".to_string()
        } else {
            let mut buffer = String::with_capacity(documents.len() * 100);
            for (i, doc) in documents.iter().enumerate() {
                if i > 0 {
                    buffer.push_str("\n\n");
                }
                writeln!(buffer, "[Document {}]", i + 1).ok();
                buffer.push_str(doc);
            }
            buffer
        };

        // Augment input with retrieved context
        let augmented_input = if let Some(obj) = input.as_object() {
            let mut new_obj = obj.clone();
            new_obj.insert(self.context_key.clone(), json!(context_text));
            Value::Object(new_obj)
        } else {
            let mut new_obj = serde_json::Map::new();
            new_obj.insert(self.context_key.clone(), json!(context_text));
            new_obj.insert("input".to_owned(), input.clone());
            Value::Object(new_obj)
        };

        // Pass to LLM chain
        let result = self.llm_chain.run(&augmented_input).await?;

        // Merge contexts
        context.append_steps(result.context);

        Ok(ChainResult {
            output: result.output,
            context,
        })
    }
}
