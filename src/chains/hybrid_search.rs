use crate::chains::{Chain, ChainContext, ChainError, ChainResult};
use crate::llm::embedding::EmbeddingProvider;
use crate::rag::{RagFusionMode, RagSearchMode, RagService};
use async_trait::async_trait;
use dripstone::traits::Backend;
use serde_json::{Value, json};
use std::fmt::Write as _;
use std::sync::Arc;

/// A scored document from search results
#[derive(Debug, Clone)]
pub struct ScoredDocument {
    pub id: String,
    pub text: String,
    pub score: f32,
    pub source: SearchSource,
}

/// Source of a search result
#[derive(Debug, Clone, PartialEq)]
pub enum SearchSource {
    Vector,
    Keyword,
    Hybrid,
}

/// A chain that combines semantic retrieval with Dripstone hybrid retrieval modes.
///
/// This implements a hybrid search approach that:
/// 1. Delegates retrieval to `RagService` backed by Dripstone
/// 2. Uses dense+sparse hybrid search or dense+full-text hybrid search
/// 3. Lets Dripstone own the fusion strategy instead of local score merging
/// 4. Passes top results to the next chain
///
/// # Example
///
/// ```rust,no_run
/// use caliche::{AgentBuilder, ProviderType};
/// use caliche::chains::{Chain, HybridSearchChain, LLMChain};
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
///     "Based on these documents:\n\n{context}\n\nAnswer: {query}"
/// );
///
/// let llm_chain = Arc::new(LLMChain::new(agent, template));
///
/// let hybrid_chain = HybridSearchChain::new(
///     embedding_provider,
///     vector_backend,
///     llm_chain,
/// )
/// .with_vector_weight(0.6)
/// .with_keyword_weight(0.4)
/// .with_top_k(10);
///
/// let result = hybrid_chain.run(&json!({
///     "query": "Rust memory safety"
/// })).await?;
/// # Ok(())
/// # }
/// ```
pub struct HybridSearchChain {
    embedding_provider: Arc<dyn EmbeddingProvider>,
    vector_backend: Arc<dyn Backend>,
    llm_chain: Arc<dyn Chain>,
    vector_weight: f32,
    keyword_weight: f32,
    top_k: usize,
    namespace: Option<String>,
    use_rrf: bool, // Reciprocal Rank Fusion
    context_key: String,
    query_key: String,
}

impl std::fmt::Debug for HybridSearchChain {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HybridSearchChain")
            .field("embedding_provider", &"Arc<dyn EmbeddingProvider>")
            .field("vector_backend", &"Arc<dyn Backend>")
            .field(
                "llm_chain",
                &format!("Arc<dyn Chain: {}>", self.llm_chain.name()),
            )
            .field("vector_weight", &self.vector_weight)
            .field("keyword_weight", &self.keyword_weight)
            .field("top_k", &self.top_k)
            .field("namespace", &self.namespace)
            .field("use_rrf", &self.use_rrf)
            .field("context_key", &self.context_key)
            .field("query_key", &self.query_key)
            .finish()
    }
}

impl HybridSearchChain {
    /// Create a new hybrid search chain
    pub fn new(
        embedding_provider: Arc<dyn EmbeddingProvider>,
        vector_backend: Arc<dyn Backend>,
        llm_chain: Arc<dyn Chain>,
    ) -> Self {
        Self {
            embedding_provider,
            vector_backend,
            llm_chain,
            vector_weight: 0.7,
            keyword_weight: 0.3,
            top_k: 10,
            namespace: None,
            use_rrf: false,
            context_key: "context".to_owned(),
            query_key: "query".to_owned(),
        }
    }

    /// Set weight for vector search results (0.0 to 1.0)
    pub fn with_vector_weight(mut self, weight: f32) -> Self {
        self.vector_weight = weight.clamp(0.0, 1.0);
        self
    }

    /// Set weight for keyword search results (0.0 to 1.0)
    pub fn with_keyword_weight(mut self, weight: f32) -> Self {
        self.keyword_weight = weight.clamp(0.0, 1.0);
        self
    }

    /// Set the number of final results to return
    pub fn with_top_k(mut self, top_k: usize) -> Self {
        self.top_k = top_k;
        self
    }

    /// Set the namespace to search in
    pub fn with_namespace(mut self, namespace: impl Into<String>) -> Self {
        self.namespace = Some(namespace.into());
        self
    }

    /// Use Reciprocal Rank Fusion instead of weighted scoring
    pub fn with_reciprocal_rank_fusion(mut self, use_rrf: bool) -> Self {
        self.use_rrf = use_rrf;
        self
    }

    /// Set the key name for retrieved context
    pub fn with_context_key(mut self, key: impl Into<String>) -> Self {
        self.context_key = key.into();
        self
    }

    /// Set the key name for the query
    pub fn with_query_key(mut self, key: impl Into<String>) -> Self {
        self.query_key = key.into();
        self
    }

    fn normalized_dense_weight(&self) -> f32 {
        let total = self.vector_weight + self.keyword_weight;
        if total > 0.0 {
            (self.vector_weight / total).clamp(0.0, 1.0)
        } else {
            0.7
        }
    }

    async fn search_documents(
        &self,
        query: &str,
    ) -> Result<
        (
            Vec<ScoredDocument>,
            RagSearchMode,
            RagFusionMode,
            Vec<String>,
        ),
        ChainError,
    > {
        let search_mode = if self.use_rrf {
            RagSearchMode::HybridDenseFullText
        } else {
            RagSearchMode::HybridDenseSparse
        };
        let fusion_mode = if self.use_rrf {
            RagFusionMode::ReciprocalRank
        } else {
            RagFusionMode::Weighted
        };

        let mut service = RagService::new_with_backend(
            Arc::clone(&self.embedding_provider),
            Arc::clone(&self.vector_backend),
            false,
            false,
            false,
        );
        if let Some(namespace) = &self.namespace {
            service = service.with_namespace(namespace.clone());
        }

        let (results, _, _, warnings, _) = service
            .enhanced_query(
                query,
                self.top_k,
                f32::MIN,
                search_mode,
                fusion_mode,
                Some(self.normalized_dense_weight()),
                Some(self.keyword_weight),
                false,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
            )
            .await
            .map_err(|e| ChainError::ExecutionFailed(format!("Hybrid retrieval failed: {e}")))?;

        let documents = results
            .into_iter()
            .map(|result| ScoredDocument {
                id: result.chunk.id,
                text: result.chunk.content,
                score: result.score,
                source: SearchSource::Hybrid,
            })
            .collect();

        Ok((documents, search_mode, fusion_mode, warnings))
    }
}

#[async_trait]
impl Chain for HybridSearchChain {
    async fn run(&self, input: &Value) -> Result<ChainResult, ChainError> {
        let mut context = ChainContext::new();

        // Extract query from input
        let query = input
            .get(&self.query_key)
            .and_then(|v| v.as_str())
            .ok_or_else(|| {
                ChainError::InvalidInput(format!("Missing '{}' in input", self.query_key))
            })?;

        let (fused_results, search_mode, fusion_mode, warnings) =
            self.search_documents(query).await?;

        context.add_step(
            "hybrid_search",
            json!({
                "query": query,
                "search_mode": search_mode.as_str(),
                "fusion_mode": fusion_mode.as_str(),
                "results": fused_results.len(),
            }),
        );

        if !warnings.is_empty() {
            context.add_step("hybrid_search_warnings", json!(warnings));
        }

        // Format context
        let context_text = if fused_results.is_empty() {
            "No relevant documents found.".to_string()
        } else {
            let mut buffer = String::with_capacity(fused_results.len() * 150);
            for (i, doc) in fused_results.iter().enumerate() {
                if i > 0 {
                    buffer.push_str("\n\n");
                }
                writeln!(
                    buffer,
                    "[Document {} - {} search - score: {:.3}]",
                    i + 1,
                    match doc.source {
                        SearchSource::Vector => "Vector",
                        SearchSource::Keyword => "Keyword",
                        SearchSource::Hybrid => "Hybrid",
                    },
                    doc.score
                )
                .ok();
                buffer.push_str(&doc.text);
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
