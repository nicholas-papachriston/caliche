#![cfg(feature = "retrieval-chains")]

use anyhow::Result;
use async_trait::async_trait;
use caliche::{
    chains::{
        Chain, ChainContext, ChainError, ChainResult, ChunkingStrategy, DocumentMapReduceChain,
        HybridSearchChain, RetrievalChain,
    },
    llm::embedding::{EmbeddingProvider, EmbeddingRequest, EmbeddingResponse},
    messages::TokenUsage,
};
use dripstone::{Chunk, MemoryBackend, traits::Backend};
use serde_json::{Value, json};
use std::sync::Arc;

/// Mock embedding provider for testing
#[derive(Debug)]
struct MockEmbeddingProvider {
    dimension: usize,
}

impl MockEmbeddingProvider {
    fn new(dimension: usize) -> Self {
        Self { dimension }
    }

    /// Generate a deterministic embedding based on text hash
    fn generate_embedding(&self, text: &str) -> Vec<f32> {
        let mut embedding = vec![0.0; self.dimension];
        let bytes = text.as_bytes();

        for (i, &byte) in bytes.iter().enumerate() {
            embedding[i % self.dimension] += (byte as f32) / 255.0;
        }

        // Normalize
        let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for val in &mut embedding {
                *val /= norm;
            }
        }

        embedding
    }
}

#[async_trait]
impl EmbeddingProvider for MockEmbeddingProvider {
    async fn embed(&self, request: EmbeddingRequest) -> Result<EmbeddingResponse> {
        let embeddings: Vec<Vec<f32>> = request
            .texts
            .iter()
            .map(|text| self.generate_embedding(text))
            .collect();

        let total_chars = request.texts.iter().map(|t| t.len()).sum::<usize>() as u32;
        let embedding_tokens = (request.texts.len() * 10) as u32;

        Ok(EmbeddingResponse {
            embeddings,
            usage: Some(TokenUsage {
                prompt_tokens: Some(total_chars),
                completion_tokens: None,
                embedding_tokens: Some(embedding_tokens),
                total_tokens: Some(embedding_tokens),
                context_window: None,
                cache_creation_tokens: None,
                cache_read_tokens: None,
                last_prompt_tokens: None,
            }),
            model: "mock-embedding-model".to_string(),
        })
    }

    fn name(&self) -> &str {
        "MockEmbeddingProvider"
    }
}

/// Mock chain for testing that returns deterministic output
#[derive(Debug)]
struct MockLLMChain {
    response: String,
}

impl MockLLMChain {
    fn new(response: impl Into<String>) -> Self {
        Self {
            response: response.into(),
        }
    }
}

#[async_trait]
impl Chain for MockLLMChain {
    async fn run(&self, _input: &Value) -> Result<ChainResult, ChainError> {
        Ok(ChainResult {
            output: json!(self.response.clone()),
            context: ChainContext::new(),
        })
    }
}

/// Helper to create and populate a test vector store
async fn create_test_vector_backend(
    embedding_provider: Arc<dyn EmbeddingProvider>,
) -> Result<Arc<dyn Backend>> {
    let vector_backend: Arc<dyn Backend> = Arc::new(MemoryBackend::new());

    let test_docs = vec![
        (
            "doc1",
            "Rust provides memory safety without garbage collection",
        ),
        (
            "doc2",
            "Borrowing allows references without ownership transfer",
        ),
        (
            "doc3",
            "Zero-cost abstractions in Rust enable high performance",
        ),
        (
            "doc4",
            "The ownership model prevents memory leaks at compile time",
        ),
        ("doc5", "Rust's type system catches errors before runtime"),
    ];

    for (idx, (doc_id, text)) in test_docs.iter().enumerate() {
        let request = EmbeddingRequest {
            texts: vec![text.to_string()],
            model: None,
        };

        let response = embedding_provider.embed(request).await?;
        let embedding = &response.embeddings[0];

        let chunk = Chunk::new(
            doc_id.to_string(),
            idx,
            text.to_string(),
            embedding.clone(),
            0,
            text.len(),
            1,
            json!({
                "text": text,
            }),
            Some("test".to_string()),
        );

        vector_backend.upsert(&[chunk]).await?;
    }

    Ok(vector_backend)
}

#[tokio::test]
async fn test_retrieval_chain_basic() -> Result<()> {
    let embedding_provider: Arc<dyn EmbeddingProvider> = Arc::new(MockEmbeddingProvider::new(128));
    let vector_backend = create_test_vector_backend(embedding_provider.clone()).await?;

    let mock_llm = Arc::new(MockLLMChain::new(
        "Rust ensures memory safety through ownership",
    ));

    let retrieval_chain = RetrievalChain::new(embedding_provider, vector_backend, mock_llm)
        .with_namespace("test")
        .with_top_k(3);

    let result = retrieval_chain
        .run(&json!({
            "query": "How does Rust handle memory safety?"
        }))
        .await?;

    assert_eq!(
        result.output.as_str().unwrap(),
        "Rust ensures memory safety through ownership"
    );

    // Check that retrieval step was recorded
    let retrieval_step = result
        .context
        .intermediate_steps
        .iter()
        .find(|(name, _)| name == "retrieval");
    assert!(retrieval_step.is_some());

    Ok(())
}

#[tokio::test]
async fn test_retrieval_chain_with_similarity_threshold() -> Result<()> {
    let embedding_provider: Arc<dyn EmbeddingProvider> = Arc::new(MockEmbeddingProvider::new(128));
    let vector_backend = create_test_vector_backend(embedding_provider.clone()).await?;

    let mock_llm = Arc::new(MockLLMChain::new("Response based on filtered results"));

    let retrieval_chain = RetrievalChain::new(embedding_provider, vector_backend, mock_llm)
        .with_namespace("test")
        .with_top_k(5)
        .with_min_similarity(0.9); // High threshold

    let result = retrieval_chain
        .run(&json!({
            "query": "completely unrelated query about cooking pasta"
        }))
        .await?;

    // Should still get a response, but with fewer or no documents
    assert!(result.output.as_str().is_some());

    Ok(())
}

#[tokio::test]
async fn test_retrieval_chain_custom_keys() -> Result<()> {
    let embedding_provider: Arc<dyn EmbeddingProvider> = Arc::new(MockEmbeddingProvider::new(128));
    let vector_backend = create_test_vector_backend(embedding_provider.clone()).await?;

    let mock_llm = Arc::new(MockLLMChain::new("Custom response"));

    let retrieval_chain = RetrievalChain::new(embedding_provider, vector_backend, mock_llm)
        .with_namespace("test")
        .with_query_key("question")
        .with_context_key("docs");

    let result = retrieval_chain
        .run(&json!({
            "question": "What is Rust?"
        }))
        .await?;

    assert!(result.output.as_str().is_some());

    Ok(())
}

#[tokio::test]
async fn test_retrieval_chain_missing_query() -> Result<()> {
    let embedding_provider: Arc<dyn EmbeddingProvider> = Arc::new(MockEmbeddingProvider::new(128));
    let vector_backend = create_test_vector_backend(embedding_provider.clone()).await?;

    let mock_llm = Arc::new(MockLLMChain::new("Should not reach here"));

    let retrieval_chain =
        RetrievalChain::new(embedding_provider, vector_backend, mock_llm).with_namespace("test");

    let result = retrieval_chain
        .run(&json!({
            "wrong_key": "some value"
        }))
        .await;

    assert!(result.is_err());
    match result {
        Err(ChainError::InvalidInput(msg)) => {
            assert!(msg.contains("query"));
        }
        _ => panic!("Expected InvalidInput error"),
    }

    Ok(())
}

#[tokio::test]
async fn test_hybrid_search_chain_basic() -> Result<()> {
    let embedding_provider: Arc<dyn EmbeddingProvider> = Arc::new(MockEmbeddingProvider::new(128));
    let vector_backend = create_test_vector_backend(embedding_provider.clone()).await?;

    let mock_llm = Arc::new(MockLLMChain::new("Hybrid search result"));

    let hybrid_chain = HybridSearchChain::new(embedding_provider, vector_backend, mock_llm)
        .with_namespace("test")
        .with_top_k(5);

    let result = hybrid_chain
        .run(&json!({
            "query": "memory safety ownership"
        }))
        .await?;

    assert_eq!(result.output.as_str().unwrap(), "Hybrid search result");

    // Check that hybrid search recorded the selected search and fusion modes.
    let has_hybrid_step = result
        .context
        .intermediate_steps
        .iter()
        .any(|(name, _)| name == "hybrid_search");
    let has_fusion_mode = result
        .context
        .intermediate_steps
        .iter()
        .any(|(name, data)| name == "hybrid_search" && data.get("fusion_mode").is_some());

    assert!(has_hybrid_step);
    assert!(has_fusion_mode);

    Ok(())
}

#[tokio::test]
async fn test_hybrid_search_chain_weighted_fusion() -> Result<()> {
    let embedding_provider: Arc<dyn EmbeddingProvider> = Arc::new(MockEmbeddingProvider::new(128));
    let vector_backend = create_test_vector_backend(embedding_provider.clone()).await?;

    let mock_llm = Arc::new(MockLLMChain::new("Weighted result"));

    let hybrid_chain = HybridSearchChain::new(embedding_provider, vector_backend, mock_llm)
        .with_namespace("test")
        .with_vector_weight(0.7)
        .with_keyword_weight(0.3)
        .with_top_k(3);

    let result = hybrid_chain
        .run(&json!({
            "query": "rust performance"
        }))
        .await?;

    assert!(result.output.as_str().is_some());

    // Check fusion method in metadata
    let fusion_step = result
        .context
        .intermediate_steps
        .iter()
        .find(|(name, _)| name == "fusion");

    if let Some((_, value)) = fusion_step {
        assert_eq!(
            value.get("method").and_then(|v| v.as_str()),
            Some("weighted")
        );
    }

    Ok(())
}

#[tokio::test]
async fn test_hybrid_search_chain_rrf() -> Result<()> {
    let embedding_provider: Arc<dyn EmbeddingProvider> = Arc::new(MockEmbeddingProvider::new(128));
    let vector_backend = create_test_vector_backend(embedding_provider.clone()).await?;

    let mock_llm = Arc::new(MockLLMChain::new("RRF result"));

    let hybrid_chain = HybridSearchChain::new(embedding_provider, vector_backend, mock_llm)
        .with_namespace("test")
        .with_reciprocal_rank_fusion(true)
        .with_top_k(3);

    let result = hybrid_chain
        .run(&json!({
            "query": "ownership borrowing"
        }))
        .await?;

    assert!(result.output.as_str().is_some());

    // Check fusion method is RRF
    let fusion_step = result
        .context
        .intermediate_steps
        .iter()
        .find(|(name, _)| name == "fusion");

    if let Some((_, value)) = fusion_step {
        assert_eq!(value.get("method").and_then(|v| v.as_str()), Some("rrf"));
    }

    Ok(())
}

#[tokio::test]
async fn test_document_map_reduce_fixed_size_chunking() -> Result<()> {
    let mock_map = Arc::new(MockLLMChain::new("Key point from chunk"));
    let mock_reduce = Arc::new(MockLLMChain::new("Final summary"));

    let doc_chain = DocumentMapReduceChain::new(mock_map, mock_reduce).with_chunking(
        ChunkingStrategy::FixedSize {
            size: 50,
            overlap: 10,
        },
    );

    let large_doc = "a".repeat(200);
    let result = doc_chain
        .run(&json!({
            "document": large_doc
        }))
        .await?;

    assert_eq!(result.output.as_str().unwrap(), "Final summary");

    // Check chunking was performed
    let chunking_step = result
        .context
        .intermediate_steps
        .iter()
        .find(|(name, _)| name == "chunking");

    assert!(chunking_step.is_some());
    if let Some((_, value)) = chunking_step {
        let num_chunks = value.get("num_chunks").and_then(|v| v.as_u64()).unwrap();
        assert!(num_chunks > 1);
    }

    Ok(())
}

#[tokio::test]
async fn test_document_map_reduce_sentence_chunking() -> Result<()> {
    let mock_map = Arc::new(MockLLMChain::new("Sentence summary"));
    let mock_reduce = Arc::new(MockLLMChain::new("Combined summary"));

    let doc_chain = DocumentMapReduceChain::new(mock_map, mock_reduce)
        .with_chunking(ChunkingStrategy::Sentence { max_size: 100 });

    let doc_with_sentences = "First sentence here. Second sentence here. \
                              Third sentence here. Fourth sentence here. \
                              Fifth sentence here.";

    let result = doc_chain
        .run(&json!({
            "document": doc_with_sentences
        }))
        .await?;

    assert_eq!(result.output.as_str().unwrap(), "Combined summary");

    Ok(())
}

#[tokio::test]
async fn test_document_map_reduce_paragraph_chunking() -> Result<()> {
    let mock_map = Arc::new(MockLLMChain::new("Paragraph summary"));
    let mock_reduce = Arc::new(MockLLMChain::new("Final output"));

    let doc_chain = DocumentMapReduceChain::new(mock_map, mock_reduce)
        .with_chunking(ChunkingStrategy::Paragraph { max_size: 200 });

    let doc_with_paragraphs = "First paragraph with some text.\n\n\
                               Second paragraph with more text.\n\n\
                               Third paragraph with even more text.";

    let result = doc_chain
        .run(&json!({
            "document": doc_with_paragraphs
        }))
        .await?;

    assert_eq!(result.output.as_str().unwrap(), "Final output");

    Ok(())
}

#[tokio::test]
async fn test_document_map_reduce_with_embeddings() -> Result<()> {
    let embedding_provider: Arc<dyn EmbeddingProvider> = Arc::new(MockEmbeddingProvider::new(128));

    let mock_map = Arc::new(MockLLMChain::new("Embedded chunk summary"));
    let mock_reduce = Arc::new(MockLLMChain::new("Embedded final summary"));

    let doc_chain = DocumentMapReduceChain::new(mock_map, mock_reduce)
        .with_chunking(ChunkingStrategy::FixedSize {
            size: 100,
            overlap: 20,
        })
        .with_embedding_provider(embedding_provider)
        .with_semantic_grouping(0.8);

    let doc = "Rust is a systems programming language. \
               It provides memory safety without garbage collection. \
               The ownership model is central to Rust. \
               Borrowing allows temporary access to data.";

    let result = doc_chain
        .run(&json!({
            "document": doc
        }))
        .await?;

    assert_eq!(result.output.as_str().unwrap(), "Embedded final summary");

    // Check that embedding step was performed
    let embedding_step = result
        .context
        .intermediate_steps
        .iter()
        .find(|(name, _)| name == "embedding");
    assert!(embedding_step.is_some());

    // Check that grouping step was performed
    let grouping_step = result
        .context
        .intermediate_steps
        .iter()
        .find(|(name, _)| name == "grouping");
    assert!(grouping_step.is_some());

    Ok(())
}

#[tokio::test]
async fn test_document_map_reduce_custom_keys() -> Result<()> {
    let mock_map = Arc::new(MockLLMChain::new("Custom map result"));
    let mock_reduce = Arc::new(MockLLMChain::new("Custom reduce result"));

    let doc_chain = DocumentMapReduceChain::new(mock_map, mock_reduce)
        .with_document_key("content")
        .with_text_key("chunk");

    let result = doc_chain
        .run(&json!({
            "content": "Short document for testing custom keys"
        }))
        .await?;

    assert_eq!(result.output.as_str().unwrap(), "Custom reduce result");

    Ok(())
}

#[tokio::test]
async fn test_document_map_reduce_empty_document() -> Result<()> {
    let mock_map = Arc::new(MockLLMChain::new("Should not be called"));
    let mock_reduce = Arc::new(MockLLMChain::new("Empty result"));

    let doc_chain = DocumentMapReduceChain::new(mock_map, mock_reduce);

    let result = doc_chain
        .run(&json!({
            "document": ""
        }))
        .await?;

    // Should handle empty documents gracefully
    assert!(result.output.as_str().is_some());

    Ok(())
}

#[tokio::test]
async fn test_chunking_strategies() {
    let mock_map = Arc::new(MockLLMChain::new("chunk"));
    let mock_reduce = Arc::new(MockLLMChain::new("final"));

    // Test different chunking strategies
    let strategies = vec![
        ChunkingStrategy::FixedSize {
            size: 100,
            overlap: 10,
        },
        ChunkingStrategy::Sentence { max_size: 200 },
        ChunkingStrategy::Paragraph { max_size: 300 },
        ChunkingStrategy::Semantic {
            threshold: 0.7,
            max_size: 150,
        },
    ];

    let test_doc = "Sentence one. Sentence two.\n\nParagraph two here. More text.";

    for strategy in strategies {
        let chain = DocumentMapReduceChain::new(mock_map.clone(), mock_reduce.clone())
            .with_chunking(strategy);

        let result = chain.run(&json!({"document": test_doc})).await;
        assert!(result.is_ok(), "Chunking strategy should work");
    }
}

#[tokio::test]
async fn test_embedding_consistency() -> Result<()> {
    // Test that the same text produces the same embedding
    let embedding_provider: Arc<dyn EmbeddingProvider> = Arc::new(MockEmbeddingProvider::new(128));

    let text = "Rust memory safety".to_string();

    let request1 = EmbeddingRequest {
        texts: vec![text.clone()],
        model: None,
    };

    let request2 = EmbeddingRequest {
        texts: vec![text.clone()],
        model: None,
    };

    let response1 = embedding_provider.embed(request1).await?;
    let response2 = embedding_provider.embed(request2).await?;

    let emb1 = &response1.embeddings[0];
    let emb2 = &response2.embeddings[0];

    // Same text should produce identical embeddings
    let dot_product: f32 = emb1.iter().zip(emb2.iter()).map(|(a, b)| a * b).sum();

    // Cosine similarity should be 1.0 (or very close) for identical texts
    assert!(
        (dot_product - 1.0).abs() < 0.01,
        "Same text should produce identical embeddings"
    );

    // Test that embeddings are normalized
    let norm: f32 = emb1.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!((norm - 1.0).abs() < 0.01, "Embeddings should be normalized");

    Ok(())
}
