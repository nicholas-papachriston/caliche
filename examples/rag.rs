//! Retrieval Augmented Generation (RAG) Example
//!
//! This example demonstrates:
//! - Basic RAG with vector similarity search
//! - Cross-encoder reranking for improved relevance
//! - Hybrid search (dense vector + BM25 keyword)
//! - Knowledge graph RAG for relationship queries
//! - Metadata filtering and advanced query options
//!
//! Best practices shown:
//! - When to use reranking (quality over speed)
//! - Tuning search weights for your data
//! - Combining semantic and keyword matching
//! - Graph-enhanced retrieval for relationship queries
//!
//! Run with:
//! ```bash
//! cargo run --example rag --features tools-rag,hybrid-search,rag-reranking
//! ```

#![expect(
    clippy::print_stdout,
    reason = "examples print their results for the reader"
)]

use anyhow::{Context, Result};
use caliche::AgentBuilder;
use caliche::prelude::ProviderType;
use caliche::rag::{GraphSearch, RagService};
use caliche::vecstore::VectorBackendConfig;
use std::sync::Arc;

#[cfg(feature = "tools-rag")]
use caliche::llm::{OpenAIProvider, Provider};

#[cfg(feature = "tools-rag")]
use dripstone::graph::{GraphBackend, memory::InMemoryGraphBackend};

#[tokio::main]
async fn main() -> Result<()> {
    dotenvy::dotenv().ok();

    println!("=== RAG Examples ===\n");

    // Example 1: Basic RAG with vector similarity
    Box::pin(example_basic_rag()).await?;

    // Example 2: RAG with reranking
    Box::pin(example_rag_reranking()).await?;

    // Example 3: Hybrid search
    Box::pin(example_hybrid_search()).await?;

    // Example 4: Graph RAG
    #[cfg(feature = "tools-rag")]
    {
        example_graph_rag().await?;
    }

    #[cfg(not(feature = "tools-rag"))]
    {
        println!("\n=== Graph RAG (skipped) ===");
        println!("Enable 'tools-rag' feature to see graph RAG examples");
    }

    Ok(())
}

async fn example_basic_rag() -> Result<()> {
    println!("Example 1: Basic RAG with Vector Similarity\n");

    let mut agent = AgentBuilder::new()
        .provider(ProviderType::OpenAI)
        .enable_tool("semantic_search")
        .build()
        .await?;

    // Index documents
    let documents = vec![
        (
            "doc1",
            "Rust is a systems programming language focused on memory safety without garbage collection.",
        ),
        (
            "doc2",
            "Python is a high-level language known for simplicity and readability.",
        ),
        (
            "doc3",
            "Rust uses ownership and borrowing to prevent data races and memory errors.",
        ),
    ];

    for (doc_id, content) in &documents {
        agent
            .chat(&format!(
                "Index this document with ID {}: {}",
                doc_id, content
            ))
            .await?;
    }

    let response = agent
        .chat("What are Rust's memory safety features?")
        .await?;

    println!("Query: What are Rust's memory safety features?");
    println!("Response: {}\n", response.text());

    Ok(())
}

async fn example_rag_reranking() -> Result<()> {
    println!("Example 2: RAG with Reranking\n");

    let mut agent = AgentBuilder::new()
        .provider(ProviderType::OpenAI)
        .enable_tool("semantic_search")
        .build()
        .await?;

    // Index multiple documents
    let documents = vec![
        (
            "doc1",
            "Rust is a systems programming language focused on memory safety without garbage collection.",
        ),
        (
            "doc2",
            "Python is a high-level language known for simplicity and readability.",
        ),
        (
            "doc3",
            "Rust uses ownership and borrowing to prevent data races and memory errors.",
        ),
        (
            "doc4",
            "Python's dynamic typing makes it easy to write but can lead to runtime errors.",
        ),
    ];

    for (doc_id, content) in &documents {
        agent
            .chat(&format!(
                "Index this document with ID {}: {}",
                doc_id, content
            ))
            .await?;
    }

    // Initial retrieval without reranking
    let response = agent
        .chat("What are Rust's memory safety features?")
        .await?;

    println!("Initial Retrieval (no reranking):");
    println!("{}\n", response.text());

    // With reranking enabled
    let response = agent
        .chat("Use rag tool with advanced_query: query='Rust memory safety', rerank=true, top_k=5")
        .await?;

    println!("With Reranking:");
    println!("{}\n", response.text());

    // Hybrid search + reranking
    let response = agent
        .chat("Use rag tool with advanced_query: query='programming language safety', search_mode='hybrid', rerank=true, top_k=10")
        .await?;

    println!("Hybrid Search + Reranking:");
    println!("{}\n", response.text());

    Ok(())
}

async fn example_hybrid_search() -> Result<()> {
    println!("Example 3: Hybrid Search (Vector + Keyword)\n");

    let mut agent = AgentBuilder::new()
        .provider(ProviderType::OpenAI)
        .enable_tool("semantic_search")
        .build()
        .await?;

    // Index some documents first
    agent
        .chat("Index this document with ID doc1: Rust is a systems programming language focused on safety and performance. It uses ownership and borrowing to prevent memory errors.")
        .await?;

    agent
        .chat("Index this document with ID doc2: Python is a high-level programming language known for its simplicity. It's widely used for web development and data science.")
        .await?;

    // Pure semantic search (default)
    let response = agent
        .chat("What programming language focuses on memory safety?")
        .await?;

    println!("Semantic Search Result:");
    println!("{}\n", response.text());

    // Hybrid search (dense + sparse)
    let response = agent
        .chat("Use rag tool with advanced_query: query='programming language', search_mode='hybrid', dense_weight=0.6, sparse_weight=0.4")
        .await?;

    println!("Hybrid Search Result:");
    println!("{}\n", response.text());

    // Keyword-heavy query (benefits from hybrid)
    let response = agent
        .chat("Use rag tool with advanced_query: query='Rust ownership', search_mode='hybrid', dense_weight=0.4, sparse_weight=0.6")
        .await?;

    println!("Keyword-Heavy Hybrid Search:");
    println!("{}\n", response.text());

    Ok(())
}

#[cfg(feature = "tools-rag")]
async fn example_graph_rag() -> Result<()> {
    println!("Example 4: Graph RAG (Knowledge Graph)\n");

    let api_key =
        std::env::var("OPENAI_API_KEY").context("OPENAI_API_KEY required for graph RAG")?;

    let embedding_provider = Arc::new(OpenAIProvider::try_new(
        api_key.clone(),
        "text-embedding-3-small".to_string(),
    )?);

    let llm_provider: Arc<dyn Provider> =
        Arc::new(OpenAIProvider::try_new(api_key, "gpt-4o-mini".to_string())?);

    // Create vector backend
    let vector_config = VectorBackendConfig::Memory {
        ttl_secs: 86400,
        max_entries: Some(100_000),
    };
    let _vector_backend = vector_config.build_backend().await?;

    // Create graph backend
    let graph_backend: Arc<dyn GraphBackend> = Arc::new(InMemoryGraphBackend::new());

    // Create RAG service with graph support (uses internal MemoryBackend)
    let rag_service = RagService::with_graph_backend(
        embedding_provider.clone(),
        graph_backend.clone(),
        Some(llm_provider.clone()),
    );

    println!("Indexing document with entity extraction...\n");

    // Sample document about a company and its employees
    let document = r#"
        Acme Corporation is a technology company founded in 2010 by Alice Johnson and Bob Smith.
        Alice Johnson serves as the CEO of Acme Corporation. She has a background in computer science
        and previously worked at TechCorp. Bob Smith is the CTO and co-founder. He specializes in
        machine learning and artificial intelligence.
        
        Acme Corporation is located in San Francisco, California. The company focuses on developing
        AI-powered solutions for enterprise customers. Their main product is an intelligent document
        processing system.
        
        The company has partnerships with several major tech companies including Microsoft and Google.
        They also work closely with Stanford University on research projects.
    "#;

    // Index the document (this will also extract entities and build the graph)
    let (chunk_count, _usage) = rag_service.index_document(document, "acme_corp").await?;
    println!("Indexed {} chunks\n", chunk_count);

    // Check graph statistics
    let node_count = graph_backend.count_nodes().await?;
    let edge_count = graph_backend.count_edges().await?;
    println!("Graph Statistics:");
    println!("   Nodes: {}", node_count);
    println!("   Edges: {}\n", edge_count);

    // Perform graph search
    println!("Performing graph-enhanced search...\n");

    let graph_search = GraphSearch::new(embedding_provider.clone(), graph_backend.clone());

    let query = "Who works at Acme Corporation?";
    println!("Query: {}\n", query);

    let result = graph_search.search(query, 5, 1).await?;

    println!("Search Results:");
    println!("   Entry nodes found: {}", result.entry_nodes.len());
    println!("   Expanded nodes: {}", result.expanded_nodes.len());
    println!("   Relationships found: {}\n", result.expanded_edges.len());

    println!("Graph Context:");
    println!("   {}\n", result.context);

    Ok(())
}
