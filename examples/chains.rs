//! Chains Example
//!
//! This example demonstrates:
//! - LLMChain: Simple prompt → LLM → output
//! - SequentialChain/ComposableChain: Multiple chains in sequence
//! - MapReduceChain: Parallel processing then reduce (JSON and simple APIs)
//! - RetrievalChain: RAG with vector similarity search
//! - HybridSearchChain: Vector + keyword search
//! - DocumentMapReduceChain: Large document processing with semantic chunking
//!
//! Best practices shown:
//! - Composing chains for complex workflows
//! - Parallel processing for efficiency
//! - RAG patterns for knowledge retrieval
//!
//! Run with:
//! ```bash
//! cargo run --example chains --features chain-features
//! ```

#![expect(
    clippy::print_stdout,
    reason = "examples print their results for the reader"
)]

use anyhow::{Result, anyhow};
use caliche::prelude::ProviderType;
use caliche::{
    AgentBuilder,
    chains::{
        Chain, ChunkingStrategy, ComposableChain, DocumentMapReduceChain, HybridSearchChain,
        LLMChain, MapReduceChain, RetrievalChain,
    },
    output::{OutputParser, StructuredOutputParser},
    prompts::{Example, FewShotPromptTemplate, PromptTemplate},
};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::sync::Arc;
use tokio::sync::Mutex;

#[cfg(feature = "retrieval-chains")]
use caliche::llm::{OpenAIProvider, embedding::EmbeddingProvider};
#[cfg(feature = "retrieval-chains")]
use dripstone::{Chunk, MemoryBackend, traits::Backend};

#[derive(Debug, Serialize, Deserialize)]
struct AnalysisResult {
    category: String,
    sentiment: String,
    summary: String,
    recommendations: Vec<String>,
}

#[tokio::main]
async fn main() -> Result<()> {
    dotenvy::dotenv().ok();

    println!("=== Chains Examples ===\n");

    // Example 1: Basic LLMChain
    example_llm_chain().await?;

    // Example 2: Sequential Chain
    example_sequential_chain().await?;

    // Example 3: MapReduceChain with JSON API
    example_mapreduce_json().await?;

    // Example 4: MapReduceChain with Simple API
    example_mapreduce_simple().await?;

    // Example 5: Structured Output with Chains
    example_structured_output().await?;

    // Example 6: Few-shot Learning with Chains
    example_few_shot().await?;

    // Retrieval chains (if feature enabled)
    #[cfg(feature = "retrieval-chains")]
    {
        example_retrieval_chains().await?;
    }

    #[cfg(not(feature = "retrieval-chains"))]
    {
        println!("\n=== Retrieval Chains (skipped) ===");
        println!("Enable 'retrieval-chains' feature to see RAG examples");
    }

    Ok(())
}

async fn example_llm_chain() -> Result<()> {
    println!("Example 1: Basic LLMChain\n");

    let agent = AgentBuilder::new()
        .provider(ProviderType::Bedrock)
        .model("us.anthropic.claude-haiku-4-5-20251001-v1:0".to_string())
        .build()
        .await?;

    let agent = Arc::new(Mutex::new(agent));

    let template = PromptTemplate::new("Summarize this text in one sentence: {text}");
    let chain = LLMChain::new(agent, template);

    let result = chain
        .run(&json!({
            "text": "Rust is a systems programming language focused on safety and performance."
        }))
        .await?;

    println!("Input: Rust is a systems programming language...");
    println!(
        "Output: {}\n",
        result
            .output
            .get("output")
            .and_then(|v| v.as_str())
            .unwrap_or("N/A")
    );

    Ok(())
}

async fn example_sequential_chain() -> Result<()> {
    println!("Example 2: Sequential Chain\n");

    let agent = AgentBuilder::new()
        .provider(ProviderType::Bedrock)
        .model("us.anthropic.claude-haiku-4-5-20251001-v1:0".to_string())
        .build()
        .await?;

    let agent = Arc::new(Mutex::new(agent));

    // Chain 1: Extract key points
    let extract_template = PromptTemplate::new(
        "Read this text and extract the key points:\n\n{input}\n\n\
         List 3-5 key points as bullet points.",
    );

    // Chain 2: Summarize the key points
    let summarize_template = PromptTemplate::new(
        "Given these key points:\n\n{output}\n\n\
         Write a one-paragraph summary that captures the essence.",
    );

    let extract_chain = LLMChain::new(agent.clone(), extract_template);
    let summarize_chain = LLMChain::new(agent.clone(), summarize_template);

    // Build and execute sequential chain using >> operator
    let seq_chain = ComposableChain::new(extract_chain) >> ComposableChain::new(summarize_chain);

    let sample_text = "Agent-based systems are computational systems composed of multiple \
        interacting autonomous agents. Each agent perceives its environment and acts upon it \
        to achieve specific goals.";

    let result = seq_chain.run(&json!({"input": sample_text})).await?;

    println!("Input: Agent-based systems...");
    let summary = if let Some(s) = result.output.as_str() {
        s.to_string()
    } else if let Some(obj) = result.output.as_object() {
        obj.get("output")
            .and_then(|v| v.as_str())
            .unwrap_or("[No output received]")
            .to_string()
    } else {
        "[No output received]".to_string()
    };
    println!("Final Summary: {}\n", summary);

    Ok(())
}

async fn example_mapreduce_json() -> Result<()> {
    println!("Example 3: MapReduceChain (JSON API)\n");

    let agent = AgentBuilder::new()
        .provider(ProviderType::Bedrock)
        .model("us.anthropic.claude-haiku-4-5-20251001-v1:0".to_string())
        .build()
        .await?;

    let agent = Arc::new(Mutex::new(agent));

    // Map: Summarize each article
    let map_chain = Arc::new(LLMChain::new(
        agent.clone(),
        PromptTemplate::new("Summarize this article in 2 sentences:\n\n{input}"),
    ));

    // Reduce: Combine all summaries
    let reduce_chain = Arc::new(LLMChain::new(
        agent.clone(),
        PromptTemplate::new("Combine these summaries into a coherent overview:\n\n{output}"),
    ));

    let chain = MapReduceChain::new(map_chain, reduce_chain);

    // Sample articles - wrap in objects for PromptTemplate
    let articles = vec![
        json!({"input": "Rust 1.75 has been released with improved error messages and faster compilation times."}),
        json!({"input": "WebAssembly continues to grow in popularity for web applications."}),
        json!({"input": "Artificial Intelligence tools are transforming software development."}),
    ];

    println!("Processing {} articles...", articles.len());
    let input = json!(articles);

    let result = chain.run(&input).await?;

    println!("\nFinal Summary:");
    let output_text = result
        .output
        .get("output")
        .and_then(|v| v.as_str())
        .unwrap_or("N/A");
    println!("  {}\n", output_text);

    Ok(())
}

async fn example_mapreduce_simple() -> Result<()> {
    println!("Example 4: MapReduceChain (Simple API)\n");

    let agent = AgentBuilder::new()
        .provider(ProviderType::Bedrock)
        .model("us.anthropic.claude-haiku-4-5-20251001-v1:0".to_string())
        .build()
        .await?;

    let agent = Arc::new(Mutex::new(agent));

    let map_chain = Arc::new(LLMChain::new(
        agent.clone(),
        PromptTemplate::new("Summarize this article in 2 sentences:\n\n{input}"),
    ));

    let reduce_chain = Arc::new(LLMChain::new(
        agent.clone(),
        PromptTemplate::new("Combine these summaries into a coherent overview:\n\n{output}"),
    ));

    let chain = MapReduceChain::new(map_chain, reduce_chain);

    // Simple string array - no JSON wrapping needed!
    let articles = vec![
        "Rust 1.75 has been released with improved error messages.",
        "WebAssembly continues to grow in popularity.",
        "AI tools are transforming software development.",
    ];

    println!("Processing {} articles with simple API...", articles.len());

    // Use the run_with_strings method
    let result = chain.run_with_strings(&articles, "input").await?;

    println!("\nFinal Summary:");
    let summary = result
        .output
        .get("output")
        .and_then(|v| v.as_str())
        .unwrap_or("N/A");
    println!("  {}\n", summary);

    Ok(())
}

async fn example_structured_output() -> Result<()> {
    println!("Example 5: Structured Output with Chains\n");

    let agent = AgentBuilder::new()
        .provider(ProviderType::Bedrock)
        .model("us.anthropic.claude-haiku-4-5-20251001-v1:0".to_string())
        .build()
        .await?;

    let agent = Arc::new(Mutex::new(agent));

    let template = PromptTemplate::new(
        "Analyze the following text and provide a structured analysis.\n\n\
         Text: {text}\n\n\
         You must respond with ONLY a valid JSON object (no other text) with exactly these fields:\n\
         - \"category\": string (e.g., \"technology\", \"programming\", \"ai\")\n\
         - \"sentiment\": string (\"positive\", \"neutral\", or \"negative\")\n\
         - \"summary\": string (one sentence summary)\n\
         - \"recommendations\": array of strings (2-3 recommendations)\n\n\
         JSON response:",
    );

    let chain = LLMChain::new(agent, template);

    let result = chain
        .run(&json!({
            "text": "Rust provides memory safety without garbage collection through its ownership system."
        }))
        .await?;

    let response = result
        .output
        .get("output")
        .and_then(|v| v.as_str())
        .unwrap_or("");

    // Parse structured output
    let parser = StructuredOutputParser::<AnalysisResult>::new();
    match parser.parse(response).await {
        Ok(analysis) => {
            println!("Category: {}", analysis.category);
            println!("Sentiment: {}", analysis.sentiment);
            println!("Summary: {}", analysis.summary);
            println!("Recommendations: {:?}\n", analysis.recommendations);
        }
        Err(e) => {
            println!("Failed to parse structured output: {}", e);
            println!(
                "Raw response: {}\n",
                caliche_contracts::text::truncate_utf8(&response, 200)
            );
        }
    }

    Ok(())
}

async fn example_few_shot() -> Result<()> {
    println!("Example 6: Few-shot Learning with Chains\n");

    let agent = AgentBuilder::new()
        .provider(ProviderType::Bedrock)
        .model("us.anthropic.claude-haiku-4-5-20251001-v1:0".to_string())
        .build()
        .await?;

    let agent = Arc::new(Mutex::new(agent));

    // Create few-shot template with examples
    let examples = vec![
        Example::new("How do I write safe Rust code?", "technical-programming"),
        Example::new("What are the latest AI trends?", "research-ai"),
        Example::new("Explain machine learning basics", "educational-ai"),
    ];

    let few_shot = FewShotPromptTemplate::new(
        examples,
        "Classify the following query into one of these categories: technical-programming, research-ai, educational-ai, general",
        "Query: {query}\nCategory:",
    );

    let prompt = few_shot.format(&json!({ "query": "How can I optimize Rust performance?" }))?;

    let mut agent_guard = agent.lock().await;
    let response = agent_guard.chat(&prompt).await?;
    drop(agent_guard);

    println!("Query: How can I optimize Rust performance?");
    println!("Classification: {}\n", response.text().trim());

    Ok(())
}

#[cfg(feature = "retrieval-chains")]
async fn example_retrieval_chains() -> Result<()> {
    println!("\n=== Retrieval Chains Examples ===\n");

    let embedding_key = std::env::var("OPENAI_API_KEY")
        .map_err(|_| anyhow!("OPENAI_API_KEY required for retrieval chains"))?;
    let embedding_provider: Arc<dyn EmbeddingProvider> = Arc::new(OpenAIProvider::try_new(
        embedding_key.clone(),
        "text-embedding-3-small".to_string(),
    )?);

    let vector_backend: Arc<dyn Backend> = Arc::new(MemoryBackend::new());

    // Index sample documents
    index_sample_documents(embedding_provider.clone(), vector_backend.clone()).await?;

    // Create agent for LLM operations
    let agent = AgentBuilder::new()
        .provider(ProviderType::Bedrock)
        .model("us.anthropic.claude-haiku-4-5-20251001-v1:0".to_string())
        .build()
        .await?;
    let agent = Arc::new(Mutex::new(agent));

    // Example 7: Basic Retrieval Chain
    println!("Example 7: Basic Retrieval Chain (RAG)\n");

    let template = PromptTemplate::new(
        "Answer this question based on the provided context. \
         Be concise and only use information from the context.\n\n\
         Context:\n{context}\n\n\
         Question: {query}\n\n\
         Answer:",
    );

    let llm_chain = Arc::new(LLMChain::new(agent.clone(), template));

    let retrieval_chain = RetrievalChain::new(
        embedding_provider.clone(),
        vector_backend.clone(),
        llm_chain,
    )
    .with_top_k(3)
    .with_min_similarity(0.5)
    .with_namespace("docs");

    let query = "What are the safety features of Rust?";
    println!("Query: {}\n", query);

    let result = retrieval_chain.run(&json!({ "query": query })).await?;

    println!("Answer:");
    let answer = result
        .output
        .get("output")
        .and_then(|v| v.as_str())
        .unwrap_or("[No output]");
    println!("{}\n", answer);

    // Example 8: Hybrid Search Chain
    println!("Example 8: Hybrid Search Chain (Vector + Keyword)\n");

    let template = PromptTemplate::new(
        "Based on these documents, provide a comprehensive answer:\n\n\
         {context}\n\n\
         Question: {query}\n\n\
         Answer:",
    );

    let llm_chain = Arc::new(LLMChain::new(agent.clone(), template));

    let hybrid_chain = HybridSearchChain::new(
        embedding_provider.clone(),
        vector_backend.clone(),
        llm_chain,
    )
    .with_vector_weight(0.6)
    .with_keyword_weight(0.4)
    .with_top_k(5)
    .with_namespace("docs");

    let query = "memory management and borrowing";
    println!("Query: {}", query);

    let result = hybrid_chain.run(&json!({ "query": query })).await?;

    let answer = result
        .output
        .get("output")
        .and_then(|v| v.as_str())
        .unwrap_or("[No output]");
    println!("Answer: {}\n", answer);

    // Example 9: Document Map-Reduce Chain
    println!("Example 9: Document Map-Reduce Chain\n");

    let large_document = create_large_sample_document();

    let map_template = PromptTemplate::new(
        "Extract the key points from this text section:\n\n{text}\n\n\
         Key points (as bullet points):",
    );
    let map_chain = Arc::new(LLMChain::new(agent.clone(), map_template));

    let reduce_template = PromptTemplate::new(
        "Combine these extracted key points into a cohesive summary:\n\n\
         {summaries}\n\n\
         Final comprehensive summary:",
    );
    let reduce_chain = Arc::new(LLMChain::new(agent.clone(), reduce_template));

    let doc_chain = DocumentMapReduceChain::new(map_chain, reduce_chain)
        .with_chunking(ChunkingStrategy::Sentence { max_size: 500 })
        .with_embedding_provider(embedding_provider)
        .with_semantic_grouping(0.8);

    println!(
        "Processing large document ({} chars)...",
        large_document.len()
    );

    let result = doc_chain
        .run(&json!({ "document": large_document }))
        .await?;

    let summary = result
        .output
        .get("output")
        .and_then(|v| v.as_str())
        .unwrap_or("[No output]");
    println!("Final Summary: {}\n", summary);

    Ok(())
}

#[cfg(feature = "retrieval-chains")]
async fn index_sample_documents(
    embedding_provider: Arc<dyn EmbeddingProvider>,
    vector_backend: Arc<dyn Backend>,
) -> Result<()> {
    let documents = vec![
        (
            "rust_safety",
            "Rust provides memory safety without garbage collection through its ownership system. Each value has a single owner, and when the owner goes out of scope, the value is dropped.",
        ),
        (
            "rust_borrowing",
            "Borrowing in Rust allows references to values without taking ownership. You can have either multiple immutable references or one mutable reference at a time.",
        ),
        (
            "rust_performance",
            "Rust offers performance comparable to C and C++ because it compiles to native code and has zero-cost abstractions.",
        ),
    ];

    println!("Indexing {} documents...", documents.len());

    for (idx, (doc_id, text)) in documents.iter().enumerate() {
        let request = caliche::llm::embedding::EmbeddingRequest {
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
                "source": "example",
            }),
            Some("docs".to_string()),
        );

        vector_backend.upsert(&[chunk]).await?;
    }

    println!("Indexed {} documents\n", documents.len());

    Ok(())
}

#[cfg(feature = "retrieval-chains")]
fn create_large_sample_document() -> String {
    vec![
        "Rust is a modern systems programming language that emphasizes safety, concurrency, and performance.",
        "The ownership system is Rust's most distinctive feature. Every value in Rust has a variable that owns it.",
        "Borrowing extends the ownership model by allowing references to values.",
        "The type system in Rust is designed to catch errors early.",
        "Rust's zero-cost abstractions mean that high-level features don't come with runtime overhead.",
    ]
    .join("\n\n")
}
