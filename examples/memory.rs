//! Memory Strategies Example
//!
//! This example demonstrates:
//! - BufferMemory: Sliding window of recent messages
//! - SummaryMemory: Automatic condensation of older messages
//! - EntityMemory: Semantic tracking of facts and entities
//! - VectorMemory: Semantic retrieval of relevant conversation turns
//! - Enhanced Context Management: Budget allocation, relevance scoring, compression, working memory, and knowledge store
//!
//! Best practices shown:
//! - Choosing the right memory strategy for your use case
//! - Token-aware memory management
//! - Advanced context management techniques
//!
//! Run with:
//! ```bash
//! cargo run --example memory --features vector-memory
//! ```

#![expect(
    clippy::print_stdout,
    reason = "examples print their results for the reader"
)]

#[cfg(feature = "vector-memory")]
use caliche::memory::VectorMemoryConfig;
use caliche::prelude::ProviderType;
use caliche::{
    AgentBuilder,
    memory::{EntityMemoryConfig, SummaryMemoryConfig},
};

fn display(label: &str, text: &str) {
    let normalized = text
        .replace(
            [
                '\u{202f}', '\u{00a0}', '\u{2009}', '\u{2002}', '\u{2003}', '\u{2004}', '\u{2005}',
                '\u{2006}', '\u{2007}', '\u{2008}', '\u{200a}', '\u{200b}', '\u{200c}', '\u{200d}',
                '\u{2060}', '\u{feff}',
            ],
            " ",
        )
        .replace("**", "");

    let mut spaced = String::new();
    let mut prev: Option<char> = None;
    for ch in normalized.chars() {
        if let Some(p) = prev {
            if p.is_lowercase() && ch.is_uppercase() {
                spaced.push(' ');
            }
        }

        match ch {
            '|' => {
                if !spaced.ends_with(' ') {
                    spaced.push(' ');
                }
                prev = Some(ch);
                continue;
            }
            ':' => {
                spaced.push(':');
                spaced.push(' ');
            }
            _ => spaced.push(ch),
        }

        prev = Some(ch);
    }

    let collapsed = spaced.split_whitespace().collect::<Vec<_>>().join(" ");

    let snippet: String = collapsed.chars().take(160).collect();
    let ellipsis = if collapsed.len() > snippet.len() {
        "…"
    } else {
        ""
    };
    println!("{label}: {snippet}{ellipsis}");
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Buffer memory keeps a sliding window of recent messages.
    let mut buffer_agent = AgentBuilder::new()
        .provider(ProviderType::LMStudio)
        .buffer_memory(BufferMemoryConfig::default().with_max_tokens(2048))
        .build()
        .await?;

    buffer_agent
        .chat("Remember: the launch codename is Horizon.")
        .await?;
    let response = buffer_agent.chat("What's our launch codename?").await?;
    display("Buffer memory agent", response.text());

    // Summary memory automatically condenses older turns when the context grows.
    let mut summary_agent = AgentBuilder::new()
        .provider(ProviderType::LMStudio)
        .summary_memory(SummaryMemoryConfig {
            max_tokens: 8192,
            max_summary_tokens: None,
            summarize_threshold_pct: 0.75,
            chunk_size: 8,
            max_summaries: 6,
        })
        .build()
        .await?;

    summary_agent
        .chat("Please remember that I prefer async Rust examples.")
        .await?;
    let response = summary_agent
        .chat("What kind of examples do I like?")
        .await?;
    display("Summary memory agent", response.text());

    // Entity memory keeps track of facts about people, places, and concepts.
    // Requires a provider that exposes embeddings (OpenAI, LMStudio, or Bedrock with embeddings enabled).
    let mut entity_agent = AgentBuilder::new()
        .provider(ProviderType::LMStudio)
        .entity_memory(EntityMemoryConfig {
            max_tokens: 4096,
            max_entities: 32,
            extraction_interval: 3,
            similarity_threshold: 0.6,
        })
        .build()
        .await?;

    entity_agent
        .chat("My teammate Nova owns the analytics pipeline.")
        .await?;
    let response = entity_agent
        .chat("Who owns the analytics pipeline?")
        .await?;
    display("Entity memory agent", response.text());

    // Vector memory stores conversation turns as embeddings and retrieves semantically
    // relevant memories based on the current query. Perfect for long conversations
    // where you need to recall contextually relevant information.
    // Requires a provider that exposes embeddings (OpenAI, LMStudio, or Bedrock with embeddings enabled).
    #[cfg(feature = "vector-memory")]
    {
        let mut vector_agent = AgentBuilder::new()
            .provider(ProviderType::LMStudio)
            .vector_memory(VectorMemoryConfig {
                max_tokens: 8192,
                max_memories: 1000,
                top_k: 5,
                similarity_threshold: 0.6, // Lower threshold for better recall
                buffer_tokens: 2048,       // Smaller buffer to force retrieval of older messages
                namespace: Some("example-conversation".to_string()),
            })
            .build()
            .await?;

        // Simulate a long conversation with multiple topics
        // Each chat() call stores the turn in vector memory
        println!("\nStoring conversation turns in vector memory...");
        vector_agent
            .chat("I'm planning a trip to Japan next month. I want to visit Tokyo and Kyoto.")
            .await?;
        vector_agent
            .chat("The weather in Japan during spring is usually mild and pleasant.")
            .await?;
        vector_agent
            .chat("I'm also working on a Rust project that uses async/await patterns.")
            .await?;
        vector_agent
            .chat("The project involves building a distributed system with multiple agents.")
            .await?;

        // Add a few more turns to push the travel plans out of the buffer
        vector_agent
            .chat("The system needs to handle concurrent requests efficiently.")
            .await?;
        vector_agent
            .chat("I'm using tokio for async runtime and channels for communication.")
            .await?;

        println!("Querying for travel plans (should retrieve from vector memory)...");
        // Even after many turns, VectorMemory can retrieve semantically relevant information
        // based on the query, not just recent messages
        let response = vector_agent
            .chat("What did I say about my travel plans to Japan?")
            .await?;
        display("Vector memory agent (travel query)", response.text());

        println!("Querying for technical project (should retrieve from vector memory)...");
        let response = vector_agent
            .chat("Tell me about the Rust project I mentioned earlier.")
            .await?;
        display("Vector memory agent (project query)", response.text());
    }

    // Enhanced Context Management Example
    // Demonstrates the new advanced context management capabilities with actual agent interaction
    println!("\n=== Enhanced Context Management with Agent Interaction ===");

    use caliche::agent::memory::{
        BufferMemory, BufferMemoryConfig, TokenCounterFactory,
        budget::{AllocationStrategy, ContextBudgetAllocator},
        compression::{CompressionStrategy, LlmMessageCompressor, ToolOutputCompressor},
        enhanced::EnhancedMemory,
        knowledge::KnowledgeStore,
        relevance::RelevanceScorer,
        working::WorkingMemoryManager,
    };
    use caliche::agent::service::{AgentService, ProviderFactory};
    use caliche::traits::credentials::EnvironmentCredentialProvider;
    use std::sync::Arc;
    use tokio::sync::RwLock;

    // Create providers for enhanced memory components
    let provider = ProviderFactory::create(
        ProviderType::LMStudio,
        "local-model".to_string(),
        Arc::new(EnvironmentCredentialProvider::new()),
        None,
        None,
        None,
        None,
    )
    .await?;

    let embedding_provider = ProviderFactory::create_embedding_provider(
        ProviderType::LMStudio,
        "local-model".to_string(),
        Arc::new(EnvironmentCredentialProvider::new()),
    )
    .await?;

    // Only proceed if we have an embedding provider (required for enhanced features)
    if let Some(embedding_provider) = embedding_provider {
        let token_counter = TokenCounterFactory::new().build();

        // Build enhanced memory components
        println!("\nBuilding enhanced memory components...");

        // 1. Context Budget Allocator
        let budget_allocator = ContextBudgetAllocator::with_strategy(
            128_000,
            AllocationStrategy::Dynamic {
                min_system: 1000,
                min_history: 3000,
                min_rag: 2000,
                reserve_tools: 2000,
            },
        );

        // 2. Relevance Scorer
        let relevance_scorer =
            RelevanceScorer::new(Arc::clone(&embedding_provider), token_counter.clone());

        // 3. Working Memory Manager
        let working_mgr = WorkingMemoryManager::new(token_counter.clone(), 2000);

        // 4. Base memory strategy
        let base_memory = Box::new(BufferMemory::new(
            BufferMemoryConfig::default().with_max_tokens(8192),
            token_counter.clone(),
        ));

        // 5. Message Compressor (optional, for tool outputs)
        let compressor = Arc::new(LlmMessageCompressor::new(
            provider.clone(),
            token_counter.clone(),
            CompressionStrategy::Summarize,
        ));
        let tool_compressor = Arc::new(ToolOutputCompressor::new(
            compressor,
            token_counter.clone(),
            1000, // Compress tool outputs over 1000 tokens
        ));

        // 6. Knowledge Store
        let knowledge_store = Arc::new(RwLock::new(KnowledgeStore::new(
            Arc::clone(&embedding_provider),
            token_counter.clone(),
            500, // Max 500 knowledge entries
        )));

        // Create Enhanced Memory
        let enhanced_memory =
            EnhancedMemory::new(base_memory, budget_allocator, relevance_scorer, working_mgr)
                .with_tool_compressor(tool_compressor)
                .with_knowledge_store(knowledge_store.clone());

        // Create agent service and set enhanced memory
        let mut enhanced_agent = AgentService::from_config(
            caliche::AgentServiceConfig::new()
                .with_provider(Some(ProviderType::LMStudio), None)
                .with_all_default_tools(),
        )
        .await?;

        // Wrap enhanced memory in a way that allows query setting
        // For this example, we'll create a wrapper that sets the query before get_context
        enhanced_agent.set_memory_strategy(Some(Box::new(enhanced_memory)));

        println!("Enhanced memory configured with:");
        println!("  - Context budget allocation (dynamic)");
        println!("  - Query-aware relevance scoring");
        println!("  - Tool output compression");
        println!("  - Working memory management");
        println!("  - Cross-session knowledge store");

        // Demonstrate enhanced memory in action
        println!("\n--- Conversation with Enhanced Memory ---");

        enhanced_agent
            .chat("My name is Alex and I prefer dark mode in all applications.")
            .await?;

        enhanced_agent
            .chat("I'm working on a project called 'Project Phoenix' that uses Rust.")
            .await?;

        enhanced_agent
            .chat("I live in San Francisco and enjoy hiking on weekends.")
            .await?;

        // Query that should retrieve relevant context using enhanced memory
        // The enhanced memory will use relevance scoring to find the most relevant messages
        println!("\nQuerying with enhanced context management...");
        let response = enhanced_agent
            .chat("What are my preferences and what project am I working on?")
            .await?;
        display("Enhanced memory agent", response.text());

        // Check knowledge store
        {
            let store = knowledge_store.read().await;
            let entries = store.all_entries();
            println!("\nKnowledge store contains {} learned facts", entries.len());
            if !entries.is_empty() {
                println!("Sample knowledge entries:");
                for entry in entries.iter().take(3) {
                    println!("  - [{}] {}", entry.category, entry.content);
                }
            }
        }

        println!("\n=== Enhanced Memory Demo Complete ===");
    } else {
        println!("\nEnhanced memory features require an embedding provider.");
        println!("Skipping enhanced memory demo (embedding provider not available).");

        // Still show the basic components
        let token_counter = TokenCounterFactory::new().build();

        println!("\nBasic Enhanced Memory Components:");
        let allocator = ContextBudgetAllocator::new(128_000);
        let budget = allocator.allocate(Some("What is the weather?"));
        println!(
            "  - Context Budget Allocator: {} tokens for history",
            budget.history_tokens
        );

        let mut working_mgr = WorkingMemoryManager::new(token_counter, 2000);
        let wm = working_mgr.get_or_create("demo_task".to_string());
        println!(
            "  - Working Memory: {} tokens capacity",
            wm.remaining_capacity()
        );
    }

    Ok(())
}
