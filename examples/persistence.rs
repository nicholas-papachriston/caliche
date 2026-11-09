//! Persistence Example
//!
//! Demonstrates conversation storage and session management.
//!
//! Best practices shown:
//! - Early validation of required environment variables
//! - Proper resource management with automatic cleanup
//! - TTL management for ephemeral data
//! - Checkpoint-based workflow recovery
//! - Efficient pagination for large datasets
//!
//! ## Prerequisites
//!
//! This example requires PostgreSQL and Redis to be running. You can start them using Docker:
//!
//! ```bash
//! # Start PostgreSQL
//! docker run -d \
//!   --name caliche-postgres \
//!   -e POSTGRES_PASSWORD=postgres \
//!   -e POSTGRES_DB=caliche \
//!   -p 127.0.0.1:5432:5432 \
//!   pgvector/pgvector:pg17
//!
//! # Start Redis
//! docker run -d \
//!   --name caliche-redis \
//!   -p 127.0.0.1:6379:6379 \
//!   redis:7.4-alpine
//! ```
//!
//! Then set environment variables:
//! ```bash
//! export DATABASE_URL="postgresql://postgres:postgres@localhost:5432/caliche"
//! export REDIS_URL="redis://localhost:6379"
//! ```
//!
//! Run with:
//! ```bash
//! cargo run --example persistence
//! ```

#![expect(
    clippy::print_stdout,
    reason = "examples print their results for the reader"
)]

use anyhow::{Context as AnyhowContext, Result};
use caliche::messages::{Message, MessageRole};
use caliche::persistence::{ConversationRecord, ConversationStore, RedisSessionStore, SessionData};
use chrono::Utc;
use serde_json::json;
use std::env;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize tracing for better observability
    tracing_subscriber::fmt::init();

    let database_url = env::var("DATABASE_URL").context(
        "DATABASE_URL not set. Please set it to your PostgreSQL connection string.\n\
         Example: export DATABASE_URL='postgresql://postgres:postgres@localhost:5432/caliche'",
    )?;

    let redis_url = env::var("REDIS_URL").context(
        "REDIS_URL not set. Please set it to your Redis connection string.\n\
         Example: export REDIS_URL='redis://localhost:6379'",
    )?;

    conversation_storage_example(&database_url).await?;
    session_management_example(&redis_url).await?;
    advanced_queries_example(&database_url).await?;

    Ok(())
}

/// Example 1: Conversation storage with PostgreSQL
async fn conversation_storage_example(database_url: &str) -> Result<()> {
    let store = ConversationStore::with_postgres(database_url).await?;

    // Create a conversation
    let messages = vec![
        Message::new(MessageRole::System, "You are a helpful AI assistant."),
        Message::new(MessageRole::User, "What is Rust?"),
        Message::new(
            MessageRole::Assistant,
            "Rust is a systems programming language focused on safety, speed, and concurrency.",
        ),
        Message::new(
            MessageRole::User,
            "Tell me more about its memory safety features.",
        ),
        Message::new(
            MessageRole::Assistant,
            "Rust achieves memory safety without garbage collection through its ownership system...",
        ),
    ];

    let conversation = ConversationRecord {
        id: uuid::Uuid::now_v7().to_string(),
        agent_id: "rust_expert".to_string(),
        messages,
        metadata: json!({
            "topic": "rust_programming",
            "user_id": "user_123",
            "session_type": "educational"
        }),
        created_at: Utc::now(),
        updated_at: Utc::now(),
        expires_at: None,
    };

    store.save(&conversation).await?;
    println!("Conversation saved with ID: {}", conversation.id);

    let retrieved = store.get(&conversation.id).await?;
    if let Some(conv) = retrieved {
        println!(
            "Retrieved conversation - Agent: {}, Messages: {}, Created: {}",
            conv.agent_id,
            conv.messages.len(),
            conv.created_at
        );
    }

    let conversations = store.list_by_agent("rust_expert", 10, 0).await?;
    println!("Found {} conversation(s) for agent", conversations.len());

    store.update_ttl(&conversation.id, 3600).await?;
    println!("TTL set to 1 hour");

    let stats = store.get_stats().await?;
    println!("\nConversation store statistics:");
    println!("  Total conversations: {}", stats.total_conversations);
    println!("  Active conversations: {}", stats.active_conversations);
    println!(
        "  Average messages per conversation: {:.1}",
        stats.avg_message_count
    );

    Ok(())
}

/// Example 2: Session management with Redis
async fn session_management_example(redis_url: &str) -> Result<()> {
    let store = RedisSessionStore::new(redis_url, Some("agent:session:".to_string()))?;

    // Create a session
    let session = SessionData {
        session_id: uuid::Uuid::now_v7().to_string(),
        agent_id: "customer_support".to_string(),
        conversation_id: uuid::Uuid::now_v7().to_string(),
        user_id: Some("user_456".to_string()),
        metadata: json!({
            "ip_address": "192.168.1.100",
            "user_agent": "Mozilla/5.0...",
            "auth_level": "verified"
        }),
        created_at: Utc::now(),
        last_accessed: Utc::now(),
    };

    store.save_session(&session, 3600).await?;
    println!("Session saved: {}", session.session_id);

    let retrieved = store.get_session(&session.session_id).await?;
    if let Some(sess) = retrieved {
        println!(
            "Session found - User ID: {:?}, Agent ID: {}, Last accessed: {}",
            sess.user_id, sess.agent_id, sess.last_accessed
        );
    }

    store.touch_session(&session.session_id, 3600).await?;
    println!("Session TTL extended");

    let preferences = json!({
        "theme": "dark",
        "language": "en",
        "notifications": true
    });
    store
        .set_json("user_456:preferences", &preferences, Some(7200))
        .await?;

    let retrieved_prefs: serde_json::Value = store
        .get_json("user_456:preferences")
        .await?
        .unwrap_or_default();
    println!("Retrieved preferences: {}", retrieved_prefs);

    let counter_key = "api:requests:count";
    for _ in 0..5 {
        let count = store.increment(counter_key).await?;
        println!("Request count: {}", count);
    }

    let stats = store.get_stats().await?;
    println!("\nRedis store statistics:");
    println!("  Pool size: {}", stats.pool_size);
    println!("  Available connections: {}", stats.available_connections);
    println!("  Total commands: {}", stats.total_commands);

    Ok(())
}

/// Example 3: Advanced queries and operations
async fn advanced_queries_example(database_url: &str) -> Result<()> {
    let conv_store = ConversationStore::with_postgres(database_url).await?;

    let search_filter = json!({"topic": "rust_programming"});
    let results = conv_store
        .search_by_metadata("", &search_filter, 10)
        .await?;
    println!("Found {} conversations about Rust", results.len());

    let topics = vec!["python", "javascript", "golang"];
    let mut conversation_ids = Vec::new();

    for topic in topics {
        let conv = ConversationRecord {
            id: uuid::Uuid::now_v7().to_string(),
            agent_id: format!("{}_expert", topic),
            messages: vec![
                Message::new(MessageRole::System, format!("Expert in {}", topic)),
                Message::new(MessageRole::User, format!("Tell me about {}", topic)),
            ],
            metadata: json!({
                "topic": topic,
                "difficulty": "intermediate",
                "created_by_example": true
            }),
            created_at: Utc::now(),
            updated_at: Utc::now(),
            expires_at: None,
        };

        conv_store.save(&conv).await?;
        conversation_ids.push(conv.id.clone());
    }

    let python_convs = conv_store.list_by_agent("python_expert", 10, 0).await?;
    println!("Python expert conversations: {}", python_convs.len());

    for conv_id in conversation_ids {
        conv_store.delete(&conv_id).await?;
    }

    let cleaned = conv_store.cleanup_expired().await?;
    println!("Cleaned up {} expired conversations", cleaned);

    Ok(())
}
