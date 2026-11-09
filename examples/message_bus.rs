//! MessageBus Example
//!
//! This example demonstrates:
//! - Direct agent-to-agent messaging
//! - Priority-based message queuing
//! - Broadcast messages to all agents
//! - Message threading with reply-to
//! - Message bus statistics and monitoring
//!
//! Best practices shown:
//! - Inter-agent communication patterns
//! - Priority handling for urgent messages
//!
//! Run with:
//! ```bash
//! cargo run --example message_bus
//! ```

#![expect(
    clippy::print_stdout,
    reason = "examples print their results for the reader"
)]

use anyhow::Result;
use caliche::orchestration::message_passing::{
    AgentMessage, BusEvent, MessageBus, MessagePriority,
};
use tokio::time::{Duration, sleep};

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize tracing for better observability
    tracing_subscriber::fmt::init();

    let bus = MessageBus::new();

    direct_messaging_example(&bus).await?;
    priority_queue_example(&bus).await?;
    broadcast_example(&bus).await?;
    message_threading_example(&bus).await?;
    statistics_example(&bus).await?;

    Ok(())
}

/// Example 1: Direct agent-to-agent messaging
async fn direct_messaging_example(bus: &MessageBus) -> Result<()> {
    // Create a direct message from agent1 to agent2
    let msg = AgentMessage::new(
        "agent_1".to_string(),
        Some("agent_2".to_string()),
        "Hello from Agent 1!".to_string(),
    );

    bus.publish(msg).await;

    if let Some(received) = bus.poll_message("agent_2").await {
        println!("Agent 2 received: {}", received.content);
        println!(
            "From: {}, Message ID: {}",
            received.from_agent_id, received.message_id
        );
    }

    Ok(())
}

/// Example 2: Priority-based message handling
async fn priority_queue_example(bus: &MessageBus) -> Result<()> {
    let low_priority = AgentMessage::new(
        "agent_1".to_string(),
        Some("agent_3".to_string()),
        "Low priority task".to_string(),
    )
    .with_priority(MessagePriority::Low);
    bus.publish(low_priority).await;

    let normal_priority = AgentMessage::new(
        "agent_1".to_string(),
        Some("agent_3".to_string()),
        "Normal priority task".to_string(),
    )
    .with_priority(MessagePriority::Normal);
    bus.publish(normal_priority).await;

    let critical_priority = AgentMessage::new(
        "agent_1".to_string(),
        Some("agent_3".to_string()),
        "CRITICAL: Urgent task!".to_string(),
    )
    .with_priority(MessagePriority::Critical);
    bus.publish(critical_priority).await;

    let high_priority = AgentMessage::new(
        "agent_1".to_string(),
        Some("agent_3".to_string()),
        "High priority task".to_string(),
    )
    .with_priority(MessagePriority::High);
    bus.publish(high_priority).await;

    println!("Agent 3 receiving messages (priority order):");
    for i in 1..=4 {
        if let Some(msg) = bus.poll_message("agent_3").await {
            println!("  {}. [Priority: {:?}] {}", i, msg.priority, msg.content);
        }
    }

    Ok(())
}

/// Example 3: Broadcast messages to all agents
async fn broadcast_example(bus: &MessageBus) -> Result<()> {
    // Create subscribers for multiple agents
    let mut subscriber1 = bus.subscribe();
    let mut subscriber2 = bus.subscribe();
    let mut subscriber3 = bus.subscribe();

    // Send broadcast message (no specific recipient)
    let broadcast_msg = AgentMessage::new(
        "coordinator".to_string(),
        None, // None = broadcast
        "System-wide announcement: Starting maintenance in 5 minutes".to_string(),
    );

    bus.publish(broadcast_msg).await;
    sleep(Duration::from_millis(10)).await;

    println!("Subscribers receiving broadcast:");
    if let Ok(event) = subscriber1.try_recv() {
        if let BusEvent::Message(msg) = event {
            println!("  Agent 1 received: {}", msg.content);
        }
    }

    if let Ok(event) = subscriber2.try_recv() {
        if let BusEvent::Message(msg) = event {
            println!("  Agent 2 received: {}", msg.content);
        }
    }

    if let Ok(event) = subscriber3.try_recv() {
        if let BusEvent::Message(msg) = event {
            println!("  Agent 3 received: {}", msg.content);
        }
    }

    Ok(())
}

/// Example 4: Message threading with reply-to
async fn message_threading_example(bus: &MessageBus) -> Result<()> {
    // Initial message
    let initial_msg = AgentMessage::new(
        "agent_1".to_string(),
        Some("agent_2".to_string()),
        "Can you process this data?".to_string(),
    );
    let initial_msg_id = initial_msg.message_id.clone();

    bus.publish(initial_msg).await;

    if let Some(received) = bus.poll_message("agent_2").await {
        let reply = AgentMessage::new(
            "agent_2".to_string(),
            Some("agent_1".to_string()),
            "Yes, data processed successfully!".to_string(),
        )
        .as_reply_to(received.message_id.clone());

        println!(
            "Agent 2 replies: {} (in reply to: {})",
            reply.content, initial_msg_id
        );
        bus.publish(reply).await;
    }

    if let Some(reply) = bus.poll_message("agent_1").await {
        println!("Agent 1 received reply: {}", reply.content);
        if let Some(reply_to_id) = &reply.reply_to {
            println!("  (Reply to: {})", reply_to_id);
        }
    }

    Ok(())
}

/// Example 5: Statistics and monitoring
async fn statistics_example(bus: &MessageBus) -> Result<()> {
    // Send some test messages with metadata
    for i in 1..=5 {
        let msg = AgentMessage::new(
            "agent_1".to_string(),
            Some("agent_2".to_string()),
            format!("Test message {}", i),
        )
        .with_metadata(serde_json::json!({
            "sequence": i,
            "test_run": "statistics_example"
        }))
        .with_priority(if i % 2 == 0 {
            MessagePriority::High
        } else {
            MessagePriority::Normal
        });

        bus.publish(msg).await;
    }

    let stats = bus.get_stats().await;

    println!("MessageBus Statistics:");
    println!("  Queue size: {}", stats.queue_size);
    println!("  Max queue size: {}", stats.max_queue_size);
    println!("  Priority breakdown:");
    println!(
        "    Low: {}, Normal: {}, High: {}, Critical: {}",
        stats.low_priority, stats.normal_priority, stats.high_priority, stats.critical_priority
    );
    println!("  Active subscribers: {}", stats.subscriber_count);

    // Clean up - poll all messages
    bus.poll_all_messages("agent_2").await;

    Ok(())
}
