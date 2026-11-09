//! Distributed Coordination Example
//!
//! This example demonstrates:
//! - Redis leader election for cluster coordination
//! - Distributed task queues
//! - Session affinity for conversations
//! - Failover and recovery
//!
//! Best practices shown:
//! - Multi-node agent deployment
//! - High availability patterns
//!
//! Run with:
//! ```bash
//! cargo run --example distributed
//! ```
//!
//! Prerequisites:
//! - Redis running
//! - Set REDIS_URL environment variable

#![expect(
    clippy::print_stdout,
    reason = "examples print their results for the reader"
)]

use anyhow::{Context, Result};
use chrono::Utc;
use serde_json::json;
use std::time::Instant;

use caliche::distributed::coordinator::Task;
use caliche::distributed::{DistributedCoordinator, NodeInfo, NodeStatus, RoutingStrategy};

#[tokio::main]
async fn main() -> Result<()> {
    dotenvy::dotenv().ok();

    let redis_url =
        std::env::var("REDIS_URL").context("REDIS_URL not set. Example: redis://localhost:6379")?;

    let node_a = DistributedCoordinator::new(
        &redis_url,
        Some("node-a".to_string()),
        Some(RoutingStrategy::LeastConnections),
        Some(3600),
    )?;
    let node_b = DistributedCoordinator::new(
        &redis_url,
        Some("node-b".to_string()),
        Some(RoutingStrategy::LeastConnections),
        Some(3600),
    )?;

    let register_start = Instant::now();
    node_a
        .register_node(NodeInfo {
            node_id: "node-a".to_string(),
            address: "127.0.0.1:7001".to_string(),
            region: Some("local".to_string()),
            capabilities: vec!["chat".to_string(), "tools".to_string()],
            current_load: 1,
            max_capacity: 10,
            last_heartbeat: Utc::now(),
            status: NodeStatus::Healthy,
        })
        .await?;
    node_b
        .register_node(NodeInfo {
            node_id: "node-b".to_string(),
            address: "127.0.0.1:7002".to_string(),
            region: Some("local".to_string()),
            capabilities: vec!["chat".to_string(), "tools".to_string()],
            current_load: 2,
            max_capacity: 10,
            last_heartbeat: Utc::now(),
            status: NodeStatus::Healthy,
        })
        .await?;

    let election_a = node_a.leader_election();
    let became_leader_a = election_a.try_become_leader().await?;
    let current_leader = election_a.get_leader().await?;

    let session_id = "demo-session";
    let task = Task::new(
        "chat".to_string(),
        json!({ "prompt": "Summarize today cluster health" }),
    )
    .with_priority(5)
    .with_session_id(session_id.to_string());

    let first_target = node_a.distribute_task(&task).await?;
    let first_polled = if first_target == "node-a" {
        node_a.poll_tasks(1).await?
    } else {
        node_b.poll_tasks(1).await?
    };

    let failover_start = Instant::now();
    let unhealthy = if first_target == "node-a" {
        "node-a"
    } else {
        "node-b"
    };
    let healthy = if unhealthy == "node-a" {
        "node-b"
    } else {
        "node-a"
    };
    if unhealthy == "node-a" {
        node_a
            .register_node(NodeInfo {
                node_id: "node-a".to_string(),
                address: "127.0.0.1:7001".to_string(),
                region: Some("local".to_string()),
                capabilities: vec!["chat".to_string(), "tools".to_string()],
                current_load: 10,
                max_capacity: 10,
                last_heartbeat: Utc::now(),
                status: NodeStatus::Unhealthy,
            })
            .await?;
    } else {
        node_b
            .register_node(NodeInfo {
                node_id: "node-b".to_string(),
                address: "127.0.0.1:7002".to_string(),
                region: Some("local".to_string()),
                capabilities: vec!["chat".to_string(), "tools".to_string()],
                current_load: 10,
                max_capacity: 10,
                last_heartbeat: Utc::now(),
                status: NodeStatus::Unhealthy,
            })
            .await?;
    }
    node_a
        .session_manager()
        .set_affinity(
            session_id.to_string(),
            unhealthy.to_string(),
            vec![healthy.to_string()],
        )
        .await?;

    let failover_task = Task::new(
        "chat".to_string(),
        json!({ "prompt": "Continue same session on healthy node" }),
    )
    .with_priority(6)
    .with_session_id(session_id.to_string());
    let second_target = node_a.distribute_task(&failover_task).await?;
    let second_polled = if second_target == "node-a" {
        node_a.poll_tasks(1).await?
    } else {
        node_b.poll_tasks(1).await?
    };

    let failover_count = usize::from(first_target != second_target);
    let cluster_stats = node_a.get_cluster_stats().await?;

    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "redis_url": redis_url,
            "leader_election": {
                "became_leader_on_node_a": became_leader_a,
                "current_leader": current_leader
            },
            "task_distribution": {
                "first_target_node": first_target,
                "first_task_polled": first_polled.as_ref().map(|t| t.task_id.clone()),
                "second_target_node_after_failover": second_target,
                "second_task_polled": second_polled.as_ref().map(|t| t.task_id.clone()),
                "task_failover_count": failover_count
            },
            "cluster_stats": cluster_stats,
            "timings_ms": {
                "register_and_initial_distribution": register_start.elapsed().as_millis(),
                "failover_and_redistribution": failover_start.elapsed().as_millis()
            }
        }))?
    );

    if became_leader_a {
        let _ = election_a.release_leadership().await;
    }

    Ok(())
}
