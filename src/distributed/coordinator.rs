use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use deadpool_redis::redis::AsyncCommands;
use deadpool_redis::{Config, Pool, Runtime};
use futures_util::StreamExt;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use uuid::Uuid;

use super::load_balancer::{LoadBalancer, RoutingStrategy};
use super::session_manager::SessionManager;

/// Information about a node in the distributed system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeInfo {
    pub node_id: String,
    pub address: String,
    pub region: Option<String>,
    pub capabilities: Vec<String>,
    pub current_load: usize,
    pub max_capacity: usize,
    pub last_heartbeat: DateTime<Utc>,
    pub status: NodeStatus,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum NodeStatus {
    Healthy,
    Degraded,
    Unhealthy,
    Offline,
}

/// Leader election using Redis/Valkey
///
/// Compatible with both Redis and Valkey (Redis protocol-compatible).
/// Pass a Redis or Valkey URL (e.g., "redis://localhost:6379" or "redis://localhost:6380").
pub struct LeaderElection {
    pool: Pool,
    node_id: String,
    leader_key: String,
    lease_duration: Duration,
}

impl LeaderElection {
    /// Create a new leader election instance
    ///
    /// # Arguments
    /// * `coordinator_url` - Redis or Valkey connection URL (e.g., "redis://localhost:6379")
    /// * `node_id` - Unique identifier for this node
    /// * `lease_duration_secs` - Leader lease duration in seconds
    pub fn new(
        coordinator_url: &str,
        node_id: impl Into<String>,
        lease_duration_secs: u64,
    ) -> Result<Self> {
        let cfg = Config::from_url(coordinator_url);
        let pool = cfg
            .create_pool(Some(Runtime::Tokio1))
            .context("Failed to create Redis/Valkey pool")?;

        Ok(Self {
            pool,
            node_id: node_id.into(),
            leader_key: "distributed:leader".to_owned(),
            lease_duration: Duration::from_secs(lease_duration_secs),
        })
    }

    /// Attempt to become leader
    pub async fn try_become_leader(&self) -> Result<bool> {
        let mut conn = self.pool.get().await?;

        // Try to set leader key with NX (only if not exists) and EX (expiration)
        let result: Option<String> = deadpool_redis::redis::cmd("SET")
            .arg(&self.leader_key)
            .arg(&self.node_id)
            .arg("NX")
            .arg("EX")
            .arg(self.lease_duration.as_secs())
            .query_async(&mut conn)
            .await?;

        Ok(result.is_some())
    }

    /// Check if this node is the leader
    pub async fn is_leader(&self) -> Result<bool> {
        let mut conn = self.pool.get().await?;
        let leader: Option<String> = conn.get(&self.leader_key).await?;

        Ok(leader.as_ref() == Some(&self.node_id))
    }

    /// Renew leader lease
    pub async fn renew_lease(&self) -> Result<bool> {
        let mut conn = self.pool.get().await?;

        // Only renew if we're still the leader
        let script = r#"
            if redis.call("GET", KEYS[1]) == ARGV[1] then
                return redis.call("EXPIRE", KEYS[1], ARGV[2])
            else
                return 0
            end
        "#;

        let result: i32 = deadpool_redis::redis::cmd("EVAL")
            .arg(script)
            .arg(1) // number of keys
            .arg(&self.leader_key)
            .arg(&self.node_id)
            .arg(self.lease_duration.as_secs())
            .query_async(&mut conn)
            .await?;

        Ok(result == 1)
    }

    /// Get current leader node ID
    pub async fn get_leader(&self) -> Result<Option<String>> {
        let mut conn = self.pool.get().await?;
        let leader: Option<String> = conn.get(&self.leader_key).await?;
        Ok(leader)
    }

    /// Release leadership (for graceful shutdown)
    pub async fn release_leadership(&self) -> Result<()> {
        let mut conn = self.pool.get().await?;

        // Only delete if we're the leader
        let script = r#"
            if redis.call("GET", KEYS[1]) == ARGV[1] then
                return redis.call("DEL", KEYS[1])
            else
                return 0
            end
        "#;

        let _: i32 = deadpool_redis::redis::cmd("EVAL")
            .arg(script)
            .arg(1) // number of keys
            .arg(&self.leader_key)
            .arg(&self.node_id)
            .query_async(&mut conn)
            .await?;

        Ok(())
    }
}

/// Cached snapshot of the cluster's node list, along with when it was fetched.
struct NodeCache {
    nodes: Vec<NodeInfo>,
    fetched_at: Instant,
}

/// How long a cached node list may be reused by `distribute_task` before a fresh Redis scan is
/// required. Node registrations carry a 60s TTL and heartbeats typically refresh well within
/// that window, so a short cache window meaningfully cuts per-task Redis round-trips without
/// making node selection stale relative to real cluster churn.
const DEFAULT_NODE_CACHE_TTL: Duration = Duration::from_secs(2);

/// Distributed coordinator for managing nodes and task distribution
pub struct DistributedCoordinator {
    pool: Pool,
    node_id: String,
    election: Arc<LeaderElection>,
    nodes: Arc<RwLock<Option<NodeCache>>>,
    node_cache_ttl: Duration,
    load_balancer: Arc<LoadBalancer>,
    session_manager: Arc<SessionManager>,
}

impl DistributedCoordinator {
    /// Create a new distributed coordinator
    ///
    /// # Arguments
    /// * `coordinator_url` - Redis or Valkey connection URL (e.g., "redis://localhost:6379" or "redis://localhost:6380")
    /// * `node_id` - Optional unique identifier for this node (auto-generated if None)
    /// * `routing_strategy` - Optional load balancing strategy (defaults to LeastConnections)
    /// * `session_ttl_secs` - Optional session TTL in seconds (defaults to 3600 = 1 hour)
    ///
    /// # Compatibility
    /// Works with both Redis and Valkey (Redis protocol-compatible).
    pub fn new(
        coordinator_url: &str,
        node_id: Option<String>,
        routing_strategy: Option<RoutingStrategy>,
        session_ttl_secs: Option<u64>,
    ) -> Result<Self> {
        let node_id = node_id.unwrap_or_else(|| Uuid::now_v7().to_string());
        let election = Arc::new(LeaderElection::new(coordinator_url, node_id.clone(), 30)?);

        let cfg = Config::from_url(coordinator_url);
        let pool = cfg
            .create_pool(Some(Runtime::Tokio1))
            .context("Failed to create Redis/Valkey pool")?;

        let load_balancer = Arc::new(LoadBalancer::new(
            routing_strategy.unwrap_or(RoutingStrategy::LeastConnections),
        ));
        let session_manager = Arc::new(SessionManager::new(
            coordinator_url,
            session_ttl_secs.unwrap_or(3600), // Default 1 hour TTL
        )?);

        Ok(Self {
            pool,
            node_id,
            election,
            nodes: Arc::new(RwLock::new(None)),
            node_cache_ttl: DEFAULT_NODE_CACHE_TTL,
            load_balancer,
            session_manager,
        })
    }

    /// Override how long `distribute_task` may reuse a cached node list before rescanning
    /// Redis (default: `DEFAULT_NODE_CACHE_TTL`).
    pub fn with_node_cache_ttl(mut self, ttl: Duration) -> Self {
        self.node_cache_ttl = ttl;
        self
    }

    /// Register this node with the cluster
    pub async fn register_node(&self, info: NodeInfo) -> Result<()> {
        let mut conn = self.pool.get().await?;
        let key = format!("distributed:nodes:{}", info.node_id);
        let value = serde_json::to_string(&info)?;

        conn.set_ex::<_, _, ()>(&key, value, 60).await?; // 60 second TTL

        tracing::info!(node_id = %info.node_id, "Registered node in cluster");
        Ok(())
    }

    /// Send heartbeat to maintain node registration
    pub async fn heartbeat(&self, node_info: NodeInfo) -> Result<()> {
        self.register_node(node_info).await
    }

    /// Get all registered nodes. Always performs a fresh scan against Redis (and refreshes
    /// the internal cache used by `distribute_task`) — call this when you need an up-to-date
    /// view (e.g. cluster stats); prefer the cache-aware path on the hot task-dispatch path.
    pub async fn get_nodes(&self) -> Result<Vec<NodeInfo>> {
        let nodes = self.scan_nodes_from_redis().await?;

        *self.nodes.write().await = Some(NodeCache {
            nodes: nodes.clone(),
            fetched_at: Instant::now(),
        });

        Ok(nodes)
    }

    /// Scan the cluster's node keys and batch-fetch their values.
    ///
    /// Uses `SCAN` (cursor-based, non-blocking on the Redis server) instead of `KEYS`
    /// (which blocks the server for O(keyspace size)), and fetches all values in one `MGET`
    /// round-trip instead of one `GET` per key.
    async fn scan_nodes_from_redis(&self) -> Result<Vec<NodeInfo>> {
        let mut conn = self.pool.get().await?;
        let pattern = "distributed:nodes:*";

        let mut keys: Vec<String> = Vec::new();
        {
            let mut key_iter: deadpool_redis::redis::AsyncIter<'_, String> =
                conn.scan_match(pattern).await?;
            while let Some(key) = key_iter.next().await {
                keys.push(key?);
            }
        }

        if keys.is_empty() {
            return Ok(Vec::new());
        }

        let values: Vec<Option<String>> = conn.mget(&keys).await?;
        let nodes = values
            .into_iter()
            .filter_map(|data| data.and_then(|d| serde_json::from_str::<NodeInfo>(&d).ok()))
            .collect();

        Ok(nodes)
    }

    /// Get the cluster's node list, reusing a recent cached scan (within `node_cache_ttl`)
    /// when available instead of rescanning Redis on every call. Used by `distribute_task`,
    /// which would otherwise re-scan the entire node keyspace on every single task dispatch.
    async fn get_nodes_cached(&self) -> Result<Vec<NodeInfo>> {
        {
            let cache = self.nodes.read().await;
            if let Some(ref cached) = *cache
                && cached.fetched_at.elapsed() < self.node_cache_ttl
            {
                return Ok(cached.nodes.clone());
            }
        }

        self.get_nodes().await
    }

    /// Distribute task to appropriate node
    pub async fn distribute_task(&self, task: &Task) -> Result<String> {
        let nodes = self.get_nodes_cached().await?;

        // Filter healthy nodes with capacity
        let available: Vec<NodeInfo> = nodes
            .iter()
            .filter(|n| n.status == NodeStatus::Healthy && n.current_load < n.max_capacity)
            .cloned()
            .collect();

        if available.is_empty() {
            anyhow::bail!("No available nodes for task distribution");
        }

        // Check for session affinity if task has a session_id
        let target_node_id = if let Some(session_id) = &task.session_id {
            if let Ok(Some(affinity)) = self.session_manager.get_affinity(session_id).await {
                // Check if affinity node is still healthy and available
                let affinity_node_available = available
                    .iter()
                    .any(|n| n.node_id == affinity.node_id && n.status == NodeStatus::Healthy);

                if affinity_node_available {
                    tracing::debug!(
                        task_id = %task.task_id,
                        session_id = %session_id,
                        node_id = %affinity.node_id,
                        "Using session affinity for task distribution"
                    );
                    affinity.node_id.clone()
                } else {
                    // Affinity node is unhealthy, try fallback nodes
                    let fallback_available = available
                        .iter()
                        .find(|n| affinity.fallback_nodes.contains(&n.node_id));

                    if let Some(fallback_node) = fallback_available {
                        tracing::warn!(
                            task_id = %task.task_id,
                            session_id = %session_id,
                            old_node_id = %affinity.node_id,
                            new_node_id = %fallback_node.node_id,
                            "Affinity node unhealthy, using fallback node"
                        );
                        // Update affinity to fallback node
                        let _ = self
                            .session_manager
                            .reassign_session(session_id, fallback_node.node_id.clone())
                            .await;
                        fallback_node.node_id.clone()
                    } else {
                        // No fallback available, use load balancer and update affinity
                        let selected = self
                            .load_balancer
                            .select_node(&available)
                            .await
                            .ok_or_else(|| {
                                anyhow::anyhow!("Failed to select node via load balancer")
                            })?;
                        tracing::info!(
                            task_id = %task.task_id,
                            session_id = %session_id,
                            old_node_id = %affinity.node_id,
                            new_node_id = %selected,
                            "Affinity node and fallbacks unavailable, using load balancer"
                        );
                        // Update affinity to new node
                        let _ = self
                            .session_manager
                            .set_affinity(
                                session_id.clone(),
                                selected.clone(),
                                Vec::new(), // No fallbacks set yet
                            )
                            .await;
                        selected
                    }
                }
            } else {
                // No affinity exists, use load balancer and set affinity
                let selected = self
                    .load_balancer
                    .select_node(&available)
                    .await
                    .ok_or_else(|| anyhow::anyhow!("Failed to select node via load balancer"))?;
                tracing::debug!(
                    task_id = %task.task_id,
                    session_id = %session_id,
                    node_id = %selected,
                    "No session affinity, using load balancer and setting affinity"
                );
                // Set affinity for future requests
                let _ = self
                    .session_manager
                    .set_affinity(
                        session_id.clone(),
                        selected.clone(),
                        Vec::new(), // No fallbacks set yet
                    )
                    .await;
                selected
            }
        } else {
            // No session_id, use load balancer
            self.load_balancer
                .select_node(&available)
                .await
                .ok_or_else(|| anyhow::anyhow!("Failed to select node via load balancer"))?
        };

        // Push task to node's queue
        let queue_key = format!("distributed:queue:{}", target_node_id);
        let task_json = serde_json::to_string(task)?;

        let mut conn = self.pool.get().await?;
        conn.rpush::<_, _, ()>(&queue_key, task_json).await?;

        tracing::info!(
            task_id = %task.task_id,
            node_id = %target_node_id,
            "Distributed task to node"
        );

        Ok(target_node_id)
    }

    /// Poll for tasks from this node's queue
    pub async fn poll_tasks(&self, timeout_secs: u64) -> Result<Option<Task>> {
        let queue_key = format!("distributed:queue:{}", self.node_id);
        let mut conn = self.pool.get().await?;

        let result: Option<(String, String)> = conn.blpop(&queue_key, timeout_secs as f64).await?;

        if let Some((_, task_json)) = result {
            let task = serde_json::from_str(&task_json)?;
            Ok(Some(task))
        } else {
            Ok(None)
        }
    }

    /// Start leader election loop
    pub fn start_leader_election_loop(self: Arc<Self>) {
        tokio::spawn(async move {
            loop {
                if let Ok(true) = self.election.try_become_leader().await {
                    tracing::info!(node_id = %self.node_id, "Became cluster leader");

                    // Keep renewing lease while we're leader
                    loop {
                        tokio::time::sleep(Duration::from_secs(10)).await;

                        match self.election.renew_lease().await {
                            Ok(true) => continue,
                            _ => {
                                tracing::warn!(node_id = %self.node_id, "Lost leadership");
                                break;
                            }
                        }
                    }
                }

                tokio::time::sleep(Duration::from_secs(5)).await;
            }
        });
    }

    /// Get node statistics
    pub async fn get_cluster_stats(&self) -> Result<ClusterStats> {
        let nodes = self.get_nodes().await?;

        let healthy_count = nodes
            .iter()
            .filter(|n| n.status == NodeStatus::Healthy)
            .count();
        let total_capacity = nodes.iter().map(|n| n.max_capacity).sum();
        let total_load = nodes.iter().map(|n| n.current_load).sum();

        Ok(ClusterStats {
            total_nodes: nodes.len(),
            healthy_nodes: healthy_count,
            total_capacity,
            total_load,
            leader: self.election.get_leader().await?,
        })
    }

    pub fn node_id(&self) -> &str {
        &self.node_id
    }

    pub fn leader_election(&self) -> Arc<LeaderElection> {
        Arc::clone(&self.election)
    }

    pub fn load_balancer(&self) -> Arc<LoadBalancer> {
        Arc::clone(&self.load_balancer)
    }

    pub fn session_manager(&self) -> Arc<SessionManager> {
        Arc::clone(&self.session_manager)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Task {
    pub task_id: String,
    pub task_type: String,
    pub payload: serde_json::Value,
    pub priority: i32,
    pub created_at: DateTime<Utc>,
    pub session_id: Option<String>,
}

impl Task {
    pub fn new(task_type: impl Into<String>, payload: serde_json::Value) -> Self {
        Self {
            task_id: Uuid::now_v7().to_string(),
            task_type: task_type.into(),
            payload,
            priority: 0,
            created_at: Utc::now(),
            session_id: None,
        }
    }

    pub fn with_priority(mut self, priority: i32) -> Self {
        self.priority = priority;
        self
    }

    pub fn with_session_id(mut self, session_id: impl Into<String>) -> Self {
        self.session_id = Some(session_id.into());
        self
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ClusterStats {
    pub total_nodes: usize,
    pub healthy_nodes: usize,
    pub total_capacity: usize,
    pub total_load: usize,
    pub leader: Option<String>,
}
