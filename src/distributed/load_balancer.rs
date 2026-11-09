use super::coordinator::NodeInfo;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::RwLock;

#[derive(Debug, Clone, Copy)]
pub enum RoutingStrategy {
    RoundRobin,
    LeastConnections,
    WeightedRoundRobin,
    RegionAware,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeHealth {
    pub node_id: String,
    pub healthy: bool,
    pub response_time_ms: u64,
    pub error_rate: f64,
    pub circuit_open: bool,
}

pub struct LoadBalancer {
    strategy: RoutingStrategy,
    round_robin_counter: Arc<RwLock<usize>>,
    node_health: Arc<RwLock<Vec<NodeHealth>>>,
    /// Preferred region for `RegionAware` strategy. Determined from environment on construction.
    preferred_region: Option<String>,
}

impl LoadBalancer {
    pub fn new(strategy: RoutingStrategy) -> Self {
        // Read preferred region from environment so region-affinity works without changing
        // the public API signature.
        let preferred_region = std::env::var("CALICHE_REGION")
            .ok()
            .filter(|s| !s.is_empty());
        Self {
            strategy,
            round_robin_counter: Arc::new(RwLock::new(0)),
            node_health: Arc::new(RwLock::new(Vec::new())),
            preferred_region,
        }
    }

    /// Create a load balancer with an explicit preferred region for `RegionAware` routing.
    pub fn new_with_region(strategy: RoutingStrategy, preferred_region: Option<String>) -> Self {
        Self {
            strategy,
            round_robin_counter: Arc::new(RwLock::new(0)),
            node_health: Arc::new(RwLock::new(Vec::new())),
            preferred_region,
        }
    }

    pub async fn select_node(&self, nodes: &[NodeInfo]) -> Option<String> {
        let healthy: Vec<_> = nodes
            .iter()
            .filter(|n| n.status == super::coordinator::NodeStatus::Healthy)
            .collect();

        if healthy.is_empty() {
            return None;
        }

        match self.strategy {
            RoutingStrategy::RoundRobin => {
                let mut counter = self.round_robin_counter.write().await;
                let idx = *counter % healthy.len();
                *counter += 1;
                Some(healthy[idx].node_id.clone())
            }
            RoutingStrategy::LeastConnections => healthy
                .iter()
                .min_by_key(|n| n.current_load)
                .map(|n| n.node_id.clone()),
            RoutingStrategy::WeightedRoundRobin => healthy
                .iter()
                .max_by_key(|n| n.max_capacity - n.current_load)
                .map(|n| n.node_id.clone()),
            RoutingStrategy::RegionAware => self.select_region_aware(&healthy),
        }
    }

    /// Region-aware node selection: prefer nodes in the same region, fall back to
    /// least-connections across all healthy nodes if no regional match is found.
    fn select_region_aware(&self, healthy: &[&NodeInfo]) -> Option<String> {
        if let Some(ref preferred) = self.preferred_region {
            // Try nodes in the preferred region first, picking the least-loaded one
            let regional: Vec<_> = healthy
                .iter()
                .filter(|n| n.region.as_deref() == Some(preferred.as_str()))
                .collect();

            if !regional.is_empty() {
                return regional
                    .iter()
                    .min_by_key(|n| n.current_load)
                    .map(|n| n.node_id.clone());
            }

            tracing::debug!(
                preferred_region = %preferred,
                "no healthy nodes in preferred region, falling back to global least-connections"
            );
        }

        // Fall back to least-connections across all healthy nodes
        healthy
            .iter()
            .min_by_key(|n| n.current_load)
            .map(|n| n.node_id.clone())
    }

    pub async fn update_health(&self, health: NodeHealth) {
        let mut health_map = self.node_health.write().await;
        if let Some(pos) = health_map.iter().position(|h| h.node_id == health.node_id) {
            health_map[pos] = health;
        } else {
            health_map.push(health);
        }
    }

    pub async fn get_health(&self, node_id: &str) -> Option<NodeHealth> {
        let health_map = self.node_health.read().await;
        health_map.iter().find(|h| h.node_id == node_id).cloned()
    }
}
