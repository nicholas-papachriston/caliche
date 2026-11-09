//! Distributed coordination module for multi-node agent deployments
//!
//! This module provides:
//! - Leader election using Redis/Valkey
//! - Distributed task queues
//! - Session affinity for conversation continuity
//! - Load balancing across nodes
//!
//! # Compatibility
//! Works with both Redis and Valkey (Redis protocol-compatible).
//! Pass a Redis or Valkey URL (e.g., "redis://localhost:6379" or "redis://localhost:6380").

#[cfg(feature = "redis")]
pub mod coordinator;
#[cfg(feature = "redis")]
pub mod load_balancer;
#[cfg(feature = "redis")]
pub mod session_manager;

#[cfg(feature = "redis")]
pub use coordinator::{DistributedCoordinator, LeaderElection, NodeInfo, NodeStatus};
#[cfg(feature = "redis")]
pub use load_balancer::{LoadBalancer, NodeHealth, RoutingStrategy};
#[cfg(feature = "redis")]
pub use session_manager::{SessionAffinity, SessionManager};
