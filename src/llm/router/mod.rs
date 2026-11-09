//! Provider routing for dynamic (provider, model) selection.
//!
//! Supports static, weighted, and cost-optimized routing strategies
//! inspired by ai-gateway patterns.

mod cost_router;
mod latency_router;
mod static_router;
mod weighted_router;

use crate::llm::ProviderType;
use anyhow::Result;
use async_trait::async_trait;
use std::collections::HashMap;

/// Selected provider and model for a request.
#[derive(Debug, Clone)]
pub struct ProviderSelection {
    pub provider: ProviderType,
    pub model: String,
}

/// Context passed to the router for selection decisions.
#[derive(Debug, Clone)]
pub struct RoutingContext {
    /// Requested provider (if any)
    pub requested_provider: Option<ProviderType>,
    /// Requested model (if any)
    pub requested_model: Option<String>,
    /// Tenant or session identifier for per-tenant routing/budgets
    pub tenant_id: Option<String>,
    /// Default provider when request does not specify one
    pub default_provider: ProviderType,
    /// Default model when request does not specify one
    pub default_model: String,
    /// Optional metadata for routing (e.g. capability hints)
    pub metadata: HashMap<String, serde_json::Value>,
}

/// Router that selects (provider, model) for each request.
#[async_trait]
pub trait ProviderRouter: Send + Sync {
    /// Select provider and model for the given request context.
    async fn select(&self, ctx: &RoutingContext) -> Result<ProviderSelection>;
}

pub use cost_router::{CostOptimizedRouter, ModelCost};
pub use latency_router::{LatencyCandidate, LatencyRouter};
pub use static_router::StaticRouter;
pub use weighted_router::{WeightedEntry, WeightedRouter};
