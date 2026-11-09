pub mod estimator;
pub mod pricing;
pub mod tracker;

pub use estimator::{ConfidenceLevel, CostBreakdown, CostEstimator};
pub use pricing::{ModelPricing, get_pricing};
pub use tracker::{CostReport, CostTracker, ModelCost, UsageRecord};

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenCost {
    pub provider: String,
    pub model: String,
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
    pub total_cost_usd: f64,
}
