use chrono::{DateTime, Utc};
use dashmap::DashMap;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UsageRecord {
    pub timestamp: DateTime<Utc>,
    pub user_id: Option<String>,
    pub tenant_id: Option<String>,
    pub agent_id: String,
    pub provider: String,
    pub model: String,
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
    pub total_tokens: u64,
    pub cost_usd: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CostReport {
    pub tenant_id: Option<String>,
    pub total_tokens: u64,
    pub total_cost_usd: f64,
    pub records_count: usize,
    pub breakdown_by_model: Vec<ModelCost>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelCost {
    pub provider: String,
    pub model: String,
    pub total_tokens: u64,
    pub cost_usd: f64,
    pub request_count: usize,
}

pub struct CostTracker {
    records: Arc<DashMap<String, Vec<UsageRecord>>>, // tenant_id -> records
    pricing: Arc<DashMap<String, ModelPricing>>,     // model_id -> pricing
}

#[derive(Debug, Clone)]
struct ModelPricing {
    prompt_price_per_1k: f64,
    completion_price_per_1k: f64,
}

impl CostTracker {
    pub fn new() -> Self {
        let tracker = Self {
            records: Arc::new(DashMap::new()),
            pricing: Arc::new(DashMap::new()),
        };

        // Initialize default pricing (example prices)
        tracker.set_pricing("openai/gpt-4o", 0.005, 0.015);
        tracker.set_pricing("anthropic/claude-3-5-sonnet", 0.003, 0.015);
        tracker.set_pricing("openrouter/qwen-plus", 0.001, 0.003);
        tracker.set_pricing("cloudflare-ai-gateway/openai/gpt-4o-mini", 0.00015, 0.00060);

        tracker
    }

    pub fn set_pricing(&self, model: &str, prompt_price: f64, completion_price: f64) {
        self.pricing.insert(
            model.to_string(),
            ModelPricing {
                prompt_price_per_1k: prompt_price,
                completion_price_per_1k: completion_price,
            },
        );
    }

    pub fn record_usage(&self, record: UsageRecord) {
        let key = record
            .tenant_id
            .clone()
            .unwrap_or_else(|| "default".to_string());
        self.records.entry(key).or_default().push(record);
    }

    pub fn calculate_cost(
        &self,
        provider: &str,
        model: &str,
        prompt_tokens: u64,
        completion_tokens: u64,
    ) -> f64 {
        let model_key = format!("{}/{}", provider, model);

        if let Some(pricing) = self.pricing.get(&model_key) {
            let prompt_cost = (prompt_tokens as f64 / 1000.0) * pricing.prompt_price_per_1k;
            let completion_cost =
                (completion_tokens as f64 / 1000.0) * pricing.completion_price_per_1k;
            prompt_cost + completion_cost
        } else {
            0.0 // Unknown model, default to 0
        }
    }

    pub fn get_report(&self, tenant_id: Option<&str>) -> CostReport {
        let key = tenant_id.unwrap_or("default");

        if let Some(records) = self.records.get(key) {
            let mut total_tokens = 0u64;
            let mut total_cost = 0.0f64;
            let mut model_stats: std::collections::HashMap<String, (u64, f64, usize)> =
                std::collections::HashMap::new();

            for record in records.iter() {
                total_tokens += record.total_tokens;
                total_cost += record.cost_usd;

                let model_key = format!("{}/{}", record.provider, record.model);
                let entry = model_stats.entry(model_key).or_insert((0, 0.0, 0));
                entry.0 += record.total_tokens;
                entry.1 += record.cost_usd;
                entry.2 += 1;
            }

            let breakdown_by_model = model_stats
                .into_iter()
                .map(|(model, (tokens, cost, count))| {
                    let parts: Vec<_> = model.split('/').collect();
                    ModelCost {
                        provider: parts[0].to_string(),
                        model: parts.get(1).unwrap_or(&"unknown").to_string(),
                        total_tokens: tokens,
                        cost_usd: cost,
                        request_count: count,
                    }
                })
                .collect();

            CostReport {
                tenant_id: tenant_id.map(|s| s.to_string()),
                total_tokens,
                total_cost_usd: total_cost,
                records_count: records.len(),
                breakdown_by_model,
            }
        } else {
            CostReport {
                tenant_id: tenant_id.map(|s| s.to_string()),
                total_tokens: 0,
                total_cost_usd: 0.0,
                records_count: 0,
                breakdown_by_model: vec![],
            }
        }
    }
}

impl Default for CostTracker {
    fn default() -> Self {
        Self::new()
    }
}
