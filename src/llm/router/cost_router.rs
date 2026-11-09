//! Cost-optimized router: selects cheapest provider/model for requested capability.

use super::{ProviderRouter, ProviderSelection, RoutingContext};
use anyhow::{Context, Result};
use async_trait::async_trait;
use std::collections::HashMap;

/// Cost per 1K tokens (input, output) for a provider/model.
#[derive(Debug, Clone)]
pub struct ModelCost {
    pub input_per_1k: f64,
    pub output_per_1k: f64,
}

/// Router that selects the cheapest provider/model from a cost table.
#[derive(Debug, Clone)]
pub struct CostOptimizedRouter {
    /// Map of "provider:model" -> cost
    cost_table: HashMap<String, ModelCost>,
    /// Fallback when requested model not in table: use default from context
    use_fallback: bool,
}

impl CostOptimizedRouter {
    /// Create a cost router with the given cost table.
    #[must_use]
    pub const fn new(cost_table: HashMap<String, ModelCost>) -> Self {
        Self {
            cost_table,
            use_fallback: true,
        }
    }

    /// Disable fallback to default when model not in table.
    #[must_use]
    pub const fn without_fallback(mut self) -> Self {
        self.use_fallback = false;
        self
    }

    /// Parse cost table from JSON: {"openai:gpt-4o-mini": {"`input_per_1k"`: 0.15, "`output_per_1k"`: 0.6}, ...}
    /// # Errors
    ///
    /// Returns an error if the language-model operation cannot be completed.
    pub fn from_json(json: &str) -> Result<Self> {
        let raw: HashMap<String, serde_json::Value> =
            serde_json::from_str(json).context("Invalid cost table JSON")?;
        let mut cost_table = HashMap::new();
        for (key, val) in raw {
            let obj = val.as_object().context("Cost table entry must be object")?;
            let input_per_1k = obj
                .get("input_per_1k")
                .and_then(serde_json::Value::as_f64)
                .unwrap_or(0.0);
            let output_per_1k = obj
                .get("output_per_1k")
                .and_then(serde_json::Value::as_f64)
                .unwrap_or(0.0);
            cost_table.insert(
                key,
                ModelCost {
                    input_per_1k,
                    output_per_1k,
                },
            );
        }
        Ok(Self::new(cost_table))
    }

    /// Get average cost (input+output)/2 for comparison when we don't know ratio.
    const fn avg_cost(c: &ModelCost) -> f64 {
        c.input_per_1k.midpoint(c.output_per_1k)
    }
}

#[async_trait]
impl ProviderRouter for CostOptimizedRouter {
    async fn select(&self, ctx: &RoutingContext) -> Result<ProviderSelection> {
        let requested_model = ctx.requested_model.as_deref().unwrap_or(&ctx.default_model);
        let requested_provider = ctx.requested_provider;

        // If request specifies provider+model and we have cost, we could still optimize
        // across equivalents; for now, if fully specified and in table, use it.
        if let (Some(prov), Some(model)) = (requested_provider, ctx.requested_model.as_ref()) {
            let key = format!("{prov}:{model}");
            if self.cost_table.contains_key(&key) {
                return Ok(ProviderSelection {
                    provider: prov,
                    model: model.clone(),
                });
            }
        }

        // Find cheapest entry. If model requested, only consider entries with that model.
        let model_filter: Option<&str> = if requested_model.is_empty() {
            None
        } else {
            Some(requested_model)
        };

        let mut best: Option<(ProviderSelection, f64)> = None;

        for (key, cost) in &self.cost_table {
            let Some((prov_str, model)) = key.split_once(':') else {
                continue;
            };
            if let Some(req) = model_filter
                && model != req
            {
                continue;
            }
            let Ok(provider) = prov_str.parse::<crate::llm::ProviderType>() else {
                continue;
            };

            let avg = Self::avg_cost(cost);
            let selection = ProviderSelection {
                provider,
                model: model.to_string(),
            };

            match best {
                None => best = Some((selection, avg)),
                Some((_, best_avg)) if avg < best_avg => best = Some((selection, avg)),
                _ => {}
            }
        }

        match best {
            Some((sel, _)) => Ok(sel),
            None if self.use_fallback => Ok(ProviderSelection {
                provider: ctx.default_provider,
                model: ctx.default_model.clone(),
            }),
            None => {
                anyhow::bail!("CostOptimizedRouter: no entries in cost table and fallback disabled")
            }
        }
    }
}
