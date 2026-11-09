//! Weighted router: distributes requests across providers by configured weights.

use super::{ProviderRouter, ProviderSelection, RoutingContext};
use anyhow::{Context, Result};
use async_trait::async_trait;
use std::sync::atomic::{AtomicUsize, Ordering};

/// A single weighted (provider, model) option.
#[derive(Debug, Clone)]
pub struct WeightedEntry {
    pub provider: crate::llm::ProviderType,
    pub model: String,
    pub weight: u32,
}

/// Router that selects provider/model by weighted random or round-robin.
#[derive(Debug)]
pub struct WeightedRouter {
    entries: Vec<WeightedEntry>,
    total_weight: u32,
    /// Round-robin counter for deterministic distribution when not using random
    round_robin: AtomicUsize,
    use_random: bool,
}

impl WeightedRouter {
    /// Create a weighted router from entries.
    /// Weights are normalized; selection uses weighted random by default.
    /// # Errors
    ///
    /// Returns an error if the language-model operation cannot be completed.
    pub fn new(entries: Vec<WeightedEntry>) -> Result<Self> {
        let total_weight: u32 = entries.iter().map(|e| e.weight).sum();
        if total_weight == 0 {
            anyhow::bail!("WeightedRouter: total weight must be > 0");
        }
        Ok(Self {
            entries,
            total_weight,
            round_robin: AtomicUsize::new(0),
            use_random: true,
        })
    }

    /// Use round-robin instead of weighted random.
    #[must_use]
    pub const fn with_round_robin(mut self) -> Self {
        self.use_random = false;
        self
    }

    /// Parse weights from env format: "openai:gpt-4o-mini:70,anthropic:claude-3-haiku:30"
    /// # Errors
    ///
    /// Returns an error if the language-model operation cannot be completed.
    pub fn from_env_weights(env_value: &str) -> Result<Self> {
        let mut entries = Vec::new();
        for part in env_value.split(',') {
            let part = part.trim();
            if part.is_empty() {
                continue;
            }
            let segs: Vec<&str> = part.splitn(3, ':').collect();
            if segs.len() != 3 {
                anyhow::bail!("Invalid weight entry '{part}': expected provider:model:weight");
            }
            let provider = segs[0]
                .parse()
                .map_err(|e: String| anyhow::anyhow!("Invalid provider: {e}"))?;
            let model = segs[1].to_string();
            let weight: u32 = segs[2]
                .parse()
                .map_err(|e| anyhow::anyhow!("Invalid weight '{}': {e}", segs[2]))?;
            entries.push(WeightedEntry {
                provider,
                model,
                weight,
            });
        }
        Self::new(entries)
    }
}

#[async_trait]
impl ProviderRouter for WeightedRouter {
    async fn select(&self, _ctx: &RoutingContext) -> Result<ProviderSelection> {
        if self.entries.is_empty() {
            anyhow::bail!("WeightedRouter has no entries");
        }

        if self.use_random {
            let mut rng = fastrand::Rng::new();
            let r = rng.u32(0..self.total_weight);
            let mut cum = 0u32;
            for e in &self.entries {
                cum += e.weight;
                if r < cum {
                    return Ok(ProviderSelection {
                        provider: e.provider,
                        model: e.model.clone(),
                    });
                }
            }
            let e = self
                .entries
                .last()
                .context("WeightedRouter entries empty")?;
            Ok(ProviderSelection {
                provider: e.provider,
                model: e.model.clone(),
            })
        } else {
            let round = self.round_robin.fetch_add(1, Ordering::Relaxed);
            let total_weight = usize::try_from(self.total_weight).unwrap_or(usize::MAX);
            let ticket = u32::try_from(round % total_weight).unwrap_or(u32::MAX);
            let mut cumulative = 0u32;
            let mut selected = self
                .entries
                .last()
                .context("WeightedRouter entries empty")?;
            for entry in &self.entries {
                cumulative = cumulative.saturating_add(entry.weight);
                if ticket < cumulative {
                    selected = entry;
                    break;
                }
            }
            Ok(ProviderSelection {
                provider: selected.provider,
                model: selected.model.clone(),
            })
        }
    }
}
