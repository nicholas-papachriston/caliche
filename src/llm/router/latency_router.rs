//! Latency-based router.
//!
//! Selects the (provider, model) pair with the lowest observed EWMA latency.
//! Caller code is expected to feed observed per-request latencies back in via
//! [`LatencyRouter::record_latency`] so the next selection reflects reality.
//!
//! Routing policy:
//! 1. If the request has a `requested_provider`/`requested_model`, honor it.
//!    The caller asked for a specific endpoint so the router is a no-op.
//! 2. Otherwise, walk the configured candidate set and pick the candidate
//!    with the lowest EWMA latency. Any candidate without a recorded latency
//!    yet is treated as "unknown / try me first" so new providers get a slot
//!    before the EWMA converges.
//! 3. Ties (equal best latency) fall back to round-robin across the tied
//!    candidates; this keeps traffic from pinning to the first option when
//!    multiple backends are equivalent.

use super::{ProviderRouter, ProviderSelection, RoutingContext};
use crate::llm::ProviderType;
use anyhow::{Context, Result};
use async_trait::async_trait;
use std::collections::HashMap;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

/// One candidate backend the router can choose from.
#[derive(Debug, Clone)]
pub struct LatencyCandidate {
    pub provider: ProviderType,
    pub model: String,
}

/// Stores per-candidate EWMA latency in milliseconds.
#[derive(Debug, Default)]
struct LatencyState {
    ewma_ms: HashMap<(ProviderType, String), f64>,
}

/// Latency-aware router.
#[derive(Debug)]
pub struct LatencyRouter {
    candidates: Vec<LatencyCandidate>,
    state: Mutex<LatencyState>,
    rr: AtomicUsize,
    /// EWMA alpha (0.0 = no new signal, 1.0 = only last observation).
    alpha: f64,
}

impl LatencyRouter {
    /// Build a latency router over a fixed set of candidates.
    ///
    /// `alpha` is the EWMA smoothing factor in [0.0, 1.0]. A value around
    /// `0.2` works well for noisy per-request latency.
    /// # Errors
    ///
    /// Returns an error if the language-model operation cannot be completed.
    pub fn new(candidates: Vec<LatencyCandidate>, alpha: f64) -> Result<Self> {
        if candidates.is_empty() {
            anyhow::bail!("LatencyRouter requires at least one candidate");
        }
        if !(0.0..=1.0).contains(&alpha) {
            anyhow::bail!("LatencyRouter alpha must be in [0.0, 1.0]; got {alpha}");
        }
        Ok(Self {
            candidates,
            state: Mutex::new(LatencyState::default()),
            rr: AtomicUsize::new(0),
            alpha,
        })
    }

    /// Parse candidates from an env-style list: `openai:gpt-4o-mini,anthropic:claude-3-haiku`.
    /// # Errors
    ///
    /// Returns an error if the language-model operation cannot be completed.
    pub fn from_env(env_value: &str, alpha: f64) -> Result<Self> {
        let mut candidates = Vec::new();
        for raw in env_value.split(',') {
            let raw = raw.trim();
            if raw.is_empty() {
                continue;
            }
            let (provider_s, model) = raw.split_once(':').with_context(|| {
                format!("Invalid latency candidate '{raw}': expected provider:model")
            })?;
            let provider: ProviderType = provider_s
                .parse()
                .map_err(|e: String| anyhow::anyhow!("Invalid provider '{provider_s}': {e}"))?;
            candidates.push(LatencyCandidate {
                provider,
                model: model.to_string(),
            });
        }
        Self::new(candidates, alpha)
    }

    /// Feed a newly-observed latency (milliseconds) for a candidate.
    pub fn record_latency(&self, provider: ProviderType, model: &str, observed_ms: f64) {
        if observed_ms.is_nan() || observed_ms < 0.0 {
            return;
        }
        let mut state = match self.state.lock() {
            Ok(s) => s,
            Err(poisoned) => poisoned.into_inner(),
        };
        let key = (provider, model.to_string());
        let entry = state.ewma_ms.entry(key).or_insert(observed_ms);
        *entry = (1.0 - self.alpha).mul_add(*entry, self.alpha * observed_ms);
        drop(state);
    }

    /// Pick a candidate using the current latency state. Exposed for tests.
    pub fn choose(&self) -> ProviderSelection {
        let state = match self.state.lock() {
            Ok(s) => s,
            Err(poisoned) => poisoned.into_inner(),
        };

        let mut unknown: Vec<&LatencyCandidate> = Vec::new();
        let mut best_latency: Option<f64> = None;
        let mut best: Vec<&LatencyCandidate> = Vec::new();

        for candidate in &self.candidates {
            let key = (candidate.provider, candidate.model.clone());
            match state.ewma_ms.get(&key) {
                Some(latency) => match best_latency {
                    Some(b) if *latency < b - f64::EPSILON => {
                        best_latency = Some(*latency);
                        best.clear();
                        best.push(candidate);
                    }
                    Some(b) if (*latency - b).abs() <= f64::EPSILON => best.push(candidate),
                    None => {
                        best_latency = Some(*latency);
                        best.push(candidate);
                    }
                    _ => {}
                },
                None => unknown.push(candidate),
            }
        }

        let pool: &[&LatencyCandidate] = if unknown.is_empty() { &best } else { &unknown };
        let idx = self.rr.fetch_add(1, Ordering::Relaxed) % pool.len().max(1);
        let chosen = pool[idx % pool.len()];
        ProviderSelection {
            provider: chosen.provider,
            model: chosen.model.clone(),
        }
    }
}

#[async_trait]
impl ProviderRouter for LatencyRouter {
    async fn select(&self, ctx: &RoutingContext) -> Result<ProviderSelection> {
        if let (Some(provider), Some(model)) = (ctx.requested_provider, &ctx.requested_model) {
            return Ok(ProviderSelection {
                provider,
                model: model.clone(),
            });
        }
        Ok(self.choose())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn c(provider: ProviderType, model: &str) -> LatencyCandidate {
        LatencyCandidate {
            provider,
            model: model.to_string(),
        }
    }

    #[test]
    fn rejects_empty_and_out_of_range_alpha() {
        assert!(LatencyRouter::new(Vec::new(), 0.2).is_err());
        assert!(LatencyRouter::new(vec![c(ProviderType::OpenAI, "m")], -0.1).is_err());
        assert!(LatencyRouter::new(vec![c(ProviderType::OpenAI, "m")], 1.5).is_err());
    }

    #[test]
    fn unknown_candidates_tried_first() {
        let r = LatencyRouter::new(
            vec![
                c(ProviderType::OpenAI, "gpt-a"),
                c(ProviderType::Anthropic, "claude-b"),
            ],
            0.2,
        )
        .expect("router");
        r.record_latency(ProviderType::OpenAI, "gpt-a", 100.0);
        let pick = r.choose();
        assert_eq!(pick.provider, ProviderType::Anthropic);
    }

    #[test]
    fn prefers_lower_latency() {
        let r = LatencyRouter::new(
            vec![
                c(ProviderType::OpenAI, "gpt-a"),
                c(ProviderType::Anthropic, "claude-b"),
            ],
            0.2,
        )
        .expect("router");
        r.record_latency(ProviderType::OpenAI, "gpt-a", 300.0);
        r.record_latency(ProviderType::Anthropic, "claude-b", 100.0);
        let pick = r.choose();
        assert_eq!(pick.provider, ProviderType::Anthropic);
    }
}
