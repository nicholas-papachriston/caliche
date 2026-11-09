//! Harness discipline metrics (tokens/turn, tool calls, compaction, estimated cost).

use crate::llm::ProviderType;
use crate::messages::TokenUsage;
use crate::utils::cost::get_pricing;
use caliche_contracts::metrics::HarnessDisciplineMetricsDto;
use serde::{Deserialize, Serialize};

/// Aggregated cost/latency discipline signals for a turn or multi-turn task.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct HarnessDisciplineMetrics {
    /// Mean tokens per turn when turn count is known.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tokens_per_turn: Option<f64>,
    /// Tool invocations counted for this scope.
    #[serde(default)]
    pub tool_calls: usize,
    /// Mean tool calls per turn when turn count is known.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_calls_per_turn: Option<f64>,
    /// Context-window compaction events in this scope.
    #[serde(default)]
    pub compactions: usize,
    /// Tool calls denied by readonly / allowed-tools filters.
    #[serde(default)]
    pub readonly_filter_hits: usize,
    /// Estimated USD cost from known provider/model pricing tables.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub estimated_cost_usd: Option<f64>,
}

impl HarnessDisciplineMetrics {
    /// Build metrics for a single turn.
    #[must_use]
    pub fn for_turn(
        usage: &TokenUsage,
        tool_calls: usize,
        compactions: usize,
        readonly_filter_hits: usize,
        provider: Option<ProviderType>,
        model: Option<&str>,
    ) -> Self {
        Self {
            tokens_per_turn: usage.total_tokens.map(f64::from),
            tool_calls,
            tool_calls_per_turn: Some(tool_calls as f64),
            compactions,
            readonly_filter_hits,
            estimated_cost_usd: estimate_usage_cost_usd(usage, provider, model),
        }
    }

    /// Build metrics for a multi-turn task/session.
    #[must_use]
    pub fn for_task(
        usage: &TokenUsage,
        turns: usize,
        tool_calls: usize,
        compactions: usize,
        readonly_filter_hits: usize,
        provider: Option<ProviderType>,
        model: Option<&str>,
    ) -> Self {
        let turns_f = if turns == 0 { None } else { Some(turns as f64) };
        Self {
            tokens_per_turn: match (usage.total_tokens, turns_f) {
                (Some(tokens), Some(t)) => Some(f64::from(tokens) / t),
                _ => None,
            },
            tool_calls,
            tool_calls_per_turn: turns_f.map(|t| tool_calls as f64 / t),
            compactions,
            readonly_filter_hits,
            estimated_cost_usd: estimate_usage_cost_usd(usage, provider, model),
        }
    }

    /// One-line summary for REPL/API status output.
    #[must_use]
    pub fn summary_line(&self) -> String {
        let tokens = self
            .tokens_per_turn
            .map(|t| format!("{t:.0}"))
            .unwrap_or_else(|| "-".to_string());
        let cost = self
            .estimated_cost_usd
            .map(|c| format!("${c:.4}"))
            .unwrap_or_else(|| "-".to_string());
        format!(
            "discipline: tokens/turn={tokens} tools={} compact={} readonly_hits={} cost={cost}",
            self.tool_calls, self.compactions, self.readonly_filter_hits
        )
    }
}

impl From<&HarnessDisciplineMetrics> for HarnessDisciplineMetricsDto {
    fn from(metrics: &HarnessDisciplineMetrics) -> Self {
        Self {
            tokens_per_turn: metrics.tokens_per_turn,
            tool_calls: metrics.tool_calls,
            tool_calls_per_turn: metrics.tool_calls_per_turn,
            compactions: metrics.compactions,
            readonly_filter_hits: metrics.readonly_filter_hits,
            estimated_cost_usd: metrics.estimated_cost_usd,
            summary: Some(metrics.summary_line()),
        }
    }
}

/// Estimate USD cost for token usage when pricing is known.
#[must_use]
pub fn estimate_usage_cost_usd(
    usage: &TokenUsage,
    provider: Option<ProviderType>,
    model: Option<&str>,
) -> Option<f64> {
    let provider = provider?;
    let model = model.filter(|m| !m.is_empty())?;
    let pricing = get_pricing(&provider.to_string(), model)?;
    let input = u64::from(usage.prompt_tokens.unwrap_or(0));
    let output = u64::from(usage.completion_tokens.unwrap_or(0));
    if input == 0 && output == 0 {
        return None;
    }
    Some(pricing.calculate(input, output))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn for_turn_sets_tokens_tool_calls_compaction_and_cost_when_priced() {
        let usage = TokenUsage::from_counts(Some(1_000), Some(500), Some(1_500));
        let metrics = HarnessDisciplineMetrics::for_turn(
            &usage,
            2,
            1,
            3,
            Some(ProviderType::OpenAI),
            Some("gpt-4o"),
        );
        assert_eq!(metrics.tokens_per_turn, Some(1_500.0));
        assert_eq!(metrics.tool_calls, 2);
        assert_eq!(metrics.tool_calls_per_turn, Some(2.0));
        assert_eq!(metrics.compactions, 1);
        assert_eq!(metrics.readonly_filter_hits, 3);
        assert!(metrics.estimated_cost_usd.is_some_and(|c| c > 0.0));
        assert!(metrics.summary_line().contains("compact=1"));
        assert!(metrics.summary_line().contains("readonly_hits=3"));
    }

    #[test]
    fn for_task_averages_tokens_across_turns() {
        let usage = TokenUsage::from_counts(Some(200), Some(100), Some(300));
        let metrics = HarnessDisciplineMetrics::for_task(
            &usage,
            3,
            6,
            2,
            1,
            Some(ProviderType::OpenAI),
            Some("gpt-4o"),
        );
        assert_eq!(metrics.tokens_per_turn, Some(100.0));
        assert_eq!(metrics.tool_calls_per_turn, Some(2.0));
        assert_eq!(metrics.compactions, 2);
        assert_eq!(metrics.readonly_filter_hits, 1);
    }

    #[test]
    fn unknown_model_omits_cost() {
        let usage = TokenUsage::from_counts(Some(10), Some(10), Some(20));
        let metrics = HarnessDisciplineMetrics::for_turn(
            &usage,
            0,
            0,
            0,
            Some(ProviderType::OpenAI),
            Some("nope"),
        );
        assert!(metrics.estimated_cost_usd.is_none());
    }
}
