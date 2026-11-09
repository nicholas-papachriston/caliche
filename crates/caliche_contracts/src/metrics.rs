use serde::{Deserialize, Serialize};

/// Harness discipline metrics shared by API responses and SDK bindings.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct HarnessDisciplineMetricsDto {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tokens_per_turn: Option<f64>,
    #[serde(default)]
    pub tool_calls: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_calls_per_turn: Option<f64>,
    #[serde(default)]
    pub compactions: usize,
    #[serde(default)]
    pub readonly_filter_hits: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub estimated_cost_usd: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub summary: Option<String>,
}

impl HarnessDisciplineMetricsDto {
    #[must_use]
    pub fn summary_line(&self) -> String {
        if let Some(summary) = &self.summary {
            return summary.clone();
        }
        let tokens = self
            .tokens_per_turn
            .map_or_else(|| "-".to_string(), |t| format!("{t:.0}"));
        let cost = self
            .estimated_cost_usd
            .map_or_else(|| "-".to_string(), |c| format!("${c:.4}"));
        format!(
            "discipline: tokens/turn={tokens} tools={} compact={} readonly_hits={} cost={cost}",
            self.tool_calls, self.compactions, self.readonly_filter_hits
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn summary_line_prefers_explicit_summary() {
        let dto = HarnessDisciplineMetricsDto {
            summary: Some("custom".to_string()),
            ..HarnessDisciplineMetricsDto::default()
        };
        assert_eq!(dto.summary_line(), "custom");
    }

    #[test]
    fn summary_line_formats_metrics_when_summary_is_absent() {
        let dto = HarnessDisciplineMetricsDto {
            tokens_per_turn: Some(123.4),
            tool_calls: 2,
            compactions: 1,
            readonly_filter_hits: 3,
            estimated_cost_usd: Some(0.01234),
            ..HarnessDisciplineMetricsDto::default()
        };
        let summary = dto.summary_line();
        assert!(summary.contains("tokens/turn=123"));
        assert!(summary.contains("tools=2"));
        assert!(summary.contains("readonly_hits=3"));
        assert!(summary.contains("$0.0123"));
    }
}
