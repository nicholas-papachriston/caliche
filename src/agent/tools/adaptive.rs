use crate::agent::tools::metadata::{ToolMetadata, ToolRelevanceScore};
use crate::agent::tools::selection::{
    ToolExecutionRecord, ToolSelectionContext, ToolSelectionResult, ToolSelector,
};
use crate::agent::tools::{filtering, scoring};
use std::collections::HashMap;

/// Adaptive selector that combines relevance scoring with execution-history feedback.
///
/// This is intentionally lightweight and in-memory: it consumes `execution_history` from the
/// provided `ToolSelectionContext` (callers can persist/aggregate however they want).
#[derive(Debug, Clone)]
pub struct AdaptiveSelector {
    /// Weight for keyword relevance (0.0-1.0)
    pub weight_keyword: f32,
    /// Weight for historical success rate (0.0-1.0)
    pub weight_success_rate: f32,
    /// Weight for recency (0.0-1.0)
    pub weight_recency: f32,
}

impl Default for AdaptiveSelector {
    fn default() -> Self {
        Self {
            weight_keyword: 0.5,
            weight_success_rate: 0.3,
            weight_recency: 0.2,
        }
    }
}

impl AdaptiveSelector {
    const fn score_weights(&self) -> scoring::ScoreWeights {
        scoring::ScoreWeights {
            keyword: self.weight_keyword,
            success_rate: self.weight_success_rate,
            recency: self.weight_recency,
        }
    }
}

impl ToolSelector for AdaptiveSelector {
    fn select(&self, context: &ToolSelectionContext) -> ToolSelectionResult {
        let mut result = ToolSelectionResult::new();
        let mut scored: Vec<(f32, ToolRelevanceScore, &ToolMetadata)> = Vec::new();
        let weights = self.score_weights();

        for tool in &context.available_tools {
            if !filtering::passes_constraints(tool, &context.constraints) {
                continue;
            }

            let mut score = ToolRelevanceScore::new(&tool.name);
            score.keyword_match = scoring::calculate_keyword_match(&context.query, tool);
            score.success_rate_factor =
                scoring::calculate_success_rate(&tool.name, &context.execution_history);
            score.recency_factor =
                scoring::calculate_recency_factor(&tool.name, &context.execution_history);
            score.calculate();

            let combined = scoring::combine_scores(
                score.keyword_match,
                score.success_rate_factor,
                score.recency_factor,
                &weights,
            );
            scored.push((combined, score, tool));
        }

        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

        let selected_count = scored.len().min(context.constraints.max_tools);
        for (_combined, score, tool) in scored.into_iter().take(selected_count) {
            result.selected_tools.push(tool.name.clone());
            result.scores.insert(tool.name.clone(), score.clone());
            result.estimated_cost += tool.cost.score();
        }

        // Keep the explanation small and user-facing.
        let selected_count_val = result.selected_tools.len();
        result.explanation = format!(
            "Selected {selected_count_val} tools using adaptive scoring (relevance + success + recency)"
        );

        result
    }

    fn name(&self) -> &'static str {
        "adaptive"
    }
}

/// Convenience helper: convert metrics from an execution history into a per-tool lookup.
#[must_use]
pub fn execution_metrics_from_history(
    history: &[ToolExecutionRecord],
) -> HashMap<String, (u32, u32)> {
    let mut map: HashMap<String, (u32, u32)> = HashMap::new();
    for r in history {
        let entry = map.entry(r.tool_name.clone()).or_insert((0, 0));
        entry.0 += 1;
        if r.success {
            entry.1 += 1;
        }
    }
    map
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::tools::metadata::{ToolCategory, ToolDependencies, ToolMetadata};
    use crate::agent::tools::selection::SelectionConstraints;

    #[test]
    fn adaptive_selector_runs() {
        let selector = AdaptiveSelector::default();
        let ctx = ToolSelectionContext {
            query: "read file".to_string(),
            available_tools: vec![ToolMetadata {
                name: "fs".to_string(),
                description: "Filesystem operations".to_string(),
                category: ToolCategory::Information,
                cost: crate::agent::tools::metadata::ToolCost::Low,
                keywords: vec!["read".to_string(), "file".to_string()],
                dependencies: ToolDependencies::default(),
                is_common: true,
                tags: Vec::new(),
            }],
            execution_history: Vec::new(),
            task_type: crate::agent::tools::selection::TaskType::Research,
            constraints: SelectionConstraints::default(),
        };

        let result = selector.select(&ctx);
        assert_eq!(result.selected_tools, vec!["fs".to_string()]);
    }
}
