use crate::agent::tools::metadata::{ToolCategory, ToolMetadata, ToolRelevanceScore};
use crate::agent::tools::{filtering, scoring};
use std::collections::HashMap;

/// Context for tool selection decisions
#[derive(Debug, Clone)]
pub struct ToolSelectionContext {
    /// The user's query or task description
    pub query: String,
    /// Available tools with metadata
    pub available_tools: Vec<ToolMetadata>,
    /// Historical execution records
    pub execution_history: Vec<ToolExecutionRecord>,
    /// Type of task being performed
    pub task_type: TaskType,
    /// Selection constraints
    pub constraints: SelectionConstraints,
}

/// Type of task being performed
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TaskType {
    /// Information gathering or research
    Research,
    /// Modifying or creating files/data
    Modification,
    /// Analyzing or processing data
    Analysis,
    /// General-purpose multi-step task
    General,
    /// Quick utility operation
    Utility,
}

/// Constraints for tool selection
#[derive(Debug, Clone)]
pub struct SelectionConstraints {
    /// Maximum number of tools to select
    pub max_tools: usize,
    /// Maximum total cost budget
    pub max_cost_budget: u32,
    /// Exclude tools with side effects
    pub avoid_side_effects: bool,
    /// Prefer high-confidence tools
    pub prefer_high_confidence: bool,
    /// Required categories
    pub required_categories: Vec<ToolCategory>,
    /// Excluded tool names
    pub excluded_tools: Vec<String>,
}

impl Default for SelectionConstraints {
    fn default() -> Self {
        Self {
            max_tools: 10,
            max_cost_budget: 500,
            avoid_side_effects: false,
            prefer_high_confidence: false,
            required_categories: Vec::new(),
            excluded_tools: Vec::new(),
        }
    }
}

/// Record of a tool execution
#[derive(Debug, Clone)]
pub struct ToolExecutionRecord {
    pub tool_name: String,
    pub arguments: serde_json::Value,
    pub success: bool,
    pub execution_time_ms: u64,
    pub output_length: usize,
    pub error_type: Option<String>,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

impl ToolExecutionRecord {
    pub fn new(tool_name: impl Into<String>, arguments: serde_json::Value, success: bool) -> Self {
        Self {
            tool_name: tool_name.into(),
            arguments,
            success,
            execution_time_ms: 0,
            output_length: 0,
            error_type: None,
            timestamp: chrono::Utc::now(),
        }
    }

    #[must_use]
    pub const fn with_timing(mut self, execution_time_ms: u64) -> Self {
        self.execution_time_ms = execution_time_ms;
        self
    }

    #[must_use]
    pub const fn with_output_size(mut self, size: usize) -> Self {
        self.output_length = size;
        self
    }

    pub fn with_error(mut self, error_type: impl Into<String>) -> Self {
        self.error_type = Some(error_type.into());
        self
    }
}

/// Result of tool selection
#[derive(Debug, Clone)]
pub struct ToolSelectionResult {
    /// Selected tool names in recommended order
    pub selected_tools: Vec<String>,
    /// Relevance scores for each tool
    pub scores: HashMap<String, ToolRelevanceScore>,
    /// Explanation of selection
    pub explanation: String,
    /// Total estimated cost
    pub estimated_cost: u32,
}

impl Default for ToolSelectionResult {
    fn default() -> Self {
        Self::new()
    }
}

impl ToolSelectionResult {
    #[must_use]
    pub fn new() -> Self {
        Self {
            selected_tools: Vec::new(),
            scores: HashMap::new(),
            explanation: String::new(),
            estimated_cost: 0,
        }
    }

    #[must_use]
    pub fn with_tools(mut self, tools: Vec<String>) -> Self {
        self.selected_tools = tools;
        self
    }

    pub fn with_explanation(mut self, explanation: impl Into<String>) -> Self {
        self.explanation = explanation.into();
        self
    }

    #[must_use]
    pub fn add_score(mut self, score: ToolRelevanceScore) -> Self {
        self.scores.insert(score.tool_name.clone(), score);
        self
    }
}

/// Trait for tool selection strategies
pub trait ToolSelector: Send + Sync {
    /// Select tools based on the given context
    fn select(&self, context: &ToolSelectionContext) -> ToolSelectionResult;

    /// Get the name of this selector strategy
    fn name(&self) -> &str;
}

/// Simple selector that returns all available tools (current behavior)
#[derive(Debug)]
pub struct SimpleSelector;

impl ToolSelector for SimpleSelector {
    fn select(&self, context: &ToolSelectionContext) -> ToolSelectionResult {
        let tool_names: Vec<String> = context
            .available_tools
            .iter()
            .filter(|t| !context.constraints.excluded_tools.contains(&t.name))
            .map(|t| t.name.clone())
            .collect();

        ToolSelectionResult::new()
            .with_tools(tool_names)
            .with_explanation("Using simple selector: all available tools")
    }

    fn name(&self) -> &'static str {
        "simple"
    }
}

/// Relevance-based selector that scores tools by keyword matching
#[derive(Debug)]
pub struct RelevanceSelector;

impl ToolSelector for RelevanceSelector {
    fn select(&self, context: &ToolSelectionContext) -> ToolSelectionResult {
        let mut result = ToolSelectionResult::new();
        let mut scores: Vec<(ToolRelevanceScore, &ToolMetadata)> = Vec::new();

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

            scores.push((score, tool));
        }

        // Sort by combined score (descending)
        scores.sort_by(|a, b| {
            b.0.combined_score
                .partial_cmp(&a.0.combined_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Apply max_tools constraint
        let selected_count = scores.len().min(context.constraints.max_tools);

        for (score, tool) in scores.iter().take(selected_count) {
            result.selected_tools.push(tool.name.clone());
            result.scores.insert(tool.name.clone(), score.clone());
            result.estimated_cost += tool.cost.score();
        }

        let selected_count_val = result.selected_tools.len();
        result.with_explanation(format!(
            "Selected {selected_count_val} tools using relevance scoring (keyword match, success rate, recency)"
        ))
    }

    fn name(&self) -> &'static str {
        "relevance"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::tools::scoring;

    #[test]
    fn test_selection_constraints_default() {
        let constraints = SelectionConstraints::default();
        assert_eq!(constraints.max_tools, 10);
        assert_eq!(constraints.max_cost_budget, 500);
        assert!(!constraints.avoid_side_effects);
    }

    #[test]
    fn test_tool_execution_record() {
        let record = ToolExecutionRecord::new("test_tool", serde_json::json!({}), true)
            .with_timing(100)
            .with_output_size(500);

        assert_eq!(record.tool_name, "test_tool");
        assert!(record.success);
        assert_eq!(record.execution_time_ms, 100);
        assert_eq!(record.output_length, 500);
    }

    #[test]
    fn test_relevance_selector_keyword_match() {
        let metadata = ToolMetadata::new(
            "fs",
            "Filesystem operations (read/write/edit/delete)",
            ToolCategory::Information,
        )
        .with_keywords(vec!["read", "file", "content"]);

        let score = scoring::calculate_keyword_match("read a file", &metadata);
        assert!(score > 0.0);
        assert!(score <= 1.0);
    }

    #[test]
    fn test_relevance_selector_success_rate() {
        let history = vec![
            ToolExecutionRecord::new("tool1", serde_json::json!({}), true),
            ToolExecutionRecord::new("tool1", serde_json::json!({}), true),
            ToolExecutionRecord::new("tool1", serde_json::json!({}), false),
        ];

        let rate = scoring::calculate_success_rate("tool1", &history);
        assert!((rate - 2.0 / 3.0).abs() < 0.01);
    }

    #[test]
    fn test_relevance_selector_new_tool() {
        let rate = scoring::calculate_success_rate("unknown_tool", &[]);
        assert_eq!(rate, 0.5);
    }
}
