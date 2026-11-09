use crate::agent::tools::scoring::{self, ScoreWeights};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;

/// Categorizes tools by their primary function
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ToolCategory {
    /// Tools that retrieve or search information (search, read, query, fetch)
    Information,
    /// Tools that modify state or execute code (write, edit, bash, execute)
    Execution,
    /// Tools that process and transform data (parse, transform, validate)
    Analysis,
    /// Tools for multi-agent coordination and handoffs
    Coordination,
    /// Utility tools (calculator, etc.)
    Utility,
}

impl ToolCategory {
    #[must_use]
    pub const fn as_str(&self) -> &str {
        match self {
            Self::Information => "information",
            Self::Execution => "execution",
            Self::Analysis => "analysis",
            Self::Coordination => "coordination",
            Self::Utility => "utility",
        }
    }
}

impl std::str::FromStr for ToolCategory {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "information" => Ok(Self::Information),
            "execution" => Ok(Self::Execution),
            "analysis" => Ok(Self::Analysis),
            "coordination" => Ok(Self::Coordination),
            "utility" => Ok(Self::Utility),
            _ => Err(format!("Invalid tool category: {s}")),
        }
    }
}

/// Represents the computational or API cost of executing a tool
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub enum ToolCost {
    /// Local operations with no external calls (calculator)
    Free,
    /// Simple API calls or lightweight operations
    Low,
    /// Moderate computation or external service calls
    Medium,
    /// Expensive operations (heavy computation, batch processing, expensive APIs)
    High,
}

impl ToolCost {
    #[must_use]
    pub const fn as_str(&self) -> &str {
        match self {
            Self::Free => "free",
            Self::Low => "low",
            Self::Medium => "medium",
            Self::High => "high",
        }
    }

    /// Returns a numeric cost score for optimization
    #[must_use]
    pub const fn score(&self) -> u32 {
        match self {
            Self::Free => 1,
            Self::Low => 5,
            Self::Medium => 15,
            Self::High => 50,
        }
    }
}

impl std::str::FromStr for ToolCost {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "free" => Ok(Self::Free),
            "low" => Ok(Self::Low),
            "medium" => Ok(Self::Medium),
            "high" => Ok(Self::High),
            _ => Err(format!("Invalid tool cost: {s}")),
        }
    }
}

/// Describes dependencies and constraints for a tool
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ToolDependencies {
    /// Whether the tool requires specific context to function
    pub requires_context: bool,
    /// Names of tools that cannot be used together with this tool
    pub mutually_exclusive: HashSet<String>,
    /// Names of tools that should be executed before this tool
    pub requires_prior_tools: HashSet<String>,
    /// Whether executing this tool has side effects (modifies state)
    pub produces_side_effects: bool,
}

impl ToolDependencies {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    #[must_use]
    pub const fn with_context(mut self) -> Self {
        self.requires_context = true;
        self
    }

    #[must_use]
    pub const fn with_side_effects(mut self) -> Self {
        self.produces_side_effects = true;
        self
    }

    #[must_use]
    pub fn mutually_exclusive_with(mut self, tools: Vec<&str>) -> Self {
        self.mutually_exclusive = tools.iter().map(std::string::ToString::to_string).collect();
        self
    }

    #[must_use]
    pub fn requires_prior(mut self, tools: Vec<&str>) -> Self {
        self.requires_prior_tools = tools.iter().map(std::string::ToString::to_string).collect();
        self
    }
}

/// Metadata that enhances a tool with additional intelligence
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolMetadata {
    /// Unique tool identifier
    pub name: String,
    /// Human-readable description
    pub description: String,
    /// Primary category of the tool
    pub category: ToolCategory,
    /// Computational/API cost of execution
    pub cost: ToolCost,
    /// Keywords for relevance matching
    pub keywords: Vec<String>,
    /// Dependencies and constraints
    pub dependencies: ToolDependencies,
    /// Whether this tool is commonly used (for prioritization)
    pub is_common: bool,
    /// Tags for filtering and grouping
    pub tags: Vec<String>,
}

impl ToolMetadata {
    pub fn new(
        name: impl Into<String>,
        description: impl Into<String>,
        category: ToolCategory,
    ) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            category,
            cost: ToolCost::Low,
            keywords: Vec::new(),
            dependencies: ToolDependencies::default(),
            is_common: false,
            tags: Vec::new(),
        }
    }

    #[must_use]
    pub const fn with_cost(mut self, cost: ToolCost) -> Self {
        self.cost = cost;
        self
    }

    #[must_use]
    pub fn with_keywords(mut self, keywords: Vec<&str>) -> Self {
        self.keywords = keywords
            .iter()
            .map(std::string::ToString::to_string)
            .collect();
        self
    }

    #[must_use]
    pub fn with_dependencies(mut self, deps: ToolDependencies) -> Self {
        self.dependencies = deps;
        self
    }

    #[must_use]
    pub const fn common(mut self) -> Self {
        self.is_common = true;
        self
    }

    #[must_use]
    pub fn with_tags(mut self, tags: Vec<&str>) -> Self {
        self.tags = tags.iter().map(std::string::ToString::to_string).collect();
        self
    }
}

/// Scoring result for tool relevance to a query
#[derive(Debug, Clone)]
pub struct ToolRelevanceScore {
    pub tool_name: String,
    /// Match score based on keywords and description (0.0-1.0)
    pub keyword_match: f32,
    /// Historical success rate (0.0-1.0)
    pub success_rate_factor: f32,
    /// Recency factor - higher for recently successful tools (0.0-1.0)
    pub recency_factor: f32,
    /// Combined relevance score (0.0-1.0)
    pub combined_score: f32,
}

impl ToolRelevanceScore {
    pub fn new(tool_name: impl Into<String>) -> Self {
        Self {
            tool_name: tool_name.into(),
            keyword_match: 0.0,
            success_rate_factor: 0.5,
            recency_factor: 0.5,
            combined_score: 0.0,
        }
    }

    /// Calculate combined score from components
    /// Weights: `keyword_match` (50%), `success_rate` (30%), recency (20%)
    pub fn calculate(&mut self) {
        self.calculate_with_weights(&ScoreWeights::default());
    }

    pub fn calculate_with_weights(&mut self, weights: &ScoreWeights) {
        self.combined_score = scoring::combine_scores(
            self.keyword_match,
            self.success_rate_factor,
            self.recency_factor,
            weights,
        );
    }
}

fn infer_category_from_name(name: &str) -> ToolCategory {
    let n = name.to_lowercase();
    if n.contains("read")
        || n.contains("list")
        || n.contains("search")
        || n.contains("grep")
        || n.contains("semantic")
        || n.contains("workspace")
        || n.contains("context_get")
    {
        ToolCategory::Information
    } else if n.contains("write")
        || n.contains("edit")
        || n.contains("bash")
        || n.contains("execute")
        || n.contains("sql_execution")
        || n.contains("git_commit")
    {
        ToolCategory::Execution
    } else if n.contains("sql_generation")
        || n.contains("data")
        || n.contains("rag")
        || n.contains("analy")
    {
        ToolCategory::Analysis
    } else if n.contains("handoff") || n.contains("context_set") {
        ToolCategory::Coordination
    } else {
        ToolCategory::Utility
    }
}

/// Heuristic category inference for tool names.
#[must_use]
pub fn inferred_category(name: &str) -> ToolCategory {
    infer_category_from_name(name)
}

/// Heuristic side-effect detection for tool names.
#[must_use]
pub fn is_likely_side_effecting_tool(name: &str) -> bool {
    matches!(infer_category_from_name(name), ToolCategory::Execution)
}

fn infer_cost_from_name(name: &str) -> ToolCost {
    let n = name.to_lowercase();
    if n.contains("calculator") {
        ToolCost::Free
    } else if n.contains("read") || n.contains("list") {
        ToolCost::Low
    } else if n.contains("grep") || n.contains("sql") || n.contains("bash") || n.contains("http") {
        ToolCost::Medium
    } else if n.contains("rag") || n.contains("document") {
        ToolCost::High
    } else {
        ToolCost::Low
    }
}

fn extract_keywords(name: &str, description: &str) -> Vec<String> {
    let mut out = Vec::new();
    for token in name
        .split(|c: char| !c.is_ascii_alphanumeric())
        .chain(description.split_whitespace())
    {
        let t = token.trim().to_lowercase();
        if t.len() >= 3 && t.len() <= 24 {
            out.push(t);
        }
        if out.len() >= 12 {
            break;
        }
    }
    out
}

/// Create `ToolMetadata` from a `ToolSchema` (heuristic category/cost/keywords).
#[must_use]
pub fn metadata_from_schema(schema: &crate::messages::ToolSchema) -> ToolMetadata {
    let category = infer_category_from_name(schema.name.as_ref());
    let cost = infer_cost_from_name(schema.name.as_ref());
    let keywords = extract_keywords(schema.name.as_ref(), schema.description.as_ref());

    ToolMetadata::new(schema.name.as_ref(), schema.description.as_ref(), category)
        .with_cost(cost)
        .with_keywords(keywords.iter().map(std::string::String::as_str).collect())
}

/// Reorder tool schemas so the most relevant tools appear first.
///
/// This does **not** remove tools; it only changes ordering (low-risk integration).
/// Uses index-based reordering and returns the original Arc when order is unchanged.
#[must_use]
pub fn order_tool_schemas_by_relevance(
    schemas: &std::sync::Arc<Vec<crate::messages::ToolSchema>>,
    query: &str,
) -> std::sync::Arc<Vec<crate::messages::ToolSchema>> {
    if query.trim().is_empty() || schemas.is_empty() {
        return std::sync::Arc::clone(schemas);
    }

    let mut indices: Vec<(f32, usize)> = Vec::with_capacity(schemas.len());
    for (i, schema) in schemas.iter().enumerate() {
        let meta = metadata_from_schema(schema);
        let score = scoring::calculate_keyword_match(query, &meta);
        indices.push((score, i));
    }

    indices.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

    // Return original Arc when order is unchanged (avoids allocation).
    let is_same_order = indices
        .iter()
        .enumerate()
        .all(|(pos, (_, idx))| pos == *idx);
    if is_same_order {
        return std::sync::Arc::clone(schemas);
    }

    let out: Vec<_> = indices
        .iter()
        .map(|(_, idx)| schemas[*idx].clone())
        .collect();
    std::sync::Arc::new(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tool_category_conversion() {
        assert_eq!(ToolCategory::Information.as_str(), "information");
    }

    #[test]
    fn test_tool_cost_ordering() {
        assert!(ToolCost::Free < ToolCost::Low);
        assert!(ToolCost::Low < ToolCost::Medium);
        assert!(ToolCost::Medium < ToolCost::High);
    }

    #[test]
    fn test_tool_cost_scoring() {
        assert_eq!(ToolCost::Free.score(), 1);
        assert_eq!(ToolCost::High.score(), 50);
    }

    #[test]
    fn test_tool_metadata_builder() {
        let meta = ToolMetadata::new("test_tool", "A test tool", ToolCategory::Utility)
            .with_cost(ToolCost::Medium)
            .with_keywords(vec!["test", "example"])
            .common();

        assert_eq!(meta.name, "test_tool");
        assert_eq!(meta.cost, ToolCost::Medium);
        assert_eq!(meta.keywords.len(), 2);
        assert!(meta.is_common);
    }

    #[test]
    fn test_relevance_score_calculation() {
        let mut score = ToolRelevanceScore::new("test_tool");
        score.keyword_match = 0.8;
        score.success_rate_factor = 0.9;
        score.recency_factor = 0.7;
        score.calculate();

        let expected = (0.8 * 0.5) + (0.9 * 0.3) + (0.7 * 0.2);
        assert!((score.combined_score - expected).abs() < 0.0001);
    }
}
