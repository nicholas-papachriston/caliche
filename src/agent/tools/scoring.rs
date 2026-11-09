use crate::agent::tools::metadata::ToolMetadata;
use crate::agent::tools::selection::ToolExecutionRecord;

#[derive(Debug, Clone, Copy)]
pub struct ScoreWeights {
    pub keyword: f32,
    pub success_rate: f32,
    pub recency: f32,
}

impl Default for ScoreWeights {
    fn default() -> Self {
        Self {
            keyword: 0.5,
            success_rate: 0.3,
            recency: 0.2,
        }
    }
}

#[must_use]
pub fn calculate_keyword_match(query: &str, metadata: &ToolMetadata) -> f32 {
    let query_lower = query.to_lowercase();
    let description_lower = metadata.description.to_lowercase();

    let mut score = 0.0f32;

    for keyword in &metadata.keywords {
        if query_lower.contains(&keyword.to_lowercase()) {
            score += 0.3;
        }
    }

    for word in query_lower.split_whitespace() {
        if word.len() > 3 && description_lower.contains(word) {
            score += 0.1;
        }
    }

    if metadata.is_common {
        score += 0.1;
    }

    (score / 5.0).min(1.0)
}

#[must_use]
pub fn calculate_success_rate(tool_name: &str, history: &[ToolExecutionRecord]) -> f32 {
    let mut total = 0u32;
    let mut success = 0u32;

    for r in history {
        if r.tool_name == tool_name {
            total += 1;
            if r.success {
                success += 1;
            }
        }
    }

    if total == 0 {
        return 0.5;
    }

    let rate = (success as f32) / (total as f32);
    rate.max(0.1)
}

#[must_use]
pub fn calculate_recency_factor(tool_name: &str, history: &[ToolExecutionRecord]) -> f32 {
    let mut newest: Option<&ToolExecutionRecord> = None;
    for r in history {
        if r.tool_name == tool_name {
            newest = match newest {
                None => Some(r),
                Some(prev) => {
                    if r.timestamp > prev.timestamp {
                        Some(r)
                    } else {
                        Some(prev)
                    }
                }
            };
        }
    }

    let Some(record) = newest else {
        return 0.5;
    };

    let age_hours = chrono::Utc::now()
        .signed_duration_since(record.timestamp)
        .num_hours();

    let decay = 1.0 / (1.0 + (age_hours as f32 / 24.0));
    (decay * 0.5 + 0.5).min(1.0)
}

#[must_use]
pub fn combine_scores(
    keyword: f32,
    success_rate: f32,
    recency: f32,
    weights: &ScoreWeights,
) -> f32 {
    let total_weight = weights.keyword + weights.success_rate + weights.recency;
    if total_weight <= f32::EPSILON {
        return keyword;
    }

    (keyword.mul_add(weights.keyword, success_rate * weights.success_rate)
        + recency * weights.recency)
        / total_weight
}
