//! Rank harness-improvement suggestions by weakness (Bennett proxy).
//!
//! Used by the trace-driven harness improvement loop: prefer changes that cover
//! more failure *classes* with fewer task-specific bindings, rather than the
//! shortest or most specific patch.

use super::weakness::{count_specificity, weakness_score};

/// A proposed harness change distilled from failure-trace analysis.
#[derive(Debug, Clone, PartialEq)]
pub struct HarnessImprovementSuggestion {
    pub id: String,
    pub summary: String,
    /// Distinct failure classes this change is expected to address.
    pub failure_classes_covered: u32,
    /// Optional explicit specificity override. When `None`, derived from `summary`.
    pub specificity: Option<u32>,
    /// Estimated risk of regressing previously passing tasks in `[0.0, 1.0]`.
    pub regression_risk: f64,
}

/// A suggestion paired with its weakness-derived rank score.
#[derive(Debug, Clone, PartialEq)]
pub struct RankedHarnessSuggestion {
    pub suggestion: HarnessImprovementSuggestion,
    /// Higher is better. Incorporates weakness and inverse regression risk.
    pub score: f64,
    pub specificity: u32,
    pub weakness: f64,
}

/// Rank suggestions by weakness maximisation with a regression penalty.
///
/// Score: `weakness * (1 - regression_risk)` where
/// `weakness = failure_classes_covered / (1 + specificity)`.
#[must_use]
pub fn rank_by_weakness(
    suggestions: Vec<HarnessImprovementSuggestion>,
) -> Vec<RankedHarnessSuggestion> {
    let mut ranked: Vec<RankedHarnessSuggestion> = suggestions
        .into_iter()
        .map(|suggestion| {
            let specificity = suggestion
                .specificity
                .unwrap_or_else(|| count_specificity(&suggestion.summary));
            let weakness = weakness_score(suggestion.failure_classes_covered, 0, specificity);
            let risk = suggestion.regression_risk.clamp(0.0, 1.0);
            let score = weakness * (1.0 - risk);
            RankedHarnessSuggestion {
                suggestion,
                score,
                specificity,
                weakness,
            }
        })
        .collect();

    ranked.sort_by(|left, right| {
        right
            .score
            .partial_cmp(&left.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| left.specificity.cmp(&right.specificity))
            .then_with(|| left.suggestion.id.cmp(&right.suggestion.id))
    });
    ranked
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prefers_broad_low_specificity_changes() {
        let ranked = rank_by_weakness(vec![
            HarnessImprovementSuggestion {
                id: "narrow".into(),
                summary: "Fix TerminalBench task /tmp/tb-042 only".into(),
                failure_classes_covered: 1,
                specificity: Some(8),
                regression_risk: 0.1,
            },
            HarnessImprovementSuggestion {
                id: "broad".into(),
                summary: "Require verification before claiming completion".into(),
                failure_classes_covered: 4,
                specificity: Some(0),
                regression_risk: 0.1,
            },
        ]);
        assert_eq!(ranked[0].suggestion.id, "broad");
        assert!(ranked[0].score > ranked[1].score);
    }

    #[test]
    fn penalises_regression_risk() {
        let ranked = rank_by_weakness(vec![
            HarnessImprovementSuggestion {
                id: "risky".into(),
                summary: "Aggressively rewrite system prompt".into(),
                failure_classes_covered: 5,
                specificity: Some(0),
                regression_risk: 0.9,
            },
            HarnessImprovementSuggestion {
                id: "safe".into(),
                summary: "Add loop-detection middleware".into(),
                failure_classes_covered: 3,
                specificity: Some(0),
                regression_risk: 0.1,
            },
        ]);
        assert_eq!(ranked[0].suggestion.id, "safe");
    }
}
