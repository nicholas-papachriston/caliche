#![cfg(feature = "discipline-budget-tests")]

//! Deterministic tokens-per-turn budget fixture for harness discipline metrics.
//!
//! No live LLM: constructs [`HarnessDisciplineMetrics`] synthetically and asserts
//! each fixture turn stays within a median-style budget.

use caliche::harness::HarnessDisciplineMetrics;
use caliche::llm::ProviderType;
use caliche::messages::TokenUsage;

/// Soft median-style ceiling used by CI budget checks (tokens per turn).
const TOKENS_PER_TURN_BUDGET: f64 = 50_000.0;

fn turn_metrics(total_tokens: u32, tool_calls: usize) -> HarnessDisciplineMetrics {
    let usage = TokenUsage::from_counts(
        Some(total_tokens.saturating_mul(2) / 3),
        Some(total_tokens / 3),
        Some(total_tokens),
    );
    HarnessDisciplineMetrics::for_turn(
        &usage,
        tool_calls,
        0,
        0,
        Some(ProviderType::OpenAI),
        Some("gpt-4o"),
    )
}

#[test]
fn synthetic_turns_stay_within_tokens_per_turn_budget() {
    let fixtures = [
        turn_metrics(1_200, 2),
        turn_metrics(8_500, 4),
        turn_metrics(24_000, 6),
        turn_metrics(49_999, 1),
    ];

    for (idx, metrics) in fixtures.iter().enumerate() {
        let tokens = metrics
            .tokens_per_turn
            .expect("fixture must set tokens_per_turn");
        assert!(
            tokens <= TOKENS_PER_TURN_BUDGET,
            "fixture turn {idx} exceeded budget: {tokens} > {TOKENS_PER_TURN_BUDGET}"
        );
        assert_eq!(metrics.compactions, 0);
        assert_eq!(metrics.readonly_filter_hits, 0);
    }

    let median = {
        let mut values: Vec<f64> = fixtures
            .iter()
            .map(|m| m.tokens_per_turn.expect("tokens"))
            .collect();
        values.sort_by(|a, b| a.partial_cmp(b).expect("finite"));
        let mid = values.len() / 2;
        if values.len() % 2 == 0 {
            (values[mid - 1] + values[mid]) / 2.0
        } else {
            values[mid]
        }
    };
    assert!(
        median <= TOKENS_PER_TURN_BUDGET,
        "median tokens/turn {median} exceeded budget {TOKENS_PER_TURN_BUDGET}"
    );
}
