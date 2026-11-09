//! Per-run cost budgets for [`WorkflowEngine`](super::WorkflowEngine).
//!
//! An orchestration pattern can fan out over data, so the node count and the token spend of
//! one run are not always known when the run starts. A budget bounds both: the engine
//! refuses to start another agent node once the run reaches the configured ceiling, and the
//! refusal is recorded as a node error instead of a silent stop.
//!
//! Token accounting depends on the executor. A
//! [`WorkflowNodeExecutor`](crate::orchestration::workflow_executor::WorkflowNodeExecutor)
//! that reports `total_tokens` (or `tokens`) in node metadata is charged for what it spends;
//! one that reports nothing never trips the token ceiling.

use super::{ExecutionContext, WorkflowEngine};
use std::collections::HashMap;

/// Metadata keys an executor may use to report the token spend of one agent node.
const TOKEN_METADATA_KEYS: &[&str] = &["total_tokens", "tokens"];

/// Recorded spend of one workflow run.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct RunUsage {
    /// Agent nodes started so far.
    pub nodes: usize,
    /// Tokens reported by the executor so far.
    pub tokens: u64,
}

impl WorkflowEngine {
    /// Reserve one agent node against the run budget.
    ///
    /// Returns the reason to record when the run may not start another node. Reserving
    /// before execution — rather than counting afterwards — keeps the ceiling honest when
    /// parallel branches start nodes at the same time.
    pub(super) fn reserve_agent_node(&self, context: &ExecutionContext) -> Result<(), String> {
        if self.max_nodes.is_none() && self.max_tokens.is_none() {
            return Ok(());
        }

        let mut usage = self
            .run_usage
            .entry(context.execution_id.clone())
            .or_default();

        if let Some(max_tokens) = self.max_tokens
            && usage.tokens >= max_tokens
        {
            return Err(format!(
                "Run token budget exhausted: {} of {max_tokens} tokens spent",
                usage.tokens
            ));
        }
        if let Some(max_nodes) = self.max_nodes
            && usage.nodes >= max_nodes
        {
            return Err(format!(
                "Run node budget exhausted: {max_nodes} agent nodes already started"
            ));
        }

        usage.nodes += 1;
        Ok(())
    }

    /// Charge the tokens one agent node reported.
    pub(super) fn charge_tokens(
        &self,
        context: &ExecutionContext,
        metadata: &HashMap<String, serde_json::Value>,
    ) {
        if self.max_tokens.is_none() {
            return;
        }
        let spent = tokens_from_metadata(metadata);
        if spent == 0 {
            return;
        }
        self.run_usage
            .entry(context.execution_id.clone())
            .or_default()
            .tokens += spent;
    }

    /// Recorded spend of one run, when the engine tracked it.
    pub fn run_usage(&self, execution_id: &str) -> Option<RunUsage> {
        self.run_usage.get(execution_id).map(|entry| *entry.value())
    }

    /// Drop the recorded spend of one run.
    pub(super) fn forget_run_usage(&self, execution_id: &str) {
        self.run_usage.remove(execution_id);
    }
}

/// Token spend an executor reported for one node, or `0` when it reported none.
fn tokens_from_metadata(metadata: &HashMap<String, serde_json::Value>) -> u64 {
    TOKEN_METADATA_KEYS
        .iter()
        .find_map(|key| metadata.get(*key).and_then(serde_json::Value::as_u64))
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn metadata(pairs: &[(&str, u64)]) -> HashMap<String, serde_json::Value> {
        pairs
            .iter()
            .map(|(key, value)| ((*key).to_string(), serde_json::json!(value)))
            .collect()
    }

    #[test]
    fn no_budget_reserves_without_tracking() {
        let engine = WorkflowEngine::new();
        let context = ExecutionContext::new("wf");
        assert!(engine.reserve_agent_node(&context).is_ok());
        assert!(engine.run_usage(&context.execution_id).is_none());
    }

    #[test]
    fn node_budget_stops_the_run() {
        let engine = WorkflowEngine::new().with_run_budgets(Some(2), None);
        let context = ExecutionContext::new("wf");

        assert!(engine.reserve_agent_node(&context).is_ok());
        assert!(engine.reserve_agent_node(&context).is_ok());
        let refused = engine
            .reserve_agent_node(&context)
            .expect_err("third node exceeds the budget");
        assert!(refused.contains("node budget"));
        assert_eq!(
            engine.run_usage(&context.execution_id).map(|u| u.nodes),
            Some(2)
        );
    }

    #[test]
    fn token_budget_stops_the_run_after_a_charge() {
        let engine = WorkflowEngine::new().with_run_budgets(None, Some(100));
        let context = ExecutionContext::new("wf");

        assert!(engine.reserve_agent_node(&context).is_ok());
        engine.charge_tokens(&context, &metadata(&[("total_tokens", 150)]));

        let refused = engine
            .reserve_agent_node(&context)
            .expect_err("the next node exceeds the token budget");
        assert!(refused.contains("token budget"));
    }

    #[test]
    fn unreported_tokens_never_trip_the_budget() {
        let engine = WorkflowEngine::new().with_run_budgets(None, Some(10));
        let context = ExecutionContext::new("wf");

        assert!(engine.reserve_agent_node(&context).is_ok());
        engine.charge_tokens(&context, &metadata(&[("model", 1)]));
        assert!(engine.reserve_agent_node(&context).is_ok());
    }

    #[test]
    fn budgets_are_per_run() {
        let engine = WorkflowEngine::new().with_run_budgets(Some(1), None);
        let first = ExecutionContext::new("wf");
        let second = ExecutionContext::new("wf");

        assert!(engine.reserve_agent_node(&first).is_ok());
        assert!(engine.reserve_agent_node(&second).is_ok());
        assert!(engine.reserve_agent_node(&first).is_err());
    }

    #[test]
    fn forgetting_a_run_drops_its_recorded_spend() {
        let engine = WorkflowEngine::new().with_run_budgets(Some(4), None);
        let context = ExecutionContext::new("wf");
        assert!(engine.reserve_agent_node(&context).is_ok());

        engine.forget_run_usage(&context.execution_id);
        assert!(engine.run_usage(&context.execution_id).is_none());
    }
}
