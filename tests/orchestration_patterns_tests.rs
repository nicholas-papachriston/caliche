//! Regression tests for the built-in orchestration patterns (fan-out/collect, hierarchical,
//! debate, supervisor). These patterns generate graphs that start with a `NodeType::Parallel`
//! node forcing control flow to a `Merge` node once branches complete. That forced transition
//! is only honored by `WorkflowEngine::execute_state_machine`, not the pure topological
//! `execute_dag` scheduler. Each `Workflow::execute` impl must route through
//! `execute_state_machine` or these patterns fail (or double-execute branch nodes) at runtime.

use anyhow::Result;
use async_trait::async_trait;
use caliche::orchestration::{
    DebatePattern, ExecutionContext, FanOutCollectPattern, HierarchicalPattern, Workflow,
    WorkflowEngine, WorkflowNodeExecutor,
};
use serde_json::Value;
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Records how many times each prompt is executed so tests can assert branch nodes run
/// exactly once (not zero, not twice) under the state-machine executor.
#[derive(Default)]
struct CountingExecutor {
    calls: AtomicUsize,
}

#[async_trait]
impl WorkflowNodeExecutor for CountingExecutor {
    async fn execute_agent(
        &self,
        prompt: &str,
        _agent_id: Option<&str>,
        _capability: Option<&str>,
        _context: &ExecutionContext,
    ) -> Result<(Value, HashMap<String, Value>)> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        Ok((serde_json::json!({ "prompt": prompt }), HashMap::new()))
    }

    async fn transform(
        &self,
        _function: &str,
        _context: &ExecutionContext,
        last_output: Option<&Value>,
    ) -> Result<Value> {
        Ok(last_output.cloned().unwrap_or(serde_json::json!(null)))
    }

    async fn merge_custom(
        &self,
        _name: &str,
        inputs: &Value,
        _context: &ExecutionContext,
    ) -> Result<Value> {
        Ok(inputs.clone())
    }
}

#[tokio::test]
async fn fanout_collect_pattern_executes_via_state_machine() {
    let executor = Arc::new(CountingExecutor::default());
    let engine = WorkflowEngine::with_executor(executor.clone());

    let pattern = FanOutCollectPattern::new("fanout_test")
        .add_branch("branch_a", "Do task A")
        .add_branch("branch_b", "Do task B")
        .add_branch("branch_c", "Do task C");

    let mut context = ExecutionContext::new("fanout_test");
    let result = pattern
        .execute(&engine, &mut context)
        .await
        .expect("fan-out/collect pattern should execute successfully via the state machine");

    assert!(!result.is_null());
    // Exactly one call per branch: the state-machine executor must not double-execute
    // branch nodes when the parallel fan-out forces control flow to the merge node.
    assert_eq!(executor.calls.load(Ordering::SeqCst), 3);
}

#[tokio::test]
async fn fanout_collect_pattern_with_final_aggregator() {
    let executor = Arc::new(CountingExecutor::default());
    let engine = WorkflowEngine::with_executor(executor.clone());

    let pattern = FanOutCollectPattern::new("fanout_aggregate_test")
        .add_branch("branch_a", "Do task A")
        .add_branch("branch_b", "Do task B")
        .with_final_aggregator("Summarize the results");

    let mut context = ExecutionContext::new("fanout_aggregate_test");
    let result = pattern
        .execute(&engine, &mut context)
        .await
        .expect("fan-out/collect pattern with aggregator should execute successfully");

    assert!(!result.is_null());
    // 2 branches + 1 aggregator node.
    assert_eq!(executor.calls.load(Ordering::SeqCst), 3);
}

#[tokio::test]
async fn hierarchical_pattern_executes_via_state_machine() {
    let executor = Arc::new(CountingExecutor::default());
    let engine = WorkflowEngine::with_executor(executor.clone());

    let pattern = HierarchicalPattern::new("hierarchical_test")
        .add_layer("layer_0", 2, "Layer 0 task")
        .add_layer("layer_1", 1, "Layer 1 task");

    let mut context = ExecutionContext::new("hierarchical_test");
    let result = pattern
        .execute(&engine, &mut context)
        .await
        .expect("hierarchical pattern should execute successfully via the state machine");

    assert!(!result.is_null());
    // 2 agents in layer 0 + 1 agent in layer 1.
    assert_eq!(executor.calls.load(Ordering::SeqCst), 3);
}

#[tokio::test]
async fn debate_pattern_executes_via_state_machine() {
    let executor = Arc::new(CountingExecutor::default());
    let engine = WorkflowEngine::with_executor(executor.clone());

    let pattern = DebatePattern::new("debate_test", "Is this pattern correct?", 2, 1);

    let mut context = ExecutionContext::new("debate_test");
    let result = pattern
        .execute(&engine, &mut context)
        .await
        .expect("debate pattern should execute successfully via the state machine");

    assert!(!result.is_null());
    // 2 debaters + evaluate_consensus + final_synthesis.
    assert_eq!(executor.calls.load(Ordering::SeqCst), 4);
}
