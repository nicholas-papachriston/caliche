//! Tests for cyclic state machine workflows

use caliche::orchestration::{Workflow, WorkflowBuilder, WorkflowEngine};
use std::sync::Arc;

#[tokio::test]
async fn test_cyclic_workflow_validation() {
    // Create a workflow with cycles and entry_point (should validate)
    let workflow = WorkflowBuilder::new("cyclic_test")
        .with_description("Test cyclic workflow")
        .add_agent("start", "Start task")
        .add_agent("loop", "Loop task")
        .with_entry_point("start")
        .add_edge_with_condition("start", "loop", "true")
        .add_edge_with_condition("loop", "start", "count < 5") // Cyclic edge!
        .build();

    // Should validate successfully (cycles allowed with entry_point)
    assert!(Workflow::validate(&workflow).is_ok());
}

#[tokio::test]
async fn test_cyclic_workflow_without_entry_point_fails() {
    // Create a workflow with cycles but NO entry_point (should fail validation)
    let workflow = WorkflowBuilder::new("invalid_cyclic")
        .add_agent("node1", "Task 1")
        .add_agent("node2", "Task 2")
        .add_edge_with_condition("node1", "node2", "true")
        .add_edge_with_condition("node2", "node1", "true") // Cycle without entry_point
        .build();

    // Should fail validation (cycles not allowed without entry_point)
    assert!(Workflow::validate(&workflow).is_err());
}

#[tokio::test]
async fn test_state_machine_execution() {
    let engine = WorkflowEngine::new();

    // Create a simple counter loop workflow
    let workflow = WorkflowBuilder::new("counter_loop")
        .with_description("Counter that loops until it reaches 5")
        .add_agent("init", "Initialize counter to 0")
        .add_agent("increment", "Increment counter")
        .add_agent("check", "Check if counter >= 5")
        .with_entry_point("init")
        .add_edge_with_condition("init", "increment", "true")
        .add_edge_with_condition("increment", "check", "true")
        .add_edge_with_condition("check", "increment", "count < 5") // Loop back
        // When count >= 5, no edge matches, so workflow ends
        .build();

    // Validate workflow
    assert!(Workflow::validate(&workflow).is_ok());

    // Register workflow
    engine
        .register_workflow(Arc::new(workflow))
        .await
        .expect("Failed to register workflow");

    // Note: Actual execution would require real agent implementation
    // This test primarily validates the structure and validation logic
}

#[tokio::test]
async fn test_entry_point_validation() {
    // Entry point that doesn't exist should fail
    let workflow = WorkflowBuilder::new("invalid_entry")
        .add_agent("node1", "Task 1")
        .with_entry_point("nonexistent_node")
        .build();

    assert!(Workflow::validate(&workflow).is_err());
}

#[tokio::test]
async fn test_conditional_transitions() {
    // Test workflow with multiple conditional paths
    let workflow = WorkflowBuilder::new("conditional_test")
        .add_agent("start", "Start")
        .add_agent("path_a", "Path A")
        .add_agent("path_b", "Path B")
        .add_agent("end", "End")
        .with_entry_point("start")
        .add_edge_with_condition("start", "path_a", "choice == 'a'")
        .add_edge_with_condition("start", "path_b", "choice == 'b'")
        .add_edge_with_condition("path_a", "end", "true")
        .add_edge_with_condition("path_b", "end", "true")
        .build();

    assert!(Workflow::validate(&workflow).is_ok());
    assert_eq!(Workflow::entry_point(&workflow), Some("start"));
}

#[tokio::test]
async fn test_dag_mode_still_works() {
    // Workflow without entry_point should still work as DAG
    let workflow = WorkflowBuilder::new("dag_test")
        .add_agent("node1", "Task 1")
        .add_agent("node2", "Task 2")
        .add_agent("node3", "Task 3")
        // No entry_point - should use DAG mode
        .build();

    assert!(Workflow::validate(&workflow).is_ok());
    assert_eq!(Workflow::entry_point(&workflow), None);
}
