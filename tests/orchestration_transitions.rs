//! Workflow conditional edge routing and execution-mode validation.

use caliche::orchestration::workflow::transition::{
    COND_BRANCH_FALSE, COND_BRANCH_TRUE, select_next_edge, validate_execution_mode,
};
use caliche::orchestration::workflow::{
    ExecutionContext, NodeType, WorkflowEdge, WorkflowEngine, WorkflowNode,
};
use serde_json::Value;

fn eval(condition: &str, context: &Value, node_output: Option<&Value>) -> bool {
    WorkflowEngine::evaluate_condition(condition, context, node_output)
}

#[test]
fn select_next_edge_honors_true_false_labels() {
    let context = ExecutionContext::new("wf");
    let true_node = "true-branch".to_string();
    let false_node = "false-branch".to_string();
    let owned = vec![
        WorkflowEdge::new("cond-1", true_node.clone()).with_label("true"),
        WorkflowEdge::new("cond-1", false_node.clone()).with_label("false"),
    ];
    let edges: Vec<&WorkflowEdge> = owned.iter().collect();

    let cond_true = serde_json::json!({ "value": true });
    let next = select_next_edge(&edges, &context, Some(&cond_true), eval);
    assert_eq!(next.as_deref(), Some(true_node.as_str()));

    let cond_false = serde_json::json!({ "value": false });
    let next = select_next_edge(&edges, &context, Some(&cond_false), eval);
    assert_eq!(next.as_deref(), Some(false_node.as_str()));
}

#[test]
fn select_next_edge_honors_cond_branch_expressions() {
    let context = ExecutionContext::new("wf");
    let true_node = "expr-true".to_string();
    let false_node = "expr-false".to_string();
    let owned = vec![
        WorkflowEdge::new("cond-1", true_node.clone()).with_condition(COND_BRANCH_TRUE),
        WorkflowEdge::new("cond-1", false_node.clone()).with_condition(COND_BRANCH_FALSE),
    ];
    let edges: Vec<&WorkflowEdge> = owned.iter().collect();

    let cond_true = serde_json::json!({ "value": true });
    let next = select_next_edge(&edges, &context, Some(&cond_true), eval);
    assert_eq!(next.as_deref(), Some(true_node.as_str()));

    let cond_false = serde_json::json!({ "value": false });
    let next = select_next_edge(&edges, &context, Some(&cond_false), eval);
    assert_eq!(next.as_deref(), Some(false_node.as_str()));
}

#[test]
fn select_next_edge_uses_first_matching_condition() {
    let mut context = ExecutionContext::new("wf");
    context.set_variable("status", serde_json::json!("approved"));
    let approved_node = "approved".to_string();
    let fallback_node = "fallback".to_string();
    let owned = vec![
        WorkflowEdge::new("gate", approved_node.clone()).with_condition("status == 'approved'"),
        WorkflowEdge::new("gate", fallback_node),
    ];
    let edges: Vec<&WorkflowEdge> = owned.iter().collect();

    let next = select_next_edge(&edges, &context, None, eval);
    assert_eq!(next.as_deref(), Some(approved_node.as_str()));
}

#[test]
fn validate_execution_mode_rejects_branching_without_entry_point() {
    let nodes = vec![WorkflowNode::new(
        "cond",
        NodeType::Condition {
            predicate: "x == 1".to_string(),
        },
    )];
    let edges = vec![
        WorkflowEdge::new("n1", "n2").with_label("true"),
        WorkflowEdge::new("n1", "n3").with_label("false"),
    ];
    let err = validate_execution_mode(&nodes, &edges, false).expect_err("branching without entry");
    assert!(
        err.to_string()
            .contains("Condition or Loop nodes require entry_point"),
        "unexpected error: {err}"
    );
}

#[test]
fn validate_execution_mode_rejects_exclusive_edges_without_entry_point() {
    let nodes = vec![WorkflowNode::new(
        "n1",
        NodeType::Agent {
            agent_id: None,
            capability: None,
            prompt_template: "do work".to_string(),
        },
    )];
    let edges = vec![
        WorkflowEdge::new("n1", "n2").with_condition("$.ok == true"),
        WorkflowEdge::new("n1", "n3").with_condition("$.ok == false"),
    ];
    let err = validate_execution_mode(&nodes, &edges, false).expect_err("exclusive edges");
    assert!(
        err.to_string()
            .contains("mutually exclusive outgoing edges"),
        "unexpected error: {err}"
    );
}

#[test]
fn validate_execution_mode_allows_branching_with_entry_point() {
    let nodes = vec![WorkflowNode::new(
        "cond",
        NodeType::Condition {
            predicate: "x == 1".to_string(),
        },
    )];
    let edges = vec![
        WorkflowEdge::new("n1", "n2").with_label("true"),
        WorkflowEdge::new("n1", "n3").with_label("false"),
    ];
    validate_execution_mode(&nodes, &edges, true).expect("entry_point enables state machine mode");
}
