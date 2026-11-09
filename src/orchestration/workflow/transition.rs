use super::{ExecutionContext, NodeType, WorkflowEdge, WorkflowNode};
use anyhow::Result;
use serde_json::Value;
use std::collections::HashMap;

/// Edge condition expressions emitted by the workflow builder for Condition branches.
pub const COND_BRANCH_TRUE: &str = "$.value == true";
pub const COND_BRANCH_FALSE: &str = "$.value == false";

/// Select the next node by evaluating outgoing edges in order (first match wins).
///
/// Honors explicit `edge.condition` expressions and `true`/`false` edge labels against a
/// Condition node's `{ "value": bool }` output.
pub fn select_next_edge(
    edges: &[&WorkflowEdge],
    context: &ExecutionContext,
    node_output: Option<&Value>,
    evaluate_condition: impl Fn(&str, &Value, Option<&Value>) -> bool,
) -> Option<String> {
    if edges.is_empty() {
        return None;
    }

    let context_value = Value::Object(
        context
            .variables
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect(),
    );

    let mut default_unconditional: Option<String> = None;

    for edge in edges {
        if let Some(ref condition) = edge.condition {
            if evaluate_condition(condition, &context_value, node_output) {
                return Some(edge.to_node.clone());
            }
            continue;
        }

        if let Some(label) = edge.label.as_deref() {
            if let Some(cond_value) = node_output
                .and_then(|o| o.get("value"))
                .and_then(|v| v.as_bool())
            {
                let matches = match label {
                    "true" => cond_value,
                    "false" => !cond_value,
                    _ => false,
                };
                if matches {
                    return Some(edge.to_node.clone());
                }
            }
            continue;
        }

        if default_unconditional.is_none() {
            default_unconditional = Some(edge.to_node.clone());
        }
    }

    default_unconditional
}

/// Validate that control-flow graphs use state-machine mode (`entry_point`).
pub fn validate_execution_mode(
    nodes: &[WorkflowNode],
    edges: &[WorkflowEdge],
    has_entry_point: bool,
) -> Result<()> {
    if has_entry_point {
        return Ok(());
    }

    let has_control_flow_node = nodes.iter().any(|n| {
        matches!(
            n.node_type,
            NodeType::Condition { .. } | NodeType::Loop { .. }
        )
    });

    if has_control_flow_node {
        anyhow::bail!(
            "Workflows with Condition or Loop nodes require entry_point() for state machine execution"
        );
    }

    let mut outgoing: HashMap<&str, Vec<&WorkflowEdge>> = HashMap::new();
    for edge in edges {
        outgoing
            .entry(edge.from_node.as_str())
            .or_default()
            .push(edge);
    }

    for (from_node, outs) in outgoing {
        if outs.len() <= 1 {
            continue;
        }
        let has_exclusive_branch = outs.iter().any(|e| {
            e.condition.is_some() || matches!(e.label.as_deref(), Some("true") | Some("false"))
        });
        if has_exclusive_branch {
            anyhow::bail!(
                "Node '{}' has mutually exclusive outgoing edges; use entry_point() for state machine execution",
                from_node
            );
        }
    }

    Ok(())
}
