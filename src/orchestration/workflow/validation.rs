//! Shared workflow structure validation used by [`Workflow`] and [`BuiltWorkflow`](crate::orchestration::builder::BuiltWorkflow).

use anyhow::Result;
use std::collections::{HashMap, HashSet};

use super::transition;
use super::{NodeType, WorkflowEdge, WorkflowNode};

/// How the workflow entry point is represented for validation.
#[derive(Debug, Clone)]
pub(crate) enum EntryPointRef<'a> {
    /// No entry point — DAG mode (cycle detection enforced).
    None,
    /// Entry point stored as a node id (`BuiltWorkflow`).
    NodeId(&'a str),
    /// Entry point resolved as a node name (`Workflow` trait default).
    NodeName(&'a str),
}

impl EntryPointRef<'_> {
    fn allows_cycles(&self) -> bool {
        !matches!(self, EntryPointRef::None)
    }
}

/// Validate nodes, edges, branching constraints, and cycle policy.
pub(crate) fn validate_workflow(
    nodes: &[WorkflowNode],
    edges: &[WorkflowEdge],
    entry_point: EntryPointRef<'_>,
) -> Result<()> {
    let nodes_map: HashMap<&str, &WorkflowNode> =
        nodes.iter().map(|n| (n.id.as_str(), n)).collect();

    let mut adj: HashMap<String, Vec<String>> = HashMap::new();
    for edge in edges {
        adj.entry(edge.from_node.clone())
            .or_default()
            .push(edge.to_node.clone());
    }

    validate_edge_endpoints(nodes, edges, &nodes_map)?;
    validate_node_branching(nodes, edges)?;
    validate_entry_point(nodes, &entry_point)?;
    transition::validate_execution_mode(nodes, edges, entry_point.allows_cycles())?;

    if entry_point.allows_cycles() {
        return Ok(());
    }
    detect_cycles(nodes, &adj)
}

fn validate_edge_endpoints(
    nodes: &[WorkflowNode],
    edges: &[WorkflowEdge],
    nodes_map: &HashMap<&str, &WorkflowNode>,
) -> Result<()> {
    for edge in edges {
        if !nodes_map.contains_key(edge.from_node.as_str()) {
            let all_node_ids: Vec<_> = nodes
                .iter()
                .map(|n| format!("{} ({})", n.id, n.name))
                .collect();
            anyhow::bail!(
                "Edge references non-existent from_node: {}. Available nodes: {:?}. Edge: {:?}",
                edge.from_node,
                all_node_ids,
                edge
            );
        }
        if !nodes_map.contains_key(edge.to_node.as_str()) {
            let all_node_ids: Vec<_> = nodes
                .iter()
                .map(|n| format!("{} ({})", n.id, n.name))
                .collect();
            let all_edges: Vec<_> = edges
                .iter()
                .map(|e| format!("{} -> {} (cond: {:?})", e.from_node, e.to_node, e.condition))
                .collect();
            anyhow::bail!(
                "Edge references non-existent to_node: {}. Available nodes: {:?}. All edges: {:?}. Problematic edge: {:?}",
                edge.to_node,
                all_node_ids,
                all_edges,
                edge
            );
        }
    }
    Ok(())
}

fn validate_node_branching(nodes: &[WorkflowNode], edges: &[WorkflowEdge]) -> Result<()> {
    let mut outgoing_count: HashMap<&str, usize> = HashMap::new();
    let mut incoming_count: HashMap<&str, usize> = HashMap::new();
    let mut loop_edge_labels: HashMap<&str, Vec<Option<&str>>> = HashMap::new();
    for edge in edges {
        let from = edge.from_node.as_str();
        let to = edge.to_node.as_str();
        *outgoing_count.entry(from).or_insert(0) += 1;
        *incoming_count.entry(to).or_insert(0) += 1;
        loop_edge_labels
            .entry(from)
            .or_default()
            .push(edge.label.as_deref());
    }

    for node in nodes {
        match &node.node_type {
            NodeType::Parallel => {
                if outgoing_count.get(node.id.as_str()).copied().unwrap_or(0) == 0 {
                    anyhow::bail!(
                        "Parallel node '{}' ({}) has no outgoing branches",
                        node.name,
                        node.id
                    );
                }
            }
            NodeType::Merge { .. } => {
                if incoming_count.get(node.id.as_str()).copied().unwrap_or(0) == 0 {
                    anyhow::bail!(
                        "Merge node '{}' ({}) has no incoming branch results",
                        node.name,
                        node.id
                    );
                }
            }
            NodeType::Loop { .. } => validate_loop_node(node, &outgoing_count, &loop_edge_labels)?,
            _ => {}
        }
    }
    Ok(())
}

fn validate_loop_node(
    node: &WorkflowNode,
    outgoing_count: &HashMap<&str, usize>,
    loop_edge_labels: &HashMap<&str, Vec<Option<&str>>>,
) -> Result<()> {
    let outgoing = outgoing_count.get(node.id.as_str()).copied().unwrap_or(0);
    if outgoing == 0 {
        anyhow::bail!(
            "Loop node '{}' ({}) has no outgoing edges",
            node.name,
            node.id
        );
    }
    if let Some(labels) = loop_edge_labels.get(node.id.as_str()) {
        let has_labeled = labels.iter().any(|label| label.is_some());
        if has_labeled && !labels.contains(&Some("body")) {
            anyhow::bail!(
                "Loop node '{}' ({}) uses labeled edges but is missing a 'body' edge",
                node.name,
                node.id
            );
        }
    }
    Ok(())
}

fn validate_entry_point(nodes: &[WorkflowNode], entry_point: &EntryPointRef<'_>) -> Result<()> {
    match entry_point {
        EntryPointRef::NodeId(id) => {
            if !nodes.iter().any(|n| n.id == *id) {
                anyhow::bail!("Entry point ID '{}' does not exist in workflow nodes", id);
            }
        }
        EntryPointRef::NodeName(name) => {
            if !nodes.iter().any(|n| n.name == *name) {
                anyhow::bail!("Entry point '{}' does not exist in workflow nodes", name);
            }
        }
        EntryPointRef::None => {}
    }
    Ok(())
}

fn detect_cycles(nodes: &[WorkflowNode], adj: &HashMap<String, Vec<String>>) -> Result<()> {
    let mut visited = HashSet::new();
    let mut rec_stack = HashSet::new();
    for node in nodes {
        if !visited.contains(&node.id) && has_cycle(&node.id, adj, &mut visited, &mut rec_stack) {
            anyhow::bail!("Workflow contains cycles. Use entry_point() for state machine mode.");
        }
    }
    Ok(())
}

fn has_cycle(
    node: &str,
    adj: &HashMap<String, Vec<String>>,
    visited: &mut HashSet<String>,
    rec_stack: &mut HashSet<String>,
) -> bool {
    visited.insert(node.to_string());
    rec_stack.insert(node.to_string());

    if let Some(neighbors) = adj.get(node) {
        for neighbor in neighbors {
            if !visited.contains(neighbor) {
                if has_cycle(neighbor, adj, visited, rec_stack) {
                    return true;
                }
            } else if rec_stack.contains(neighbor) {
                return true;
            }
        }
    }

    rec_stack.remove(node);
    false
}
