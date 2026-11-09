//! Parallel fan-out and merge execution for [`WorkflowEngine`](super::WorkflowEngine).
//!
//! Branches run as independent Tokio tasks behind a shared concurrency semaphore. The merge
//! strategy that the branches converge on is resolved before they start, so `FirstComplete`
//! can stop waiting — and cancel the losers — as soon as one branch reaches the merge node.

use super::constants::META_FORCED_NEXT_NODE_ID;
use super::{
    BranchExecutionResult, ExecutionContext, MergeStrategy, NodeExecutionResult, NodeType,
    WorkflowEdge, WorkflowEngine, WorkflowNode, WorkflowState, branch_timeout_from_metadata,
    build_nodes_map, build_outgoing_edges,
};
use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use futures_util::future::{join_all, select_all};
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Arc;
use tokio::task::JoinHandle;

/// Branch outcome after the fan-out completes: per-branch output plus the shared merge node.
#[derive(Debug)]
struct BranchCollection {
    merge_target: Option<String>,
    branch_outputs: serde_json::Map<String, serde_json::Value>,
    /// Set when the branches could not be reconciled (no merge reached, or divergent merges).
    failure: Option<String>,
}

impl WorkflowEngine {
    #[allow(clippy::too_many_arguments)]
    pub(super) async fn run_parallel_node(
        &self,
        node: &WorkflowNode,
        nodes: &[WorkflowNode],
        edges: &[WorkflowEdge],
        context: &mut ExecutionContext,
        nodes_map: &HashMap<&str, &WorkflowNode>,
        outgoing_edges: &HashMap<&str, Vec<&WorkflowEdge>>,
        metadata: &mut HashMap<String, serde_json::Value>,
        start_time: DateTime<Utc>,
    ) -> Result<NodeExecutionResult> {
        let edges_from = outgoing_edges
            .get(node.id.as_str())
            .cloned()
            .unwrap_or_default();
        if edges_from.is_empty() {
            return Ok(NodeExecutionResult::err(
                node.id.clone(),
                "Parallel node has no outgoing edges",
                start_time,
                HashMap::new(),
            ));
        }

        let branch_starts: Vec<String> = edges_from.iter().map(|e| e.to_node.clone()).collect();
        let first_complete = matches!(
            planned_merge_strategy(nodes_map, outgoing_edges, &branch_starts),
            Some(MergeStrategy::FirstComplete)
        );

        let handles = self
            .spawn_branches(node, nodes, edges, context, &branch_starts)
            .await?;

        let collection = if first_complete {
            collect_first_complete(handles).await?
        } else {
            collect_all(handles).await?
        };

        if let Some(failure) = collection.failure {
            return Ok(NodeExecutionResult::err(
                node.id.clone(),
                failure,
                start_time,
                HashMap::new(),
            ));
        }

        if let Some(merge_id) = collection.merge_target {
            let key = format!("__parallel_merge_inputs_{}", merge_id);
            context.set_variable(
                key,
                serde_json::Value::Object(collection.branch_outputs.clone()),
            );
            metadata.insert(
                META_FORCED_NEXT_NODE_ID.to_owned(),
                serde_json::Value::String(merge_id),
            );
        }
        Ok(NodeExecutionResult::ok(
            node.id.clone(),
            serde_json::Value::Object(collection.branch_outputs),
            start_time,
            std::mem::take(metadata),
        ))
    }

    /// Spawn one task per branch, each holding a concurrency permit for its whole run.
    ///
    /// Acquiring a permit before spawning — rather than inside the task — also caps how many
    /// Tokio tasks and cloned `ExecutionContext`s exist at once, not just how many are
    /// actively executing.
    async fn spawn_branches(
        &self,
        node: &WorkflowNode,
        nodes: &[WorkflowNode],
        edges: &[WorkflowEdge],
        context: &ExecutionContext,
        branch_starts: &[String],
    ) -> Result<Vec<JoinHandle<BranchExecutionResult>>> {
        let nodes_arc: Arc<Vec<WorkflowNode>> = Arc::new(nodes.to_vec());
        let edges_arc: Arc<Vec<WorkflowEdge>> = Arc::new(edges.to_vec());
        let branch_timeout = branch_timeout_from_metadata(&node.metadata);

        let mut handles: Vec<JoinHandle<BranchExecutionResult>> =
            Vec::with_capacity(branch_starts.len());
        for branch_start in branch_starts {
            let branch_start = branch_start.clone();
            let engine = self.clone();
            let nodes_clone = Arc::clone(&nodes_arc);
            let edges_clone = Arc::clone(&edges_arc);
            let mut branch_ctx = context.clone();
            branch_ctx.state = WorkflowState::Running;
            let permit = Arc::clone(&self.parallel_branch_semaphore)
                .acquire_owned()
                .await
                .context("parallel branch semaphore was unexpectedly closed")?;
            let timeout = branch_timeout;

            handles.push(tokio::spawn(async move {
                let _permit = permit;
                let nodes_slice = nodes_clone.as_slice();
                let edges_slice = edges_clone.as_slice();
                let nodes_map = build_nodes_map(nodes_slice);
                let outgoing_edges_map = build_outgoing_edges(edges_slice);

                let run = engine.execute_until_first_merge(
                    &nodes_map,
                    &outgoing_edges_map,
                    branch_start.as_str(),
                    nodes_slice,
                    edges_slice,
                    &mut branch_ctx,
                );

                let (results, found_merge) = match timeout {
                    Some(duration) => {
                        tokio::time::timeout(duration, run).await.map_err(|_| {
                            anyhow::anyhow!(
                                "Parallel branch starting at {} timed out after {}ms",
                                branch_start,
                                duration.as_millis()
                            )
                        })??
                    }
                    None => run.await?,
                };
                Ok((branch_start, results, found_merge))
            }));
        }
        Ok(handles)
    }

    pub(super) async fn run_merge_node(
        &self,
        node: &WorkflowNode,
        _nodes: &[WorkflowNode],
        _edges: &[WorkflowEdge],
        context: &ExecutionContext,
        metadata: &mut HashMap<String, serde_json::Value>,
        start_time: DateTime<Utc>,
    ) -> Result<NodeExecutionResult> {
        let NodeType::Merge { strategy } = &node.node_type else {
            return Ok(NodeExecutionResult::err(
                node.id.clone(),
                format!("expected Merge node, got {:?}", node.node_type),
                start_time,
                std::mem::take(metadata),
            ));
        };
        let key = format!("__parallel_merge_inputs_{}", node.id);
        let Some(value) = context.get_variable(key.as_str()).cloned() else {
            return Ok(NodeExecutionResult::err(
                node.id.clone(),
                format!(
                    "Merge node {} missing parallel inputs (expected {})",
                    node.id, key
                ),
                start_time,
                std::mem::take(metadata),
            ));
        };

        let merged = match strategy {
            MergeStrategy::Concatenate => merge_concatenate(&value),
            MergeStrategy::Vote => merge_vote(&value),
            MergeStrategy::FirstComplete => merge_first(&value),
            MergeStrategy::AllComplete => value,
            MergeStrategy::Custom(name) => self
                .executor
                .merge_custom(name.as_str(), &value, context)
                .await
                .with_context(|| format!("Custom merge strategy '{}' failed", name))?,
        };
        Ok(NodeExecutionResult::ok(
            node.id.clone(),
            merged,
            start_time,
            std::mem::take(metadata),
        ))
    }
}

/// Wait for every branch, then reconcile their merge targets.
async fn collect_all(handles: Vec<JoinHandle<BranchExecutionResult>>) -> Result<BranchCollection> {
    let branch_results: Vec<BranchExecutionResult> = join_all(handles)
        .await
        .into_iter()
        .map(|r| {
            r.map_err(|e| anyhow::anyhow!("Parallel branch task panicked or was cancelled: {}", e))
                .and_then(|inner| inner)
        })
        .collect();

    let mut collection = BranchCollection {
        merge_target: None,
        branch_outputs: serde_json::Map::new(),
        failure: None,
    };

    for res in branch_results {
        let (branch_start, results, found_merge) = res?;

        let Some(found_merge) = found_merge else {
            collection.failure = Some(format!(
                "Parallel branch starting at {} did not reach a Merge node",
                branch_start
            ));
            return Ok(collection);
        };

        match collection.merge_target.as_deref() {
            None => collection.merge_target = Some(found_merge.clone()),
            Some(existing) if existing == found_merge => {}
            Some(existing) => {
                collection.failure = Some(format!(
                    "Parallel branches converged on different Merge nodes: expected {:?}, got {}",
                    existing, found_merge
                ));
                return Ok(collection);
            }
        }

        let last_out = results
            .last()
            .and_then(|r| r.output.clone())
            .unwrap_or(serde_json::json!(null));
        collection.branch_outputs.insert(branch_start, last_out);
    }

    Ok(collection)
}

/// Take the first branch that reaches the merge node and cancel the rest.
///
/// A branch that fails, or that ends without reaching a merge node, does not win: the wait
/// continues with the remaining branches. Only when every branch fails does the last error
/// propagate, so one bad branch cannot mask a good one.
async fn collect_first_complete(
    handles: Vec<JoinHandle<BranchExecutionResult>>,
) -> Result<BranchCollection> {
    let abort_handles: Vec<tokio::task::AbortHandle> =
        handles.iter().map(JoinHandle::abort_handle).collect();
    let mut pending = handles;
    let mut last_error: Option<anyhow::Error> = None;
    let mut incomplete_branch: Option<String> = None;

    while !pending.is_empty() {
        let (joined, _index, remaining) = select_all(pending).await;
        pending = remaining;

        let outcome = joined
            .map_err(|e| anyhow::anyhow!("Parallel branch task panicked or was cancelled: {}", e))
            .and_then(|inner| inner);

        match outcome {
            Ok((branch_start, results, Some(found_merge))) => {
                // Cancel the losers before returning; aborting a finished task is a no-op.
                for abort in &abort_handles {
                    abort.abort();
                }
                let last_out = results
                    .last()
                    .and_then(|r| r.output.clone())
                    .unwrap_or(serde_json::json!(null));
                let mut branch_outputs = serde_json::Map::new();
                branch_outputs.insert(branch_start, last_out);
                return Ok(BranchCollection {
                    merge_target: Some(found_merge),
                    branch_outputs,
                    failure: None,
                });
            }
            Ok((branch_start, _results, None)) => {
                incomplete_branch.get_or_insert(branch_start);
            }
            Err(e) => last_error = Some(e),
        }
    }

    if let Some(error) = last_error {
        return Err(error);
    }
    Ok(BranchCollection {
        merge_target: None,
        branch_outputs: serde_json::Map::new(),
        failure: Some(format!(
            "Parallel branch starting at {} did not reach a Merge node",
            incomplete_branch.unwrap_or_else(|| "<unknown>".to_string())
        )),
    })
}

/// Resolve the merge strategy the branches will converge on, before they run.
///
/// Returns `None` when the branches reach different merge nodes or none at all; the caller
/// then falls back to waiting for every branch, which reports the mismatch as a node error.
fn planned_merge_strategy(
    nodes_map: &HashMap<&str, &WorkflowNode>,
    outgoing_edges: &HashMap<&str, Vec<&WorkflowEdge>>,
    branch_starts: &[String],
) -> Option<MergeStrategy> {
    let mut merge_id: Option<String> = None;
    for start in branch_starts {
        let found = first_reachable_merge(nodes_map, outgoing_edges, start)?;
        match merge_id.as_deref() {
            None => merge_id = Some(found),
            Some(existing) if existing == found => {}
            Some(_) => return None,
        }
    }
    let merge_id = merge_id?;
    match nodes_map.get(merge_id.as_str()).map(|n| &n.node_type) {
        Some(NodeType::Merge { strategy }) => Some(strategy.clone()),
        _ => None,
    }
}

/// Breadth-first search for the first `Merge` node reachable from `start`.
fn first_reachable_merge(
    nodes_map: &HashMap<&str, &WorkflowNode>,
    outgoing_edges: &HashMap<&str, Vec<&WorkflowEdge>>,
    start: &str,
) -> Option<String> {
    let mut seen: HashSet<&str> = HashSet::new();
    let mut queue: VecDeque<&str> = VecDeque::new();
    queue.push_back(start);

    while let Some(node_id) = queue.pop_front() {
        if !seen.insert(node_id) {
            continue;
        }
        let node = nodes_map.get(node_id)?;
        if matches!(node.node_type, NodeType::Merge { .. }) {
            return Some(node.id.clone());
        }
        if let Some(edges) = outgoing_edges.get(node_id) {
            for edge in edges {
                queue.push_back(edge.to_node.as_str());
            }
        }
    }
    None
}

fn merge_concatenate(value: &serde_json::Value) -> serde_json::Value {
    let Some(obj) = value.as_object() else {
        return value.clone();
    };
    let mut keys: Vec<&String> = obj.keys().collect();
    keys.sort();
    let values: Vec<serde_json::Value> = keys
        .into_iter()
        .filter_map(|key| obj.get(key).cloned())
        .collect();
    if values.iter().all(serde_json::Value::is_array) {
        let mut out = Vec::new();
        for v in values {
            if let Some(arr) = v.as_array() {
                out.extend(arr.iter().cloned());
            }
        }
        return serde_json::Value::Array(out);
    }
    if values.iter().all(serde_json::Value::is_string) {
        let joined = values
            .into_iter()
            .filter_map(|v| v.as_str().map(str::to_string))
            .collect::<Vec<_>>()
            .join("\n");
        return serde_json::Value::String(joined);
    }
    serde_json::Value::Array(values)
}

fn merge_first(value: &serde_json::Value) -> serde_json::Value {
    let Some(obj) = value.as_object() else {
        return value.clone();
    };
    let mut keys: Vec<&String> = obj.keys().collect();
    keys.sort();
    if let Some(first_key) = keys.first()
        && let Some(v) = obj.get(*first_key)
    {
        return v.clone();
    }
    serde_json::json!(null)
}

fn merge_vote(value: &serde_json::Value) -> serde_json::Value {
    let Some(obj) = value.as_object() else {
        return value.clone();
    };
    let mut counts: HashMap<String, usize> = HashMap::new();
    for v in obj.values() {
        if let Some(s) = v.as_str() {
            let entry = counts.entry(s.to_string()).or_insert(0);
            *entry += 1;
        }
    }
    let mut ranked: Vec<(String, usize)> = counts.into_iter().collect();
    ranked.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    if let Some((winner, _)) = ranked.into_iter().next() {
        return serde_json::Value::String(winner);
    }
    merge_first(value)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::orchestration::workflow::{MergeStrategy, NodeType, WorkflowEdge, WorkflowNode};

    fn agent_node(name: &str) -> WorkflowNode {
        WorkflowNode::new(
            name,
            NodeType::Agent {
                agent_id: None,
                capability: None,
                prompt_template: "task".to_string(),
            },
        )
    }

    #[test]
    fn planned_strategy_resolves_shared_merge_node() {
        let parallel = WorkflowNode::new("fan", NodeType::Parallel);
        let left = agent_node("left");
        let right = agent_node("right");
        let merge = WorkflowNode::new(
            "merge",
            NodeType::Merge {
                strategy: MergeStrategy::FirstComplete,
            },
        );
        let nodes = vec![parallel.clone(), left.clone(), right.clone(), merge.clone()];
        let edges = vec![
            WorkflowEdge::new(parallel.id.clone(), left.id.clone()),
            WorkflowEdge::new(parallel.id, right.id.clone()),
            WorkflowEdge::new(left.id.clone(), merge.id.clone()),
            WorkflowEdge::new(right.id.clone(), merge.id),
        ];
        let nodes_map = build_nodes_map(&nodes);
        let outgoing = build_outgoing_edges(&edges);

        let strategy = planned_merge_strategy(&nodes_map, &outgoing, &[left.id, right.id]);
        assert!(matches!(strategy, Some(MergeStrategy::FirstComplete)));
    }

    #[test]
    fn planned_strategy_is_none_for_divergent_merges() {
        let left = agent_node("left");
        let right = agent_node("right");
        let merge_a = WorkflowNode::new(
            "merge_a",
            NodeType::Merge {
                strategy: MergeStrategy::FirstComplete,
            },
        );
        let merge_b = WorkflowNode::new(
            "merge_b",
            NodeType::Merge {
                strategy: MergeStrategy::AllComplete,
            },
        );
        let nodes = vec![
            left.clone(),
            right.clone(),
            merge_a.clone(),
            merge_b.clone(),
        ];
        let edges = vec![
            WorkflowEdge::new(left.id.clone(), merge_a.id),
            WorkflowEdge::new(right.id.clone(), merge_b.id),
        ];
        let nodes_map = build_nodes_map(&nodes);
        let outgoing = build_outgoing_edges(&edges);

        assert!(planned_merge_strategy(&nodes_map, &outgoing, &[left.id, right.id]).is_none());
    }

    #[test]
    fn merge_helpers_reduce_branch_maps() {
        let value = serde_json::json!({"a": "one", "b": "one", "c": "two"});
        assert_eq!(merge_vote(&value), serde_json::json!("one"));
        assert_eq!(merge_first(&value), serde_json::json!("one"));
        assert_eq!(
            merge_concatenate(&serde_json::json!({"a": [1], "b": [2]})),
            serde_json::json!([1, 2])
        );
    }

    #[tokio::test]
    async fn first_complete_cancels_slower_branches() {
        let winner: JoinHandle<BranchExecutionResult> = tokio::spawn(async {
            Ok((
                "fast".to_string(),
                vec![NodeExecutionResult::ok(
                    "fast",
                    serde_json::json!("done"),
                    Utc::now(),
                    HashMap::new(),
                )],
                Some("merge".to_string()),
            ))
        });
        let loser: JoinHandle<BranchExecutionResult> = tokio::spawn(async {
            tokio::time::sleep(std::time::Duration::from_secs(30)).await;
            Ok(("slow".to_string(), Vec::new(), Some("merge".to_string())))
        });
        let loser_abort = loser.abort_handle();

        let collection = collect_first_complete(vec![winner, loser])
            .await
            .expect("first complete succeeds");

        assert_eq!(collection.merge_target.as_deref(), Some("merge"));
        assert_eq!(collection.branch_outputs.len(), 1);
        assert!(collection.branch_outputs.contains_key("fast"));

        // `abort` only signals the task; the runtime finishes it on a later poll, so give the
        // scheduler a bounded number of turns before the assertion.
        for _ in 0..100 {
            if loser_abort.is_finished() {
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(5)).await;
        }
        assert!(
            loser_abort.is_finished(),
            "the slower branch must be cancelled once a branch wins"
        );
    }

    #[tokio::test]
    async fn first_complete_skips_a_failing_branch() {
        let failing: JoinHandle<BranchExecutionResult> =
            tokio::spawn(async { Err(anyhow::anyhow!("branch blew up")) });
        let winner: JoinHandle<BranchExecutionResult> = tokio::spawn(async {
            tokio::time::sleep(std::time::Duration::from_millis(20)).await;
            Ok((
                "slow-but-good".to_string(),
                vec![NodeExecutionResult::ok(
                    "slow-but-good",
                    serde_json::json!("ok"),
                    Utc::now(),
                    HashMap::new(),
                )],
                Some("merge".to_string()),
            ))
        });

        let collection = collect_first_complete(vec![failing, winner])
            .await
            .expect("a later healthy branch still wins");
        assert!(collection.branch_outputs.contains_key("slow-but-good"));
    }

    #[tokio::test]
    async fn first_complete_propagates_when_all_branches_fail() {
        let a: JoinHandle<BranchExecutionResult> =
            tokio::spawn(async { Err(anyhow::anyhow!("a failed")) });
        let b: JoinHandle<BranchExecutionResult> =
            tokio::spawn(async { Err(anyhow::anyhow!("b failed")) });

        let err = collect_first_complete(vec![a, b])
            .await
            .expect_err("all branches failed");
        assert!(err.to_string().contains("failed"));
    }
}
