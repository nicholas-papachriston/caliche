use anyhow::{Context, Result};
use async_trait::async_trait;
use chrono::{DateTime, Utc};
use dashmap::DashMap;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;
use tokio::sync::{RwLock, Semaphore};
use uuid::Uuid;

use crate::config::OrchestrationConfig;
use crate::orchestration::workflow_executor::{NoopWorkflowNodeExecutor, WorkflowNodeExecutor};

mod agent_node;
mod budget;
mod constants;
mod engine;
mod engine_exec;
mod parallel;
pub mod transition;
pub(crate) mod validation;
use constants::{MAX_STEPS, META_FORCED_NEXT_NODE_ID, VAR_ERROR, VAR_LAST_OUTPUT, VAR_RESULT};
use transition::select_next_edge;

/// Policy applied when a node returns `success: false`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum NodeErrorPolicy {
    /// Stop workflow execution and mark the run failed (default).
    #[default]
    Halt,
    /// Record the failure and continue along the next edge.
    Continue,
    /// Retry an agent node before the failure is recorded, then halt.
    ///
    /// Only agent nodes retry; other node types fail as they do under [`Self::Halt`].
    Retry,
}

/// Default age after which completed executions may be pruned.
pub const DEFAULT_EXECUTION_RETENTION_SECS: u64 = 86_400;

/// Workflow execution state
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum WorkflowState {
    Pending,
    Running,
    Paused,
    Completed,
    Failed,
    Cancelled,
}

/// Node type in the workflow DAG
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NodeType {
    /// Execute an agent task
    Agent {
        agent_id: Option<String>, // None = auto-select by capability
        capability: Option<String>,
        prompt_template: String,
    },
    /// Conditional branching
    Condition {
        predicate: String, // JSONPath or expression
    },
    /// Parallel execution
    Parallel,
    /// Merge parallel results
    Merge { strategy: MergeStrategy },
    /// Loop/iteration
    Loop {
        max_iterations: usize,
        condition: String,
    },
    /// Custom transformation
    Transform { function: String },
    /// Human-in-the-loop interrupt - pauses execution for human approval/feedback
    HumanInterrupt {
        message: String,                      // Message to display to human reviewer
        required_fields: Option<Vec<String>>, // Fields that must be provided (e.g., ["approval", "notes"])
    },
}

/// Strategy for merging parallel results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MergeStrategy {
    Concatenate,
    Vote, // Majority vote
    FirstComplete,
    AllComplete,
    Custom(String),
}

/// A node in the workflow DAG
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkflowNode {
    pub id: String,
    pub name: String,
    pub node_type: NodeType,
    pub metadata: serde_json::Value,
}

impl WorkflowNode {
    pub fn new(name: impl Into<String>, node_type: NodeType) -> Self {
        Self {
            id: Uuid::now_v7().to_string(),
            name: name.into(),
            node_type,
            metadata: serde_json::json!({}),
        }
    }

    pub fn with_metadata(mut self, metadata: serde_json::Value) -> Self {
        self.metadata = metadata;
        self
    }
}

/// Edge connecting workflow nodes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkflowEdge {
    pub from_node: String,
    pub to_node: String,
    pub condition: Option<String>, // Optional condition for conditional edges
    pub label: Option<String>,
}

impl WorkflowEdge {
    pub fn new(from_node: impl Into<String>, to_node: impl Into<String>) -> Self {
        Self {
            from_node: from_node.into(),
            to_node: to_node.into(),
            condition: None,
            label: None,
        }
    }

    pub fn with_condition(mut self, condition: impl Into<String>) -> Self {
        self.condition = Some(condition.into());
        self
    }

    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into());
        self
    }
}

/// Pending human interrupt
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PendingInterrupt {
    pub interrupt_id: String,
    pub execution_id: String,
    pub node_id: String,
    pub node_name: String,
    pub message: String,
    pub required_fields: Option<Vec<String>>,
    pub created_at: DateTime<Utc>,
    pub context_snapshot: HashMap<String, serde_json::Value>,
}

/// Execution context for a workflow
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionContext {
    pub workflow_id: String,
    pub execution_id: String,
    pub variables: HashMap<String, serde_json::Value>,
    pub node_results: HashMap<String, NodeExecutionResult>,
    pub start_time: DateTime<Utc>,
    pub end_time: Option<DateTime<Utc>>,
    pub state: WorkflowState,
    pub pending_interrupt: Option<PendingInterrupt>,
    pub current_node_id: Option<String>,
    #[serde(skip)]
    cooperative_cancel: Option<Arc<AtomicBool>>,
}

impl ExecutionContext {
    pub fn new(workflow_id: impl Into<String>) -> Self {
        Self {
            workflow_id: workflow_id.into(),
            execution_id: Uuid::now_v7().to_string(),
            variables: HashMap::new(),
            node_results: HashMap::new(),
            start_time: Utc::now(),
            end_time: None,
            state: WorkflowState::Pending,
            pending_interrupt: None,
            current_node_id: None,
            cooperative_cancel: None,
        }
    }

    /// Returns true when [`WorkflowEngine::cancel_execution`] was called for this run.
    pub fn is_cancelled(&self) -> bool {
        self.cooperative_cancel
            .as_ref()
            .is_some_and(|flag| flag.load(Ordering::Relaxed))
    }

    pub fn set_variable(&mut self, key: impl Into<String>, value: serde_json::Value) {
        self.variables.insert(key.into(), value);
    }

    pub fn get_variable(&self, key: &str) -> Option<&serde_json::Value> {
        self.variables.get(key)
    }

    pub fn record_node_result(&mut self, node_id: impl Into<String>, result: NodeExecutionResult) {
        self.node_results.insert(node_id.into(), result);
    }

    pub fn get_node_result(&self, node_id: &str) -> Option<&NodeExecutionResult> {
        self.node_results.get(node_id)
    }
}

/// Result of a parallel branch: (branch_start_id, node_results, merge_target_if_found).
type BranchExecutionResult = Result<(String, Vec<NodeExecutionResult>, Option<String>)>;

/// Result of executing a node
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeExecutionResult {
    pub node_id: String,
    pub success: bool,
    pub output: Option<serde_json::Value>,
    pub error: Option<String>,
    pub start_time: DateTime<Utc>,
    pub end_time: DateTime<Utc>,
    pub metadata: HashMap<String, serde_json::Value>,
}

impl NodeExecutionResult {
    /// Build a successful result.
    pub fn ok(
        node_id: impl Into<String>,
        output: serde_json::Value,
        start_time: DateTime<Utc>,
        metadata: HashMap<String, serde_json::Value>,
    ) -> Self {
        Self {
            node_id: node_id.into(),
            success: true,
            output: Some(output),
            error: None,
            start_time,
            end_time: Utc::now(),
            metadata,
        }
    }

    /// Build an error result (execution failed but we still return Ok(NodeExecutionResult)).
    pub fn err(
        node_id: impl Into<String>,
        error: impl Into<String>,
        start_time: DateTime<Utc>,
        metadata: HashMap<String, serde_json::Value>,
    ) -> Self {
        Self {
            node_id: node_id.into(),
            success: false,
            output: None,
            error: Some(error.into()),
            start_time,
            end_time: Utc::now(),
            metadata,
        }
    }
}

/// Parse optional per-node parallel branch timeout from workflow metadata.
///
/// Only a JSON integer `branch_timeout_ms` is accepted; strings/floats are ignored
/// so misconfigured metadata fails open (no timeout) rather than panicking.
pub(crate) fn branch_timeout_from_metadata(
    metadata: &serde_json::Value,
) -> Option<std::time::Duration> {
    metadata
        .get("branch_timeout_ms")
        .and_then(|v| v.as_u64())
        .map(std::time::Duration::from_millis)
}

fn build_nodes_map(nodes: &[WorkflowNode]) -> HashMap<&str, &WorkflowNode> {
    nodes.iter().map(|n| (n.id.as_str(), n)).collect()
}

fn build_outgoing_edges<'a>(edges: &'a [WorkflowEdge]) -> HashMap<&'a str, Vec<&'a WorkflowEdge>> {
    let mut out: HashMap<&'a str, Vec<&'a WorkflowEdge>> = HashMap::new();
    for edge in edges {
        out.entry(edge.from_node.as_str()).or_default().push(edge);
    }
    out
}

fn get_forced_next_node(results: &[NodeExecutionResult]) -> Option<String> {
    results
        .last()
        .and_then(|r| r.metadata.get(META_FORCED_NEXT_NODE_ID))
        .and_then(|v| v.as_str())
        .map(|s| s.to_string())
}

fn record_and_update_context(
    context: &mut ExecutionContext,
    node: &WorkflowNode,
    result: &NodeExecutionResult,
) {
    context.record_node_result(node.id.clone(), result.clone());
    let output = result.output.clone().unwrap_or(serde_json::json!(null));
    publish_template_outputs(context, node, &output);
}

/// True when `key` is safe to expose as a workflow prompt placeholder.
fn is_template_alias_key(key: &str) -> bool {
    !key.is_empty()
        && !key.starts_with("__")
        && key
            .chars()
            .all(|character| character.is_ascii_alphanumeric() || character == '_')
}

/// Publish node output under the keys templates actually use: `__last_output`,
/// `last_output`, `node_{id}_output`, plus the node id and name when they are
/// valid placeholder identifiers (`{{research}}`).
pub(super) fn publish_template_outputs(
    context: &mut ExecutionContext,
    node: &WorkflowNode,
    output: &serde_json::Value,
) {
    context.set_variable(VAR_LAST_OUTPUT.to_owned(), output.clone());
    context.set_variable("last_output".to_owned(), output.clone());
    context.set_variable(format!("node_{}_output", node.id), output.clone());
    if is_template_alias_key(&node.id) {
        context.set_variable(node.id.clone(), output.clone());
    }
    if node.name != node.id && is_template_alias_key(&node.name) {
        context.set_variable(node.name.clone(), output.clone());
    }
}

/// Trait for workflow definition
#[async_trait]
pub trait Workflow: Send + Sync + std::fmt::Debug {
    /// Get workflow ID
    fn id(&self) -> &str;

    /// Get workflow name
    fn name(&self) -> &str;

    /// Get workflow description
    fn description(&self) -> &str;

    /// Get workflow nodes
    fn nodes(&self) -> &[WorkflowNode];

    /// Get workflow edges
    fn edges(&self) -> &[WorkflowEdge];

    /// Get entry point for state machine execution (None for DAG mode)
    fn entry_point(&self) -> Option<&str> {
        None
    }

    /// Execute the workflow
    async fn execute(
        &self,
        engine: &WorkflowEngine,
        context: &mut ExecutionContext,
    ) -> Result<serde_json::Value>;

    /// Validate workflow structure
    fn validate(&self) -> Result<()> {
        let entry_point = match self.entry_point() {
            Some(name) => validation::EntryPointRef::NodeName(name),
            None => validation::EntryPointRef::None,
        };
        validation::validate_workflow(self.nodes(), self.edges(), entry_point)
    }
}

/// Default cap on how many branches of a single `NodeType::Parallel` node are allowed to run
/// concurrently. Without a bound, a workflow with a large fan-out spawns one Tokio task (each
/// carrying its own cloned `ExecutionContext`) per branch with no limit, which can exhaust
/// memory/fds or overwhelm downstream providers when the branch count is data-driven.
const PARALLEL_BRANCH_CONCURRENCY_DEFAULT: usize = 16;

fn configured_parallel_branch_concurrency() -> usize {
    std::env::var("CALICHE_WORKFLOW_PARALLEL_CONCURRENCY")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(PARALLEL_BRANCH_CONCURRENCY_DEFAULT)
}

/// Default retry attempts for an agent node under [`NodeErrorPolicy::Retry`].
const DEFAULT_NODE_RETRY_ATTEMPTS: u32 = 2;

/// Default base backoff between agent node retries.
const DEFAULT_NODE_RETRY_BACKOFF: Duration = Duration::from_millis(250);

/// Workflow execution engine
pub struct WorkflowEngine {
    workflows: Arc<DashMap<String, Arc<dyn Workflow>>>,
    /// Per-execution context. Inner `RwLock` covers mutable run state.
    executions: Arc<DashMap<String, Arc<RwLock<ExecutionContext>>>>,
    /// Cooperative cancellation flags keyed by execution id.
    cancellation_flags: Arc<DashMap<String, Arc<AtomicBool>>>,
    executor: Arc<dyn WorkflowNodeExecutor>,
    node_error_policy: NodeErrorPolicy,
    /// Bounds how many branches of a `NodeType::Parallel` node run at once (see
    /// `run_parallel_node`). Shared across clones of the engine so the limit applies
    /// per-engine-instance rather than per execution.
    parallel_branch_semaphore: Arc<Semaphore>,
    /// Recorded outputs of agent nodes that declared an idempotency key.
    idempotency_cache: Arc<DashMap<String, serde_json::Value>>,
    node_retry_attempts: u32,
    node_retry_backoff: Duration,
    /// Maximum agent nodes one run may start. `None` means no node ceiling.
    max_nodes: Option<usize>,
    /// Maximum tokens one run may spend, as reported by the executor. `None` means no
    /// token ceiling.
    max_tokens: Option<u64>,
    /// Recorded spend per execution id (see [`budget`]).
    run_usage: Arc<DashMap<String, budget::RunUsage>>,
}

#[cfg(test)]
mod tests;
