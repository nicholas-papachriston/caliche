use crate::orchestration::ExecutionContext;
use anyhow::Result;
use async_trait::async_trait;
use serde_json::Value;
use std::collections::HashMap;
use std::sync::Arc;

/// Pluggable execution interface for workflow nodes.
///
/// The core workflow engine is intentionally decoupled from any particular agent/service layer.
/// Consumers (API server, CLI, JS bindings, etc.) can provide an executor that knows how to:
/// - run an agent completion for `NodeType::Agent`
/// - run custom transforms for `NodeType::Transform`
#[async_trait]
pub trait WorkflowNodeExecutor: Send + Sync {
    /// Execute an agent node and return `(output, metadata)`.
    ///
    /// - `agent_id`: explicit agent identifier, if the runtime supports it.
    /// - `capability`: capability hint for selection, if the runtime supports it.
    async fn execute_agent(
        &self,
        prompt: &str,
        agent_id: Option<&str>,
        capability: Option<&str>,
        context: &ExecutionContext,
    ) -> Result<(Value, HashMap<String, Value>)>;

    /// Execute a transform node.
    ///
    /// Implementations may interpret `function` as a DSL, a registered function name,
    /// or something more elaborate. The engine will pass the current execution context
    /// and the most recent node output (if any).
    async fn transform(
        &self,
        function: &str,
        context: &ExecutionContext,
        last_output: Option<&Value>,
    ) -> Result<Value>;

    /// Execute a custom merge strategy for `MergeStrategy::Custom(name)`.
    ///
    /// - `name`: merge strategy identifier from the workflow definition
    /// - `inputs`: the raw parallel branch outputs stored by the engine (typically an object)
    async fn merge_custom(
        &self,
        name: &str,
        inputs: &Value,
        context: &ExecutionContext,
    ) -> Result<Value>;
}

/// Default executor used when the consumer does not configure one.
///
/// This avoids silently returning mock results. Instead, workflow execution will fail
/// with a clear error message when it reaches an agent/transform node.
#[derive(Debug, Default)]
pub struct NoopWorkflowNodeExecutor;

impl NoopWorkflowNodeExecutor {
    pub fn new() -> Arc<Self> {
        Arc::new(Self)
    }
}

#[async_trait]
impl WorkflowNodeExecutor for NoopWorkflowNodeExecutor {
    async fn execute_agent(
        &self,
        _prompt: &str,
        _agent_id: Option<&str>,
        _capability: Option<&str>,
        _context: &ExecutionContext,
    ) -> Result<(Value, HashMap<String, Value>)> {
        anyhow::bail!("No WorkflowNodeExecutor configured (agent execution unavailable)")
    }

    async fn transform(
        &self,
        _function: &str,
        _context: &ExecutionContext,
        _last_output: Option<&Value>,
    ) -> Result<Value> {
        anyhow::bail!("No WorkflowNodeExecutor configured (transform execution unavailable)")
    }

    async fn merge_custom(
        &self,
        name: &str,
        _inputs: &Value,
        _context: &ExecutionContext,
    ) -> Result<Value> {
        anyhow::bail!(
            "No WorkflowNodeExecutor configured (custom merge '{}' unavailable)",
            name
        )
    }
}
