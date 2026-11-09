//! JavaScript bindings for Caliche Workflow Orchestration
#![expect(
    clippy::needless_pass_by_value,
    reason = "NAPI must own values converted from JavaScript arguments at the ABI boundary"
)]

use caliche::orchestration::builder::{BuiltWorkflow, WorkflowBuilder, merge_strategy_from_str};
use caliche::orchestration::workflow::{ExecutionContext, Workflow, WorkflowEngine};
use napi_derive::napi;
use std::cell::RefCell;
use std::sync::Arc;

/// JavaScript wrapper for `WorkflowBuilder`
#[napi]
pub struct JsWorkflowBuilder {
    inner: RefCell<Option<WorkflowBuilder>>,
}

#[napi]
impl JsWorkflowBuilder {
    /// Create a new `WorkflowBuilder`
    #[napi(constructor, catch_unwind)]
    pub fn new(name: String) -> Self {
        Self {
            inner: RefCell::new(Some(WorkflowBuilder::new(name))),
        }
    }

    /// Add an agent node
    #[napi(catch_unwind)]
    ///
    /// # Errors
    ///
    /// Returns a NAPI error when validation, synchronization, serialization,
    /// or an underlying runtime operation fails.
    pub fn add_agent(&self, name: String, prompt_template: String) -> napi::Result<()> {
        let mut inner = self.inner.borrow_mut();
        let current = inner
            .take()
            .ok_or_else(|| napi::Error::from_reason("Builder already consumed"))?;
        *inner = Some(current.add_agent(name, prompt_template));
        Ok(())
    }

    /// Add an agent node with capability
    #[napi(catch_unwind)]
    pub fn add_agent_with_capability(
        &self,
        name: String,
        capability: String,
        prompt_template: String,
    ) -> napi::Result<()> {
        let mut inner = self.inner.borrow_mut();
        let current = inner
            .take()
            .ok_or_else(|| napi::Error::from_reason("Builder already consumed"))?;
        *inner = Some(current.add_agent_with_capability(name, capability, prompt_template));
        Ok(())
    }

    /// Add a condition node
    #[napi(catch_unwind)]
    ///
    /// # Errors
    ///
    /// Returns a NAPI error when validation, synchronization, serialization,
    /// or an underlying runtime operation fails.
    pub fn add_condition(&self, name: String, predicate: String) -> napi::Result<()> {
        let mut inner = self.inner.borrow_mut();
        let current = inner
            .take()
            .ok_or_else(|| napi::Error::from_reason("Builder already consumed"))?;
        *inner = Some(current.add_condition(name, predicate));
        Ok(())
    }

    /// Add a parallel node (fan-out with merge strategy `all_complete`).
    #[napi(catch_unwind)]
    pub fn add_parallel(&self, branches: Vec<ParallelBranch>) -> napi::Result<()> {
        let mut inner = self.inner.borrow_mut();
        let current = inner
            .take()
            .ok_or_else(|| napi::Error::from_reason("Builder already consumed"))?;
        let rust_branches: Vec<(String, String)> =
            branches.into_iter().map(|b| (b.name, b.prompt)).collect();
        if rust_branches.is_empty() {
            *inner = Some(current);
            return Err(napi::Error::from_reason(
                "add_parallel requires at least one branch",
            ));
        }
        *inner = Some(current.parallel(rust_branches));
        Ok(())
    }

    /// Fan out to parallel branches without merging; call `collect_merge` (or `add_merge`) after.
    #[napi(catch_unwind)]
    ///
    /// # Errors
    ///
    /// Returns a NAPI error when validation, synchronization, serialization,
    /// or an underlying runtime operation fails.
    pub fn fan_out(&self, branches: Vec<ParallelBranch>) -> napi::Result<()> {
        let mut inner = self.inner.borrow_mut();
        let current = inner
            .take()
            .ok_or_else(|| napi::Error::from_reason("Builder already consumed"))?;
        let rust_branches: Vec<(String, String)> =
            branches.into_iter().map(|b| (b.name, b.prompt)).collect();
        if rust_branches.is_empty() {
            *inner = Some(current);
            return Err(napi::Error::from_reason(
                "fan_out requires at least one branch",
            ));
        }
        *inner = Some(current.fan_out(rust_branches));
        Ok(())
    }

    /// Complete a pending fan-out with a merge node and strategy (see `merge_strategy_from_str`).
    #[napi(js_name = "collectMerge", catch_unwind)]
    ///
    /// # Errors
    ///
    /// Returns a NAPI error when validation, synchronization, serialization,
    /// or an underlying runtime operation fails.
    pub fn collect_merge(&self, merge_node_name: String, strategy: String) -> napi::Result<()> {
        let mut inner = self.inner.borrow_mut();
        let current = inner
            .take()
            .ok_or_else(|| napi::Error::from_reason("Builder already consumed"))?;
        if !current.has_pending_fan_out() {
            *inner = Some(current);
            return Err(napi::Error::from_reason(
                "collect_merge requires a pending fan-out: call fan_out first",
            ));
        }
        let strat = merge_strategy_from_str(&strategy)
            .map_err(|e| napi::Error::from_reason(e.to_string()))?;
        *inner = Some(current.collect_named(merge_node_name, strat));
        Ok(())
    }

    /// Fan out then merge with the given strategy (single call).
    #[napi(js_name = "addParallelWithMergeStrategy", catch_unwind)]
    ///
    /// # Errors
    ///
    /// Returns a NAPI error when validation, synchronization, serialization,
    /// or an underlying runtime operation fails.
    pub fn add_parallel_with_merge_strategy(
        &self,
        branches: Vec<ParallelBranch>,
        strategy: String,
    ) -> napi::Result<()> {
        let mut inner = self.inner.borrow_mut();
        let current = inner
            .take()
            .ok_or_else(|| napi::Error::from_reason("Builder already consumed"))?;
        let rust_branches: Vec<(String, String)> =
            branches.into_iter().map(|b| (b.name, b.prompt)).collect();
        if rust_branches.is_empty() {
            *inner = Some(current);
            return Err(napi::Error::from_reason(
                "add_parallel_with_merge_strategy requires at least one branch",
            ));
        }
        let strat = merge_strategy_from_str(&strategy)
            .map_err(|e| napi::Error::from_reason(e.to_string()))?;
        *inner = Some(current.parallel_with_strategy(rust_branches, strat));
        Ok(())
    }

    /// Add a merge node for a pending fan-out. Errors if there is no pending fan-out.
    #[napi(catch_unwind)]
    ///
    /// # Errors
    ///
    /// Returns a NAPI error when validation, synchronization, serialization,
    /// or an underlying runtime operation fails.
    pub fn add_merge(&self, name: String, strategy: String) -> napi::Result<()> {
        self.collect_merge(name, strategy)
    }

    /// Add a loop node (named loop container).
    #[napi(js_name = "addLoop", catch_unwind)]
    pub fn add_loop(
        &self,
        name: String,
        max_iterations: u32,
        condition: String,
        body_prompt: String,
    ) -> napi::Result<()> {
        let mut inner = self.inner.borrow_mut();
        let current = inner
            .take()
            .ok_or_else(|| napi::Error::from_reason("Builder already consumed"))?;
        *inner = Some(current.loop_while_with_name(
            name,
            condition,
            max_iterations as usize,
            body_prompt,
        ));
        Ok(())
    }

    /// Add a human-in-the-loop interrupt node.
    #[napi(js_name = "addHumanInterrupt", catch_unwind)]
    ///
    /// # Errors
    ///
    /// Returns a NAPI error when validation, synchronization, serialization,
    /// or an underlying runtime operation fails.
    pub fn add_human_interrupt(
        &self,
        name: String,
        message: String,
        required_fields: Option<Vec<String>>,
    ) -> napi::Result<()> {
        let mut inner = self.inner.borrow_mut();
        let current = inner
            .take()
            .ok_or_else(|| napi::Error::from_reason("Builder already consumed"))?;
        *inner = Some(current.human_interrupt(name, message, required_fields));
        Ok(())
    }

    /// Connect two nodes by name or id with an optional edge condition (default `"true"`).
    #[napi(js_name = "connectNodes", catch_unwind)]
    ///
    /// # Errors
    ///
    /// Returns a NAPI error when validation, synchronization, serialization,
    /// or an underlying runtime operation fails.
    pub fn connect_nodes(
        &self,
        from: String,
        to: String,
        condition: Option<String>,
    ) -> napi::Result<()> {
        let mut inner = self.inner.borrow_mut();
        let current = inner
            .take()
            .ok_or_else(|| napi::Error::from_reason("Builder already consumed"))?;
        let cond = condition.unwrap_or_else(|| "true".to_string());
        let next = current
            .try_add_edge_with_condition(from, to, cond)
            .map_err(|e| napi::Error::from_reason(e.to_string()))?;
        *inner = Some(next);
        Ok(())
    }

    /// Set workflow entry point (node name or id) for state-machine mode.
    #[napi(js_name = "setEntryPoint", catch_unwind)]
    ///
    /// # Errors
    ///
    /// Returns a NAPI error when validation, synchronization, serialization,
    /// or an underlying runtime operation fails.
    pub fn set_entry_point(&self, name: String) -> napi::Result<()> {
        let mut inner = self.inner.borrow_mut();
        let current = inner
            .take()
            .ok_or_else(|| napi::Error::from_reason("Builder already consumed"))?;
        *inner = Some(current.with_entry_point(name));
        Ok(())
    }

    /// Build the workflow
    #[napi(catch_unwind)]
    pub fn build(&self) -> napi::Result<JsWorkflow> {
        let mut inner = self.inner.borrow_mut();
        let builder = inner
            .take()
            .ok_or_else(|| napi::Error::from_reason("Builder already consumed"))?;
        let built = builder.build();
        Ok(JsWorkflow {
            inner: Arc::new(built),
        })
    }
}

#[napi(object)]
pub struct ParallelBranch {
    pub name: String,
    pub prompt: String,
}

/// JavaScript wrapper for `BuiltWorkflow`
#[napi]
pub struct JsWorkflow {
    inner: Arc<BuiltWorkflow>,
}

#[napi]
impl JsWorkflow {
    /// Get workflow ID
    #[napi(catch_unwind)]
    pub fn id(&self) -> String {
        self.inner.id().to_string()
    }

    /// Get workflow name
    #[napi(catch_unwind)]
    pub fn name(&self) -> String {
        self.inner.name().to_string()
    }

    /// Get workflow description
    #[napi(catch_unwind)]
    pub fn description(&self) -> String {
        self.inner.description().to_string()
    }

    /// Serialize workflow definition to JSON (for inspection/tests).
    #[napi(catch_unwind)]
    ///
    /// # Errors
    ///
    /// Returns a NAPI error when validation, synchronization, serialization,
    /// or an underlying runtime operation fails.
    pub fn to_json(&self) -> napi::Result<String> {
        self.inner
            .to_json()
            .map_err(|e| napi::Error::from_reason(e.to_string()))
    }

    /// Execute this workflow
    #[napi]
    pub async fn execute(
        &self,
        initial_state: Option<serde_json::Value>,
    ) -> napi::Result<WorkflowExecutionResult> {
        let engine = WorkflowEngine::new();
        let workflow_arc: Arc<dyn Workflow> = self.inner.clone();
        let workflow_id = self.inner.id().to_string();
        let mut context = if let Some(init_state) = initial_state {
            // Create context with initial state data in variables
            let mut ctx = ExecutionContext::new(workflow_id);
            // Store initial state as a variable
            ctx.set_variable("initial_state".to_string(), init_state);
            ctx
        } else {
            ExecutionContext::new(workflow_id)
        };

        let result = workflow_arc.execute(&engine, &mut context).await;
        match result {
            Ok(value) => {
                // Serialize context state to JSON
                let state_value = serde_json::json!({
                    "workflow_id": context.workflow_id,
                    "execution_id": context.execution_id,
                    "variables": context.variables,
                    "state": format!("{:?}", context.state),
                });
                Ok(WorkflowExecutionResult {
                    result: value,
                    state: state_value,
                })
            }
            Err(e) => Err(napi::Error::from_reason(format!(
                "Workflow execution failed: {e}"
            ))),
        }
    }
}

/// JavaScript wrapper for `WorkflowEngine`
#[napi]
pub struct JsWorkflowEngine {
    pub(crate) inner: Arc<WorkflowEngine>,
}

#[napi]
impl JsWorkflowEngine {
    /// Create a new `WorkflowEngine`
    #[napi(constructor, catch_unwind)]
    pub fn new() -> Self {
        Self {
            inner: Arc::new(WorkflowEngine::new()),
        }
    }

    /// Register a built workflow for later execution.
    #[napi]
    ///
    /// # Errors
    ///
    /// Returns a NAPI error when validation, synchronization, serialization,
    /// or an underlying runtime operation fails.
    pub async fn register_workflow(&self, workflow: &JsWorkflow) -> napi::Result<()> {
        let wf: Arc<dyn Workflow> = workflow.inner.clone();
        self.inner
            .register_workflow(wf)
            .await
            .map_err(|e| napi::Error::from_reason(e.to_string()))
    }
}

#[napi(object)]
pub struct WorkflowExecutionResult {
    pub result: serde_json::Value,
    pub state: serde_json::Value,
}
