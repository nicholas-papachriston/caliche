//! Python bindings for Caliche Workflow Orchestration

use crate::block_on_detached;
use crate::conversions::{json_to_python_dict, python_to_json_value};
use caliche::orchestration::builder::{BuiltWorkflow, WorkflowBuilder, merge_strategy_from_str};
use caliche::orchestration::workflow::{Workflow, WorkflowEngine, WorkflowState};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use std::collections::HashMap;
use std::sync::Arc;

/// Python wrapper for `WorkflowBuilder`
#[pyclass]
pub struct PyWorkflowBuilder {
    inner: Option<WorkflowBuilder>,
}

#[pymethods]
impl PyWorkflowBuilder {
    #[new]
    pub fn new(name: String) -> Self {
        Self {
            inner: Some(WorkflowBuilder::new(name)),
        }
    }

    /// Add an agent node
    pub fn add_agent(&mut self, name: String, prompt_template: String) -> PyResult<()> {
        let builder = self
            .inner
            .take()
            .ok_or_else(|| PyRuntimeError::new_err("Builder already consumed"))?;
        self.inner = Some(builder.add_agent(name, prompt_template));
        Ok(())
    }

    /// Add an agent node with capability
    pub fn add_agent_with_capability(
        &mut self,
        name: String,
        capability: String,
        prompt_template: String,
    ) -> PyResult<()> {
        let builder = self
            .inner
            .take()
            .ok_or_else(|| PyRuntimeError::new_err("Builder already consumed"))?;
        self.inner = Some(builder.add_agent_with_capability(name, capability, prompt_template));
        Ok(())
    }

    /// Add a condition node (uses branch internally)
    pub fn add_condition(&mut self, name: &str, predicate: String) -> PyResult<()> {
        let builder = self
            .inner
            .take()
            .ok_or_else(|| PyRuntimeError::new_err("Builder already consumed"))?;
        self.inner =
            Some(builder.branch(predicate, format!("{name}_true"), format!("{name}_false")));
        Ok(())
    }

    /// Add a parallel node
    pub fn add_parallel(&mut self, branches: &Bound<'_, PyList>) -> PyResult<()> {
        let mut branch_vec = Vec::new();
        for item in branches.iter() {
            if let Ok(dict) = item.cast::<PyDict>() {
                let name = dict
                    .get_item("name")?
                    .ok_or_else(|| PyRuntimeError::new_err("Missing 'name' in branch"))?
                    .extract::<String>()?;
                let prompt = dict
                    .get_item("prompt")?
                    .ok_or_else(|| PyRuntimeError::new_err("Missing 'prompt' in branch"))?
                    .extract::<String>()?;
                branch_vec.push((name, prompt));
            }
        }

        if branch_vec.is_empty() {
            return Err(PyRuntimeError::new_err(
                "add_parallel requires at least one branch dict with name and prompt",
            ));
        }

        let builder = self
            .inner
            .take()
            .ok_or_else(|| PyRuntimeError::new_err("Builder already consumed"))?;
        self.inner = Some(builder.parallel(branch_vec));
        Ok(())
    }

    /// Fan out to parallel branches without merging; call `collect_merge` or `add_merge` next.
    pub fn fan_out(&mut self, branches: &Bound<'_, PyList>) -> PyResult<()> {
        let mut branch_vec = Vec::new();
        for item in branches.iter() {
            if let Ok(dict) = item.cast::<PyDict>() {
                let name = dict
                    .get_item("name")?
                    .ok_or_else(|| PyRuntimeError::new_err("Missing 'name' in branch"))?
                    .extract::<String>()?;
                let prompt = dict
                    .get_item("prompt")?
                    .ok_or_else(|| PyRuntimeError::new_err("Missing 'prompt' in branch"))?
                    .extract::<String>()?;
                branch_vec.push((name, prompt));
            }
        }

        if branch_vec.is_empty() {
            return Err(PyRuntimeError::new_err(
                "fan_out requires at least one branch dict with name and prompt",
            ));
        }

        let builder = self
            .inner
            .take()
            .ok_or_else(|| PyRuntimeError::new_err("Builder already consumed"))?;
        self.inner = Some(builder.fan_out(branch_vec));
        Ok(())
    }

    /// Complete a pending fan-out with a merge node and strategy.
    pub fn collect_merge(&mut self, merge_node_name: String, strategy: &str) -> PyResult<()> {
        let builder = self
            .inner
            .take()
            .ok_or_else(|| PyRuntimeError::new_err("Builder already consumed"))?;
        if !builder.has_pending_fan_out() {
            self.inner = Some(builder);
            return Err(PyRuntimeError::new_err(
                "collect_merge requires a pending fan-out: call fan_out first",
            ));
        }
        let strat = merge_strategy_from_str(strategy)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        self.inner = Some(builder.collect_named(merge_node_name, strat));
        Ok(())
    }

    /// Parallel branches with an explicit merge strategy.
    pub fn add_parallel_with_merge_strategy(
        &mut self,
        branches: &Bound<'_, PyList>,
        strategy: &str,
    ) -> PyResult<()> {
        let mut branch_vec = Vec::new();
        for item in branches.iter() {
            if let Ok(dict) = item.cast::<PyDict>() {
                let name = dict
                    .get_item("name")?
                    .ok_or_else(|| PyRuntimeError::new_err("Missing 'name' in branch"))?
                    .extract::<String>()?;
                let prompt = dict
                    .get_item("prompt")?
                    .ok_or_else(|| PyRuntimeError::new_err("Missing 'prompt' in branch"))?
                    .extract::<String>()?;
                branch_vec.push((name, prompt));
            }
        }

        if branch_vec.is_empty() {
            return Err(PyRuntimeError::new_err(
                "add_parallel_with_merge_strategy requires at least one branch dict with name and prompt",
            ));
        }

        let builder = self
            .inner
            .take()
            .ok_or_else(|| PyRuntimeError::new_err("Builder already consumed"))?;
        let strat = merge_strategy_from_str(strategy)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        self.inner = Some(builder.parallel_with_strategy(branch_vec, strat));
        Ok(())
    }

    /// Add a merge node for a pending fan-out (requires prior `fan_out`).
    pub fn add_merge(&mut self, name: String, strategy: &str) -> PyResult<()> {
        self.collect_merge(name, strategy)
    }

    /// Add a loop node
    pub fn add_loop(
        &mut self,
        _name: String,
        max_iterations: usize,
        condition: String,
        body_prompt: String,
    ) -> PyResult<()> {
        let builder = self
            .inner
            .take()
            .ok_or_else(|| PyRuntimeError::new_err("Builder already consumed"))?;
        self.inner = Some(builder.loop_while(condition, max_iterations, body_prompt));
        Ok(())
    }
    /// Add a human interrupt node
    #[pyo3(signature = (name, message, required_fields=None))]
    pub fn add_human_interrupt(
        &mut self,
        name: String,
        message: String,
        required_fields: Option<&Bound<'_, PyList>>,
    ) -> PyResult<()> {
        let fields = required_fields.map(|list| {
            let mut vec = Vec::new();
            for item in list.iter() {
                if let Ok(s) = item.extract::<String>() {
                    vec.push(s);
                }
            }
            vec
        });

        let builder = self
            .inner
            .take()
            .ok_or_else(|| PyRuntimeError::new_err("Builder already consumed"))?;
        self.inner = Some(builder.human_interrupt(name, message, fields));
        Ok(())
    }

    /// Connect nodes
    #[pyo3(signature = (from, to, condition=None))]
    pub fn connect_nodes(
        &mut self,
        from: String,
        to: String,
        condition: Option<String>,
    ) -> PyResult<()> {
        let builder = self
            .inner
            .take()
            .ok_or_else(|| PyRuntimeError::new_err("Builder already consumed"))?;
        let cond = condition.unwrap_or_else(|| "true".to_string());
        let builder = builder
            .try_add_edge_with_condition(from, to, cond)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        self.inner = Some(builder);
        Ok(())
    }

    /// Set entry point
    pub fn set_entry_point(&mut self, name: String) -> PyResult<()> {
        let builder = self
            .inner
            .take()
            .ok_or_else(|| PyRuntimeError::new_err("Builder already consumed"))?;
        self.inner = Some(builder.with_entry_point(name));
        Ok(())
    }

    /// Build the workflow
    pub fn build(&mut self) -> PyResult<PyWorkflow> {
        let built = self
            .inner
            .take()
            .ok_or_else(|| PyRuntimeError::new_err("Builder already consumed"))?;
        Ok(PyWorkflow {
            inner: Arc::new(built.build()),
        })
    }
}

/// Python wrapper for `BuiltWorkflow`
#[pyclass]
pub struct PyWorkflow {
    inner: Arc<BuiltWorkflow>,
}

#[pymethods]
impl PyWorkflow {
    /// Get workflow ID
    pub fn id(&self) -> String {
        self.inner.id().to_string()
    }

    /// Get workflow name
    pub fn name(&self) -> String {
        self.inner.name().to_string()
    }

    /// Get workflow description
    pub fn description(&self) -> String {
        self.inner.description().to_string()
    }

    /// Serialize workflow definition to JSON.
    pub fn to_json(&self) -> PyResult<String> {
        self.inner
            .to_json()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }
}

/// Python wrapper for `WorkflowEngine`
#[pyclass]
pub struct PyWorkflowEngine {
    pub(crate) inner: Arc<WorkflowEngine>,
}

impl PyWorkflowEngine {
    /// Get inner `WorkflowEngine` (for use by other modules)
    pub(crate) fn get_inner(&self) -> Arc<WorkflowEngine> {
        self.inner.clone()
    }
}

#[pymethods]
impl PyWorkflowEngine {
    #[new]
    pub fn new() -> Self {
        Self {
            inner: Arc::new(WorkflowEngine::new()),
        }
    }

    /// Register a workflow with the engine
    pub fn register_workflow(&self, py: Python<'_>, workflow: &PyWorkflow) -> PyResult<()> {
        let engine_clone = self.inner.clone();
        let workflow_clone = workflow.inner.clone();

        block_on_detached(py, async move {
            let workflow_arc: Arc<dyn Workflow> = workflow_clone.clone();
            engine_clone.register_workflow(workflow_arc).await
        })?
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to register workflow: {e}")))?;

        Ok(())
    }

    /// Start workflow execution
    pub fn start_execution(&self, py: Python<'_>, workflow_id: String) -> PyResult<String> {
        let engine_clone = self.inner.clone();

        block_on_detached(py, async move {
            engine_clone.start_execution(&workflow_id).await
        })?
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to start execution: {e}")))
    }

    /// Get execution result
    pub fn get_execution_result(
        &self,
        py: Python<'_>,
        execution_id: String,
    ) -> PyResult<Py<PyAny>> {
        let engine_clone = self.inner.clone();

        let result = block_on_detached(py, async move {
            engine_clone.get_execution_result(&execution_id).await
        })?;

        match result {
            Ok(value) => {
                let py_dict = json_to_python_dict(py, &value)?;
                Ok(py_dict.into())
            }
            Err(e) => Err(PyRuntimeError::new_err(format!(
                "Failed to get execution result: {e}"
            ))),
        }
    }

    /// Get execution state
    pub fn get_execution_state(&self, py: Python<'_>, execution_id: String) -> PyResult<String> {
        let engine_clone = self.inner.clone();

        let state = block_on_detached(py, async move {
            if let Some(context_arc) = engine_clone.get_execution(&execution_id).await {
                let state_str = {
                    let context = context_arc.read().await;
                    match context.state {
                        WorkflowState::Pending => "pending",
                        WorkflowState::Running => "running",
                        WorkflowState::Paused => "paused",
                        WorkflowState::Completed => "completed",
                        WorkflowState::Failed => "failed",
                        WorkflowState::Cancelled => "cancelled",
                    }
                };
                Some(state_str.to_string())
            } else {
                None
            }
        })?;

        state.ok_or_else(|| PyRuntimeError::new_err("Execution not found"))
    }

    /// Get pending interrupts
    pub fn get_pending_interrupts(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let engine_clone = self.inner.clone();

        let interrupts =
            block_on_detached(
                py,
                async move { engine_clone.list_pending_interrupts().await },
            )?;

        let list = PyList::empty(py);
        for interrupt in interrupts {
            let dict = PyDict::new(py);
            dict.set_item("interrupt_id", interrupt.interrupt_id)?;
            dict.set_item("execution_id", interrupt.execution_id)?;
            dict.set_item("node_id", interrupt.node_id)?;
            dict.set_item("node_name", interrupt.node_name)?;
            dict.set_item("message", interrupt.message)?;
            if let Some(ref fields) = interrupt.required_fields {
                let fields_list = PyList::empty(py);
                for field in fields {
                    fields_list.append(field)?;
                }
                dict.set_item("required_fields", fields_list)?;
            }
            list.append(dict)?;
        }
        Ok(list.into())
    }

    /// Resume execution with human input
    pub fn resume_execution(
        &self,
        interrupt_id: String,
        feedback: &Bound<'_, PyDict>,
    ) -> PyResult<()> {
        let py = feedback.py();
        let mut feedback_map = HashMap::new();
        for (key, value) in feedback {
            let key_str = key.extract::<String>()?;
            let json_value = python_to_json_value(&value)?;
            feedback_map.insert(key_str, json_value);
        }

        let engine_clone = self.inner.clone();

        block_on_detached(py, async move {
            engine_clone
                .resume_execution(&interrupt_id, feedback_map)
                .await
        })?
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to resume execution: {e}")))?;

        Ok(())
    }

    /// Pause execution
    pub fn pause_execution(&self, py: Python<'_>, execution_id: String) -> PyResult<()> {
        let engine_clone = self.inner.clone();

        let found = block_on_detached(py, async move {
            if let Some(context_arc) = engine_clone.get_execution(&execution_id).await {
                let mut context = context_arc.write().await;
                context.state = WorkflowState::Paused;
                true
            } else {
                false
            }
        })?;

        if found {
            Ok(())
        } else {
            Err(PyRuntimeError::new_err("Execution not found"))
        }
    }

    /// Cancel execution
    pub fn cancel_execution(&self, py: Python<'_>, execution_id: String) -> PyResult<()> {
        let engine_clone = self.inner.clone();

        let found = block_on_detached(py, async move {
            if let Some(context_arc) = engine_clone.get_execution(&execution_id).await {
                let mut context = context_arc.write().await;
                context.state = WorkflowState::Cancelled;
                context.end_time = Some(chrono::Utc::now());
                true
            } else {
                false
            }
        })?;

        if found {
            Ok(())
        } else {
            Err(PyRuntimeError::new_err("Execution not found"))
        }
    }
}

#[cfg(test)]
mod merge_strategy_tests {
    use caliche::orchestration::builder::merge_strategy_from_str;
    use caliche::orchestration::workflow::MergeStrategy;

    #[test]
    fn parses_builtin_labels() {
        assert!(matches!(
            merge_strategy_from_str("concatenate").expect("concatenate"),
            MergeStrategy::Concatenate
        ));
        assert!(matches!(
            merge_strategy_from_str("vote").expect("vote"),
            MergeStrategy::Vote
        ));
        assert!(matches!(
            merge_strategy_from_str("first-complete").expect("first-complete"),
            MergeStrategy::FirstComplete
        ));
        assert!(matches!(
            merge_strategy_from_str("all_complete").expect("all_complete"),
            MergeStrategy::AllComplete
        ));
    }

    #[test]
    fn custom_merge_name_trimmed() {
        let m = merge_strategy_from_str("Custom:  my_merge  ").expect("custom");
        assert!(matches!(m, MergeStrategy::Custom(ref s) if s == "my_merge"));
    }

    #[test]
    fn custom_merge_requires_name() {
        assert!(merge_strategy_from_str("custom:").is_err());
        assert!(merge_strategy_from_str("custom:   ").is_err());
    }

    #[test]
    fn unknown_strategy_errors() {
        assert!(merge_strategy_from_str("not_a_strategy").is_err());
    }
}
