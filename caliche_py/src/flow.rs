//! Python bindings for `AgentFlow`

use crate::agent::Agent as PyAgent;
use crate::get_runtime;
use caliche::agent::Agent as RustAgent;
use caliche::orchestration::{AgentFlow as RustAgentFlow, FlowBuilder as RustFlowBuilder};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

#[pyclass]
pub struct FlowBuilder {
    inner: RustFlowBuilder,
}

#[pymethods]
impl FlowBuilder {
    /// Create a new flow builder
    #[new]
    pub fn new() -> Self {
        Self {
            inner: RustFlowBuilder::new(),
        }
    }

    /// Add an agent node to the flow
    ///
    /// # Arguments
    /// * `name` - Unique identifier for this agent node
    /// * `agent` - The agent instance (`PyAgent`)
    ///
    /// Note: This method clones the Agent from `AgentService`. The original agent
    /// can still be used after adding it to the flow.
    pub fn add_node<'a>(
        mut slf: PyRefMut<'a, Self>,
        name: String,
        agent: &PyAgent,
    ) -> PyResult<PyRefMut<'a, Self>> {
        // Access the inner field directly since we're in the same crate
        let rust_agent: RustAgent = if let Some(ref agent_arc) = agent.inner {
            let rt = get_runtime()?;
            rt.block_on(async {
                let agent_guard = agent_arc.lock().await;
                agent_guard.clone_agent()
            })
        } else {
            return Err(PyRuntimeError::new_err("Agent not initialized"));
        };

        let builder = std::mem::replace(&mut slf.inner, RustFlowBuilder::new());
        slf.inner = builder.add_node(name, rust_agent);
        Ok(slf)
    }

    /// Set the entry point for the flow
    pub fn set_entry_point(mut slf: PyRefMut<Self>, name: String) -> PyRefMut<Self> {
        let builder = std::mem::replace(&mut slf.inner, RustFlowBuilder::new());
        slf.inner = builder.set_entry_point(name);
        slf
    }

    /// Build the `AgentFlow`
    pub fn build(&mut self) -> PyResult<AgentFlow> {
        let flow = std::mem::take(&mut self.inner)
            .build()
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to build flow: {e}")))?;

        Ok(AgentFlow { inner: flow })
    }
}

#[pyclass]
pub struct AgentFlow {
    inner: RustAgentFlow,
}

#[pymethods]
impl AgentFlow {
    /// Create a new flow builder
    #[staticmethod]
    pub fn builder() -> FlowBuilder {
        FlowBuilder::new()
    }

    /// Run the flow with an initial input message
    ///
    /// # Arguments
    /// * `initial_input` - The initial message to start the flow
    ///
    /// # Returns
    /// A dictionary with '`final_response`' and '`execution_log`'
    pub fn run(&self, py: Python<'_>, initial_input: &str) -> PyResult<Py<PyAny>> {
        let rt = get_runtime()?;

        let flow_clone = &self.inner;
        let result = rt
            .block_on(async move { flow_clone.run(initial_input).await })
            .map_err(|e| PyRuntimeError::new_err(format!("Flow execution failed: {e}")))?;

        // Convert result to Python dict
        let result_dict = PyDict::new(py);
        result_dict.set_item("final_response", result.final_response)?;

        // Convert execution log
        let log_list = PyList::empty(py);
        for step in result.execution_log {
            let step_dict = PyDict::new(py);
            step_dict.set_item("agent_name", step.agent_name)?;
            step_dict.set_item("input", step.input)?;
            if let Some(response) = step.response {
                step_dict.set_item("response", response)?;
            }
            if let Some(handoff) = step.handoff {
                let handoff_dict = PyDict::new(py);
                handoff_dict.set_item("target_agent", handoff.target_agent)?;
                handoff_dict.set_item("reason", handoff.reason)?;
                if let Some(instructions) = handoff.instructions {
                    handoff_dict.set_item("instructions", instructions)?;
                }
                if let Some(persona) = handoff.persona {
                    handoff_dict.set_item("persona", persona)?;
                }
                step_dict.set_item("handoff", handoff_dict)?;
            }
            let usage_dict = PyDict::new(py);
            if let Some(prompt) = step.usage.prompt_tokens {
                usage_dict.set_item("prompt_tokens", prompt)?;
            }
            if let Some(completion) = step.usage.completion_tokens {
                usage_dict.set_item("completion_tokens", completion)?;
            }
            if let Some(total) = step.usage.total_tokens {
                usage_dict.set_item("total_tokens", total)?;
            }
            step_dict.set_item("usage", usage_dict)?;
            log_list.append(step_dict)?;
        }
        result_dict.set_item("execution_log", log_list)?;

        Ok(result_dict.into())
    }

    /// Generate a Mermaid diagram representation of the flow
    pub fn to_mermaid(&self) -> String {
        self.inner.to_mermaid()
    }

    /// Get the entry point node name
    pub fn entry_point(&self) -> String {
        self.inner.entry_point().to_string()
    }

    /// Get all node names
    pub fn node_names(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let names = self.inner.node_names();
        let py_list = PyList::empty(py);
        for name in names {
            py_list.append(name)?;
        }
        Ok(py_list.into())
    }
}
