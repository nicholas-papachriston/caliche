//! Python bindings for Caliche Reasoning Patterns

use crate::agent::Agent;
use crate::conversions::json_to_python_value;
use crate::get_runtime;
use caliche::reasoning::plan_execute::PlanExecutePattern;
use caliche::reasoning::react::ReActPattern;
use caliche::reasoning::reflexion::ReflexionPattern;
use caliche::reasoning::self_ask::SelfAskPattern;
use caliche::reasoning::tree_of_thoughts::TreeOfThoughtsPattern;
use caliche::reasoning::{ReasoningPattern, ReasoningResult, StepType};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::sync::Arc;

fn recreate_agent_service(
    provider: caliche::ProviderType,
    model: String,
) -> PyResult<caliche::AgentService> {
    get_runtime()?
        .block_on(async {
            caliche::AgentService::from_config(
                caliche::AgentServiceConfig::new()
                    .with_provider(Some(provider), Some(model))
                    .with_all_default_tools(),
            )
            .await
        })
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to recreate agent service: {e}")))
}

/// Python wrapper for `ReActPattern`
#[pyclass]
pub struct PyReActPattern {
    inner: Arc<ReActPattern>,
}

#[pymethods]
impl PyReActPattern {
    #[new]
    pub fn new(agent: &Agent, max_iterations: usize) -> PyResult<Self> {
        // Convert Arc<std::sync::Mutex<AgentService>> to Arc<tokio::sync::Mutex<AgentService>>
        // by extracting AgentService and wrapping it in tokio::sync::Mutex
        let agent_arc = agent
            .inner
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("Agent not initialized"))?;
        let rt = get_runtime()?;

        let (provider, model) = rt.block_on(async {
            let agent_guard = agent_arc.lock().await;
            (agent_guard.provider(), agent_guard.model().to_string())
        });
        let agent_service = recreate_agent_service(provider, model)?;

        let pattern = ReActPattern::new(agent_service, max_iterations);
        Ok(Self {
            inner: Arc::new(pattern),
        })
    }

    /// Execute the `ReAct` pattern
    pub fn execute(&self, query: String) -> PyResult<Py<PyAny>> {
        let pattern_clone = self.inner.clone();
        let rt = get_runtime()?;

        let result = rt.block_on(async move { pattern_clone.execute(&query).await });

        Python::attach(|py| match result {
            Ok(reasoning_result) => {
                let dict = create_reasoning_result_dict(py, &reasoning_result)?;
                Ok(dict.into())
            }
            Err(e) => Err(PyRuntimeError::new_err(format!(
                "ReAct execution failed: {e}"
            ))),
        })
    }
}

/// Python wrapper for `PlanExecutePattern`
#[pyclass]
pub struct PyPlanExecutePattern {
    inner: Arc<PlanExecutePattern>,
}

#[pymethods]
impl PyPlanExecutePattern {
    #[new]
    pub fn new(agent: &Agent, max_iterations: usize) -> PyResult<Self> {
        let agent_arc = agent
            .inner
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("Agent not initialized"))?;
        let rt = get_runtime()?;

        let (provider, model) = rt.block_on(async {
            let agent_guard = agent_arc.lock().await;
            (agent_guard.provider(), agent_guard.model().to_string())
        });
        let agent_service = recreate_agent_service(provider, model)?;

        let pattern = PlanExecutePattern::new(agent_service, max_iterations);
        Ok(Self {
            inner: Arc::new(pattern),
        })
    }

    /// Execute the `PlanExecute` pattern
    pub fn execute(&self, query: String) -> PyResult<Py<PyAny>> {
        let pattern_clone = self.inner.clone();
        let rt = get_runtime()?;

        let result = rt.block_on(async move { pattern_clone.execute(&query).await });

        Python::attach(|py| match result {
            Ok(reasoning_result) => {
                let dict = create_reasoning_result_dict(py, &reasoning_result)?;
                Ok(dict.into())
            }
            Err(e) => Err(PyRuntimeError::new_err(format!(
                "PlanExecute execution failed: {e}"
            ))),
        })
    }
}

/// Python wrapper for `ReflexionPattern`
#[pyclass]
pub struct PyReflexionPattern {
    inner: Arc<ReflexionPattern>,
}

#[pymethods]
impl PyReflexionPattern {
    #[new]
    pub fn new(agent: &Agent, max_iterations: usize) -> PyResult<Self> {
        let agent_arc = agent
            .inner
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("Agent not initialized"))?;
        let rt = get_runtime()?;

        let (provider, model) = rt.block_on(async {
            let agent_guard = agent_arc.lock().await;
            (agent_guard.provider(), agent_guard.model().to_string())
        });
        let agent_service = recreate_agent_service(provider, model)?;

        let pattern = ReflexionPattern::new(agent_service, max_iterations);
        Ok(Self {
            inner: Arc::new(pattern),
        })
    }

    /// Execute the Reflexion pattern
    pub fn execute(&self, query: String) -> PyResult<Py<PyAny>> {
        let pattern_clone = self.inner.clone();
        let rt = get_runtime()?;

        let result = rt.block_on(async move { pattern_clone.execute(&query).await });

        Python::attach(|py| match result {
            Ok(reasoning_result) => {
                let dict = create_reasoning_result_dict(py, &reasoning_result)?;
                Ok(dict.into())
            }
            Err(e) => Err(PyRuntimeError::new_err(format!(
                "Reflexion execution failed: {e}"
            ))),
        })
    }
}

/// Python wrapper for `SelfAskPattern`
#[pyclass]
pub struct PySelfAskPattern {
    inner: Arc<SelfAskPattern>,
}

#[pymethods]
impl PySelfAskPattern {
    #[new]
    pub fn new(agent: &Agent, max_depth: usize) -> PyResult<Self> {
        let agent_arc = agent
            .inner
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("Agent not initialized"))?;
        let rt = get_runtime()?;

        let (provider, model) = rt.block_on(async {
            let agent_guard = agent_arc.lock().await;
            (agent_guard.provider(), agent_guard.model().to_string())
        });
        let agent_service = recreate_agent_service(provider, model)?;

        let pattern = SelfAskPattern::new(agent_service, max_depth);
        Ok(Self {
            inner: Arc::new(pattern),
        })
    }

    /// Execute the `SelfAsk` pattern
    pub fn execute(&self, query: String) -> PyResult<Py<PyAny>> {
        let pattern_clone = self.inner.clone();
        let rt = get_runtime()?;

        let result = rt.block_on(async move { pattern_clone.execute(&query).await });

        Python::attach(|py| match result {
            Ok(reasoning_result) => {
                let dict = create_reasoning_result_dict(py, &reasoning_result)?;
                Ok(dict.into())
            }
            Err(e) => Err(PyRuntimeError::new_err(format!(
                "SelfAsk execution failed: {e}"
            ))),
        })
    }
}

/// Python wrapper for `TreeOfThoughtsPattern`
#[pyclass]
pub struct PyTreeOfThoughtsPattern {
    inner: Arc<TreeOfThoughtsPattern>,
}

#[pymethods]
impl PyTreeOfThoughtsPattern {
    #[new]
    #[pyo3(signature = (agent, max_depth, branching_factor, beam_width = None))]
    pub fn new(
        agent: &Agent,
        max_depth: usize,
        branching_factor: usize,
        beam_width: Option<usize>,
    ) -> PyResult<Self> {
        let agent_arc = agent
            .inner
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("Agent not initialized"))?;
        let rt = get_runtime()?;

        let (provider, model) = rt.block_on(async {
            let agent_guard = agent_arc.lock().await;
            (agent_guard.provider(), agent_guard.model().to_string())
        });
        let agent_service = recreate_agent_service(provider, model)?;

        let mut pattern = TreeOfThoughtsPattern::new(agent_service, max_depth, branching_factor);
        if let Some(bw) = beam_width {
            pattern = pattern.with_beam_width(bw);
        }
        Ok(Self {
            inner: Arc::new(pattern),
        })
    }

    /// Execute the `TreeOfThoughts` pattern
    pub fn execute(&self, query: String) -> PyResult<Py<PyAny>> {
        let pattern_clone = self.inner.clone();
        let rt = get_runtime()?;

        let result = rt.block_on(async move { pattern_clone.execute(&query).await });

        Python::attach(|py| match result {
            Ok(reasoning_result) => {
                let dict = create_reasoning_result_dict(py, &reasoning_result)?;
                Ok(dict.into())
            }
            Err(e) => Err(PyRuntimeError::new_err(format!(
                "TreeOfThoughts execution failed: {e}"
            ))),
        })
    }
}

fn create_reasoning_result_dict<'py>(
    py: Python<'py>,
    result: &ReasoningResult,
) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);

    // Set final_answer
    dict.set_item("final_answer", result.final_answer.clone())?;

    // Set success
    dict.set_item("success", result.success)?;

    // Set error if present
    if let Some(error) = &result.error {
        dict.set_item("error", error.clone())?;
    }

    // Build steps list
    let steps_list = pyo3::types::PyList::empty(py);
    for step in &result.steps {
        let step_dict = PyDict::new(py);
        let step_type_str = match step.step_type {
            StepType::Thought => "thought",
            StepType::Action => "action",
            StepType::Observation => "observation",
            StepType::Plan => "plan",
            StepType::Reflection => "reflection",
            StepType::Answer => "answer",
        };
        step_dict.set_item("step_type", step_type_str)?;
        step_dict.set_item("content", step.content.clone())?;

        if let Some(metadata) = &step.metadata {
            // Convert metadata Value to Python dict
            let metadata_dict = json_to_python_dict(py, metadata)?;
            step_dict.set_item("metadata", metadata_dict)?;
        }

        steps_list.append(step_dict)?;
    }
    dict.set_item("steps", steps_list)?;

    // Set token_usage if present
    if let Some(usage) = &result.token_usage {
        let usage_dict = PyDict::new(py);
        if let Some(prompt_tokens) = usage.prompt_tokens {
            usage_dict.set_item("prompt_tokens", prompt_tokens)?;
        }
        if let Some(completion_tokens) = usage.completion_tokens {
            usage_dict.set_item("completion_tokens", completion_tokens)?;
        }
        if let Some(total_tokens) = usage.total_tokens {
            usage_dict.set_item("total_tokens", total_tokens)?;
        }
        dict.set_item("token_usage", usage_dict)?;
    }

    Ok(dict)
}

fn json_to_python_dict<'py>(
    py: Python<'py>,
    value: &serde_json::Value,
) -> PyResult<Bound<'py, PyDict>> {
    match value {
        serde_json::Value::Object(obj) => {
            let dict = PyDict::new(py);
            for (k, v) in obj {
                let py_value = json_to_python_value(py, v)?;
                dict.set_item(k, py_value)?;
            }
            Ok(dict)
        }
        _ => Err(PyRuntimeError::new_err("Expected JSON object")),
    }
}
