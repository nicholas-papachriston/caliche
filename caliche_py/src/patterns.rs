//! Python bindings for Caliche Orchestration Patterns

use crate::conversions::json_to_python_dict;
use crate::get_runtime;
use caliche::orchestration::patterns::{
    DebatePattern, HierarchicalPattern, PipelinePattern, SupervisorPattern,
};
use caliche::orchestration::workflow::{ExecutionContext, Workflow};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use std::sync::Arc;

/// Python wrapper for `DebatePattern`
#[pyclass]
pub struct PyDebatePattern {
    inner: Arc<DebatePattern>,
}

#[pymethods]
impl PyDebatePattern {
    #[new]
    #[pyo3(signature = (name, topic, debater_count, max_rounds, moderator_prompt = None, description = None))]
    pub fn new(
        name: String,
        topic: String,
        debater_count: usize,
        max_rounds: usize,
        moderator_prompt: Option<String>,
        description: Option<String>,
    ) -> Self {
        let mut pattern = DebatePattern::new(name, topic, debater_count, max_rounds);
        if let Some(prompt) = moderator_prompt {
            pattern = pattern.with_moderator_prompt(prompt);
        }
        if let Some(desc) = description {
            pattern = pattern.with_description(desc);
        }
        Self {
            inner: Arc::new(pattern),
        }
    }

    /// Execute the debate pattern
    pub fn execute(&self, engine: &workflow::PyWorkflowEngine) -> PyResult<Py<PyAny>> {
        let pattern_clone = self.inner.clone();
        let engine_clone = engine.get_inner();
        let workflow_id = pattern_clone.id().to_string();
        let rt = get_runtime()?;

        let result = rt.block_on(async {
            let mut context = ExecutionContext::new(workflow_id);
            pattern_clone.execute(&engine_clone, &mut context).await
        });

        Python::attach(|py| match result {
            Ok(value) => {
                let py_dict = json_to_python_dict(py, &value)?;
                Ok(py_dict.into())
            }
            Err(e) => Err(PyRuntimeError::new_err(format!(
                "Debate pattern execution failed: {e}"
            ))),
        })
    }

    /// Get pattern ID
    pub fn id(&self) -> String {
        self.inner.id().to_string()
    }

    /// Get pattern name
    pub fn name(&self) -> String {
        self.inner.name().to_string()
    }

    /// Get pattern description
    pub fn description(&self) -> String {
        self.inner.description().to_string()
    }
}

/// Python wrapper for `SupervisorPattern`
#[pyclass]
pub struct PySupervisorPattern {
    inner: Arc<SupervisorPattern>,
}

#[pymethods]
impl PySupervisorPattern {
    #[new]
    #[pyo3(signature = (name, supervisor_prompt, worker_count, worker_capability = None, description = None))]
    pub fn new(
        name: String,
        supervisor_prompt: String,
        worker_count: usize,
        worker_capability: Option<String>,
        description: Option<String>,
    ) -> Self {
        let mut pattern = SupervisorPattern::new(name, supervisor_prompt, worker_count);
        if let Some(cap) = worker_capability {
            pattern = pattern.with_worker_capability(cap);
        }
        if let Some(desc) = description {
            pattern = pattern.with_description(desc);
        }
        Self {
            inner: Arc::new(pattern),
        }
    }

    /// Execute the supervisor pattern
    pub fn execute(&self, engine: &workflow::PyWorkflowEngine) -> PyResult<Py<PyAny>> {
        let pattern_clone = self.inner.clone();
        let engine_clone = engine.get_inner();
        let workflow_id = pattern_clone.id().to_string();
        let rt = get_runtime()?;

        let result = rt.block_on(async {
            let mut context = ExecutionContext::new(workflow_id);
            pattern_clone.execute(&engine_clone, &mut context).await
        });

        Python::attach(|py| match result {
            Ok(value) => {
                let py_dict = json_to_python_dict(py, &value)?;
                Ok(py_dict.into())
            }
            Err(e) => Err(PyRuntimeError::new_err(format!(
                "Supervisor pattern execution failed: {e}"
            ))),
        })
    }

    /// Get pattern ID
    pub fn id(&self) -> String {
        self.inner.id().to_string()
    }

    /// Get pattern name
    pub fn name(&self) -> String {
        self.inner.name().to_string()
    }

    /// Get pattern description
    pub fn description(&self) -> String {
        self.inner.description().to_string()
    }
}

/// Python wrapper for `HierarchicalPattern`
#[pyclass]
pub struct PyHierarchicalPattern {
    inner: Arc<HierarchicalPattern>,
}

#[pymethods]
impl PyHierarchicalPattern {
    #[new]
    #[pyo3(signature = (name, description = None))]
    pub fn new(name: String, description: Option<String>) -> Self {
        let mut pattern = HierarchicalPattern::new(name);
        if let Some(desc) = description {
            pattern = pattern.with_description(desc);
        }
        Self {
            inner: Arc::new(pattern),
        }
    }

    /// Execute the hierarchical pattern
    pub fn execute(&self, engine: &workflow::PyWorkflowEngine) -> PyResult<Py<PyAny>> {
        let pattern_clone = self.inner.clone();
        let engine_clone = engine.get_inner();
        let workflow_id = pattern_clone.id().to_string();
        let rt = get_runtime()?;

        let result = rt.block_on(async {
            let mut context = ExecutionContext::new(workflow_id);
            pattern_clone.execute(&engine_clone, &mut context).await
        });

        Python::attach(|py| match result {
            Ok(value) => {
                let py_dict = json_to_python_dict(py, &value)?;
                Ok(py_dict.into())
            }
            Err(e) => Err(PyRuntimeError::new_err(format!(
                "Hierarchical pattern execution failed: {e}"
            ))),
        })
    }

    /// Get pattern ID
    pub fn id(&self) -> String {
        self.inner.id().to_string()
    }

    /// Get pattern name
    pub fn name(&self) -> String {
        self.inner.name().to_string()
    }

    /// Get pattern description
    pub fn description(&self) -> String {
        self.inner.description().to_string()
    }
}

/// Python wrapper for `HierarchicalPattern` builder
#[pyclass]
pub struct PyHierarchicalPatternBuilder {
    name: String,
    description: Option<String>,
    layers: Vec<(String, usize, String, Option<String>)>, // (name, agent_count, prompt, capability)
}

#[pymethods]
impl PyHierarchicalPatternBuilder {
    #[new]
    pub const fn new(name: String) -> Self {
        Self {
            name,
            description: None,
            layers: Vec::new(),
        }
    }

    pub fn with_description(&mut self, description: String) {
        self.description = Some(description);
    }

    pub fn add_layer(&mut self, name: String, agent_count: usize, prompt_template: String) {
        self.layers.push((name, agent_count, prompt_template, None));
    }

    pub fn add_layer_with_capability(
        &mut self,
        name: String,
        agent_count: usize,
        prompt_template: String,
        capability: String,
    ) {
        self.layers
            .push((name, agent_count, prompt_template, Some(capability)));
    }

    pub fn build(&self) -> PyHierarchicalPattern {
        let mut pattern = HierarchicalPattern::new(self.name.clone());
        if let Some(ref desc) = self.description {
            pattern = pattern.with_description(desc.clone());
        }
        for (name, agent_count, prompt, _capability) in &self.layers {
            // Note: HierarchicalPattern::add_layer doesn't support capability yet
            // This would need to be added to the Rust side
            pattern = pattern.add_layer(name.clone(), *agent_count, prompt.clone());
        }
        PyHierarchicalPattern {
            inner: Arc::new(pattern),
        }
    }
}

/// Python wrapper for `PipelinePattern`
#[pyclass]
pub struct PyPipelinePattern {
    inner: Arc<PipelinePattern>,
}

#[pymethods]
impl PyPipelinePattern {
    #[new]
    #[pyo3(signature = (name, description = None))]
    pub fn new(name: String, description: Option<String>) -> Self {
        let mut pattern = PipelinePattern::new(name);
        if let Some(desc) = description {
            pattern = pattern.with_description(desc);
        }
        Self {
            inner: Arc::new(pattern),
        }
    }

    /// Execute the pipeline pattern
    pub fn execute(&self, engine: &workflow::PyWorkflowEngine) -> PyResult<Py<PyAny>> {
        let pattern_clone = self.inner.clone();
        let engine_clone = engine.get_inner();
        let workflow_id = pattern_clone.id().to_string();
        let rt = get_runtime()?;

        let result = rt.block_on(async {
            let mut context = ExecutionContext::new(workflow_id);
            pattern_clone.execute(&engine_clone, &mut context).await
        });

        Python::attach(|py| match result {
            Ok(value) => {
                let py_dict = json_to_python_dict(py, &value)?;
                Ok(py_dict.into())
            }
            Err(e) => Err(PyRuntimeError::new_err(format!(
                "Pipeline pattern execution failed: {e}"
            ))),
        })
    }

    /// Get pattern ID
    pub fn id(&self) -> String {
        self.inner.id().to_string()
    }

    /// Get pattern name
    pub fn name(&self) -> String {
        self.inner.name().to_string()
    }

    /// Get pattern description
    pub fn description(&self) -> String {
        self.inner.description().to_string()
    }
}

/// Python wrapper for `PipelinePattern` builder
#[pyclass]
pub struct PyPipelinePatternBuilder {
    name: String,
    description: Option<String>,
    stages: Vec<(String, Option<String>, String, Option<String>)>, // (name, capability, prompt, transform)
}

#[pymethods]
impl PyPipelinePatternBuilder {
    #[new]
    pub const fn new(name: String) -> Self {
        Self {
            name,
            description: None,
            stages: Vec::new(),
        }
    }

    pub fn with_description(&mut self, description: String) {
        self.description = Some(description);
    }

    pub fn add_stage(&mut self, name: String, prompt_template: String) {
        self.stages.push((name, None, prompt_template, None));
    }

    pub fn add_stage_with_capability(
        &mut self,
        name: String,
        capability: String,
        prompt_template: String,
    ) {
        self.stages
            .push((name, Some(capability), prompt_template, None));
    }

    pub fn add_transform_stage(&mut self, name: String, transform: String) {
        self.stages
            .push((name, None, String::new(), Some(transform)));
    }

    pub fn build(&self) -> PyPipelinePattern {
        let mut pattern = PipelinePattern::new(self.name.clone());
        if let Some(desc) = &self.description {
            pattern = pattern.with_description(desc.clone());
        }
        for (name, capability, prompt, transform) in &self.stages {
            if let Some(t) = transform {
                pattern = pattern.add_transform_stage(name.clone(), t.clone());
            } else if let Some(cap) = capability {
                pattern =
                    pattern.add_stage_with_capability(name.clone(), cap.clone(), prompt.clone());
            } else {
                pattern = pattern.add_stage(name.clone(), prompt.clone());
            }
        }
        PyPipelinePattern {
            inner: Arc::new(pattern),
        }
    }
}

// Use PyWorkflowEngine from workflow module - patterns need access to inner
use crate::workflow;
