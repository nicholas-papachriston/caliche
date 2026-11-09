//! JavaScript bindings for Caliche Orchestration Patterns

use caliche::orchestration::patterns::{
    DebatePattern as RustDebatePattern, HierarchicalPattern as RustHierarchicalPattern,
    PipelinePattern as RustPipelinePattern, SupervisorPattern as RustSupervisorPattern,
};
use caliche::orchestration::workflow::{ExecutionContext, Workflow, WorkflowEngine};
use napi_derive::napi;
use std::sync::Arc;

/// JavaScript wrapper for `DebatePattern`
#[napi]
pub struct DebatePattern {
    inner: Arc<RustDebatePattern>,
}

#[napi]
impl DebatePattern {
    /// Create a new `DebatePattern`
    #[napi(factory, catch_unwind)]
    pub fn new(
        name: String,
        topic: String,
        debater_count: u32,
        max_rounds: u32,
        moderator_prompt: Option<String>,
        description: Option<String>,
    ) -> Self {
        let mut pattern =
            RustDebatePattern::new(name, topic, debater_count as usize, max_rounds as usize);
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

    /// Execute a `DebatePattern`
    #[napi]
    ///
    /// # Errors
    ///
    /// Returns a NAPI error when validation, synchronization, serialization,
    /// or an underlying runtime operation fails.
    pub async fn execute(&self) -> napi::Result<serde_json::Value> {
        let pattern_clone = self.inner.clone();
        let engine = caliche::orchestration::workflow::WorkflowEngine::new();
        let workflow_id = pattern_clone.id().to_string();

        let mut context = ExecutionContext::new(workflow_id);
        let workflow_arc: Arc<dyn Workflow> = pattern_clone.clone();
        let result = workflow_arc.execute(&engine, &mut context).await;

        match result {
            Ok(value) => Ok(value),
            Err(e) => Err(napi::Error::from_reason(format!(
                "Debate pattern execution failed: {e}"
            ))),
        }
    }

    /// Get `DebatePattern` ID
    #[napi(catch_unwind)]
    pub fn get_id(&self) -> String {
        self.inner.id().to_string()
    }

    /// Get `DebatePattern` name
    #[napi(catch_unwind)]
    pub fn get_name(&self) -> String {
        self.inner.name().to_string()
    }

    /// Get `DebatePattern` description
    #[napi(catch_unwind)]
    pub fn get_description(&self) -> String {
        self.inner.description().to_string()
    }
}

/// JavaScript wrapper for `HierarchicalPattern`
#[napi]
pub struct HierarchicalPattern {
    inner: Arc<RustHierarchicalPattern>,
}

#[napi]
impl HierarchicalPattern {
    /// Create a new `HierarchicalPattern`
    #[napi(factory, catch_unwind)]
    pub fn new(name: String, description: Option<String>) -> Self {
        let mut pattern = RustHierarchicalPattern::new(name);
        if let Some(desc) = description {
            pattern = pattern.with_description(desc);
        }
        Self {
            inner: Arc::new(pattern),
        }
    }

    /// Add a layer to `HierarchicalPattern`
    #[napi(catch_unwind)]
    pub fn add_layer(
        &self,
        name: String,
        agent_count: u32,
        prompt_template: String,
        _capability: Option<String>,
    ) -> Self {
        // Clone the pattern and add the layer
        let pattern_inner =
            Arc::try_unwrap(self.inner.clone()).unwrap_or_else(|arc| (*arc).clone());

        let new_pattern = pattern_inner.add_layer(name, agent_count as usize, prompt_template);
        Self {
            inner: Arc::new(new_pattern),
        }
    }

    /// Execute a `HierarchicalPattern`
    #[napi]
    ///
    /// # Errors
    ///
    /// Returns a NAPI error when validation, synchronization, serialization,
    /// or an underlying runtime operation fails.
    pub async fn execute(&self) -> napi::Result<serde_json::Value> {
        let pattern_clone = self.inner.clone();
        let engine = WorkflowEngine::new();
        let workflow_id = pattern_clone.id().to_string();

        let mut context = ExecutionContext::new(workflow_id);
        let workflow_arc: Arc<dyn Workflow> = pattern_clone.clone();
        let result = workflow_arc.execute(&engine, &mut context).await;

        match result {
            Ok(value) => Ok(value),
            Err(e) => Err(napi::Error::from_reason(format!(
                "Hierarchical pattern execution failed: {e}"
            ))),
        }
    }

    /// Get `HierarchicalPattern` ID
    #[napi(catch_unwind)]
    pub fn get_id(&self) -> String {
        self.inner.id().to_string()
    }

    /// Get `HierarchicalPattern` name
    #[napi(catch_unwind)]
    pub fn get_name(&self) -> String {
        self.inner.name().to_string()
    }

    /// Get `HierarchicalPattern` description
    #[napi(catch_unwind)]
    pub fn get_description(&self) -> String {
        self.inner.description().to_string()
    }
}

/// JavaScript wrapper for `PipelinePattern`
#[napi]
pub struct PipelinePattern {
    inner: Arc<RustPipelinePattern>,
}

#[napi]
impl PipelinePattern {
    /// Create a new `PipelinePattern`
    #[napi(factory, catch_unwind)]
    pub fn new(name: String, description: Option<String>) -> Self {
        let mut pattern = RustPipelinePattern::new(name);
        if let Some(desc) = description {
            pattern = pattern.with_description(desc);
        }
        Self {
            inner: Arc::new(pattern),
        }
    }

    /// Add a stage to `PipelinePattern`
    #[napi(catch_unwind)]
    pub fn add_stage(
        &self,
        name: String,
        prompt: String,
        capability: Option<String>,
        transform: Option<String>,
    ) -> Self {
        let pattern_inner =
            Arc::try_unwrap(self.inner.clone()).unwrap_or_else(|arc| (*arc).clone());

        let new_pattern = if let Some(t) = transform {
            pattern_inner.add_transform_stage(name, t)
        } else if let Some(cap) = capability {
            pattern_inner.add_stage_with_capability(name, cap, prompt)
        } else {
            pattern_inner.add_stage(name, prompt)
        };

        Self {
            inner: Arc::new(new_pattern),
        }
    }

    /// Execute a `PipelinePattern`
    #[napi]
    ///
    /// # Errors
    ///
    /// Returns a NAPI error when validation, synchronization, serialization,
    /// or an underlying runtime operation fails.
    pub async fn execute(&self) -> napi::Result<serde_json::Value> {
        let pattern_clone = self.inner.clone();
        let engine = WorkflowEngine::new();
        let workflow_id = pattern_clone.id().to_string();

        let mut context = ExecutionContext::new(workflow_id);
        let workflow_arc: Arc<dyn Workflow> = pattern_clone.clone();
        let result = workflow_arc.execute(&engine, &mut context).await;

        match result {
            Ok(value) => Ok(value),
            Err(e) => Err(napi::Error::from_reason(format!(
                "Pipeline pattern execution failed: {e}"
            ))),
        }
    }

    /// Get `PipelinePattern` ID
    #[napi(catch_unwind)]
    pub fn get_id(&self) -> String {
        self.inner.id().to_string()
    }

    /// Get `PipelinePattern` name
    #[napi(catch_unwind)]
    pub fn get_name(&self) -> String {
        self.inner.name().to_string()
    }

    /// Get `PipelinePattern` description
    #[napi(catch_unwind)]
    pub fn get_description(&self) -> String {
        self.inner.description().to_string()
    }
}

/// JavaScript wrapper for `SupervisorPattern`
#[napi]
pub struct SupervisorPattern {
    inner: Arc<RustSupervisorPattern>,
}

#[napi]
impl SupervisorPattern {
    /// Create a new `SupervisorPattern`
    #[napi(factory, catch_unwind)]
    pub fn new(
        name: String,
        supervisor_prompt: String,
        worker_count: u32,
        worker_capability: Option<String>,
        description: Option<String>,
    ) -> Self {
        let mut pattern =
            RustSupervisorPattern::new(name, supervisor_prompt, worker_count as usize);
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

    /// Execute a `SupervisorPattern`
    #[napi]
    ///
    /// # Errors
    ///
    /// Returns a NAPI error when validation, synchronization, serialization,
    /// or an underlying runtime operation fails.
    pub async fn execute(&self) -> napi::Result<serde_json::Value> {
        let pattern_clone = self.inner.clone();
        let engine = WorkflowEngine::new();
        let workflow_id = pattern_clone.id().to_string();

        let mut context = ExecutionContext::new(workflow_id);
        let workflow_arc: Arc<dyn Workflow> = pattern_clone.clone();
        let result = workflow_arc.execute(&engine, &mut context).await;

        match result {
            Ok(value) => Ok(value),
            Err(e) => Err(napi::Error::from_reason(format!(
                "Supervisor pattern execution failed: {e}"
            ))),
        }
    }

    /// Get `SupervisorPattern` ID
    #[napi(catch_unwind)]
    pub fn get_id(&self) -> String {
        self.inner.id().to_string()
    }

    /// Get `SupervisorPattern` name
    #[napi(catch_unwind)]
    pub fn get_name(&self) -> String {
        self.inner.name().to_string()
    }

    /// Get `SupervisorPattern` description
    #[napi(catch_unwind)]
    pub fn get_description(&self) -> String {
        self.inner.description().to_string()
    }
}
