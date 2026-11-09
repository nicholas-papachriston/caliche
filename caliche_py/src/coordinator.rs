//! Python bindings for Caliche `AgentCoordinator` and `MessageBus`

use crate::agent::Agent as PyAgent;
use crate::block_on_detached;
use caliche::orchestration::coordinator::{
    AgentCapability, AgentCoordinator, AgentMetadata, LoadBalancingStrategy,
};
use caliche::orchestration::message_passing::MessageBus;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use std::sync::Arc;

// Helper to convert Python string to AgentCapability
fn python_string_to_capability(cap_str: &str) -> AgentCapability {
    match cap_str {
        "research" => AgentCapability::Research,
        "writing" => AgentCapability::Writing,
        "coding" => AgentCapability::Coding,
        "data_analysis" => AgentCapability::DataAnalysis,
        "planning" => AgentCapability::Planning,
        "coordination" => AgentCapability::Coordination,
        s => AgentCapability::Custom(s.to_string()),
    }
}

/// Python wrapper for `AgentCoordinator`
#[pyclass]
pub struct PyAgentCoordinator {
    inner: Arc<AgentCoordinator>,
}

#[pymethods]
impl PyAgentCoordinator {
    #[new]
    #[pyo3(signature = (strategy = None))]
    pub fn new(strategy: Option<String>) -> Self {
        let coordinator = strategy.map_or_else(AgentCoordinator::new, |strategy_str| {
            let load_balancing = match strategy_str.as_str() {
                "round_robin" => LoadBalancingStrategy::RoundRobin,
                "weighted" => LoadBalancingStrategy::Weighted,
                _ => LoadBalancingStrategy::LeastConnections,
            };
            AgentCoordinator::with_strategy(load_balancing)
        });

        Self {
            inner: Arc::new(coordinator),
        }
    }

    /// Register an agent with the coordinator
    pub fn register_agent(
        &self,
        py: Python<'_>,
        name: String,
        role: String,
        capabilities: &Bound<'_, PyList>,
        agent: &PyAgent,
    ) -> PyResult<String> {
        // Convert capabilities list
        let mut cap_vec = Vec::new();
        for item in capabilities.iter() {
            if let Ok(cap_str) = item.extract::<String>() {
                cap_vec.push(python_string_to_capability(&cap_str));
            }
        }

        // Create metadata
        let metadata = AgentMetadata::new(name, role, cap_vec);

        // Get agent service from Python Agent wrapper
        let agent_arc = agent
            .inner
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("Agent not initialized"))?
            .clone();

        // Extract provider and tool registry from the existing agent service,
        // then register with the coordinator.
        let coordinator_clone = self.inner.clone();
        block_on_detached(py, async move {
            let (provider, tool_registry) = {
                let agent_guard = agent_arc.lock().await;
                (
                    agent_guard.llm_provider(),
                    agent_guard.tool_registry().clone(),
                )
            };
            let rust_agent = caliche::agent::Agent::new(provider, tool_registry);
            coordinator_clone.register_agent(metadata, rust_agent).await
        })?
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to register agent: {e}")))
    }

    /// Unregister an agent
    pub fn unregister_agent(&self, py: Python<'_>, agent_id: String) -> PyResult<()> {
        let coordinator_clone = self.inner.clone();

        block_on_detached(py, async move {
            coordinator_clone.unregister_agent(&agent_id).await
        })?
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to unregister agent: {e}")))
    }

    /// Delegate a task to the best available agent
    #[pyo3(signature = (message, capability=None))]
    pub fn delegate_task(
        &self,
        py: Python<'_>,
        message: String,
        capability: Option<&str>,
    ) -> PyResult<Py<PyAny>> {
        let coordinator_clone = self.inner.clone();

        let cap = capability.map(python_string_to_capability);

        let result = block_on_detached(py, async move {
            coordinator_clone
                .delegate_task(&message, cap.as_ref())
                .await
        })?;

        match result {
            Ok((agent_id, response, usage)) => {
                let dict = PyDict::new(py);
                dict.set_item("agent_id", agent_id)?;
                dict.set_item("response", response)?;
                let usage_dict = PyDict::new(py);
                usage_dict.set_item("prompt_tokens", usage.prompt_tokens.unwrap_or(0))?;
                usage_dict.set_item("completion_tokens", usage.completion_tokens.unwrap_or(0))?;
                usage_dict.set_item("total_tokens", usage.total_tokens.unwrap_or(0))?;
                dict.set_item("usage", usage_dict)?;
                Ok(dict.into())
            }
            Err(e) => Err(PyRuntimeError::new_err(format!(
                "Failed to delegate task: {e}"
            ))),
        }
    }

    /// Execute a task on a specific agent
    pub fn execute_on_agent(
        &self,
        py: Python<'_>,
        agent_id: String,
        message: String,
    ) -> PyResult<Py<PyAny>> {
        let coordinator_clone = self.inner.clone();

        let result = block_on_detached(py, async move {
            coordinator_clone
                .execute_on_agent(&agent_id, &message)
                .await
        })?;

        match result {
            Ok((response, usage)) => {
                let dict = PyDict::new(py);
                dict.set_item("response", response)?;
                let usage_dict = PyDict::new(py);
                usage_dict.set_item("prompt_tokens", usage.prompt_tokens.unwrap_or(0))?;
                usage_dict.set_item("completion_tokens", usage.completion_tokens.unwrap_or(0))?;
                usage_dict.set_item("total_tokens", usage.total_tokens.unwrap_or(0))?;
                dict.set_item("usage", usage_dict)?;
                Ok(dict.into())
            }
            Err(e) => Err(PyRuntimeError::new_err(format!(
                "Failed to execute on agent: {e}"
            ))),
        }
    }

    /// Send message from one agent to another
    pub fn send_message(
        &self,
        py: Python<'_>,
        from_agent_id: String,
        to_agent_id: String,
        content: String,
    ) -> PyResult<()> {
        let coordinator_clone = self.inner.clone();

        block_on_detached(py, async move {
            coordinator_clone
                .send_message(&from_agent_id, &to_agent_id, content)
                .await
        })?
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to send message: {e}")))
    }

    /// Broadcast message to all agents
    pub fn broadcast(
        &self,
        py: Python<'_>,
        from_agent_id: String,
        content: String,
    ) -> PyResult<()> {
        let coordinator_clone = self.inner.clone();

        block_on_detached(py, async move {
            coordinator_clone.broadcast(&from_agent_id, content).await
        })?
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to broadcast: {e}")))
    }

    /// Get coordinator statistics
    pub fn get_stats(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let coordinator_clone = self.inner.clone();

        let stats = block_on_detached(py, async move { coordinator_clone.get_stats().await })?;

        let dict = PyDict::new(py);
        dict.set_item("total_agents", stats.total_agents)?;
        dict.set_item("total_load", stats.total_load)?;
        dict.set_item("total_capacity", stats.total_capacity)?;

        let agent_stats_list = PyList::empty(py);
        for agent_stat in stats.agent_stats {
            let stat_dict = PyDict::new(py);
            stat_dict.set_item("agent_id", agent_stat.agent_id)?;
            stat_dict.set_item("name", agent_stat.name)?;
            stat_dict.set_item("current_load", agent_stat.current_load)?;
            stat_dict.set_item("max_concurrent_tasks", agent_stat.max_concurrent_tasks)?;
            agent_stats_list.append(stat_dict)?;
        }
        dict.set_item("agent_stats", agent_stats_list)?;

        Ok(dict.into())
    }

    /// Get list of agent IDs
    pub fn list_agent_ids(&self) -> Vec<String> {
        self.inner.list_agent_ids()
    }

    /// Get agent count
    pub fn agent_count(&self) -> usize {
        self.inner.agent_count()
    }

    /// Get message bus
    pub fn get_message_bus(&self) -> PyMessageBus {
        PyMessageBus {
            inner: self.inner.message_bus(),
        }
    }
}

/// Python wrapper for `MessageBus`
#[pyclass]
pub struct PyMessageBus {
    inner: Arc<MessageBus>,
}

#[pymethods]
impl PyMessageBus {
    /// Get message bus statistics
    pub fn get_stats(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let bus_clone = self.inner.clone();

        let stats = block_on_detached(py, async move { bus_clone.get_stats().await })?;

        let dict = PyDict::new(py);
        dict.set_item("queue_size", stats.queue_size)?;
        dict.set_item("max_queue_size", stats.max_queue_size)?;
        dict.set_item("low_priority", stats.low_priority)?;
        dict.set_item("normal_priority", stats.normal_priority)?;
        dict.set_item("high_priority", stats.high_priority)?;
        dict.set_item("critical_priority", stats.critical_priority)?;
        dict.set_item("subscriber_count", stats.subscriber_count)?;

        Ok(dict.into())
    }

    /// Get queue size
    pub fn queue_size(&self, py: Python<'_>) -> PyResult<usize> {
        let bus_clone = self.inner.clone();

        block_on_detached(py, async move { bus_clone.queue_size().await })
    }
}
