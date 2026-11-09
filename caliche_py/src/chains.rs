//! Python bindings for Caliche Chains

use crate::agent::Agent;
use crate::block_on_detached;
use crate::conversions::{json_to_python_dict, python_to_json_value};
use caliche::chains::{Chain, CompositionSequentialChain, LLMChain};
use caliche::prompts::PromptTemplate;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use serde_json::Value;
use std::sync::Arc;

fn recreate_agent_service(
    py: Python<'_>,
    provider: caliche::ProviderType,
    model: String,
) -> PyResult<caliche::AgentService> {
    block_on_detached(py, async {
        caliche::AgentService::from_config(
            caliche::AgentServiceConfig::new()
                .with_provider(Some(provider), Some(model))
                .with_all_default_tools(),
        )
        .await
    })?
    .map_err(|e| PyRuntimeError::new_err(format!("Failed to recreate agent service: {e}")))
}

/// Python wrapper for `LLMChain`
#[pyclass]
pub struct PyLLMChain {
    inner: Arc<LLMChain>,
}

#[pymethods]
impl PyLLMChain {
    #[new]
    pub fn new(py: Python<'_>, agent: &Agent, prompt_template: &str) -> PyResult<Self> {
        // Convert Arc<std::sync::Mutex<AgentService>> to Arc<tokio::sync::Mutex<AgentService>>
        let agent_arc = agent
            .inner
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("Agent not initialized"))?
            .clone();

        let (provider, model) = block_on_detached(py, async move {
            let agent_guard = agent_arc.lock().await;
            (agent_guard.provider(), agent_guard.model().to_string())
        })?;
        let agent_service = recreate_agent_service(py, provider, model)?;

        let prompt = PromptTemplate::new(prompt_template);
        let chain = caliche::chains::LLMChain::new(
            Arc::new(tokio::sync::Mutex::new(agent_service)),
            prompt,
        );
        Ok(Self {
            inner: Arc::new(chain),
        })
    }

    /// Run the chain with input variables
    pub fn run(&self, variables: &Bound<'_, PyDict>) -> PyResult<Py<PyAny>> {
        let py = variables.py();

        // Convert Python dict to JSON Value
        let mut json_map = serde_json::Map::new();
        for (key, value) in variables {
            let key_str = key.extract::<String>()?;
            let json_value = python_to_json_value(&value)?;
            json_map.insert(key_str, json_value);
        }
        let input_value = Value::Object(json_map);

        let chain_clone = self.inner.clone();

        // Run async operation
        let result = block_on_detached(py, async move { chain_clone.run(&input_value).await })?;

        match result {
            Ok(chain_result) => {
                let output_value = &chain_result.output;
                let output_dict = json_to_python_dict(py, output_value)?;
                Ok(output_dict.into_any().unbind())
            }
            Err(e) => Err(PyRuntimeError::new_err(format!(
                "Chain execution failed: {e}"
            ))),
        }
    }
}

/// Python wrapper for `SequentialChain`
#[pyclass]
pub struct PySequentialChain {
    chains: Vec<(String, Arc<dyn Chain>)>,
    #[expect(dead_code, reason = "stored but not read yet")]
    return_intermediate: bool,
}

#[pymethods]
impl PySequentialChain {
    #[new]
    pub fn new() -> Self {
        Self {
            chains: Vec::new(),
            return_intermediate: false,
        }
    }

    /// Add a chain to the sequence
    pub fn add_chain(&mut self, name: String, chain: &PyLLMChain) {
        self.chains
            .push((name, chain.inner.clone() as Arc<dyn Chain>));
    }

    /// Run the sequential chain
    pub fn run(&self, variables: &Bound<'_, PyDict>) -> PyResult<Py<PyAny>> {
        let py = variables.py();

        // Convert Python dict to JSON Value
        let mut json_map = serde_json::Map::new();
        for (key, value) in variables {
            let key_str = key.extract::<String>()?;
            let json_value = python_to_json_value(&value)?;
            json_map.insert(key_str, json_value);
        }
        let input_value = Value::Object(json_map);

        // Build sequential chain by chaining pairs
        if self.chains.is_empty() {
            return Err(PyRuntimeError::new_err(
                "No chains added to sequential chain",
            ));
        }

        // Start with the first chain
        let mut chain: Arc<dyn Chain> = self.chains[0].1.clone();

        // Chain each subsequent chain sequentially
        for (_, chain_ref) in self.chains.iter().skip(1) {
            let sequential = CompositionSequentialChain::new(chain, chain_ref.clone());
            chain = Arc::new(sequential);
        }

        // Note: return_intermediate is not supported with SequentialChain
        // The intermediate results are always included in the ChainContext

        let result = block_on_detached(py, async move { chain.run(&input_value).await })?;

        match result {
            Ok(chain_result) => {
                let output_value = &chain_result.output;
                let output_dict = json_to_python_dict(py, output_value)?;
                Ok(output_dict.into_any().unbind())
            }
            Err(e) => Err(PyRuntimeError::new_err(format!(
                "Chain execution failed: {e}"
            ))),
        }
    }
}

/// Python wrapper for `MapReduceChain`
#[pyclass]
pub struct PyMapReduceChain {
    inner: Arc<caliche::chains::MapReduceChain>,
}

#[pymethods]
impl PyMapReduceChain {
    #[new]
    pub fn new(
        py: Python<'_>,
        map_agent: &Agent,
        map_prompt: &str,
        reduce_agent: &Agent,
        reduce_prompt: &str,
    ) -> PyResult<Self> {
        let map_agent_service = {
            let agent_arc = map_agent
                .inner
                .as_ref()
                .ok_or_else(|| PyRuntimeError::new_err("Map agent not initialized"))?
                .clone();
            let (provider, model) = block_on_detached(py, async move {
                let agent_guard = agent_arc.lock().await;
                (agent_guard.provider(), agent_guard.model().to_string())
            })?;
            recreate_agent_service(py, provider, model).map_err(|e| {
                PyRuntimeError::new_err(format!("Failed to recreate map agent service: {e}"))
            })?
        };

        let reduce_agent_service = {
            let agent_arc = reduce_agent
                .inner
                .as_ref()
                .ok_or_else(|| PyRuntimeError::new_err("Reduce agent not initialized"))?
                .clone();
            let (provider, model) = block_on_detached(py, async move {
                let agent_guard = agent_arc.lock().await;
                (agent_guard.provider(), agent_guard.model().to_string())
            })?;
            recreate_agent_service(py, provider, model).map_err(|e| {
                PyRuntimeError::new_err(format!("Failed to recreate reduce agent service: {e}"))
            })?
        };

        let map_prompt_template = PromptTemplate::new(map_prompt);
        let reduce_prompt_template = PromptTemplate::new(reduce_prompt);

        let map_chain = Arc::new(caliche::chains::LLMChain::new(
            Arc::new(tokio::sync::Mutex::new(map_agent_service)),
            map_prompt_template,
        )) as Arc<dyn Chain>;
        let reduce_chain = Arc::new(caliche::chains::LLMChain::new(
            Arc::new(tokio::sync::Mutex::new(reduce_agent_service)),
            reduce_prompt_template,
        )) as Arc<dyn Chain>;

        let chain = caliche::chains::MapReduceChain::new(map_chain, reduce_chain);
        Ok(Self {
            inner: Arc::new(chain),
        })
    }

    /// Run the map-reduce chain
    pub fn run(&self, items: &Bound<'_, PyList>) -> PyResult<Py<PyAny>> {
        let py = items.py();

        // Convert Python list of dicts to JSON Values
        let mut json_items = Vec::new();
        for item in items.iter() {
            let item_dict = item.cast::<PyDict>()?;
            let mut json_map = serde_json::Map::new();
            for (key, value) in item_dict {
                let key_str = key.extract::<String>()?;
                let json_value = python_to_json_value(&value)?;
                json_map.insert(key_str, json_value);
            }
            json_items.push(Value::Object(json_map));
        }

        let chain_clone = self.inner.clone();
        let input = Value::Array(json_items);

        let result = block_on_detached(py, async move { chain_clone.run(&input).await })?;

        match result {
            Ok(chain_result) => {
                let output_value = &chain_result.output;
                let output_dict = json_to_python_dict(py, output_value)?;
                Ok(output_dict.into_any().unbind())
            }
            Err(e) => Err(PyRuntimeError::new_err(format!(
                "Chain execution failed: {e}"
            ))),
        }
    }
}

/// Python wrapper for `ParallelChain`
#[pyclass]
pub struct PyParallelChain {
    inner: Arc<caliche::chains::ParallelChain>,
}

#[pymethods]
impl PyParallelChain {
    #[new]
    pub fn new(chain1: &PyLLMChain, chain2: &PyLLMChain) -> Self {
        let chain1_arc = chain1.inner.clone() as Arc<dyn Chain>;
        let chain2_arc = chain2.inner.clone() as Arc<dyn Chain>;
        Self {
            inner: Arc::new(caliche::chains::ParallelChain::new(chain1_arc, chain2_arc)),
        }
    }

    /// Run the parallel chain
    pub fn run(&self, variables: &Bound<'_, PyDict>) -> PyResult<Py<PyAny>> {
        let py = variables.py();

        // Convert Python dict to JSON Value
        let mut json_map = serde_json::Map::new();
        for (key, value) in variables {
            let key_str = key.extract::<String>()?;
            let json_value = python_to_json_value(&value)?;
            json_map.insert(key_str, json_value);
        }
        let input_value = Value::Object(json_map);

        let chain_clone = self.inner.clone();

        let result = block_on_detached(py, async move { chain_clone.run(&input_value).await })?;

        match result {
            Ok(chain_result) => {
                let output_value = &chain_result.output;
                let output_dict = json_to_python_dict(py, output_value)?;
                Ok(output_dict.into_any().unbind())
            }
            Err(e) => Err(PyRuntimeError::new_err(format!(
                "Parallel chain execution failed: {e}"
            ))),
        }
    }
}

/// Python wrapper for `ChainBuilder`
#[pyclass]
pub struct PyChainBuilder {
    agent: Option<Agent>,
    template: Option<String>,
    output_key: Option<String>,
}

#[pymethods]
impl PyChainBuilder {
    #[new]
    pub fn new(agent: &Agent) -> Self {
        Self {
            agent: Some(Agent {
                inner: agent.inner.clone(),
            }),
            template: None,
            output_key: None,
        }
    }

    pub fn with_template(&mut self, template: String) {
        self.template = Some(template);
    }

    pub fn with_output_key(&mut self, key: String) {
        self.output_key = Some(key);
    }

    /// Build the chain
    pub fn build(&mut self, py: Python<'_>) -> PyResult<PyLLMChain> {
        let agent = self
            .agent
            .take()
            .ok_or_else(|| PyRuntimeError::new_err("Agent already consumed"))?;
        let template_str = self
            .template
            .take()
            .ok_or_else(|| PyRuntimeError::new_err("Template is required"))?;

        let agent_arc = agent
            .inner
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("Agent not initialized"))?
            .clone();

        let (provider, model) = block_on_detached(py, async move {
            let agent_guard = agent_arc.lock().await;
            (agent_guard.provider(), agent_guard.model().to_string())
        })?;
        let agent_service = recreate_agent_service(py, provider, model)?;

        let prompt = PromptTemplate::new(&template_str);
        let mut chain = caliche::chains::LLMChain::new(
            Arc::new(tokio::sync::Mutex::new(agent_service)),
            prompt,
        );

        if let Some(ref key) = self.output_key {
            chain = chain.with_output_key(key.clone());
        }

        Ok(PyLLMChain {
            inner: Arc::new(chain),
        })
    }
}
