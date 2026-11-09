//! Python bindings for Agent

use crate::block_on_detached;
use crate::get_runtime;
use crate::shared_state::SharedState as PySharedState;
use caliche::AgentService;
use caliche::messages::PerformanceMetrics;
use futures::StreamExt;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use pythonize::depythonize;
use std::sync::Arc;
use tokio::sync::mpsc;

fn performance_metrics_to_pydict<'py>(
    py: Python<'py>,
    metrics: Option<&PerformanceMetrics>,
) -> PyResult<Bound<'py, PyDict>> {
    let metrics_dict = PyDict::new(py);
    if let Some(m) = metrics {
        if let Some(v) = m.total_duration_ms {
            metrics_dict.set_item("total_duration_ms", v)?;
        }
        if let Some(v) = m.llm_duration_ms {
            metrics_dict.set_item("llm_duration_ms", v)?;
        }
        if let Some(v) = m.tool_duration_ms {
            metrics_dict.set_item("tool_duration_ms", v)?;
        }
        if let Some(v) = m.overhead_ms {
            metrics_dict.set_item("overhead_ms", v)?;
        }
        if let Some(v) = m.step_count {
            metrics_dict.set_item("step_count", v)?;
        }
        if let Some(v) = m.time_to_first_token_ms {
            metrics_dict.set_item("time_to_first_token_ms", v)?;
        }
        if let Some(steps) = &m.steps {
            let steps_list = PyList::empty(py);
            for step in steps {
                let step_dict = PyDict::new(py);
                step_dict.set_item("step", step.step)?;
                step_dict.set_item("llm_duration_ms", step.llm_duration_ms)?;
                step_dict.set_item("tool_duration_ms", step.tool_duration_ms)?;
                step_dict.set_item("tools_executed", step.tools_executed.clone())?;
                steps_list.append(step_dict)?;
            }
            metrics_dict.set_item("steps", steps_list)?;
        }
    }
    Ok(metrics_dict)
}

#[pyclass]
pub struct Agent {
    pub(crate) inner: Option<Arc<tokio::sync::Mutex<AgentService>>>,
}

#[pymethods]
impl Agent {
    /// Create a new agent with default settings
    #[new]
    pub const fn new() -> Self {
        // Note: This is a placeholder - actual initialization should be async
        // For now, we'll use a builder pattern
        Self { inner: None }
    }

    /// Create an isolated request-scoped agent for concurrent work.
    ///
    /// The clone shares immutable provider and tool configuration while using
    /// independent session and conversation state.
    pub fn clone_for_request(&self, py: Python<'_>) -> PyResult<Self> {
        let agent_arc = self.inner.as_ref().ok_or_else(|| {
            PyRuntimeError::new_err("Agent not initialized. Use AgentBuilder to create an agent.")
        })?;
        let agent_clone = Arc::clone(agent_arc);
        let service = block_on_detached(py, async move {
            let agent = agent_clone.lock().await;
            agent.clone_for_request()
        })?;
        Ok(Self {
            inner: Some(Arc::new(tokio::sync::Mutex::new(service))),
        })
    }

    /// Chat with structured output (returns JSON string that can be parsed)
    pub fn chat_structured(
        &mut self,
        py: Python<'_>,
        message: &str,
        schema: &Bound<'_, PyDict>,
    ) -> PyResult<(String, Py<PyAny>)> {
        if let Some(ref agent_arc) = self.inner {
            let schema_json: serde_json::Value = depythonize(schema)
                .map_err(|e| PyRuntimeError::new_err(format!("Invalid JSON schema: {e}")))?;

            let agent_clone = Arc::clone(agent_arc);
            let msg = message.to_string();

            // Release GIL while waiting for LLM response
            let (parsed_value, usage) = block_on_detached(py, async move {
                let mut agent = agent_clone.lock().await;
                agent
                    .chat_structured::<serde_json::Value>(&msg, schema_json)
                    .await
            })?
            .map_err(|e| PyRuntimeError::new_err(format!("Structured chat failed: {e}")))?;

            // Convert parsed value back to JSON string
            let response_json = serde_json::to_string(&parsed_value).map_err(|e| {
                PyRuntimeError::new_err(format!("Failed to serialize response: {e}"))
            })?;

            let usage_dict = PyDict::new(py);
            if let Some(ref usage_val) = usage {
                if let Some(prompt) = usage_val.prompt_tokens {
                    usage_dict.set_item("prompt_tokens", prompt)?;
                }
                if let Some(completion) = usage_val.completion_tokens {
                    usage_dict.set_item("completion_tokens", completion)?;
                }
                if let Some(total) = usage_val.total_tokens {
                    usage_dict.set_item("total_tokens", total)?;
                }
                if let Some(embedding) = usage_val.embedding_tokens {
                    usage_dict.set_item("embedding_tokens", embedding)?;
                }
            }

            Ok((response_json, usage_dict.unbind().into()))
        } else {
            Err(PyRuntimeError::new_err(
                "Agent not initialized. Use AgentBuilder to create an agent.",
            ))
        }
    }

    /// Chat with the agent. Returns (`response_text`, `usage_dict`, `metrics_dict`).
    /// `metrics_dict` may be None (as a dict with no keys) when metrics are not available.
    pub fn chat(
        &mut self,
        py: Python<'_>,
        message: &str,
    ) -> PyResult<(String, Py<PyAny>, Py<PyAny>)> {
        if let Some(ref agent_arc) = self.inner {
            let agent_clone = Arc::clone(agent_arc);
            let msg = message.to_string();

            // Release GIL while waiting for LLM response
            let chat_response = block_on_detached(py, async move {
                let mut agent = agent_clone.lock().await;
                agent.chat(&msg).await
            })?
            .map_err(|e| PyRuntimeError::new_err(format!("Chat failed: {e}")))?;

            let response_text = chat_response.text().to_string();
            let usage = chat_response.usage();

            let usage_dict = PyDict::new(py);
            if let Some(prompt) = usage.prompt_tokens {
                usage_dict.set_item("prompt_tokens", prompt)?;
            }
            if let Some(completion) = usage.completion_tokens {
                usage_dict.set_item("completion_tokens", completion)?;
            }
            if let Some(total) = usage.total_tokens {
                usage_dict.set_item("total_tokens", total)?;
            }
            if let Some(embedding) = usage.embedding_tokens {
                usage_dict.set_item("embedding_tokens", embedding)?;
            }

            let metrics_dict = performance_metrics_to_pydict(py, chat_response.metrics.as_ref())?;

            Ok((
                response_text,
                usage_dict.unbind().into(),
                metrics_dict.unbind().into(),
            ))
        } else {
            Err(PyRuntimeError::new_err(
                "Agent not initialized. Use AgentBuilder to create an agent.",
            ))
        }
    }

    /// Native async chat — returns a Python awaitable bound to the asyncio loop.
    pub fn chat_async<'py>(
        &mut self,
        py: Python<'py>,
        message: String,
    ) -> PyResult<Bound<'py, PyAny>> {
        let agent_arc = self.inner.clone().ok_or_else(|| {
            PyRuntimeError::new_err("Agent not initialized. Use AgentBuilder to create an agent.")
        })?;
        let _ = get_runtime()?;
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let chat_response = {
                let mut agent = agent_arc.lock().await;
                agent
                    .chat(&message)
                    .await
                    .map_err(|e| PyRuntimeError::new_err(format!("Chat failed: {e}")))?
            };
            Python::attach(|py| -> PyResult<(String, Py<PyAny>, Py<PyAny>)> {
                let response_text = chat_response.text().to_string();
                let usage = chat_response.usage();
                let usage_dict = PyDict::new(py);
                if let Some(prompt) = usage.prompt_tokens {
                    usage_dict.set_item("prompt_tokens", prompt)?;
                }
                if let Some(completion) = usage.completion_tokens {
                    usage_dict.set_item("completion_tokens", completion)?;
                }
                if let Some(total) = usage.total_tokens {
                    usage_dict.set_item("total_tokens", total)?;
                }
                if let Some(embedding) = usage.embedding_tokens {
                    usage_dict.set_item("embedding_tokens", embedding)?;
                }
                let metrics_dict =
                    performance_metrics_to_pydict(py, chat_response.metrics.as_ref())?;
                Ok((
                    response_text,
                    usage_dict.unbind().into(),
                    metrics_dict.unbind().into(),
                ))
            })
        })
    }

    /// Get the provider name
    pub fn provider(&self, py: Python<'_>) -> PyResult<String> {
        self.inner.as_ref().map_or_else(
            || Err(PyRuntimeError::new_err("Agent not initialized")),
            |agent_arc| {
                let agent_clone = Arc::clone(agent_arc);
                block_on_detached(py, async move {
                    let agent = agent_clone.lock().await;
                    agent.provider().to_string()
                })
            },
        )
    }

    /// Get the model name
    pub fn model(&self, py: Python<'_>) -> PyResult<String> {
        self.inner.as_ref().map_or_else(
            || Err(PyRuntimeError::new_err("Agent not initialized")),
            |agent_arc| {
                let agent_clone = Arc::clone(agent_arc);
                block_on_detached(py, async move {
                    let agent = agent_clone.lock().await;
                    agent.model().to_string()
                })
            },
        )
    }

    /// Get the shared state
    ///
    /// Returns a reference to the agent's shared state, wrapped in a `SharedState` `PyClass`.
    pub fn shared_state(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        if let Some(ref agent_arc) = self.inner {
            let agent_clone = Arc::clone(agent_arc);
            let rust_agent = block_on_detached(py, async move {
                let agent_service = agent_clone.lock().await;
                agent_service.clone_agent()
            })?;
            let shared_state = rust_agent.shared_state;

            // Wrap the SharedState in the PyClass
            let py_shared_state = PySharedState {
                inner: shared_state,
            };
            Ok(Py::new(py, py_shared_state)?.into_any())
        } else {
            Err(PyRuntimeError::new_err("Agent not initialized"))
        }
    }

    /// Access the shared tool registry (deferred/active tool controls).
    pub fn tool_registry(&self, py: Python<'_>) -> PyResult<crate::tool_registry::PyToolRegistry> {
        if let Some(ref agent_arc) = self.inner {
            let agent_clone = Arc::clone(agent_arc);
            let registry = block_on_detached(py, async move {
                let agent = agent_clone.lock().await;
                Arc::clone(agent.tool_registry())
            })?;
            Ok(crate::tool_registry::PyToolRegistry::new(registry))
        } else {
            Err(PyRuntimeError::new_err(
                "Agent not initialized. Use AgentBuilder to create an agent.",
            ))
        }
    }

    /// List all available tools
    pub fn list_tools(&self, py: Python<'_>) -> PyResult<Vec<Py<PyAny>>> {
        if let Some(ref agent_arc) = self.inner {
            let agent_clone = Arc::clone(agent_arc);
            let tools: Vec<(String, String, serde_json::Value)> =
                block_on_detached(py, async move {
                    let agent = agent_clone.lock().await;
                    agent.list_tools()
                })?;

            // Convert to Python list of dicts
            let mut tools_list = Vec::new();
            for (name, description, parameters) in tools {
                let tool_dict = PyDict::new(py);
                tool_dict.set_item("name", name).map_err(|e| {
                    PyRuntimeError::new_err(format!("Failed to set tool name: {e}"))
                })?;
                tool_dict
                    .set_item("description", description)
                    .map_err(|e| {
                        PyRuntimeError::new_err(format!("Failed to set tool description: {e}"))
                    })?;
                let py_params =
                    crate::conversions::json_to_python_value(py, &parameters).map_err(|e| {
                        PyRuntimeError::new_err(format!("Failed to convert parameters: {e}"))
                    })?;
                tool_dict.set_item("parameters", py_params).map_err(|e| {
                    PyRuntimeError::new_err(format!("Failed to set tool parameters: {e}"))
                })?;
                tools_list.push(tool_dict.into());
            }

            Ok(tools_list)
        } else {
            Err(PyRuntimeError::new_err("Agent not initialized"))
        }
    }

    /// Stream chat responses. Optionally resume from a durable stream event ID.
    #[pyo3(signature = (message, *, resume_from=None))]
    pub fn stream_chat(
        &mut self,
        py: Python<'_>,
        message: String,
        resume_from: Option<String>,
    ) -> PyResult<Py<PyAny>> {
        if let Some(ref agent_arc) = self.inner {
            let (tx, rx) = mpsc::unbounded_channel();
            let message_clone = message;
            let agent_clone = Arc::clone(agent_arc);

            let rt_handle = get_runtime()?.handle().clone();
            rt_handle.spawn(async move {
                let request_agent = {
                    let agent = agent_clone.lock().await;
                    agent.clone_for_request()
                };
                let mut stream = request_agent.stream_chat(&message_clone);
                while let Some(item) = stream.next().await {
                    if tx.send(item).is_err() {
                        break;
                    }
                }
            });

            let receiver_stream = futures::stream::unfold(rx, |mut rx| async move {
                rx.recv().await.map(|item| (item, rx))
            });

            let boxed_stream: std::pin::Pin<
                Box<
                    dyn futures::Stream<Item = anyhow::Result<caliche::messages::StreamEvent>>
                        + Send,
                >,
            > = Box::pin(receiver_stream);

            let stream_iter = crate::streaming::StreamIterator::new(boxed_stream, resume_from);

            Ok(Py::new(py, stream_iter)?.into_any())
        } else {
            Err(PyRuntimeError::new_err("Agent not initialized"))
        }
    }
}

impl Agent {
    pub(crate) fn from_service(service: AgentService) -> Self {
        Self {
            inner: Some(Arc::new(tokio::sync::Mutex::new(service))),
        }
    }
}
