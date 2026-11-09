//! Python bindings for MCP server functionality

use crate::conversions::cached_tool_parameters;
use crate::python_async::call_python_maybe_async;
use async_trait::async_trait;
use caliche::agent::tools::{Tool, ToolMeta, ToolResult};
use caliche::mcp::protocol::{LATEST_PROTOCOL_VERSION, SUPPORTED_PROTOCOL_VERSIONS};
use caliche::mcp::server::{McpServer, McpServerBuilder};
use caliche::mcp::transports::stdio::run_stdio_transport;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::PyAnyMethods;
use pythonize::pythonize;
use serde_json::Value;
use std::sync::{Arc, Mutex, OnceLock};

fn extract_bool_field(obj: &Bound<'_, PyAny>, name: &str) -> Option<bool> {
    let attr = obj.getattr(name).ok()?;
    (!attr.is_callable())
        .then(|| attr.extract::<bool>().ok())
        .flatten()
}

fn extract_bool_method(obj: &Bound<'_, PyAny>, name: &str) -> Option<bool> {
    let attr = obj.getattr(name).ok()?;
    if attr.is_callable() {
        attr.call0().ok()?.extract::<bool>().ok()
    } else {
        attr.extract::<bool>().ok()
    }
}

fn extract_string_field(obj: &Bound<'_, PyAny>, name: &str) -> Option<String> {
    let attr = obj.getattr(name).ok()?;
    if attr.is_callable() {
        attr.call0().ok()?.extract::<String>().ok()
    } else {
        attr.extract::<String>().ok()
    }
}

fn extract_optional_string_field(obj: &Bound<'_, PyAny>, name: &str) -> Option<String> {
    let attr = obj.getattr(name).ok()?;
    if attr.is_callable() {
        attr.call0()
            .ok()?
            .extract::<Option<String>>()
            .ok()
            .flatten()
    } else if attr.is_none() {
        None
    } else {
        attr.extract::<String>().ok()
    }
}

/// Python wrapper for Rust Tool trait for MCP
#[derive(Debug)]
pub struct PyMcpTool {
    tool_obj: Py<PyAny>,
    name: OnceLock<String>,
    description: OnceLock<String>,
    parameters_cache: Arc<Mutex<Option<Value>>>,
}

impl PyMcpTool {
    pub fn new(tool_obj: Py<PyAny>) -> Self {
        Self {
            tool_obj,
            name: OnceLock::new(),
            description: OnceLock::new(),
            parameters_cache: Arc::new(Mutex::new(None)),
        }
    }

    fn get_name(&self) -> &str {
        self.name.get_or_init(|| {
            Python::attach(|py| {
                self.tool_obj
                    .call_method0(py, "name")
                    .and_then(|result| result.extract::<String>(py))
                    .unwrap_or_else(|_| "unknown_tool".to_string())
            })
        })
    }

    fn get_description(&self) -> &str {
        self.description.get_or_init(|| {
            Python::attach(|py| {
                self.tool_obj
                    .call_method0(py, "description")
                    .and_then(|result| result.extract::<String>(py))
                    .unwrap_or_else(|_| "No description".to_string())
            })
        })
    }

    fn get_parameters(&self) -> Value {
        cached_tool_parameters(&self.tool_obj, &self.parameters_cache)
    }

    async fn execute_async(&self, arguments: Value) -> ToolResult {
        let py_result = Python::attach(|py| -> PyResult<Py<PyAny>> {
            let execute_method = self.tool_obj.getattr(py, "execute")?;
            let py_args = pythonize(py, &arguments).map_err(|e| {
                PyRuntimeError::new_err(format!("Failed to serialize arguments: {e}"))
            })?;
            execute_method.call1(py, (py_args,))
        });

        let py_result = match py_result {
            Ok(result) => result,
            Err(e) => {
                return ToolResult::error(format!("Python tool execution error: {e}"));
            }
        };

        let final_result = call_python_maybe_async(py_result).await;

        match final_result {
            Ok(result_obj) => Python::attach(|py| {
                let bound = result_obj.bind(py);
                Self::process_tool_result(py, bound)
                    .unwrap_or_else(|e| ToolResult::error(format!("Failed to process result: {e}")))
            }),
            Err(e) => ToolResult::error(format!("Python execution error: {e}")),
        }
    }

    fn process_tool_result(_py: Python, py_result: &Bound<PyAny>) -> PyResult<ToolResult> {
        if py_result.hasattr("is_success")? || py_result.hasattr("success")? {
            let success_val = extract_bool_field(py_result, "success")
                .or_else(|| extract_bool_method(py_result, "is_success"))
                .unwrap_or(true);
            let content_val = extract_string_field(py_result, "content").unwrap_or_default();
            let error_val = extract_optional_string_field(py_result, "error");

            return Ok(ToolResult {
                success: success_val,
                content: content_val,
                typed_payload: None,
                error: error_val,
                usage: None,
                warnings: None,
            });
        }

        if let Ok(dict) = py_result.cast::<pyo3::types::PyDict>() {
            let success = match dict.get_item("success") {
                Ok(Some(v)) => v.extract::<bool>().ok().unwrap_or(true),
                _ => true,
            };

            let content = match dict.get_item("content") {
                Ok(Some(v)) => v.extract::<String>().ok().unwrap_or_else(|| {
                    v.cast::<pyo3::types::PyString>()
                        .ok()
                        .and_then(|s| s.extract::<String>().ok())
                        .unwrap_or_default()
                }),
                _ => String::new(),
            };

            let error = match dict.get_item("error") {
                Ok(Some(v)) => v.extract::<String>().ok(),
                _ => None,
            };

            return Ok(ToolResult {
                success,
                content,
                typed_payload: None,
                error,
                usage: None,
                warnings: None,
            });
        }

        if let Ok(content) = py_result.extract::<String>() {
            return Ok(ToolResult::success(content));
        }

        let content = py_result
            .repr()
            .and_then(|r| r.extract::<String>())
            .unwrap_or_else(|_| "Unknown result".to_string());
        Ok(ToolResult::success(content))
    }
}

impl ToolMeta for PyMcpTool {
    fn name(&self) -> &str {
        self.get_name()
    }

    fn description(&self) -> &str {
        self.get_description()
    }

    fn parameters(&self) -> Value {
        self.get_parameters()
    }
}

#[async_trait]
impl Tool for PyMcpTool {
    async fn execute(&self, arguments: Value) -> ToolResult {
        self.execute_async(arguments).await
    }
}

/// Python MCP Server Builder
#[pyclass]
pub struct PyMcpServerBuilder {
    builder: Arc<Mutex<McpServerBuilder>>,
}

impl PyMcpServerBuilder {
    fn update_builder(
        &self,
        update: impl FnOnce(McpServerBuilder) -> McpServerBuilder,
    ) -> PyResult<()> {
        let mut builder = self
            .builder
            .lock()
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to lock builder: {e}")))?;
        *builder = update(std::mem::take(&mut *builder));
        drop(builder);
        Ok(())
    }
}

#[pymethods]
impl PyMcpServerBuilder {
    #[new]
    fn new() -> Self {
        Self {
            builder: Arc::new(Mutex::new(McpServerBuilder::new())),
        }
    }

    fn name(&mut self, name: String) -> PyResult<()> {
        self.update_builder(|builder| builder.name(name))
    }

    fn version(&mut self, version: String) -> PyResult<()> {
        self.update_builder(|builder| builder.version(version))
    }

    fn enable_tools(&mut self, enabled: bool) -> PyResult<()> {
        self.update_builder(|builder| builder.enable_tools(enabled))
    }

    fn enable_prompts(&mut self, enabled: bool) -> PyResult<()> {
        self.update_builder(|builder| builder.enable_prompts(enabled))
    }

    fn enable_resources(&mut self, enabled: bool) -> PyResult<()> {
        self.update_builder(|builder| builder.enable_resources(enabled))
    }

    fn add_tool(&mut self, tool_obj: Py<PyAny>) -> PyResult<()> {
        let py_tool = PyMcpTool::new(tool_obj);
        let tool: Arc<dyn Tool> = Arc::new(py_tool);
        self.update_builder(|builder| builder.add_tool(tool))
    }

    fn build(&self) -> PyResult<PyMcpServer> {
        let mut builder = self
            .builder
            .lock()
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to lock builder: {e}")))?;
        // Move the builder out to preserve all configured registries and options.
        let builder_to_build = std::mem::take(&mut *builder);

        drop(builder);

        let rt = crate::get_runtime()?;
        let server = rt.block_on(async move {
            builder_to_build
                .build()
                .map_err(|e| PyRuntimeError::new_err(format!("Failed to build MCP server: {e}")))
        })?;

        Ok(PyMcpServer {
            server: Arc::new(Mutex::new(server)),
        })
    }
}

/// Python MCP Server
#[pyclass]
pub struct PyMcpServer {
    server: Arc<Mutex<McpServer>>,
}

#[pymethods]
impl PyMcpServer {
    /// Run the server with stdio transport
    fn run_stdio(&self) -> PyResult<()> {
        // Take ownership of the server from the mutex
        let server = self
            .server
            .lock()
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to lock server: {e}")))?;
        // Clone the server (it implements Clone)
        let server_clone = server.clone();
        drop(server);

        let rt = crate::get_runtime()?;
        rt.block_on(async move {
            let handler = server_clone.create_handler();
            run_stdio_transport(handler)
                .await
                .map_err(|e| PyRuntimeError::new_err(format!("Failed to run stdio transport: {e}")))
        })?;

        Ok(())
    }
}

/// Get the latest MCP protocol version
#[pyfunction]
pub const fn get_latest_protocol_version() -> &'static str {
    LATEST_PROTOCOL_VERSION
}

/// Get supported MCP protocol versions
#[pyfunction]
pub fn get_supported_protocol_versions() -> Vec<&'static str> {
    SUPPORTED_PROTOCOL_VERSIONS.to_vec()
}
