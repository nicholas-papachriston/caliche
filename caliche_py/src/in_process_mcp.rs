//! Python bindings for in-process MCP servers

use crate::tool::PyTool;
use caliche::agent::tools::Tool;
use caliche::mcp::in_process_server::{
    InProcessMcpServer as RustInProcessMcpServer,
    InProcessMcpServerBuilder as RustInProcessMcpServerBuilder,
};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pythonize::pythonize;
use std::sync::Arc;

/// Python wrapper for `InProcessMcpServerBuilder`
///
/// Stores tools separately since Rust builder doesn't support mutation
#[pyclass]
pub struct PyInProcessMcpServerBuilder {
    name: String,
    version: String,
    tools: Vec<Arc<dyn Tool>>,
}

#[pymethods]
impl PyInProcessMcpServerBuilder {
    #[new]
    fn new(name: String) -> Self {
        Self {
            name,
            version: env!("CARGO_PKG_VERSION").to_string(),
            tools: Vec::new(),
        }
    }

    fn version(mut self_: PyRefMut<Self>, version: String) -> PyRefMut<Self> {
        self_.version = version;
        self_
    }

    fn add_tool(mut self_: PyRefMut<Self>, tool_obj: Py<PyAny>) -> PyRefMut<Self> {
        let py_tool = Arc::new(PyTool::new(tool_obj)) as Arc<dyn Tool>;
        self_.tools.push(py_tool);
        self_
    }

    fn build(&self) -> PyResult<PyInProcessMcpServer> {
        let mut builder =
            RustInProcessMcpServerBuilder::new(self.name.clone()).version(self.version.clone());

        // Add all tools at once
        if !self.tools.is_empty() {
            builder = builder.add_tools(self.tools.clone());
        }

        let server = builder
            .build()
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to build MCP server: {e}")))?;

        Ok(PyInProcessMcpServer {
            inner: Arc::new(server),
        })
    }
}

/// Python wrapper for `InProcessMcpServer`
#[pyclass]
pub struct PyInProcessMcpServer {
    pub(crate) inner: Arc<RustInProcessMcpServer>,
}

#[pymethods]
impl PyInProcessMcpServer {
    #[getter]
    fn name(&self) -> String {
        self.inner.name().to_string()
    }

    #[getter]
    fn version(&self) -> String {
        self.inner.version().to_string()
    }

    /// List all available tools as MCP tools
    fn list_tools(&self) -> Vec<Py<PyAny>> {
        Python::attach(|py| {
            let mut tools = Vec::new();
            for mcp_tool in self.inner.list_tools() {
                let dict = pyo3::types::PyDict::new(py);
                if let Err(e) = dict.set_item("name", mcp_tool.name.as_ref()) {
                    #[expect(
                        clippy::print_stderr,
                        reason = "surface MCP tool export failures on the Python process stderr"
                    )]
                    {
                        eprintln!("Warning: Failed to set tool name: {e}");
                    }
                    continue;
                }
                if let Err(e) = dict.set_item("description", mcp_tool.description.as_ref()) {
                    #[expect(
                        clippy::print_stderr,
                        reason = "surface MCP tool export failures on the Python process stderr"
                    )]
                    {
                        eprintln!("Warning: Failed to set tool description: {e}");
                    }
                    continue;
                }
                let schema = match pythonize(py, &mcp_tool.input_schema) {
                    Ok(s) => s,
                    Err(e) => {
                        #[expect(
                            clippy::print_stderr,
                            reason = "surface MCP tool export failures on the Python process stderr"
                        )]
                        {
                            eprintln!("Warning: Failed to pythonize input schema: {e}");
                        }
                        continue;
                    }
                };
                if let Err(e) = dict.set_item("input_schema", schema) {
                    #[expect(
                        clippy::print_stderr,
                        reason = "surface MCP tool export failures on the Python process stderr"
                    )]
                    {
                        eprintln!("Warning: Failed to set input schema: {e}");
                    }
                    continue;
                }
                tools.push(dict.into());
            }
            tools
        })
    }

    /// Call a tool by name with arguments
    fn call_tool(&self, name: &str, arguments: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        let py_args = {
            pythonize::depythonize(arguments)
                .map_err(|e| PyRuntimeError::new_err(format!("Failed to parse arguments: {e}")))
        }?;

        let rt = crate::get_runtime()?;
        let result = rt.block_on(async {
            self.inner
                .call_tool(name, py_args)
                .await
                .map_err(|e| PyRuntimeError::new_err(format!("Tool execution failed: {e}")))
        })?;

        Python::attach(|py| {
            let dict = pyo3::types::PyDict::new(py);
            if let Some(ref content) = result.content {
                let mut content_list: Vec<Py<PyAny>> = Vec::new();
                for part in content {
                    match part {
                        caliche::mcp::protocol::ContentPart::Text { text } => {
                            let part_dict = pyo3::types::PyDict::new(py);
                            if let Err(e) = part_dict.set_item("type", "text") {
                                return Err(PyRuntimeError::new_err(format!(
                                    "Failed to set content type: {e}"
                                )));
                            }
                            if let Err(e) = part_dict.set_item("text", text.clone()) {
                                return Err(PyRuntimeError::new_err(format!(
                                    "Failed to set text content: {e}"
                                )));
                            }
                            content_list.push(part_dict.into());
                        }
                        caliche::mcp::protocol::ContentPart::Resource { resource } => {
                            let part_dict = pyo3::types::PyDict::new(py);
                            if let Err(e) = part_dict.set_item("type", "resource") {
                                return Err(PyRuntimeError::new_err(format!(
                                    "Failed to set resource type: {e}"
                                )));
                            }
                            let resource_dict = pyo3::types::PyDict::new(py);
                            if let Err(e) = resource_dict.set_item("uri", resource.uri.clone()) {
                                return Err(PyRuntimeError::new_err(format!(
                                    "Failed to set resource URI: {e}"
                                )));
                            }
                            if let Some(ref text) = resource.text
                                && let Err(e) = resource_dict.set_item("text", text.clone())
                            {
                                return Err(PyRuntimeError::new_err(format!(
                                    "Failed to set resource text: {e}"
                                )));
                            }
                            if let Some(ref mime_type) = resource.mime_type
                                && let Err(e) =
                                    resource_dict.set_item("mimeType", mime_type.clone())
                            {
                                return Err(PyRuntimeError::new_err(format!(
                                    "Failed to set mime type: {e}"
                                )));
                            }
                            if let Err(e) = part_dict.set_item("resource", resource_dict) {
                                return Err(PyRuntimeError::new_err(format!(
                                    "Failed to set resource: {e}"
                                )));
                            }
                            content_list.push(part_dict.into());
                        }
                        caliche::mcp::protocol::ContentPart::Image { data, mime_type } => {
                            let part_dict = pyo3::types::PyDict::new(py);
                            part_dict.set_item("type", "image").map_err(|e| {
                                PyRuntimeError::new_err(format!("Failed to set type: {e}"))
                            })?;
                            part_dict.set_item("data", data.clone()).map_err(|e| {
                                PyRuntimeError::new_err(format!("Failed to set data: {e}"))
                            })?;
                            part_dict
                                .set_item("mimeType", mime_type.clone())
                                .map_err(|e| {
                                    PyRuntimeError::new_err(format!("Failed to set mimeType: {e}"))
                                })?;
                            content_list.push(part_dict.into());
                        }
                        caliche::mcp::protocol::ContentPart::Audio { data, mime_type } => {
                            let part_dict = pyo3::types::PyDict::new(py);
                            part_dict.set_item("type", "audio").map_err(|e| {
                                PyRuntimeError::new_err(format!("Failed to set type: {e}"))
                            })?;
                            part_dict.set_item("data", data.clone()).map_err(|e| {
                                PyRuntimeError::new_err(format!("Failed to set data: {e}"))
                            })?;
                            part_dict
                                .set_item("mimeType", mime_type.clone())
                                .map_err(|e| {
                                    PyRuntimeError::new_err(format!("Failed to set mimeType: {e}"))
                                })?;
                            content_list.push(part_dict.into());
                        }
                        caliche::mcp::protocol::ContentPart::ResourceLink {
                            uri,
                            name,
                            description,
                            mime_type,
                        } => {
                            let part_dict = pyo3::types::PyDict::new(py);
                            part_dict.set_item("type", "resource_link").map_err(|e| {
                                PyRuntimeError::new_err(format!("Failed to set type: {e}"))
                            })?;
                            part_dict.set_item("uri", uri.clone()).map_err(|e| {
                                PyRuntimeError::new_err(format!("Failed to set uri: {e}"))
                            })?;
                            if let Some(name) = name {
                                part_dict.set_item("name", name.clone()).map_err(|e| {
                                    PyRuntimeError::new_err(format!("Failed to set name: {e}"))
                                })?;
                            }
                            if let Some(description) = description {
                                part_dict
                                    .set_item("description", description.clone())
                                    .map_err(|e| {
                                        PyRuntimeError::new_err(format!(
                                            "Failed to set description: {e}"
                                        ))
                                    })?;
                            }
                            if let Some(mime_type) = mime_type {
                                part_dict
                                    .set_item("mimeType", mime_type.clone())
                                    .map_err(|e| {
                                        PyRuntimeError::new_err(format!(
                                            "Failed to set mimeType: {e}"
                                        ))
                                    })?;
                            }
                            content_list.push(part_dict.into());
                        }
                    }
                }
                if let Err(e) = dict.set_item("content", content_list) {
                    return Err(PyRuntimeError::new_err(format!(
                        "Failed to set content: {e}"
                    )));
                }
            }
            if let Some(is_error) = result.is_error
                && let Err(e) = dict.set_item("is_error", is_error)
            {
                return Err(PyRuntimeError::new_err(format!(
                    "Failed to set is_error: {e}"
                )));
            }
            Ok(dict.into())
        })
    }
}

impl PyInProcessMcpServer {
    /// Get the inner server (for internal use)
    pub(crate) fn get_inner(&self) -> Arc<RustInProcessMcpServer> {
        Arc::clone(&self.inner)
    }
}
