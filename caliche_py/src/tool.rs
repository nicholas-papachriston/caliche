//! Python bindings for custom Tool implementation

use crate::conversions::cached_tool_parameters;
use crate::python_async::{call_python_maybe_async, call_python_result};
use async_trait::async_trait;
use caliche::agent::tools::{Tool, ToolMeta, ToolResult};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::PyAnyMethods;
use pyo3::types::{PyDict, PyString};
use pythonize::pythonize;
use serde_json::Value;
use std::sync::{Arc, Mutex, OnceLock};

/// Python wrapper for Rust Tool trait
///
/// This allows Python objects to implement the Tool interface
/// by calling Python methods from Rust.
#[derive(Debug)]
pub struct PyTool {
    tool_obj: Py<PyAny>,
    // Use OnceLock for immutable fields that are set once and read many times
    // This avoids Box::leak and allows returning &str references
    name: OnceLock<String>,
    description: OnceLock<String>,
    parameters_cache: Arc<Mutex<Option<Value>>>,
}

impl PyTool {
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

    fn tool_is_async(&self) -> bool {
        Python::attach(|py| {
            self.tool_obj
                .call_method0(py, "is_async")
                .and_then(|result| result.extract::<bool>(py))
                .unwrap_or(true)
        })
    }

    async fn execute_async(&self, arguments: Value) -> ToolResult {
        let is_async = self.tool_is_async();
        // Convert arguments to Python dict using pythonize for efficiency
        let py_result = Python::attach(|py| -> PyResult<Py<PyAny>> {
            // Get the execute method
            let execute_method = self.tool_obj.getattr(py, "execute")?;

            // Use pythonize to convert Value to Python object directly
            let py_args = pythonize(py, &arguments).map_err(|e| {
                PyRuntimeError::new_err(format!("Failed to serialize arguments: {e}"))
            })?;

            // Call execute method
            execute_method.call1(py, (py_args,))
        });

        let py_result = match py_result {
            Ok(result) => result,
            Err(e) => {
                return ToolResult::error(format!("Python tool execution error: {e}"));
            }
        };

        // Sync tools skip coroutine detection; async tools use the shared bridge.
        let final_result = if is_async {
            call_python_maybe_async(py_result).await
        } else {
            call_python_result(py_result, false).await
        };

        match final_result {
            Ok(result_obj) => Python::attach(|py| {
                let bound = result_obj.bind(py);
                Self::process_tool_result(py, bound)
                    .unwrap_or_else(|e| ToolResult::error(format!("Failed to process result: {e}")))
            }),
            Err(e) => ToolResult::error(format!("Python execution error: {e}")),
        }
    }

    fn process_tool_result(py: Python, py_result: &Bound<PyAny>) -> PyResult<ToolResult> {
        // Try to extract ToolResult from Python
        // The Python tool should return a ToolResult object or a dict/string

        // Check if it's a ToolResult instance - support attribute getters and method aliases.
        if py_result.hasattr("is_success")? || py_result.hasattr("success")? {
            let success_val = extract_bool_field(py_result, "success")
                .or_else(|| extract_bool_method(py_result, "is_success"))
                .unwrap_or(true);
            let content_val = extract_string_field(py_result, "content").unwrap_or_default();
            let error_val = extract_optional_string_field(py_result, "error");
            let typed_payload = extract_typed_payload(py, py_result);

            return Ok(ToolResult {
                success: success_val,
                content: content_val,
                typed_payload,
                error: error_val,
                usage: None,
                warnings: None,
            });
        }

        // Check if it's a dict with success/content/error
        if let Ok(dict) = py_result.cast::<PyDict>() {
            let success = match dict.get_item("success") {
                Ok(Some(v)) => v.extract::<bool>().ok().unwrap_or(true),
                _ => true,
            };

            let content = match dict.get_item("content") {
                Ok(Some(v)) => v.extract::<String>().ok().unwrap_or_else(|| {
                    // Try to convert to string via cast
                    v.cast::<PyString>()
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

            let typed_payload = match dict.get_item("typed_payload") {
                Ok(Some(v)) => pythonize::depythonize::<Value>(&v).ok(),
                _ => None,
            };

            return Ok(ToolResult {
                success,
                content,
                typed_payload,
                error,
                usage: None,
                warnings: None,
            });
        }

        // Try to convert to string
        if let Ok(content) = py_result.extract::<String>() {
            return Ok(ToolResult::success(content));
        }

        // Last resort: convert to string representation
        let content = py_result
            .repr()
            .and_then(|r| r.extract::<String>())
            .unwrap_or_else(|_| "Unknown result".to_string());
        Ok(ToolResult::success(content))
    }
}

impl ToolMeta for PyTool {
    fn name(&self) -> &str {
        // OnceLock::get_or_init ensures the value is initialized once and returns &str
        // The reference is valid for the lifetime of self since OnceLock stores the String
        self.get_name()
    }

    fn description(&self) -> &str {
        // OnceLock::get_or_init ensures the value is initialized once and returns &str
        // The reference is valid for the lifetime of self since OnceLock stores the String
        self.get_description()
    }

    fn parameters(&self) -> Value {
        self.get_parameters()
    }
}

#[async_trait]
impl Tool for PyTool {
    async fn execute(&self, arguments: Value) -> ToolResult {
        self.execute_async(arguments).await
    }
}

fn extract_bool_field(obj: &Bound<'_, PyAny>, name: &str) -> Option<bool> {
    let attr = obj.getattr(name).ok()?;
    if attr.is_callable() {
        return None;
    }
    attr.extract::<bool>().ok()
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
            .ok()
            .and_then(|value| value.extract::<Option<String>>().ok())
            .flatten()
    } else if attr.is_none() {
        None
    } else {
        attr.extract::<String>().ok()
    }
}

fn extract_typed_payload(py: Python<'_>, obj: &Bound<'_, PyAny>) -> Option<Value> {
    let _ = py;
    let attr = obj.getattr("typed_payload").ok()?;
    if attr.is_none() {
        return None;
    }
    if attr.is_callable() {
        let value = attr.call0().ok()?;
        return pythonize::depythonize::<Value>(&value).ok();
    }
    pythonize::depythonize::<Value>(&attr).ok()
}
