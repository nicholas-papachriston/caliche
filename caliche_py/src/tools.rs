//! Python bindings for `ToolResult`

use pyo3::prelude::*;

/// Result of a tool execution.
///
/// Exposed to Python with attribute access (`result.success`, `result.content`,
/// `result.error`) matching the SDK and tests.
#[pyclass]
pub struct ToolResult {
    success: bool,
    content: String,
    error: Option<String>,
}

#[pymethods]
impl ToolResult {
    #[new]
    #[pyo3(signature = (success, content, error=None))]
    pub const fn new(success: bool, content: String, error: Option<String>) -> Self {
        Self {
            success,
            content,
            error,
        }
    }

    #[getter]
    pub const fn success(&self) -> bool {
        self.success
    }

    #[getter]
    pub fn content(&self) -> &str {
        &self.content
    }

    #[getter]
    pub fn error(&self) -> Option<&str> {
        self.error.as_deref()
    }
}
