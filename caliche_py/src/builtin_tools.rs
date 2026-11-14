//! Python bindings for Caliche Built-in Tools

use crate::mutation::PyMutationStyle;
use crate::{conversions::json_to_python_dict, get_runtime};
use caliche::agent::hooks::HookRegistry;
use caliche::agent::tools::execution::HookExecutionMiddleware;
use caliche::agent::tools::{
    ApplyPatchTool, CalculatorTool, CodeExecutionTool, FsTool, HttpTool, MutationStyle, Tool,
};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use pythonize::depythonize;
use std::sync::Arc;

/// The tool gate for a call from Python.
pub(crate) fn tool_gate(workspace: Option<String>) -> HookExecutionMiddleware {
    let gate = HookExecutionMiddleware::new(HookRegistry::new());
    match workspace {
        Some(root) => gate.with_workspace(root),
        None => gate,
    }
}

/// Create a `CalculatorTool` instance
#[pyfunction]
pub fn create_calculator_tool() -> PyBuiltinTool {
    PyBuiltinTool {
        inner: Arc::new(CalculatorTool) as Arc<dyn Tool>,
    }
}

/// Create an `FsTool` instance.
///
/// When `mutation_style` is `ApplyPatch`, write/edit ops are disabled on `fs`
/// (use [`create_apply_patch_tool`] instead).
#[pyfunction]
#[pyo3(signature = (mutation_style=None))]
pub fn create_fs_tool(mutation_style: Option<PyMutationStyle>) -> PyBuiltinTool {
    let style = mutation_style.map_or(MutationStyle::EditWrite, MutationStyle::from);
    PyBuiltinTool {
        inner: Arc::new(FsTool::with_mutation_style(style)) as Arc<dyn Tool>,
    }
}

/// Create an `ApplyPatchTool` instance (Codex-style freeform patch).
#[pyfunction]
pub fn create_apply_patch_tool() -> PyBuiltinTool {
    PyBuiltinTool {
        inner: Arc::new(ApplyPatchTool::new()) as Arc<dyn Tool>,
    }
}

/// Create an `HttpTool` instance (unified HTTP tool supporting request, `fetch_html`, and `api_call` operations)
#[pyfunction]
pub fn create_http_tool() -> PyBuiltinTool {
    PyBuiltinTool {
        inner: Arc::new(HttpTool::new()) as Arc<dyn Tool>,
    }
}

/// Create a `CodeExecutionTool` instance
#[pyfunction]
pub fn create_code_execution_tool() -> PyBuiltinTool {
    PyBuiltinTool {
        inner: Arc::new(CodeExecutionTool) as Arc<dyn Tool>,
    }
}

/// Python wrapper for built-in tools
#[pyclass]
pub struct PyBuiltinTool {
    pub(crate) inner: Arc<dyn Tool>,
}

#[pymethods]
impl PyBuiltinTool {
    pub fn name(&self) -> String {
        self.inner.name().to_string()
    }

    pub fn description(&self) -> String {
        self.inner.description().to_string()
    }

    pub fn parameters(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let params = self.inner.parameters();
        json_to_python_dict(py, &params).map(std::convert::Into::into)
    }

    /// Execute the tool through the tool gate (async work is driven on the
    /// shared runtime).
    ///
    /// `workspace` is the root the filesystem tools may use; without it,
    /// every filesystem call is refused. A refused call returns
    /// `success: False` and the tool does not run.
    #[pyo3(signature = (arguments, workspace=None))]
    pub fn execute(
        &self,
        py: Python<'_>,
        arguments: &Bound<'_, PyAny>,
        workspace: Option<String>,
    ) -> PyResult<Py<PyAny>> {
        let args = depythonize(arguments)
            .map_err(|e| PyRuntimeError::new_err(format!("Invalid tool arguments: {e}")))?;
        let tool = Arc::clone(&self.inner);
        let rt = get_runtime()?;
        let result = py.detach(|| {
            rt.block_on(async move { tool_gate(workspace).run_tool(tool.as_ref(), args).await })
        });

        let dict = PyDict::new(py);
        dict.set_item("success", result.success)?;
        dict.set_item("content", result.content)?;
        if let Some(err) = result.error {
            dict.set_item("error", err)?;
        }
        if let Some(payload) = result.typed_payload {
            let py_payload = crate::conversions::json_to_python_value(py, &payload)?;
            dict.set_item("typed_payload", py_payload)?;
        }
        Ok(dict.into())
    }
}
