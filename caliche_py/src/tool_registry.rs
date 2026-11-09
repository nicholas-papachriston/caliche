//! Python bindings for deferred/active tool registry controls.

use crate::builtin_tools::PyBuiltinTool;
use crate::get_runtime;
use crate::tool::PyTool;
use caliche::agent::tools::{Tool, ToolRegistry};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::{PyAnyMethods, PyDict, PyList};
use std::sync::Arc;

/// Thin handle over the shared Rust [`ToolRegistry`].
#[pyclass(name = "ToolRegistry")]
#[derive(Clone)]
pub struct PyToolRegistry {
    pub(crate) inner: Arc<ToolRegistry>,
}

impl PyToolRegistry {
    pub(crate) const fn new(inner: Arc<ToolRegistry>) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyToolRegistry {
    /// Create an empty registry (useful for deferred/active catalog tests).
    #[new]
    fn py_new() -> Self {
        Self {
            inner: Arc::new(ToolRegistry::new()),
        }
    }

    /// Register a tool without exposing it until [`Self::activate`].
    pub fn register_deferred(&self, tool_obj: Py<PyAny>) -> PyResult<()> {
        let tool = extract_tool(tool_obj)?;
        self.inner.register_deferred(tool);
        Ok(())
    }

    /// Register a tool and keep current active-set semantics.
    pub fn register(&self, tool_obj: Py<PyAny>) -> PyResult<()> {
        let tool = extract_tool(tool_obj)?;
        self.inner.register(tool);
        Ok(())
    }

    /// Activate named tools (creating an active set if unrestricted).
    pub fn activate(&self, names: Vec<String>) {
        self.inner.activate(names);
    }

    /// Deactivate named tools.
    pub fn deactivate(&self, names: Vec<String>) {
        self.inner.deactivate(names);
    }

    /// Replace the active set. Pass `None` to activate all registered tools.
    #[pyo3(signature = (names=None))]
    pub fn set_active_tools(&self, names: Option<Vec<String>>) {
        match names {
            Some(names) => self.inner.set_active_tools(Some(names)),
            None => self.inner.activate_all(),
        }
    }

    /// Clear the active filter so every registered tool is active.
    pub fn activate_all(&self) {
        self.inner.activate_all();
    }

    pub fn is_active(&self, name: &str) -> bool {
        self.inner.is_active(name)
    }

    pub fn active_tool_names(&self) -> Vec<String> {
        self.inner.active_tool_names()
    }

    /// Execute a tool by name (honors active-set filtering).
    pub fn execute(
        &self,
        py: Python<'_>,
        name: &str,
        arguments: &Bound<'_, PyAny>,
    ) -> PyResult<Py<PyAny>> {
        let args = pythonize::depythonize(arguments)
            .map_err(|e| PyRuntimeError::new_err(format!("Invalid tool arguments: {e}")))?;
        let rt = get_runtime()?;
        let registry = Arc::clone(&self.inner);
        let name = name.to_string();
        let result = py.detach(|| rt.block_on(async move { registry.execute(&name, args).await }));

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

    /// Schemas currently exposed to the model (active tools only).
    pub fn get_tool_schemas(&self, py: Python<'_>) -> PyResult<Vec<Py<PyAny>>> {
        let schemas = self.inner.get_tool_schemas();
        let mut out = Vec::with_capacity(schemas.len());
        for schema in schemas.iter() {
            let dict = PyDict::new(py);
            dict.set_item("name", schema.name.as_ref())?;
            dict.set_item("description", schema.description.as_ref())?;
            let params = serde_json::json!({
                "type": schema.parameters.schema_type.as_ref(),
                "properties": schema.parameters.properties,
                "required": schema.parameters.required.iter().map(std::convert::AsRef::as_ref).collect::<Vec<_>>(),
            });
            let py_params = crate::conversions::json_to_python_value(py, &params)?;
            dict.set_item("parameters", py_params)?;
            out.push(dict.into());
        }
        Ok(out)
    }

    pub fn list(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let items = self.inner.list();
        let list = PyList::empty(py);
        for (name, description, parameters) in items {
            let dict = PyDict::new(py);
            dict.set_item("name", name)?;
            dict.set_item("description", description)?;
            let py_params = crate::conversions::json_to_python_value(py, &parameters)?;
            dict.set_item("parameters", py_params)?;
            list.append(dict)?;
        }
        Ok(list.into())
    }
}

fn extract_tool(tool_obj: Py<PyAny>) -> PyResult<Arc<dyn Tool>> {
    Python::attach(|py| {
        let bound = tool_obj.bind(py);
        if let Ok(builtin) = bound.extract::<PyRef<'_, PyBuiltinTool>>() {
            return Ok(Arc::clone(&builtin.inner));
        }
        Ok(Arc::new(PyTool::new(tool_obj)) as Arc<dyn Tool>)
    })
}
