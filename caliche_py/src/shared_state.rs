//! Python bindings for `SharedState`

use crate::block_on_detached;
use caliche::orchestration::state::SharedState as RustSharedState;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use serde_json::Value;

#[pyclass]
pub struct SharedState {
    pub(crate) inner: RustSharedState,
}

#[pymethods]
impl SharedState {
    /// Create a new empty shared state
    #[new]
    pub fn new() -> Self {
        Self {
            inner: RustSharedState::new(),
        }
    }

    /// Create shared state from an initial value
    #[staticmethod]
    pub fn from_value(py: Python<'_>, value: Py<PyAny>) -> PyResult<Self> {
        // Convert Python object to JSON string
        let json_str = py
            .import("json")?
            .getattr("dumps")?
            .call1((value,))?
            .extract::<String>()?;

        let json_value: Value = serde_json::from_str(&json_str)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to parse JSON: {e}")))?;

        Ok(Self {
            inner: RustSharedState::from_value(json_value),
        })
    }

    /// Get a value from the state using a JSON pointer path
    ///
    /// # Arguments
    /// * `path` - JSON pointer path (e.g., "/`order_id`" or "/customer/name")
    ///
    /// # Returns
    /// The value at the path, or None if not found
    pub fn get(&self, py: Python<'_>, path: &str) -> PyResult<Py<PyAny>> {
        let state_clone = self.inner.clone();
        let path = path.to_string();
        let result = block_on_detached(py, async move { state_clone.get(&path).await })?
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to get value: {e}")))?;

        match result {
            Some(value) => {
                // Convert JSON Value to Python object
                let json_str = serde_json::to_string(&value)
                    .map_err(|e| PyRuntimeError::new_err(format!("Failed to serialize: {e}")))?;
                let json_module = py.import("json")?;
                let py_value = json_module.getattr("loads")?.call1((json_str,))?.unbind();
                Ok(py_value)
            }
            None => Ok(py.None()),
        }
    }

    /// Set a value at a JSON pointer path
    ///
    /// # Arguments
    /// * `path` - JSON pointer path (e.g., "/`order_id`" or "/customer/name")
    /// * `value` - The value to set (can be any JSON-serializable Python object)
    pub fn set(&self, py: Python<'_>, path: &str, value: Py<PyAny>) -> PyResult<()> {
        // Convert Python object to JSON Value
        let json_str = py
            .import("json")?
            .getattr("dumps")?
            .call1((value,))?
            .extract::<String>()?;
        let json_value: Value = serde_json::from_str(&json_str)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to parse JSON: {e}")))?;

        let state_clone = self.inner.clone();
        let path = path.to_string();
        block_on_detached(py, async move { state_clone.set(&path, json_value).await })?
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to set value: {e}")))?;

        Ok(())
    }

    /// Merge a value into the state (deep merge for objects)
    ///
    /// # Arguments
    /// * `value` - The value to merge (must be a dict/object)
    pub fn merge(&self, py: Python<'_>, value: Py<PyAny>) -> PyResult<()> {
        // Convert Python object to JSON Value
        let json_str = py
            .import("json")?
            .getattr("dumps")?
            .call1((value,))?
            .extract::<String>()?;
        let json_value: Value = serde_json::from_str(&json_str)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to parse JSON: {e}")))?;

        let state_clone = self.inner.clone();
        block_on_detached(py, async move { state_clone.merge(json_value).await })?
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to merge value: {e}")))?;

        Ok(())
    }

    /// Get the entire state as a dictionary
    pub fn get_all(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let state_clone = self.inner.clone();
        let value = block_on_detached(py, async move { state_clone.get_all().await })?;

        // Convert JSON Value to Python dict
        let json_str = serde_json::to_string(&value)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to serialize: {e}")))?;
        let json_module = py.import("json")?;
        let py_dict = json_module.getattr("loads")?.call1((json_str,))?.unbind();
        Ok(py_dict)
    }

    /// Clear the state
    pub fn clear(&self, py: Python<'_>) -> PyResult<()> {
        let state_clone = self.inner.clone();
        block_on_detached(py, async move {
            state_clone.clear().await;
        })?;

        Ok(())
    }
}
