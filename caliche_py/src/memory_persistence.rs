//! Python bindings for `MemoryPersistence` trait

use crate::python_async::call_python_maybe_async;
use async_trait::async_trait;
use caliche::agent::memory::{MemoryPersistence, MemorySnapshot};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use pythonize::pythonize;
use serde_json::Value;

/// Python wrapper for Rust `MemoryPersistence` trait
///
/// This allows Python objects to implement the `MemoryPersistence` interface
/// by calling Python methods from Rust.
#[derive(Debug)]
pub struct PyMemoryPersistence {
    persistence_obj: Py<PyAny>,
}

impl PyMemoryPersistence {
    pub const fn new(persistence_obj: Py<PyAny>) -> Self {
        Self { persistence_obj }
    }
}

#[async_trait]
impl MemoryPersistence for PyMemoryPersistence {
    async fn save_snapshot(
        &self,
        conversation_id: &str,
        snapshot: &MemorySnapshot,
    ) -> anyhow::Result<()> {
        // Convert snapshot to Python dict using pythonize for efficient serialization
        let snapshot_dict = Python::attach(|py| -> PyResult<Py<PyAny>> {
            let dict = PyDict::new(py);

            // Use pythonize to directly convert messages (skip JSON string intermediate step)
            let messages_py = pythonize(py, &snapshot.messages).map_err(|e| {
                PyRuntimeError::new_err(format!("Failed to serialize messages: {e}"))
            })?;
            dict.set_item("messages", messages_py)?;

            // Use pythonize for metadata as well
            let metadata_py = pythonize(py, &snapshot.metadata).map_err(|e| {
                PyRuntimeError::new_err(format!("Failed to serialize metadata: {e}"))
            })?;
            dict.set_item("metadata", metadata_py)?;

            Ok(dict.into())
        })?;

        // Call Python save_snapshot method
        let py_result = Python::attach(|py| -> PyResult<Py<PyAny>> {
            let method = self.persistence_obj.getattr(py, "save_snapshot")?;
            method.call1(py, (conversation_id, snapshot_dict))
        })?;

        // Handle both sync and async Python functions using shared helper
        call_python_maybe_async(py_result).await?;

        Ok(())
    }

    async fn load_snapshot(&self, conversation_id: &str) -> anyhow::Result<Option<MemorySnapshot>> {
        // Call Python load_snapshot method
        let result = Python::attach(|py| -> PyResult<Py<PyAny>> {
            let method = self.persistence_obj.getattr(py, "load_snapshot")?;
            method.call1(py, (conversation_id,))
        });

        let py_result = match result {
            Ok(r) => r,
            Err(e) => return Err(anyhow::anyhow!("Python load_snapshot error: {e}")),
        };

        // Handle both sync and async Python functions using shared helper
        let snapshot_obj = call_python_maybe_async(py_result).await?;

        // Convert Python result to MemorySnapshot using pythonize for efficient deserialization
        Python::attach(|py| -> PyResult<Option<MemorySnapshot>> {
            let bound_snapshot = snapshot_obj.bind(py);
            if bound_snapshot.is_none() {
                return Ok(None);
            }

            let dict = bound_snapshot.cast::<PyDict>()?;

            // Extract messages using pythonize (more efficient than manual conversion)
            let messages: Vec<caliche::messages::Message> = match dict.get_item("messages") {
                Ok(Some(msgs_val)) => {
                    // Use pythonize to convert Python object to Rust type directly
                    pythonize::depythonize(msgs_val.as_ref()).map_err(|e| {
                        PyRuntimeError::new_err(format!("Failed to deserialize messages: {e}"))
                    })?
                }
                _ => Vec::new(),
            };

            // Extract metadata using pythonize
            let metadata: Value = match dict.get_item("metadata") {
                Ok(Some(meta_val)) => pythonize::depythonize(meta_val.as_ref())
                    .unwrap_or_else(|_| Value::Object(serde_json::Map::new())),
                _ => Value::Object(serde_json::Map::new()),
            };

            Ok(Some(MemorySnapshot { messages, metadata }))
        })
        .map_err(|e| anyhow::anyhow!("Failed to convert snapshot: {e}"))
    }

    async fn clear_snapshot(&self, conversation_id: &str) -> anyhow::Result<()> {
        // Call Python clear_snapshot method
        let result = Python::attach(|py| -> PyResult<Py<PyAny>> {
            let method = self.persistence_obj.getattr(py, "clear_snapshot")?;
            method.call1(py, (conversation_id,))
        });

        match result {
            Ok(r) => {
                // Handle both sync and async Python functions using shared helper
                call_python_maybe_async(r).await?;
                Ok(())
            }
            Err(e) => Err(anyhow::anyhow!("Python clear_snapshot error: {e}")),
        }
    }
}
