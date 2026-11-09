//! Python bindings for Prompt Template Storage

use super::storage::PyStorageConfig;
use crate::get_runtime;
use caliche::prompts::{PromptTemplate, PromptTemplateStore};
use pyo3::prelude::*;
use std::sync::Arc;

/// Python wrapper for `PromptTemplateStore`
#[pyclass]
pub struct PyPromptTemplateStore {
    inner: Arc<tokio::sync::Mutex<PromptTemplateStore>>,
}

#[pymethods]
impl PyPromptTemplateStore {
    /// Create a `PromptTemplateStore` with storage config
    #[staticmethod]
    fn new(config: &PyStorageConfig) -> PyResult<Self> {
        let config_clone = config.inner().clone();
        let rt = get_runtime()?;

        let store = rt
            .block_on(async { PromptTemplateStore::new(config_clone).await })
            .map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                    "Failed to create prompt store: {e}"
                ))
            })?;

        Ok(Self {
            inner: Arc::new(tokio::sync::Mutex::new(store)),
        })
    }

    /// Save a prompt template
    fn save(&self, name: String, template_str: String) -> PyResult<()> {
        let template = PromptTemplate::new(&template_str);
        let store_clone = self.inner.clone();
        let rt = get_runtime()?;

        rt.block_on(async {
            let store_guard = store_clone.lock().await;
            store_guard.save(&name, &template).await
        })
        .map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Failed to save prompt: {e}"))
        })
    }

    /// Load a prompt template
    fn load(&self, name: String) -> PyResult<String> {
        let store_clone = self.inner.clone();
        let rt = get_runtime()?;

        let template = rt
            .block_on(async {
                let store_guard = store_clone.lock().await;
                store_guard.load(&name).await
            })
            .map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                    "Failed to load prompt: {e}"
                ))
            })?;

        Ok(template.template().to_string())
    }

    /// List all prompt template names
    fn list(&self) -> PyResult<Vec<String>> {
        let store_clone = self.inner.clone();
        let rt = get_runtime()?;

        rt.block_on(async {
            let store_guard = store_clone.lock().await;
            store_guard.list().await
        })
        .map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Failed to list prompts: {e}"
            ))
        })
    }

    /// Delete a prompt template
    fn delete(&self, name: String) -> PyResult<()> {
        let store_clone = self.inner.clone();
        let rt = get_runtime()?;

        rt.block_on(async {
            let store_guard = store_clone.lock().await;
            store_guard.delete(&name).await
        })
        .map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Failed to delete prompt: {e}"
            ))
        })
    }

    /// Check if a prompt template exists
    fn exists(&self, name: String) -> PyResult<bool> {
        let store_clone = self.inner.clone();
        let rt = get_runtime()?;

        rt.block_on(async {
            let store_guard = store_clone.lock().await;
            store_guard.exists(&name).await
        })
        .map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Failed to check prompt existence: {e}"
            ))
        })
    }
}
