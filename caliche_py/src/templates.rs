//! Python bindings for Caliche Templates

use caliche::templates::{AgentTemplate, TemplateRegistry};
use pyo3::prelude::*;
use std::sync::Arc;

#[cfg(feature = "s3")]
use super::storage::PyStorageConfig;
use crate::get_runtime;

/// Python wrapper for `TemplateRegistry`
#[pyclass]
pub struct PyTemplateRegistry {
    inner: Arc<tokio::sync::Mutex<TemplateRegistry>>,
}

#[pymethods]
impl PyTemplateRegistry {
    /// Create a new `TemplateRegistry`
    #[new]
    fn new() -> Self {
        Self {
            inner: Arc::new(tokio::sync::Mutex::new(TemplateRegistry::new())),
        }
    }

    /// Create a `TemplateRegistry` with storage config
    #[cfg(feature = "s3")]
    #[staticmethod]
    fn with_storage(config: &PyStorageConfig) -> PyResult<Self> {
        let config_clone = config.inner().clone();
        let rt = get_runtime()?;

        let registry = rt
            .block_on(async { TemplateRegistry::with_storage(config_clone).await })
            .map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                    "Failed to create registry with storage: {e}"
                ))
            })?;

        Ok(Self {
            inner: Arc::new(tokio::sync::Mutex::new(registry)),
        })
    }

    /// Add a template directory to scan
    fn add_template_dir(&self, dir: String) -> PyResult<()> {
        let registry_clone = self.inner.clone();
        let rt = get_runtime()?;

        rt.block_on(async {
            let mut reg = registry_clone.lock().await;
            reg.add_template_dir(dir);
        });

        Ok(())
    }

    /// Load templates from configured sources
    fn load_templates(&self) -> PyResult<usize> {
        let registry_clone = self.inner.clone();
        let rt = get_runtime()?;

        rt.block_on(async {
            let reg = registry_clone.lock().await;
            reg.load_templates().await
        })
        .map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Failed to load templates: {e}"
            ))
        })
    }

    /// Get a template by ID
    fn get(&self, id: &str) -> PyResult<Option<PyAgentTemplate>> {
        let registry_clone = self.inner.clone();
        let rt = get_runtime()?;

        let template = rt.block_on(async {
            let reg = registry_clone.lock().await;
            reg.get(id)
        });

        Ok(template.map(|t| PyAgentTemplate {
            inner: t.as_ref().clone(),
        }))
    }

    /// List all templates
    fn list(&self) -> PyResult<Vec<PyAgentTemplate>> {
        let registry_clone = self.inner.clone();
        let rt = get_runtime()?;

        let templates = rt.block_on(async {
            let reg = registry_clone.lock().await;
            reg.list()
        });

        Ok(templates
            .into_iter()
            .map(|t| PyAgentTemplate {
                inner: t.as_ref().clone(),
            })
            .collect())
    }
}

/// Python wrapper for `AgentTemplate`
#[pyclass]
pub struct PyAgentTemplate {
    inner: AgentTemplate,
}

#[pymethods]
impl PyAgentTemplate {
    /// Get template name
    fn name(&self) -> String {
        self.inner.name.clone()
    }

    /// Get template version
    fn version(&self) -> String {
        self.inner.version.clone()
    }

    /// Get template description
    fn description(&self) -> String {
        self.inner.description.clone()
    }

    /// Instantiate an agent from this template
    fn instantiate(&self) -> PyResult<crate::builder::AgentBuilder> {
        use crate::builder::AgentBuilder as PyAgentBuilder;
        use caliche::AgentBuilder as RustAgentBuilder;

        let rt = get_runtime()?;
        let rust_builder: RustAgentBuilder = rt
            .block_on(async { self.inner.instantiate().await })
            .map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                    "Failed to instantiate template: {e}"
                ))
            })?;

        Ok(PyAgentBuilder::from_rust_builder(rust_builder))
    }
}
