//! Python bindings for Caliche RAG Tool

use caliche::agent::tools::RagTool as RustRagTool;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use std::sync::Arc;

use crate::agent::Agent as PyAgent;
use crate::get_runtime;

/// Python wrapper for `RagTool`
#[pyclass]
pub struct PyRagTool {
    pub(crate) inner: Arc<RustRagTool>,
}

#[pymethods]
impl PyRagTool {
    /// Create a new RAG tool with default in-memory backend
    #[new]
    pub fn new(agent: &PyAgent) -> PyResult<Self> {
        let agent_arc = agent
            .inner
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("Agent not initialized"))?;

        let rt = get_runtime()?;
        let embedding_provider = rt
            .block_on(async {
                let agent_guard = agent_arc.lock().await;
                agent_guard.embedding_provider().ok_or_else(|| {
                    anyhow::anyhow!("Agent does not have an embedding provider configured")
                })
            })
            .map_err(|e| {
                PyRuntimeError::new_err(format!("Failed to get embedding provider: {e}"))
            })?;

        let rag_tool = RustRagTool::new(embedding_provider, None::<String>);

        Ok(Self {
            inner: Arc::new(rag_tool),
        })
    }

    /// Get the underlying RAG service.
    ///
    /// This allows sharing the same backend between `RagTool` and `RagService` instances.
    pub fn service(&self) -> PyResult<crate::rag::PyRagService> {
        let service = self.inner.service();
        crate::rag::PyRagService::from_service(service)
    }
}
