//! Python bindings for durable event stream backends.
//!
//! Exposes Memory and (when built with `full` / redis features) Redis backends
//! so Python can configure durable streams for replay and resume.

use caliche::streams::{DurableStreamBackend, MemoryDurableStreamBackend};
use pyo3::prelude::*;
use std::sync::Arc;

/// Python wrapper for a durable stream backend (memory, Redis, or mmap).
///
/// Use the class methods to create a backend, then pass to
/// :meth:`AgentBuilder.durable_stream_backend` when building an agent.
/// The backend is used when the agent is run behind a harness (e.g. API)
/// to persist events for replay and resume.
#[pyclass(name = "DurableStreamBackend")]
pub struct DurableStreamBackendWrapper {
    pub(crate) inner: Arc<dyn DurableStreamBackend>,
}

#[pymethods]
impl DurableStreamBackendWrapper {
    /// Create an in-memory durable stream backend with a fixed capacity.
    #[staticmethod]
    fn memory(capacity: usize) -> Self {
        Self {
            inner: Arc::new(MemoryDurableStreamBackend::new(capacity)),
        }
    }

    /// Create a Redis/Valkey Streams backend for durable event storage.
    ///
    /// Requires the bindings crate to be built with Redis durable-stream support
    /// (`--features durable-streams-redis`, included by `--features full`).
    #[staticmethod]
    fn redis(redis_url: &str) -> PyResult<Self> {
        #[cfg(feature = "durable-streams-redis")]
        {
            use caliche::streams::RedisDurableStreamBackend;
            let backend = RedisDurableStreamBackend::new(redis_url)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{e}")))?;
            Ok(Self {
                inner: Arc::new(backend),
            })
        }
        #[cfg(not(feature = "durable-streams-redis"))]
        {
            let _ = redis_url;
            Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Redis durable streams require building caliche_py with --features durable-streams-redis",
            ))
        }
    }
}
