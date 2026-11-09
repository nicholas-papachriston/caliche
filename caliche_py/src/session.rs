//! Python bindings for `AgentSession`

use crate::conversions::{py_message_to_rust, rust_message_to_py};
use crate::get_runtime;
use crate::shared_state::SharedState as PySharedState;
use caliche::agent::session::AgentSession as RustAgentSession;
use caliche::messages::Message;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

/// Python wrapper for `AgentSession`
#[pyclass]
pub struct PyAgentSession {
    inner: RustAgentSession,
}

#[pymethods]
impl PyAgentSession {
    #[new]
    #[pyo3(signature = (id, history=None, shared_state=None))]
    fn new(
        id: String,
        history: Option<Vec<Py<PyAny>>>,
        shared_state: Option<PyRef<PySharedState>>,
    ) -> PyResult<Self> {
        let rust_session = if let Some(shared_state_ref) = shared_state {
            RustAgentSession::with_shared_state(id, shared_state_ref.inner.clone())
        } else if let Some(hist) = history {
            // Convert Python messages to Rust messages
            let rust_messages: Result<Vec<Message>, String> = Python::attach(|py| {
                hist.into_iter()
                    .map(|py_msg| {
                        let bound = py_msg.bind(py);
                        py_message_to_rust(bound)
                            .map_err(|e| format!("Failed to convert message: {e}"))
                    })
                    .collect()
            });
            RustAgentSession::with_history(id, rust_messages.map_err(PyRuntimeError::new_err)?)
        } else {
            RustAgentSession::new(id)
        };

        Ok(Self {
            inner: rust_session,
        })
    }

    #[staticmethod]
    fn with_history(id: String, history: Vec<Py<PyAny>>) -> PyResult<Self> {
        let rust_messages: Result<Vec<Message>, String> = Python::attach(|py| {
            history
                .into_iter()
                .map(|py_msg| {
                    let bound = py_msg.bind(py);
                    py_message_to_rust(bound).map_err(|e| format!("Failed to convert message: {e}"))
                })
                .collect()
        });
        Ok(Self {
            inner: RustAgentSession::with_history(
                id,
                rust_messages.map_err(PyRuntimeError::new_err)?,
            ),
        })
    }

    #[getter]
    fn id(&self) -> String {
        self.inner.id.clone()
    }

    /// Fork this session into a new isolated session (async to fork memory).
    fn fork(&self, new_id: String) -> PyResult<Self> {
        let rt = get_runtime()?;
        let forked = rt
            .block_on(self.inner.fork(new_id))
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to fork session: {e}")))?;
        Ok(Self { inner: forked })
    }

    /// Merge another session's conversation history and memory into this session.
    fn merge(&mut self, other: &Self) -> PyResult<()> {
        let rt = get_runtime()?;
        rt.block_on(self.inner.merge(&other.inner))
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to merge session: {e}")))
    }

    /// Get the current conversation history
    fn get_history(&self, py: Python) -> PyResult<Vec<Py<PyAny>>> {
        let rust_messages = self
            .inner
            .get_history()
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to get history: {e}")))?;

        // Convert Rust messages to Python messages
        rust_messages
            .into_iter()
            .map(|msg| rust_message_to_py(py, &msg))
            .collect()
    }

    /// Add a message to the conversation history
    fn add_message(&self, message: &Bound<'_, PyAny>) -> PyResult<()> {
        let rust_message = py_message_to_rust(message)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to convert message: {e}")))?;

        self.inner
            .add_message(rust_message)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to add message: {e}")))
    }

    /// Clear the conversation history
    fn clear_history(&self) -> PyResult<()> {
        self.inner
            .clear_history()
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to clear history: {e}")))
    }

    /// Get the shared state
    fn shared_state(&self, py: Python) -> PyResult<Py<PyAny>> {
        let py_shared_state = PySharedState {
            inner: self.inner.shared_state.clone(),
        };
        Ok(Py::new(py, py_shared_state)?.into_any())
    }
}

impl From<RustAgentSession> for PyAgentSession {
    fn from(session: RustAgentSession) -> Self {
        Self { inner: session }
    }
}
