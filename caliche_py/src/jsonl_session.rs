//! Python bindings for pi-inspired JSONL session trees.

use crate::block_on_detached;
use caliche::agent::session::{
    JsonlSessionEntry, JsonlSessionHeader, JsonlSessionStore, JsonlSessionTree, new_entry_id,
    parse_jsonl_session,
};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::PyList;
use pythonize::{depythonize, pythonize};

/// Session header (first JSONL line).
#[pyclass(name = "JsonlSessionHeader")]
#[derive(Clone)]
pub struct PyJsonlSessionHeader {
    pub(crate) inner: JsonlSessionHeader,
}

#[pymethods]
impl PyJsonlSessionHeader {
    #[new]
    #[pyo3(signature = (id, cwd=None))]
    fn new(id: String, cwd: Option<String>) -> Self {
        Self {
            inner: JsonlSessionHeader::new(id, cwd),
        }
    }

    #[getter]
    fn id(&self) -> String {
        self.inner.id.clone()
    }

    #[getter]
    const fn version(&self) -> u32 {
        self.inner.version
    }

    #[getter]
    fn cwd(&self) -> Option<String> {
        self.inner.cwd.clone()
    }

    #[getter]
    fn parent_session(&self) -> Option<String> {
        self.inner.parent_session.clone()
    }

    fn set_parent_session(&mut self, parent_session: Option<String>) {
        self.inner.parent_session = parent_session;
    }

    fn to_dict(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        Ok(pythonize(py, &self.inner)?.unbind())
    }
}

/// One tree-linked session entry.
#[pyclass(name = "JsonlSessionEntry")]
#[derive(Clone)]
pub struct PyJsonlSessionEntry {
    pub(crate) inner: JsonlSessionEntry,
}

#[pymethods]
impl PyJsonlSessionEntry {
    #[staticmethod]
    #[pyo3(signature = (id, message, parent_id=None))]
    fn message(
        id: String,
        message: &Bound<'_, PyAny>,
        parent_id: Option<String>,
    ) -> PyResult<Self> {
        let message_value = depythonize(message)
            .map_err(|e| PyRuntimeError::new_err(format!("Invalid message payload: {e}")))?;
        Ok(Self {
            inner: JsonlSessionEntry::message(id, parent_id, message_value),
        })
    }

    #[staticmethod]
    fn branch_summary(
        id: String,
        from_id: String,
        summary: String,
        parent_id: Option<String>,
    ) -> Self {
        Self {
            inner: JsonlSessionEntry::BranchSummary {
                id,
                parent_id,
                timestamp: chrono::Utc::now(),
                summary,
                from_id,
            },
        }
    }

    #[staticmethod]
    #[pyo3(signature = (id, summary, parent_id=None, tokens_before=None))]
    fn compaction_summary(
        id: String,
        summary: String,
        parent_id: Option<String>,
        tokens_before: Option<usize>,
    ) -> Self {
        Self {
            inner: JsonlSessionEntry::CompactionSummary {
                id,
                parent_id,
                timestamp: chrono::Utc::now(),
                summary,
                tokens_before,
            },
        }
    }

    #[staticmethod]
    #[pyo3(signature = (id, provider, model_id, parent_id=None))]
    fn model_change(
        id: String,
        provider: String,
        model_id: String,
        parent_id: Option<String>,
    ) -> Self {
        Self {
            inner: JsonlSessionEntry::ModelChange {
                id,
                parent_id,
                timestamp: chrono::Utc::now(),
                provider,
                model_id,
            },
        }
    }

    #[getter]
    fn id(&self) -> String {
        self.inner.id().to_string()
    }

    #[getter]
    fn parent_id(&self) -> Option<String> {
        self.inner.parent_id().map(str::to_string)
    }

    fn to_dict(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        Ok(pythonize(py, &self.inner)?.unbind())
    }
}

/// In-memory JSONL session tree.
#[pyclass(name = "JsonlSessionTree")]
pub struct PyJsonlSessionTree {
    inner: JsonlSessionTree,
}

#[pymethods]
impl PyJsonlSessionTree {
    #[new]
    fn new(header: &PyJsonlSessionHeader) -> Self {
        Self {
            inner: JsonlSessionTree::new(header.inner.clone()),
        }
    }

    #[getter]
    fn header(&self) -> PyJsonlSessionHeader {
        PyJsonlSessionHeader {
            inner: self.inner.header.clone(),
        }
    }

    fn append(&mut self, entry: &PyJsonlSessionEntry) -> PyResult<()> {
        self.inner
            .append(entry.inner.clone())
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    fn leaves(&self) -> Vec<String> {
        self.inner
            .leaves()
            .into_iter()
            .map(str::to_string)
            .collect()
    }

    fn branch_path(&self, py: Python<'_>, leaf_id: &str) -> PyResult<Py<PyAny>> {
        let path = self
            .inner
            .branch_path(leaf_id)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        let list = PyList::empty(py);
        for entry in path {
            let dict = pythonize(py, entry)?;
            list.append(dict)?;
        }
        Ok(list.into())
    }

    fn restore_branch_messages(&self, py: Python<'_>, leaf_id: &str) -> PyResult<Py<PyAny>> {
        let messages = self
            .inner
            .restore_branch_messages(leaf_id)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        let list = PyList::empty(py);
        for message in messages {
            list.append(pythonize(py, &message)?)?;
        }
        Ok(list.into())
    }

    fn branch_from(
        &mut self,
        from_id: &str,
        branch_entry_id: String,
        summary: String,
    ) -> PyResult<String> {
        self.inner
            .branch_from(from_id, branch_entry_id, summary)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    const fn entry_count(&self) -> usize {
        self.inner.entries.len()
    }

    /// Return a copy of all entries (root-order append sequence).
    fn entries(&self) -> Vec<PyJsonlSessionEntry> {
        self.inner
            .entries
            .iter()
            .cloned()
            .map(|inner| PyJsonlSessionEntry { inner })
            .collect()
    }
}

/// Append-only JSONL session store on disk.
#[pyclass(name = "JsonlSessionStore")]
pub struct PyJsonlSessionStore {
    inner: JsonlSessionStore,
}

#[pymethods]
impl PyJsonlSessionStore {
    #[new]
    fn new(path: String) -> Self {
        Self {
            inner: JsonlSessionStore::new(path),
        }
    }

    #[getter]
    fn path(&self) -> String {
        self.inner.path().display().to_string()
    }

    fn create(&self, py: Python<'_>, header: &PyJsonlSessionHeader) -> PyResult<()> {
        block_on_detached(py, self.inner.create(&header.inner))?
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    fn append_entry(&self, py: Python<'_>, entry: &PyJsonlSessionEntry) -> PyResult<()> {
        block_on_detached(py, self.inner.append_entry(&entry.inner))?
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    fn load(&self, py: Python<'_>) -> PyResult<PyJsonlSessionTree> {
        let tree = block_on_detached(py, self.inner.load())?
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(PyJsonlSessionTree { inner: tree })
    }
}

/// Parse a full JSONL session document.
#[pyfunction]
#[pyo3(name = "parse_jsonl_session")]
pub fn py_parse_jsonl_session(text: &str) -> PyResult<PyJsonlSessionTree> {
    let tree = parse_jsonl_session(text).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    Ok(PyJsonlSessionTree { inner: tree })
}

/// Generate a short hex entry id.
#[pyfunction]
#[pyo3(name = "new_entry_id")]
pub fn py_new_entry_id() -> String {
    new_entry_id()
}
