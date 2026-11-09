//! Python bindings for Caliche Document Loaders

use crate::get_runtime;
use caliche::loaders::{
    CsvLoader, Document, DocumentLoader, HtmlLoader, JsonLoader, MarkdownLoader, TextSplitter,
};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

/// Python wrapper for Document
#[pyclass]
pub struct PyDocument {
    inner: Document,
}

#[pymethods]
impl PyDocument {
    #[new]
    pub fn new(content: String) -> Self {
        Self {
            inner: Document::new(content),
        }
    }

    pub fn with_metadata(&mut self, key: String, value: String) {
        self.inner = self.inner.clone().with_metadata(key, value);
    }

    pub fn content(&self) -> String {
        self.inner.content.clone()
    }

    pub fn metadata(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let dict = PyDict::new(py);
        for (k, v) in &self.inner.metadata {
            dict.set_item(k, v)?;
        }
        Ok(dict.into())
    }
}

/// Python wrapper for `TextSplitter`
#[pyclass]
pub struct PyTextSplitter {
    inner: TextSplitter,
}

#[pymethods]
impl PyTextSplitter {
    #[new]
    pub fn new(chunk_size: usize, chunk_overlap: usize) -> Self {
        Self {
            inner: TextSplitter::new(chunk_size, chunk_overlap),
        }
    }

    /// Split text into chunks
    pub fn split_text(&self, text: &str) -> Vec<String> {
        self.inner.split_text(text)
    }

    /// Split documents into chunks
    pub fn split_documents(&self, documents: &Bound<'_, PyList>) -> PyResult<Vec<PyDocument>> {
        let mut rust_docs = Vec::new();
        for item in documents.iter() {
            let py_doc = item.extract::<PyRef<'_, PyDocument>>()?;
            rust_docs.push(py_doc.inner.clone());
        }

        let split_docs = self.inner.split_documents(&rust_docs);
        Ok(split_docs
            .into_iter()
            .map(|doc| PyDocument { inner: doc })
            .collect())
    }
}

/// Python wrapper for `CsvLoader`
#[pyclass]
pub struct PyCsvLoader {
    inner: CsvLoader,
}

#[pymethods]
impl PyCsvLoader {
    #[new]
    pub fn new(path: String) -> Self {
        Self {
            inner: CsvLoader::new(path),
        }
    }

    pub fn with_content_column(&mut self, column: String) {
        self.inner = self.inner.clone().with_content_column(column);
    }

    pub fn with_metadata_columns(&mut self, columns: &Bound<'_, PyList>) {
        let mut cols = Vec::new();
        for item in columns.iter() {
            if let Ok(s) = item.extract::<String>() {
                cols.push(s);
            }
        }
        self.inner = self.inner.clone().with_metadata_columns(&cols);
    }

    /// Load documents from CSV
    pub fn load(&self) -> PyResult<Vec<PyDocument>> {
        let loader_clone = self.inner.clone();
        let rt = get_runtime()?;

        let result = rt.block_on(async { loader_clone.load().await });

        match result {
            Ok(docs) => Ok(docs.into_iter().map(|d| PyDocument { inner: d }).collect()),
            Err(e) => Err(PyRuntimeError::new_err(format!("Failed to load CSV: {e}"))),
        }
    }

    pub fn name(&self) -> String {
        self.inner.name().to_string()
    }
}

/// Python wrapper for `JsonLoader`
#[pyclass]
pub struct PyJsonLoader {
    inner: JsonLoader,
}

#[pymethods]
impl PyJsonLoader {
    #[new]
    pub fn new(path: String) -> Self {
        Self {
            inner: JsonLoader::new(path),
        }
    }

    pub fn with_content_key(&mut self, key: String) {
        self.inner = self.inner.clone().with_content_key(key);
    }

    pub fn with_jsonl(&mut self, is_jsonl: bool) {
        self.inner = self.inner.clone().with_jsonl(is_jsonl);
    }

    /// Load documents from JSON/JSONL
    pub fn load(&self) -> PyResult<Vec<PyDocument>> {
        let loader_clone = self.inner.clone();
        let rt = get_runtime()?;

        let result = rt.block_on(async { loader_clone.load().await });

        match result {
            Ok(docs) => Ok(docs.into_iter().map(|d| PyDocument { inner: d }).collect()),
            Err(e) => Err(PyRuntimeError::new_err(format!("Failed to load JSON: {e}"))),
        }
    }

    pub fn name(&self) -> String {
        self.inner.name().to_string()
    }
}

/// Python wrapper for `MarkdownLoader`
#[pyclass]
pub struct PyMarkdownLoader {
    inner: MarkdownLoader,
}

#[pymethods]
impl PyMarkdownLoader {
    #[new]
    pub fn new(path: String) -> Self {
        Self {
            inner: MarkdownLoader::new(path),
        }
    }

    pub fn with_frontmatter(&mut self, extract: bool) {
        self.inner = self.inner.clone().with_frontmatter(extract);
    }

    /// Load documents from Markdown
    pub fn load(&self) -> PyResult<Vec<PyDocument>> {
        let loader_clone = self.inner.clone();
        let rt = get_runtime()?;

        let result = rt.block_on(async { loader_clone.load().await });

        match result {
            Ok(docs) => Ok(docs.into_iter().map(|d| PyDocument { inner: d }).collect()),
            Err(e) => Err(PyRuntimeError::new_err(format!(
                "Failed to load Markdown: {e}"
            ))),
        }
    }

    pub fn name(&self) -> String {
        self.inner.name().to_string()
    }
}

/// Python wrapper for `HtmlLoader`
#[pyclass]
pub struct PyHtmlLoader {
    inner: HtmlLoader,
}

#[pymethods]
impl PyHtmlLoader {
    #[new]
    pub fn new(path: String) -> Self {
        Self {
            inner: HtmlLoader::new(path),
        }
    }

    pub fn with_remove_scripts(&mut self, remove: bool) {
        self.inner = self.inner.clone().with_remove_scripts(remove);
    }

    pub fn with_remove_styles(&mut self, remove: bool) {
        self.inner = self.inner.clone().with_remove_styles(remove);
    }

    /// Load documents from HTML
    pub fn load(&self) -> PyResult<Vec<PyDocument>> {
        let loader_clone = self.inner.clone();
        let rt = get_runtime()?;

        let result = rt.block_on(async { loader_clone.load().await });

        match result {
            Ok(docs) => Ok(docs.into_iter().map(|d| PyDocument { inner: d }).collect()),
            Err(e) => Err(PyRuntimeError::new_err(format!("Failed to load HTML: {e}"))),
        }
    }

    pub fn name(&self) -> String {
        self.inner.name().to_string()
    }
}
