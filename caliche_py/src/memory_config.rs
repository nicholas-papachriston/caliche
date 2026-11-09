//! Python bindings for memory configuration

use caliche::agent::memory::{
    BufferMemoryConfig, EntityMemoryConfig, SummaryMemoryConfig, VectorMemoryConfig,
};
use pyo3::prelude::*;

#[pyclass]
#[derive(Clone)]
pub struct PyBufferMemoryConfig {
    inner: BufferMemoryConfig,
}

#[pymethods]
impl PyBufferMemoryConfig {
    #[new]
    #[pyo3(signature = (max_tokens=4096, min_messages=4, preserve_system=true, summary_placeholder=true))]
    pub const fn new(
        max_tokens: usize,
        min_messages: usize,
        preserve_system: bool,
        summary_placeholder: bool,
    ) -> Self {
        Self {
            inner: BufferMemoryConfig {
                max_tokens,
                min_messages,
                preserve_system,
                summary_placeholder,
            },
        }
    }

    #[getter]
    pub const fn max_tokens(&self) -> usize {
        self.inner.max_tokens
    }

    #[getter]
    pub const fn min_messages(&self) -> usize {
        self.inner.min_messages
    }

    #[getter]
    pub const fn preserve_system(&self) -> bool {
        self.inner.preserve_system
    }

    #[getter]
    pub const fn summary_placeholder(&self) -> bool {
        self.inner.summary_placeholder
    }
}

impl From<PyBufferMemoryConfig> for BufferMemoryConfig {
    fn from(py_config: PyBufferMemoryConfig) -> Self {
        py_config.inner
    }
}

#[pyclass]
#[derive(Clone)]
pub struct PySummaryMemoryConfig {
    inner: SummaryMemoryConfig,
}

#[pymethods]
impl PySummaryMemoryConfig {
    #[new]
    #[pyo3(signature = (max_tokens=8192, summarize_threshold_pct=0.8, chunk_size=12, max_summaries=8))]
    pub const fn new(
        max_tokens: usize,
        summarize_threshold_pct: f32,
        chunk_size: usize,
        max_summaries: usize,
    ) -> Self {
        Self {
            inner: SummaryMemoryConfig {
                max_tokens,
                max_summary_tokens: None,
                summarize_threshold_pct,
                chunk_size,
                max_summaries,
            },
        }
    }

    #[getter]
    pub const fn max_tokens(&self) -> usize {
        self.inner.max_tokens
    }

    #[getter]
    pub const fn summarize_threshold_pct(&self) -> f32 {
        self.inner.summarize_threshold_pct
    }

    #[getter]
    pub const fn chunk_size(&self) -> usize {
        self.inner.chunk_size
    }

    #[getter]
    pub const fn max_summaries(&self) -> usize {
        self.inner.max_summaries
    }
}

impl From<PySummaryMemoryConfig> for SummaryMemoryConfig {
    fn from(py_config: PySummaryMemoryConfig) -> Self {
        py_config.inner
    }
}

#[pyclass]
#[derive(Clone)]
pub struct PyEntityMemoryConfig {
    inner: EntityMemoryConfig,
}

#[pymethods]
impl PyEntityMemoryConfig {
    #[new]
    #[pyo3(signature = (max_tokens=6144, max_entities=128, extraction_interval=3, similarity_threshold=0.65))]
    pub const fn new(
        max_tokens: usize,
        max_entities: usize,
        extraction_interval: usize,
        similarity_threshold: f32,
    ) -> Self {
        Self {
            inner: EntityMemoryConfig {
                max_tokens,
                max_entities,
                extraction_interval,
                similarity_threshold,
            },
        }
    }

    #[getter]
    pub const fn max_tokens(&self) -> usize {
        self.inner.max_tokens
    }

    #[getter]
    pub const fn max_entities(&self) -> usize {
        self.inner.max_entities
    }

    #[getter]
    pub const fn extraction_interval(&self) -> usize {
        self.inner.extraction_interval
    }

    #[getter]
    pub const fn similarity_threshold(&self) -> f32 {
        self.inner.similarity_threshold
    }
}

impl From<PyEntityMemoryConfig> for EntityMemoryConfig {
    fn from(py_config: PyEntityMemoryConfig) -> Self {
        py_config.inner
    }
}

#[pyclass]
#[derive(Clone)]
pub struct PyVectorMemoryConfig {
    inner: VectorMemoryConfig,
}

#[pymethods]
impl PyVectorMemoryConfig {
    #[new]
    #[pyo3(signature = (max_tokens=8192, max_memories=1000, top_k=5, similarity_threshold=0.7, buffer_tokens=4096, namespace=None))]
    pub const fn new(
        max_tokens: usize,
        max_memories: usize,
        top_k: usize,
        similarity_threshold: f32,
        buffer_tokens: usize,
        namespace: Option<String>,
    ) -> Self {
        Self {
            inner: VectorMemoryConfig {
                max_tokens,
                max_memories,
                top_k,
                similarity_threshold,
                buffer_tokens,
                namespace,
            },
        }
    }

    #[getter]
    pub const fn max_tokens(&self) -> usize {
        self.inner.max_tokens
    }

    #[getter]
    pub const fn max_memories(&self) -> usize {
        self.inner.max_memories
    }

    #[getter]
    pub const fn top_k(&self) -> usize {
        self.inner.top_k
    }

    #[getter]
    pub const fn similarity_threshold(&self) -> f32 {
        self.inner.similarity_threshold
    }

    #[getter]
    pub const fn buffer_tokens(&self) -> usize {
        self.inner.buffer_tokens
    }

    #[getter]
    pub fn namespace(&self) -> Option<String> {
        self.inner.namespace.clone()
    }
}

impl From<PyVectorMemoryConfig> for VectorMemoryConfig {
    fn from(py_config: PyVectorMemoryConfig) -> Self {
        py_config.inner
    }
}
