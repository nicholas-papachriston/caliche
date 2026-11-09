//! Python bindings for Caliche RAG Service

use caliche::rag::{RagFusionMode, RagSearchMode, RagService as RustRagService};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use std::sync::Arc;

use crate::agent::Agent as PyAgent;
use crate::block_on_detached;
use crate::conversions::python_to_json_value;

/// Python wrapper for `RagService`
#[pyclass]
pub struct PyRagService {
    inner: Arc<RustRagService>,
}

impl PyRagService {
    /// Create a `PyRagService` from an existing Rust `RagService` Arc.
    ///
    /// This allows sharing the same backend between `RagTool` and `RagService` instances.
    pub const fn from_service(service: Arc<RustRagService>) -> PyResult<Self> {
        Ok(Self { inner: service })
    }
}

#[pymethods]
impl PyRagService {
    /// Create a new RAG service with default in-memory backend
    #[new]
    pub fn new(py: Python<'_>, agent: &PyAgent) -> PyResult<Self> {
        let agent_arc = agent
            .inner
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("Agent not initialized"))?
            .clone();

        let embedding_provider = block_on_detached(py, async move {
            let agent_guard = agent_arc.lock().await;
            agent_guard.embedding_provider().ok_or_else(|| {
                anyhow::anyhow!("Agent does not have an embedding provider configured")
            })
        })?
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to get embedding provider: {e}")))?;

        let rag_service = RustRagService::new(embedding_provider);

        Ok(Self {
            inner: Arc::new(rag_service),
        })
    }

    /// Index a document
    pub fn index_document(
        &self,
        py: Python<'_>,
        content: String,
        doc_id: String,
    ) -> PyResult<Py<PyAny>> {
        let service_clone = self.inner.clone();

        let (chunk_count, usage) = block_on_detached(py, async move {
            service_clone.index_document(&content, &doc_id).await
        })?
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to index document: {e}")))?;

        let result = PyDict::new(py);
        result.set_item("chunk_count", chunk_count)?;
        if let Some(u) = usage {
            let usage_dict = PyDict::new(py);
            if let Some(embedding_tokens) = u.embedding_tokens {
                usage_dict.set_item("embedding_tokens", embedding_tokens)?;
            }
            if let Some(total_tokens) = u.total_tokens {
                usage_dict.set_item("total_tokens", total_tokens)?;
            }
            result.set_item("usage", usage_dict)?;
        }
        Ok(result.into_any().unbind())
    }

    /// Query for relevant chunks
    #[pyo3(signature = (query, top_k, min_similarity=None))]
    pub fn query(
        &self,
        py: Python<'_>,
        query: String,
        top_k: usize,
        min_similarity: Option<f32>,
    ) -> PyResult<Py<PyAny>> {
        let min_sim = min_similarity.unwrap_or(0.3);
        let service_clone = self.inner.clone();

        let (results, top_similarity, usage) = block_on_detached(py, async move {
            service_clone.query(&query, top_k, min_sim).await
        })?
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to query: {e}")))?;

        let result = PyDict::new(py);

        // Results array
        let results_list = PyList::empty(py);
        for r in results {
            results_list.append(r)?;
        }
        result.set_item("results", results_list)?;

        // Top similarity
        if let Some(sim) = top_similarity {
            result.set_item("top_similarity", sim)?;
        }

        // Usage
        if let Some(u) = usage {
            let usage_dict = PyDict::new(py);
            if let Some(embedding_tokens) = u.embedding_tokens {
                usage_dict.set_item("embedding_tokens", embedding_tokens)?;
            }
            if let Some(total_tokens) = u.total_tokens {
                usage_dict.set_item("total_tokens", total_tokens)?;
            }
            result.set_item("usage", usage_dict)?;
        }

        Ok(result.into_any().unbind())
    }

    /// Enhanced query with hybrid search, reranking, and filtering.
    ///
    /// `filter_dsl` accepts a FilterDto-shaped dict, e.g.
    /// `{"type": "equal", "field": "category", "value": "docs"}`.
    #[pyo3(signature = (query, top_k, min_similarity=None, search_mode=None, dense_weight=None, sparse_weight=None, rerank=None, metadata_filter=None, max_context_tokens=None, filter_dsl=None))]
    pub fn enhanced_query(
        &self,
        py: Python<'_>,
        query: String,
        top_k: usize,
        min_similarity: Option<f32>,
        search_mode: Option<String>,
        dense_weight: Option<f32>,
        sparse_weight: Option<f32>,
        rerank: Option<bool>,
        metadata_filter: Option<&Bound<'_, PyDict>>,
        max_context_tokens: Option<usize>,
        filter_dsl: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Py<PyAny>> {
        let min_sim = min_similarity.unwrap_or(0.3);
        let search_mode = match search_mode.as_deref() {
            Some("hybrid") => RagSearchMode::HybridDenseSparse,
            Some("full_text") => RagSearchMode::FullText,
            Some("hybrid_full_text") => RagSearchMode::HybridDenseFullText,
            Some("dense") | Some(_) | None => RagSearchMode::Dense,
        };
        let rerank_enabled = rerank.unwrap_or(false);

        // Convert metadata filter
        let metadata_filter_value = if let Some(filter_dict) = metadata_filter {
            let mut map = serde_json::Map::new();
            for (key, value) in filter_dict.iter() {
                let key_str = key.extract::<String>()?;
                let value_json = python_to_json_value(&value)?;
                map.insert(key_str, value_json);
            }
            serde_json::Value::Object(map)
        } else {
            serde_json::Value::Null
        };

        let dsl_filter = if let Some(filter_dict) = filter_dsl {
            let value = python_to_json_value(filter_dict.as_any())?;
            Some(
                caliche::rag::filter_dto::filter_from_value(&value)
                    .map_err(|e| PyRuntimeError::new_err(format!("Invalid filter_dsl: {e}")))?,
            )
        } else {
            None
        };

        let service_clone = self.inner.clone();

        let (results, top_sim, usage, warnings, context_stats, texts) =
            block_on_detached(py, async move {
                let (results, top_sim, usage, warnings, context_stats) = service_clone
                    .enhanced_query(
                        &query,
                        top_k,
                        min_sim,
                        search_mode,
                        RagFusionMode::Weighted,
                        dense_weight,
                        sparse_weight,
                        rerank_enabled,
                        if metadata_filter_value.is_null() {
                            None
                        } else {
                            Some(&metadata_filter_value)
                        },
                        dsl_filter,
                        max_context_tokens,
                        None, // Provider type
                        None, // Model
                        None, // use_mmr
                        None, // mmr_lambda
                        None, // use_temporal_decay
                        None, // temporal_decay_half_life_days
                    )
                    .await?;

                let mut texts = Vec::new();
                for result in &results {
                    let text = service_clone.get_chunk_text(&result.chunk.id).await;
                    texts.push(text);
                }
                Ok::<_, anyhow::Error>((results, top_sim, usage, warnings, context_stats, texts))
            })?
            .map_err(|e| {
                PyRuntimeError::new_err(format!("Failed to perform enhanced query: {e}"))
            })?;

        let result = PyDict::new(py);

        // Results array
        let results_list = PyList::empty(py);
        for (result_item, text_opt) in results.iter().zip(texts.iter()) {
            let result_dict = PyDict::new(py);
            result_dict.set_item("id", &result_item.chunk.id)?;
            result_dict.set_item("score", result_item.score)?;
            if let Some(text) = text_opt {
                result_dict.set_item("text", text)?;
            }
            // metadata is serde_json::Value, not Option
            let metadata = &result_item.chunk.metadata;
            if !metadata.is_null() {
                let metadata_dict = PyDict::new(py);
                if let Some(obj) = metadata.as_object() {
                    for (k, v) in obj {
                        let py_val: Py<PyAny> = match v {
                            serde_json::Value::String(s) => {
                                s.clone().into_pyobject(py)?.into_any().unbind()
                            }
                            serde_json::Value::Number(n) => {
                                if let Some(i) = n.as_i64() {
                                    i.into_pyobject(py)?.into_any().unbind()
                                } else if let Some(f) = n.as_f64() {
                                    f.into_pyobject(py)?.into_any().unbind()
                                } else {
                                    py.None()
                                }
                            }
                            serde_json::Value::Bool(b) => b
                                .into_pyobject(py)?
                                .as_borrowed()
                                .to_owned()
                                .into_any()
                                .unbind(),
                            _ => py.None(),
                        };
                        metadata_dict.set_item(k, py_val)?;
                    }
                }
                result_dict.set_item("metadata", metadata_dict)?;
            }
            results_list.append(result_dict)?;
        }
        result.set_item("results", results_list)?;

        // Top similarity
        if let Some(sim) = top_sim {
            result.set_item("top_similarity", sim)?;
        }

        // Usage
        if let Some(u) = usage {
            let usage_dict = PyDict::new(py);
            if let Some(embedding_tokens) = u.embedding_tokens {
                usage_dict.set_item("embedding_tokens", embedding_tokens)?;
            }
            if let Some(total_tokens) = u.total_tokens {
                usage_dict.set_item("total_tokens", total_tokens)?;
            }
            result.set_item("usage", usage_dict)?;
        }

        // Warnings
        let warnings_list = PyList::empty(py);
        for warning in warnings {
            warnings_list.append(warning)?;
        }
        result.set_item("warnings", warnings_list)?;

        // Context stats
        if let Some(stats) = context_stats {
            let stats_dict = PyDict::new(py);
            stats_dict.set_item("included_chunks", stats.included_chunks)?;
            stats_dict.set_item("omitted_chunks", stats.omitted_chunks)?;
            stats_dict.set_item("total_chunks", stats.total_chunks)?;
            stats_dict.set_item("truncated", stats.truncated)?;
            result.set_item("context_stats", stats_dict)?;
        }

        Ok(result.into_any().unbind())
    }

    /// Graph search (requires graph backend)
    #[pyo3(signature = (query, top_k, max_hops=None))]
    pub fn graph_search(
        &self,
        py: Python<'_>,
        query: String,
        top_k: usize,
        max_hops: Option<usize>,
    ) -> PyResult<Py<PyAny>> {
        let max_h = max_hops.unwrap_or(2);
        let service_clone = self.inner.clone();

        let graph_result = block_on_detached(py, async move {
            service_clone.graph_search(&query, top_k, max_h).await
        })?
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to perform graph search: {e}")))?;

        let result = PyDict::new(py);

        // Chunks array
        let chunks_list = PyList::empty(py);
        for chunk in graph_result.entry_nodes {
            let chunk_dict = PyDict::new(py);
            chunk_dict.set_item("id", chunk.id)?;
            chunk_dict.set_item("type", chunk.node_type)?;
            chunks_list.append(chunk_dict)?;
        }
        result.set_item("chunks", chunks_list)?;

        // Entities array
        let entities_list = PyList::empty(py);
        for entity in graph_result.expanded_nodes {
            let entity_dict = PyDict::new(py);
            entity_dict.set_item("id", entity.id)?;
            entity_dict.set_item("type", entity.node_type)?;
            entities_list.append(entity_dict)?;
        }
        result.set_item("entities", entities_list)?;

        Ok(result.into_any().unbind())
    }

    /// Get statistics
    pub fn stats(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let service_clone = self.inner.clone();

        let (embedding_count, text_count) =
            block_on_detached(py, async move { service_clone.stats().await })?
                .map_err(|e| PyRuntimeError::new_err(format!("Failed to get stats: {e}")))?;

        let stats = PyDict::new(py);
        stats.set_item("embedding_count", embedding_count)?;
        stats.set_item("text_count", text_count)?;
        Ok(stats.into_any().unbind())
    }

    /// Clear all indexed documents
    pub fn clear(&self, py: Python<'_>) -> PyResult<()> {
        let service_clone = self.inner.clone();

        block_on_detached(py, async move { service_clone.clear().await })?
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to clear: {e}")))
    }
}
