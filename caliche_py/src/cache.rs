//! Python bindings for Caliche Caching

use crate::conversions::json_to_python_dict;
use crate::get_runtime;
use caliche::llm::cache::{CachePolicy, CacheStats, CachedResponse, LlmCache};
use caliche::messages::TokenUsage;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

/// Python wrapper for `CachePolicy`
#[pyclass]
pub struct PyCachePolicy {
    inner: CachePolicy,
}

#[pymethods]
impl PyCachePolicy {
    #[new]
    #[pyo3(signature = (policy_type, threshold = None))]
    pub fn new(policy_type: &str, threshold: Option<f32>) -> PyResult<Self> {
        let policy = match policy_type.to_lowercase().as_str() {
            "exact_match" | "exact" => CachePolicy::ExactMatch,
            "semantic_similarity" | "semantic" => CachePolicy::SemanticSimilarity {
                threshold: threshold.unwrap_or(0.95),
            },
            "disabled" => CachePolicy::Disabled,
            _ => {
                return Err(PyRuntimeError::new_err(
                    "Invalid policy type: must be 'exact_match', 'semantic_similarity', or 'disabled'",
                ));
            }
        };
        Ok(Self { inner: policy })
    }

    #[pyo3(name = "to_string")]
    pub fn as_string(&self) -> String {
        match &self.inner {
            CachePolicy::Disabled => "disabled".to_string(),
            CachePolicy::ExactMatch => "exact_match".to_string(),
            CachePolicy::SemanticSimilarity { .. } => "semantic_similarity".to_string(),
        }
    }
}

/// Python wrapper for `CachedResponse`
#[pyclass]
pub struct PyCachedResponse {
    inner: CachedResponse,
}

#[pymethods]
impl PyCachedResponse {
    #[new]
    pub fn new(
        content: String,
        prompt_tokens: Option<u32>,
        completion_tokens: Option<u32>,
    ) -> Self {
        Self {
            inner: CachedResponse {
                content,
                token_usage: TokenUsage {
                    prompt_tokens,
                    completion_tokens,
                    total_tokens: prompt_tokens.zip(completion_tokens).map(|(p, c)| p + c),
                    embedding_tokens: None,
                    context_window: None,
                    cache_creation_tokens: None,
                    cache_read_tokens: None,
                    last_prompt_tokens: None,
                },
                metadata: serde_json::json!({}),
            },
        }
    }

    pub fn content(&self) -> String {
        self.inner.content.clone()
    }

    pub fn token_usage(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let dict = PyDict::new(py);
        if let Some(pt) = self.inner.token_usage.prompt_tokens {
            dict.set_item("prompt_tokens", pt)?;
        }
        if let Some(ct) = self.inner.token_usage.completion_tokens {
            dict.set_item("completion_tokens", ct)?;
        }
        if let Some(tt) = self.inner.token_usage.total_tokens {
            dict.set_item("total_tokens", tt)?;
        }
        if let Some(created) = self.inner.token_usage.cache_creation_tokens {
            dict.set_item("cache_creation_tokens", created)?;
        }
        if let Some(read) = self.inner.token_usage.cache_read_tokens {
            dict.set_item("cache_read_tokens", read)?;
        }
        if let Some(last) = self.inner.token_usage.last_prompt_tokens {
            dict.set_item("last_prompt_tokens", last)?;
        }
        Ok(dict.into())
    }

    pub fn metadata(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        json_to_python_dict(py, &self.inner.metadata).map(std::convert::Into::into)
    }
}

/// Python wrapper for `CacheStats`
#[pyclass]
pub struct PyCacheStats {
    inner: CacheStats,
}

#[pymethods]
impl PyCacheStats {
    pub const fn hits(&self) -> u64 {
        self.inner.hits
    }

    pub const fn misses(&self) -> u64 {
        self.inner.misses
    }

    pub const fn size(&self) -> usize {
        self.inner.size
    }

    pub const fn hit_rate(&self) -> f64 {
        self.inner.hit_rate
    }

    pub fn to_dict(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let dict = PyDict::new(py);
        dict.set_item("hits", self.inner.hits)?;
        dict.set_item("misses", self.inner.misses)?;
        dict.set_item("size", self.inner.size)?;
        dict.set_item("hit_rate", self.inner.hit_rate)?;
        Ok(dict.into())
    }
}

/// Python wrapper for `LlmCache`
#[pyclass]
pub struct PyLlmCache {
    inner: LlmCache,
}

#[pymethods]
impl PyLlmCache {
    #[new]
    #[pyo3(signature = (max_size, ttl_secs, policy))]
    pub fn new(max_size: usize, ttl_secs: u64, policy: &PyCachePolicy) -> Self {
        Self {
            inner: LlmCache::new(max_size, ttl_secs, policy.inner.clone()),
        }
    }

    /// Get cached response
    pub fn get(&self, prompt: &str, model: &str) -> PyResult<Option<PyCachedResponse>> {
        let cache_clone = &self.inner;
        let rt = get_runtime()?;

        let result = rt.block_on(async { cache_clone.get(prompt, model).await });

        match result {
            Ok(Some(response)) => Ok(Some(PyCachedResponse { inner: response })),
            Ok(None) => Ok(None),
            Err(e) => Err(PyRuntimeError::new_err(format!("Cache get failed: {e}"))),
        }
    }

    /// Put response in cache
    pub fn put(&self, prompt: &str, model: &str, response: &PyCachedResponse) -> PyResult<()> {
        let cache_clone = &self.inner;
        let rt = get_runtime()?;

        let result =
            rt.block_on(async { cache_clone.put(prompt, model, response.inner.clone()).await });

        match result {
            Ok(()) => Ok(()),
            Err(e) => Err(PyRuntimeError::new_err(format!("Cache put failed: {e}"))),
        }
    }

    /// Clear cache
    pub fn clear(&self) -> PyResult<()> {
        let cache_clone = &self.inner;
        let rt = get_runtime()?;

        let result = rt.block_on(async { cache_clone.clear().await });

        match result {
            Ok(()) => Ok(()),
            Err(e) => Err(PyRuntimeError::new_err(format!("Cache clear failed: {e}"))),
        }
    }

    /// Get cache statistics
    pub fn get_stats(&self) -> PyResult<PyCacheStats> {
        let cache_clone = &self.inner;
        let rt = get_runtime()?;

        let result = rt.block_on(async { cache_clone.get_stats().await });

        match result {
            Ok(stats) => Ok(PyCacheStats { inner: stats }),
            Err(e) => Err(PyRuntimeError::new_err(format!("Failed to get stats: {e}"))),
        }
    }
}
