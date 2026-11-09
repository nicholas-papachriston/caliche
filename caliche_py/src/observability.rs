//! Python bindings for Caliche Observability

use crate::get_runtime;
use caliche::observability::logging::{
    LogCallback, LogEntry, get_logging_config, init_logging_with_level, set_log_callback,
};
use caliche::observability::metrics::{MetricsRecorder, register_metrics};
use caliche::observability::{ToolMetrics, ToolStats, categorize_error, init_telemetry};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::sync::Arc;
use tracing::Level;

/// Python wrapper for `ToolStats`
#[pyclass]
pub struct PyToolStats {
    inner: ToolStats,
}

#[pymethods]
impl PyToolStats {
    pub const fn total_calls(&self) -> u64 {
        self.inner.total_calls
    }

    pub const fn successful_calls(&self) -> u64 {
        self.inner.successful_calls
    }

    pub const fn failed_calls(&self) -> u64 {
        self.inner.failed_calls
    }

    pub const fn total_duration_ms(&self) -> u64 {
        self.inner.total_duration_ms
    }

    pub fn success_rate(&self) -> f64 {
        self.inner.success_rate()
    }

    pub fn avg_duration_ms(&self) -> f64 {
        self.inner.avg_duration_ms()
    }

    pub fn error_types(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let dict = PyDict::new(py);
        for (k, v) in &self.inner.error_types {
            dict.set_item(k, *v)?;
        }
        Ok(dict.into())
    }

    pub fn last_executed(&self) -> Option<String> {
        self.inner.last_executed.map(|dt| dt.to_rfc3339())
    }

    pub fn to_dict(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let dict = PyDict::new(py);
        dict.set_item("total_calls", self.inner.total_calls)?;
        dict.set_item("successful_calls", self.inner.successful_calls)?;
        dict.set_item("failed_calls", self.inner.failed_calls)?;
        dict.set_item("total_duration_ms", self.inner.total_duration_ms)?;
        dict.set_item("success_rate", self.inner.success_rate())?;
        dict.set_item("avg_duration_ms", self.inner.avg_duration_ms())?;
        let error_dict = PyDict::new(py);
        for (k, v) in &self.inner.error_types {
            error_dict.set_item(k, *v)?;
        }
        dict.set_item("error_types", error_dict)?;
        if let Some(ref dt) = self.inner.last_executed {
            dict.set_item("last_executed", dt.to_rfc3339())?;
        }
        Ok(dict.into())
    }
}

/// Python wrapper for `ToolMetrics`
#[pyclass]
pub struct PyToolMetrics {
    inner: ToolMetrics,
}

#[pymethods]
impl PyToolMetrics {
    #[new]
    pub fn new() -> Self {
        Self {
            inner: ToolMetrics::new(),
        }
    }

    /// Record a tool execution
    pub fn record_execution(
        &self,
        tool_name: &str,
        duration_ms: u64,
        success: bool,
        error_type: Option<&str>,
    ) -> PyResult<()> {
        let metrics_clone = &self.inner;
        let rt = get_runtime()?;

        rt.block_on(async {
            metrics_clone
                .record_execution(tool_name, duration_ms, success, error_type)
                .await;
        });
        Ok(())
    }

    /// Get statistics for a specific tool
    pub fn get_tool_stats(&self, tool_name: &str) -> PyResult<Option<PyToolStats>> {
        let metrics_clone = &self.inner;
        let rt = get_runtime()?;

        let result = rt.block_on(async { metrics_clone.get_tool_stats(tool_name).await });

        Ok(result.map(|stats| PyToolStats { inner: stats }))
    }

    /// Get statistics for all tools
    pub fn get_all_stats(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let metrics_clone = &self.inner;
        let rt = get_runtime()?;

        let result = rt.block_on(async { metrics_clone.get_all_stats().await });

        let dict = PyDict::new(py);
        for (k, v) in result {
            let stats = PyToolStats { inner: v };
            dict.set_item(k, stats)?;
        }
        Ok(dict.into())
    }

    /// Clear all metrics
    pub fn clear(&self) -> PyResult<()> {
        let metrics_clone = &self.inner;
        let rt = get_runtime()?;

        rt.block_on(async { metrics_clone.clear().await });
        Ok(())
    }
}

/// Python wrapper for `MetricsRecorder`
#[pyclass]
pub struct PyMetricsRecorder {
    inner: MetricsRecorder,
}

#[pymethods]
impl PyMetricsRecorder {
    #[new]
    pub fn new() -> PyResult<Self> {
        let inner = MetricsRecorder::new().map_err(|e| {
            PyRuntimeError::new_err(format!("Failed to create metrics recorder: {e}"))
        })?;
        Ok(Self { inner })
    }

    /// Increment request counter
    pub fn increment_requests(&self, count: u64) {
        self.inner.request_total.inc_by(count);
    }

    /// Record request duration
    pub fn record_duration(&self, seconds: f64) {
        self.inner.request_duration.observe(seconds);
    }

    /// Increment token usage
    pub fn increment_tokens(&self, count: u64) {
        self.inner.token_usage.inc_by(count);
    }

    /// Increment cache hits
    pub fn increment_cache_hits(&self, count: u64) {
        self.inner.cache_hits.inc_by(count);
    }

    /// Increment cache misses
    pub fn increment_cache_misses(&self, count: u64) {
        self.inner.cache_misses.inc_by(count);
    }

    /// Set active workflows count
    pub fn set_active_workflows(&self, count: i64) {
        self.inner.active_workflows.set(count);
    }

    /// Set agent pool size
    pub fn set_agent_pool_size(&self, size: i64) {
        self.inner.agent_pool_size.set(size);
    }
}

/// Initialize logging
#[pyfunction]
#[pyo3(signature = (json_format = false, level = None))]
pub fn py_init_logging(json_format: bool, level: Option<&str>) -> PyResult<()> {
    let log_level = level.and_then(|l| match l.to_lowercase().as_str() {
        "trace" => Some(Level::TRACE),
        "debug" => Some(Level::DEBUG),
        "info" => Some(Level::INFO),
        "warn" => Some(Level::WARN),
        "error" => Some(Level::ERROR),
        _ => None,
    });

    match init_logging_with_level(json_format, log_level) {
        Ok(()) => Ok(()),
        Err(e) => Err(PyRuntimeError::new_err(format!(
            "Failed to init logging: {e}"
        ))),
    }
}

/// Set a log callback function
#[pyfunction]
pub fn py_set_log_callback(callback: Py<PyAny>) {
    let callback: LogCallback = Arc::new(move |entry: &LogEntry| {
        Python::attach(|py| {
            let dict = PyDict::new(py);
            dict.set_item("level", entry.level.as_str()).ok();
            dict.set_item("target", entry.target.as_str()).ok();
            dict.set_item("message", entry.message.as_str()).ok();
            dict.set_item("timestamp", entry.timestamp.to_rfc3339())
                .ok();

            let fields_dict = PyDict::new(py);
            for (k, v) in &entry.fields {
                fields_dict.set_item(k, v).ok();
            }
            dict.set_item("fields", fields_dict).ok();

            let _ = callback.call1(py, (dict,));
        });
    });

    set_log_callback(callback);
}

/// Get current logging configuration
#[pyfunction]
pub fn py_get_logging_config(py: Python) -> PyResult<Py<PyDict>> {
    let (json_format, level, initialized) = get_logging_config();
    let dict = PyDict::new(py);
    dict.set_item("json_format", json_format)?;
    dict.set_item("level", format!("{level:?}"))?;
    dict.set_item("initialized", initialized)?;
    Ok(dict.into())
}

/// Initialize telemetry
#[pyfunction]
#[pyo3(signature = (service_name, otlp_endpoint = None))]
pub fn py_init_telemetry(service_name: &str, otlp_endpoint: Option<&str>) -> PyResult<()> {
    match init_telemetry(service_name, otlp_endpoint) {
        Ok(()) => Ok(()),
        Err(e) => Err(PyRuntimeError::new_err(format!(
            "Failed to init telemetry: {e}"
        ))),
    }
}

/// Register metrics
#[pyfunction]
pub fn py_register_metrics() -> PyResult<PyMetricsRecorder> {
    let inner = register_metrics()
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to register metrics: {e}")))?;
    Ok(PyMetricsRecorder { inner })
}

/// Categorize error
#[pyfunction]
pub fn py_categorize_error(error: &str) -> String {
    categorize_error(error)
}
