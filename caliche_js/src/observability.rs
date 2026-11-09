//! JavaScript bindings for Caliche Observability
#![expect(
    clippy::needless_pass_by_value,
    reason = "NAPI must own values converted from JavaScript arguments at the ABI boundary"
)]

use caliche::observability::metrics::MetricsRecorder as RustMetricsRecorder;
use caliche::observability::{
    ToolMetrics as RustToolMetrics, categorize_error as rust_categorize_error,
};
use napi_derive::napi;
use std::sync::Arc;
use tracing::Level;

/// JavaScript wrapper for `ToolStats`
#[napi(object)]
pub struct ToolStats {
    pub total_calls: f64,
    pub successful_calls: f64,
    pub failed_calls: f64,
    pub total_duration_ms: f64,
    pub error_types: std::collections::HashMap<String, f64>,
    pub last_executed: Option<String>,
}

fn u64_to_js_number(value: u64, field: &str) -> napi::Result<f64> {
    const MAX_SAFE_INTEGER: u64 = (1_u64 << 53) - 1;
    if value > MAX_SAFE_INTEGER {
        return Err(napi::Error::from_reason(format!(
            "{field} exceeds JavaScript's maximum safe integer"
        )));
    }
    #[expect(
        clippy::cast_precision_loss,
        reason = "the preceding JavaScript safe-integer bound makes this conversion exact"
    )]
    Ok(value as f64)
}

fn f64_to_u64(value: f64, field: &str) -> napi::Result<u64> {
    let u64_exclusive_upper_bound = 2.0_f64.powi(64);
    if !value.is_finite()
        || value < 0.0
        || value >= u64_exclusive_upper_bound
        || value.fract() != 0.0
    {
        return Err(napi::Error::from_reason(format!(
            "{field} must be a non-negative integer within the 64-bit range"
        )));
    }
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "finiteness, sign, integral value, and range are checked above"
    )]
    Ok(value as u64)
}

impl TryFrom<caliche::observability::ToolStats> for ToolStats {
    type Error = napi::Error;

    fn try_from(stats: caliche::observability::ToolStats) -> napi::Result<Self> {
        Ok(Self {
            total_calls: u64_to_js_number(stats.total_calls, "total calls")?,
            successful_calls: u64_to_js_number(stats.successful_calls, "successful calls")?,
            failed_calls: u64_to_js_number(stats.failed_calls, "failed calls")?,
            total_duration_ms: u64_to_js_number(stats.total_duration_ms, "total duration")?,
            error_types: stats
                .error_types
                .into_iter()
                .map(|(key, value)| Ok((key, u64_to_js_number(value, "error count")?)))
                .collect::<napi::Result<_>>()?,
            last_executed: stats.last_executed.map(|dt| dt.to_rfc3339()),
        })
    }
}

#[napi]
impl ToolStats {
    /// Get success rate
    #[napi(catch_unwind)]
    pub fn success_rate(&self) -> f64 {
        if self.total_calls == 0.0 {
            return 0.0;
        }
        self.successful_calls / self.total_calls
    }

    /// Get average duration ms
    #[napi(catch_unwind)]
    pub fn avg_duration_ms(&self) -> f64 {
        if self.total_calls == 0.0 {
            return 0.0;
        }
        self.total_duration_ms / self.total_calls
    }
}

/// JavaScript wrapper for `ToolMetrics`
#[napi]
pub struct ToolMetrics {
    inner: Arc<RustToolMetrics>,
}

#[napi]
impl ToolMetrics {
    /// Create a new `ToolMetrics`
    #[napi(constructor, catch_unwind)]
    pub fn new() -> Self {
        Self {
            inner: Arc::new(caliche::observability::ToolMetrics::new()),
        }
    }

    /// Record a tool execution
    #[napi]
    ///
    /// # Errors
    ///
    /// Returns a NAPI error when validation, synchronization, serialization,
    /// or an underlying runtime operation fails.
    pub async fn record_execution(
        &self,
        tool_name: String,
        duration_ms: f64,
        success: bool,
        error_type: Option<String>,
    ) -> napi::Result<()> {
        self.inner
            .record_execution(
                &tool_name,
                f64_to_u64(duration_ms, "duration_ms")?,
                success,
                error_type.as_deref(),
            )
            .await;
        Ok(())
    }

    /// Get statistics for a specific tool
    #[napi]
    pub async fn get_tool_stats(&self, tool_name: String) -> napi::Result<Option<ToolStats>> {
        self.inner
            .get_tool_stats(&tool_name)
            .await
            .map(TryInto::try_into)
            .transpose()
    }

    /// Get statistics for all tools
    #[napi]
    pub async fn get_all_stats(
        &self,
    ) -> napi::Result<std::collections::HashMap<String, ToolStats>> {
        let stats = self.inner.get_all_stats().await;
        stats
            .into_iter()
            .map(|(key, value)| Ok((key, value.try_into()?)))
            .collect()
    }

    /// Clear all metrics
    #[napi]
    ///
    /// # Errors
    ///
    /// Returns a NAPI error when validation, synchronization, serialization,
    /// or an underlying runtime operation fails.
    pub async fn clear(&self) -> napi::Result<()> {
        self.inner.clear().await;
        Ok(())
    }
}

/// JavaScript wrapper for `MetricsRecorder`
#[napi]
#[expect(
    dead_code,
    reason = "the handle owns the recorder, and no exported method reads it yet"
)]
pub struct MetricsRecorder {
    inner: Arc<RustMetricsRecorder>,
}

#[napi]
impl MetricsRecorder {
    /// Create a new `MetricsRecorder`
    #[napi(constructor, catch_unwind)]
    ///
    /// # Errors
    ///
    /// Returns a NAPI error when validation, synchronization, serialization,
    /// or an underlying runtime operation fails.
    pub fn new() -> napi::Result<Self> {
        let inner = RustMetricsRecorder::new().map_err(|e| {
            napi::Error::from_reason(format!("Failed to create metrics recorder: {e}"))
        })?;
        Ok(Self {
            inner: Arc::new(inner),
        })
    }
}

/// Initialize logging
#[napi(catch_unwind)]
pub fn init_logging(json_format: Option<bool>, level: Option<String>) -> napi::Result<()> {
    use caliche::observability::logging::init_logging_with_level;

    let json_format = json_format.unwrap_or(false);
    let level_enum = level.and_then(|l| match l.to_lowercase().as_str() {
        "trace" => Some(Level::TRACE),
        "debug" => Some(Level::DEBUG),
        "info" => Some(Level::INFO),
        "warn" => Some(Level::WARN),
        "error" => Some(Level::ERROR),
        _ => None,
    });

    init_logging_with_level(json_format, level_enum)
        .map_err(|e| napi::Error::from_reason(format!("Failed to init logging: {e}")))
}

/// Get current logging configuration
#[napi(catch_unwind)]
pub fn get_logging_config() -> LoggingConfig {
    use caliche::observability::logging::get_logging_config;

    let (json_format, level, initialized) = get_logging_config();
    LoggingConfig {
        json_format,
        level: format!("{level:?}"),
        initialized,
    }
}

/// Initialize telemetry
#[napi(catch_unwind)]
///
/// # Errors
///
/// Returns a NAPI error when validation, synchronization, serialization,
/// or an underlying runtime operation fails.
pub fn init_telemetry(service_name: String, otlp_endpoint: Option<String>) -> napi::Result<()> {
    caliche::observability::init_telemetry(&service_name, otlp_endpoint.as_deref())
        .map_err(|e| napi::Error::from_reason(format!("Failed to init telemetry: {e}")))
}

/// Categorize error
#[napi(catch_unwind)]
pub fn categorize_error(error: String) -> String {
    rust_categorize_error(&error)
}

#[napi(object)]
pub struct LoggingConfig {
    pub json_format: bool,
    pub level: String,
    pub initialized: bool,
}
