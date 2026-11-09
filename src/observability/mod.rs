//! Observability: logging, metrics, and tracing.
//!
//! - **Audit logging**: PostgreSQL, webhook, and file backends
//! - **Structured logging**: JSON logging with request ID tracking
//! - **Metrics**: Prometheus counters and histograms
//! - **Tool metrics**: Call counts, durations, and error rates
//! - **Hotpath perf**: Gated `caliche::perf` spans for edit/search (off when not measuring)
//! - **Tracing**: OpenTelemetry distributed tracing
//!
//! # Examples
//!
//! ## Initialize Logging
//!
//! ```rust,no_run
//! use caliche::observability::init_logging;
//!
//! init_logging(false); // false = text format, true = JSON format
//! ```
//!
//! ## Tool Metrics
//!
//! ```rust,no_run
//! use caliche::observability::ToolMetrics;
//! use std::sync::Arc;
//!
//! let metrics = Arc::new(ToolMetrics::new());
//! // Metrics are automatically tracked when used with Agent
//! ```
//!
//! ## Audit Logging
//!
//! ```rust,no_run
//! use caliche::observability::{AuditLog, AuditLogBackend};
//!
//! # async fn example() -> anyhow::Result<()> {
//! let audit_log = AuditLog::new(
//!     "req-123".to_string(),
//!     "chat".to_string(),
//!     "/api/chat".to_string(),
//!     "success".to_string(),
//! );
//! // Send to audit backend
//! # Ok(())
//! # }
//! ```

pub mod audit_log;
pub mod cache_metrics;
pub mod logging;
pub mod metrics;
#[cfg(feature = "otlp")]
pub mod otlp;
pub mod perf;
pub mod tool_metrics;
pub mod tracing_setup;
pub mod verbosity;

#[cfg(feature = "testing")]
pub mod inspector;

pub use audit_log::{AuditLog, AuditLogBackend, AuditLogger};
pub use cache_metrics::{CacheMetrics, CacheMetricsSnapshot, CacheStats};
pub use logging::{
    LogCallback, LogEntry, get_logging_config, init_logging, init_logging_with_level,
    set_log_callback,
};
pub use metrics::{MetricsRecorder, register_metrics};
#[cfg(feature = "otlp")]
pub use otlp::{OtlpConfig, OtlpProtocol, init_otlp_pipeline, maybe_redact, otlp_enabled_from_env};
pub use perf::{
    PerfEvent, configure as configure_perf, enabled as perf_enabled, span as perf_span,
};
pub use tool_metrics::{ToolMetrics, ToolStats, categorize_error};
pub use tracing_setup::init_telemetry;
pub use verbosity::{OutputVerbosity, OutputVerbosityParseError};
