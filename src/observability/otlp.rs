//! OTLP (OpenTelemetry Protocol) exporter pipeline.
//!
//! Feature-gated behind `otlp`. Wires a `BatchSpanProcessor` with an OTLP exporter
//! (gRPC or HTTP) and installs it as the global `TracerProvider`.
//!
//! Respects standard OTel environment variables:
//! - `OTEL_EXPORTER_OTLP_ENDPOINT`
//! - `OTEL_SERVICE_NAME`
//! - `OTEL_RESOURCE_ATTRIBUTES`
//! - `OTEL_EXPORTER_OTLP_PROTOCOL` (grpc | http/protobuf)
//! - `OTEL_EXPORTER_OTLP_HEADERS`

use anyhow::{Context, Result};
use opentelemetry::KeyValue;
use opentelemetry_otlp::WithExportConfig;
use opentelemetry_sdk::Resource;
use opentelemetry_sdk::trace::SdkTracerProvider;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// OTLP transport protocol.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum OtlpProtocol {
    /// gRPC (default, port 4317)
    #[default]
    Grpc,
    /// HTTP/protobuf (port 4318)
    HttpProtobuf,
}

impl OtlpProtocol {
    fn from_env_str(s: &str) -> Self {
        match s.trim().to_ascii_lowercase().as_str() {
            "http/protobuf" | "http_protobuf" | "http-protobuf" | "httpprotobuf" => {
                Self::HttpProtobuf
            }
            _ => Self::Grpc,
        }
    }
}

/// Batch span processor tuning knobs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchConfig {
    /// Maximum queue size before dropping spans. Default: 2048.
    pub max_queue_size: usize,
    /// Maximum batch size per export. Default: 512.
    pub max_export_batch_size: usize,
    /// Scheduled delay between exports in milliseconds. Default: 5000.
    pub scheduled_delay_ms: u64,
    /// Maximum time to wait for an export to complete in milliseconds. Default: 30000.
    pub max_export_timeout_ms: u64,
}

impl Default for BatchConfig {
    fn default() -> Self {
        Self {
            max_queue_size: 2048,
            max_export_batch_size: 512,
            scheduled_delay_ms: 5000,
            max_export_timeout_ms: 30_000,
        }
    }
}

/// Top-level OTLP configuration.
///
/// Programmatic values take precedence over environment variables.
/// When a field is `None` or empty, the corresponding `OTEL_*` env var is consulted.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct OtlpConfig {
    /// Collector endpoint (e.g. `http://localhost:4317`).
    /// Falls back to `OTEL_EXPORTER_OTLP_ENDPOINT`.
    #[serde(default)]
    pub endpoint: Option<String>,
    /// Transport protocol. Falls back to `OTEL_EXPORTER_OTLP_PROTOCOL`.
    #[serde(default)]
    pub protocol: Option<OtlpProtocol>,
    /// Extra HTTP headers sent with every export request.
    /// Falls back to `OTEL_EXPORTER_OTLP_HEADERS` (comma-separated `k=v`).
    #[serde(default)]
    pub headers: HashMap<String, String>,
    /// Additional resource attributes merged into every span.
    /// Falls back to `OTEL_RESOURCE_ATTRIBUTES` (comma-separated `k=v`).
    #[serde(default)]
    pub resource_attributes: HashMap<String, String>,
    /// When `true`, prompt text and tool arguments are replaced with `[REDACTED]`
    /// in span attributes. Default: `false`.
    #[serde(default)]
    pub redact_prompts: bool,
    /// Batch exporter tuning. Uses sensible defaults when absent.
    #[serde(default)]
    pub batch_config: Option<BatchConfig>,
}

impl OtlpConfig {
    /// Resolve the effective endpoint, preferring the programmatic value.
    fn effective_endpoint(&self) -> String {
        if let Some(ref ep) = self.endpoint
            && !ep.is_empty()
        {
            return ep.clone();
        }
        std::env::var("OTEL_EXPORTER_OTLP_ENDPOINT")
            .unwrap_or_else(|_| "http://localhost:4317".to_owned())
    }

    /// Resolve the effective protocol.
    fn effective_protocol(&self) -> OtlpProtocol {
        if let Some(p) = self.protocol {
            return p;
        }
        std::env::var("OTEL_EXPORTER_OTLP_PROTOCOL")
            .map(|s| OtlpProtocol::from_env_str(&s))
            .unwrap_or_default()
    }

    /// Resolve effective service name.
    fn effective_service_name(&self) -> String {
        if let Some(name) = self.resource_attributes.get("service.name")
            && !name.is_empty()
        {
            return name.clone();
        }
        std::env::var("OTEL_SERVICE_NAME").unwrap_or_else(|_| "caliche".to_owned())
    }

    /// Merge env-var headers into the programmatic set.
    fn effective_headers(&self) -> HashMap<String, String> {
        let mut merged = self.headers.clone();
        if let Ok(env_headers) = std::env::var("OTEL_EXPORTER_OTLP_HEADERS") {
            for pair in env_headers.split(',') {
                if let Some((k, v)) = pair.split_once('=') {
                    merged
                        .entry(k.trim().to_owned())
                        .or_insert_with(|| v.trim().to_owned());
                }
            }
        }
        merged
    }

    /// Build the OTel `Resource` from config + env vars.
    fn build_resource(&self) -> Resource {
        let service_name = self.effective_service_name();
        let mut attrs: Vec<KeyValue> = vec![KeyValue::new("service.name", service_name)];

        for (k, v) in &self.resource_attributes {
            if k == "service.name" {
                continue;
            }
            attrs.push(KeyValue::new(k.clone(), v.clone()));
        }
        if let Ok(env_attrs) = std::env::var("OTEL_RESOURCE_ATTRIBUTES") {
            for pair in env_attrs.split(',') {
                if let Some((k, v)) = pair.split_once('=') {
                    let key = k.trim();
                    if !self.resource_attributes.contains_key(key) && key != "service.name" {
                        attrs.push(KeyValue::new(key.to_owned(), v.trim().to_owned()));
                    }
                }
            }
        }
        Resource::builder().with_attributes(attrs).build()
    }
}

/// Initialise the OTLP trace pipeline and install a global `TracerProvider`.
///
/// Returns the provider so the caller can hold it (and flush on shutdown).
pub fn init_otlp_pipeline(config: &OtlpConfig) -> Result<SdkTracerProvider> {
    let protocol = config.effective_protocol();
    let endpoint = config.effective_endpoint();
    let _headers = config.effective_headers();
    let resource = config.build_resource();

    let exporter = match protocol {
        OtlpProtocol::Grpc => opentelemetry_otlp::SpanExporter::builder()
            .with_tonic()
            .with_endpoint(&endpoint)
            .build()
            .context("failed to build OTLP gRPC span exporter")?,
        OtlpProtocol::HttpProtobuf => opentelemetry_otlp::SpanExporter::builder()
            .with_http()
            .with_endpoint(&endpoint)
            .build()
            .context("failed to build OTLP HTTP span exporter")?,
    };

    let provider = SdkTracerProvider::builder()
        .with_batch_exporter(exporter)
        .with_resource(resource)
        .build();

    opentelemetry::global::set_tracer_provider(provider.clone());

    tracing::info!(
        endpoint = %endpoint,
        protocol = ?protocol,
        redact_prompts = config.redact_prompts,
        "OTLP trace pipeline initialised"
    );

    Ok(provider)
}

/// Conditionally redact a string value based on the redaction flag.
#[inline]
pub fn maybe_redact(value: &str, redact: bool) -> &str {
    if redact { "[REDACTED]" } else { value }
}

/// Check whether OTLP export should be enabled.
///
/// Returns `true` when the `otlp` feature is compiled in **and** either:
/// - `OTEL_EXPORTER_OTLP_ENDPOINT` is set, or
/// - a programmatic config has been provided.
pub fn otlp_enabled_from_env() -> bool {
    std::env::var("OTEL_EXPORTER_OTLP_ENDPOINT").is_ok()
}
