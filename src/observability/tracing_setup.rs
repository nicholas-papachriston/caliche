//! Telemetry initialisation.
//!
//! When the `otlp` feature is enabled **and** an OTLP endpoint is configured
//! (via `OtlpConfig` or the `OTEL_EXPORTER_OTLP_ENDPOINT` env var), spans are
//! exported to the collector. Otherwise telemetry init is a no-op (or installs
//! a lightweight no-op provider when the OTel SDK is present but unused).

use anyhow::Result;

/// Initialise the OpenTelemetry tracer provider.
///
/// * With the `otlp` feature **and** a reachable endpoint, a full OTLP pipeline
///   is started (batch processor + gRPC/HTTP exporter).
/// * With the `otlp` feature but no endpoint, a lightweight no-op provider is
///   installed so the rest of the codebase can use `tracing` spans without a
///   runtime cost.
/// * Without `otlp`, this is a no-op so lean presets omit the OpenTelemetry SDK.
pub fn init_telemetry(service_name: &str, _otlp_endpoint: Option<&str>) -> Result<()> {
    #[cfg(feature = "otlp")]
    {
        use opentelemetry_sdk::trace::SdkTracerProvider;

        let should_enable = _otlp_endpoint.is_some() || super::otlp::otlp_enabled_from_env();
        if should_enable {
            let mut config = super::otlp::OtlpConfig::default();
            if let Some(ep) = _otlp_endpoint {
                config.endpoint = Some(ep.to_owned());
            }
            config
                .resource_attributes
                .insert("service.name".to_owned(), service_name.to_owned());
            let _provider = super::otlp::init_otlp_pipeline(&config)?;
            return Ok(());
        }

        let provider = SdkTracerProvider::builder().build();
        opentelemetry::global::set_tracer_provider(provider);
        tracing::info!(
            "OpenTelemetry tracing initialized for service: {} (no-op exporter)",
            service_name
        );
        return Ok(());
    }

    #[cfg(not(feature = "otlp"))]
    {
        tracing::info!(
            "OpenTelemetry tracing skipped for service: {} (otlp feature disabled)",
            service_name
        );
        Ok(())
    }
}
