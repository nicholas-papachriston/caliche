//! Prometheus metrics endpoint handler.
//!
//! Exposes both the process-local [`AppState::metrics_recorder`] registry and
//! the global default `prometheus::gather()` output so counters registered by
//! either path surface in a single `/metrics` response.

use crate::AppState;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::Response;
use prometheus::{Encoder, TextEncoder};

/// GET /metrics - Prometheus metrics endpoint.
///
/// Concatenates app-local metrics (from `state.metrics_recorder`) with any
/// metrics registered on the global prometheus registry. Metric families are
/// printed in a single text document so scrapers see a stable union.
pub async fn prometheus_metrics_handler(
    State(state): State<AppState>,
) -> Result<Response<String>, (StatusCode, String)> {
    let encoder = TextEncoder::new();

    // Gather from the app-local registry first so HTTP-layer metrics take
    // precedence for their names.
    let app_registry = state.metrics_recorder.registry();
    let app_families = app_registry.gather();
    let mut buffer = Vec::with_capacity(2 * 1024);
    if let Err(e) = encoder.encode(&app_families, &mut buffer) {
        return Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("Failed to encode app metrics: {e}"),
        ));
    }

    // Global default registry (populated by any `prometheus::register_*` call
    // elsewhere in the binary). De-duplicate metric family names so we do not
    // double-emit counters that were registered against both registries.
    let mut seen_families = std::collections::HashSet::new();
    for family in &app_families {
        seen_families.insert(family.name().to_string());
    }
    let global_families = prometheus::gather();
    let global_filtered: Vec<_> = global_families
        .into_iter()
        .filter(|family| !seen_families.contains(family.name()))
        .collect();
    if let Err(e) = encoder.encode(&global_filtered, &mut buffer) {
        return Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("Failed to encode global metrics: {e}"),
        ));
    }

    let response = String::from_utf8(buffer).map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("Failed to convert metrics to string: {e}"),
        )
    })?;

    Response::builder()
        .status(StatusCode::OK)
        .header("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
        .body(response)
        .map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("Failed to build response: {e}"),
            )
        })
}
