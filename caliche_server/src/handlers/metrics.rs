//! Tool and cache metrics API handlers.

use crate::AppState;
use axum::extract::{Path, Query, State};
use axum::http::StatusCode;
use axum::response::Json;
use caliche::observability::{CacheMetricsSnapshot, ToolStats};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tracing::Instrument;

#[derive(Debug, Deserialize)]
pub struct MetricsQuery {
    #[serde(default)]
    pub tool: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct ToolMetricsResponse {
    pub tools: HashMap<String, ToolMetricsDto>,
}

#[derive(Debug, Serialize)]
pub struct ToolMetricsDto {
    pub total_calls: u64,
    pub success_rate: f64,
    pub avg_duration_ms: f64,
    pub p95_duration_ms: Option<u64>, // Placeholder - would need histogram for real p95
    pub error_types: HashMap<String, u64>,
    pub last_executed: Option<String>,
}

impl From<ToolStats> for ToolMetricsDto {
    fn from(stats: ToolStats) -> Self {
        Self {
            total_calls: stats.total_calls,
            success_rate: stats.success_rate(),
            avg_duration_ms: stats.avg_duration_ms(),
            p95_duration_ms: None, // Would need histogram data for accurate p95
            error_types: stats.error_types,
            last_executed: stats.last_executed.map(|dt| dt.to_rfc3339()),
        }
    }
}

/// GET /api/metrics/tools - Get metrics for all tools
pub async fn get_tools_metrics_handler(
    State(state): State<AppState>,
    Query(query): Query<MetricsQuery>,
) -> Result<Json<ToolMetricsResponse>, (StatusCode, String)> {
    let span = tracing::span!(tracing::Level::INFO, "metrics_tools");

    async move {
        let all_stats = state.tool_metrics.get_all_stats().await;

        let mut tools_map = HashMap::new();

        if let Some(tool_name) = query.tool {
            // Return specific tool metrics
            if let Some(stats) = all_stats.get(&tool_name) {
                tools_map.insert(tool_name, ToolMetricsDto::from(stats.clone()));
            }
        } else {
            // Return all tools metrics
            for (name, stats) in all_stats {
                tools_map.insert(name, ToolMetricsDto::from(stats));
            }
        }

        Ok(Json(ToolMetricsResponse { tools: tools_map }))
    }
    .instrument(span)
    .await
}

/// GET /api/metrics/tools/:name - Get metrics for a specific tool
pub async fn get_tool_metrics_handler(
    State(state): State<AppState>,
    Path(tool_name): Path<String>,
) -> Result<Json<ToolMetricsDto>, (StatusCode, String)> {
    let span = tracing::span!(tracing::Level::INFO, "metrics_tool", tool = %tool_name);

    async move {
        match state.tool_metrics.get_tool_stats(&tool_name).await {
            Some(stats) => Ok(Json(ToolMetricsDto::from(stats))),
            None => Err((
                StatusCode::NOT_FOUND,
                format!("No metrics found for tool: {}", tool_name),
            )),
        }
    }
    .instrument(span)
    .await
}

/// GET /api/metrics/cache - Get cache performance metrics
pub async fn get_cache_metrics_handler(
    State(state): State<AppState>,
) -> Json<CacheMetricsSnapshot> {
    let span = tracing::span!(tracing::Level::INFO, "metrics_cache");
    let _guard = span.enter();

    Json(state.cache_metrics.snapshot())
}
