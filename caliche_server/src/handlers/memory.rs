//! Memory API handlers: stats, inspect, search, clear, export, import.

use crate::adapter::ApiAdapter;
use axum::{
    extract::{Path, Query, State},
    http::StatusCode,
    response::Json,
};
use caliche::agent::memory::{MemoryInspection, MemorySnapshot, MemoryStats, MemoryStrategy};
use serde::Deserialize;
use std::sync::Arc;
use tokio::sync::RwLock;

pub type SessionsState = Arc<ApiAdapter>;

async fn session_memory(
    adapter: &ApiAdapter,
    session_id: &str,
) -> Result<Arc<RwLock<Box<dyn MemoryStrategy>>>, (StatusCode, String)> {
    let session = adapter
        .get_session(session_id)
        .await
        .ok_or_else(|| (StatusCode::NOT_FOUND, "Session not found".to_string()))?;
    let state = session.read().await;
    let harness = state.harness.read().await;
    harness.service().session().memory.clone().ok_or_else(|| {
        (
            StatusCode::NOT_FOUND,
            serde_json::json!({"error": "no memory strategy configured"}).to_string(),
        )
    })
}

/// GET /api/agent/sessions/{id}/memory/stats
pub async fn memory_stats_handler(
    State(adapter): State<SessionsState>,
    Path(session_id): Path<String>,
) -> Result<Json<MemoryStats>, (StatusCode, String)> {
    let memory = session_memory(&adapter, &session_id).await?;
    let stats = memory.read().await.stats();
    Ok(Json(stats))
}

/// GET /api/agent/sessions/{id}/memory/inspect
pub async fn memory_inspect_handler(
    State(adapter): State<SessionsState>,
    Path(session_id): Path<String>,
) -> Result<Json<MemoryInspection>, (StatusCode, String)> {
    let memory = session_memory(&adapter, &session_id).await?;
    let inspection = memory
        .read()
        .await
        .inspect()
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    Ok(Json(inspection))
}

#[derive(Debug, Deserialize)]
pub struct SearchQuery {
    pub q: Option<String>,
}

/// GET /api/agent/sessions/{id}/memory/search?q=...
pub async fn memory_search_handler(
    State(adapter): State<SessionsState>,
    Path(session_id): Path<String>,
    Query(query): Query<SearchQuery>,
) -> Result<Json<Vec<(caliche::messages::Message, f32)>>, (StatusCode, String)> {
    let memory = session_memory(&adapter, &session_id).await?;
    let q = query.q.as_deref().unwrap_or("");
    let results = memory
        .read()
        .await
        .search(q)
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    Ok(Json(results))
}

/// POST /api/agent/sessions/{id}/memory/clear
pub async fn memory_clear_handler(
    State(adapter): State<SessionsState>,
    Path(session_id): Path<String>,
) -> Result<StatusCode, (StatusCode, String)> {
    let memory = session_memory(&adapter, &session_id).await?;
    let mut guard = memory.write().await;
    guard
        .clear()
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    Ok(StatusCode::NO_CONTENT)
}

/// GET /api/agent/sessions/{id}/memory/export
pub async fn memory_export_handler(
    State(adapter): State<SessionsState>,
    Path(session_id): Path<String>,
) -> Result<Json<MemorySnapshot>, (StatusCode, String)> {
    let memory = session_memory(&adapter, &session_id).await?;
    let guard = memory.read().await;
    let messages = guard
        .get_context()
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    let stats = guard.stats();
    let snapshot = MemorySnapshot {
        messages,
        metadata: serde_json::to_value(&stats).unwrap_or(serde_json::Value::Null),
    };
    Ok(Json(snapshot))
}

/// POST /api/agent/sessions/{id}/memory/import - body: MemorySnapshot as JSON
pub async fn memory_import_handler(
    State(adapter): State<SessionsState>,
    Path(session_id): Path<String>,
    Json(snapshot): Json<MemorySnapshot>,
) -> Result<StatusCode, (StatusCode, String)> {
    let memory = session_memory(&adapter, &session_id).await?;
    let mut guard = memory.write().await;
    guard
        .clear()
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    if !snapshot.messages.is_empty() {
        guard
            .add_messages(snapshot.messages)
            .await
            .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    }
    Ok(StatusCode::NO_CONTENT)
}
