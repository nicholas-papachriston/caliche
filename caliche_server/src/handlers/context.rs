//! Context management API handlers.
//!
//! These handlers provide workspace awareness and context tracking
//! for agent sessions.

use crate::handlers::sessions::SessionsState;
use axum::{
    Json,
    extract::{Path, State},
    http::StatusCode,
};
use caliche::harness::{ContextItem, GitContext};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Response for workspace info
#[derive(Debug, Serialize)]
pub struct WorkspaceInfoResponse {
    pub workspace_root: String,
    pub project_type: Option<String>,
    pub git_context: Option<GitContextResponse>,
    pub tracked_files: Vec<TrackedFileInfo>,
}

/// Git context info
#[derive(Debug, Serialize)]
pub struct GitContextResponse {
    pub branch: String,
    pub has_changes: bool,
    pub staged_count: usize,
    pub modified_count: usize,
    pub remote_url: Option<String>,
}

impl From<GitContext> for GitContextResponse {
    fn from(ctx: GitContext) -> Self {
        Self {
            branch: ctx.branch,
            has_changes: ctx.has_changes,
            staged_count: ctx.staged_count,
            modified_count: ctx.modified_count,
            remote_url: ctx.remote_url,
        }
    }
}

/// Tracked file info
#[derive(Debug, Serialize)]
pub struct TrackedFileInfo {
    pub path: String,
    pub size_bytes: u64,
    pub language: Option<String>,
    pub modified: bool,
    pub read_count: usize,
}

/// Request to track a file
#[derive(Debug, Deserialize)]
pub struct TrackFileRequest {
    pub path: String,
}

/// Response for context items
#[derive(Debug, Serialize)]
pub struct ContextItemsResponse {
    pub items: Vec<ContextItemInfo>,
}

/// Context item info
#[derive(Debug, Serialize)]
pub struct ContextItemInfo {
    pub item_type: String,
    pub content: String,
    pub relevance: f32,
    pub source: Option<String>,
}

impl From<ContextItem> for ContextItemInfo {
    fn from(item: ContextItem) -> Self {
        Self {
            item_type: format!("{:?}", item.item_type),
            content: item.content,
            relevance: item.relevance,
            source: item.source.map(|p| p.display().to_string()),
        }
    }
}

/// GET /api/agent/sessions/{id}/context - Get session context info
pub async fn get_context_handler(
    State(adapter): State<SessionsState>,
    Path(session_id): Path<String>,
) -> Result<Json<WorkspaceInfoResponse>, (StatusCode, String)> {
    let session = adapter.get_session(&session_id).await.ok_or_else(|| {
        (
            StatusCode::NOT_FOUND,
            format!("Session not found: {}", session_id),
        )
    })?;

    let state = session.read().await;
    let context_manager = state.harness.read().await.context_manager();
    let ctx = context_manager.read().await;

    let tracked_files: Vec<TrackedFileInfo> = ctx
        .tracked_files()
        .map(|fc| TrackedFileInfo {
            path: fc.path.display().to_string(),
            size_bytes: fc.size_bytes,
            language: fc.language.clone(),
            modified: fc.modified,
            read_count: fc.read_count,
        })
        .collect();

    let project_type = ctx.project_type().map(|pt| format!("{:?}", pt));
    let git_context = ctx.git_context().cloned().map(GitContextResponse::from);

    Ok(Json(WorkspaceInfoResponse {
        workspace_root: ctx.workspace_root().display().to_string(),
        project_type,
        git_context,
        tracked_files,
    }))
}

/// POST /api/agent/sessions/{id}/context/track - Track a file
pub async fn track_file_handler(
    State(adapter): State<SessionsState>,
    Path(session_id): Path<String>,
    Json(request): Json<TrackFileRequest>,
) -> Result<Json<TrackedFileInfo>, (StatusCode, String)> {
    let session = adapter.get_session(&session_id).await.ok_or_else(|| {
        (
            StatusCode::NOT_FOUND,
            format!("Session not found: {}", session_id),
        )
    })?;

    let state = session.read().await;
    let context_manager = state.harness.read().await.context_manager();
    let mut ctx = context_manager.write().await;

    let path = PathBuf::from(&request.path);
    let file_context = ctx
        .track_file(&path)
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    Ok(Json(TrackedFileInfo {
        path: file_context.path.display().to_string(),
        size_bytes: file_context.size_bytes,
        language: file_context.language,
        modified: file_context.modified,
        read_count: file_context.read_count,
    }))
}

/// POST /api/agent/sessions/{id}/context/refresh-git - Refresh git context
pub async fn refresh_git_handler(
    State(adapter): State<SessionsState>,
    Path(session_id): Path<String>,
) -> Result<Json<Option<GitContextResponse>>, (StatusCode, String)> {
    let session = adapter.get_session(&session_id).await.ok_or_else(|| {
        (
            StatusCode::NOT_FOUND,
            format!("Session not found: {}", session_id),
        )
    })?;

    let state = session.read().await;
    let context_manager = state.harness.read().await.context_manager();
    let mut ctx = context_manager.write().await;

    let git_context = ctx
        .refresh_git_context()
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?
        .map(GitContextResponse::from);

    Ok(Json(git_context))
}

/// POST /api/agent/sessions/{id}/context/detect-project - Detect project type
pub async fn detect_project_handler(
    State(adapter): State<SessionsState>,
    Path(session_id): Path<String>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    let session = adapter.get_session(&session_id).await.ok_or_else(|| {
        (
            StatusCode::NOT_FOUND,
            format!("Session not found: {}", session_id),
        )
    })?;

    let state = session.read().await;
    let context_manager = state.harness.read().await.context_manager();
    let mut ctx = context_manager.write().await;

    let project_type = ctx
        .detect_project()
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    Ok(Json(serde_json::json!({
        "project_type": format!("{:?}", project_type)
    })))
}

/// GET /api/agent/sessions/{id}/context/relevant - Get relevant context for a query
#[derive(Debug, Deserialize)]
pub struct RelevantContextQuery {
    pub query: String,
}

pub async fn get_relevant_context_handler(
    State(adapter): State<SessionsState>,
    Path(session_id): Path<String>,
    axum::extract::Query(query): axum::extract::Query<RelevantContextQuery>,
) -> Result<Json<ContextItemsResponse>, (StatusCode, String)> {
    let session = adapter.get_session(&session_id).await.ok_or_else(|| {
        (
            StatusCode::NOT_FOUND,
            format!("Session not found: {}", session_id),
        )
    })?;

    let state = session.read().await;
    let context_manager = state.harness.read().await.context_manager();
    let ctx = context_manager.read().await;

    let items: Vec<ContextItemInfo> = ctx
        .get_relevant_context(&query.query)
        .into_iter()
        .map(ContextItemInfo::from)
        .collect();

    Ok(Json(ContextItemsResponse { items }))
}
