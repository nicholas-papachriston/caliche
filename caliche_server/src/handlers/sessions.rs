//! Session management API handlers.
//!
//! These handlers provide session-based agent interactions with
//! SSE event streaming support.

use crate::adapter::{ApiAdapter, ApiAdapterError, SessionInfo};
use crate::handlers::chat_contracts::{core_mode_to_dto, dto_mode_value_to_core};
use crate::middleware::OptionalUserContext;
use crate::middleware::session_auth::owner_from_user;
use axum::{
    extract::{Path, Query, State},
    http::{HeaderMap, StatusCode, header},
    response::{
        Html, IntoResponse, Json, Response,
        sse::{Event, Sse},
    },
};
use caliche::agent::session::{ShareVisibility, render_share_html};
use caliche::harness::SessionConfig;
use caliche::harness::diff_contracts::{
    diff_response_from_summary, redo_response_from_noop, redo_response_from_success,
    undo_response_from_noop, undo_response_from_success,
};
use caliche_contracts::metrics::HarnessDisciplineMetricsDto;
use caliche_contracts::sessions::normalize_session_title;
pub use caliche_contracts::sessions::{
    CheckpointResponseDto as CheckpointResponse, CreateSessionRequestDto as CreateSessionRequest,
    CreateSessionResponseDto as CreateSessionResponse, DiffResponseDto as DiffResponse,
    ModeResponseDto as ModeResponse, RedoResponseDto as RedoResponse,
    RollbackResponseDto as RollbackResponse, ShareCreateRequestDto as ShareCreateRequest,
    ShareCreateResponseDto as ShareCreateResponse, ShareResolveResponseDto as ShareResolveResponse,
    ShareVisibilityDto, UndoResponseDto as UndoResponse,
};
use caliche_contracts::settings::InteractionModeDto;
use futures_util::StreamExt;
use futures_util::stream::Stream;
use serde::{Deserialize, Serialize};
use std::convert::Infallible;
use std::sync::Arc;
use std::time::Duration;

/// Request to execute a turn.
///
/// `provider` and `model` are the REPL's current selection. When set, the live
/// session is retargeted before this turn and later turns on that id keep it.
#[derive(Debug, Deserialize)]
pub struct ExecuteTurnRequest {
    pub input: String,
    #[serde(default)]
    pub provider: Option<String>,
    #[serde(default)]
    pub model: Option<String>,
    /// An explicit context window for this session, from `/context-window`.
    #[serde(default)]
    pub context_window: Option<usize>,
}

/// Response for turn execution
#[derive(Debug, Serialize)]
pub struct TurnResponse {
    pub response: String,
    pub turn_number: usize,
    pub tools_used: bool,
    pub duration_ms: u64,
    pub usage: UsageInfo,
    pub discipline: DisciplineResponse,
}

/// Token usage information
#[derive(Debug, Serialize)]
pub struct UsageInfo {
    pub prompt_tokens: Option<u64>,
    pub completion_tokens: Option<u64>,
    pub total_tokens: Option<u64>,
}

/// Harness discipline metrics for turn/task responses.
pub type DisciplineResponse = HarnessDisciplineMetricsDto;

/// Request to execute a task
#[derive(Debug, Deserialize)]
pub struct ExecuteTaskRequest {
    pub task: String,
}

/// Response for task execution
#[derive(Debug, Serialize)]
pub struct TaskResponse {
    pub response: String,
    pub turns: usize,
    pub duration_ms: u64,
    pub files_modified: Vec<String>,
    pub checkpoint_id: Option<String>,
    pub total_usage: UsageInfo,
    pub discipline: DisciplineResponse,
}

/// Request to rollback
#[derive(Debug, Deserialize)]
pub struct RollbackRequest {
    pub checkpoint_id: String,
}

/// Request to create checkpoint
#[derive(Debug, Deserialize)]
pub struct CheckpointRequest {
    pub label: String,
}

/// State containing the API adapter
pub type SessionsState = Arc<ApiAdapter>;

pub(crate) fn map_adapter_error(error: anyhow::Error) -> (StatusCode, String) {
    if let Some(ApiAdapterError::SessionNotFound(session_id)) =
        error.downcast_ref::<ApiAdapterError>()
    {
        return (
            StatusCode::NOT_FOUND,
            format!("Session not found: {session_id}"),
        );
    }
    if let Some(ApiAdapterError::TurnCancelled(session_id)) =
        error.downcast_ref::<ApiAdapterError>()
    {
        return (
            StatusCode::CONFLICT,
            format!("Session turn cancelled: {session_id}"),
        );
    }
    let message = crate::redact::redact_secrets(&format!("{error:#}"));
    if message.contains("session turn cancelled") || message.contains("Stream cancelled by user") {
        return (StatusCode::CONFLICT, message);
    }
    if message.contains("invalid provider") {
        return (StatusCode::BAD_REQUEST, message);
    }
    if message.contains("sharing is disabled") {
        return (StatusCode::FORBIDDEN, message);
    }
    (StatusCode::INTERNAL_SERVER_ERROR, message)
}

fn visibility_to_dto(visibility: ShareVisibility) -> ShareVisibilityDto {
    match visibility {
        ShareVisibility::Public => ShareVisibilityDto::Public,
        ShareVisibility::Authenticated => ShareVisibilityDto::Authenticated,
    }
}

fn visibility_from_dto(visibility: ShareVisibilityDto) -> ShareVisibility {
    match visibility {
        ShareVisibilityDto::Public => ShareVisibility::Public,
        ShareVisibilityDto::Authenticated => ShareVisibility::Authenticated,
    }
}

fn authorize_share_resolve(
    visibility: ShareVisibility,
    user: &OptionalUserContext,
) -> Result<(), (StatusCode, String)> {
    match visibility {
        ShareVisibility::Public => Ok(()),
        ShareVisibility::Authenticated => {
            if user.0.is_some() {
                Ok(())
            } else {
                Err((
                    StatusCode::UNAUTHORIZED,
                    "authenticated share requires a valid principal".to_string(),
                ))
            }
        }
    }
}

fn share_resolve_response(record: caliche::agent::session::SharePayload) -> ShareResolveResponse {
    ShareResolveResponse {
        slug: record.slug,
        session_id: record.session_id,
        created_at: record.created_at.to_rfc3339(),
        messages: record.messages,
        redaction_count: record.redaction_count,
        visibility: visibility_to_dto(record.visibility),
    }
}

/// Response for `POST /api/agent/sessions/{id}/cancel` and `.../stop`.
#[derive(Debug, Serialize)]
pub struct CancelSessionResponse {
    pub session_id: String,
    pub cancelled: bool,
}

/// GET /api/agent/sessions — every live session in this process.
///
/// Returns `200` and `[]` when the in-memory cache is empty. The array is the
/// process-wide cache (`ApiAdapter` / `AgentRuntime`), not a per-caller view.
/// It is not filtered by JWT `sub` or roles. When JWT or require-auth is on,
/// the route uses the same auth middleware as the other session routes; that
/// gate authorizes the call and does not filter the rows.
pub async fn list_sessions_handler(State(adapter): State<SessionsState>) -> Json<Vec<SessionInfo>> {
    Json(adapter.list_process_sessions().await)
}

/// POST /api/agent/sessions - Create a new session
pub async fn create_session_handler(
    State(adapter): State<SessionsState>,
    OptionalUserContext(user): OptionalUserContext,
    Json(request): Json<CreateSessionRequest>,
) -> Result<Json<CreateSessionResponse>, (StatusCode, String)> {
    let session_config = SessionConfig {
        session_id: None,
        workspace: request.workspace.map(std::path::PathBuf::from),
        context_files: request
            .context_files
            .into_iter()
            .map(std::path::PathBuf::from)
            .collect(),
        system_prompt_addition: request.system_prompt,
        max_turns: request.max_turns,
        history: Vec::new(),
    };

    let owner = user.as_ref().map(owner_from_user);
    let title = normalize_session_title(request.title.as_deref())
        .map_err(|err| (StatusCode::BAD_REQUEST, err.to_string()))?;
    match Box::pin(adapter.create_session(Some(session_config), owner, title)).await {
        Ok(session_id) => Ok(Json(CreateSessionResponse {
            session_id,
            created_at: chrono::Utc::now().to_rfc3339(),
        })),
        Err(e) => Err((StatusCode::INTERNAL_SERVER_ERROR, e.to_string())),
    }
}

/// GET /api/agent/sessions/{id} - Get session info
pub async fn get_session_handler(
    State(adapter): State<SessionsState>,
    Path(session_id): Path<String>,
) -> Result<Json<SessionInfo>, (StatusCode, String)> {
    match adapter.get_session_info(&session_id).await {
        Some(info) => Ok(Json(info)),
        None => Err((
            StatusCode::NOT_FOUND,
            format!("Session not found: {}", session_id),
        )),
    }
}

/// POST /api/agent/sessions/{id}/cancel and /stop.
///
/// Signals the in-flight turn (cancel flag + harness hard-stop) and removes the
/// session. The next turn on that id is `404`. Idle time-to-idle eviction is
/// not this teardown.
///
/// Auth matches turn and task: the session router middleware. With auth off,
/// any caller who can reach the process can cancel any session id. A pasted
/// UI bearer is an operator credential.
pub async fn cancel_session_handler(
    State(adapter): State<SessionsState>,
    Path(session_id): Path<String>,
) -> Result<Json<CancelSessionResponse>, (StatusCode, String)> {
    if adapter.cancel_session(&session_id).await {
        Ok(Json(CancelSessionResponse {
            session_id,
            cancelled: true,
        }))
    } else {
        Err((
            StatusCode::NOT_FOUND,
            format!("Session not found: {session_id}"),
        ))
    }
}

/// DELETE /api/agent/sessions/{id} - Delete a session
pub async fn delete_session_handler(
    State(adapter): State<SessionsState>,
    Path(session_id): Path<String>,
) -> Result<StatusCode, (StatusCode, String)> {
    if adapter.delete_session(&session_id).await {
        Ok(StatusCode::NO_CONTENT)
    } else {
        Err((
            StatusCode::NOT_FOUND,
            format!("Session not found: {}", session_id),
        ))
    }
}

/// POST /api/agent/sessions/{id}/turn - Execute a turn
pub async fn execute_turn_handler(
    State(adapter): State<SessionsState>,
    Path(session_id): Path<String>,
    Json(request): Json<ExecuteTurnRequest>,
) -> Result<Json<TurnResponse>, (StatusCode, String)> {
    match adapter
        .execute_turn_with_selection(
            &session_id,
            &request.input,
            request.provider.as_deref(),
            request.model.as_deref(),
            request.context_window,
        )
        .await
    {
        Ok(result) => {
            // Get the turn count from the session after execution
            let turn_number = adapter
                .get_session_info(&session_id)
                .await
                .map(|info| info.turn_count)
                .unwrap_or(0);

            Ok(Json(TurnResponse {
                response: result.response,
                turn_number,
                tools_used: result.tools_used,
                duration_ms: result.duration_ms,
                usage: UsageInfo {
                    prompt_tokens: result.usage.prompt_tokens.map(|t| t as u64),
                    completion_tokens: result.usage.completion_tokens.map(|t| t as u64),
                    total_tokens: result.usage.total_tokens.map(|t| t as u64),
                },
                discipline: HarnessDisciplineMetricsDto::from(&result.discipline),
            }))
        }
        Err(e) => Err(super::turn_failure::reject_failed_turn(&adapter, &session_id, e).await),
    }
}

/// POST /api/agent/sessions/{id}/task - Execute a complete task
pub async fn execute_task_handler(
    State(adapter): State<SessionsState>,
    Path(session_id): Path<String>,
    Json(request): Json<ExecuteTaskRequest>,
) -> Result<Json<TaskResponse>, (StatusCode, String)> {
    match adapter.execute_task(&session_id, &request.task).await {
        Ok(result) => Ok(Json(TaskResponse {
            response: result.response,
            turns: result.turns,
            duration_ms: result.duration_ms,
            files_modified: result
                .files_modified
                .iter()
                .map(|p| p.display().to_string())
                .collect(),
            checkpoint_id: result.checkpoint_id,
            total_usage: UsageInfo {
                prompt_tokens: result.total_usage.prompt_tokens.map(|t| t as u64),
                completion_tokens: result.total_usage.completion_tokens.map(|t| t as u64),
                total_tokens: result.total_usage.total_tokens.map(|t| t as u64),
            },
            discipline: HarnessDisciplineMetricsDto::from(&result.discipline),
        })),
        Err(e) => Err(map_adapter_error(e)),
    }
}

/// GET /api/agent/sessions/{id}/events - SSE event stream
pub async fn session_events_handler(
    State(adapter): State<SessionsState>,
    Path(session_id): Path<String>,
) -> Result<Sse<impl Stream<Item = Result<Event, Infallible>>>, (StatusCode, String)> {
    let event_rx = adapter
        .subscribe_events(&session_id)
        .await
        .map_err(map_adapter_error)?;

    let stream = super::session_sse::session_event_frames(event_rx)
        .map(|frame| Ok::<Event, Infallible>(frame.into_event()));

    Ok(Sse::new(stream).keep_alive(
        axum::response::sse::KeepAlive::new()
            .interval(Duration::from_secs(15))
            .text("keep-alive"),
    ))
}

/// POST /api/agent/sessions/{id}/rollback - Rollback to checkpoint
pub async fn rollback_handler(
    State(adapter): State<SessionsState>,
    Path(session_id): Path<String>,
    Json(request): Json<RollbackRequest>,
) -> Result<Json<RollbackResponse>, (StatusCode, String)> {
    match adapter.rollback(&session_id, &request.checkpoint_id).await {
        Ok(files) => Ok(Json(RollbackResponse {
            files_restored: files.iter().map(|p| p.display().to_string()).collect(),
        })),
        Err(e) => Err(map_adapter_error(e)),
    }
}

/// POST /api/agent/sessions/{id}/checkpoint - Create checkpoint
pub async fn checkpoint_handler(
    State(adapter): State<SessionsState>,
    Path(session_id): Path<String>,
    Json(request): Json<CheckpointRequest>,
) -> Result<Json<CheckpointResponse>, (StatusCode, String)> {
    match adapter.checkpoint(&session_id, &request.label).await {
        Ok(checkpoint_id) => Ok(Json(CheckpointResponse { checkpoint_id })),
        Err(e) => Err(map_adapter_error(e)),
    }
}

pub async fn undo_handler(
    State(adapter): State<SessionsState>,
    Path(session_id): Path<String>,
) -> Result<Json<UndoResponse>, (StatusCode, String)> {
    match adapter.undo_last(&session_id).await {
        Ok(Some(diff)) => {
            let summary = adapter
                .session_diffs(&session_id, None)
                .await
                .map_err(map_adapter_error)?;
            Ok(Json(undo_response_from_success(&diff, &summary)))
        }
        Ok(None) => {
            let summary = adapter
                .session_diffs(&session_id, None)
                .await
                .map_err(map_adapter_error)?;
            Ok(Json(undo_response_from_noop("Nothing to undo", &summary)))
        }
        Err(e) => Err(map_adapter_error(e)),
    }
}

/// Query for GET .../diff
#[derive(Debug, Deserialize)]
pub struct DiffQuery {
    pub message_id: Option<uuid::Uuid>,
}

/// GET /api/agent/sessions/{id}/diff - List session diffs (optionally filter by message_id)
pub async fn diff_handler(
    State(adapter): State<SessionsState>,
    Path(session_id): Path<String>,
    Query(query): Query<DiffQuery>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    match adapter.session_diffs(&session_id, query.message_id).await {
        Ok(summary) => Ok(Json(serde_json::json!(diff_response_from_summary(
            &summary
        )))),
        Err(e) => Err(map_adapter_error(e)),
    }
}

/// POST /api/agent/sessions/{id}/revert - Revert last (or given) message's file changes
#[derive(Debug, Deserialize)]
pub struct RevertRequest {
    pub message_id: Option<uuid::Uuid>,
}

pub async fn revert_handler(
    State(adapter): State<SessionsState>,
    Path(session_id): Path<String>,
    Json(body): Json<RevertRequest>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    if let Some(message_id) = body.message_id {
        match adapter.undo_to_message(&session_id, message_id).await {
            Ok(diffs) if diffs.is_empty() => Ok(Json(serde_json::json!({
                "reverted": false,
                "message": "No matching message_id to revert"
            }))),
            Ok(diffs) => {
                let files_restored: usize = diffs.iter().map(|d| d.snapshots.len()).sum();
                let files: Vec<String> = diffs
                    .iter()
                    .flat_map(|d| d.snapshots.iter().map(|s| s.path.display().to_string()))
                    .collect();
                let message_ids: Vec<String> =
                    diffs.iter().map(|d| d.message_id.to_string()).collect();
                Ok(Json(serde_json::json!({
                    "reverted": true,
                    "message_id": message_id.to_string(),
                    "reverted_message_ids": message_ids,
                    "files_restored": files_restored,
                    "files": files,
                })))
            }
            Err(e) => Err(map_adapter_error(e)),
        }
    } else {
        match adapter.undo_last(&session_id).await {
            Ok(Some(diff)) => Ok(Json(serde_json::json!({
                "reverted": true,
                "message_id": diff.message_id.to_string(),
                "files_restored": diff.snapshots.len(),
                "files": diff
                    .snapshots
                    .iter()
                    .map(|snap| snap.path.display().to_string())
                    .collect::<Vec<_>>()
            }))),
            Ok(None) => Ok(Json(serde_json::json!({
                "reverted": false,
                "message": "Nothing to revert"
            }))),
            Err(e) => Err(map_adapter_error(e)),
        }
    }
}

pub async fn unrevert_handler(
    State(adapter): State<SessionsState>,
    Path(session_id): Path<String>,
) -> Result<Json<RedoResponse>, (StatusCode, String)> {
    match adapter.redo_last(&session_id).await {
        Ok(Some(diff)) => {
            let summary = adapter
                .session_diffs(&session_id, None)
                .await
                .map_err(map_adapter_error)?;
            Ok(Json(redo_response_from_success(&diff, &summary)))
        }
        Ok(None) => {
            let summary = adapter
                .session_diffs(&session_id, None)
                .await
                .map_err(map_adapter_error)?;
            Ok(Json(redo_response_from_noop("Nothing to redo", &summary)))
        }
        Err(e) => Err(map_adapter_error(e)),
    }
}

/// Body for `POST /api/agent/sessions/{id}/history`.
#[derive(Debug, Deserialize)]
pub struct RestoreHistoryRequest {
    pub messages: Vec<caliche::messages::Message>,
}

/// How many messages the resumed session will send on the next turn.
#[derive(Debug, Serialize)]
pub struct RestoreHistoryResponse {
    pub messages: usize,
    pub summarized: bool,
}

/// POST /api/agent/sessions/{id}/history - Reload a saved transcript into the model context.
pub async fn restore_history_handler(
    State(adapter): State<SessionsState>,
    Path(session_id): Path<String>,
    Json(request): Json<RestoreHistoryRequest>,
) -> Result<Json<RestoreHistoryResponse>, (StatusCode, String)> {
    let fitted = adapter
        .restore_resume_context(&session_id, request.messages)
        .await
        .map_err(map_adapter_error)?;
    Ok(Json(RestoreHistoryResponse {
        messages: fitted.messages.len(),
        summarized: fitted.summarized,
    }))
}

/// POST /api/agent/sessions/{id}/clear - Clear conversation history
pub async fn clear_history_handler(
    State(adapter): State<SessionsState>,
    Path(session_id): Path<String>,
) -> Result<StatusCode, (StatusCode, String)> {
    match adapter.clear_history(&session_id).await {
        Ok(()) => Ok(StatusCode::NO_CONTENT),
        Err(e) => Err(map_adapter_error(e)),
    }
}

/// POST /api/agent/sessions/{id}/share - Create a shareable slug for a session.
pub async fn share_handler(
    State(adapter): State<SessionsState>,
    Path(session_id): Path<String>,
    Json(request): Json<ShareCreateRequest>,
) -> Result<Json<ShareCreateResponse>, (StatusCode, String)> {
    let visibility = request.visibility.map(visibility_from_dto);
    let record = adapter
        .share_session_with_visibility(&session_id, visibility)
        .await
        .map_err(map_adapter_error)?;
    let url = adapter.public_share_url(&record.slug);
    Ok(Json(ShareCreateResponse {
        slug: record.slug,
        session_id: record.session_id,
        created_at: record.created_at.to_rfc3339(),
        url,
        redaction_count: record.redaction_count,
        visibility: visibility_to_dto(record.visibility),
    }))
}

/// GET /api/agent/shares/{slug} - Resolve a share slug to a redacted transcript (JSON).
pub async fn get_share_handler(
    State(adapter): State<SessionsState>,
    Path(slug): Path<String>,
    user: OptionalUserContext,
) -> Result<Json<ShareResolveResponse>, (StatusCode, String)> {
    match adapter.get_share(&slug) {
        Some(record) => {
            authorize_share_resolve(record.visibility, &user)?;
            Ok(Json(share_resolve_response(record)))
        }
        None => Err((StatusCode::NOT_FOUND, "Share not found".to_string())),
    }
}

/// GET /s/{slug} - Public share view (HTML by default, JSON when Accept prefers JSON).
pub async fn public_share_handler(
    State(adapter): State<SessionsState>,
    Path(slug): Path<String>,
    headers: HeaderMap,
    user: OptionalUserContext,
) -> Result<Response, (StatusCode, String)> {
    let Some(record) = adapter.get_share(&slug) else {
        return Err((StatusCode::NOT_FOUND, "Share not found".to_string()));
    };
    authorize_share_resolve(record.visibility, &user)?;
    let prefer_json = headers
        .get(header::ACCEPT)
        .and_then(|value| value.to_str().ok())
        .is_some_and(|accept| {
            let accept = accept.to_ascii_lowercase();
            accept.contains("application/json") && !accept.contains("text/html")
        });
    if prefer_json {
        return Ok(Json(share_resolve_response(record)).into_response());
    }
    Ok(Html(render_share_html(&record)).into_response())
}

/// DELETE /api/agent/sessions/{id}/share - Remove shares for a session.
pub async fn unshare_handler(
    State(adapter): State<SessionsState>,
    Path(session_id): Path<String>,
) -> Result<StatusCode, (StatusCode, String)> {
    adapter.delete_session_share(&session_id);
    Ok(StatusCode::NO_CONTENT)
}

/// DELETE /api/agent/shares/{slug} - Remove one share slug.
pub async fn delete_share_handler(
    State(adapter): State<SessionsState>,
    Path(slug): Path<String>,
) -> Result<StatusCode, (StatusCode, String)> {
    if adapter.delete_share_slug(&slug) {
        Ok(StatusCode::NO_CONTENT)
    } else {
        Err((StatusCode::NOT_FOUND, "Share not found".to_string()))
    }
}

// ============================================================================
// Mode Management
// ============================================================================

/// Request to set session mode
#[derive(Debug, Deserialize)]
pub struct SetModeRequest {
    pub mode: InteractionModeDto,
}

/// GET /api/agent/sessions/{id}/mode - Get current session mode
pub async fn get_session_mode_handler(
    State(adapter): State<SessionsState>,
    Path(session_id): Path<String>,
) -> Result<Json<ModeResponse>, (StatusCode, String)> {
    // Check if session exists
    if adapter.get_session_info(&session_id).await.is_none() {
        return Err((
            StatusCode::NOT_FOUND,
            format!("Session not found: {}", session_id),
        ));
    }

    // For now, sessions default to Agent mode
    // In a full implementation, mode would be stored in the session
    let mode = caliche::agent::modes::InteractionMode::Agent;

    Ok(Json(ModeResponse {
        mode: core_mode_to_dto(mode),
        write_tools_enabled: !mode.is_readonly(),
        description: mode.description().to_string(),
    }))
}

/// POST /api/agent/sessions/{id}/mode - Set session mode
pub async fn set_session_mode_handler(
    State(adapter): State<SessionsState>,
    Path(session_id): Path<String>,
    Json(request): Json<SetModeRequest>,
) -> Result<Json<ModeResponse>, (StatusCode, String)> {
    // Check if session exists
    if adapter.get_session_info(&session_id).await.is_none() {
        return Err((
            StatusCode::NOT_FOUND,
            format!("Session not found: {}", session_id),
        ));
    }

    // In a full implementation, this would update the session's mode
    // For now, we acknowledge the request and return the mode info
    let mode = dto_mode_value_to_core(request.mode);

    tracing::info!("Session {} mode set to: {}", session_id, mode);

    Ok(Json(ModeResponse {
        mode: core_mode_to_dto(mode),
        write_tools_enabled: !mode.is_readonly(),
        description: mode.description().to_string(),
    }))
}

// ============================================================================
// Plan Document Management
// ============================================================================

/// Response for plan operations
#[derive(Debug, Serialize)]
pub struct PlanResponse {
    pub title: Option<String>,
    pub requirements: Vec<String>,
    pub clarifications: Vec<caliche::agent::modes::Clarification>,
    pub tasks: Vec<caliche::agent::modes::PlanTask>,
    pub notes: Vec<String>,
    pub markdown: String,
}

/// GET /api/agent/sessions/{id}/plan - Get session plan document
pub async fn get_session_plan_handler(
    State(adapter): State<SessionsState>,
    Path(session_id): Path<String>,
) -> Result<Json<PlanResponse>, (StatusCode, String)> {
    // Check if session exists
    if adapter.get_session_info(&session_id).await.is_none() {
        return Err((
            StatusCode::NOT_FOUND,
            format!("Session not found: {}", session_id),
        ));
    }

    // In a full implementation, this would retrieve the plan from session state
    // For now, return an empty plan
    let plan = caliche::agent::modes::PlanDocument::new();

    Ok(Json(PlanResponse {
        title: plan.title.clone(),
        requirements: plan.requirements.clone(),
        clarifications: plan.clarifications.clone(),
        tasks: plan.tasks.clone(),
        notes: plan.notes.clone(),
        markdown: plan.to_markdown(),
    }))
}

/// DELETE /api/agent/sessions/{id}/plan - Clear session plan document
pub async fn clear_session_plan_handler(
    State(adapter): State<SessionsState>,
    Path(session_id): Path<String>,
) -> Result<StatusCode, (StatusCode, String)> {
    // Check if session exists
    if adapter.get_session_info(&session_id).await.is_none() {
        return Err((
            StatusCode::NOT_FOUND,
            format!("Session not found: {}", session_id),
        ));
    }

    // In a full implementation, this would clear the plan from session state
    tracing::info!("Session {} plan cleared", session_id);

    Ok(StatusCode::NO_CONTENT)
}

#[cfg(test)]
mod title_tests {
    use super::{
        CreateSessionRequest, OptionalUserContext, create_session_handler, list_sessions_handler,
    };
    use crate::adapter::ApiAdapter;
    use axum::Json;
    use axum::extract::State;
    use axum::http::StatusCode;
    use caliche::harness::HarnessConfig;
    use caliche_contracts::sessions::SESSION_TITLE_MAX_CHARS;
    use std::sync::Arc;
    use std::time::Duration;

    #[tokio::test]
    #[serial_test::serial]
    async fn optional_title_is_listed_and_overlong_title_is_rejected() {
        #[expect(unsafe_code, reason = "takes the process env guard; see SAFETY")]
        // SAFETY: this test is serial and starts no thread that reads the environment while the guard lives.
        let mut env = unsafe { caliche_test_env::EnvGuard::acquire() };
        env.set("OPENROUTER_API_KEY", "test-key-for-session-title");

        let adapter = Arc::new(ApiAdapter::with_session_ttl(
            HarnessConfig::default(),
            Duration::from_secs(60),
            None,
        ));

        let created = create_session_handler(
            State(Arc::clone(&adapter)),
            OptionalUserContext(None),
            Json(CreateSessionRequest {
                title: Some("  desk  ".to_string()),
                ..CreateSessionRequest::default()
            }),
        )
        .await
        .expect("titled create");

        let bare = create_session_handler(
            State(Arc::clone(&adapter)),
            OptionalUserContext(None),
            Json(CreateSessionRequest::default()),
        )
        .await
        .expect("untitled create");

        let Json(listed) = list_sessions_handler(State(Arc::clone(&adapter))).await;
        let titled = listed
            .iter()
            .find(|row| row.session_id == created.0.session_id)
            .expect("titled row");
        assert_eq!(titled.title.as_deref(), Some("desk"));
        let untitled = listed
            .iter()
            .find(|row| row.session_id == bare.0.session_id)
            .expect("untitled row");
        assert!(untitled.title.is_none());

        let overlong = "x".repeat(SESSION_TITLE_MAX_CHARS + 1);
        let rejected = create_session_handler(
            State(adapter),
            OptionalUserContext(None),
            Json(CreateSessionRequest {
                title: Some(overlong),
                ..CreateSessionRequest::default()
            }),
        )
        .await
        .expect_err("overlong title");
        assert_eq!(rejected.0, StatusCode::BAD_REQUEST);
        assert!(rejected.1.contains("80"));
    }
}

#[cfg(test)]
mod resume_history_tests {
    use super::{RestoreHistoryRequest, restore_history_handler};
    use crate::adapter::ApiAdapter;
    use axum::Json;
    use axum::extract::{Path, State};
    use caliche::harness::HarnessConfig;
    use caliche::memory::MemoryStrategy;
    use caliche::messages::Message;
    use std::sync::Arc;
    use std::time::Duration;

    async fn session_memory_text(adapter: &ApiAdapter, session_id: &str) -> String {
        let session = adapter.get_session(session_id).await.expect("live session");
        let memory = {
            let state = session.read().await;
            let harness = state.harness.read().await;
            harness.service().memory().expect("resume memory")
        };
        let context = memory.read().await.get_context().await.expect("context");
        context
            .iter()
            .map(Message::extract_text)
            .collect::<Vec<_>>()
            .join("\n")
    }

    #[tokio::test]
    #[serial_test::serial]
    async fn restore_history_handler_writes_request_body() {
        #[expect(unsafe_code, reason = "takes the process env guard; see SAFETY")]
        // SAFETY: this test is serial and starts no thread that reads the environment while the guard lives.
        let mut env = unsafe { caliche_test_env::EnvGuard::acquire() };
        env.set("OPENROUTER_API_KEY", "test-key-for-resume-history");

        let adapter = Arc::new(ApiAdapter::with_session_ttl(
            HarnessConfig::default(),
            Duration::from_secs(60),
            None,
        ));
        let session_id = adapter
            .create_session(None, None, None)
            .await
            .expect("session");
        let messages = vec![
            Message::user("Please rename parser.rs"),
            Message::assistant("Renamed parser.rs with the fs tool"),
        ];
        let Json(response) = restore_history_handler(
            State(Arc::clone(&adapter)),
            Path(session_id.clone()),
            Json(RestoreHistoryRequest {
                messages: messages.clone(),
            }),
        )
        .await
        .expect("restore history");
        assert_eq!(response.messages, messages.len());
        assert!(!response.summarized);

        let memory = session_memory_text(&adapter, &session_id).await;
        assert!(
            memory.contains("Please rename parser.rs"),
            "handler did not write the posted user turn: {memory}"
        );
        assert!(
            memory.contains("Renamed parser.rs with the fs tool"),
            "handler did not write the posted assistant turn: {memory}"
        );
    }
}
