//! Enhanced Agent Streaming Handler using AgentHarness.
//!
//! This handler provides full event streaming with tool progress,
//! thinking events, and file change tracking.
//!
//! Note: task execution is spawned with `tokio::spawn` and is not automatically aborted when the
//! SSE client disconnects; the event subscription may end while the task continues until completion.
//! Prefer bounded `max_turns` for public ingress. Improving cancel-on-disconnect would require an
//! explicit cancellation token wired into `execute_task`.

use crate::AgentRouterState;
use crate::adapter::ResumedSessionItem;
use axum::{
    extract::State,
    http::StatusCode,
    response::{
        Json,
        sse::{Event, KeepAlive, Sse},
    },
};
use caliche::agent::hooks::parse_permission_mode_str;
use caliche::harness::{
    AgentEvent, BroadcastResync, HarnessConfig, SessionConfig, public_agent_sse_event_name,
};
use futures_util::StreamExt;
use futures_util::stream::Stream;
use serde::{Deserialize, Serialize};
use std::convert::Infallible;
use std::time::Duration;

/// Request for agent task execution
#[derive(Debug, Deserialize)]
pub struct AgentTaskRequest {
    /// The task to execute
    pub task: String,
    /// Provider to use (optional, uses default)
    #[serde(default)]
    pub provider: Option<String>,
    /// Model to use (optional, uses default)
    #[serde(default)]
    pub model: Option<String>,
    /// Tools to enable (optional, uses all defaults)
    #[serde(default)]
    pub tools: Option<Vec<String>>,
    /// Maximum turns for the task
    #[serde(default)]
    pub max_turns: Option<usize>,
    /// Workspace directory (optional)
    #[serde(default)]
    pub workspace: Option<String>,
    /// System prompt override (optional)
    #[serde(default)]
    pub system_prompt: Option<String>,
    /// Whether to auto-checkpoint (default: true)
    #[serde(default = "default_true")]
    pub auto_checkpoint: bool,
    /// Accept all file edits without confirmation
    #[serde(default)]
    pub accept_edits: bool,
    /// Permission mode (default, accept-edits, accept-all, deny-all, interactive)
    #[serde(default)]
    pub permission_mode: Option<String>,
}

fn default_true() -> bool {
    true
}

/// Response for task completion
#[derive(Debug, Serialize)]
pub struct AgentTaskResponse {
    pub session_id: String,
    pub response: String,
    pub turns: usize,
    pub duration_ms: u64,
    pub files_modified: Vec<String>,
    pub checkpoint_id: Option<String>,
    pub total_usage: UsageInfo,
}

/// Token usage information
#[derive(Debug, Serialize)]
pub struct UsageInfo {
    pub prompt_tokens: Option<u64>,
    pub completion_tokens: Option<u64>,
    pub total_tokens: Option<u64>,
}

/// Request for streaming agent execution
#[derive(Debug, Deserialize)]
pub struct AgentStreamRequest {
    /// The task/message to execute
    pub message: String,
    /// Provider to use
    #[serde(default)]
    pub provider: Option<String>,
    /// Model to use
    #[serde(default)]
    pub model: Option<String>,
    /// Tools to enable
    #[serde(default)]
    pub tools: Option<Vec<String>>,
    /// Maximum turns
    #[serde(default)]
    pub max_turns: Option<usize>,
    /// Workspace directory
    #[serde(default)]
    pub workspace: Option<String>,
    /// System prompt override
    #[serde(default)]
    pub system_prompt: Option<String>,
    /// Event types to include (if empty, include all)
    #[serde(default)]
    pub include_events: Vec<String>,
    /// Whether to include thinking events
    #[serde(default = "default_true")]
    pub include_thinking: bool,
    /// Whether to include tool events
    #[serde(default = "default_true")]
    pub include_tools: bool,
    /// Resume cursor. The client passes the SSE `id` from a task or chat frame back as `after`.
    #[serde(default)]
    pub after: Option<String>,
    /// Permission mode (default, accept-edits, accept-all, deny-all, interactive)
    #[serde(default)]
    pub permission_mode: Option<String>,
}

fn event_is_included(event: &AgentEvent, include_thinking: bool, include_tools: bool) -> bool {
    match event {
        AgentEvent::Thinking { .. } => include_thinking,
        AgentEvent::ToolQueued { .. }
        | AgentEvent::ToolExecuting { .. }
        | AgentEvent::ToolComplete { .. }
        | AgentEvent::ToolError { .. } => include_tools,
        _ => true,
    }
}

/// One task or chat SSE frame before it becomes an axum [`Event`].
#[derive(Debug, PartialEq)]
struct ResumedFrame {
    name: &'static str,
    data: String,
    id: Option<String>,
}

/// Every frame's data is internally tagged JSON (`{"type": ..., ...}`),
/// including the resync.
fn frame_from_resumed(
    item: ResumedSessionItem,
    include_thinking: bool,
    include_tools: bool,
) -> Option<ResumedFrame> {
    match item {
        ResumedSessionItem::Resync(resync) => Some(ResumedFrame {
            name: BroadcastResync::SSE_EVENT,
            data: resync.tagged_data(),
            id: None,
        }),
        ResumedSessionItem::Event { id, event } => {
            if !event_is_included(&event, include_thinking, include_tools) {
                return None;
            }
            Some(ResumedFrame {
                name: public_agent_sse_event_name(&event),
                data: serde_json::to_string(&event).unwrap_or_default(),
                id: (!id.is_empty()).then_some(id),
            })
        }
    }
}

fn sse_from_resumed(
    result: anyhow::Result<ResumedSessionItem>,
    include_thinking: bool,
    include_tools: bool,
) -> Option<Result<Event, Infallible>> {
    let item = match result {
        Ok(item) => item,
        Err(error) => {
            tracing::warn!(error = %error, "durable session stream read failed; frame skipped");
            return None;
        }
    };
    let frame = frame_from_resumed(item, include_thinking, include_tools)?;
    let mut event = Event::default().event(frame.name).data(frame.data);
    if let Some(id) = frame.id {
        event = event.id(id);
    }
    Some(Ok(event))
}

fn parse_permission_mode(
    value: Option<&str>,
    accept_edits: bool,
) -> Result<caliche::harness::config::PermissionModeConfig, (StatusCode, String)> {
    let parsed =
        parse_permission_mode_str(value).map_err(|error| (StatusCode::BAD_REQUEST, error))?;
    if accept_edits {
        return Ok(caliche::harness::config::PermissionModeConfig::AcceptEdits);
    }
    Ok(parsed)
}

/// POST /api/agent/execute - Execute a task and return final result
pub async fn execute_task_handler(
    State(state): State<AgentRouterState>,
    Json(request): Json<AgentTaskRequest>,
) -> Result<Json<AgentTaskResponse>, (StatusCode, String)> {
    let adapter = &state.adapter;
    let harness_config = build_harness_config(&request)?;
    let session_config = SessionConfig {
        workspace: request.workspace.map(std::path::PathBuf::from),
        max_turns: request.max_turns,
        system_prompt_addition: None,
        ..Default::default()
    };

    // Create session
    let session_id =
        Box::pin(adapter.create_session_with_harness_config(harness_config, Some(session_config)))
            .await
            .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    // Execute task
    let result = adapter
        .execute_task(&session_id, &request.task)
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    Ok(Json(AgentTaskResponse {
        session_id,
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
    }))
}

/// POST /api/agent/stream - Stream agent execution events via SSE
pub async fn stream_task_handler(
    State(state): State<AgentRouterState>,
    Json(request): Json<AgentStreamRequest>,
) -> Result<Sse<impl Stream<Item = Result<Event, Infallible>>>, (StatusCode, String)> {
    let adapter = &state.adapter;

    // Build harness config
    let harness_config = HarnessConfig {
        provider: request.provider.as_ref().and_then(|p| p.parse().ok()),
        model: request.model.clone(),
        tools: request.tools.clone(),
        workspace_root: request.workspace.as_ref().map(std::path::PathBuf::from),
        max_turns: request.max_turns.unwrap_or(50),
        system_prompt: request.system_prompt.clone(),
        auto_checkpoint: true,
        permission_mode: parse_permission_mode(request.permission_mode.as_deref(), false)?,
        ..Default::default()
    };

    let session_config = SessionConfig {
        workspace: request.workspace.map(std::path::PathBuf::from),
        max_turns: request.max_turns,
        system_prompt_addition: None,
        ..Default::default()
    };

    // Create session
    let session_id =
        Box::pin(adapter.create_session_with_harness_config(harness_config, Some(session_config)))
            .await
            .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    // Subscribe to events (with optional resume from durable stream)
    let event_stream = adapter
        .subscribe_events_with_resume(&session_id, request.after.clone())
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    let message = request.message.clone();
    let include_thinking = request.include_thinking;
    let include_tools = request.include_tools;
    let adapter_clone = adapter.clone();
    let session_id_clone = session_id.clone();

    // Spawn task execution
    tokio::spawn(async move {
        if let Err(e) = adapter_clone
            .execute_task(&session_id_clone, &message)
            .await
        {
            tracing::error!("Task execution error: {}", e);
        }
    });

    // Task and chat frames carry an SSE id. The client passes that id back as `after`.
    let stream = event_stream.filter_map(move |result| {
        std::future::ready(sse_from_resumed(result, include_thinking, include_tools))
    });

    Ok(Sse::new(stream).keep_alive(
        KeepAlive::new()
            .interval(Duration::from_secs(15))
            .text("keep-alive"),
    ))
}

/// POST /api/agent/chat/stream - Single turn streaming with events
pub async fn chat_stream_with_events_handler(
    State(state): State<AgentRouterState>,
    Json(request): Json<AgentStreamRequest>,
) -> Result<Sse<impl Stream<Item = Result<Event, Infallible>>>, (StatusCode, String)> {
    let adapter = &state.adapter;
    let session_config = SessionConfig {
        workspace: request.workspace.as_ref().map(std::path::PathBuf::from),
        max_turns: Some(1), // Single turn
        system_prompt_addition: None,
        ..Default::default()
    };

    // Create session
    let session_id = Box::pin(adapter.create_session_with_harness_config(
        HarnessConfig {
            provider: request.provider.as_ref().and_then(|p| p.parse().ok()),
            model: request.model.clone(),
            tools: request.tools.clone(),
            workspace_root: request.workspace.as_ref().map(std::path::PathBuf::from),
            max_turns: 1,
            system_prompt: request.system_prompt.clone(),
            permission_mode: parse_permission_mode(request.permission_mode.as_deref(), false)?,
            ..Default::default()
        },
        Some(session_config),
    ))
    .await
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    // Subscribe to events (with optional resume from durable stream)
    let event_stream = adapter
        .subscribe_events_with_resume(&session_id, request.after.clone())
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    let message = request.message.clone();
    let include_thinking = request.include_thinking;
    let include_tools = request.include_tools;
    let adapter_clone = adapter.clone();
    let session_id_clone = session_id.clone();

    // Spawn turn execution
    tokio::spawn(async move {
        if let Err(e) = adapter_clone
            .execute_turn(&session_id_clone, &message)
            .await
        {
            tracing::error!("Turn execution error: {}", e);
        }
    });

    // Task and chat frames carry an SSE id. The client passes that id back as `after`.
    let stream = event_stream.filter_map(move |result| {
        std::future::ready(sse_from_resumed(result, include_thinking, include_tools))
    });

    Ok(Sse::new(stream).keep_alive(
        KeepAlive::new()
            .interval(Duration::from_secs(15))
            .text("keep-alive"),
    ))
}

/// Build HarnessConfig from request
fn build_harness_config(request: &AgentTaskRequest) -> Result<HarnessConfig, (StatusCode, String)> {
    Ok(HarnessConfig {
        provider: request.provider.as_ref().and_then(|p| p.parse().ok()),
        model: request.model.clone(),
        tools: request.tools.clone(),
        workspace_root: request.workspace.as_ref().map(std::path::PathBuf::from),
        max_turns: request.max_turns.unwrap_or(50),
        auto_checkpoint: request.auto_checkpoint,
        system_prompt: request.system_prompt.clone(),
        permission_mode: parse_permission_mode(
            request.permission_mode.as_deref(),
            request.accept_edits,
        )?,
        ..Default::default()
    })
}

/// Request for enhanced agent with reasoning pattern
#[derive(Debug, Deserialize)]
pub struct EnhancedAgentRequest {
    /// The task/message to execute
    pub message: String,
    /// Reasoning pattern to use: "react", "plan_execute", "self_ask", "reflexion", "tree_of_thoughts"
    #[serde(default)]
    pub reasoning_pattern: Option<String>,
    /// Provider to use
    #[serde(default)]
    pub provider: Option<String>,
    /// Model to use
    #[serde(default)]
    pub model: Option<String>,
    /// Tools to enable
    #[serde(default)]
    pub tools: Option<Vec<String>>,
    /// Maximum iterations/turns
    #[serde(default)]
    pub max_iterations: Option<usize>,
    /// Workspace directory
    #[serde(default)]
    pub workspace: Option<String>,
    /// System prompt override
    #[serde(default)]
    pub system_prompt: Option<String>,
}

/// Response for enhanced agent execution
#[derive(Debug, Serialize)]
pub struct EnhancedAgentResponse {
    pub session_id: String,
    pub response: String,
    pub reasoning_pattern: Option<String>,
    pub steps: Vec<ReasoningStepInfo>,
    pub success: bool,
    pub error: Option<String>,
    pub usage: UsageInfo,
}

/// Reasoning step info
#[derive(Debug, Serialize)]
pub struct ReasoningStepInfo {
    pub step_type: String,
    pub content: String,
    pub tool_calls: Option<Vec<String>>,
}

/// POST /api/agent/enhanced - Execute with optional reasoning pattern
#[cfg(feature = "reasoning-patterns")]
pub async fn enhanced_agent_handler(
    State(state): State<AgentRouterState>,
    Json(request): Json<EnhancedAgentRequest>,
) -> Result<Json<EnhancedAgentResponse>, (StatusCode, String)> {
    let adapter = &state.adapter;
    use caliche::agent::service::AgentService;
    use caliche::reasoning::{ReasoningPattern, ReasoningResult};

    // Create session for tracking
    let session_config = SessionConfig {
        workspace: request.workspace.as_ref().map(std::path::PathBuf::from),
        max_turns: request.max_iterations,
        system_prompt_addition: request.system_prompt.clone(),
        ..Default::default()
    };

    let session_id = adapter
        .create_session(Some(session_config), None, None)
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    // If no reasoning pattern, use standard execution
    if request.reasoning_pattern.is_none() {
        let result = adapter
            .execute_task(&session_id, &request.message)
            .await
            .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

        return Ok(Json(EnhancedAgentResponse {
            session_id,
            response: result.response,
            reasoning_pattern: None,
            steps: Vec::new(),
            success: true,
            error: None,
            usage: UsageInfo {
                prompt_tokens: result.total_usage.prompt_tokens.map(|t| t as u64),
                completion_tokens: result.total_usage.completion_tokens.map(|t| t as u64),
                total_tokens: result.total_usage.total_tokens.map(|t| t as u64),
            },
        }));
    }

    // Create agent service for reasoning patterns
    let tool_names = request.tools.clone();
    let service = AgentService::from_config(
        caliche::AgentServiceConfig::new()
            .with_provider(
                request.provider.as_ref().and_then(|p| p.parse().ok()),
                request.model.clone(),
            )
            .with_tool_names_or_defaults(tool_names),
    )
    .await
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    let max_iterations = request.max_iterations.unwrap_or(5);
    let pattern_name = request.reasoning_pattern.as_deref().unwrap_or("react");

    // Execute with selected reasoning pattern
    let result: ReasoningResult = match pattern_name {
        "react" => {
            let pattern = caliche::reasoning::react::ReActPattern::new(service, max_iterations);
            pattern.execute(&request.message).await
        }
        "plan_execute" | "plan-execute" => {
            let pattern =
                caliche::reasoning::plan_execute::PlanExecutePattern::new(service, max_iterations);
            pattern.execute(&request.message).await
        }
        "self_ask" | "self-ask" => {
            let pattern =
                caliche::reasoning::self_ask::SelfAskPattern::new(service, max_iterations);
            pattern.execute(&request.message).await
        }
        "reflexion" => {
            let pattern =
                caliche::reasoning::reflexion::ReflexionPattern::new(service, max_iterations);
            pattern.execute(&request.message).await
        }
        "tree_of_thoughts" | "tot" => {
            let pattern = caliche::reasoning::tree_of_thoughts::TreeOfThoughtsPattern::new(
                service, 3, // branches
                2, // depth
            );
            pattern.execute(&request.message).await
        }
        _ => {
            return Err((
                StatusCode::BAD_REQUEST,
                format!("Unknown reasoning pattern: {}", pattern_name),
            ));
        }
    }
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    let steps: Vec<ReasoningStepInfo> = result
        .steps
        .iter()
        .map(|s| ReasoningStepInfo {
            step_type: format!("{:?}", s.step_type),
            content: s.content.clone(),
            tool_calls: s
                .metadata
                .as_ref()
                .and_then(|m| m.get("tool_calls"))
                .and_then(|v| v.as_array())
                .map(|arr| {
                    arr.iter()
                        .filter_map(|v| v.as_str().map(|s| s.to_string()))
                        .collect()
                }),
        })
        .collect();

    Ok(Json(EnhancedAgentResponse {
        session_id,
        response: result.final_answer,
        reasoning_pattern: Some(pattern_name.to_string()),
        steps,
        success: result.success,
        error: result.error,
        usage: UsageInfo {
            prompt_tokens: result
                .token_usage
                .as_ref()
                .and_then(|u| u.prompt_tokens)
                .map(|t| t as u64),
            completion_tokens: result
                .token_usage
                .as_ref()
                .and_then(|u| u.completion_tokens)
                .map(|t| t as u64),
            total_tokens: result
                .token_usage
                .as_ref()
                .and_then(|u| u.total_tokens)
                .map(|t| t as u64),
        },
    }))
}

/// POST /api/agent/enhanced - Stub when reasoning patterns not enabled
#[cfg(not(feature = "reasoning-patterns"))]
pub async fn enhanced_agent_handler(
    State(_state): State<AgentRouterState>,
    Json(_request): Json<EnhancedAgentRequest>,
) -> Result<Json<EnhancedAgentResponse>, (StatusCode, String)> {
    Err((
        StatusCode::NOT_IMPLEMENTED,
        "Reasoning patterns feature not enabled".to_string(),
    ))
}

#[cfg(test)]
mod permission_mode_tests {
    use super::parse_permission_mode;
    use axum::http::StatusCode;
    use caliche::harness::config::PermissionModeConfig;

    #[test]
    fn unknown_permission_mode_is_rejected_when_accept_edits_is_set() {
        let error = parse_permission_mode(Some("nope"), true).expect_err("unknown mode");
        assert_eq!(error.0, StatusCode::BAD_REQUEST);
        assert!(error.1.contains("nope"));
    }

    #[test]
    fn accept_edits_flag_selects_accept_edits_for_a_known_mode() {
        let mode = parse_permission_mode(Some("deny-all"), true).expect("known mode");
        assert_eq!(mode, PermissionModeConfig::AcceptEdits);
    }

    #[test]
    fn missing_permission_mode_stays_default() {
        let mode = parse_permission_mode(None, false).expect("empty mode");
        assert_eq!(mode, PermissionModeConfig::Default);
    }
}

#[cfg(test)]
mod resumed_frame_tests {
    use super::{ResumedFrame, frame_from_resumed};
    use crate::adapter::ResumedSessionItem;
    use caliche::harness::{AgentEvent, BroadcastResync};

    fn data(frame: &ResumedFrame) -> serde_json::Value {
        serde_json::from_str(&frame.data).expect("json")
    }

    #[test]
    fn resync_and_event_frames_share_the_tagged_shape() {
        let resync = frame_from_resumed(
            ResumedSessionItem::Resync(BroadcastResync::from_skipped(2)),
            true,
            true,
        )
        .expect("resync frame");
        assert_eq!(resync.name, "resync");
        assert_eq!(resync.id, None);
        assert_eq!(
            data(&resync),
            serde_json::json!({ "type": "resync", "skipped": 2 })
        );

        let event = frame_from_resumed(
            ResumedSessionItem::Event {
                id: "7".to_owned(),
                event: AgentEvent::Info {
                    message: "m2".to_owned(),
                },
            },
            true,
            true,
        )
        .expect("event frame");
        assert_eq!(event.name, "info");
        assert_eq!(event.id.as_deref(), Some("7"));
        assert_eq!(
            data(&event),
            serde_json::json!({ "type": "info", "message": "m2" })
        );
    }

    #[test]
    fn a_live_event_has_no_id_and_an_excluded_event_has_no_frame() {
        let live = frame_from_resumed(
            ResumedSessionItem::Event {
                id: String::new(),
                event: AgentEvent::Info {
                    message: "m2".to_owned(),
                },
            },
            true,
            true,
        )
        .expect("live frame");
        assert_eq!(live.id, None);

        let thinking = frame_from_resumed(
            ResumedSessionItem::Event {
                id: "8".to_owned(),
                event: AgentEvent::Thinking {
                    content: "plan".to_owned(),
                    action_id: String::new(),
                },
            },
            false,
            true,
        );
        assert_eq!(thinking, None);
    }
}
