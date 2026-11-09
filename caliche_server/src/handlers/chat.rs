use crate::AppState;
use crate::handlers::chat_contracts::{
    core_mode_to_dto, core_usage_to_dto, dto_history_to_core, dto_mode_to_core,
};
use crate::handlers::chat_history::trim_trailing_non_conversational_turns;
use crate::handlers::chat_service::resolve_request_service;
use crate::handlers::multipart_chat::parse_multipart_chat;
use crate::middleware::OptionalUserContext;
use axum::extract::{Multipart, State};
use axum::http::StatusCode;
use axum::response::Json;
use caliche::messages::Message;
use caliche_contracts::chat::{
    ChatRequestDto, ChatResponseDto, SourceCitationDto as SourceCitation, parse_sources_footer,
    strip_sources_footer,
};
use serde_json::Value;
use std::collections::HashSet;
use tracing::Instrument;
use uuid::Uuid;

/// POST /api/chat/multipart - Send a message with file attachments.
pub async fn chat_multipart_handler(
    State(state): State<AppState>,
    OptionalUserContext(user_context): OptionalUserContext,
    multipart: Multipart,
) -> Result<Json<ChatResponseDto>, (StatusCode, String)> {
    let parsed = parse_multipart_chat(multipart).await?;
    chat_handler_with_history(
        State(state),
        OptionalUserContext(user_context),
        Json(parsed.request),
        parsed.history_override,
    )
    .await
}

/// POST /api/chat - Send a message to the agent
pub async fn chat_handler(
    State(state): State<AppState>,
    OptionalUserContext(user_context): OptionalUserContext,
    Json(request): Json<ChatRequestDto>,
) -> Result<Json<ChatResponseDto>, (StatusCode, String)> {
    chat_handler_with_history(
        State(state),
        OptionalUserContext(user_context),
        Json(request),
        None,
    )
    .await
}

async fn chat_handler_with_history(
    State(state): State<AppState>,
    OptionalUserContext(user_context): OptionalUserContext,
    Json(request): Json<ChatRequestDto>,
    history_override: Option<Vec<Message>>,
) -> Result<Json<ChatResponseDto>, (StatusCode, String)> {
    // Generate request ID for this API call
    let request_id = Uuid::now_v7().to_string();
    let span = tracing::span!(tracing::Level::INFO, "api_chat", request_id = %request_id);

    async move {
        tracing::info!("Received chat request");

        let context_window_override = request.context_window.or(state.context_window);

        let (mut service, provider_type, model) =
            resolve_request_service(&state, user_context.as_ref(), &request).await?;

        let model_for_logging = model.clone();

        if let Some(context_window) = context_window_override {
            service
                .set_context_window(context_window)
                .map_err(|e| (StatusCode::BAD_REQUEST, e.to_string()))?;
        }

        // Set per-request metrics
        service.set_metrics(state.tool_metrics.clone());

        // Set per-request user context if available
        // Convert API UserContext to caliche UserContext
        if let Some(ref ctx) = user_context {
            let agent_ctx = caliche::UserContext {
                user_id: ctx.user_id.clone(),
                tenant_id: ctx.tenant_id.clone(),
                email: ctx.email.clone(),
                roles: ctx.roles.clone(),
                custom_claims: ctx.custom_claims.clone(),
            };
            service.set_user_context(Some(agent_ctx));
            tracing::debug!(
                "Set user context: user_id={}, tenant_id={:?}",
                ctx.user_id,
                ctx.tenant_id
            );
        }

        // Apply interaction mode restrictions
        let effective_mode = dto_mode_to_core(request.mode);
        let show_sources = service.show_sources();
        if effective_mode.is_readonly() {
            use caliche::agent::modes::{readonly_blocked_tools, readonly_mode_prompt};

            // Restrict to read-only tools
            let readonly_tools: Vec<String> = service
                .list_tools()
                .iter()
                .map(|(name, _, _)| name.to_string())
                .filter(|name| !readonly_blocked_tools().contains(&name.as_str()))
                .collect();
            service.agent_mut().set_allowed_tools(Some(readonly_tools));

            // Apply mode-specific system prompt
            if let Some(mode_prompt) = readonly_mode_prompt(effective_mode) {
                let original = service.agent().system_prompt().to_string();
                let new_prompt = format!("{}\n\n{}", mode_prompt, original);
                service.agent_mut().set_system_prompt(new_prompt);
            }

            tracing::info!("Applied {} mode restrictions", effective_mode);
        }

        // Use chat_with_history if history is provided, otherwise use regular chat
        let history = history_override.or_else(|| {
            let history = dto_history_to_core(request.messages);
            (!history.is_empty()).then_some(history)
        });

        let result = if let Some(history) = history {
            tracing::info!("API received {} history messages", history.len());
            for (i, msg) in history.iter().enumerate() {
                tracing::debug!("  API History[{}]: {:?}", i, msg.role);
            }
            service
                .chat_with_history(
                    &request.message,
                    trim_trailing_non_conversational_turns(history),
                )
                .await
        } else {
            tracing::info!("API: No history provided, starting fresh conversation");
            service.chat(&request.message).await
        };

        match result {
            Ok(chat_response) => {
                #[cfg(feature = "router")]
                if let (Some(tracker), Some(ctx)) = (&state.spend_tracker, &user_context) {
                    if let Some(ref tid) = ctx.tenant_id {
                        let tokens = chat_response.usage().total_tokens.unwrap_or(0) as u64;
                        if tokens > 0 {
                            tracker.record_tokens(tid, tokens).await;
                        }
                    }
                }

                let mut response_text = chat_response.text().to_string();
                let mut sources = if chat_response.used_tools() {
                    extract_sources_from_response(&response_text)
                } else {
                    parse_sources_footer(&response_text)
                };
                if !show_sources {
                    response_text = strip_sources_footer(&response_text).to_string();
                    sources = None;
                }

                Ok(Json(ChatResponseDto {
                    response: response_text,
                    request_id,
                    usage: core_usage_to_dto(chat_response.usage()),
                    sources,
                    mode: core_mode_to_dto(effective_mode),
                }))
            }
            Err(e) => {
                let error_msg = format!("Failed to process chat: {}", e);
                tracing::error!(
                    request_id = %request_id,
                    provider = ?provider_type,
                    model = %model_for_logging,
                    error = %e,
                    error_debug = ?e,
                    "Chat request failed"
                );
                let status = if e
                    .downcast_ref::<caliche::errors::AgentError>()
                    .is_some_and(|err| {
                        matches!(
                            err,
                            caliche::errors::AgentError::ContextWindowExceeded { .. }
                        )
                    }) {
                    StatusCode::BAD_REQUEST
                } else {
                    StatusCode::INTERNAL_SERVER_ERROR
                };
                Err((status, error_msg))
            }
        }
    }
    .instrument(span)
    .await
}

/// Extract source citations from tool execution results
///
/// This function attempts to extract source information from tool results
/// by parsing common patterns in the response text. It looks for:
/// - RAG tool results: document IDs, collection names, similarity scores
/// - HTTP tool results: URLs from responses
/// - File tool results: file paths
/// - Web browser tool results: URLs visited
fn extract_sources_from_response(response_text: &str) -> Option<Vec<SourceCitation>> {
    let mut sources = Vec::new();
    let mut seen_resources = HashSet::new();

    // Try to extract sources from JSON-like structures in the response
    // RAG tools often return JSON with document information
    if let Ok(json_value) = serde_json::from_str::<Value>(response_text) {
        extract_sources_from_json(&json_value, &mut sources, &mut seen_resources);
    } else {
        // Try to find JSON objects within the text
        extract_sources_from_text(response_text, &mut sources, &mut seen_resources);
    }

    // Extract URLs from the response text
    extract_urls_from_text(response_text, &mut sources, &mut seen_resources);

    // Extract file paths from the response text
    extract_file_paths_from_text(response_text, &mut sources, &mut seen_resources);

    if sources.is_empty() {
        None
    } else {
        Some(sources)
    }
}

/// Extract sources from a JSON value (for RAG tool results)
fn extract_sources_from_json(
    value: &Value,
    sources: &mut Vec<SourceCitation>,
    seen: &mut HashSet<String>,
) {
    match value {
        Value::Object(obj) => {
            // Look for common RAG result fields
            if let Some(doc_id) = obj.get("doc_id").or_else(|| obj.get("document_id"))
                && let Some(doc_id_str) = doc_id.as_str()
            {
                let resource = doc_id_str.to_string();
                if seen.insert(resource.clone()) {
                    let mut metadata = serde_json::Map::new();
                    if let Some(collection) = obj.get("collection") {
                        metadata.insert("collection".to_string(), collection.clone());
                    }
                    if let Some(score) = obj.get("similarity").or_else(|| obj.get("score")) {
                        metadata.insert("similarity".to_string(), score.clone());
                    }
                    if let Some(title) = obj.get("title").or_else(|| obj.get("doc_title")) {
                        metadata.insert("title".to_string(), title.clone());
                    }

                    sources.push(SourceCitation {
                        source_type: "document".to_string(),
                        resource,
                        metadata: if metadata.is_empty() {
                            None
                        } else {
                            Some(Value::Object(metadata))
                        },
                    });
                }
            }

            // Recursively check nested objects and arrays
            for (_, v) in obj {
                extract_sources_from_json(v, sources, seen);
            }
        }
        Value::Array(arr) => {
            for item in arr {
                extract_sources_from_json(item, sources, seen);
            }
        }
        _ => {}
    }
}

/// Extract sources from text by looking for JSON-like patterns
fn extract_sources_from_text(
    text: &str,
    sources: &mut Vec<SourceCitation>,
    seen: &mut HashSet<String>,
) {
    // Look for JSON objects in the text (simple pattern matching)
    // This is a basic implementation - can be enhanced with proper JSON parsing
    let json_pattern = regex::Regex::new(r#"\{"doc_id"\s*:\s*"([^"]+)""#).ok();

    if let Some(pattern) = json_pattern {
        for cap in pattern.captures_iter(text) {
            if let Some(doc_id) = cap.get(1) {
                let resource = doc_id.as_str().to_string();
                if seen.insert(resource.clone()) {
                    sources.push(SourceCitation {
                        source_type: "document".to_string(),
                        resource,
                        metadata: None,
                    });
                }
            }
        }
    }
}

/// Extract URLs from text
fn extract_urls_from_text(
    text: &str,
    sources: &mut Vec<SourceCitation>,
    seen: &mut HashSet<String>,
) {
    // Simple URL pattern matching
    let url_pattern =
        regex::Regex::new(r#"https?://[^\s<>"{}|\\^`\[\]]+[^\s<>"{}|\\^`\[\].,;:!?]"#).ok();

    if let Some(pattern) = url_pattern {
        for cap in pattern.find_iter(text) {
            let url = cap
                .as_str()
                .trim_end_matches(|c: char| c.is_ascii_punctuation() && c != '/');
            if seen.insert(url.to_string()) {
                sources.push(SourceCitation {
                    source_type: "url".to_string(),
                    resource: url.to_string(),
                    metadata: None,
                });
            }
        }
    }
}

/// Extract file paths from text
fn extract_file_paths_from_text(
    text: &str,
    sources: &mut Vec<SourceCitation>,
    seen: &mut HashSet<String>,
) {
    // Look for file paths (basic pattern - can be enhanced)
    let file_pattern = regex::Regex::new(
        r#"(?:^|\s)(?:\./|/|~/?)(?:[a-zA-Z0-9_\-./]+/)*[a-zA-Z0-9_\-]+\.[a-zA-Z0-9]+"#,
    )
    .ok();

    if let Some(pattern) = file_pattern {
        for cap in pattern.find_iter(text) {
            let path = cap.as_str().trim();
            // Filter out common false positives
            if !path.contains("://") && !path.starts_with("http") && seen.insert(path.to_string()) {
                sources.push(SourceCitation {
                    source_type: "file".to_string(),
                    resource: path.to_string(),
                    metadata: None,
                });
            }
        }
    }
}
