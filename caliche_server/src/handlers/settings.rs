//! GET/PUT /api/settings for shared agent configuration (CLI-hosted mode).

use crate::handlers::chat_contracts::{core_mode_to_dto, dto_mode_value_to_core};
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::Json;
use caliche::{AgentSettings, llm::ProviderType};
use caliche_contracts::settings::AgentSettingsDto;

use crate::AppState;

/// GET /api/settings - Return current shared service settings.
/// Returns 404 when no shared service (standalone API mode).
pub async fn get_settings_handler(
    State(state): State<AppState>,
) -> Result<Json<AgentSettingsDto>, (StatusCode, String)> {
    let shared = state.shared_service.as_ref().ok_or_else(|| {
        (
            StatusCode::NOT_FOUND,
            "No shared service (standalone API)".to_string(),
        )
    })?;
    let guard = shared.read().await;
    let settings = guard.snapshot_settings();
    Ok(Json(core_settings_to_dto(settings)))
}

/// PUT /api/settings - Apply settings to the shared service and write the
/// durable provider/model/sampling record beside `mcp.json`.
///
/// The request's `system_prompt` is ignored. GET still returns the effective
/// prompt for inspection. Save must not copy that text into `base_system_prompt`
/// or the durable file, or a later startup would freeze `AGENTS.md`.
/// Returns 404 when no shared service (standalone API mode).
pub async fn put_settings_handler(
    State(state): State<AppState>,
    Json(settings): Json<AgentSettingsDto>,
) -> Result<Json<AgentSettingsDto>, (StatusCode, String)> {
    let mut settings = dto_settings_to_core(settings)?;
    settings.system_prompt = None;
    let shared = state.shared_service.as_ref().ok_or_else(|| {
        (
            StatusCode::NOT_FOUND,
            "No shared service (standalone API)".to_string(),
        )
    })?;
    let mut guard = shared.write().await;
    guard
        .apply_settings(settings)
        .await
        .map_err(|e| (StatusCode::BAD_REQUEST, e.to_string()))?;
    let updated = guard.snapshot_settings();
    drop(guard);
    let settings_path = crate::durable_settings::path_beside_mcp(&state.mcp_config_path);
    let record = crate::durable_settings::DurableSettings::from_agent_settings(&updated);
    crate::durable_settings::save(&settings_path, &record)
        .await
        .map_err(|err| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("Failed to persist settings: {err}"),
            )
        })?;
    if let Some(ref tx) = state.settings_broadcast {
        let _ = tx.send(updated.clone());
    }
    if let Some(ref hook) = state.on_settings_applied {
        hook(&updated);
    }
    Ok(Json(core_settings_to_dto(updated)))
}

fn core_settings_to_dto(settings: AgentSettings) -> AgentSettingsDto {
    let retrieval = caliche::capabilities::vector_retrieval_status();
    AgentSettingsDto {
        provider: settings.provider.to_string(),
        model: settings.model,
        system_prompt: settings.system_prompt,
        temperature: settings.temperature,
        max_tokens: settings.max_tokens,
        context_window: settings.context_window,
        tools: settings.tools,
        mode: settings.mode.map(core_mode_to_dto),
        embedding_provider: settings.embedding_provider,
        embedding_model: settings.embedding_model,
        show_sources: settings.show_sources,
        vector_backend: retrieval.backend,
        vector_backend_state: Some(retrieval.state.to_owned()),
    }
}

fn dto_settings_to_core(settings: AgentSettingsDto) -> Result<AgentSettings, (StatusCode, String)> {
    let provider = settings
        .provider
        .trim()
        .to_ascii_lowercase()
        .parse::<ProviderType>()
        .map_err(|e| (StatusCode::BAD_REQUEST, format!("Invalid provider: {e}")))?;

    Ok(AgentSettings {
        provider,
        model: settings.model,
        system_prompt: settings.system_prompt,
        temperature: settings.temperature,
        max_tokens: settings.max_tokens,
        context_window: settings.context_window,
        tools: settings.tools,
        mode: settings.mode.map(dto_mode_value_to_core),
        embedding_provider: settings.embedding_provider,
        embedding_model: settings.embedding_model,
        show_sources: settings.show_sources,
    })
}
