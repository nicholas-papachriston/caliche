//! Resolve the agent service for a chat request.
//!
//! When the CLI hosts the API, chat uses a clone of the shared service so header
//! model and embedding changes stay on the live agent. Per-request provider/model
//! values call `change_model` on that clone instead of building a bare service.

use crate::middleware::UserContext;
#[cfg(feature = "router")]
use crate::select_provider;
use crate::{AppState, get_or_create_provider, get_or_create_tool_registry};
use axum::http::StatusCode;
use caliche::agent::service::AgentService;
#[cfg(not(feature = "router"))]
use caliche::agent::service::resolve_runtime_identity_with_discovery;
use caliche::llm::ProviderType;
use caliche_contracts::chat::ChatRequestDto;

/// True when the request asks for a different provider or model than the shared service.
#[must_use]
pub(crate) fn request_changes_shared_model(
    current_provider: ProviderType,
    current_model: &str,
    requested_provider: Option<ProviderType>,
    requested_model: Option<&str>,
) -> bool {
    if let Some(provider) = requested_provider
        && provider != current_provider
    {
        return true;
    }
    requested_model.is_some_and(|model| model != current_model)
}

pub(crate) async fn resolve_request_service(
    state: &AppState,
    user_context: Option<&UserContext>,
    request: &ChatRequestDto,
) -> Result<(AgentService, ProviderType, String), (StatusCode, String)> {
    if let Some(ref shared) = state.shared_service {
        return resolve_shared_service(shared, request).await;
    }
    build_standalone_service(state, user_context, request).await
}

async fn resolve_shared_service(
    shared: &std::sync::Arc<tokio::sync::RwLock<AgentService>>,
    request: &ChatRequestDto,
) -> Result<(AgentService, ProviderType, String), (StatusCode, String)> {
    let requested_provider = match request.provider.as_deref() {
        None => None,
        Some(value) => Some(
            value
                .parse::<ProviderType>()
                .map_err(|err| (StatusCode::BAD_REQUEST, format!("Invalid provider: {err}")))?,
        ),
    };
    let requested_model = request.model.as_deref();

    let mut service = {
        let guard = shared.read().await;
        guard.clone_for_request()
    };

    if request_changes_shared_model(
        service.provider(),
        service.model(),
        requested_provider,
        requested_model,
    ) {
        service
            .change_model(requested_provider, request.model.clone())
            .await
            .map_err(|err| (StatusCode::BAD_REQUEST, err.to_string()))?;
    }

    if let Some(ref tools) = request.tools {
        if tools.is_empty() {
            service.agent_mut().set_allowed_tools(None);
        } else {
            service.agent_mut().set_allowed_tools(Some(tools.clone()));
        }
    }

    let provider_type = service.provider();
    let model = service.model().to_string();
    Ok((service, provider_type, model))
}

async fn build_standalone_service(
    state: &AppState,
    user_context: Option<&UserContext>,
    request: &ChatRequestDto,
) -> Result<(AgentService, ProviderType, String), (StatusCode, String)> {
    #[cfg(feature = "router")]
    let (provider_type, model) = {
        let requested_provider =
            match request.provider.as_deref() {
                None => None,
                Some(value) => Some(value.parse::<ProviderType>().map_err(|err| {
                    (StatusCode::BAD_REQUEST, format!("Invalid provider: {err}"))
                })?),
            };
        let tenant_id = user_context.and_then(|ctx| ctx.tenant_id.as_deref());
        if let Some(ref tracker) = state.spend_tracker
            && let Some(tid) = tenant_id
        {
            tracker
                .check_budget_tokens(tid, 4096)
                .await
                .map_err(|err| (StatusCode::TOO_MANY_REQUESTS, err.to_string()))?;
        }
        let selection =
            select_provider(state, requested_provider, request.model.clone(), tenant_id)
                .await
                .map_err(|err| {
                    (
                        StatusCode::INTERNAL_SERVER_ERROR,
                        format!("Provider selection failed: {err}"),
                    )
                })?;
        (selection.provider, selection.model)
    };

    #[cfg(not(feature = "router"))]
    let (provider_type, model) = {
        let _ = user_context;
        let provider_type = if let Some(provider_str) = request.provider.as_deref() {
            provider_str
                .parse()
                .map_err(|err| (StatusCode::BAD_REQUEST, format!("Invalid provider: {err}")))?
        } else {
            state.default_provider
        };
        let model = if request.model.is_none() && request.provider.is_none() {
            state.default_model.clone()
        } else {
            resolve_runtime_identity_with_discovery(
                Some(provider_type),
                request.model.clone(),
                None,
            )
            .await
            .map_err(|err| (StatusCode::BAD_REQUEST, err.to_string()))?
            .model
        };
        (provider_type, model)
    };

    let provider = get_or_create_provider(state, provider_type, model.clone())
        .await
        .map_err(|err| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("Failed to get or create provider: {err}"),
            )
        })?;

    let tool_registry = get_or_create_tool_registry(state, provider_type, request.tools.clone())
        .await
        .map_err(|err| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("Failed to get or create tool registry: {err}"),
            )
        })?;

    let service =
        AgentService::from_components(provider, tool_registry, provider_type, model.clone(), None)
            .map_err(|err| {
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    format!("Failed to create agent service: {err}"),
                )
            })?;
    Ok((service, provider_type, model))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn request_changes_shared_model_detects_provider_and_model_overrides() {
        let openai = "openai".parse::<ProviderType>().expect("openai provider");
        let anthropic = "anthropic"
            .parse::<ProviderType>()
            .expect("anthropic provider");

        assert!(!request_changes_shared_model(
            openai,
            "gpt-4o",
            Some(openai),
            Some("gpt-4o")
        ));
        assert!(!request_changes_shared_model(openai, "gpt-4o", None, None));
        assert!(request_changes_shared_model(
            openai,
            "gpt-4o",
            Some(openai),
            Some("gpt-4.1")
        ));
        assert!(request_changes_shared_model(
            openai,
            "gpt-4o",
            Some(anthropic),
            Some("gpt-4o")
        ));
    }
}
