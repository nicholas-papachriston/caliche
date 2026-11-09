use axum::extract::{Query, State};
use axum::http::StatusCode;
use axum::response::Json;
use caliche::llm::{
    ModelCatalogEntry, ModelCatalogOptions, ProviderCatalogEntry, ProviderType,
    discover_provider_catalog, discover_provider_models, flatten_catalog_models,
};
use caliche_contracts::models::{
    ModelCatalogEntryDto, ModelsCatalogResponseDto, ProviderCatalogEntryDto,
    ProviderModelsResponseDto,
};
use std::collections::HashMap;
use std::sync::Arc;

use crate::AppState;

fn catalog_options() -> ModelCatalogOptions {
    ModelCatalogOptions::from_environment()
}

fn model_to_dto(model: ModelCatalogEntry) -> ModelCatalogEntryDto {
    ModelCatalogEntryDto {
        id: model.id,
        name: model.name,
        provider_id: model.provider_id,
        provider_display_name: model.provider_display_name,
        description: model.description,
        context_length: model.context_length,
    }
}

fn provider_to_dto(provider: ProviderCatalogEntry) -> ProviderCatalogEntryDto {
    ProviderCatalogEntryDto {
        id: provider.id,
        display_name: provider.display_name,
        configured: provider.configured,
        local: provider.local,
        models: provider.models.into_iter().map(model_to_dto).collect(),
    }
}

async fn get_cached_provider_models(
    state: &AppState,
    provider: ProviderType,
) -> Arc<ProviderModelsResponseDto> {
    let key = format!("provider:{provider}");
    Box::pin(state.model_list_cache.get_with(key, async move {
        let provider = discover_provider_models(provider, &catalog_options())
            .await
            .unwrap_or_else(|error| {
                tracing::warn!("Failed to discover provider models: {error}");
                ProviderCatalogEntry {
                    id: provider.to_string(),
                    display_name: provider.to_string(),
                    configured: false,
                    local: false,
                    models: Vec::new(),
                }
            });
        let models = provider.models.iter().cloned().map(model_to_dto).collect();
        Arc::new(ProviderModelsResponseDto {
            provider: provider_to_dto(provider),
            models,
        })
    }))
    .await
}

/// GET /api/models - List available models for a provider
#[tracing::instrument(name = "api_models_by_provider", skip_all, fields(provider))]
pub async fn models_handler(
    State(state): State<AppState>,
    Query(params): Query<HashMap<String, String>>,
) -> Result<Json<ProviderModelsResponseDto>, (StatusCode, String)> {
    let provider_str = params
        .get("provider")
        .ok_or_else(|| {
            (
                StatusCode::BAD_REQUEST,
                "Missing 'provider' query parameter".to_string(),
            )
        })?
        .as_str();

    let provider: ProviderType = provider_str
        .parse()
        .map_err(|e| (StatusCode::BAD_REQUEST, format!("Invalid provider: {}", e)))?;

    let models = Box::pin(get_cached_provider_models(&state, provider)).await;

    Ok(Json(models.as_ref().clone()))
}

/// GET /api/models - List all available models from all providers for UI consumption.
/// When API keys are set, fetches live model lists from provider APIs (e.g. OpenRouter)
/// so the UI shows the full catalog; otherwise uses static enum fallbacks.
#[tracing::instrument(name = "api_models_all", skip_all)]
pub async fn all_models_handler(State(state): State<AppState>) -> Json<ModelsCatalogResponseDto> {
    let all_models = Box::pin(state.all_models_cache.get_with("all".to_string(), async {
        let providers = discover_provider_catalog(&catalog_options())
            .await
            .unwrap_or_else(|error| {
                tracing::warn!("Failed to discover model catalog: {error}");
                Vec::new()
            });
        let models = flatten_catalog_models(&providers);
        Arc::new(ModelsCatalogResponseDto {
            providers: providers.into_iter().map(provider_to_dto).collect(),
            models: models.into_iter().map(model_to_dto).collect(),
        })
    }))
    .await;

    Json(all_models.as_ref().clone())
}
