use axum::extract::{Path as AxumPath, Query};
use axum::http::{StatusCode, header};
use axum::response::{IntoResponse, Response};
use serde::{Deserialize, Serialize};

use crate::commands::credentials_service::CredentialService;

use super::shared::internal_error;

#[derive(Debug, Deserialize)]
pub(crate) struct ConfigShowQuery {
    format: Option<String>,
    verbose: Option<bool>,
}

#[derive(Debug, Serialize)]
pub(crate) struct ConfigShowJsonResponse {
    sources: Vec<String>,
    config: crate::config::Config,
}

#[derive(Debug, Deserialize)]
pub(crate) struct SecretQuery {
    unmask: Option<bool>,
    yes_i_understand_this_prints_secrets: Option<bool>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct ConfiguredQuery {
    configured: Option<bool>,
}

#[derive(Debug, Serialize)]
pub(crate) struct ProviderStatusDto {
    name: String,
    description: String,
    has_config_key: bool,
    has_env_key: bool,
    is_configured: bool,
    default_model: Option<String>,
    is_default: bool,
    is_running: Option<bool>,
    available_models: Option<Vec<String>>,
    status: String,
}

#[derive(Debug, Serialize)]
pub(crate) struct CredentialValueDto {
    name: String,
    value: Option<String>,
    masked: bool,
    configured: bool,
}

#[derive(Debug, Deserialize)]
pub(crate) struct SetCredentialRequest {
    name: Option<String>,
    provider: Option<String>,
    key: Option<String>,
    env: Option<String>,
}

#[derive(Debug, Serialize)]
pub(crate) struct MutationResponse {
    changed: bool,
}

#[derive(Debug, Deserialize)]
pub(crate) struct SetDefaultProviderRequest {
    provider: String,
}

#[derive(Debug, Serialize)]
pub(crate) struct DefaultProviderResponse {
    provider: Option<String>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct SetDefaultModelRequest {
    provider: String,
    model: String,
}

#[derive(Debug, Serialize)]
pub(crate) struct DefaultModelResponse {
    provider: String,
    model: String,
}

#[derive(Debug, Serialize)]
pub(crate) struct GatewayCredentialStatusDto {
    name: String,
    description: String,
    env_var: String,
    configured: bool,
}

#[derive(Debug, Deserialize)]
pub(crate) struct ConfigExportQuery {
    yes_i_understand_this_prints_secrets: Option<bool>,
}
pub(crate) async fn config_show_handler(
    Query(query): Query<ConfigShowQuery>,
) -> Result<Response, (StatusCode, String)> {
    let mut manager = crate::config::ConfigManager::with_standard_sources();
    let (config, sources) = manager.load_with_sources().await.map_err(internal_error)?;
    match query.format.as_deref().unwrap_or("json") {
        "json" => {
            let body = serde_json::to_string_pretty(&ConfigShowJsonResponse {
                sources: if query.verbose.unwrap_or(false) {
                    sources
                } else {
                    Vec::new()
                },
                config: config.clone(),
            })
            .map_err(internal_error)?;
            Ok(([(header::CONTENT_TYPE, "application/json")], body).into_response())
        }
        "toml" => {
            let mut body = String::new();
            if query.verbose.unwrap_or(false) && !sources.is_empty() {
                body.push_str(&format!("# Sources: {}\n\n", sources.join(", ")));
            }
            body.push_str(&toml::to_string_pretty(config).map_err(internal_error)?);
            Ok(([(header::CONTENT_TYPE, "text/plain; charset=utf-8")], body).into_response())
        }
        other => Err((
            StatusCode::BAD_REQUEST,
            format!("unsupported config format: {other}"),
        )),
    }
}

pub(crate) async fn config_schema_handler() -> Result<Response, (StatusCode, String)> {
    let schema = schemars::schema_for!(crate::config::Config);
    let body = serde_json::to_string_pretty(&schema).map_err(internal_error)?;
    Ok(([(header::CONTENT_TYPE, "application/json")], body).into_response())
}

pub(crate) async fn config_path_handler() -> Result<Response, (StatusCode, String)> {
    let path = crate::storage::database::Database::default_db_path()
        .map_err(internal_error)?
        .display()
        .to_string();
    Ok(([(header::CONTENT_TYPE, "text/plain; charset=utf-8")], path).into_response())
}

pub(crate) async fn config_providers_handler(
    Query(query): Query<ConfiguredQuery>,
) -> Result<axum::Json<Vec<ProviderStatusDto>>, (StatusCode, String)> {
    let service = CredentialService::new().await.map_err(internal_error)?;
    let providers = service
        .list_provider_status(query.configured.unwrap_or(false))
        .await
        .into_iter()
        .map(|status| {
            let status_display = status.status_display().to_string();
            ProviderStatusDto {
                name: status.name,
                description: status.description,
                has_config_key: status.has_config_key,
                has_env_key: status.has_env_key,
                is_configured: status.is_configured,
                default_model: status.default_model,
                is_default: status.is_default,
                is_running: status.is_running,
                available_models: status.available_models,
                status: status_display,
            }
        })
        .collect();
    Ok(axum::Json(providers))
}

pub(crate) async fn config_get_provider_handler(
    AxumPath(provider): AxumPath<String>,
    Query(query): Query<SecretQuery>,
) -> Result<axum::Json<CredentialValueDto>, (StatusCode, String)> {
    let service = CredentialService::new().await.map_err(internal_error)?;
    let key = service.get_provider_key(&provider).await;
    let unmask = query.unmask.unwrap_or(false);
    if unmask && key.is_some() && !query.yes_i_understand_this_prints_secrets.unwrap_or(false) {
        return Err(secret_confirmation_error("config provider get --unmask"));
    }
    Ok(axum::Json(CredentialValueDto {
        name: provider,
        value: key
            .as_deref()
            .map(|value| CredentialService::format_credential_value(value, unmask)),
        masked: !unmask,
        configured: key.is_some(),
    }))
}

pub(crate) async fn config_set_provider_handler(
    axum::Json(request): axum::Json<SetCredentialRequest>,
) -> Result<axum::Json<MutationResponse>, (StatusCode, String)> {
    let provider = request
        .provider
        .clone()
        .or_else(|| request.name.clone())
        .ok_or_else(|| (StatusCode::BAD_REQUEST, "provider is required".to_string()))?;
    let value = credential_value_from_request(&request)?;
    let service = CredentialService::new().await.map_err(internal_error)?;
    service
        .set_provider_key(&provider, &value)
        .await
        .map_err(internal_error)?;
    Ok(axum::Json(MutationResponse { changed: true }))
}

pub(crate) async fn config_remove_provider_handler(
    AxumPath(provider): AxumPath<String>,
) -> Result<axum::Json<MutationResponse>, (StatusCode, String)> {
    let service = CredentialService::new().await.map_err(internal_error)?;
    let changed = service
        .remove_provider_key(&provider)
        .await
        .map_err(internal_error)?;
    Ok(axum::Json(MutationResponse { changed }))
}

pub(crate) async fn config_default_provider_handler()
-> Result<axum::Json<DefaultProviderResponse>, (StatusCode, String)> {
    let service = CredentialService::new().await.map_err(internal_error)?;
    Ok(axum::Json(DefaultProviderResponse {
        provider: service.default_provider().await,
    }))
}

pub(crate) async fn config_set_default_provider_handler(
    axum::Json(request): axum::Json<SetDefaultProviderRequest>,
) -> Result<axum::Json<DefaultProviderResponse>, (StatusCode, String)> {
    let service = CredentialService::new().await.map_err(internal_error)?;
    service
        .set_default_provider(&request.provider)
        .await
        .map_err(internal_error)?;
    Ok(axum::Json(DefaultProviderResponse {
        provider: Some(request.provider),
    }))
}

pub(crate) async fn config_set_default_model_handler(
    axum::Json(request): axum::Json<SetDefaultModelRequest>,
) -> Result<axum::Json<DefaultModelResponse>, (StatusCode, String)> {
    let service = CredentialService::new().await.map_err(internal_error)?;
    service
        .set_default_model(&request.provider, &request.model)
        .await
        .map_err(internal_error)?;
    Ok(axum::Json(DefaultModelResponse {
        provider: request.provider,
        model: request.model,
    }))
}

pub(crate) async fn config_gateway_credentials_handler(
    Query(query): Query<ConfiguredQuery>,
) -> Result<axum::Json<Vec<GatewayCredentialStatusDto>>, (StatusCode, String)> {
    let service = CredentialService::new().await.map_err(internal_error)?;
    let credentials = service
        .gateway_status_rows(query.configured.unwrap_or(false))
        .await
        .into_iter()
        .map(
            |(name, env_var, description, configured)| GatewayCredentialStatusDto {
                name: name.to_string(),
                description: description.to_string(),
                env_var: env_var.to_string(),
                configured,
            },
        )
        .collect();
    Ok(axum::Json(credentials))
}

pub(crate) async fn config_get_gateway_credential_handler(
    AxumPath(name): AxumPath<String>,
    Query(query): Query<SecretQuery>,
) -> Result<axum::Json<CredentialValueDto>, (StatusCode, String)> {
    ensure_known_gateway_credential(&name)?;
    config_get_named_credential(name, query).await
}

pub(crate) async fn config_set_gateway_credential_handler(
    axum::Json(request): axum::Json<SetCredentialRequest>,
) -> Result<axum::Json<MutationResponse>, (StatusCode, String)> {
    let name = request
        .name
        .clone()
        .or_else(|| request.provider.clone())
        .ok_or_else(|| (StatusCode::BAD_REQUEST, "name is required".to_string()))?
        .to_ascii_lowercase();
    ensure_known_gateway_credential(&name)?;
    let value = credential_value_from_request(&request)?;
    let service = CredentialService::new().await.map_err(internal_error)?;
    service
        .set_gateway_credential(&name, &value)
        .await
        .map_err(internal_error)?;
    Ok(axum::Json(MutationResponse { changed: true }))
}

pub(crate) async fn config_remove_gateway_credential_handler(
    AxumPath(name): AxumPath<String>,
) -> Result<axum::Json<MutationResponse>, (StatusCode, String)> {
    ensure_known_gateway_credential(&name)?;
    let service = CredentialService::new().await.map_err(internal_error)?;
    let changed = service
        .remove_provider_key(&name)
        .await
        .map_err(internal_error)?;
    Ok(axum::Json(MutationResponse { changed }))
}

pub(crate) async fn config_get_named_credential(
    name: String,
    query: SecretQuery,
) -> Result<axum::Json<CredentialValueDto>, (StatusCode, String)> {
    let service = CredentialService::new().await.map_err(internal_error)?;
    let key = service.get_provider_key(&name).await;
    let unmask = query.unmask.unwrap_or(false);
    if unmask && key.is_some() && !query.yes_i_understand_this_prints_secrets.unwrap_or(false) {
        return Err(secret_confirmation_error("config credential get --unmask"));
    }
    Ok(axum::Json(CredentialValueDto {
        name,
        value: key
            .as_deref()
            .map(|value| CredentialService::format_credential_value(value, unmask)),
        masked: !unmask,
        configured: key.is_some(),
    }))
}

pub(crate) async fn config_export_handler(
    AxumPath(shell): AxumPath<String>,
    Query(query): Query<ConfigExportQuery>,
) -> Result<Response, (StatusCode, String)> {
    if !query.yes_i_understand_this_prints_secrets.unwrap_or(false) {
        return Err(secret_confirmation_error("config export"));
    }
    validate_export_shell(&shell)?;
    let service = CredentialService::new().await.map_err(internal_error)?;
    let lines = service
        .build_export_lines(&shell)
        .await
        .map_err(internal_error)?;
    let body = format!("{}\n", lines.join("\n"));
    Ok(([(header::CONTENT_TYPE, "text/plain; charset=utf-8")], body).into_response())
}

pub(crate) fn credential_value_from_request(
    request: &SetCredentialRequest,
) -> Result<String, (StatusCode, String)> {
    match (&request.key, &request.env) {
        (Some(key), _) if !key.is_empty() => Ok(key.clone()),
        (_, Some(env_var)) => caliche_contracts::credential_env::var(env_var).map_err(|_| {
            (
                StatusCode::BAD_REQUEST,
                format!("environment variable not set: {env_var}"),
            )
        }),
        _ => Err((
            StatusCode::BAD_REQUEST,
            "key or env is required".to_string(),
        )),
    }
}

pub(crate) fn ensure_known_gateway_credential(name: &str) -> Result<(), (StatusCode, String)> {
    if CredentialService::is_known_gateway(name) {
        Ok(())
    } else {
        Err((
            StatusCode::BAD_REQUEST,
            format!("unknown gateway credential: {name}"),
        ))
    }
}

pub(crate) fn validate_export_shell(shell: &str) -> Result<(), (StatusCode, String)> {
    CredentialService::validate_shell(shell)
        .map_err(|error| (StatusCode::BAD_REQUEST, error.to_string()))
}

pub(crate) fn secret_confirmation_error(action: &str) -> (StatusCode, String) {
    (
        StatusCode::BAD_REQUEST,
        CredentialService::secret_confirmation_message(action),
    )
}
