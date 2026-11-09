use axum::extract::Query;
use axum::http::StatusCode;
use serde::{Deserialize, Serialize};

use crate::commands::credentials_service::CredentialService;

use super::shared::{bad_request, internal_error, resolve_dir};

#[derive(Debug, Serialize)]
pub(crate) struct VaultStatusResponse {
    cwd: String,
    sources: Vec<VaultSourceDto>,
    env_providers: Vec<String>,
}

#[derive(Debug, Serialize)]
pub(crate) struct VaultSourceDto {
    label: String,
    hash: String,
    providers: Vec<String>,
    is_linked: bool,
}

#[derive(Debug, Serialize)]
pub(crate) struct VaultScopeDto {
    path: String,
    hash: String,
    label: Option<String>,
    providers: Vec<String>,
    updated_at: String,
}

#[derive(Debug, Serialize)]
pub(crate) struct VaultCredentialsResponse {
    scope: String,
    hash: String,
    credentials: Vec<VaultCredentialDto>,
}

#[derive(Debug, Serialize)]
pub(crate) struct VaultCredentialDto {
    provider: String,
    masked_key: String,
}

#[derive(Debug, Deserialize)]
pub(crate) struct VaultScopeQuery {
    scope: Option<String>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct VaultLinkRequest {
    source_dir: String,
    provider: Option<String>,
}

#[derive(Debug, Serialize)]
pub(crate) struct VaultLinkResponse {
    source_dir: String,
    provider: String,
    target_scope: String,
    removed: Option<bool>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct VaultImportRequest {
    dir: Option<String>,
    parents: Option<bool>,
    destination: Option<VaultImportDestination>,
    /// Must be `true` to import credentials over HTTP (mirrors the interactive `[G/d/n]` prompt).
    confirm: Option<bool>,
}

#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum VaultImportDestination {
    Global,
    Directory,
}

#[derive(Debug, Serialize)]
pub(crate) struct VaultImportResponse {
    dir: String,
    destination: String,
    detected: Vec<VaultDetectedCredentialDto>,
    stored: usize,
}

#[derive(Debug, Serialize)]
pub(crate) struct VaultDetectedCredentialDto {
    provider: String,
    env_var: String,
    source_path: String,
}

#[derive(Debug, Deserialize)]
pub(crate) struct VaultPromoteRequest {
    provider: String,
}

#[derive(Debug, Serialize)]
pub(crate) struct VaultPromoteResponse {
    provider: String,
    promoted: bool,
}

pub(crate) async fn vault_status_handler()
-> Result<axum::Json<VaultStatusResponse>, (StatusCode, String)> {
    let service = CredentialService::new().await.map_err(internal_error)?;
    let status = service.vault_status().await;
    Ok(axum::Json(VaultStatusResponse {
        cwd: status.cwd,
        sources: status
            .sources
            .into_iter()
            .map(|source| VaultSourceDto {
                label: source.label,
                hash: source.hash,
                providers: source.providers,
                is_linked: source.is_linked,
            })
            .collect(),
        env_providers: status.env_providers,
    }))
}

pub(crate) async fn vault_scopes_handler()
-> Result<axum::Json<Vec<VaultScopeDto>>, (StatusCode, String)> {
    let service = CredentialService::new().await.map_err(internal_error)?;
    let scopes = service
        .vault_list_scopes()
        .await
        .into_iter()
        .map(|scope| VaultScopeDto {
            path: scope.path,
            hash: scope.hash,
            label: scope.label,
            providers: scope.providers,
            updated_at: scope.updated_at,
        })
        .collect();
    Ok(axum::Json(scopes))
}

pub(crate) async fn vault_credentials_handler(
    Query(query): Query<VaultScopeQuery>,
) -> Result<axum::Json<VaultCredentialsResponse>, (StatusCode, String)> {
    let service = CredentialService::new().await.map_err(internal_error)?;
    let (scope_label, scope_hash) = if let Some(scope) = query.scope {
        let hash = CredentialService::scope_hash_for_path(&scope);
        (scope, hash)
    } else {
        (
            service.current_scope_path(),
            service.current_scope_hash().to_string(),
        )
    };

    let mut credentials: Vec<VaultCredentialDto> = service
        .vault_scope_credentials(&scope_hash)
        .into_iter()
        .map(|(provider, key)| VaultCredentialDto {
            provider,
            masked_key: CredentialService::mask_for_display(&key),
        })
        .collect();
    credentials.sort_by(|a, b| a.provider.cmp(&b.provider));

    Ok(axum::Json(VaultCredentialsResponse {
        scope: scope_label,
        hash: scope_hash,
        credentials,
    }))
}

pub(crate) async fn vault_link_handler(
    axum::Json(request): axum::Json<VaultLinkRequest>,
) -> Result<axum::Json<VaultLinkResponse>, (StatusCode, String)> {
    let source_path = resolve_dir(&request.source_dir).map_err(bad_request)?;
    let source_dir = source_path.to_string_lossy().to_string();
    let provider = request.provider.unwrap_or_else(|| "*".to_string());
    let service = CredentialService::new().await.map_err(internal_error)?;
    service
        .vault_link(&source_dir, &provider)
        .await
        .map_err(internal_error)?;
    let target_scope = service.current_scope_path();
    Ok(axum::Json(VaultLinkResponse {
        source_dir,
        provider,
        target_scope,
        removed: None,
    }))
}

pub(crate) async fn vault_unlink_handler(
    axum::Json(request): axum::Json<VaultLinkRequest>,
) -> Result<axum::Json<VaultLinkResponse>, (StatusCode, String)> {
    let source_path = resolve_dir(&request.source_dir).map_err(bad_request)?;
    let source_dir = source_path.to_string_lossy().to_string();
    let provider = request.provider;
    let service = CredentialService::new().await.map_err(internal_error)?;
    let removed = service
        .vault_unlink(&source_dir, provider.as_deref())
        .await
        .map_err(internal_error)?;
    let target_scope = service.current_scope_path();
    Ok(axum::Json(VaultLinkResponse {
        source_dir,
        provider: provider.unwrap_or_else(|| "*".to_string()),
        target_scope,
        removed: Some(removed),
    }))
}

pub(crate) async fn vault_import_handler(
    axum::Json(request): axum::Json<VaultImportRequest>,
) -> Result<axum::Json<VaultImportResponse>, (StatusCode, String)> {
    CredentialService::validate_vault_import_confirm(request.confirm)
        .map_err(|e| (StatusCode::BAD_REQUEST, e.to_string()))?;

    let dir = match request.dir {
        Some(dir) => resolve_dir(&dir).map_err(bad_request)?,
        None => std::env::current_dir().map_err(internal_error)?,
    };
    let detected =
        crate::storage::env_scanner::scan_for_credentials(&dir, request.parents.unwrap_or(false));
    let destination = request
        .destination
        .unwrap_or(VaultImportDestination::Global);
    let stored = if detected.is_empty() {
        0
    } else {
        match destination {
            VaultImportDestination::Global => {
                let service = CredentialService::new().await.map_err(internal_error)?;
                service
                    .vault_import_global(&detected)
                    .await
                    .map_err(internal_error)?
            }
            VaultImportDestination::Directory => {
                let service = CredentialService::new_with_scope(Some(&dir))
                    .await
                    .map_err(internal_error)?;
                service
                    .vault_import_directory(&detected)
                    .await
                    .map_err(internal_error)?
            }
        }
    };

    Ok(axum::Json(VaultImportResponse {
        dir: dir.display().to_string(),
        destination: match destination {
            VaultImportDestination::Global => "global".to_string(),
            VaultImportDestination::Directory => "directory".to_string(),
        },
        detected: detected
            .into_iter()
            .map(|credential| VaultDetectedCredentialDto {
                provider: credential.provider_name,
                env_var: credential.env_var,
                source_path: credential.source_path.display().to_string(),
            })
            .collect(),
        stored,
    }))
}

pub(crate) async fn vault_promote_handler(
    axum::Json(request): axum::Json<VaultPromoteRequest>,
) -> Result<axum::Json<VaultPromoteResponse>, (StatusCode, String)> {
    let service = CredentialService::new().await.map_err(internal_error)?;
    let promoted = service
        .vault_promote(&request.provider)
        .await
        .map_err(internal_error)?;
    Ok(axum::Json(VaultPromoteResponse {
        provider: request.provider,
        promoted,
    }))
}
