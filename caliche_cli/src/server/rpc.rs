use axum::extract::{Path as AxumPath, Query};
use axum::http::{StatusCode, header};
use axum::response::Response;
use serde::Deserialize;
use serde::Serialize;

use crate::server::cluster::register_cli_gateway_rpc_methods;
use crate::server::routes::*;

pub(crate) async fn response_result_to_value(
    result: Result<Response, (StatusCode, String)>,
) -> Result<serde_json::Value, caliche_server::rpc::types::JsonRpcError> {
    let response = result.map_err(|(status, message)| rpc_handler_error(status, message))?;
    let status = response.status();
    let content_type = response
        .headers()
        .get(header::CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .map(ToOwned::to_owned);
    let bytes = axum::body::to_bytes(response.into_body(), 8 * 1024 * 1024)
        .await
        .map_err(|e| rpc_internal_error(e.to_string()))?;
    let body = String::from_utf8(bytes.to_vec()).map_err(|e| rpc_internal_error(e.to_string()))?;
    Ok(serde_json::json!({
        "status": status.as_u16(),
        "content_type": content_type,
        "body": body
    }))
}

pub(crate) fn to_rpc_value<T: Serialize>(
    result: Result<axum::Json<T>, (StatusCode, String)>,
) -> Result<serde_json::Value, caliche_server::rpc::types::JsonRpcError> {
    let axum::Json(value) =
        result.map_err(|(status, message)| rpc_handler_error(status, message))?;
    serde_json::to_value(value).map_err(rpc_serde_error)
}

pub(crate) fn parse_rpc_params<T: for<'de> Deserialize<'de>>(
    params: serde_json::Value,
) -> Result<T, caliche_server::rpc::types::JsonRpcError> {
    serde_json::from_value(params).map_err(|e| {
        caliche_server::rpc::types::JsonRpcError::invalid_params(Some(serde_json::json!({
            "error": e.to_string()
        })))
    })
}

pub(crate) fn required_string(
    params: &serde_json::Value,
    key: &str,
) -> Result<String, caliche_server::rpc::types::JsonRpcError> {
    params
        .get(key)
        .and_then(serde_json::Value::as_str)
        .filter(|value| !value.trim().is_empty())
        .map(ToOwned::to_owned)
        .ok_or_else(|| {
            caliche_server::rpc::types::JsonRpcError::invalid_params(Some(serde_json::json!({
                "missing": key
            })))
        })
}

pub(crate) fn rpc_handler_error(
    status: StatusCode,
    message: String,
) -> caliche_server::rpc::types::JsonRpcError {
    caliche_server::rpc::types::JsonRpcError::server_error(
        -32000 - i32::from(status.as_u16()),
        format!("HTTP {}: {}", status.as_u16(), message),
        Some(serde_json::json!({ "status": status.as_u16() })),
    )
}

pub(crate) fn rpc_internal_error(
    error: impl std::fmt::Display,
) -> caliche_server::rpc::types::JsonRpcError {
    caliche_server::rpc::types::JsonRpcError::internal_error(Some(serde_json::json!({
        "error": error.to_string()
    })))
}

pub(crate) fn rpc_serde_error(
    error: serde_json::Error,
) -> caliche_server::rpc::types::JsonRpcError {
    rpc_internal_error(error)
}
pub(crate) fn register_cli_rpc_methods(router: &mut caliche_server::rpc::dispatch::MethodRouter) {
    use caliche_server::rpc::dispatch::MethodKind;
    use serde_json::json;

    router.register(
        "cli.completions",
        MethodKind::RequestResponse,
        |params, _id, _ctx| async move {
            let shell = required_string(&params, "shell")?;
            response_result_to_value(completions_handler(AxumPath(shell)).await).await
        },
    );
    router.register("cli.man", MethodKind::RequestResponse, |_params, _id, _ctx| async move {
        response_result_to_value(man_page_handler().await).await
    });
    router.register(
        "cli.manAll",
        MethodKind::RequestResponse,
        |_params, _id, _ctx| async move { to_rpc_value(man_pages_handler().await) },
    );
    router.register(
        "cli.lsp.status",
        MethodKind::RequestResponse,
        |_params, _id, _ctx| async move { to_rpc_value(lsp_status_handler().await) },
    );
    router.register(
        "cli.lsp.languages",
        MethodKind::RequestResponse,
        |_params, _id, _ctx| async move { to_rpc_value(lsp_languages_handler().await) },
    );
    router.register(
        "cli.lsp.install",
        MethodKind::RequestResponse,
        |params, _id, _ctx| async move {
            let request = parse_rpc_params::<LspInstallRequest>(params)?;
            to_rpc_value(lsp_install_handler(axum::Json(request)).await)
        },
    );
    router.register(
        "cli.agents.list",
        MethodKind::RequestResponse,
        |_params, _id, _ctx| async move { to_rpc_value(agent_profiles_handler().await) },
    );
    router.register(
        "cli.agents.get",
        MethodKind::RequestResponse,
        |params, _id, _ctx| async move {
            let name = required_string(&params, "name")?;
            to_rpc_value(agent_profile_handler(AxumPath(name)).await)
        },
    );
    router.register(
        "cli.doctor",
        MethodKind::RequestResponse,
        |_params, _id, _ctx| async move { response_result_to_value(doctor_handler().await).await },
    );

    register_cli_config_rpc_methods(router);
    register_cli_vault_rpc_methods(router);
    register_cli_gateway_rpc_methods(router);

    router.register(
        "cli.routes",
        MethodKind::RequestResponse,
        |_params, _id, _ctx| async move {
            Ok(json!({
                "http_routes": [
                    "/api/cli/completions/{shell}",
                    "/api/cli/man",
                    "/api/cli/man/all",
                    "/api/cli/lsp/status",
                    "/api/cli/lsp/languages",
                    "/api/cli/lsp/install",
                    "/api/cli/agents",
                    "/api/cli/agents/{name}",
                    "/api/cli/doctor",
                    "/api/cli/config",
                    "/api/cli/config/schema",
                    "/api/cli/config/path",
                    "/api/cli/config/providers",
                    "/api/cli/config/provider",
                    "/api/cli/config/provider/{provider}",
                    "/api/cli/config/default-provider",
                    "/api/cli/config/default-model",
                    "/api/cli/config/gateway",
                    "/api/cli/config/gateway/{name}",
                    "/api/cli/config/export/{shell}",
                    "/api/cli/vault/status",
                    "/api/cli/vault/scopes",
                    "/api/cli/vault/credentials",
                    "/api/cli/vault/link",
                    "/api/cli/vault/unlink",
                    "/api/cli/vault/import",
                    "/api/cli/vault/promote"
                ],
                "rpc_only_methods": [
                    "cli.gateway.runtime.start",
                    "cli.gateway.runtime.stop",
                    "cli.gateway.runtime.stopAll",
                    "cli.gateway.runtime.restart",
                    "cli.gateway.runtime.status",
                    "cli.gateway.runtime.list",
                    "cli.gateway.forward.start",
                    "cli.gateway.forward.stop",
                    "cli.gateway.forward.restart",
                    "cli.gateway.forward.status",
                    "cli.gateway.forward.list",
                    "cli.gateway.logs",
                    "cli.gateway.logs.latest",
                    "cli.gateway.doctor",
                    "cli.gateway.llm.status",
                    "cli.gateway.llm.set",
                    "cli.gateway.llm.clear",
                    "cli.gateway.route.list",
                    "cli.gateway.route.pin",
                    "cli.gateway.route.unpin"
                ]
            }))
        },
    );
}

pub(crate) fn register_cli_config_rpc_methods(
    router: &mut caliche_server::rpc::dispatch::MethodRouter,
) {
    use caliche_server::rpc::dispatch::MethodKind;
    use serde_json::json;

    router.register(
        "cli.config.get",
        MethodKind::RequestResponse,
        |params, _id, _ctx| async move {
            let query = parse_rpc_params::<ConfigShowQuery>(params)?;
            response_result_to_value(config_show_handler(Query(query)).await).await
        },
    );
    router.register(
        "cli.config.schema",
        MethodKind::RequestResponse,
        |_params, _id, _ctx| async move {
            response_result_to_value(config_schema_handler().await).await
        },
    );
    router.register(
        "cli.config.path",
        MethodKind::RequestResponse,
        |_params, _id, _ctx| async move { response_result_to_value(config_path_handler().await).await },
    );
    router.register(
        "cli.config.providers",
        MethodKind::RequestResponse,
        |params, _id, _ctx| async move {
            let query = parse_rpc_params::<ConfiguredQuery>(params)?;
            to_rpc_value(config_providers_handler(Query(query)).await)
        },
    );
    router.register(
        "cli.config.provider.get",
        MethodKind::RequestResponse,
        |params, _id, _ctx| async move {
            let provider = required_string(&params, "provider")?;
            let query = parse_rpc_params::<SecretQuery>(params)?;
            to_rpc_value(config_get_provider_handler(AxumPath(provider), Query(query)).await)
        },
    );
    router.register(
        "cli.config.provider.set",
        MethodKind::RequestResponse,
        |params, _id, _ctx| async move {
            let request = parse_rpc_params::<SetCredentialRequest>(params)?;
            to_rpc_value(config_set_provider_handler(axum::Json(request)).await)
        },
    );
    router.register(
        "cli.config.provider.remove",
        MethodKind::RequestResponse,
        |params, _id, _ctx| async move {
            let provider = required_string(&params, "provider")?;
            to_rpc_value(config_remove_provider_handler(AxumPath(provider)).await)
        },
    );
    router.register(
        "cli.config.defaultProvider.get",
        MethodKind::RequestResponse,
        |_params, _id, _ctx| async move { to_rpc_value(config_default_provider_handler().await) },
    );
    router.register(
        "cli.config.defaultProvider.set",
        MethodKind::RequestResponse,
        |params, _id, _ctx| async move {
            let request = parse_rpc_params::<SetDefaultProviderRequest>(params)?;
            to_rpc_value(config_set_default_provider_handler(axum::Json(request)).await)
        },
    );
    router.register(
        "cli.config.defaultModel.set",
        MethodKind::RequestResponse,
        |params, _id, _ctx| async move {
            let request = parse_rpc_params::<SetDefaultModelRequest>(params)?;
            to_rpc_value(config_set_default_model_handler(axum::Json(request)).await)
        },
    );
    router.register(
        "cli.config.gatewayCredentials",
        MethodKind::RequestResponse,
        |params, _id, _ctx| async move {
            let query = parse_rpc_params::<ConfiguredQuery>(params)?;
            to_rpc_value(config_gateway_credentials_handler(Query(query)).await)
        },
    );
    router.register(
        "cli.config.gatewayCredential.get",
        MethodKind::RequestResponse,
        |params, _id, _ctx| async move {
            let name = required_string(&params, "name")?;
            let query = parse_rpc_params::<SecretQuery>(params)?;
            to_rpc_value(config_get_gateway_credential_handler(AxumPath(name), Query(query)).await)
        },
    );
    router.register(
        "cli.config.gatewayCredential.set",
        MethodKind::RequestResponse,
        |params, _id, _ctx| async move {
            let request = parse_rpc_params::<SetCredentialRequest>(params)?;
            to_rpc_value(config_set_gateway_credential_handler(axum::Json(request)).await)
        },
    );
    router.register(
        "cli.config.gatewayCredential.remove",
        MethodKind::RequestResponse,
        |params, _id, _ctx| async move {
            let name = required_string(&params, "name")?;
            to_rpc_value(config_remove_gateway_credential_handler(AxumPath(name)).await)
        },
    );
    router.register(
        "cli.config.export",
        MethodKind::RequestResponse,
        |params, _id, _ctx| async move {
            let shell = params
                .get("shell")
                .and_then(serde_json::Value::as_str)
                .unwrap_or("bash")
                .to_string();
            let query = parse_rpc_params::<ConfigExportQuery>(params)?;
            response_result_to_value(config_export_handler(AxumPath(shell), Query(query)).await)
                .await
        },
    );
    router.register(
        "cli.config.methods",
        MethodKind::RequestResponse,
        |_params, _id, _ctx| async move {
            Ok(json!([
                "cli.config.get",
                "cli.config.schema",
                "cli.config.path",
                "cli.config.providers",
                "cli.config.provider.get",
                "cli.config.provider.set",
                "cli.config.provider.remove",
                "cli.config.defaultProvider.get",
                "cli.config.defaultProvider.set",
                "cli.config.defaultModel.set",
                "cli.config.gatewayCredentials",
                "cli.config.gatewayCredential.get",
                "cli.config.gatewayCredential.set",
                "cli.config.gatewayCredential.remove",
                "cli.config.export"
            ]))
        },
    );
}

pub(crate) fn register_cli_vault_rpc_methods(
    router: &mut caliche_server::rpc::dispatch::MethodRouter,
) {
    use caliche_server::rpc::dispatch::MethodKind;

    router.register(
        "cli.vault.status",
        MethodKind::RequestResponse,
        |_params, _id, _ctx| async move { to_rpc_value(vault_status_handler().await) },
    );
    router.register(
        "cli.vault.scopes",
        MethodKind::RequestResponse,
        |_params, _id, _ctx| async move { to_rpc_value(vault_scopes_handler().await) },
    );
    router.register(
        "cli.vault.credentials",
        MethodKind::RequestResponse,
        |params, _id, _ctx| async move {
            let query = parse_rpc_params::<VaultScopeQuery>(params)?;
            to_rpc_value(vault_credentials_handler(Query(query)).await)
        },
    );
    router.register(
        "cli.vault.link",
        MethodKind::RequestResponse,
        |params, _id, _ctx| async move {
            let request = parse_rpc_params::<VaultLinkRequest>(params)?;
            to_rpc_value(vault_link_handler(axum::Json(request)).await)
        },
    );
    router.register(
        "cli.vault.unlink",
        MethodKind::RequestResponse,
        |params, _id, _ctx| async move {
            let request = parse_rpc_params::<VaultLinkRequest>(params)?;
            to_rpc_value(vault_unlink_handler(axum::Json(request)).await)
        },
    );
    router.register(
        "cli.vault.import",
        MethodKind::RequestResponse,
        |params, _id, _ctx| async move {
            let request = parse_rpc_params::<VaultImportRequest>(params)?;
            to_rpc_value(vault_import_handler(axum::Json(request)).await)
        },
    );
    router.register(
        "cli.vault.promote",
        MethodKind::RequestResponse,
        |params, _id, _ctx| async move {
            let request = parse_rpc_params::<VaultPromoteRequest>(params)?;
            to_rpc_value(vault_promote_handler(axum::Json(request)).await)
        },
    );
}
