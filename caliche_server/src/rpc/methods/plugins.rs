use crate::handlers;
use crate::rpc::connection::ConnectionCtx;
use crate::rpc::dispatch::{MethodKind, MethodRouter};
use crate::rpc::helpers::{map_handler_err, parse_params, require_param_str, to_rpc_value};
use crate::rpc::types::JsonRpcError;
use axum::Json;
use axum::extract::{Path, State};
use serde_json::{Value, json};

pub fn register(router: &mut MethodRouter) {
    register_plugins(router);
    register_mcp(router);
    register_endpoints(router);
}

fn register_plugins(router: &mut MethodRouter) {
    router.register(
        "plugins.list",
        MethodKind::RequestResponse,
        |_params, _id, ctx: ConnectionCtx| async move { plugins_list(ctx).await },
    );

    router.register(
        "plugins.getCatalog",
        MethodKind::RequestResponse,
        |_params, _id, ctx: ConnectionCtx| async move { plugins_get_catalog(ctx).await },
    );

    router.register(
        "plugins.get",
        MethodKind::RequestResponse,
        |params, _id, ctx: ConnectionCtx| async move { plugins_get(params, ctx).await },
    );

    router.register(
        "plugins.export",
        MethodKind::RequestResponse,
        |params, _id, ctx: ConnectionCtx| async move { plugins_export(params, ctx).await },
    );

    router.register(
        "plugins.validate",
        MethodKind::RequestResponse,
        |params, _id, ctx: ConnectionCtx| async move { plugins_validate(params, ctx).await },
    );

    router.register(
        "plugins.save",
        MethodKind::RequestResponse,
        |params, _id, ctx: ConnectionCtx| async move { plugins_save(params, ctx).await },
    );

    router.register(
        "plugins.delete",
        MethodKind::RequestResponse,
        |params, _id, ctx: ConnectionCtx| async move { plugins_delete(params, ctx).await },
    );
}

fn register_mcp(router: &mut MethodRouter) {
    router.register(
        "mcp.getConfig",
        MethodKind::RequestResponse,
        |_params, _id, ctx: ConnectionCtx| async move { mcp_get_config(ctx).await },
    );

    router.register(
        "mcp.updateConfig",
        MethodKind::RequestResponse,
        |params, _id, ctx: ConnectionCtx| async move { mcp_update_config(params, ctx).await },
    );

    router.register(
        "mcp.testConfig",
        MethodKind::RequestResponse,
        |params, _id, ctx: ConnectionCtx| async move { mcp_test_config(params, ctx).await },
    );

    #[cfg(feature = "mcp")]
    {
        router.register(
            "mcp.capabilities",
            MethodKind::RequestResponse,
            |_params, _id, ctx: ConnectionCtx| async move { mcp_capabilities(ctx).await },
        );
        router.register(
            "mcp.tools",
            MethodKind::RequestResponse,
            |_params, _id, ctx: ConnectionCtx| async move { mcp_tools(ctx).await },
        );
        router.register(
            "mcp.prompts",
            MethodKind::RequestResponse,
            |_params, _id, ctx: ConnectionCtx| async move { mcp_prompts(ctx).await },
        );
        router.register(
            "mcp.resources",
            MethodKind::RequestResponse,
            |_params, _id, ctx: ConnectionCtx| async move { mcp_resources(ctx).await },
        );
    }
}

fn register_endpoints(router: &mut MethodRouter) {
    router.register(
        "endpoints.list",
        MethodKind::RequestResponse,
        |_params, _id, ctx: ConnectionCtx| async move { endpoints_list(ctx).await },
    );

    router.register(
        "endpoints.create",
        MethodKind::RequestResponse,
        |params, _id, ctx: ConnectionCtx| async move { endpoints_create(params, ctx).await },
    );

    router.register(
        "endpoints.get",
        MethodKind::RequestResponse,
        |params, _id, ctx: ConnectionCtx| async move { endpoints_get(params, ctx).await },
    );

    router.register(
        "endpoints.start",
        MethodKind::RequestResponse,
        |params, _id, ctx: ConnectionCtx| async move { endpoints_start(params, ctx).await },
    );

    router.register(
        "endpoints.stop",
        MethodKind::RequestResponse,
        |params, _id, ctx: ConnectionCtx| async move { endpoints_stop(params, ctx).await },
    );

    router.register(
        "endpoints.delete",
        MethodKind::RequestResponse,
        |params, _id, ctx: ConnectionCtx| async move { endpoints_delete(params, ctx).await },
    );
}

async fn plugins_list(ctx: ConnectionCtx) -> Result<Value, JsonRpcError> {
    let Json(response) = handlers::plugins::list_plugins_handler(State(ctx.app_state))
        .await
        .map_err(|(status, message)| map_handler_err(status, message))?;
    to_rpc_value(response)
}

async fn plugins_get_catalog(ctx: ConnectionCtx) -> Result<Value, JsonRpcError> {
    let Json(response) = handlers::plugins::plugin_catalog_handler(State(ctx.app_state))
        .await
        .map_err(|(status, message)| map_handler_err(status, message))?;
    to_rpc_value(response)
}

async fn plugins_get(params: Value, ctx: ConnectionCtx) -> Result<Value, JsonRpcError> {
    let name = require_param_str(&params, "name")?;
    let Json(response) = handlers::plugins::get_plugin_handler(State(ctx.app_state), Path(name))
        .await
        .map_err(|(status, message)| map_handler_err(status, message))?;
    to_rpc_value(response)
}

async fn plugins_export(params: Value, ctx: ConnectionCtx) -> Result<Value, JsonRpcError> {
    let name = require_param_str(&params, "name")?;
    let Json(response) = handlers::plugins::export_plugin_handler(State(ctx.app_state), Path(name))
        .await
        .map_err(|(status, message)| map_handler_err(status, message))?;
    to_rpc_value(response)
}

async fn plugins_validate(params: Value, ctx: ConnectionCtx) -> Result<Value, JsonRpcError> {
    let request = parse_params::<handlers::plugins::PluginDefinitionRequest>(params)?;
    let Json(response) =
        handlers::plugins::validate_plugin_handler(State(ctx.app_state), Json(request))
            .await
            .map_err(|(status, message)| map_handler_err(status, message))?;
    to_rpc_value(response)
}

async fn plugins_save(params: Value, ctx: ConnectionCtx) -> Result<Value, JsonRpcError> {
    let request = parse_params::<handlers::plugins::PluginDefinitionRequest>(params)?;
    let Json(response) =
        handlers::plugins::save_plugin_handler(State(ctx.app_state), Json(request))
            .await
            .map_err(|(status, message)| map_handler_err(status, message))?;
    to_rpc_value(response)
}

async fn plugins_delete(params: Value, ctx: ConnectionCtx) -> Result<Value, JsonRpcError> {
    let name = require_param_str(&params, "name")?;
    let status = handlers::plugins::delete_plugin_handler(State(ctx.app_state), Path(name))
        .await
        .map_err(|(status, message)| map_handler_err(status, message))?;
    Ok(json!({"status": status.as_u16()}))
}

async fn mcp_get_config(ctx: ConnectionCtx) -> Result<Value, JsonRpcError> {
    let Json(response) = handlers::mcp_config::get_mcp_config_handler(State(ctx.app_state))
        .await
        .map_err(|(status, message)| map_handler_err(status, message))?;
    to_rpc_value(response)
}

async fn mcp_update_config(params: Value, ctx: ConnectionCtx) -> Result<Value, JsonRpcError> {
    let request = parse_params::<handlers::mcp_config::McpConfigResponse>(params)?;
    let Json(response) =
        handlers::mcp_config::put_mcp_config_handler(State(ctx.app_state), Json(request))
            .await
            .map_err(|(status, message)| map_handler_err(status, message))?;
    to_rpc_value(response)
}

async fn mcp_test_config(params: Value, ctx: ConnectionCtx) -> Result<Value, JsonRpcError> {
    let request = parse_params::<handlers::mcp_config::McpServerConfigPayload>(params)?;
    let Json(response) =
        handlers::mcp_config::test_mcp_config_handler(State(ctx.app_state), Json(request))
            .await
            .map_err(|(status, message)| map_handler_err(status, message))?;
    to_rpc_value(response)
}

#[cfg(feature = "mcp")]
async fn mcp_capabilities(ctx: ConnectionCtx) -> Result<Value, JsonRpcError> {
    let Json(response) = handlers::mcp::capabilities_handler(State(ctx.app_state))
        .await
        .map_err(|(status, message)| map_handler_err(status, message))?;
    to_rpc_value(response)
}

#[cfg(feature = "mcp")]
async fn mcp_tools(ctx: ConnectionCtx) -> Result<Value, JsonRpcError> {
    let Json(response) = handlers::mcp::tools_handler(State(ctx.app_state))
        .await
        .map_err(|(status, message)| map_handler_err(status, message))?;
    to_rpc_value(response)
}

#[cfg(feature = "mcp")]
async fn mcp_prompts(ctx: ConnectionCtx) -> Result<Value, JsonRpcError> {
    let Json(response) = handlers::mcp::prompts_handler(State(ctx.app_state))
        .await
        .map_err(|(status, message)| map_handler_err(status, message))?;
    to_rpc_value(response)
}

#[cfg(feature = "mcp")]
async fn mcp_resources(ctx: ConnectionCtx) -> Result<Value, JsonRpcError> {
    let Json(response) = handlers::mcp::resources_handler(State(ctx.app_state))
        .await
        .map_err(|(status, message)| map_handler_err(status, message))?;
    to_rpc_value(response)
}

async fn endpoints_list(ctx: ConnectionCtx) -> Result<Value, JsonRpcError> {
    let Json(response) = handlers::endpoints::list_endpoints(State(ctx.app_state))
        .await
        .map_err(|(status, message)| map_handler_err(status, message))?;
    to_rpc_value(response)
}

async fn endpoints_create(params: Value, ctx: ConnectionCtx) -> Result<Value, JsonRpcError> {
    let request = parse_params::<handlers::endpoints::CreateEndpointRequest>(params)?;
    let Json(response) = handlers::endpoints::create_endpoint(State(ctx.app_state), Json(request))
        .await
        .map_err(|(status, message)| map_handler_err(status, message))?;
    to_rpc_value(response)
}

async fn endpoints_get(params: Value, ctx: ConnectionCtx) -> Result<Value, JsonRpcError> {
    let id = require_param_str(&params, "id")?;
    let Json(response) = handlers::endpoints::get_endpoint(State(ctx.app_state), Path(id))
        .await
        .map_err(|(status, message)| map_handler_err(status, message))?;
    to_rpc_value(response)
}

async fn endpoints_start(params: Value, ctx: ConnectionCtx) -> Result<Value, JsonRpcError> {
    let id = require_param_str(&params, "id")?;
    let Json(response) = handlers::endpoints::start_endpoint(State(ctx.app_state), Path(id))
        .await
        .map_err(|(status, message)| map_handler_err(status, message))?;
    to_rpc_value(response)
}

async fn endpoints_stop(params: Value, ctx: ConnectionCtx) -> Result<Value, JsonRpcError> {
    let id = require_param_str(&params, "id")?;
    let Json(response) = handlers::endpoints::stop_endpoint(State(ctx.app_state), Path(id))
        .await
        .map_err(|(status, message)| map_handler_err(status, message))?;
    to_rpc_value(response)
}

async fn endpoints_delete(params: Value, ctx: ConnectionCtx) -> Result<Value, JsonRpcError> {
    let id = require_param_str(&params, "id")?;
    let Json(response) = handlers::endpoints::delete_endpoint(State(ctx.app_state), Path(id))
        .await
        .map_err(|(status, message)| map_handler_err(status, message))?;
    Ok(response)
}
