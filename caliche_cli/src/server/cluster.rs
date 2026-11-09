use std::path::PathBuf;

use caliche_server::rpc::connection::ConnectionCtx;
use caliche_server::rpc::dispatch::{MethodKind, MethodRouter};
use caliche_server::rpc::types::JsonRpcError;
use serde_json::{Value, json};

use crate::server::rpc::{required_string, rpc_internal_error, rpc_serde_error};

type RpcResult = Result<Value, JsonRpcError>;

pub(crate) fn register_cli_gateway_rpc_methods(router: &mut MethodRouter) {
    register_gateway_runtime_rpc_methods(router);
    register_gateway_forward_rpc_methods(router);
    register_gateway_ops_rpc_methods(router);
    register_gateway_llm_rpc_methods(router);
    register_gateway_route_rpc_methods(router);
}

fn register_gateway_runtime_rpc_methods(router: &mut MethodRouter) {
    router.register(
        "cli.gateway.runtime.start",
        MethodKind::RequestResponse,
        rpc_runtime_start,
    );
    router.register(
        "cli.gateway.runtime.stop",
        MethodKind::RequestResponse,
        rpc_runtime_stop,
    );
    router.register(
        "cli.gateway.runtime.stopAll",
        MethodKind::RequestResponse,
        rpc_runtime_stop_all,
    );
    router.register(
        "cli.gateway.runtime.restart",
        MethodKind::RequestResponse,
        rpc_runtime_restart,
    );
    router.register(
        "cli.gateway.runtime.status",
        MethodKind::RequestResponse,
        rpc_runtime_status,
    );
    router.register(
        "cli.gateway.runtime.list",
        MethodKind::RequestResponse,
        rpc_runtime_list,
    );
}

fn register_gateway_forward_rpc_methods(router: &mut MethodRouter) {
    router.register(
        "cli.gateway.forward.start",
        MethodKind::RequestResponse,
        rpc_forward_start,
    );
    router.register(
        "cli.gateway.forward.stop",
        MethodKind::RequestResponse,
        rpc_forward_stop,
    );
    router.register(
        "cli.gateway.forward.restart",
        MethodKind::RequestResponse,
        rpc_forward_restart,
    );
    router.register(
        "cli.gateway.forward.status",
        MethodKind::RequestResponse,
        rpc_forward_status,
    );
    router.register(
        "cli.gateway.forward.list",
        MethodKind::RequestResponse,
        rpc_forward_list,
    );
}

fn register_gateway_ops_rpc_methods(router: &mut MethodRouter) {
    router.register(
        "cli.gateway.logs",
        MethodKind::RequestResponse,
        rpc_gateway_logs,
    );
    router.register(
        "cli.gateway.logs.latest",
        MethodKind::RequestResponse,
        rpc_gateway_logs_latest,
    );
    router.register(
        "cli.gateway.doctor",
        MethodKind::RequestResponse,
        rpc_gateway_doctor,
    );
}

fn register_gateway_llm_rpc_methods(router: &mut MethodRouter) {
    router.register(
        "cli.gateway.llm.status",
        MethodKind::RequestResponse,
        rpc_llm_status,
    );
    router.register(
        "cli.gateway.llm.set",
        MethodKind::RequestResponse,
        rpc_llm_set,
    );
    router.register(
        "cli.gateway.llm.clear",
        MethodKind::RequestResponse,
        rpc_llm_clear,
    );
}

fn register_gateway_route_rpc_methods(router: &mut MethodRouter) {
    router.register(
        "cli.gateway.route.list",
        MethodKind::RequestResponse,
        rpc_route_list,
    );
    router.register(
        "cli.gateway.route.pin",
        MethodKind::RequestResponse,
        rpc_route_pin,
    );
    router.register(
        "cli.gateway.route.unpin",
        MethodKind::RequestResponse,
        rpc_route_unpin,
    );
}

async fn rpc_runtime_start(params: Value, _id: Option<Value>, _ctx: ConnectionCtx) -> RpcResult {
    let (port, cwd) = gateway_port_and_cwd(&params)?;
    let status = crate::gateway_runtime::start_gateway_runtime(cwd.as_deref(), port)
        .await
        .map_err(rpc_internal_error)?;
    runtime_status_to_value(status.running, status.entry)
}

async fn rpc_runtime_stop(params: Value, _id: Option<Value>, _ctx: ConnectionCtx) -> RpcResult {
    let cwd = gateway_cwd(&params)?;
    let stopped = crate::gateway_runtime::stop_gateway_runtime(cwd.as_deref())
        .await
        .map_err(rpc_internal_error)?;
    Ok(json!({ "stopped": stopped }))
}

async fn rpc_runtime_stop_all(
    _params: Value,
    _id: Option<Value>,
    _ctx: ConnectionCtx,
) -> RpcResult {
    let stopped = crate::gateway_runtime::stop_all_gateway_runtimes()
        .await
        .map_err(rpc_internal_error)?;
    Ok(json!({ "stopped": stopped }))
}

async fn rpc_runtime_restart(params: Value, _id: Option<Value>, _ctx: ConnectionCtx) -> RpcResult {
    let (port, cwd) = gateway_port_and_cwd(&params)?;
    let status = crate::gateway_runtime::restart_gateway_runtime(cwd.as_deref(), port)
        .await
        .map_err(rpc_internal_error)?;
    runtime_status_to_value(status.running, status.entry)
}

async fn rpc_runtime_status(params: Value, _id: Option<Value>, _ctx: ConnectionCtx) -> RpcResult {
    let cwd = gateway_cwd(&params)?;
    let status = crate::gateway_runtime::status_gateway_runtime(cwd.as_deref())
        .await
        .map_err(rpc_internal_error)?;
    match status {
        Some(status) => runtime_status_to_value(status.running, status.entry),
        None => Ok(json!({ "running": false, "entry": null })),
    }
}

async fn rpc_runtime_list(_params: Value, _id: Option<Value>, _ctx: ConnectionCtx) -> RpcResult {
    let rows = crate::gateway_runtime::list_gateway_runtimes()
        .await
        .map_err(rpc_internal_error)?;
    let values = rows
        .into_iter()
        .map(|row| runtime_status_to_value(row.running, row.entry))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(json!(values))
}

async fn rpc_forward_start(params: Value, _id: Option<Value>, _ctx: ConnectionCtx) -> RpcResult {
    let (port, cwd) = gateway_port_and_cwd(&params)?;
    let provider = gateway_forward_provider(&params)?;
    let status = crate::gateway_runtime::start_gateway_forwarding(cwd.as_deref(), port, provider)
        .await
        .map_err(rpc_internal_error)?;
    forward_status_to_value(status)
}

async fn rpc_forward_stop(params: Value, _id: Option<Value>, _ctx: ConnectionCtx) -> RpcResult {
    let cwd = gateway_cwd(&params)?;
    let stopped = crate::gateway_runtime::stop_gateway_forwarding(cwd.as_deref())
        .await
        .map_err(rpc_internal_error)?;
    Ok(json!({ "stopped": stopped }))
}

async fn rpc_forward_restart(params: Value, _id: Option<Value>, _ctx: ConnectionCtx) -> RpcResult {
    let (port, cwd) = gateway_port_and_cwd(&params)?;
    let provider = gateway_forward_provider(&params)?;
    let status = crate::gateway_runtime::restart_gateway_forwarding(cwd.as_deref(), port, provider)
        .await
        .map_err(rpc_internal_error)?;
    forward_status_to_value(status)
}

async fn rpc_forward_status(params: Value, _id: Option<Value>, _ctx: ConnectionCtx) -> RpcResult {
    let cwd = gateway_cwd(&params)?;
    let status = crate::gateway_runtime::status_gateway_forwarding(cwd.as_deref())
        .await
        .map_err(rpc_internal_error)?;
    match status {
        Some(status) => forward_status_to_value(status),
        None => Ok(json!(null)),
    }
}

async fn rpc_forward_list(_params: Value, _id: Option<Value>, _ctx: ConnectionCtx) -> RpcResult {
    let rows = crate::gateway_runtime::list_gateway_forwardings()
        .await
        .map_err(rpc_internal_error)?;
    let values = rows
        .into_iter()
        .map(forward_status_to_value)
        .collect::<Result<Vec<_>, _>>()?;
    Ok(json!(values))
}

async fn rpc_gateway_logs(params: Value, _id: Option<Value>, _ctx: ConnectionCtx) -> RpcResult {
    let cwd = gateway_cwd(&params)?;
    let (lines, include_forwarding) = gateway_log_query(&params);
    let logs = crate::gateway_runtime::read_gateway_logs(cwd.as_deref(), lines, include_forwarding)
        .await
        .map_err(rpc_internal_error)?;
    serde_json::to_value(logs).map_err(rpc_serde_error)
}

async fn rpc_gateway_logs_latest(
    params: Value,
    _id: Option<Value>,
    _ctx: ConnectionCtx,
) -> RpcResult {
    let (lines, include_forwarding) = gateway_log_query(&params);
    let logs = crate::gateway_runtime::read_gateway_logs_fallback(lines, include_forwarding)
        .await
        .map_err(rpc_internal_error)?;
    serde_json::to_value(logs).map_err(rpc_serde_error)
}

async fn rpc_gateway_doctor(params: Value, _id: Option<Value>, _ctx: ConnectionCtx) -> RpcResult {
    let cwd = gateway_cwd(&params)?;
    let include_all = params
        .get("include_all")
        .or_else(|| params.get("all"))
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let rows = crate::gateway_runtime::doctor_gateway(cwd.as_deref(), include_all)
        .await
        .map_err(rpc_internal_error)?;
    serde_json::to_value(rows).map_err(rpc_serde_error)
}

async fn rpc_llm_status(params: Value, _id: Option<Value>, _ctx: ConnectionCtx) -> RpcResult {
    let cwd = gateway_cwd(&params)?;
    let status = crate::gateway_runtime::gateway_llm_config_status(cwd.as_deref())
        .await
        .map_err(rpc_internal_error)?;
    serde_json::to_value(status).map_err(rpc_serde_error)
}

async fn rpc_llm_set(params: Value, _id: Option<Value>, _ctx: ConnectionCtx) -> RpcResult {
    let cwd = gateway_cwd(&params)?;
    let provider = params.get("provider").and_then(Value::as_str);
    let model = params.get("model").and_then(Value::as_str);
    let status = crate::gateway_runtime::set_gateway_llm_config(cwd.as_deref(), provider, model)
        .await
        .map_err(rpc_internal_error)?;
    serde_json::to_value(status).map_err(rpc_serde_error)
}

async fn rpc_llm_clear(params: Value, _id: Option<Value>, _ctx: ConnectionCtx) -> RpcResult {
    let cwd = gateway_cwd(&params)?;
    let status = crate::gateway_runtime::clear_gateway_llm_config(cwd.as_deref())
        .await
        .map_err(rpc_internal_error)?;
    serde_json::to_value(status).map_err(rpc_serde_error)
}

async fn rpc_route_list(params: Value, _id: Option<Value>, _ctx: ConnectionCtx) -> RpcResult {
    let cwd = gateway_cwd(&params)?;
    let rows = crate::gateway_runtime::list_gateway_routes(cwd.as_deref())
        .await
        .map_err(rpc_internal_error)?;
    serde_json::to_value(rows).map_err(rpc_serde_error)
}

async fn rpc_route_pin(params: Value, _id: Option<Value>, _ctx: ConnectionCtx) -> RpcResult {
    let cwd = gateway_cwd(&params)?;
    let route_key = required_string(&params, "route_key")?;
    let target_agent_id = required_string(&params, "target_agent_id")?;
    crate::gateway_runtime::pin_gateway_route(cwd.as_deref(), &route_key, &target_agent_id)
        .await
        .map_err(rpc_internal_error)?;
    Ok(json!({ "changed": true }))
}

async fn rpc_route_unpin(params: Value, _id: Option<Value>, _ctx: ConnectionCtx) -> RpcResult {
    let cwd = gateway_cwd(&params)?;
    let route_key = required_string(&params, "route_key")?;
    crate::gateway_runtime::unpin_gateway_route(cwd.as_deref(), &route_key)
        .await
        .map_err(rpc_internal_error)?;
    Ok(json!({ "changed": true }))
}

fn gateway_log_query(params: &Value) -> (usize, bool) {
    let lines = params
        .get("lines")
        .and_then(Value::as_u64)
        .and_then(|v| usize::try_from(v).ok())
        .unwrap_or(200);
    let include_forwarding = params
        .get("include_forwarding")
        .and_then(Value::as_bool)
        .unwrap_or(true);
    (lines, include_forwarding)
}

fn runtime_status_to_value(
    running: bool,
    entry: crate::gateway_runtime::GatewayRuntimeEntry,
) -> RpcResult {
    serde_json::to_value(entry)
        .map(|entry| json!({ "running": running, "entry": entry }))
        .map_err(rpc_serde_error)
}

pub(crate) fn gateway_cwd(params: &Value) -> Result<Option<PathBuf>, JsonRpcError> {
    Ok(params
        .get("cwd")
        .or_else(|| params.get("dir"))
        .and_then(Value::as_str)
        .filter(|value| !value.trim().is_empty())
        .map(PathBuf::from))
}

pub(crate) fn gateway_port_and_cwd(
    params: &Value,
) -> Result<(Option<u16>, Option<PathBuf>), JsonRpcError> {
    let cwd = gateway_cwd(params)?;
    let port = match params.get("port").and_then(Value::as_u64) {
        Some(value) => Some(u16::try_from(value).map_err(|_| {
            JsonRpcError::invalid_params(Some(json!({
                "invalid": "port"
            })))
        })?),
        None => None,
    };
    Ok((port, cwd))
}

pub(crate) fn gateway_forward_provider(
    params: &Value,
) -> Result<Option<crate::gateway_runtime::ForwardProviderKind>, JsonRpcError> {
    params
        .get("provider")
        .and_then(Value::as_str)
        .filter(|value| !value.trim().is_empty())
        .map(|value| {
            crate::gateway_runtime::ForwardProviderKind::parse(value).ok_or_else(|| {
                JsonRpcError::invalid_params(Some(json!({
                    "invalid": "provider",
                    "supported": ["cloudflare", "cloudflare-quick", "quick"]
                })))
            })
        })
        .transpose()
}

pub(crate) fn forward_status_to_value(status: crate::gateway_runtime::ForwardStatus) -> RpcResult {
    Ok(json!({
        "cwd": status.cwd,
        "provider": status.provider.as_str(),
        "tunnel_pid": status.tunnel_pid,
        "local_port": status.local_port,
        "public_url": status.public_url,
        "log_path": status.log_path,
        "running": status.running
    }))
}
