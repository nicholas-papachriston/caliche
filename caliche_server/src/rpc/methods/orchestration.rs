use crate::handlers;
use crate::models::{
    CoordinatorRequest, MultiAgentRequest, ResumeWorkflowRequest, WorkflowRequest,
};
use crate::rpc::connection::ConnectionCtx;
use crate::rpc::dispatch::{MethodKind, MethodRouter};
use crate::rpc::helpers::{map_handler_err, parse_params, require_param_str, to_rpc_value};
use crate::rpc::types::JsonRpcError;
use axum::Json;
use axum::extract::{Path, State};
use futures_util::FutureExt;
use serde_json::{Value, json};
use std::collections::HashMap;

pub fn register(router: &mut MethodRouter) {
    register_workflows(router);
    register_harnesses(router);
    register_workflow_builder(router);
    register_gateway(router);
    register_cluster(router);
}

fn register_workflows(router: &mut MethodRouter) {
    router.register(
        "orchestration.executeWorkflow",
        MethodKind::RequestResponse,
        |params, _id, ctx: ConnectionCtx| async move { execute_workflow(params, ctx).await },
    );
    router.register(
        "orchestration.getWorkflowStatus",
        MethodKind::RequestResponse,
        |params, _id, ctx: ConnectionCtx| async move { get_workflow_status(params, ctx).await },
    );
    router.register(
        "orchestration.pauseWorkflow",
        MethodKind::RequestResponse,
        |params, _id, ctx: ConnectionCtx| async move { pause_workflow(params, ctx).await },
    );
    router.register(
        "orchestration.resumeWorkflow",
        MethodKind::RequestResponse,
        |params, _id, ctx: ConnectionCtx| async move { resume_workflow(params, ctx).await },
    );
    router.register(
        "orchestration.listInterrupts",
        MethodKind::RequestResponse,
        |_params, _id, ctx: ConnectionCtx| async move { list_interrupts(ctx).await },
    );
    router.register(
        "orchestration.getInterrupt",
        MethodKind::RequestResponse,
        |params, _id, ctx: ConnectionCtx| async move { get_interrupt(params, ctx).await },
    );
    router.register(
        "orchestration.resumeInterrupt",
        MethodKind::RequestResponse,
        |params, _id, ctx: ConnectionCtx| async move { resume_interrupt(params, ctx).await },
    );
    router.register(
        "orchestration.multiAgent",
        MethodKind::RequestResponse,
        |params, _id, ctx: ConnectionCtx| async move { multi_agent(params, ctx).await },
    );
    router.register(
        "orchestration.coordinator",
        MethodKind::RequestResponse,
        |params, _id, ctx: ConnectionCtx| coordinator(params, ctx).boxed(),
    );
}

fn register_harnesses(router: &mut MethodRouter) {
    router.register(
        "orchestration.listHarnesses",
        MethodKind::RequestResponse,
        |_params, _id, ctx: ConnectionCtx| async move { list_harnesses(ctx).await },
    );
    router.register(
        "orchestration.createHarness",
        MethodKind::RequestResponse,
        |params, _id, ctx: ConnectionCtx| async move { create_harness(params, ctx).await },
    );
    router.register(
        "orchestration.deleteHarness",
        MethodKind::RequestResponse,
        |params, _id, ctx: ConnectionCtx| async move { delete_harness(params, ctx).await },
    );
    router.register(
        "orchestration.dispatchHarness",
        MethodKind::RequestResponse,
        |params, _id, ctx: ConnectionCtx| async move { dispatch_harness(params, ctx).await },
    );
}

fn register_workflow_builder(router: &mut MethodRouter) {
    router.register(
        "workflows.save",
        MethodKind::RequestResponse,
        |params, _id, ctx: ConnectionCtx| async move { workflows_save(params, ctx).await },
    );
    router.register(
        "workflows.load",
        MethodKind::RequestResponse,
        |params, _id, ctx: ConnectionCtx| async move { workflows_load(params, ctx).await },
    );
    router.register(
        "workflows.validate",
        MethodKind::RequestResponse,
        |params, _id, ctx: ConnectionCtx| async move { workflows_validate(params, ctx).await },
    );
    router.register(
        "workflows.execute",
        MethodKind::RequestResponse,
        |params, _id, ctx: ConnectionCtx| async move { workflows_execute(params, ctx).await },
    );
}

fn register_gateway(router: &mut MethodRouter) {
    router.register(
        "gateway.listAdapters",
        MethodKind::RequestResponse,
        |_params, _id, ctx: ConnectionCtx| async move { list_adapters(ctx).await },
    );
    router.register(
        "gateway.enableAdapter",
        MethodKind::RequestResponse,
        |params, _id, ctx: ConnectionCtx| async move { enable_adapter(params, ctx).await },
    );
    router.register(
        "gateway.disableAdapter",
        MethodKind::RequestResponse,
        |params, _id, ctx: ConnectionCtx| async move { disable_adapter(params, ctx).await },
    );
    router.register(
        "gateway.listBindings",
        MethodKind::RequestResponse,
        |params, _id, ctx: ConnectionCtx| async move { list_bindings(params, ctx).await },
    );
    router.register(
        "gateway.overrideBinding",
        MethodKind::RequestResponse,
        |params, _id, ctx: ConnectionCtx| async move { override_binding(params, ctx).await },
    );
    router.register(
        "gateway.simulateIngress",
        MethodKind::RequestResponse,
        |params, _id, ctx: ConnectionCtx| async move { simulate_ingress(params, ctx).await },
    );
    router.register(
        "gateway.simulateEnvelopeIngress",
        MethodKind::RequestResponse,
        |params, _id, ctx: ConnectionCtx| async move {
            simulate_envelope_ingress(params, ctx).await
        },
    );
    router.register(
        "gateway.deliveryDiagnostics",
        MethodKind::RequestResponse,
        |_params, _id, ctx: ConnectionCtx| async move { delivery_diagnostics(ctx).await },
    );
}

fn register_cluster(router: &mut MethodRouter) {
    router.register(
        "gateway.registerClusterNode",
        MethodKind::RequestResponse,
        |params, _id, ctx: ConnectionCtx| async move { register_cluster_node(params, ctx).await },
    );
    router.register(
        "gateway.listClusterNodes",
        MethodKind::RequestResponse,
        |_params, _id, _ctx: ConnectionCtx| async move { list_cluster_nodes().await },
    );
    router.register(
        "gateway.listClusterRoutes",
        MethodKind::RequestResponse,
        |_params, _id, _ctx: ConnectionCtx| async move { list_cluster_routes().await },
    );
    router.register(
        "gateway.pinClusterRoute",
        MethodKind::RequestResponse,
        |params, _id, _ctx: ConnectionCtx| async move { pin_cluster_route(params).await },
    );
    router.register(
        "gateway.unpinClusterRoute",
        MethodKind::RequestResponse,
        |params, _id, _ctx: ConnectionCtx| async move { unpin_cluster_route(params).await },
    );
}

async fn execute_workflow(params: Value, ctx: ConnectionCtx) -> Result<Value, JsonRpcError> {
    let request = parse_params::<WorkflowRequest>(params)?;
    let Json(response) =
        handlers::orchestration::execute_workflow_handler(State(ctx.app_state), Json(request))
            .await
            .map_err(|(status, message)| map_handler_err(status, message))?;
    to_rpc_value(response)
}

async fn get_workflow_status(params: Value, ctx: ConnectionCtx) -> Result<Value, JsonRpcError> {
    let id = require_param_str(&params, "id")?;
    let Json(response) =
        handlers::orchestration::get_workflow_status_handler(State(ctx.app_state), Path(id))
            .await
            .map_err(|(status, message)| map_handler_err(status, message))?;
    to_rpc_value(response)
}

async fn pause_workflow(params: Value, ctx: ConnectionCtx) -> Result<Value, JsonRpcError> {
    let id = require_param_str(&params, "id")?;
    let response = handlers::orchestration::pause_workflow_handler(State(ctx.app_state), Path(id))
        .await
        .map_err(|(status, message)| map_handler_err(status, message))?;
    Ok(json!({"status": response.as_u16()}))
}

async fn resume_workflow(params: Value, ctx: ConnectionCtx) -> Result<Value, JsonRpcError> {
    let id = require_param_str(&params, "id")?;
    let response = handlers::orchestration::resume_workflow_handler(State(ctx.app_state), Path(id))
        .await
        .map_err(|(status, message)| map_handler_err(status, message))?;
    Ok(json!({"status": response.as_u16()}))
}

async fn list_interrupts(ctx: ConnectionCtx) -> Result<Value, JsonRpcError> {
    let Json(response) = handlers::orchestration::list_interrupts_handler(State(ctx.app_state))
        .await
        .map_err(|(status, message)| map_handler_err(status, message))?;
    to_rpc_value(response)
}

async fn get_interrupt(params: Value, ctx: ConnectionCtx) -> Result<Value, JsonRpcError> {
    let id = require_param_str(&params, "id")?;
    let Json(response) =
        handlers::orchestration::get_interrupt_handler(State(ctx.app_state), Path(id))
            .await
            .map_err(|(status, message)| map_handler_err(status, message))?;
    to_rpc_value(response)
}

async fn resume_interrupt(params: Value, ctx: ConnectionCtx) -> Result<Value, JsonRpcError> {
    let id = require_param_str(&params, "id")?;
    let feedback: HashMap<String, Value> = params
        .get("feedback")
        .and_then(Value::as_object)
        .map(|m| m.clone().into_iter().collect())
        .unwrap_or_default();
    let Json(response) = handlers::orchestration::resume_interrupt_handler(
        State(ctx.app_state),
        Path(id.clone()),
        Json(ResumeWorkflowRequest {
            interrupt_id: id,
            feedback,
        }),
    )
    .await
    .map_err(|(status, message)| map_handler_err(status, message))?;
    to_rpc_value(response)
}

async fn multi_agent(params: Value, ctx: ConnectionCtx) -> Result<Value, JsonRpcError> {
    let request = parse_params::<MultiAgentRequest>(params)?;
    let Json(response) =
        handlers::orchestration::multi_agent_handler(State(ctx.app_state), Json(request))
            .await
            .map_err(|(status, message)| map_handler_err(status, message))?;
    to_rpc_value(response)
}

async fn coordinator(params: Value, ctx: ConnectionCtx) -> Result<Value, JsonRpcError> {
    let request = parse_params::<CoordinatorRequest>(params)?;
    let Json(response) = Box::pin(handlers::orchestration::coordinator_handler(
        State(ctx.app_state),
        Json(request),
    ))
    .await
    .map_err(|(status, message)| map_handler_err(status, message))?;
    to_rpc_value(response)
}

async fn list_harnesses(ctx: ConnectionCtx) -> Result<Value, JsonRpcError> {
    let Json(response) = handlers::harnesses::list_harnesses(State(ctx.app_state))
        .await
        .map_err(|(status, message)| map_handler_err(status, message))?;
    to_rpc_value(response)
}

async fn create_harness(params: Value, ctx: ConnectionCtx) -> Result<Value, JsonRpcError> {
    let request = parse_params::<handlers::harnesses::CreateHarnessRequest>(params)?;
    let Json(response) = handlers::harnesses::create_harness(State(ctx.app_state), Json(request))
        .await
        .map_err(|(status, message)| map_handler_err(status, message))?;
    to_rpc_value(response)
}

async fn delete_harness(params: Value, ctx: ConnectionCtx) -> Result<Value, JsonRpcError> {
    let id = require_param_str(&params, "id")?;
    let Json(response) = handlers::harnesses::delete_harness(State(ctx.app_state), Path(id))
        .await
        .map_err(|(status, message)| map_handler_err(status, message))?;
    Ok(response)
}

async fn dispatch_harness(params: Value, ctx: ConnectionCtx) -> Result<Value, JsonRpcError> {
    let id = require_param_str(&params, "id")?;
    let message = require_param_str(&params, "message")?;
    let Json(response) = handlers::harnesses::dispatch_harness(
        State(ctx.app_state),
        Path(id),
        Json(handlers::harnesses::DispatchHarnessRequest { message }),
    )
    .await
    .map_err(|(status, message)| map_handler_err(status, message))?;
    to_rpc_value(response)
}

async fn workflows_save(params: Value, ctx: ConnectionCtx) -> Result<Value, JsonRpcError> {
    let Json(response) =
        handlers::orchestration::save_workflow_handler(State(ctx.app_state), Json(params))
            .await
            .map_err(|(status, message)| map_handler_err(status, message))?;
    Ok(response)
}

async fn workflows_load(params: Value, ctx: ConnectionCtx) -> Result<Value, JsonRpcError> {
    let id = require_param_str(&params, "id")?;
    let Json(response) =
        handlers::orchestration::load_workflow_handler(State(ctx.app_state), Path(id))
            .await
            .map_err(|(status, message)| map_handler_err(status, message))?;
    Ok(response)
}

async fn workflows_validate(params: Value, ctx: ConnectionCtx) -> Result<Value, JsonRpcError> {
    let Json(response) =
        handlers::orchestration::validate_workflow_handler(State(ctx.app_state), Json(params))
            .await
            .map_err(|(status, message)| map_handler_err(status, message))?;
    Ok(response)
}

async fn workflows_execute(params: Value, ctx: ConnectionCtx) -> Result<Value, JsonRpcError> {
    let Json(response) = handlers::orchestration::execute_workflow_builder_handler(
        State(ctx.app_state),
        Json(params),
    )
    .await
    .map_err(|(status, message)| map_handler_err(status, message))?;
    Ok(response)
}

async fn list_adapters(ctx: ConnectionCtx) -> Result<Value, JsonRpcError> {
    let Json(response) = handlers::gateway::list_adapters_handler(State(ctx.sessions_state))
        .await
        .map_err(|(status, message)| map_handler_err(status, message))?;
    to_rpc_value(response)
}

async fn enable_adapter(params: Value, ctx: ConnectionCtx) -> Result<Value, JsonRpcError> {
    let id = require_param_str(&params, "id")?;
    let status = handlers::gateway::enable_adapter_handler(State(ctx.sessions_state), Path(id))
        .await
        .map_err(|(status, message)| map_handler_err(status, message))?;
    Ok(json!({"status": status.as_u16()}))
}

async fn disable_adapter(params: Value, ctx: ConnectionCtx) -> Result<Value, JsonRpcError> {
    let id = require_param_str(&params, "id")?;
    let status = handlers::gateway::disable_adapter_handler(State(ctx.sessions_state), Path(id))
        .await
        .map_err(|(status, message)| map_handler_err(status, message))?;
    Ok(json!({"status": status.as_u16()}))
}

async fn list_bindings(params: Value, ctx: ConnectionCtx) -> Result<Value, JsonRpcError> {
    let session_id = require_param_str(&params, "session_id")?;
    let Json(response) =
        handlers::gateway::list_bindings_handler(State(ctx.sessions_state), Path(session_id))
            .await
            .map_err(|(status, message)| map_handler_err(status, message))?;
    to_rpc_value(response)
}

async fn override_binding(params: Value, ctx: ConnectionCtx) -> Result<Value, JsonRpcError> {
    let req = parse_params::<handlers::gateway::OverrideBindingRequest>(params)?;
    let Json(response) =
        handlers::gateway::override_binding_handler(State(ctx.sessions_state), Json(req))
            .await
            .map_err(|(status, message)| map_handler_err(status, message))?;
    to_rpc_value(response)
}

async fn simulate_ingress(params: Value, ctx: ConnectionCtx) -> Result<Value, JsonRpcError> {
    let req = parse_params::<handlers::gateway::SimulateIngressRequest>(params)?;
    let Json(response) = Box::pin(handlers::gateway::simulate_ingress_handler(
        State(ctx.sessions_state),
        Json(req),
    ))
    .await
    .map_err(|(status, message)| map_handler_err(status, message))?;
    to_rpc_value(response)
}

async fn simulate_envelope_ingress(
    params: Value,
    ctx: ConnectionCtx,
) -> Result<Value, JsonRpcError> {
    let req = parse_params::<handlers::gateway::EnvelopeSimulateRequest>(params)?;
    let Json(response) = Box::pin(handlers::gateway::simulate_envelope_ingress_handler(
        State(ctx.sessions_state),
        Json(req),
    ))
    .await
    .map_err(|(status, message)| map_handler_err(status, message))?;
    to_rpc_value(response)
}

async fn delivery_diagnostics(ctx: ConnectionCtx) -> Result<Value, JsonRpcError> {
    let Json(response) = handlers::gateway::delivery_diagnostics_handler(State(ctx.sessions_state))
        .await
        .map_err(|(status, message)| map_handler_err(status, message))?;
    to_rpc_value(response)
}

async fn register_cluster_node(params: Value, ctx: ConnectionCtx) -> Result<Value, JsonRpcError> {
    let req = parse_params::<handlers::gateway::ClusterRegisterRequest>(params)?;
    // RPC clients cannot pass an HTTP header, but a `UserContext` with
    // the cluster-admin role is sufficient authentication.
    let headers = axum::http::HeaderMap::new();
    let user = crate::middleware::OptionalUserContext(ctx.user_context);
    let status = handlers::gateway::register_cluster_node_handler(headers, user, Json(req))
        .await
        .map_err(|(status, message)| map_handler_err(status, message))?;
    Ok(json!({"status": status.as_u16()}))
}

async fn list_cluster_nodes() -> Result<Value, JsonRpcError> {
    let Json(response) = handlers::gateway::list_cluster_nodes_handler()
        .await
        .map_err(|(status, message)| map_handler_err(status, message))?;
    to_rpc_value(response)
}

async fn list_cluster_routes() -> Result<Value, JsonRpcError> {
    let Json(response) = handlers::gateway::list_cluster_routes_handler()
        .await
        .map_err(|(status, message)| map_handler_err(status, message))?;
    to_rpc_value(response)
}

async fn pin_cluster_route(params: Value) -> Result<Value, JsonRpcError> {
    let req = parse_params::<handlers::gateway::PinClusterRouteRequest>(params)?;
    let status = handlers::gateway::pin_cluster_route_handler(Json(req))
        .await
        .map_err(|(status, message)| map_handler_err(status, message))?;
    Ok(json!({"status": status.as_u16()}))
}

async fn unpin_cluster_route(params: Value) -> Result<Value, JsonRpcError> {
    let req = parse_params::<handlers::gateway::UnpinClusterRouteRequest>(params)?;
    let status = handlers::gateway::unpin_cluster_route_handler(Json(req))
        .await
        .map_err(|(status, message)| map_handler_err(status, message))?;
    Ok(json!({"status": status.as_u16()}))
}
