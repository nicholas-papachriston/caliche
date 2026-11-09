use crate::handlers;
#[cfg(any(
    feature = "chains",
    feature = "document-loaders",
    feature = "reasoning-patterns",
    feature = "tools-rag"
))]
use crate::models::*;
use crate::rpc::connection::ConnectionCtx;
use crate::rpc::dispatch::{MethodKind, MethodRouter};
use crate::rpc::helpers::{map_handler_err, parse_params, require_param_str, to_rpc_value};
use crate::rpc::types::JsonRpcError;
use axum::Json;
use axum::extract::{Path, Query, State};
use serde_json::Value;

pub fn register(router: &mut MethodRouter) {
    #[cfg(feature = "reasoning-patterns")]
    register_reasoning(router);
    #[cfg(feature = "chains")]
    register_chains(router);
    #[cfg(any(feature = "document-loaders", feature = "tools-rag"))]
    register_documents(router);
    #[cfg(feature = "templates")]
    register_templates(router);
}

#[cfg(feature = "reasoning-patterns")]
fn register_reasoning(router: &mut MethodRouter) {
    router.register(
        "reasoning.react",
        MethodKind::RequestResponse,
        |params, _id, ctx| async move { reasoning_react(params, ctx).await },
    );
    router.register(
        "reasoning.planExecute",
        MethodKind::RequestResponse,
        |params, _id, ctx| async move { reasoning_plan_execute(params, ctx).await },
    );
    router.register(
        "reasoning.selfAsk",
        MethodKind::RequestResponse,
        |params, _id, ctx| async move { reasoning_self_ask(params, ctx).await },
    );
    router.register(
        "reasoning.reflexion",
        MethodKind::RequestResponse,
        |params, _id, ctx| async move { reasoning_reflexion(params, ctx).await },
    );
    router.register(
        "reasoning.treeOfThoughts",
        MethodKind::RequestResponse,
        |params, _id, ctx| async move { reasoning_tree_of_thoughts(params, ctx).await },
    );
    router.register(
        "reasoning.listPatterns",
        MethodKind::RequestResponse,
        |_params, _id, _ctx| async move { reasoning_list_patterns().await },
    );
}

#[cfg(feature = "chains")]
fn register_chains(router: &mut MethodRouter) {
    router.register(
        "chains.llm",
        MethodKind::RequestResponse,
        |params, _id, ctx| async move { chains_llm(params, ctx).await },
    );
    router.register(
        "chains.sequential",
        MethodKind::RequestResponse,
        |params, _id, ctx| async move { chains_sequential(params, ctx).await },
    );
    router.register(
        "chains.mapReduce",
        MethodKind::RequestResponse,
        |params, _id, ctx| async move { chains_map_reduce(params, ctx).await },
    );
    router.register(
        "chains.retrieval",
        MethodKind::RequestResponse,
        |params, _id, ctx| async move { chains_retrieval(params, ctx).await },
    );
    router.register(
        "chains.hybridSearch",
        MethodKind::RequestResponse,
        |params, _id, ctx| async move { chains_hybrid_search(params, ctx).await },
    );
    router.register(
        "chains.documentMapReduce",
        MethodKind::RequestResponse,
        |params, _id, ctx| async move { chains_document_map_reduce(params, ctx).await },
    );
}

#[cfg(any(feature = "document-loaders", feature = "tools-rag"))]
fn register_documents(router: &mut MethodRouter) {
    router.register(
        "documents.split",
        MethodKind::RequestResponse,
        |params, _id, _ctx| async move { documents_split(params).await },
    );
    #[cfg(feature = "document-loaders")]
    router.register(
        "documents.load",
        MethodKind::RequestResponse,
        |params, _id, _ctx| async move { documents_load(params).await },
    );
    #[cfg(feature = "tools-rag")]
    {
        router.register(
            "documents.index",
            MethodKind::RequestResponse,
            |params, _id, ctx| async move { documents_index(params, ctx).await },
        );
        router.register(
            "documents.batchIndex",
            MethodKind::RequestResponse,
            |params, _id, ctx| async move { documents_batch_index(params, ctx).await },
        );
        router.register(
            "documents.listCollections",
            MethodKind::RequestResponse,
            |_params, _id, ctx| async move { documents_list_collections(ctx).await },
        );
        router.register(
            "documents.deleteCollection",
            MethodKind::RequestResponse,
            |params, _id, ctx| async move { documents_delete_collection(params, ctx).await },
        );
        router.register(
            "documents.search",
            MethodKind::RequestResponse,
            |params, _id, ctx| async move { documents_search(params, ctx).await },
        );
    }
}

#[cfg(feature = "templates")]
fn register_templates(router: &mut MethodRouter) {
    router.register(
        "templates.list",
        MethodKind::RequestResponse,
        |_params, _id, ctx| async move { templates_list(ctx).await },
    );
    router.register(
        "templates.get",
        MethodKind::RequestResponse,
        |params, _id, ctx| async move { templates_get(params, ctx).await },
    );
    router.register(
        "templates.validate",
        MethodKind::RequestResponse,
        |params, _id, ctx| async move { templates_validate(params, ctx).await },
    );
    router.register(
        "templates.instantiate",
        MethodKind::RequestResponse,
        |params, _id, ctx| async move { templates_instantiate(params, ctx).await },
    );
    router.register(
        "templates.upload",
        MethodKind::RequestResponse,
        |params, _id, ctx| async move { templates_upload(params, ctx).await },
    );
    router.register(
        "templates.delete",
        MethodKind::RequestResponse,
        |params, _id, ctx| async move { templates_delete(params, ctx).await },
    );
    router.register(
        "templates.getCategories",
        MethodKind::RequestResponse,
        |_params, _id, ctx| async move { templates_get_categories(ctx).await },
    );
    router.register(
        "templates.search",
        MethodKind::RequestResponse,
        |params, _id, ctx| async move { templates_search(params, ctx).await },
    );
    router.register(
        "templates.fork",
        MethodKind::RequestResponse,
        |params, _id, ctx| async move { templates_fork(params, ctx).await },
    );
    router.register(
        "templates.export",
        MethodKind::RequestResponse,
        |params, _id, ctx| async move { templates_export(params, ctx).await },
    );
    router.register(
        "templates.import",
        MethodKind::RequestResponse,
        |params, _id, ctx| async move { templates_import(params, ctx).await },
    );
    router.register(
        "templates.instantiateWithVars",
        MethodKind::RequestResponse,
        |params, _id, ctx| async move { templates_instantiate_with_vars(params, ctx).await },
    );
}

#[cfg(feature = "reasoning-patterns")]
async fn reasoning_react(params: Value, ctx: ConnectionCtx) -> Result<Value, JsonRpcError> {
    let request = parse_params::<ReasoningRequest>(params)?;
    let Json(response) = handlers::reasoning::react_handler(State(ctx.app_state), Json(request))
        .await
        .map_err(|(s, m)| map_handler_err(s, m))?;
    to_rpc_value(response)
}

#[cfg(feature = "reasoning-patterns")]
async fn reasoning_plan_execute(params: Value, ctx: ConnectionCtx) -> Result<Value, JsonRpcError> {
    let request = parse_params::<ReasoningRequest>(params)?;
    let Json(response) =
        handlers::reasoning::plan_execute_handler(State(ctx.app_state), Json(request))
            .await
            .map_err(|(s, m)| map_handler_err(s, m))?;
    to_rpc_value(response)
}

#[cfg(feature = "reasoning-patterns")]
async fn reasoning_self_ask(params: Value, ctx: ConnectionCtx) -> Result<Value, JsonRpcError> {
    let request = parse_params::<ReasoningRequest>(params)?;
    let Json(response) = handlers::reasoning::self_ask_handler(State(ctx.app_state), Json(request))
        .await
        .map_err(|(s, m)| map_handler_err(s, m))?;
    to_rpc_value(response)
}

#[cfg(feature = "reasoning-patterns")]
async fn reasoning_reflexion(params: Value, ctx: ConnectionCtx) -> Result<Value, JsonRpcError> {
    let request = parse_params::<ReasoningRequest>(params)?;
    let Json(response) =
        handlers::reasoning::reflexion_handler(State(ctx.app_state), Json(request))
            .await
            .map_err(|(s, m)| map_handler_err(s, m))?;
    to_rpc_value(response)
}

#[cfg(feature = "reasoning-patterns")]
async fn reasoning_tree_of_thoughts(
    params: Value,
    ctx: ConnectionCtx,
) -> Result<Value, JsonRpcError> {
    let request = parse_params::<TreeOfThoughtsRequest>(params)?;
    let Json(response) =
        handlers::reasoning::tree_of_thoughts_handler(State(ctx.app_state), Json(request))
            .await
            .map_err(|(s, m)| map_handler_err(s, m))?;
    to_rpc_value(response)
}

#[cfg(feature = "reasoning-patterns")]
async fn reasoning_list_patterns() -> Result<Value, JsonRpcError> {
    let Json(response) = handlers::reasoning::list_patterns_handler().await;
    to_rpc_value(response)
}

#[cfg(feature = "chains")]
async fn chains_llm(params: Value, ctx: ConnectionCtx) -> Result<Value, JsonRpcError> {
    let request = parse_params::<LLMChainRequest>(params)?;
    let Json(response) = handlers::chains::llm_chain_handler(State(ctx.app_state), Json(request))
        .await
        .map_err(|(s, m)| map_handler_err(s, m))?;
    to_rpc_value(response)
}

#[cfg(feature = "chains")]
async fn chains_sequential(params: Value, ctx: ConnectionCtx) -> Result<Value, JsonRpcError> {
    let request = parse_params::<SequentialChainRequest>(params)?;
    let Json(response) =
        handlers::chains::sequential_chain_handler(State(ctx.app_state), Json(request))
            .await
            .map_err(|(s, m)| map_handler_err(s, m))?;
    to_rpc_value(response)
}

#[cfg(feature = "chains")]
async fn chains_map_reduce(params: Value, ctx: ConnectionCtx) -> Result<Value, JsonRpcError> {
    let request = parse_params::<MapReduceRequest>(params)?;
    let Json(response) = handlers::chains::map_reduce_handler(State(ctx.app_state), Json(request))
        .await
        .map_err(|(s, m)| map_handler_err(s, m))?;
    to_rpc_value(response)
}

#[cfg(feature = "chains")]
async fn chains_retrieval(params: Value, ctx: ConnectionCtx) -> Result<Value, JsonRpcError> {
    let request = parse_params::<RetrievalChainRequest>(params)?;
    let Json(response) =
        handlers::chains::retrieval_chain_handler(State(ctx.app_state), Json(request))
            .await
            .map_err(|(s, m)| map_handler_err(s, m))?;
    to_rpc_value(response)
}

#[cfg(feature = "chains")]
async fn chains_hybrid_search(params: Value, ctx: ConnectionCtx) -> Result<Value, JsonRpcError> {
    let request = parse_params::<HybridSearchRequest>(params)?;
    let Json(response) =
        handlers::chains::hybrid_search_handler(State(ctx.app_state), Json(request))
            .await
            .map_err(|(s, m)| map_handler_err(s, m))?;
    to_rpc_value(response)
}

#[cfg(feature = "chains")]
async fn chains_document_map_reduce(
    params: Value,
    ctx: ConnectionCtx,
) -> Result<Value, JsonRpcError> {
    let request = parse_params::<DocumentMapReduceRequest>(params)?;
    let Json(response) =
        handlers::chains::document_map_reduce_handler(State(ctx.app_state), Json(request))
            .await
            .map_err(|(s, m)| map_handler_err(s, m))?;
    to_rpc_value(response)
}

#[cfg(any(feature = "document-loaders", feature = "tools-rag"))]
async fn documents_split(params: Value) -> Result<Value, JsonRpcError> {
    let request = parse_params::<DocumentSplitRequest>(params)?;
    let Json(response) = handlers::documents::split_documents_handler(Json(request))
        .await
        .map_err(|(s, m)| map_handler_err(s, m))?;
    to_rpc_value(response)
}

#[cfg(feature = "document-loaders")]
async fn documents_load(params: Value) -> Result<Value, JsonRpcError> {
    let request = parse_params::<DocumentLoadRequest>(params)?;
    let Json(response) = handlers::documents::load_documents_handler(Json(request))
        .await
        .map_err(|(s, m)| map_handler_err(s, m))?;
    to_rpc_value(response)
}

#[cfg(feature = "tools-rag")]
async fn documents_index(params: Value, ctx: ConnectionCtx) -> Result<Value, JsonRpcError> {
    let request = parse_params::<DocumentIndexRequest>(params)?;
    let Json(response) =
        handlers::documents::index_documents_handler(State(ctx.app_state), Json(request))
            .await
            .map_err(|(s, m)| map_handler_err(s, m))?;
    to_rpc_value(response)
}

#[cfg(feature = "tools-rag")]
async fn documents_batch_index(params: Value, ctx: ConnectionCtx) -> Result<Value, JsonRpcError> {
    let request = parse_params::<DocumentBatchIndexRequest>(params)?;
    let Json(response) =
        handlers::documents::batch_index_documents_handler(State(ctx.app_state), Json(request))
            .await
            .map_err(|(s, m)| map_handler_err(s, m))?;
    to_rpc_value(response)
}

#[cfg(feature = "tools-rag")]
async fn documents_list_collections(ctx: ConnectionCtx) -> Result<Value, JsonRpcError> {
    let Json(response) = handlers::documents::list_collections_handler(State(ctx.app_state))
        .await
        .map_err(|(s, m)| map_handler_err(s, m))?;
    to_rpc_value(response)
}

#[cfg(feature = "tools-rag")]
async fn documents_delete_collection(
    params: Value,
    ctx: ConnectionCtx,
) -> Result<Value, JsonRpcError> {
    let name = require_param_str(&params, "name")?;
    let status = handlers::documents::delete_collection_handler(State(ctx.app_state), Path(name))
        .await
        .map_err(|(s, m)| map_handler_err(s, m))?;
    Ok(serde_json::json!({"status": status.as_u16()}))
}

#[cfg(feature = "tools-rag")]
async fn documents_search(params: Value, ctx: ConnectionCtx) -> Result<Value, JsonRpcError> {
    let request = parse_params::<DocumentSearchRequest>(params)?;
    let Json(response) =
        handlers::documents::search_documents_handler(State(ctx.app_state), Json(request))
            .await
            .map_err(|(s, m)| map_handler_err(s, m))?;
    to_rpc_value(response)
}

#[cfg(feature = "templates")]
async fn templates_list(ctx: ConnectionCtx) -> Result<Value, JsonRpcError> {
    let Json(response) = handlers::templates::list_templates_handler(State(ctx.app_state))
        .await
        .map_err(|(s, m)| map_handler_err(s, m))?;
    to_rpc_value(response)
}

#[cfg(feature = "templates")]
async fn templates_get(params: Value, ctx: ConnectionCtx) -> Result<Value, JsonRpcError> {
    let id = require_param_str(&params, "id")?;
    let Json(response) = handlers::templates::get_template_handler(State(ctx.app_state), Path(id))
        .await
        .map_err(|(s, m)| map_handler_err(s, m))?;
    to_rpc_value(response)
}

#[cfg(feature = "templates")]
async fn templates_validate(params: Value, ctx: ConnectionCtx) -> Result<Value, JsonRpcError> {
    let request = parse_params::<handlers::templates::ValidateTemplateRequest>(params)?;
    let Json(response) =
        handlers::templates::validate_template_handler(State(ctx.app_state), Json(request))
            .await
            .map_err(|(s, m)| map_handler_err(s, m))?;
    to_rpc_value(response)
}

#[cfg(feature = "templates")]
async fn templates_instantiate(params: Value, ctx: ConnectionCtx) -> Result<Value, JsonRpcError> {
    let request = parse_params::<handlers::templates::InstantiateTemplateRequest>(params)?;
    let Json(response) = Box::pin(handlers::templates::instantiate_template_handler(
        State(ctx.app_state),
        Json(request),
    ))
    .await
    .map_err(|(s, m)| map_handler_err(s, m))?;
    to_rpc_value(response)
}

#[cfg(feature = "templates")]
async fn templates_upload(params: Value, ctx: ConnectionCtx) -> Result<Value, JsonRpcError> {
    let request = parse_params::<caliche::templates::AgentTemplate>(params)?;
    let Json(response) =
        handlers::templates::upload_template_handler(State(ctx.app_state), Json(request))
            .await
            .map_err(|(s, m)| map_handler_err(s, m))?;
    to_rpc_value(response)
}

#[cfg(feature = "templates")]
async fn templates_delete(params: Value, ctx: ConnectionCtx) -> Result<Value, JsonRpcError> {
    let id = require_param_str(&params, "id")?;
    let status = handlers::templates::delete_template_handler(State(ctx.app_state), Path(id))
        .await
        .map_err(|(s, m)| map_handler_err(s, m))?;
    Ok(serde_json::json!({"status": status.as_u16()}))
}

#[cfg(feature = "templates")]
async fn templates_get_categories(ctx: ConnectionCtx) -> Result<Value, JsonRpcError> {
    let Json(response) = handlers::templates::get_categories_handler(State(ctx.app_state))
        .await
        .map_err(|(s, m)| map_handler_err(s, m))?;
    to_rpc_value(response)
}

#[cfg(feature = "templates")]
async fn templates_search(params: Value, ctx: ConnectionCtx) -> Result<Value, JsonRpcError> {
    let query = parse_params::<handlers::templates::SearchQuery>(params)?;
    let Json(response) =
        handlers::templates::search_templates_handler(State(ctx.app_state), Query(query))
            .await
            .map_err(|(s, m)| map_handler_err(s, m))?;
    to_rpc_value(response)
}

#[cfg(feature = "templates")]
async fn templates_fork(params: Value, ctx: ConnectionCtx) -> Result<Value, JsonRpcError> {
    let request = parse_params::<handlers::templates::ForkTemplateRequest>(params)?;
    let Json(response) =
        handlers::templates::fork_template_handler(State(ctx.app_state), Json(request))
            .await
            .map_err(|(s, m)| map_handler_err(s, m))?;
    to_rpc_value(response)
}

#[cfg(feature = "templates")]
async fn templates_export(params: Value, ctx: ConnectionCtx) -> Result<Value, JsonRpcError> {
    let id = require_param_str(&params, "id")?;
    let response = handlers::templates::export_template_handler(State(ctx.app_state), Path(id))
        .await
        .map_err(|(s, m)| map_handler_err(s, m))?;
    Ok(serde_json::json!({"yaml": response}))
}

#[cfg(feature = "templates")]
async fn templates_import(params: Value, ctx: ConnectionCtx) -> Result<Value, JsonRpcError> {
    let request = parse_params::<handlers::templates::ImportTemplateRequest>(params)?;
    let Json(response) =
        handlers::templates::import_template_handler(State(ctx.app_state), Json(request))
            .await
            .map_err(|(s, m)| map_handler_err(s, m))?;
    to_rpc_value(response)
}

#[cfg(feature = "templates")]
async fn templates_instantiate_with_vars(
    params: Value,
    ctx: ConnectionCtx,
) -> Result<Value, JsonRpcError> {
    let request = parse_params::<handlers::templates::InstantiateWithVariablesRequest>(params)?;
    let Json(response) = Box::pin(handlers::templates::instantiate_with_variables_handler(
        State(ctx.app_state),
        Json(request),
    ))
    .await
    .map_err(|(s, m)| map_handler_err(s, m))?;
    to_rpc_value(response)
}
