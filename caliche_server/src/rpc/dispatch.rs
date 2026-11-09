use crate::rpc::connection::ConnectionCtx;
use crate::rpc::types::{
    ERR_WS_REQUIRED, JsonRpcError, JsonRpcRequest, JsonRpcResponse, app_error,
};
use futures_util::FutureExt;
use serde_json::{Value, json};
use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MethodKind {
    RequestResponse,
    Streaming,
    Subscription,
}

type BoxMethodFuture = Pin<Box<dyn Future<Output = Result<Value, JsonRpcError>> + Send>>;
type MethodHandler =
    Arc<dyn Fn(Value, Option<Value>, ConnectionCtx) -> BoxMethodFuture + Send + Sync>;

#[derive(Clone)]
pub struct MethodEntry {
    pub kind: MethodKind,
    pub handler: MethodHandler,
}

#[derive(Clone, Default)]
pub struct MethodRouter {
    methods: HashMap<String, MethodEntry>,
}

impl MethodRouter {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn register<F, Fut>(&mut self, name: &str, kind: MethodKind, f: F)
    where
        F: Fn(Value, Option<Value>, ConnectionCtx) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<Value, JsonRpcError>> + Send + 'static,
    {
        self.methods.insert(
            name.to_string(),
            MethodEntry {
                kind,
                handler: Arc::new(move |params, id, ctx| f(params, id, ctx).boxed()),
            },
        );
    }

    pub fn kind_of(&self, method: &str) -> Option<MethodKind> {
        self.methods.get(method).map(|entry| entry.kind)
    }

    pub async fn dispatch(
        &self,
        req: JsonRpcRequest,
        ctx: ConnectionCtx,
    ) -> Option<JsonRpcResponse> {
        if req.jsonrpc.as_ref() != caliche_contracts::rpc::JSONRPC_VERSION {
            return Some(JsonRpcResponse::error(
                req.id,
                JsonRpcError::invalid_request(Some(json!({"reason":"jsonrpc must be 2.0"}))),
            ));
        }

        let Some(entry) = self.methods.get(req.method.as_ref()) else {
            return Some(JsonRpcResponse::error(
                req.id,
                JsonRpcError::method_not_found(req.method.as_ref()),
            ));
        };

        if entry.kind == MethodKind::Subscription && !ctx.sink.supports_subscriptions() {
            return Some(JsonRpcResponse::error(
                req.id,
                app_error(
                    ERR_WS_REQUIRED,
                    "subscriptions require WebSocket transport",
                    Some(json!({"method": req.method})),
                ),
            ));
        }

        let id = req.id.clone();
        let params = req.params.unwrap_or_else(|| json!({}));
        let result = (entry.handler)(params, req.id.clone(), ctx).await;

        match result {
            Ok(value) => id.map(|id_val| JsonRpcResponse::success(Some(id_val), value)),
            Err(err) => id.map(|id_val| JsonRpcResponse::error(Some(id_val), err)),
        }
    }

    pub async fn dispatch_batch(
        &self,
        reqs: Vec<JsonRpcRequest>,
        ctx: ConnectionCtx,
    ) -> Vec<JsonRpcResponse> {
        let mut out = Vec::new();
        for req in reqs {
            if let Some(resp) = self.dispatch(req, ctx.clone()).await {
                out.push(resp);
            }
        }
        out
    }
}
