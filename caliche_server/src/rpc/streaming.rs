use crate::rpc::connection::ResponseSink;
use crate::rpc::types::{JsonRpcError, JsonRpcResponse};
use anyhow::Result;
use serde_json::{Value, json};
use std::borrow::Cow;

#[derive(Clone)]
pub struct StreamNotifier {
    sink: ResponseSink,
    request_id: Value,
    notification_method: Cow<'static, str>,
}

impl StreamNotifier {
    pub fn new(
        sink: ResponseSink,
        request_id: Value,
        notification_method: impl Into<Cow<'static, str>>,
    ) -> Self {
        Self {
            sink,
            request_id,
            notification_method: notification_method.into(),
        }
    }

    pub async fn notify(&self, data: Value) -> Result<()> {
        let notification = json!({
            "jsonrpc": caliche_contracts::rpc::JSONRPC_VERSION,
            "method": self.notification_method,
            "params": {
                "request_id": self.request_id,
                "event": data
            }
        });
        self.sink.send_value(&notification).await
    }

    pub async fn complete(&self, result: Value) -> Result<()> {
        let response = JsonRpcResponse::success(Some(self.request_id.clone()), result);
        self.sink.send_response(&response).await
    }

    pub async fn error(&self, err: JsonRpcError) -> Result<()> {
        let response = JsonRpcResponse::error(Some(self.request_id.clone()), err);
        self.sink.send_response(&response).await
    }
}
