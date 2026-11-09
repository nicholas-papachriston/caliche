use crate::adapter::ApiAdapter;
use crate::middleware::UserContext;
use crate::rpc::types::{ERR_FEATURE_DISABLED, JsonRpcResponse, app_error};
use crate::{AgentRouterState, AppState};
use anyhow::Result;
use axum::extract::ws::{Message, WebSocket};
use futures_util::SinkExt;
use futures_util::stream::SplitSink;
use serde_json::Value;
use serde_json::json;
use std::collections::HashSet;
use std::sync::Arc;
use tokio::sync::{Mutex, RwLock, mpsc};
use tokio_util::sync::CancellationToken;

/// Depth of the bounded SSE channel used by the JSON-RPC HTTP transport.
/// Chosen large enough to absorb typical streaming bursts without blocking
/// the producer, but small enough that a stuck consumer does not cause
/// unbounded memory growth.
pub const SSE_CHANNEL_CAPACITY: usize = 256;

#[derive(Clone)]
pub enum ResponseSink {
    WebSocket(Arc<Mutex<SplitSink<WebSocket, Message>>>),
    Sse(mpsc::Sender<Value>),
}

impl ResponseSink {
    pub async fn send_response(&self, msg: &JsonRpcResponse) -> Result<()> {
        self.send_value(&serde_json::to_value(msg)?).await
    }

    pub async fn send_value(&self, msg: &Value) -> Result<()> {
        match self {
            Self::WebSocket(sender) => {
                let payload = serde_json::to_string(msg)?;
                let mut guard = sender.lock().await;
                guard.send(Message::Text(payload.into())).await?;
                Ok(())
            }
            Self::Sse(sender) => match sender.try_send(msg.clone()) {
                Ok(()) => Ok(()),
                Err(mpsc::error::TrySendError::Full(value)) => {
                    tracing::warn!(
                        target: "caliche_server::rpc::sse",
                        capacity = SSE_CHANNEL_CAPACITY,
                        "SSE channel full; waiting briefly for a free slot"
                    );
                    // Wait briefly so a slow consumer applies backpressure.
                    // A timeout drops this frame and returns an error; frames
                    // already queued are kept. Callers that discard the error
                    // lose the frame without a trace.
                    tokio::time::timeout(std::time::Duration::from_millis(200), sender.send(value))
                        .await
                        .map_err(|_| anyhow::anyhow!("SSE channel send timed out"))?
                        .map_err(|e| anyhow::anyhow!("SSE channel closed: {e}"))
                }
                Err(mpsc::error::TrySendError::Closed(_)) => {
                    Err(anyhow::anyhow!("SSE channel closed"))
                }
            },
        }
    }

    pub fn supports_subscriptions(&self) -> bool {
        matches!(self, Self::WebSocket(_))
    }
}

#[derive(Clone)]
pub struct ConnectionCtx {
    pub sink: ResponseSink,
    pub app_state: AppState,
    pub agent_state: AgentRouterState,
    pub sessions_state: Arc<ApiAdapter>,
    pub subscriptions: Arc<RwLock<HashSet<String>>>,
    pub user_context: Option<UserContext>,
    /// Fires when the connection should stop performing streaming work. Wired
    /// to WS close events and HTTP disconnects by the transport layer, and may
    /// be tripped explicitly via a `chat.cancel` RPC.
    pub cancel_token: CancellationToken,
}

impl ConnectionCtx {
    pub fn ensure_ws_for_subscriptions(&self) -> Result<()> {
        if self.sink.supports_subscriptions() {
            Ok(())
        } else {
            Err(anyhow::anyhow!(
                app_error(
                    ERR_FEATURE_DISABLED,
                    "subscriptions require WebSocket transport",
                    Some(json!({"reason":"ws_required"})),
                )
                .message
                .to_string()
            ))
        }
    }
}
