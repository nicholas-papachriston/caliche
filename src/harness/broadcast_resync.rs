//! Gap signal for one slow consumer of an [`AgentEvent`](crate::harness::AgentEvent) broadcast.
//!
//! Tokio's broadcast channel keeps a fixed ring. A receiver that falls behind
//! observes [`RecvError::Lagged`] and the missed messages are gone from that
//! receiver. Other receivers, and a durable stream when one is attached, still
//! have what they kept. The lag is per receiver, so the harness does not emit it.
//!
//! # Client contract
//!
//! A [`BroadcastResync`] means this receiver's live tail has a hole. The next
//! event after the resync is the oldest event still in the ring, which is later
//! than the last event the client saw. Before acting on later events the client
//! must:
//!
//! 1. Re-read durable session state: message history and the latest checkpoint.
//! 2. Resume a durable task or chat stream with `after` set to the last event
//!    id the client stored. `skipped` counts missed broadcasts. Resume from
//!    that stored id. The session event socket
//!    (`GET /api/agent/sessions/{id}/events`) is live only: it does not set
//!    SSE ids and it does not read `Last-Event-ID`. Reconnecting it starts a
//!    new tail and does not replay the hole.
//! 3. Treat any in-flight tool call or turn as unknown until history or a later
//!    event confirms it.
//!
//! Surfaces that carry this type:
//!
//! - Session SSE (`GET /api/agent/sessions/{id}/events`) emits `event: resync`
//!   with the envelope from [`BroadcastResync::sse_data`], like its other frames.
//! - The task and chat SSE streams emit `event: resync` with
//!   [`BroadcastResync::tagged_data`], in the same tagged shape as their raw
//!   [`AgentEvent`](crate::harness::AgentEvent) frames.
//! - JSON-RPC session subscriptions emit method [`BroadcastResync::RPC_METHOD`]
//!   with `session_id` and `skipped`.
//! - `caliche code` prints [`BroadcastResync::cli_line`].

use futures_util::stream::{Stream, unfold};
use serde::{Deserialize, Serialize};
use tokio::sync::broadcast;
use tokio::sync::broadcast::error::RecvError;

use crate::messages::STREAM_EVENT_SCHEMA_VERSION;

/// One broadcast receiver missed `skipped` events.
///
/// Synthesized at the receiver. The harness never puts this value on the
/// event bus, so it is not replayed from a durable stream.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct BroadcastResync {
    /// Broadcasts this receiver did not observe.
    pub skipped: u64,
}

impl BroadcastResync {
    /// SSE `event:` name. The JSON `type` in [`Self::sse_data`] is the same string.
    pub const SSE_EVENT: &'static str = "resync";

    /// JSON-RPC notification method for a session subscription gap.
    ///
    /// Same value as [`caliche_contracts::rpc::methods::SESSIONS_RESYNC`].
    /// Params are `{ "session_id": string, "skipped": number }`. `skipped` is
    /// this struct's field. See the module-level client contract.
    pub const RPC_METHOD: &'static str = caliche_contracts::rpc::methods::SESSIONS_RESYNC;

    /// Build a resync from the count Tokio reported.
    #[must_use]
    pub const fn from_skipped(skipped: u64) -> Self {
        Self { skipped }
    }

    /// Session-SSE data line. `payload.skipped` is the missed count.
    #[must_use]
    pub fn sse_data(&self) -> String {
        let envelope = serde_json::json!({
            "version": STREAM_EVENT_SCHEMA_VERSION,
            "type": Self::SSE_EVENT,
            "payload": self,
        });
        serde_json::to_string(&envelope).unwrap_or_default()
    }

    /// Task and chat SSE data line: `{"type":"resync","skipped":n}`.
    ///
    /// Those streams send each [`AgentEvent`](crate::harness::AgentEvent) as
    /// its internally tagged JSON, so the resync uses the same shape.
    #[must_use]
    pub fn tagged_data(&self) -> String {
        serde_json::json!({ "type": Self::SSE_EVENT, "skipped": self.skipped }).to_string()
    }

    /// One line for the `caliche code` event printer.
    ///
    /// The number is `skipped`. The rest tells the operator to re-read session
    /// state before trusting later lines.
    #[must_use]
    pub fn cli_line(&self) -> String {
        format!(
            "[resync] skipped {} events; re-read session state before trusting later events\n",
            self.skipped
        )
    }
}

/// One `broadcast::Receiver::recv` outcome.
#[derive(Debug, PartialEq, Eq)]
pub enum BroadcastRecv<T> {
    /// The next event still in the ring.
    Event(T),
    /// This receiver missed [`BroadcastResync::skipped`] events.
    Resync(BroadcastResync),
    /// The sender is gone. The live tail is finished.
    Closed,
}

/// Classify one broadcast receive.
///
/// [`RecvError::Lagged`] becomes [`BroadcastRecv::Resync`] with Tokio's count.
/// The following receive is a separate call and still returns the oldest
/// retained event.
#[must_use]
pub fn classify_recv<T>(result: Result<T, RecvError>) -> BroadcastRecv<T> {
    match result {
        Ok(event) => BroadcastRecv::Event(event),
        Err(RecvError::Lagged(skipped)) => {
            BroadcastRecv::Resync(BroadcastResync::from_skipped(skipped))
        }
        Err(RecvError::Closed) => BroadcastRecv::Closed,
    }
}

/// A receiver's live tail as a stream.
///
/// A lag is an `Err` item carrying the skipped count, and the next item is the
/// oldest retained event. The stream ends when the sender closes.
pub fn broadcast_tail<T>(
    rx: broadcast::Receiver<T>,
) -> impl Stream<Item = Result<T, BroadcastResync>> + Send + Unpin
where
    T: Clone + Send + 'static,
{
    Box::pin(unfold(rx, |mut rx| async move {
        let item = match classify_recv(rx.recv().await) {
            BroadcastRecv::Event(event) => Ok(event),
            BroadcastRecv::Resync(resync) => Err(resync),
            BroadcastRecv::Closed => return None,
        };
        Some((item, rx))
    }))
}

#[cfg(test)]
mod tests {
    use super::{BroadcastRecv, BroadcastResync, broadcast_tail, classify_recv};
    use crate::harness::AgentEvent;
    use futures_util::StreamExt;
    use tokio::sync::broadcast;

    fn info(name: &str) -> AgentEvent {
        AgentEvent::Info {
            message: name.to_owned(),
        }
    }

    #[tokio::test]
    async fn lagged_receiver_reports_the_skipped_count_and_keeps_the_retained_event() {
        let (tx, mut rx) = broadcast::channel(1);
        tx.send(info("m0")).expect("send");
        tx.send(info("m1")).expect("send");
        tx.send(info("m2")).expect("send");

        match classify_recv(rx.recv().await) {
            BroadcastRecv::Resync(resync) => assert_eq!(resync, BroadcastResync::from_skipped(2)),
            other => panic!("expected resync, got {other:?}"),
        }
        match classify_recv(rx.recv().await) {
            BroadcastRecv::Event(AgentEvent::Info { message }) => assert_eq!(message, "m2"),
            other => panic!("expected retained event, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn tail_yields_the_resync_then_the_retained_event_then_ends() {
        let (tx, rx) = broadcast::channel(1);
        tx.send(info("m0")).expect("send");
        tx.send(info("m1")).expect("send");
        tx.send(info("m2")).expect("send");
        drop(tx);

        let items: Vec<Result<String, BroadcastResync>> = broadcast_tail(rx)
            .map(|item| {
                item.map(|event| match event {
                    AgentEvent::Info { message } => message,
                    other => panic!("unexpected event {other:?}"),
                })
            })
            .collect()
            .await;
        assert_eq!(
            items,
            [Err(BroadcastResync::from_skipped(2)), Ok("m2".to_owned())]
        );
    }

    #[test]
    fn tagged_data_matches_the_agent_event_frame_shape() {
        let data: serde_json::Value =
            serde_json::from_str(&BroadcastResync::from_skipped(2).tagged_data()).expect("json");
        assert_eq!(data, serde_json::json!({ "type": "resync", "skipped": 2 }));
    }
}
