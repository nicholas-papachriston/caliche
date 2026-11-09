//! SSE frames for a session's live agent-event broadcast.
//!
//! A lagged receiver becomes a [`BroadcastResync`] frame. The client contract
//! is on that type.

use axum::response::sse::Event;
use caliche::harness::{AgentEvent, BroadcastResync, broadcast_tail, public_agent_sse_event_name};
use futures_util::StreamExt;
use futures_util::stream::Stream;
use tokio::sync::broadcast;

use crate::transport::sse::envelope_from_serializable;

pub(super) struct SessionSseFrame {
    pub(super) name: &'static str,
    pub(super) data: String,
}

impl SessionSseFrame {
    pub(super) fn into_event(self) -> Event {
        Event::default().event(self.name).data(self.data)
    }
}

pub(super) fn resync_frame(resync: BroadcastResync) -> SessionSseFrame {
    SessionSseFrame {
        name: BroadcastResync::SSE_EVENT,
        data: resync.sse_data(),
    }
}

fn frame_from_agent_event(event: &AgentEvent) -> SessionSseFrame {
    let data = envelope_from_serializable(event)
        .and_then(|value| serde_json::to_string(&value))
        .unwrap_or_default();
    SessionSseFrame {
        name: public_agent_sse_event_name(event),
        data,
    }
}

/// Live session frames. The stream ends when the sender closes.
pub(super) fn session_event_frames(
    rx: broadcast::Receiver<AgentEvent>,
) -> impl Stream<Item = SessionSseFrame> + Send {
    broadcast_tail(rx).map(|item| match item {
        Ok(event) => frame_from_agent_event(&event),
        Err(resync) => resync_frame(resync),
    })
}

#[cfg(test)]
mod tests {
    use super::session_event_frames;
    use caliche::harness::{AgentEvent, BroadcastResync};
    use futures_util::StreamExt;
    use tokio::sync::broadcast;

    fn info(name: &str) -> AgentEvent {
        AgentEvent::Info {
            message: name.to_owned(),
        }
    }

    #[tokio::test]
    async fn session_sse_emits_resync_with_the_skipped_count() {
        let (tx, rx) = broadcast::channel(1);
        tx.send(info("m0")).expect("send");
        tx.send(info("m1")).expect("send");
        tx.send(info("m2")).expect("send");

        let mut frames = session_event_frames(rx);
        let resync = frames.next().await.expect("resync frame");
        assert_eq!(resync.name, BroadcastResync::SSE_EVENT);
        let body: serde_json::Value = serde_json::from_str(&resync.data).expect("json");
        assert_eq!(body["type"], BroadcastResync::SSE_EVENT);
        assert_eq!(body["payload"], serde_json::json!({ "skipped": 2 }));

        let kept = frames.next().await.expect("retained event");
        assert_eq!(kept.name, "info");
        let body: serde_json::Value = serde_json::from_str(&kept.data).expect("json");
        assert_eq!(
            body["payload"],
            serde_json::json!({ "type": "info", "message": "m2" })
        );
    }
}
