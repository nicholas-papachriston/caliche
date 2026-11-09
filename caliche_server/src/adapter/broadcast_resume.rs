//! Live tail of a session broadcast, including a resync when this receiver lags.
//!
//! The client contract for the resync value is [`BroadcastResync`].

use caliche::harness::{AgentEvent, BroadcastResync, broadcast_tail};
use futures_util::StreamExt;
use futures_util::stream::Stream;
use tokio::sync::broadcast;

/// One item from [`super::ApiAdapter::subscribe_events_with_resume`].
#[derive(Debug)]
pub(crate) enum ResumedSessionItem {
    /// An agent event. `id` is empty when the tail is the live broadcast.
    Event { id: String, event: AgentEvent },
    /// This subscriber missed `resync.skipped` broadcasts.
    Resync(BroadcastResync),
}

/// Map a live broadcast receiver to resume items.
///
/// A closed sender ends the stream. A lag is an item, so the caller can show
/// it before the next retained event.
pub(crate) fn live_broadcast_resume(
    rx: broadcast::Receiver<AgentEvent>,
) -> impl Stream<Item = anyhow::Result<ResumedSessionItem>> + Send {
    broadcast_tail(rx).map(|item| {
        Ok(match item {
            Ok(event) => ResumedSessionItem::Event {
                id: String::new(),
                event,
            },
            Err(resync) => ResumedSessionItem::Resync(resync),
        })
    })
}

#[cfg(test)]
mod tests {
    use super::{ResumedSessionItem, live_broadcast_resume};
    use caliche::harness::AgentEvent;
    use futures_util::StreamExt;
    use tokio::sync::broadcast;

    fn info(name: &str) -> AgentEvent {
        AgentEvent::Info {
            message: name.to_owned(),
        }
    }

    #[tokio::test]
    async fn live_resume_emits_resync_with_the_skipped_count() {
        let (tx, rx) = broadcast::channel(1);
        tx.send(info("m0")).expect("send");
        tx.send(info("m1")).expect("send");
        tx.send(info("m2")).expect("send");

        let mut stream = live_broadcast_resume(rx);
        match stream.next().await.expect("item").expect("ok") {
            ResumedSessionItem::Resync(resync) => assert_eq!(resync.skipped, 2),
            other => panic!("expected resync, got {other:?}"),
        }
        match stream.next().await.expect("item").expect("ok") {
            ResumedSessionItem::Event {
                event: AgentEvent::Info { message },
                ..
            } => assert_eq!(message, "m2"),
            other => panic!("expected retained event, got {other:?}"),
        }
    }
}
