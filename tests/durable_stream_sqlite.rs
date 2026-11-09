//! The SQLite durable stream backend stores and replays events. `durable-streams-sqlite`
//! compiles only the SQLite driver of `sqlx`.
#![cfg(feature = "durable-streams-sqlite")]

use caliche::messages::StreamEvent;
use caliche::streams::{
    DurableEvent, DurableStreamBackend, STREAM_START, SqliteDurableStreamBackend,
};

fn text(event: &DurableEvent) -> Option<&str> {
    match event.as_stream_event() {
        Some(StreamEvent::TextChunk { content }) => Some(content),
        _ => None,
    }
}

#[tokio::test]
async fn sqlite_stream_replays_appended_events_in_order() {
    let dir = tempfile::tempdir().unwrap();
    let db = dir.path().join("streams.db");
    let backend = SqliteDurableStreamBackend::new(db.to_str().unwrap())
        .await
        .unwrap();

    for chunk in ["first", "second"] {
        let event = StreamEvent::TextChunk {
            content: chunk.to_string(),
        };
        backend
            .append("run-1", &DurableEvent::stream(event, None))
            .await
            .unwrap();
    }

    let events = backend.read_after("run-1", STREAM_START, 10).await.unwrap();
    let texts: Vec<Option<&str>> = events.iter().map(|(_, e)| text(e)).collect();
    assert_eq!(texts, vec![Some("first"), Some("second")]);
}
