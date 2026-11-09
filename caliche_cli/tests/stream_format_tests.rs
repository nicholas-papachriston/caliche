//! One-shot stream formatting integration tests.

use caliche::messages::StreamEvent;
use caliche_cli::commands::OutputFormat;
use caliche_cli::output::process_oneshot_stream;
use futures_util::stream;

#[tokio::test]
async fn json_format_accumulates_via_processor() {
    let events = stream::iter(vec![
        Ok(StreamEvent::TextChunk {
            content: "hi".to_string(),
        }),
        Ok(StreamEvent::Done),
    ]);

    process_oneshot_stream(events, &OutputFormat::Json, None)
        .await
        .expect("json one-shot stream should complete");
}
