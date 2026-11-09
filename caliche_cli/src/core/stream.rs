//! CLI adapters for core runtime stream processing.

use crate::core::state::InteractiveState;
use caliche::agent::runtime::stream::CancellationSignal;

pub use caliche::agent::runtime::{EventSink, StreamResult};

impl CancellationSignal for crate::cancellation::CancellationToken {
    fn is_cancelled(&self) -> bool {
        self.is_cancelled()
    }
}

pub struct StreamProcessor;

impl StreamProcessor {
    pub async fn process_stream<S>(
        stream: S,
        state: &mut InteractiveState,
        sink: &mut dyn EventSink,
        cancellation_token: Option<&crate::cancellation::CancellationToken>,
    ) -> anyhow::Result<StreamResult>
    where
        S: futures_util::Stream<Item = anyhow::Result<caliche::messages::StreamEvent>> + Unpin,
    {
        caliche::agent::runtime::StreamProcessor::process_stream(
            stream,
            sink,
            cancellation_token.map(|t| t as &dyn CancellationSignal),
            |usage| state.update_token_usage(usage),
        )
        .await
    }

    pub fn dispatch_event(sink: &mut dyn EventSink, event: caliche::messages::StreamEvent) {
        caliche::agent::runtime::StreamProcessor::dispatch_event(sink, event);
    }

    /// Process a stream without updating interactive session state (e.g. git handlers).
    pub async fn process_stream_sink<S>(
        stream: S,
        sink: &mut dyn EventSink,
        cancellation_token: Option<&crate::cancellation::CancellationToken>,
    ) -> anyhow::Result<StreamResult>
    where
        S: futures_util::Stream<Item = anyhow::Result<caliche::messages::StreamEvent>> + Unpin,
    {
        caliche::agent::runtime::StreamProcessor::process_stream(
            stream,
            sink,
            cancellation_token.map(|t| t as &dyn CancellationSignal),
            |_usage| {},
        )
        .await
    }

    /// Process a one-shot stream for non-interactive [`OutputFormat`](crate::commands::OutputFormat) output.
    pub async fn process_oneshot<S>(
        stream: S,
        output_format: &crate::commands::OutputFormat,
        cancellation_token: Option<&crate::cancellation::CancellationToken>,
    ) -> anyhow::Result<()>
    where
        S: futures_util::Stream<Item = anyhow::Result<caliche::messages::StreamEvent>> + Unpin,
    {
        crate::output::stream_format::process_oneshot_stream(
            stream,
            output_format,
            cancellation_token,
        )
        .await
    }
}
