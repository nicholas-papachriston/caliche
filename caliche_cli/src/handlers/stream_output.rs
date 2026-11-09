//! Shared [`EventSink`] adapters for non-REPL command handlers.

use crate::core::stream::{EventSink, StreamProcessor, StreamResult};
use caliche::messages::TokenUsage;
use std::io::Write;

/// Controls how auxiliary stream events are rendered for plain writer sinks.
#[derive(Debug, Clone, Copy)]
pub struct PlainStreamSinkOptions {
    /// When true, emit a line when a tool call starts.
    pub show_tool_starts: bool,
}

impl Default for PlainStreamSinkOptions {
    fn default() -> Self {
        Self {
            show_tool_starts: true,
        }
    }
}

/// Writes stream events to a single output writer with plain (non-ANSI) formatting.
pub struct PlainStreamSink<'a, W: Write + Send> {
    out: &'a mut W,
    options: PlainStreamSinkOptions,
}

impl<'a, W: Write + Send> PlainStreamSink<'a, W> {
    pub fn new(out: &'a mut W, options: PlainStreamSinkOptions) -> Self {
        Self { out, options }
    }
}

impl<W: Write + Send> EventSink for PlainStreamSink<'_, W> {
    fn on_text_chunk(&mut self, text: &str) {
        let _ = write!(self.out, "{text}");
        let _ = self.out.flush();
    }

    fn on_tool_start(&mut self, name: &str, _args: &serde_json::Value) {
        if self.options.show_tool_starts {
            let _ = writeln!(self.out, "\n[{name}]");
        }
    }

    fn on_tool_complete(&mut self, _name: &str, _result: &str) {}

    fn on_thinking(&mut self, _content: &str) {}

    fn on_complete(&mut self, _usage: &TokenUsage) {}

    fn on_audio_chunk(&mut self, data: &[u8], format: &str) {
        let _ = writeln!(
            self.out,
            "\n[audio chunk: {} bytes, format={format}]",
            data.len()
        );
    }

    fn on_image_generated(&mut self, data: &[u8], media_type: &str) {
        let _ = writeln!(
            self.out,
            "\n[image generated: mime={media_type}, bytes={}]",
            data.len()
        );
    }

    fn on_attachment_processed(&mut self, attachment_type: &str, filename: &str, summary: &str) {
        let _ = writeln!(
            self.out,
            "\n[attachment processed: kind={attachment_type}, name={filename}, summary={summary}]"
        );
    }
}

/// Colored stdout/stderr sink used by single-message streaming mode.
pub struct ColoredStdioSink {
    show_tool_results: bool,
    /// Whether any text chunks were written to stdout.
    pub had_text_output: bool,
}

impl ColoredStdioSink {
    pub fn interactive_stream() -> Self {
        Self {
            show_tool_results: true,
            had_text_output: false,
        }
    }
}

impl EventSink for ColoredStdioSink {
    fn on_text_chunk(&mut self, text: &str) {
        self.had_text_output = true;
        print!("{text}");
        let _ = std::io::stdout().flush();
    }

    fn on_tool_start(&mut self, name: &str, _args: &serde_json::Value) {
        eprintln!("\n\x1b[33m[{name}]\x1b[0m");
    }

    fn on_tool_complete(&mut self, _name: &str, result: &str) {
        if !self.show_tool_results {
            return;
        }
        let preview = result.chars().take(200).collect::<String>();
        eprintln!("\x1b[2m{preview}\x1b[0m");
    }

    fn on_thinking(&mut self, _content: &str) {}

    fn on_complete(&mut self, _usage: &TokenUsage) {}

    fn on_audio_chunk(&mut self, data: &[u8], format: &str) {
        eprintln!(
            "\x1b[36m[audio chunk: {} bytes, format={format}]\x1b[0m",
            data.len()
        );
    }

    fn on_image_generated(&mut self, data: &[u8], media_type: &str) {
        eprintln!(
            "\x1b[36m[image generated: mime={media_type}, bytes={}]\x1b[0m",
            data.len()
        );
    }

    fn on_attachment_processed(&mut self, attachment_type: &str, filename: &str, summary: &str) {
        eprintln!(
            "\x1b[36m[attachment processed: kind={attachment_type}, name={filename}, summary={summary}]\x1b[0m"
        );
    }
}

/// Process a chat stream through [`StreamProcessor`] into a plain writer sink.
pub async fn process_plain_stream<W, S>(
    stream: S,
    out: &mut W,
    options: PlainStreamSinkOptions,
) -> anyhow::Result<StreamResult>
where
    W: Write + Send,
    S: futures_util::Stream<Item = anyhow::Result<caliche::messages::StreamEvent>> + Unpin,
{
    let mut sink = PlainStreamSink::new(out, options);
    StreamProcessor::process_stream_sink(stream, &mut sink, None).await
}

/// Process a chat stream through [`StreamProcessor`] into the colored stdio sink.
pub async fn process_colored_stdio_stream<S>(
    stream: S,
    sink: &mut ColoredStdioSink,
) -> anyhow::Result<StreamResult>
where
    S: futures_util::Stream<Item = anyhow::Result<caliche::messages::StreamEvent>> + Unpin,
{
    StreamProcessor::process_stream_sink(stream, sink, None).await
}
