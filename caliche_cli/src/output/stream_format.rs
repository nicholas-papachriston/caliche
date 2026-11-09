//! One-shot stream formatting for `OutputFormat` (Json, Jsonl, Text, Stream).

use crate::cancellation::CancellationToken;
use crate::commands::OutputFormat;
use anyhow::Context;
use caliche::agent::runtime::stream::CancellationSignal;
use caliche::agent::runtime::{EventSink, StreamProcessor};
use caliche::messages::{Attachment, StreamEvent};
use futures_util::Stream;
use std::io::Write;

/// Print a stderr banner listing multimodal attachment names.
pub fn eprint_multimodal_banner(attachments: &[Attachment]) {
    let names: Vec<String> = attachments
        .iter()
        .filter_map(|a| a.filename.clone())
        .collect();
    eprintln!(
        "\x1b[36m[multimodal: {} attachment(s): {}]\x1b[0m",
        attachments.len(),
        if names.is_empty() {
            "unnamed".to_string()
        } else {
            names.join(", ")
        }
    );
}

/// Stream a single multimodal message through the runtime processor and format output.
pub async fn run_multimodal_oneshot(
    stream: impl Stream<Item = anyhow::Result<StreamEvent>> + Unpin,
    output_format: &OutputFormat,
    cancellation_token: Option<&CancellationToken>,
) -> anyhow::Result<()> {
    process_oneshot_stream(stream, output_format, cancellation_token).await
}

/// Process a one-shot agent stream with stdout/stderr behavior keyed by `OutputFormat`.
pub async fn process_oneshot_stream<S>(
    stream: S,
    output_format: &OutputFormat,
    cancellation_token: Option<&CancellationToken>,
) -> anyhow::Result<()>
where
    S: Stream<Item = anyhow::Result<StreamEvent>> + Unpin,
{
    let mut sink = OneShotStreamSink::new(output_format);
    let result = StreamProcessor::process_stream(
        stream,
        &mut sink,
        cancellation_token.map(|t| t as &dyn CancellationSignal),
        |_| {},
    )
    .await
    .context("one-shot stream processing failed")?;

    if matches!(output_format, OutputFormat::Json) {
        emit_json_response(&result.text)?;
    }

    sink.finish()?;
    Ok(())
}

fn emit_json_response(text: &str) -> anyhow::Result<()> {
    let json = serde_json::json!({ "response": text });
    println!("{}", serde_json::to_string_pretty(&json)?);
    Ok(())
}

fn emit_jsonl_line(value: &serde_json::Value) -> anyhow::Result<()> {
    println!("{}", serde_json::to_string(value)?);
    Ok(())
}

/// Formats stream events for non-interactive CLI output.
struct OneShotStreamSink<'a> {
    format: &'a OutputFormat,
    saw_text: bool,
    io_error: Option<anyhow::Error>,
}

impl<'a> OneShotStreamSink<'a> {
    fn new(format: &'a OutputFormat) -> Self {
        Self {
            format,
            saw_text: false,
            io_error: None,
        }
    }

    fn record_io_error(&mut self, err: anyhow::Error) {
        if self.io_error.is_none() {
            self.io_error = Some(err);
        }
    }

    fn emit_jsonl(&mut self, value: &serde_json::Value) {
        if let Err(e) = emit_jsonl_line(value) {
            self.record_io_error(e);
        }
    }

    fn finish(&mut self) -> anyhow::Result<()> {
        if matches!(self.format, OutputFormat::Text | OutputFormat::Stream) && self.saw_text {
            println!();
        }
        if let Some(err) = self.io_error.take() {
            return Err(err);
        }
        Ok(())
    }
}

impl EventSink for OneShotStreamSink<'_> {
    fn on_text_chunk(&mut self, text: &str) {
        match self.format {
            OutputFormat::Json => {}
            OutputFormat::Jsonl => {
                let line = serde_json::json!({
                    "type": "text_chunk",
                    "content": text,
                });
                self.emit_jsonl(&line);
            }
            OutputFormat::Text | OutputFormat::Stream => {
                print!("{text}");
                if let Err(e) = std::io::stdout().flush() {
                    self.record_io_error(e.into());
                }
                self.saw_text = true;
            }
        }
    }

    fn on_tool_start(&mut self, name: &str, _args: &serde_json::Value) {
        match self.format {
            OutputFormat::Jsonl => {
                let line = serde_json::json!({
                    "type": "tool_call_start",
                    "name": name,
                });
                self.emit_jsonl(&line);
            }
            OutputFormat::Text | OutputFormat::Stream => {
                eprintln!("\n\x1b[33m[{name}]\x1b[0m");
            }
            OutputFormat::Json => {}
        }
    }

    fn on_tool_complete(&mut self, _name: &str, result: &str) {
        match self.format {
            OutputFormat::Jsonl => {
                let line = serde_json::json!({
                    "type": "tool_call_result",
                    "result": result,
                });
                self.emit_jsonl(&line);
            }
            OutputFormat::Text | OutputFormat::Stream => {
                let preview = result.chars().take(200).collect::<String>();
                eprintln!("\x1b[2m{preview}\x1b[0m");
            }
            OutputFormat::Json => {}
        }
    }

    fn on_attachment_processed(&mut self, attachment_type: &str, filename: &str, summary: &str) {
        match self.format {
            OutputFormat::Jsonl => {
                let line = serde_json::json!({
                    "type": "attachment_processed",
                    "attachment_type": attachment_type,
                    "filename": filename,
                    "summary": summary,
                });
                self.emit_jsonl(&line);
            }
            OutputFormat::Text | OutputFormat::Stream => {
                eprintln!("\x1b[36m[processed: {attachment_type} {filename} — {summary}]\x1b[0m");
            }
            OutputFormat::Json => {}
        }
    }

    fn on_thinking(&mut self, _content: &str) {}

    fn on_complete(&mut self, _usage: &caliche::messages::TokenUsage) {}
}
