//! Python bindings for streaming support

use crate::get_runtime;
use caliche::messages::{ACTION_ID_FIELD, PerformanceMetrics, StreamEvent, TokenUsage};
use caliche::streams::{DurableEventPayload, DurableStreamBackend, DurableStreamReader};
use futures::StreamExt;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyList};
use pythonize::pythonize;
use std::pin::Pin;
use std::sync::Arc;

/// Maximum number of consecutive text chunks to batch before returning to Python
const TEXT_CHUNK_BATCH_SIZE: usize = 8;

type BoxedStream = Pin<Box<dyn futures::Stream<Item = anyhow::Result<StreamEvent>> + Send>>;

/// Python iterator for streaming chat events.
#[pyclass(unsendable)]
pub struct StreamIterator {
    /// Wrapped in `Option` so `__next__` can temporarily take ownership and
    /// `py.detach` while awaiting the next event (Python tools need the GIL).
    stream: Option<BoxedStream>,
    pending_event: Option<StreamEvent>,
    resume_from: Option<String>,
    resume_emitted: bool,
    /// In-memory replay buffer populated when resume/replay ids are set.
    replay_buffer: Vec<(String, StreamEvent)>,
    next_event_seq: u64,
    /// Optional durable reader for true backend replay.
    durable_reader: Option<DurableStreamReader>,
}

fn is_stream_start_id(event_id: &str) -> bool {
    event_id.is_empty() || event_id == "0"
}

fn events_after<'a>(
    buffer: &'a [(String, StreamEvent)],
    after_id: &str,
) -> Vec<&'a (String, StreamEvent)> {
    let mut past = is_stream_start_id(after_id);
    let mut out = Vec::new();
    for entry in buffer {
        if !past {
            if entry.0 == after_id {
                past = true;
            }
            continue;
        }
        out.push(entry);
    }
    out
}

fn add_usage(dict: &Bound<'_, PyDict>, py: Python<'_>, usage: &TokenUsage) -> PyResult<()> {
    let usage_dict = PyDict::new(py);
    if let Some(value) = usage.prompt_tokens {
        usage_dict.set_item("prompt_tokens", value)?;
    }
    if let Some(value) = usage.completion_tokens {
        usage_dict.set_item("completion_tokens", value)?;
    }
    if let Some(value) = usage.total_tokens {
        usage_dict.set_item("total_tokens", value)?;
    }
    if let Some(value) = usage.cache_creation_tokens {
        usage_dict.set_item("cache_creation_tokens", value)?;
    }
    if let Some(value) = usage.cache_read_tokens {
        usage_dict.set_item("cache_read_tokens", value)?;
    }
    dict.set_item("usage", usage_dict)
}

fn add_metadata(
    dict: &Bound<'_, PyDict>,
    py: Python<'_>,
    usage: Option<TokenUsage>,
    finish_reason: Option<beef::lean::Cow<'static, str>>,
    metrics: Option<PerformanceMetrics>,
) -> PyResult<()> {
    dict.set_item("type", "metadata")?;
    if let Some(usage) = usage {
        add_usage(dict, py, &usage)?;
    }
    if let Some(reason) = finish_reason {
        dict.set_item("finish_reason", reason.as_ref())?;
    }
    if let Some(metrics) = metrics {
        dict.set_item("metrics", performance_metrics_to_pydict(py, &metrics)?)?;
    }
    Ok(())
}

fn add_image(
    dict: &Bound<'_, PyDict>,
    py: Python<'_>,
    data: &[u8],
    media_type: String,
    revised_prompt: Option<String>,
) -> PyResult<()> {
    dict.set_item("type", "image_generated")?;
    dict.set_item("media_type", media_type)?;
    dict.set_item("size_bytes", data.len())?;
    dict.set_item("data", PyBytes::new(py, data))?;
    if let Some(prompt) = revised_prompt {
        dict.set_item("revised_prompt", prompt)?;
    }
    Ok(())
}

fn add_tool_call(
    dict: &Bound<'_, PyDict>,
    py: Python<'_>,
    id: &str,
    name: &str,
    arguments: &serde_json::Value,
    thought_signature: Option<String>,
    action_id: Option<String>,
) -> PyResult<()> {
    dict.set_item("type", "tool_call_start")?;
    dict.set_item("id", id)?;
    dict.set_item("name", name)?;
    let py_args = pythonize(py, arguments)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to convert tool arguments: {e}")))?;
    dict.set_item("arguments", py_args)?;
    if let Some(signature) = thought_signature {
        dict.set_item("thought_signature", signature)?;
    }
    if let Some(action) = action_id {
        dict.set_item(ACTION_ID_FIELD, action)?;
    }
    Ok(())
}

fn add_compaction(
    dict: &Bound<'_, PyDict>,
    py: Python<'_>,
    tokens_before: usize,
    tokens_after: usize,
    messages_removed: usize,
    summary: Option<String>,
    report: Option<caliche::messages::CompactionReport>,
) -> PyResult<()> {
    dict.set_item("type", "compaction")?;
    dict.set_item("tokens_before", tokens_before)?;
    dict.set_item("tokens_after", tokens_after)?;
    dict.set_item("messages_removed", messages_removed)?;
    if let Some(summary) = summary {
        dict.set_item("summary", summary)?;
    }
    if let Some(report) = report.filter(|r| !r.is_empty()) {
        let value = serde_json::to_value(&report)
            .map_err(|err| PyRuntimeError::new_err(err.to_string()))?;
        dict.set_item("report", pythonize(py, &value)?)?;
    }
    Ok(())
}

fn add_phase_transition(
    dict: &Bound<'_, PyDict>,
    py: Python<'_>,
    phase_index: usize,
    phase_total: usize,
    tool_names: &[String],
) -> PyResult<()> {
    dict.set_item("type", "phase_transition")?;
    dict.set_item("phase_index", phase_index)?;
    dict.set_item("phase_total", phase_total)?;
    dict.set_item("tool_names", PyList::new(py, tool_names)?)?;
    Ok(())
}

const fn lifecycle_event_name(event: &StreamEvent) -> &'static str {
    match event {
        StreamEvent::AgentStart => "agent_start",
        StreamEvent::AgentEnd => "agent_end",
        StreamEvent::TurnStart => "turn_start",
        StreamEvent::TurnEnd => "turn_end",
        _ => "unknown",
    }
}

fn event_to_pydict(
    py: Python<'_>,
    event: StreamEvent,
    event_id: Option<&str>,
) -> PyResult<Py<PyAny>> {
    let dict = PyDict::new(py);
    if let Some(id) = event_id {
        dict.set_item("event_id", id)?;
    } else {
        dict.set_item("event_id", py.None())?;
    }
    match event {
        StreamEvent::TextChunk { content } => {
            dict.set_item("type", "text_chunk")?;
            dict.set_item("content", content)?;
        }
        StreamEvent::ThinkingChunk { content, action_id } => {
            dict.set_item("type", "thinking_chunk")?;
            dict.set_item("content", content)?;
            if let Some(action) = action_id {
                dict.set_item(ACTION_ID_FIELD, action)?;
            }
        }
        StreamEvent::ToolCallStart {
            id,
            name,
            arguments,
            thought_signature,
            action_id,
        } => add_tool_call(
            &dict,
            py,
            id.as_ref(),
            name.as_ref(),
            &arguments,
            thought_signature,
            action_id,
        )?,
        StreamEvent::ToolCallResult {
            name,
            result,
            id,
            action_id,
            ..
        } => {
            dict.set_item("type", "tool_call_result")?;
            dict.set_item("name", name.as_ref())?;
            dict.set_item("result", result)?;
            if let Some(id) = id {
                dict.set_item("id", id)?;
            }
            if let Some(action) = action_id {
                dict.set_item(ACTION_ID_FIELD, action)?;
            }
        }
        StreamEvent::AudioChunk { data, format } => {
            dict.set_item("type", "audio_chunk")?;
            dict.set_item("format", format)?;
            dict.set_item("size_bytes", data.len())?;
            dict.set_item("data", PyBytes::new(py, &data))?;
        }
        StreamEvent::ImageGenerated {
            data,
            media_type,
            revised_prompt,
        } => add_image(&dict, py, &data, media_type, revised_prompt)?,
        StreamEvent::AttachmentProcessed {
            filename,
            attachment_type,
            summary,
        } => {
            dict.set_item("type", "attachment_processed")?;
            dict.set_item("filename", filename)?;
            dict.set_item("attachment_type", attachment_type)?;
            dict.set_item("summary", summary)?;
        }
        StreamEvent::Metadata {
            usage,
            finish_reason,
            metrics,
        } => add_metadata(&dict, py, usage, finish_reason, metrics)?,
        StreamEvent::Done => {
            dict.set_item("type", "done")?;
        }
        StreamEvent::LimitReached {
            max_iterations,
            usage,
        } => {
            dict.set_item("type", "limit_reached")?;
            dict.set_item("max_iterations", max_iterations)?;
            add_usage(&dict, py, &usage)?;
        }
        StreamEvent::Compaction {
            tokens_before,
            tokens_after,
            messages_removed,
            summary,
            report,
        } => add_compaction(
            &dict,
            py,
            tokens_before,
            tokens_after,
            messages_removed,
            summary,
            report,
        )?,
        StreamEvent::PhaseTransition {
            phase_index,
            phase_total,
            tool_names,
        } => add_phase_transition(&dict, py, phase_index, phase_total, &tool_names)?,
        StreamEvent::Status { message } => {
            dict.set_item("type", "status")?;
            dict.set_item("message", message)?;
        }
        event @ (StreamEvent::AgentStart
        | StreamEvent::AgentEnd
        | StreamEvent::TurnStart
        | StreamEvent::TurnEnd) => dict.set_item("type", lifecycle_event_name(&event))?,
    }
    Ok(dict.into())
}

fn performance_metrics_to_pydict<'py>(
    py: Python<'py>,
    metrics: &PerformanceMetrics,
) -> PyResult<Bound<'py, PyDict>> {
    let metrics_dict = PyDict::new(py);
    if let Some(v) = metrics.total_duration_ms {
        metrics_dict.set_item("total_duration_ms", v)?;
    }
    if let Some(v) = metrics.llm_duration_ms {
        metrics_dict.set_item("llm_duration_ms", v)?;
    }
    if let Some(v) = metrics.tool_duration_ms {
        metrics_dict.set_item("tool_duration_ms", v)?;
    }
    if let Some(v) = metrics.overhead_ms {
        metrics_dict.set_item("overhead_ms", v)?;
    }
    if let Some(v) = metrics.step_count {
        metrics_dict.set_item("step_count", v)?;
    }
    if let Some(v) = metrics.time_to_first_token_ms {
        metrics_dict.set_item("time_to_first_token_ms", v)?;
    }
    if let Some(steps) = &metrics.steps {
        let steps_list = PyList::empty(py);
        for step in steps {
            let step_dict = PyDict::new(py);
            step_dict.set_item("step", step.step)?;
            step_dict.set_item("llm_duration_ms", step.llm_duration_ms)?;
            step_dict.set_item("tool_duration_ms", step.tool_duration_ms)?;
            step_dict.set_item("tools_executed", step.tools_executed.clone())?;
            steps_list.append(step_dict)?;
        }
        metrics_dict.set_item("steps", steps_list)?;
    }
    Ok(metrics_dict)
}

impl StreamIterator {
    const fn should_buffer(&self) -> bool {
        self.resume_from.is_some() || self.durable_reader.is_some()
    }

    fn push_replay(&mut self, event: &StreamEvent) -> String {
        self.next_event_seq = self.next_event_seq.saturating_add(1);
        let id = format!("evt-{}", self.next_event_seq);
        self.replay_buffer.push((id.clone(), event.clone()));
        id
    }

    fn emit_event(&mut self, py: Python<'_>, event: StreamEvent) -> PyResult<Py<PyAny>> {
        if self.should_buffer() {
            let id = self.push_replay(&event);
            return event_to_pydict(py, event, Some(id.as_str()));
        }
        event_to_pydict(py, event, None)
    }
}

#[pymethods]
impl StreamIterator {
    const fn __iter__(slf: PyRef<Self>) -> PyRef<Self> {
        slf
    }

    fn __next__(mut slf: PyRefMut<Self>) -> PyResult<Option<Py<PyAny>>> {
        let py = slf.py();
        let rt = get_runtime()?;

        if !slf.resume_emitted {
            if let Some(ref event_id) = slf.resume_from.clone() {
                slf.resume_emitted = true;
                let dict = PyDict::new(py);
                dict.set_item("type", "resume")?;
                dict.set_item("event_id", event_id.as_str())?;
                return Ok(Some(dict.into()));
            }
            slf.resume_emitted = true;
        }

        if let Some(event) = slf.pending_event.take() {
            return Ok(Some(slf.emit_event(py, event)?));
        }

        let mut text_buf = String::new();
        let mut chunk_count = 0usize;

        loop {
            // Take the stream out so the await future is `Send`, then release the GIL
            // while blocking. Holding the GIL here deadlocks Python custom tools that
            // the background agent stream must execute on Tokio workers.
            let mut stream = slf.stream.take().ok_or_else(|| {
                PyRuntimeError::new_err("Stream iterator is already exhausted or in use")
            })?;
            let next_item = py.detach(|| rt.block_on(stream.next()));
            slf.stream = Some(stream);
            match next_item {
                Some(Ok(StreamEvent::TextChunk { content })) => {
                    text_buf.push_str(&content);
                    chunk_count += 1;
                    if chunk_count >= TEXT_CHUNK_BATCH_SIZE {
                        return Ok(Some(slf.emit_event(
                            py,
                            StreamEvent::TextChunk {
                                content: std::mem::take(&mut text_buf),
                            },
                        )?));
                    }
                }
                Some(Ok(other_event)) => {
                    if !text_buf.is_empty() {
                        slf.pending_event = Some(other_event);
                        return Ok(Some(slf.emit_event(
                            py,
                            StreamEvent::TextChunk {
                                content: std::mem::take(&mut text_buf),
                            },
                        )?));
                    }
                    return Ok(Some(slf.emit_event(py, other_event)?));
                }
                Some(Err(e)) => {
                    return Err(PyRuntimeError::new_err(format!("Stream error: {e}")));
                }
                None => {
                    if !text_buf.is_empty() {
                        return Ok(Some(slf.emit_event(
                            py,
                            StreamEvent::TextChunk {
                                content: std::mem::take(&mut text_buf),
                            },
                        )?));
                    }
                    return Ok(None);
                }
            }
        }
    }

    /// Event ID this stream was asked to resume from, if any.
    #[getter]
    fn resume_from(&self) -> Option<String> {
        self.resume_from.clone()
    }

    /// Replay stream events after `event_id`.
    ///
    /// Prefer a attached [`DurableStreamReader`] when present; otherwise return
    /// events from the in-memory buffer populated while consuming this iterator
    /// (when resume/replay ids are set).
    #[pyo3(name = "replay_from")]
    fn replay_from_py(&self, py: Python<'_>, event_id: &str) -> PyResult<Vec<Py<PyAny>>> {
        if let Some(ref reader) = self.durable_reader {
            let rt = get_runtime()?;
            let batch = py
                .detach(|| rt.block_on(reader.read_after(event_id, 1_000)))
                .map_err(|e| PyRuntimeError::new_err(format!("Durable replay failed: {e}")))?;
            let mut out = Vec::with_capacity(batch.len());
            for (id, durable) in batch {
                if let DurableEventPayload::Stream(event) = durable.payload {
                    out.push(event_to_pydict(py, event, Some(id.as_str()))?);
                }
            }
            return Ok(out);
        }

        let mut out = Vec::new();
        for (id, event) in events_after(&self.replay_buffer, event_id) {
            out.push(event_to_pydict(py, event.clone(), Some(id.as_str()))?);
        }
        Ok(out)
    }
}

impl StreamIterator {
    pub fn new(stream: BoxedStream, resume_from: Option<String>) -> Self {
        Self {
            stream: Some(stream),
            pending_event: None,
            resume_from,
            resume_emitted: false,
            replay_buffer: Vec::new(),
            next_event_seq: 0,
            durable_reader: None,
        }
    }

    /// Attach a durable stream reader for backend-backed `replay_from`.
    pub fn with_durable_reader(mut self, reader: DurableStreamReader) -> Self {
        self.durable_reader = Some(reader);
        self
    }

    /// Convenience: attach a reader built from a backend + stream key.
    pub fn with_durable_backend(
        self,
        backend: Arc<dyn DurableStreamBackend>,
        stream_key: impl Into<String>,
    ) -> Self {
        self.with_durable_reader(DurableStreamReader::new(backend, stream_key))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn events_after_skips_through_cursor() {
        let buffer = vec![
            (
                "evt-1".to_string(),
                StreamEvent::TextChunk {
                    content: "a".to_string(),
                },
            ),
            (
                "evt-2".to_string(),
                StreamEvent::TextChunk {
                    content: "b".to_string(),
                },
            ),
            ("evt-3".to_string(), StreamEvent::Done),
        ];
        let after = events_after(&buffer, "evt-1");
        assert_eq!(after.len(), 2);
        assert_eq!(after[0].0, "evt-2");
        assert_eq!(after[1].0, "evt-3");
    }

    #[test]
    fn events_after_stream_start_returns_all() {
        let buffer = vec![
            ("evt-1".to_string(), StreamEvent::AgentStart),
            (
                "evt-2".to_string(),
                StreamEvent::TextChunk {
                    content: "hi".to_string(),
                },
            ),
        ];
        assert_eq!(events_after(&buffer, "0").len(), 2);
        assert_eq!(events_after(&buffer, "").len(), 2);
    }
}
