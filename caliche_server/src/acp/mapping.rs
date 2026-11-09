use agent_client_protocol_schema::v1::{
    BlobResourceContents, ContentBlock, ContentChunk, EmbeddedResourceResource, ResourceLink,
    SessionUpdate, TextContent, TextResourceContents, ToolCall, ToolCallContent, ToolCallStatus,
    ToolCallUpdate, ToolCallUpdateFields, ToolKind,
};
use caliche::messages::{ContentPart, Message, MessageContent, MessageRole, StreamEvent};
use serde_json::Value;
use std::collections::{HashMap, VecDeque};

/// Unmatched `ToolCallStart` ids, in order, keyed by tool name.
///
/// `StreamEvent::ToolCallResult` has no call id. ACP `tool_call_update.toolCallId`
/// must be the id from the matching start. Results that cannot be paired are
/// dropped so the wire id is never the tool name.
#[derive(Debug, Default)]
pub struct PendingToolCalls {
    by_name: HashMap<String, VecDeque<String>>,
}

impl PendingToolCalls {
    pub fn push(&mut self, name: &str, id: String) {
        self.by_name
            .entry(name.to_string())
            .or_default()
            .push_back(id);
    }

    pub fn pop(&mut self, name: &str) -> Option<String> {
        let queue = self.by_name.get_mut(name)?;
        let id = queue.pop_front()?;
        if queue.is_empty() {
            self.by_name.remove(name);
        }
        Some(id)
    }
}

pub fn prompt_to_plain_text(prompt: &[ContentBlock]) -> anyhow::Result<String> {
    let mut lines = Vec::new();

    for block in prompt {
        match block {
            ContentBlock::Text(text) => lines.push(text.text.clone()),
            ContentBlock::ResourceLink(link) => lines.push(resource_link_line(link)),
            ContentBlock::Resource(resource) => {
                lines.push(embedded_resource_line(&resource.resource)?)
            }
            ContentBlock::Image(_) | ContentBlock::Audio(_) => {
                anyhow::bail!("image and audio prompt blocks are not enabled");
            }
            _ => anyhow::bail!("unsupported prompt content block"),
        }
    }

    if lines.is_empty() {
        anyhow::bail!("prompt must include at least one content block");
    }

    Ok(lines.join("\n\n"))
}

fn resource_link_line(link: &ResourceLink) -> String {
    let mut line = String::from("Resource: ");
    line.push_str(&link.uri);
    if let Some(value) = &link.mime_type {
        line.push_str(" (");
        line.push_str(value);
        line.push(')');
    }
    if let Some(value) = &link.title {
        line.push_str(" - ");
        line.push_str(value);
    }
    line
}

fn embedded_resource_line(resource: &EmbeddedResourceResource) -> anyhow::Result<String> {
    match resource {
        EmbeddedResourceResource::TextResourceContents(text) => Ok(text_resource_line(text)),
        EmbeddedResourceResource::BlobResourceContents(blob) => Ok(blob_resource_line(blob)),
        _ => anyhow::bail!("unsupported embedded resource"),
    }
}

fn text_resource_line(text: &TextResourceContents) -> String {
    format!("Resource {}:\n{}", text.uri, text.text)
}

fn blob_resource_line(blob: &BlobResourceContents) -> String {
    format!("Resource: {}", blob.uri)
}

/// Replay harness history as ACP `session/update` bodies.
///
/// System rows are omitted. Tool results without a call id are omitted so the
/// wire id is never invented. Assistant tool calls are emitted as `tool_call`
/// and the matching tool row as `tool_call_update`.
pub fn updates_from_history(history: &[Message]) -> Vec<SessionUpdate> {
    let mut updates = Vec::new();
    for message in history {
        append_history_message(&mut updates, message);
    }
    updates
}

fn append_history_message(updates: &mut Vec<SessionUpdate>, message: &Message) {
    match message.role {
        MessageRole::System => {}
        MessageRole::User => append_text_content(updates, ReplayRole::User, &message.content),
        MessageRole::Assistant => append_assistant(updates, message),
        MessageRole::Tool => append_tool_result(updates, message),
    }
}

fn append_assistant(updates: &mut Vec<SessionUpdate>, message: &Message) {
    if let Some(thought) = stored_reasoning(message) {
        updates.push(thought_update(thought.to_string()));
    }
    append_text_content(updates, ReplayRole::Agent, &message.content);
    if let Some(calls) = message.content.as_tool_calls() {
        for call in calls {
            let name = call.name.as_ref();
            updates.push(tool_call_update_new(
                call.id.clone().into_owned(),
                format!("Running {name}"),
                tool_kind_for_name(name),
                Some(call.arguments.clone()),
            ));
        }
    }
}

fn append_tool_result(updates: &mut Vec<SessionUpdate>, message: &Message) {
    let Some(tool_call_id) = message.content.as_tool_call_id() else {
        return;
    };
    if tool_call_id.is_empty() {
        return;
    }
    let text = message.content.extract_text();
    let status = if tool_result_failed(&text) {
        ToolCallStatus::Failed
    } else {
        ToolCallStatus::Completed
    };
    updates.push(tool_status_update(
        tool_call_id,
        None,
        None,
        status,
        text,
        None,
    ));
}

fn tool_result_failed(text: &str) -> bool {
    let trimmed = text.trim_start();
    trimmed.starts_with("Error:") || trimmed.starts_with("Tool call validation failed")
}

fn stored_reasoning(message: &Message) -> Option<&str> {
    let metadata = message.metadata.as_ref()?;
    let value = metadata
        .get("reasoning_content")
        .or_else(|| metadata.get("reasoning"))?;
    let text = value.as_str()?.trim();
    if text.is_empty() { None } else { Some(text) }
}

#[derive(Clone, Copy)]
enum ReplayRole {
    User,
    Agent,
}

fn append_text_content(
    updates: &mut Vec<SessionUpdate>,
    role: ReplayRole,
    content: &MessageContent,
) {
    if let Some(parts) = content.as_multimodal() {
        for part in parts {
            if let ContentPart::Text { text } = part {
                push_text(updates, role, text.as_ref());
            }
        }
        return;
    }
    if let Some(text) = content.as_text() {
        push_text(updates, role, text);
    }
}

fn push_text(updates: &mut Vec<SessionUpdate>, role: ReplayRole, text: &str) {
    if text.is_empty() {
        return;
    }
    updates.push(message_update(role, text.to_string()));
}

fn text_block(text: impl Into<String>) -> ContentBlock {
    ContentBlock::Text(TextContent::new(text))
}

fn message_update(role: ReplayRole, text: String) -> SessionUpdate {
    let chunk = ContentChunk::new(text_block(text));
    match role {
        ReplayRole::User => SessionUpdate::UserMessageChunk(chunk),
        ReplayRole::Agent => SessionUpdate::AgentMessageChunk(chunk),
    }
}

fn thought_update(text: String) -> SessionUpdate {
    SessionUpdate::AgentThoughtChunk(ContentChunk::new(text_block(text)))
}

fn tool_call_update_new(
    tool_call_id: String,
    title: String,
    kind: ToolKind,
    raw_input: Option<Value>,
) -> SessionUpdate {
    let mut call = ToolCall::new(tool_call_id, title)
        .kind(kind)
        .status(ToolCallStatus::Pending);
    if let Some(raw_input) = raw_input {
        call = call.raw_input(raw_input);
    }
    SessionUpdate::ToolCall(call)
}

fn tool_status_update(
    tool_call_id: impl Into<String>,
    title: Option<String>,
    kind: Option<ToolKind>,
    status: ToolCallStatus,
    text: String,
    raw_output: Option<Value>,
) -> SessionUpdate {
    let mut fields = ToolCallUpdateFields::new()
        .status(status)
        .content(vec![ToolCallContent::from(text_block(text))]);
    if let Some(title) = title {
        fields = fields.title(title);
    }
    if let Some(kind) = kind {
        fields = fields.kind(kind);
    }
    if let Some(raw_output) = raw_output {
        fields = fields.raw_output(raw_output);
    }
    SessionUpdate::ToolCallUpdate(ToolCallUpdate::new(tool_call_id.into(), fields))
}

pub fn user_updates_from_prompt(prompt: &[ContentBlock]) -> Vec<SessionUpdate> {
    prompt
        .iter()
        .cloned()
        .map(|content| SessionUpdate::UserMessageChunk(ContentChunk::new(content)))
        .collect()
}

pub fn updates_from_stream_event(
    event: StreamEvent,
    pending: &mut PendingToolCalls,
) -> Vec<SessionUpdate> {
    match event {
        StreamEvent::TextChunk { content } => vec![message_update(ReplayRole::Agent, content)],
        StreamEvent::ThinkingChunk {
            content,
            action_id: _,
        } => vec![thought_update(content)],
        StreamEvent::ToolCallStart {
            id,
            name,
            arguments,
            ..
        } => {
            let tool_call_id = id.into_owned();
            pending.push(name.as_ref(), tool_call_id.clone());
            vec![tool_call_update_new(
                tool_call_id,
                format!("Running {name}"),
                tool_kind_for_name(name.as_ref()),
                Some(arguments),
            )]
        }
        StreamEvent::ToolCallResult {
            name,
            result,
            typed_payload,
            id: _,
            action_id: _,
        } => {
            let Some(tool_call_id) = pending.pop(name.as_ref()) else {
                return Vec::new();
            };
            vec![tool_status_update(
                tool_call_id,
                Some(format!("Completed {name}")),
                Some(tool_kind_for_name(name.as_ref())),
                ToolCallStatus::Completed,
                result,
                typed_payload,
            )]
        }
        StreamEvent::Done => Vec::new(),
        StreamEvent::Metadata { .. }
        | StreamEvent::LimitReached { .. }
        | StreamEvent::Compaction { .. }
        | StreamEvent::PhaseTransition { .. }
        | StreamEvent::Status { .. }
        | StreamEvent::AudioChunk { .. }
        | StreamEvent::ImageGenerated { .. }
        | StreamEvent::AttachmentProcessed { .. }
        | StreamEvent::AgentStart
        | StreamEvent::AgentEnd
        | StreamEvent::TurnStart
        | StreamEvent::TurnEnd => Vec::new(),
    }
}

pub(crate) fn tool_kind_for_name(tool_name: &str) -> ToolKind {
    let lowered = tool_name.to_ascii_lowercase();
    if lowered.contains("read") || lowered.contains("file") {
        ToolKind::Read
    } else if lowered.contains("write") || lowered.contains("edit") {
        ToolKind::Edit
    } else if lowered.contains("delete") || lowered.contains("remove") {
        ToolKind::Delete
    } else if lowered.contains("move") || lowered.contains("rename") {
        ToolKind::Move
    } else if lowered.contains("search") || lowered.contains("grep") {
        ToolKind::Search
    } else if lowered.contains("shell")
        || lowered.contains("command")
        || lowered.contains("terminal")
        || lowered.contains("exec")
    {
        ToolKind::Execute
    } else if lowered.contains("fetch") || lowered.contains("http") || lowered.contains("web") {
        ToolKind::Fetch
    } else if lowered.contains("think") || lowered.contains("plan") {
        ToolKind::Think
    } else {
        ToolKind::Other
    }
}

#[cfg(test)]
mod tests {
    use super::{PendingToolCalls, updates_from_stream_event};
    use agent_client_protocol_schema::v1::{ContentBlock, SessionUpdate};
    use caliche::messages::{Message, StreamEvent};
    use serde_json::json;

    fn event(value: serde_json::Value) -> StreamEvent {
        serde_json::from_value(value).expect("stream event")
    }

    fn tool_call_id(update: &SessionUpdate) -> &str {
        match update {
            SessionUpdate::ToolCall(call) => call.tool_call_id.0.as_ref(),
            SessionUpdate::ToolCallUpdate(update) => update.tool_call_id.0.as_ref(),
            _ => panic!("expected a tool update, got {update:?}"),
        }
    }

    #[test]
    fn tool_result_reuses_start_id_not_tool_name() {
        let mut pending = PendingToolCalls::default();
        let start = updates_from_stream_event(
            event(json!({
                "type": "tool_call_start",
                "id": "call-7",
                "name": "fs",
                "arguments": {"op": "read"}
            })),
            &mut pending,
        );
        let update = updates_from_stream_event(
            event(json!({
                "type": "tool_call_result",
                "name": "fs",
                "result": "file body"
            })),
            &mut pending,
        );
        assert_eq!(tool_call_id(&start[0]), "call-7");
        assert_eq!(tool_call_id(&update[0]), "call-7");
        assert_ne!(tool_call_id(&update[0]), "fs");
    }

    #[test]
    fn same_tool_name_pairs_in_start_order() {
        let mut pending = PendingToolCalls::default();
        updates_from_stream_event(
            event(json!({"type":"tool_call_start","id":"a","name":"fs","arguments":{}})),
            &mut pending,
        );
        updates_from_stream_event(
            event(json!({"type":"tool_call_start","id":"b","name":"fs","arguments":{}})),
            &mut pending,
        );
        let first = updates_from_stream_event(
            event(json!({"type":"tool_call_result","name":"fs","result":"one"})),
            &mut pending,
        );
        let second = updates_from_stream_event(
            event(json!({"type":"tool_call_result","name":"fs","result":"two"})),
            &mut pending,
        );
        assert_eq!(tool_call_id(&first[0]), "a");
        assert_eq!(tool_call_id(&second[0]), "b");
    }

    #[test]
    fn history_replay_skips_system_and_keeps_tool_ids() {
        use super::updates_from_history;

        let history = vec![
            message(json!({"role": "system", "content": "hidden"})),
            message(json!({"role": "user", "content": "What's the capital of France?"})),
            message(json!({
                "role": "assistant",
                "content": "Paris",
                "metadata": {"reasoning_content": "recall"}
            })),
            message(json!({
                "role": "assistant",
                "content": {
                    "text": "",
                    "tool_calls": [{
                        "id": "call-7",
                        "name": "fs",
                        "arguments": {"op": "read"}
                    }]
                }
            })),
            message(json!({
                "role": "tool",
                "content": {"text": "file body", "tool_call_id": "call-7"}
            })),
            message(json!({"role": "tool", "content": "orphan"})),
        ];
        let updates = updates_from_history(&history);
        assert!(
            !format!("{updates:?}").contains("hidden"),
            "system text must not replay"
        );
        assert!(matches!(
            &updates[0],
            SessionUpdate::UserMessageChunk(chunk)
                if matches!(&chunk.content, ContentBlock::Text(text) if text.text == "What's the capital of France?")
        ));
        assert!(matches!(&updates[1], SessionUpdate::AgentThoughtChunk(_)));
        assert!(matches!(&updates[2], SessionUpdate::AgentMessageChunk(_)));
        assert_eq!(tool_call_id(&updates[3]), "call-7");
        assert_eq!(tool_call_id(&updates[4]), "call-7");
        assert_eq!(updates.len(), 5);
    }

    fn message(value: serde_json::Value) -> Message {
        serde_json::from_value(value).expect("message")
    }

    #[test]
    fn unmatched_tool_result_is_dropped() {
        let mut pending = PendingToolCalls::default();
        let update = updates_from_stream_event(
            event(json!({"type":"tool_call_result","name":"fs","result":"orphan"})),
            &mut pending,
        );
        assert!(update.is_empty());
    }
}
