mod arc_message;
pub mod compaction_report;
pub mod stream_contracts;

pub use arc_message::{ArcMessage, MessageBatch, MessageExt};
pub use stream_contracts::{
    ACTION_ID_FIELD, STREAM_EVENT_SCHEMA_VERSION, client_event_from_stream_event,
    envelope_dto_from_stream_event, envelope_value_from_stream_event, public_sse_event_name,
    token_usage_to_dto,
};

pub mod chat_response;
pub mod stream_events;

pub use chat_response::ChatResponse;
pub use compaction_report::{CompactionChange, CompactionChangeKind, CompactionReport};
pub use stream_events::{StreamEvent, StreamEventEnvelope};

use crate::utils::beef_helpers::{
    deserialize_cow_str, deserialize_option_cow_str, serialize_cow_str, serialize_option_cow_str,
};
use beef::lean::Cow;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// Metadata key for provider chain-of-thought stored on a [`Message`].
///
/// Review surfaces read this field. Model replay strips it so the prompt
/// budget counts only text that is actually sent.
pub const REASONING_CONTENT_METADATA_KEY: &str = "reasoning_content";

/// Content part for multimodal messages (text, images, audio, videos, documents)
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentPart {
    Text {
        #[serde(
            serialize_with = "serialize_cow_str",
            deserialize_with = "deserialize_cow_str"
        )]
        text: Cow<'static, str>,
    },
    Image {
        /// Base64-encoded image data
        #[serde(
            serialize_with = "serialize_cow_str",
            deserialize_with = "deserialize_cow_str"
        )]
        data: Cow<'static, str>,
        /// Media type (e.g., "image/png", "image/jpeg")
        #[serde(
            serialize_with = "serialize_cow_str",
            deserialize_with = "deserialize_cow_str"
        )]
        media_type: Cow<'static, str>,
    },
    ImageUrl {
        /// URL to the image
        #[serde(
            serialize_with = "serialize_cow_str",
            deserialize_with = "deserialize_cow_str"
        )]
        url: Cow<'static, str>,
    },
    Video {
        /// Base64-encoded video data
        #[serde(
            serialize_with = "serialize_cow_str",
            deserialize_with = "deserialize_cow_str"
        )]
        data: Cow<'static, str>,
        /// Media type (e.g., "video/mp4", "video/webm")
        #[serde(
            serialize_with = "serialize_cow_str",
            deserialize_with = "deserialize_cow_str"
        )]
        media_type: Cow<'static, str>,
    },
    VideoUrl {
        /// URL to the video
        #[serde(
            serialize_with = "serialize_cow_str",
            deserialize_with = "deserialize_cow_str"
        )]
        url: Cow<'static, str>,
    },
    Audio {
        /// Base64-encoded audio data
        #[serde(
            serialize_with = "serialize_cow_str",
            deserialize_with = "deserialize_cow_str"
        )]
        data: Cow<'static, str>,
        /// Media type (e.g., "audio/mpeg", "audio/wav")
        #[serde(
            serialize_with = "serialize_cow_str",
            deserialize_with = "deserialize_cow_str"
        )]
        media_type: Cow<'static, str>,
    },
    AudioUrl {
        /// URL to the audio
        #[serde(
            serialize_with = "serialize_cow_str",
            deserialize_with = "deserialize_cow_str"
        )]
        url: Cow<'static, str>,
    },
    Document {
        /// Base64-encoded document data
        #[serde(
            serialize_with = "serialize_cow_str",
            deserialize_with = "deserialize_cow_str"
        )]
        data: Cow<'static, str>,
        /// Media type (e.g., "application/pdf")
        #[serde(
            serialize_with = "serialize_cow_str",
            deserialize_with = "deserialize_cow_str"
        )]
        media_type: Cow<'static, str>,
        /// Optional file name
        #[serde(
            default,
            skip_serializing_if = "Option::is_none",
            serialize_with = "serialize_option_cow_str",
            deserialize_with = "deserialize_option_cow_str"
        )]
        filename: Option<Cow<'static, str>>,
    },
    DocumentUrl {
        /// URL to the document
        #[serde(
            serialize_with = "serialize_cow_str",
            deserialize_with = "deserialize_cow_str"
        )]
        url: Cow<'static, str>,
        /// Optional file name
        #[serde(
            default,
            skip_serializing_if = "Option::is_none",
            serialize_with = "serialize_option_cow_str",
            deserialize_with = "deserialize_option_cow_str"
        )]
        filename: Option<Cow<'static, str>>,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum MessageContent {
    /// Plain text; stored as Arc to allow zero-copy when appending assistant response from Arc<String>.
    Text(Arc<String>),
    Multimodal(Vec<ContentPart>),
    Structured {
        #[serde(skip_serializing_if = "Option::is_none")]
        text: Option<Arc<String>>,
        #[serde(skip_serializing_if = "Option::is_none")]
        tool_calls: Option<Vec<ToolCallInfo>>,
        #[serde(
            default,
            skip_serializing_if = "Option::is_none",
            serialize_with = "serialize_option_cow_str",
            deserialize_with = "deserialize_option_cow_str"
        )]
        tool_call_id: Option<Cow<'static, str>>,
    },
}

impl MessageContent {
    pub fn text(text: impl Into<String>) -> Self {
        MessageContent::Text(Arc::new(text.into()))
    }

    /// Extract text as an owned String (clones the data)
    pub fn extract_text(&self) -> String {
        match self {
            MessageContent::Text(text) => (**text).clone(),
            MessageContent::Multimodal(parts) => parts
                .iter()
                .filter_map(|part| match part {
                    ContentPart::Text { text } => Some(text.clone()),
                    _ => None,
                })
                .collect::<Vec<_>>()
                .join(" "),
            MessageContent::Structured { text, .. } => text
                .as_deref()
                .map(|s| (*s).clone())
                .unwrap_or_else(|| "".to_string()),
        }
    }

    /// Get a reference to the text content without cloning (zero-copy)
    /// Returns the first text part if available
    pub fn as_text(&self) -> Option<&str> {
        match self {
            MessageContent::Text(text) => Some(text.as_str()),
            MessageContent::Multimodal(parts) => parts.iter().find_map(|part| match part {
                ContentPart::Text { text } => Some(text.as_ref()),
                _ => None,
            }),
            MessageContent::Structured { text, .. } => text.as_deref().map(|s| s.as_str()),
        }
    }

    /// Get text as Cow<str> - borrows when possible, clones only if needed
    pub fn text_cow(&self) -> beef::Cow<'_, str> {
        match self {
            MessageContent::Text(text) => beef::Cow::borrowed(text.as_str()),
            MessageContent::Multimodal(parts) => {
                if let Some(text_part) = parts.iter().find_map(|part| match part {
                    ContentPart::Text { text } => Some(text.as_ref()),
                    _ => None,
                }) {
                    beef::Cow::borrowed(text_part)
                } else {
                    beef::Cow::borrowed("")
                }
            }
            MessageContent::Structured { text, .. } => text
                .as_deref()
                .map(|s| beef::Cow::borrowed(s.as_str()))
                .unwrap_or_else(|| beef::Cow::borrowed("")),
        }
    }

    /// Get a reference to tool calls if present
    pub fn as_tool_calls(&self) -> Option<&[ToolCallInfo]> {
        match self {
            MessageContent::Text(_) => None,
            MessageContent::Multimodal(_) => None,
            MessageContent::Structured { tool_calls, .. } => tool_calls.as_deref(),
        }
    }

    /// Get a reference to the tool call ID if present
    pub fn as_tool_call_id(&self) -> Option<&str> {
        match self {
            MessageContent::Text(_) => None,
            MessageContent::Multimodal(_) => None,
            MessageContent::Structured { tool_call_id, .. } => tool_call_id.as_deref(),
        }
    }

    /// Check if this content contains multimodal parts (images/videos)
    pub fn is_multimodal(&self) -> bool {
        matches!(self, MessageContent::Multimodal(_))
    }

    /// Get multimodal parts if present
    pub fn as_multimodal(&self) -> Option<&[ContentPart]> {
        match self {
            MessageContent::Multimodal(parts) => Some(parts.as_slice()),
            _ => None,
        }
    }
}

impl From<String> for MessageContent {
    fn from(text: String) -> Self {
        MessageContent::Text(Arc::new(text))
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: MessageRole,
    pub content: MessageContent,
    /// Optional metadata for provider-specific features (e.g., prompt caching)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<serde_json::Value>,
}

impl Message {
    pub fn new(role: MessageRole, content: impl Into<String>) -> Self {
        Self {
            role,
            content: MessageContent::Text(Arc::new(content.into())),
            metadata: None,
        }
    }

    /// Attach provider reasoning/thinking text under `metadata.reasoning_content`.
    #[must_use]
    pub fn with_reasoning_content(mut self, reasoning: impl Into<String>) -> Self {
        let text = reasoning.into();
        if text.trim().is_empty() {
            return self;
        }
        let mut meta = self
            .metadata
            .take()
            .unwrap_or_else(|| serde_json::json!({}));
        if let Some(obj) = meta.as_object_mut() {
            obj.insert(
                REASONING_CONTENT_METADATA_KEY.to_string(),
                serde_json::Value::String(text),
            );
        } else {
            meta = serde_json::json!({ REASONING_CONTENT_METADATA_KEY: text });
        }
        self.metadata = Some(meta);
        self
    }

    /// Read provider reasoning stored by [`Self::with_reasoning_content`].
    #[must_use]
    pub fn reasoning_content(&self) -> Option<&str> {
        self.metadata
            .as_ref()
            .and_then(|meta| meta.get(REASONING_CONTENT_METADATA_KEY))
            .and_then(|value| value.as_str())
            .filter(|text| !text.is_empty())
    }

    /// Remove stored chain-of-thought, leaving other metadata in place.
    ///
    /// Saved transcripts keep the field for review. Resume and `/load` clear
    /// it on the copy that is sent back to the model.
    pub fn clear_reasoning_content(&mut self) {
        let Some(mut meta) = self.metadata.take() else {
            return;
        };
        if let Some(obj) = meta.as_object_mut() {
            obj.remove(REASONING_CONTENT_METADATA_KEY);
        }
        if meta.as_object().is_some_and(|obj| !obj.is_empty()) || !meta.is_object() {
            self.metadata = Some(meta);
        }
    }

    /// Clone without [`REASONING_CONTENT_METADATA_KEY`].
    #[must_use]
    pub fn without_reasoning_content(&self) -> Self {
        let mut message = self.clone();
        message.clear_reasoning_content();
        message
    }

    /// Assistant text plus optional `metadata.reasoning_content`.
    ///
    /// Empty or whitespace-only reasoning is omitted, matching
    /// [`Self::with_reasoning_content`]. Display policy is separate: callers
    /// can hide this text and still keep it on the message.
    #[must_use]
    pub fn assistant_with_reasoning(
        content: impl Into<String>,
        reasoning: impl Into<String>,
    ) -> Self {
        Self::assistant(content).with_reasoning_content(reasoning)
    }

    /// Create a user message (convenience constructor)
    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: MessageRole::User,
            content: MessageContent::Text(Arc::new(content.into())),
            metadata: None,
        }
    }

    /// Create an assistant message (convenience constructor)
    pub fn assistant(content: impl Into<String>) -> Self {
        Self {
            role: MessageRole::Assistant,
            content: MessageContent::Text(Arc::new(content.into())),
            metadata: None,
        }
    }

    /// Create a system message (convenience constructor)
    pub fn system(content: impl Into<String>) -> Self {
        Self {
            role: MessageRole::System,
            content: MessageContent::Text(Arc::new(content.into())),
            metadata: None,
        }
    }

    /// Create a tool message (convenience constructor)
    pub fn tool(content: impl Into<String>) -> Self {
        Self {
            role: MessageRole::Tool,
            content: MessageContent::Text(Arc::new(content.into())),
            metadata: None,
        }
    }

    /// Create a user message with an image attachment
    pub fn user_with_image(text: &str, image_data: Vec<u8>, media_type: &str) -> Self {
        use base64::{Engine as _, engine::general_purpose::STANDARD as BASE64};
        let base64_data = BASE64.encode(&image_data);
        Self {
            role: MessageRole::User,
            content: MessageContent::Multimodal(vec![
                ContentPart::Text {
                    text: Cow::owned(text.to_string()),
                },
                ContentPart::Image {
                    data: Cow::owned(base64_data),
                    media_type: Cow::owned(media_type.to_string()),
                },
            ]),
            metadata: None,
        }
    }

    /// Create a user message with an image URL
    pub fn user_with_image_url(text: &str, url: &str) -> Self {
        Self {
            role: MessageRole::User,
            content: MessageContent::Multimodal(vec![
                ContentPart::Text {
                    text: Cow::owned(text.to_string()),
                },
                ContentPart::ImageUrl {
                    url: Cow::owned(url.to_string()),
                },
            ]),
            metadata: None,
        }
    }

    /// Create a user message with an audio attachment
    pub fn user_with_audio(text: &str, audio_data: Vec<u8>, media_type: &str) -> Self {
        let attachment = Attachment {
            data: audio_data,
            media_type: media_type.to_string(),
            filename: None,
            attachment_type: AttachmentType::Audio,
        };
        Self::user_with_attachments(text, vec![attachment])
    }

    /// Create a user message with a document attachment
    pub fn user_with_document(
        text: &str,
        document_data: Vec<u8>,
        media_type: &str,
        filename: Option<String>,
    ) -> Self {
        let attachment = Attachment {
            data: document_data,
            media_type: media_type.to_string(),
            filename,
            attachment_type: AttachmentType::Document,
        };
        Self::user_with_attachments(text, vec![attachment])
    }

    /// Create a user message with mixed attachments
    pub fn user_with_attachments(text: &str, attachments: Vec<Attachment>) -> Self {
        let mut parts = vec![ContentPart::Text {
            text: Cow::owned(text.to_string()),
        }];
        for attachment in attachments {
            parts.push(attachment.to_content_part());
        }
        Self {
            role: MessageRole::User,
            content: MessageContent::Multimodal(parts),
            metadata: None,
        }
    }

    /// Extract text as an owned String (clones the data)
    pub fn extract_text(&self) -> String {
        self.content.extract_text()
    }

    /// Get a reference to the text content without cloning (zero-copy)
    pub fn as_text(&self) -> Option<&str> {
        self.content.as_text()
    }
}

/// Generic attachment for multimodal messages.
#[derive(Debug, Clone)]
pub struct Attachment {
    pub data: Vec<u8>,
    pub media_type: String,
    pub filename: Option<String>,
    pub attachment_type: AttachmentType,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AttachmentType {
    Image,
    Audio,
    Video,
    Document,
}

impl Attachment {
    /// Create an attachment from raw data.
    pub fn new(
        data: Vec<u8>,
        media_type: String,
        filename: Option<String>,
        attachment_type: AttachmentType,
    ) -> Self {
        Self {
            data,
            media_type,
            filename,
            attachment_type,
        }
    }

    /// Create an attachment from a file path.
    pub async fn from_file(path: &std::path::Path) -> anyhow::Result<Self> {
        use tokio::fs;
        let data = fs::read(path).await?;
        let media_type = detect_media_type(path);
        let attachment_type = detect_attachment_type(&media_type);
        let filename = path
            .file_name()
            .and_then(std::ffi::OsStr::to_str)
            .map(std::string::ToString::to_string);
        Ok(Self {
            data,
            media_type,
            filename,
            attachment_type,
        })
    }

    /// Convert attachment to a message content part.
    pub fn to_content_part(&self) -> ContentPart {
        use base64::{Engine as _, engine::general_purpose::STANDARD as BASE64};
        let base64_data = BASE64.encode(&self.data);
        match self.attachment_type {
            AttachmentType::Image => ContentPart::Image {
                data: Cow::owned(base64_data),
                media_type: Cow::owned(self.media_type.clone()),
            },
            AttachmentType::Audio => ContentPart::Audio {
                data: Cow::owned(base64_data),
                media_type: Cow::owned(self.media_type.clone()),
            },
            AttachmentType::Video => ContentPart::Video {
                data: Cow::owned(base64_data),
                media_type: Cow::owned(self.media_type.clone()),
            },
            AttachmentType::Document => ContentPart::Document {
                data: Cow::owned(base64_data),
                media_type: Cow::owned(self.media_type.clone()),
                filename: self.filename.clone().map(Cow::owned),
            },
        }
    }
}

const MEDIA_TYPES: &[(&str, &str)] = &[
    ("png", "image/png"),
    ("jpg", "image/jpeg"),
    ("jpeg", "image/jpeg"),
    ("gif", "image/gif"),
    ("webp", "image/webp"),
    ("bmp", "image/bmp"),
    ("tiff", "image/tiff"),
    ("tif", "image/tiff"),
    ("mp3", "audio/mpeg"),
    ("wav", "audio/wav"),
    ("ogg", "audio/ogg"),
    ("flac", "audio/flac"),
    ("aac", "audio/aac"),
    ("m4a", "audio/mp4"),
    ("mp4", "video/mp4"),
    ("webm", "video/webm"),
    ("mov", "video/quicktime"),
    ("pdf", "application/pdf"),
    (
        "docx",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ),
    (
        "xlsx",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    ),
    (
        "pptx",
        "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    ),
    ("csv", "text/csv"),
    ("tsv", "text/tab-separated-values"),
    ("txt", "text/plain"),
    ("rtf", "application/rtf"),
];

fn media_type_for_extension(ext: &str) -> &'static str {
    for (name, media_type) in MEDIA_TYPES {
        if ext.eq_ignore_ascii_case(name) {
            return media_type;
        }
    }
    "application/octet-stream"
}

/// Detect media type from file extension.
pub fn detect_media_type(path: &std::path::Path) -> String {
    path.extension()
        .and_then(|extension| extension.to_str())
        .map(media_type_for_extension)
        .unwrap_or("application/octet-stream")
        .to_string()
}

/// Detect high-level attachment type from media type.
pub fn detect_attachment_type(media_type: &str) -> AttachmentType {
    if media_type.starts_with("image/") {
        AttachmentType::Image
    } else if media_type.starts_with("audio/") {
        AttachmentType::Audio
    } else if media_type.starts_with("video/") {
        AttachmentType::Video
    } else {
        AttachmentType::Document
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Hash, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum MessageRole {
    System,
    User,
    Assistant,
    #[serde(rename = "tool")]
    Tool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCallInfo {
    #[serde(
        serialize_with = "serialize_cow_str",
        deserialize_with = "deserialize_cow_str"
    )]
    pub id: Cow<'static, str>,
    #[serde(
        serialize_with = "serialize_cow_str",
        deserialize_with = "deserialize_cow_str"
    )]
    pub name: Cow<'static, str>,
    pub arguments: serde_json::Value,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thought_signature: Option<String>,
    /// Time-sortable UUID v7 shared by this call's start and result events.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub action_id: Option<String>,
    /// Cached JSON string representation of arguments
    /// Set this to avoid repeated serialization in provider implementations
    #[serde(skip)]
    pub(crate) arguments_json: Option<std::sync::Arc<String>>,
}

impl ToolCallInfo {
    /// Get arguments as JSON string, using cache if available.
    /// Clones directly from the cached `Arc<String>` when present, avoiding an extra `Arc` bump.
    pub fn arguments_json(&self) -> String {
        if let Some(ref cached) = self.arguments_json {
            cached.as_ref().clone()
        } else {
            serde_json::to_string(&self.arguments).unwrap_or_else(|_| "{}".to_owned())
        }
    }

    /// Get arguments as shared Arc<String>, using cache if available.
    /// Cheap to clone; preferred for provider conversion paths.
    pub fn arguments_json_arc(&self) -> std::sync::Arc<String> {
        if let Some(ref cached) = self.arguments_json {
            std::sync::Arc::clone(cached)
        } else {
            let s = serde_json::to_string(&self.arguments).unwrap_or_else(|_| "{}".to_string());
            std::sync::Arc::new(s)
        }
    }

    /// Create from response with pre-serialized arguments.
    ///
    /// Accepts any `Into<String>` for `id`, `name`, and `arguments_json`.
    pub fn from_json(
        id: impl Into<String>,
        name: impl Into<String>,
        arguments_json: impl Into<String>,
    ) -> Self {
        let raw = arguments_json.into();
        match serde_json::from_str::<serde_json::Value>(&raw) {
            Ok(arguments) => Self {
                id: Cow::owned(id.into()),
                name: Cow::owned(name.into()),
                arguments,
                thought_signature: None,
                action_id: None,
                arguments_json: Some(std::sync::Arc::new(raw)),
            },
            Err(e) => {
                tracing::warn!(
                    "ToolCallInfo::from_json: failed to parse arguments JSON, using empty object: {e}"
                );
                // Keep cached wire JSON aligned with the parsed `arguments` value so
                // providers never send a stale invalid payload via `arguments_json()`.
                Self {
                    id: Cow::owned(id.into()),
                    name: Cow::owned(name.into()),
                    arguments: serde_json::json!({}),
                    thought_signature: None,
                    action_id: None,
                    arguments_json: Some(std::sync::Arc::new("{}".to_string())),
                }
            }
        }
    }

    /// Get id as a String (converts from Cow)
    pub fn id_string(&self) -> String {
        self.id.to_string()
    }

    /// Get name as a String (converts from Cow)
    pub fn name_string(&self) -> String {
        self.name.to_string()
    }

    /// Get id as &str
    pub fn id_str(&self) -> &str {
        self.id.as_ref()
    }

    /// Get name as &str
    pub fn name_str(&self) -> &str {
        self.name.as_ref()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolSchema {
    #[serde(
        serialize_with = "serialize_cow_str",
        deserialize_with = "deserialize_cow_str"
    )]
    pub name: Cow<'static, str>,
    #[serde(
        serialize_with = "serialize_cow_str",
        deserialize_with = "deserialize_cow_str"
    )]
    pub description: Cow<'static, str>,
    pub parameters: ParametersSchema,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParametersSchema {
    #[serde(
        rename = "type",
        serialize_with = "serialize_cow_str",
        deserialize_with = "deserialize_cow_str"
    )]
    pub schema_type: Cow<'static, str>,
    pub properties: serde_json::Value,
    #[serde(
        skip_serializing_if = "Vec::is_empty",
        serialize_with = "crate::utils::beef_helpers::serialize_vec_cow_str",
        deserialize_with = "crate::utils::beef_helpers::deserialize_vec_cow_str"
    )]
    pub required: Vec<Cow<'static, str>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ToolChoice {
    Auto,
    None,
    Required,
    #[serde(untagged)]
    Specific(
        #[serde(
            serialize_with = "serialize_cow_str",
            deserialize_with = "deserialize_cow_str"
        )]
        Cow<'static, str>,
    ),
}

/// Provider reasoning effort hint for models that expose a tunable reasoning budget.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum ReasoningEffort {
    Low,
    Medium,
    High,
}

impl ReasoningEffort {
    /// Parse an inbox / settings token (`low` / `medium` / `high` / `none`).
    ///
    /// `none` / `off` / empty clears the hint. Unknown tokens return `None`.
    #[must_use]
    pub fn from_inbox_token(raw: &str) -> Option<Option<Self>> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "none" | "off" | "" => Some(None),
            "low" => Some(Some(Self::Low)),
            "medium" => Some(Some(Self::Medium)),
            "high" => Some(Some(Self::High)),
            _ => None,
        }
    }

    /// OpenAI-compatible wire value.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Low => "low",
            Self::Medium => "medium",
            Self::High => "high",
        }
    }

    /// Anthropic extended-thinking `budget_tokens` mapping.
    pub const fn anthropic_budget_tokens(self) -> u32 {
        match self {
            Self::Low => 1_024,
            Self::Medium => 8_192,
            Self::High => 32_768,
        }
    }

    /// Google Gemini `thinkingBudget` mapping.
    pub const fn google_thinking_budget(self) -> i32 {
        match self {
            Self::Low => 512,
            Self::Medium => 4_096,
            Self::High => 16_384,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletionRequest {
    pub messages: Arc<Vec<Message>>,
    pub max_tokens: Option<u32>,
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_effort: Option<ReasoningEffort>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Arc<Vec<ToolSchema>>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<ToolChoice>,
    /// Response format for structured outputs (e.g., JSON mode)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_format: Option<crate::output::ResponseFormat>,
}

impl CompletionRequest {
    /// Create a new CompletionRequest with default values
    pub fn new(messages: Arc<Vec<Message>>) -> Self {
        Self {
            messages,
            max_tokens: None,
            temperature: None,
            reasoning_effort: None,
            tools: None,
            tool_choice: None,
            response_format: None,
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, Default, PartialEq, Eq)]
pub struct TokenUsage {
    /// Tokens used for LLM prompts (chat completions)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_tokens: Option<u32>,
    /// Tokens used for LLM completions (chat responses)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub completion_tokens: Option<u32>,
    /// Tokens used for embedding generation (distinct from LLM tokens)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub embedding_tokens: Option<u32>,
    /// Total tokens across all types (prompt + completion + embedding)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub total_tokens: Option<u32>,
    /// Model context window size (for computing utilization percentage)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub context_window: Option<u32>,
    /// Tokens written to a provider prompt cache (Anthropic cache creation).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_creation_tokens: Option<u32>,
    /// Tokens read from a provider prompt cache (Anthropic cache read / OpenAI cached_tokens).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_read_tokens: Option<u32>,
    /// Prompt tokens from the latest model request in this usage snapshot.
    ///
    /// `prompt_tokens` sums every model call in the turn. Context utilization
    /// uses this field so a multi-step turn is not reported as one giant prompt.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_prompt_tokens: Option<u32>,
}

impl TokenUsage {
    /// Create TokenUsage from prompt, completion, and optional total counts.
    /// If total is None, it is computed from prompt + completion when both are Some.
    pub fn from_counts(prompt: Option<u32>, completion: Option<u32>, total: Option<u32>) -> Self {
        Self {
            prompt_tokens: prompt,
            completion_tokens: completion,
            embedding_tokens: None,
            total_tokens: total,
            context_window: None,
            cache_creation_tokens: None,
            cache_read_tokens: None,
            last_prompt_tokens: None,
        }
        .ensure_total_tokens()
    }

    /// Create TokenUsage for embedding responses.
    /// Sets embedding_tokens and total_tokens to `n`. Use `input_tokens` when the API
    /// reports input/prompt tokens separately (e.g. number of items embedded).
    pub fn from_embedding_tokens(n: u32, input_tokens: Option<u32>) -> Self {
        Self {
            prompt_tokens: input_tokens,
            completion_tokens: if input_tokens.is_some() {
                Some(0)
            } else {
                None
            },
            embedding_tokens: Some(n),
            total_tokens: Some(n),
            context_window: None,
            cache_creation_tokens: None,
            cache_read_tokens: None,
            last_prompt_tokens: None,
        }
    }

    /// Attach provider prompt-cache counters. Zero values are stored as `None`.
    #[must_use]
    pub fn with_cache_tokens(mut self, creation: Option<u32>, read: Option<u32>) -> Self {
        self.cache_creation_tokens = creation.filter(|&n| n > 0);
        self.cache_read_tokens = read.filter(|&n| n > 0);
        self
    }

    /// Ensure total_tokens is calculated if it's missing but we have the components
    pub fn ensure_total_tokens(mut self) -> Self {
        if self.total_tokens.is_none() {
            let mut total = 0u32;
            let mut has_any = false;

            if let Some(prompt) = self.prompt_tokens {
                total = total.saturating_add(prompt);
                has_any = true;
            }
            if let Some(completion) = self.completion_tokens {
                total = total.saturating_add(completion);
                has_any = true;
            }
            if let Some(embedding) = self.embedding_tokens {
                total = total.saturating_add(embedding);
                has_any = true;
            }

            if has_any && total > 0 {
                self.total_tokens = Some(total);
            }
        }
        self
    }
}

impl std::ops::Add for TokenUsage {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        /// Merge two `Option<u32>` with saturating addition to prevent overflow.
        fn merge(a: Option<u32>, b: Option<u32>) -> Option<u32> {
            match (a, b) {
                (Some(a), Some(b)) => Some(a.saturating_add(b)),
                (some @ Some(_), None) | (None, some @ Some(_)) => some,
                (None, None) => None,
            }
        }

        let result = Self {
            prompt_tokens: merge(self.prompt_tokens, other.prompt_tokens),
            completion_tokens: merge(self.completion_tokens, other.completion_tokens),
            embedding_tokens: merge(self.embedding_tokens, other.embedding_tokens),
            total_tokens: merge(self.total_tokens, other.total_tokens),
            context_window: self.context_window.or(other.context_window),
            cache_creation_tokens: merge(self.cache_creation_tokens, other.cache_creation_tokens),
            cache_read_tokens: merge(self.cache_read_tokens, other.cache_read_tokens),
            last_prompt_tokens: other.last_prompt_tokens.or(self.last_prompt_tokens),
        };
        // Ensure total_tokens is calculated if missing
        result.ensure_total_tokens()
    }
}

/// Per-step timing breakdown for profiling.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct StepMetrics {
    /// Step index (0-based)
    pub step: u32,
    /// LLM call duration for this step (ms)
    pub llm_duration_ms: u64,
    /// Tool execution duration for this step (ms)
    pub tool_duration_ms: u64,
    /// Tool names executed in this step
    pub tools_executed: Vec<String>,
}

/// Performance metrics for a chat or streaming request.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    /// Total wall-clock time for the entire request (ms)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub total_duration_ms: Option<u64>,
    /// Total time spent waiting on LLM provider calls (ms)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub llm_duration_ms: Option<u64>,
    /// Total time spent executing tools (ms)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_duration_ms: Option<u64>,
    /// Framework overhead = total - llm - tool (ms)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub overhead_ms: Option<u64>,
    /// Number of LLM round-trips (steps)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub step_count: Option<u32>,
    /// Time to first token/response from LLM in the first step (ms)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub time_to_first_token_ms: Option<u64>,
    /// Per-step breakdown (optional)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub steps: Option<Vec<StepMetrics>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletionResponse {
    #[serde(flatten)]
    pub content: CompletionContent,
    #[serde(
        default,
        serialize_with = "serialize_option_cow_str",
        deserialize_with = "deserialize_option_cow_str"
    )]
    pub finish_reason: Option<Cow<'static, str>>,
    pub usage: Option<TokenUsage>,
    /// Provider thinking/reasoning trace (e.g. OpenAI-compat `reasoning_content`).
    /// Persisted onto assistant [`Message`] metadata for tool-call replay.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum CompletionContent {
    /// Most specific: text + media + optional tool calls.
    Multimodal {
        content: String,
        media: Vec<CompletionMediaItem>,
        tool_calls: Vec<ToolCallInfo>,
    },
    /// Text with tool calls. Must be tried before [`Self::Text`] so flattened
    /// `CompletionResponse` JSON does not drop `tool_calls` into an ignored field.
    WithToolCalls {
        content: String,
        tool_calls: Vec<ToolCallInfo>,
    },
    Text {
        content: String,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletionMediaItem {
    pub media_type: String,
    pub data: Vec<u8>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub filename: Option<String>,
}

impl CompletionContent {
    /// Extract text as an owned String (clones the data)
    pub fn extract_text(&self) -> String {
        match self {
            CompletionContent::Text { content } => content.clone(),
            CompletionContent::WithToolCalls { content, .. } => content.clone(),
            CompletionContent::Multimodal { content, .. } => content.clone(),
        }
    }

    /// Get a reference to the text content without cloning (zero-copy)
    pub fn as_text(&self) -> &str {
        match self {
            CompletionContent::Text { content } => content.as_str(),
            CompletionContent::WithToolCalls { content, .. } => content.as_str(),
            CompletionContent::Multimodal { content, .. } => content.as_str(),
        }
    }

    /// Extract tool calls as an owned Vec (clones the data)
    pub fn extract_tool_calls(&self) -> Vec<ToolCallInfo> {
        match self {
            CompletionContent::Text { .. } => vec![],
            CompletionContent::WithToolCalls { tool_calls, .. } => tool_calls.clone(),
            CompletionContent::Multimodal { tool_calls, .. } => tool_calls.clone(),
        }
    }

    /// Get a reference to tool calls without cloning (zero-copy)
    pub fn as_tool_calls(&self) -> &[ToolCallInfo] {
        match self {
            CompletionContent::Text { .. } => &[],
            CompletionContent::WithToolCalls { tool_calls, .. } => tool_calls.as_slice(),
            CompletionContent::Multimodal { tool_calls, .. } => tool_calls.as_slice(),
        }
    }

    /// Check if this content has tool calls
    pub fn has_tool_calls(&self) -> bool {
        matches!(
            self,
            CompletionContent::WithToolCalls { .. } | CompletionContent::Multimodal { .. }
        )
    }
}

impl From<String> for CompletionContent {
    fn from(content: String) -> Self {
        CompletionContent::Text { content }
    }
}

#[cfg(test)]
#[path = "message_tests.rs"]
mod message_tests;
