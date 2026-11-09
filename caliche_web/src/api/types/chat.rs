//! Chat, streaming, and multimodal message types.

use serde::{Deserialize, Serialize};

pub use caliche_contracts::chat::{
    ChatMessageDto as ChatMessage, ChatMessageRoleDto as MessageRole,
    ChatRequestDto as ChatRequest, ChatResponseDto as ChatResponse, TokenUsageDto as TokenUsage,
};
pub use caliche_contracts::stream::{
    ClientStreamEventDto as StreamingEvent, ClientStreamEventTypeDto as StreamingEventType,
    StreamEventEnvelopeDto,
};

/// Health check response
#[derive(Debug, Clone, Serialize, Deserialize)]
#[allow(
    dead_code,
    reason = "nothing builds it yet; lib.rs and main.rs both compile this module, so expect cannot hold in both"
)]
pub struct HealthResponse {
    pub status: String,
}

/// A media file attached by the user before sending
#[derive(Debug, Clone, PartialEq)]
pub struct MediaAttachment {
    /// Original filename
    pub filename: String,
    /// MIME type (e.g. "image/png", "audio/mp3")
    pub media_type: String,
    /// Raw bytes of the file
    pub bytes: Vec<u8>,
    /// Data URL for in-browser image preview (`data:image/...;base64,...`)
    pub preview_url: Option<String>,
}

/// A media item stored inside a Message for display purposes
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MessageMediaItem {
    pub filename: String,
    pub media_type: String,
    /// Data URL for image preview; None for non-image types
    pub preview_url: Option<String>,
}
