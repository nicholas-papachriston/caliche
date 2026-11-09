//! Shared multipart form parser for chat and chat-stream upload endpoints.

use crate::handlers::chat_contracts::{dto_history_to_core, parse_mode_text};
use crate::handlers::multipart_limits::{
    MAX_ATTACHMENTS, read_attachment_bytes, read_text_field, too_many_attachments,
};
use axum::extract::Multipart;
use axum::http::StatusCode;
use caliche::messages::Message;
use caliche_contracts::chat::{ChatMessageDto, ChatRequestDto};
use caliche_contracts::settings::InteractionModeDto;

/// Parsed multipart chat payload shared by sync and streaming handlers.
pub struct ParsedMultipartChat {
    pub request: ChatRequestDto,
    pub history_override: Option<Vec<Message>>,
}

/// Parse a multipart chat upload into a [`ChatRequestDto`] and optional history override.
pub async fn parse_multipart_chat(
    mut multipart: Multipart,
) -> Result<ParsedMultipartChat, (StatusCode, String)> {
    let mut message = String::new();
    let mut provider: Option<String> = None;
    let mut model: Option<String> = None;
    let mut tools: Option<Vec<String>> = None;
    let mut messages: Option<Vec<ChatMessageDto>> = None;
    let mut mode: Option<InteractionModeDto> = None;
    let mut context_window: Option<usize> = None;
    let mut attachments: Vec<caliche::messages::Attachment> = Vec::new();

    while let Some(field) = multipart.next_field().await.map_err(|e| {
        (
            StatusCode::BAD_REQUEST,
            format!("Invalid multipart data: {}", e),
        )
    })? {
        let field_name = field.name().unwrap_or_default().to_string();
        if field_name == "attachments" || field_name == "attachments[]" {
            if attachments.len() >= MAX_ATTACHMENTS {
                return Err(too_many_attachments());
            }
            let (filename, data, media_type) = read_attachment_bytes(field).await?;
            attachments.push(caliche::messages::Attachment {
                data,
                media_type: media_type.clone(),
                filename,
                attachment_type: caliche::messages::detect_attachment_type(&media_type),
            });
            continue;
        }

        let text = read_text_field(field, &field_name).await?;
        match field_name.as_str() {
            "message" => message = text,
            "provider" => provider = Some(text),
            "model" => model = Some(text),
            "tools" => {
                tools = serde_json::from_str::<Vec<String>>(&text).ok();
            }
            "messages" => {
                messages = serde_json::from_str::<Vec<ChatMessageDto>>(&text).ok();
            }
            "mode" => {
                mode = parse_mode_text(&text);
            }
            "context_window" => {
                context_window = text.trim().parse::<usize>().ok();
            }
            _ => {}
        }
    }

    let final_message = if attachments.is_empty() {
        message.clone()
    } else {
        "Please process the previously attached multimodal input.".to_string()
    };

    let history_override = if attachments.is_empty() {
        None
    } else {
        let mut history = dto_history_to_core(messages.take());
        history.push(Message::user_with_attachments(&message, attachments));
        Some(history)
    };

    let request = ChatRequestDto {
        message: final_message,
        provider,
        model,
        context_window,
        tools,
        messages,
        mode,
    };

    Ok(ParsedMultipartChat {
        request,
        history_override,
    })
}
