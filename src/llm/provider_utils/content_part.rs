//! Shared conversion of [`ContentPart`] to provider-specific JSON content blocks.
//!
//! Anthropic/Bedrock use a "source" shape; OpenAI/Azure use "`image_url`". This module
//! centralizes the mapping so providers that emit JSON content arrays can reuse it.

use crate::messages::ContentPart;

/// Format for multimodal content blocks in provider request JSON.
#[derive(Clone, Copy, Debug)]
pub enum ContentPartJsonFormat {
    /// Anthropic/Bedrock: `{ type, source: { type, ... } }` for images.
    Anthropic,
    /// OpenAI/Azure: `{ type: "image_url", image_url: { url } }` for images.
    OpenAI,
    /// Google Gemini: `{ text }` and `{ inline_data: { mime_type, data } }`.
    Google,
    /// Bedrock Anthropic payload shape.
    Bedrock,
}

/// Converts content parts to a JSON array of content blocks for the given provider format.
#[must_use]
pub fn content_parts_to_json_array(
    parts: &[ContentPart],
    format: ContentPartJsonFormat,
) -> Vec<serde_json::Value> {
    parts
        .iter()
        .map(|part| content_part_to_json(part, format))
        .collect()
}

fn content_part_to_json(part: &ContentPart, format: ContentPartJsonFormat) -> serde_json::Value {
    match part {
        ContentPart::Text { text } => serde_json::json!({
            "type": "text",
            "text": text
        }),
        ContentPart::Image { data, media_type } => image_to_json(data, media_type, format),
        ContentPart::ImageUrl { url } => image_url_to_json(url, format),
        ContentPart::Video { data, media_type } => video_to_json(data, media_type, format),
        ContentPart::VideoUrl { url } => video_url_to_json(url, format),
        ContentPart::Audio { data, media_type } => audio_to_json(data, media_type, format),
        ContentPart::AudioUrl { url } => audio_url_to_json(url, format),
        ContentPart::Document {
            data,
            media_type,
            filename,
        } => document_to_json(data, media_type, filename.as_deref(), format),
        ContentPart::DocumentUrl { url, filename } => {
            document_url_to_json(url, filename.as_deref(), format)
        }
    }
}

fn image_to_json(data: &str, media_type: &str, format: ContentPartJsonFormat) -> serde_json::Value {
    match format {
        ContentPartJsonFormat::Anthropic => serde_json::json!({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": media_type,
                "data": data
            }
        }),
        ContentPartJsonFormat::OpenAI => serde_json::json!({
            "type": "image_url",
            "image_url": {
                "url": format!("data:{};base64,{}", media_type, data)
            }
        }),
        ContentPartJsonFormat::Google | ContentPartJsonFormat::Bedrock => serde_json::json!({
            "inline_data": {
                "mime_type": media_type,
                "data": data
            }
        }),
    }
}

fn image_url_to_json(url: &str, format: ContentPartJsonFormat) -> serde_json::Value {
    match format {
        ContentPartJsonFormat::Anthropic => serde_json::json!({
            "type": "image",
            "source": {
                "type": "url",
                "url": url
            }
        }),
        ContentPartJsonFormat::OpenAI => serde_json::json!({
            "type": "image_url",
            "image_url": {
                "url": url
            }
        }),
        ContentPartJsonFormat::Google | ContentPartJsonFormat::Bedrock => serde_json::json!({
            "text": format!("[Image URL attachment: {}]", url)
        }),
    }
}

fn video_to_json(data: &str, media_type: &str, format: ContentPartJsonFormat) -> serde_json::Value {
    match format {
        ContentPartJsonFormat::Anthropic => serde_json::json!({
            "type": "text",
            "text": "[Video content not supported by Anthropic API]"
        }),
        ContentPartJsonFormat::OpenAI => serde_json::json!({
            "type": "video_url",
            "video_url": {
                "url": format!("data:{};base64,{}", media_type, data)
            }
        }),
        ContentPartJsonFormat::Google | ContentPartJsonFormat::Bedrock => serde_json::json!({
            "inline_data": {
                "mime_type": media_type,
                "data": data
            }
        }),
    }
}

fn video_url_to_json(url: &str, format: ContentPartJsonFormat) -> serde_json::Value {
    match format {
        ContentPartJsonFormat::Anthropic => serde_json::json!({
            "type": "text",
            "text": "[Video content not supported by Anthropic API]"
        }),
        ContentPartJsonFormat::OpenAI => serde_json::json!({
            "type": "video_url",
            "video_url": {
                "url": url
            }
        }),
        ContentPartJsonFormat::Google | ContentPartJsonFormat::Bedrock => serde_json::json!({
            "text": format!("[Video URL attachment: {}]", url)
        }),
    }
}

fn audio_to_json(data: &str, media_type: &str, format: ContentPartJsonFormat) -> serde_json::Value {
    match format {
        ContentPartJsonFormat::Anthropic => serde_json::json!({
            "type": "text",
            "text": "[Audio attachment provided; native audio input unavailable for this provider path]"
        }),
        ContentPartJsonFormat::OpenAI => serde_json::json!({
            "type": "input_audio",
            "input_audio": {
                "data": data,
                "format": media_type.split('/').nth(1).unwrap_or("wav")
            }
        }),
        ContentPartJsonFormat::Google | ContentPartJsonFormat::Bedrock => serde_json::json!({
            "inline_data": {
                "mime_type": media_type,
                "data": data
            }
        }),
    }
}

fn audio_url_to_json(url: &str, format: ContentPartJsonFormat) -> serde_json::Value {
    match format {
        ContentPartJsonFormat::Anthropic => serde_json::json!({
            "type": "text",
            "text": format!("[Audio URL attachment: {}]", url)
        }),
        ContentPartJsonFormat::OpenAI => serde_json::json!({
            "type": "input_audio",
            "input_audio": {
                "url": url
            }
        }),
        ContentPartJsonFormat::Google | ContentPartJsonFormat::Bedrock => serde_json::json!({
            "text": format!("[Audio URL attachment: {}]", url)
        }),
    }
}

fn document_to_json(
    data: &str,
    media_type: &str,
    filename: Option<&str>,
    format: ContentPartJsonFormat,
) -> serde_json::Value {
    match format {
        ContentPartJsonFormat::Anthropic | ContentPartJsonFormat::Bedrock => {
            serde_json::json!({
                "type": "document",
                "source": {
                    "type": "base64",
                    "media_type": media_type,
                    "data": data
                },
                "title": filename
            })
        }
        ContentPartJsonFormat::OpenAI => serde_json::json!({
            "type": "file",
            "file": {
                "filename": filename,
                "file_data": format!("data:{};base64,{}", media_type, data)
            }
        }),
        ContentPartJsonFormat::Google => serde_json::json!({
            "inline_data": {
                "mime_type": media_type,
                "data": data
            }
        }),
    }
}

fn document_url_to_json(
    url: &str,
    filename: Option<&str>,
    format: ContentPartJsonFormat,
) -> serde_json::Value {
    match format {
        ContentPartJsonFormat::Anthropic | ContentPartJsonFormat::Bedrock => {
            serde_json::json!({
                "type": "document",
                "source": {
                    "type": "url",
                    "url": url
                },
                "title": filename
            })
        }
        ContentPartJsonFormat::OpenAI => serde_json::json!({
            "type": "file",
            "file": {
                "filename": filename,
                "file_data": url
            }
        }),
        ContentPartJsonFormat::Google => serde_json::json!({
            "text": format!("[Document URL attachment{}: {}]",
                filename.map(|f| format!(" ({f})")).unwrap_or_default(),
                url
            )
        }),
    }
}
