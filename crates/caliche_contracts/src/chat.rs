use serde::{Deserialize, Serialize};

use crate::settings::InteractionModeDto;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ChatMessageRoleDto {
    User,
    Assistant,
}

impl core::fmt::Display for ChatMessageRoleDto {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::User => write!(f, "user"),
            Self::Assistant => write!(f, "assistant"),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChatMessageDto {
    pub role: ChatMessageRoleDto,
    pub content: String,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChatRequestDto {
    pub message: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provider: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub context_window: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<String>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub messages: Option<Vec<ChatMessageDto>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mode: Option<InteractionModeDto>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct TokenUsageDto {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub completion_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub total_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub embedding_tokens: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_creation_tokens: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_read_tokens: Option<u32>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SourceCitationDto {
    #[serde(rename = "type")]
    pub source_type: String,
    pub resource: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<serde_json::Value>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChatResponseDto {
    pub response: String,
    pub request_id: String,
    pub usage: TokenUsageDto,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sources: Option<Vec<SourceCitationDto>>,
    pub mode: InteractionModeDto,
}

/// Remove a trailing provider citation footer (`\n\nSources:\n...`) when present.
#[must_use]
pub fn strip_sources_footer(text: &str) -> &str {
    match text.rfind("\n\nSources:\n") {
        Some(idx) => &text[..idx],
        None => text,
    }
}

/// Parse a trailing `Sources:` footer into citation DTOs.
#[must_use]
pub fn parse_sources_footer(text: &str) -> Option<Vec<SourceCitationDto>> {
    let idx = text.rfind("\n\nSources:\n")?;
    let footer = text.get(idx.saturating_add("\n\nSources:\n".len())..)?;
    let mut sources = Vec::new();
    for line in footer.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let resource = extract_source_resource(trimmed);
        if resource.is_empty() {
            continue;
        }
        sources.push(SourceCitationDto {
            source_type: source_type_for_resource(resource),
            resource: resource.to_string(),
            metadata: None,
        });
    }
    if sources.is_empty() {
        None
    } else {
        Some(sources)
    }
}

fn extract_source_resource(line: &str) -> &str {
    let trimmed = line.trim_start_matches('-').trim();
    if let Some(rest) = trimmed.strip_prefix('[')
        && let Some(close) = rest.find(']')
    {
        return rest[close.saturating_add(1)..].trim();
    }
    trimmed
}

fn source_type_for_resource(resource: &str) -> String {
    if resource.starts_with("http://") || resource.starts_with("https://") {
        "url".to_string()
    } else {
        "file".to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chat_request_dto_roundtrips_tools_mode_and_history() {
        let request = ChatRequestDto {
            message: "hello".into(),
            provider: Some("openai".into()),
            model: Some("gpt-4o".into()),
            context_window: Some(4_000),
            tools: Some(vec!["fs".into(), "bash".into()]),
            messages: Some(vec![
                ChatMessageDto {
                    role: ChatMessageRoleDto::User,
                    content: "hi".into(),
                },
                ChatMessageDto {
                    role: ChatMessageRoleDto::Assistant,
                    content: "hello".into(),
                },
            ]),
            mode: Some(InteractionModeDto::Plan),
        };
        let value = serde_json::to_value(&request).expect("serialize");
        assert_eq!(value["mode"], "plan");
        assert_eq!(value["tools"].as_array().expect("tools").len(), 2);
        assert_eq!(value["messages"].as_array().expect("messages").len(), 2);
        let decoded: ChatRequestDto = serde_json::from_value(value).expect("deserialize");
        assert_eq!(decoded, request);
    }

    #[test]
    fn chat_response_dto_roundtrips_usage_and_mode() {
        let response = ChatResponseDto {
            response: "ok".into(),
            request_id: "req-1".into(),
            usage: TokenUsageDto {
                prompt_tokens: Some(1),
                completion_tokens: Some(2),
                total_tokens: Some(3),
                embedding_tokens: None,
                cache_creation_tokens: None,
                cache_read_tokens: None,
            },
            sources: Some(vec![SourceCitationDto {
                source_type: "file".into(),
                resource: "src/main.rs".into(),
                metadata: Some(serde_json::json!({"line": 1})),
            }]),
            mode: InteractionModeDto::Agent,
        };
        let value = serde_json::to_value(&response).expect("serialize");
        assert_eq!(value["mode"], "agent");
        assert_eq!(value["usage"]["total_tokens"], 3);
        let decoded: ChatResponseDto = serde_json::from_value(value).expect("deserialize");
        assert_eq!(decoded, response);
    }

    #[test]
    fn strip_sources_footer_removes_trailing_citation_block() {
        let text = "Answer.\n\nSources:\n[1] https://example.com\n[2] src/lib.rs";
        assert_eq!(strip_sources_footer(text), "Answer.");
        assert_eq!(strip_sources_footer("no footer"), "no footer");
    }

    #[test]
    fn parse_sources_footer_reads_numbered_and_dashed_entries() {
        let text = "Answer.\n\nSources:\n[1] https://example.com\n- src/main.rs\n";
        let sources = parse_sources_footer(text).expect("sources");
        assert_eq!(sources.len(), 2);
        assert_eq!(sources[0].source_type, "url");
        assert_eq!(sources[0].resource, "https://example.com");
        assert_eq!(sources[1].source_type, "file");
        assert_eq!(sources[1].resource, "src/main.rs");
    }
}
