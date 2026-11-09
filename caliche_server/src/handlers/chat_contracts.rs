use caliche::agent::modes::InteractionMode;
use caliche::messages::{Message, MessageRole, TokenUsage};
use caliche_contracts::chat::{ChatMessageDto, ChatMessageRoleDto, TokenUsageDto};
use caliche_contracts::settings::InteractionModeDto;

pub(crate) fn dto_mode_to_core(mode: Option<InteractionModeDto>) -> InteractionMode {
    mode.map_or_else(InteractionMode::default, dto_mode_value_to_core)
}

pub(crate) const fn dto_mode_value_to_core(mode: InteractionModeDto) -> InteractionMode {
    match mode {
        InteractionModeDto::Agent => InteractionMode::Agent,
        InteractionModeDto::Ask => InteractionMode::Ask,
        InteractionModeDto::Plan => InteractionMode::Plan,
    }
}

pub(crate) const fn core_mode_to_dto(mode: InteractionMode) -> InteractionModeDto {
    match mode {
        InteractionMode::Agent => InteractionModeDto::Agent,
        InteractionMode::Ask => InteractionModeDto::Ask,
        InteractionMode::Plan => InteractionModeDto::Plan,
    }
}

pub(crate) fn parse_mode_text(text: &str) -> Option<InteractionModeDto> {
    serde_json::from_str::<InteractionModeDto>(text)
        .ok()
        .or_else(|| match text.trim().to_ascii_lowercase().as_str() {
            "agent" => Some(InteractionModeDto::Agent),
            "ask" => Some(InteractionModeDto::Ask),
            "plan" => Some(InteractionModeDto::Plan),
            _ => None,
        })
}

pub(crate) fn dto_history_to_core(messages: Option<Vec<ChatMessageDto>>) -> Vec<Message> {
    messages
        .unwrap_or_default()
        .into_iter()
        .map(|message| {
            let role = match message.role {
                ChatMessageRoleDto::User => MessageRole::User,
                ChatMessageRoleDto::Assistant => MessageRole::Assistant,
            };
            Message::new(role, message.content)
        })
        .collect()
}

pub(crate) fn core_usage_to_dto(usage: &TokenUsage) -> TokenUsageDto {
    TokenUsageDto {
        prompt_tokens: usage.prompt_tokens,
        completion_tokens: usage.completion_tokens,
        total_tokens: usage.total_tokens,
        embedding_tokens: usage.embedding_tokens,
        cache_creation_tokens: usage.cache_creation_tokens,
        cache_read_tokens: usage.cache_read_tokens,
    }
}
