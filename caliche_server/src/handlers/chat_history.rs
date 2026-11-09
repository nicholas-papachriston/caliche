use caliche::messages::{Message, MessageRole};

/// Normalize browser-supplied history before appending the current user turn.
///
/// `AgentService::chat_with_history` appends the request's current user message
/// after this history, so a trailing assistant message is a valid completed
/// prior response, not assistant prefill.
pub(crate) fn trim_trailing_non_conversational_turns(mut history: Vec<Message>) -> Vec<Message> {
    while matches!(
        history.last().map(|message| &message.role),
        Some(MessageRole::Tool | MessageRole::System)
    ) {
        history.pop();
    }
    history
}

#[cfg(test)]
mod tests {
    use super::*;
    use caliche::messages::MessageContent;

    fn message(role: MessageRole, content: &str) -> Message {
        Message {
            role,
            content: MessageContent::text(content.to_string()),
            metadata: None,
        }
    }

    #[test]
    fn trims_trailing_tool_and_system_turns() {
        let history = vec![
            message(MessageRole::User, "what is this repo?"),
            message(MessageRole::Assistant, "<tool_call>{...}</tool_call>"),
            message(MessageRole::Tool, "tool result"),
            message(MessageRole::System, "internal note"),
        ];

        let trimmed = trim_trailing_non_conversational_turns(history);

        assert_eq!(trimmed.len(), 2);
        assert!(matches!(trimmed[0].role, MessageRole::User));
        assert!(matches!(trimmed[1].role, MessageRole::Assistant));
    }

    #[test]
    fn keeps_history_that_ends_with_assistant() {
        let history = vec![
            message(MessageRole::User, "first"),
            message(MessageRole::Assistant, "answer"),
        ];

        let trimmed = trim_trailing_non_conversational_turns(history);

        assert_eq!(trimmed.len(), 2);
        assert!(matches!(
            trimmed.last().map(|message| &message.role),
            Some(MessageRole::Assistant)
        ));
    }
}
