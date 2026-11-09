//! Chat-specific prompt templates for constructing multi-turn conversations.

use super::template::{PromptTemplate, TemplateError};
use serde_json::Value;

/// A role in a chat conversation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    System,
    User,
    Assistant,
}

impl Role {
    pub fn as_str(&self) -> &str {
        match self {
            Role::System => "system",
            Role::User => "user",
            Role::Assistant => "assistant",
        }
    }
}

/// A single message in a chat template.
#[derive(Debug, Clone)]
pub struct ChatMessage {
    role: Role,
    template: PromptTemplate,
}

impl ChatMessage {
    pub fn new(role: Role, template: impl Into<String>) -> Self {
        Self {
            role,
            template: PromptTemplate::new(template),
        }
    }

    pub fn system(template: impl Into<String>) -> Self {
        Self::new(Role::System, template)
    }

    pub fn user(template: impl Into<String>) -> Self {
        Self::new(Role::User, template)
    }

    pub fn assistant(template: impl Into<String>) -> Self {
        Self::new(Role::Assistant, template)
    }

    pub fn role(&self) -> Role {
        self.role
    }

    pub fn format(&self, values: &Value) -> Result<String, TemplateError> {
        self.template.format(values)
    }
}

/// A template for chat-based prompts with multiple messages.
///
/// Useful for constructing system prompts, user queries, and assistant responses
/// with variable substitution.
///
/// # Examples
///
/// ```rust
/// use caliche::prompts::{ChatPromptTemplate, ChatMessage};
/// use serde_json::json;
///
/// let template = ChatPromptTemplate::new(vec![
///     ChatMessage::system("You are a helpful {role}."),
///     ChatMessage::user("Please help me with: {task}"),
/// ]);
///
/// let messages = template.format(&json!({
///     "role": "assistant",
///     "task": "writing code"
/// })).expect("Template formatting should succeed");
/// ```
#[derive(Debug, Clone)]
pub struct ChatPromptTemplate {
    messages: Vec<ChatMessage>,
}

impl ChatPromptTemplate {
    /// Create a new chat prompt template with messages.
    pub fn new(messages: Vec<ChatMessage>) -> Self {
        Self { messages }
    }

    /// Create with a system message.
    pub fn with_system(system: impl Into<String>) -> Self {
        Self {
            messages: vec![ChatMessage::system(system)],
        }
    }

    /// Add a message to the template.
    pub fn add_message(mut self, message: ChatMessage) -> Self {
        self.messages.push(message);
        self
    }

    /// Add a system message.
    pub fn add_system(mut self, template: impl Into<String>) -> Self {
        self.messages.push(ChatMessage::system(template));
        self
    }

    /// Add a user message.
    pub fn add_user(mut self, template: impl Into<String>) -> Self {
        self.messages.push(ChatMessage::user(template));
        self
    }

    /// Add an assistant message.
    pub fn add_assistant(mut self, template: impl Into<String>) -> Self {
        self.messages.push(ChatMessage::assistant(template));
        self
    }

    /// Format all messages with the given variables.
    ///
    /// Returns a vector of (role, content) tuples.
    pub fn format(&self, values: &Value) -> Result<Vec<(String, String)>, TemplateError> {
        let mut result = Vec::new();

        for message in &self.messages {
            let content = message.format(values)?;
            result.push((message.role().as_str().to_owned(), content));
        }

        Ok(result)
    }

    /// Format as a single string with role labels.
    ///
    /// Useful for models that don't have a chat API.
    pub fn format_as_string(&self, values: &Value) -> Result<String, TemplateError> {
        let messages = self.format(values)?;
        let mut parts = Vec::new();

        for (role, content) in messages {
            parts.push(format!("{}: {}", role.to_uppercase(), content));
        }

        Ok(parts.join("\n\n"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_chat_template_basic() {
        let template = ChatPromptTemplate::new(vec![
            ChatMessage::system("You are a {role}."),
            ChatMessage::user("Help me with {task}."),
        ]);

        let messages = template
            .format(&json!({
                "role": "teacher",
                "task": "math"
            }))
            .expect("Should format chat template");

        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0].0, "system");
        assert_eq!(messages[0].1, "You are a teacher.");
        assert_eq!(messages[1].0, "user");
        assert_eq!(messages[1].1, "Help me with math.");
    }

    #[test]
    fn test_chat_template_builder() {
        let template = ChatPromptTemplate::with_system("System: {sys}")
            .add_user("User: {usr}")
            .add_assistant("Assistant: {ast}");

        let messages = template
            .format(&json!({
                "sys": "s1",
                "usr": "u1",
                "ast": "a1"
            }))
            .expect("Should format chat template");

        assert_eq!(messages.len(), 3);
        assert_eq!(messages[0].0, "system");
        assert_eq!(messages[1].0, "user");
        assert_eq!(messages[2].0, "assistant");
    }

    #[test]
    fn test_format_as_string() {
        let template = ChatPromptTemplate::new(vec![
            ChatMessage::system("System message"),
            ChatMessage::user("User message"),
        ]);

        let result = template
            .format_as_string(&json!({}))
            .expect("Should format template as string");
        assert!(result.contains("SYSTEM: System message"));
        assert!(result.contains("USER: User message"));
    }
}
