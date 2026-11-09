//! Tests for sandbox mode and mock providers

#[cfg(feature = "testing")]
mod tests {
    use caliche::agent::service::sandbox::{AgentSandbox, MockBehavior};
    use caliche::llm::ProviderType;

    #[tokio::test]
    async fn test_sandbox_creation() {
        let sandbox = AgentSandbox::new()
            .mock_provider(ProviderType::OpenAI, MockBehavior::Deterministic)
            .build();

        let mock_provider = sandbox.create_mock_provider(ProviderType::OpenAI);
        assert!(mock_provider.is_some());
    }

    #[tokio::test]
    async fn test_mock_provider_deterministic() {
        use caliche::llm::Provider;
        use caliche::messages::CompletionRequest;

        let sandbox = AgentSandbox::new()
            .mock_provider(ProviderType::OpenAI, MockBehavior::Deterministic)
            .build();

        let provider = sandbox
            .create_mock_provider(ProviderType::OpenAI)
            .expect("Failed to create mock provider");

        let request = CompletionRequest {
            messages: std::sync::Arc::new(vec![caliche::messages::Message {
                role: caliche::messages::MessageRole::User,
                content: caliche::messages::MessageContent::text("Test message".to_string()),
                metadata: None,
            }]),
            max_tokens: Some(100),
            temperature: Some(0.7),
            reasoning_effort: None,
            tools: None,
            tool_choice: None,
            response_format: None,
        };

        let response = provider
            .complete(request)
            .await
            .expect("Failed to complete");
        let content = response.content.extract_text();
        assert!(!content.is_empty());
        assert!(response.usage.is_some());
    }

    #[tokio::test]
    async fn test_mock_provider_fixed() {
        use caliche::llm::Provider;
        use caliche::messages::CompletionRequest;

        let sandbox = AgentSandbox::new()
            .mock_provider(
                ProviderType::OpenAI,
                MockBehavior::Fixed("Fixed response".to_string()),
            )
            .build();

        let provider = sandbox
            .create_mock_provider(ProviderType::OpenAI)
            .expect("Failed to create mock provider");

        let request = CompletionRequest {
            messages: std::sync::Arc::new(vec![caliche::messages::Message {
                role: caliche::messages::MessageRole::User,
                content: caliche::messages::MessageContent::text("Any message".to_string()),
                metadata: None,
            }]),
            max_tokens: Some(100),
            temperature: Some(0.7),
            reasoning_effort: None,
            tools: None,
            tool_choice: None,
            response_format: None,
        };

        let response = provider
            .complete(request)
            .await
            .expect("Failed to complete");
        let content = response.content.extract_text();
        assert_eq!(content, "Fixed response");
    }

    #[tokio::test]
    async fn test_mock_provider_streaming() {
        use caliche::llm::Provider;
        use caliche::messages::CompletionRequest;
        use futures_util::StreamExt;

        let sandbox = AgentSandbox::new()
            .mock_provider(ProviderType::OpenAI, MockBehavior::Deterministic)
            .build();

        let provider = sandbox
            .create_mock_provider(ProviderType::OpenAI)
            .expect("Failed to create mock provider");

        let request = CompletionRequest {
            messages: std::sync::Arc::new(vec![caliche::messages::Message {
                role: caliche::messages::MessageRole::User,
                content: caliche::messages::MessageContent::text("Test".to_string()),
                metadata: None,
            }]),
            max_tokens: Some(100),
            temperature: Some(0.7),
            reasoning_effort: None,
            tools: None,
            tool_choice: None,
            response_format: None,
        };

        let mut stream = provider.stream(request);
        let mut chunks = Vec::new();
        while let Some(event) = stream.next().await {
            chunks.push(event.expect("Stream error"));
        }

        assert!(!chunks.is_empty());
    }
}
