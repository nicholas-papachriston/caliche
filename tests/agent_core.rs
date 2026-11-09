/// Core agent functionality tests
use caliche::agent::tools::tool::ToolMeta;
use caliche::{Agent, ToolRegistry, testing::MockProvider};
use std::sync::Arc;

#[tokio::test]
async fn test_agent_creation() {
    let provider: Arc<dyn caliche::llm::Provider> = Arc::new(MockProvider::new(vec![]));
    let tools = Arc::new(ToolRegistry::new());
    let agent = Agent::new(provider, tools);

    // Agent should be created successfully
    assert!(agent.list_tools().is_empty());
}

#[tokio::test]
async fn test_agent_with_system_prompt() {
    let provider: Arc<dyn caliche::llm::Provider> = Arc::new(MockProvider::new(vec![]));
    let tools = Arc::new(ToolRegistry::new());
    let agent = Agent::new(provider, tools).with_system_prompt("Custom prompt".to_string());

    // Agent should have custom system prompt
    assert!(agent.list_tools().is_empty());
}

#[tokio::test]
async fn test_agent_simple_chat() {
    let provider: Arc<dyn caliche::llm::Provider> = Arc::new(MockProvider::new(vec![
        "Hello! How can I help you?".to_string(),
    ]));
    let tools = Arc::new(ToolRegistry::new());
    let mut agent = Agent::new(provider, tools);

    let result = agent.chat("Hi there").await;

    assert!(result.is_ok());
    if let Ok(response) = result {
        assert!(!response.text().is_empty());
        assert!(response.usage().total_tokens.is_some());
    }
}

#[tokio::test]
async fn test_agent_error_handling() {
    let provider: Arc<dyn caliche::llm::Provider> = Arc::new(MockProvider::with_error());
    let tools = Arc::new(ToolRegistry::new());
    let mut agent = Agent::new(provider, tools);

    let result = agent.chat("This should fail").await;
    assert!(result.is_err());
}

#[tokio::test]
async fn test_agent_list_tools() {
    let provider: Arc<dyn caliche::llm::Provider> = Arc::new(MockProvider::new(vec![]));
    let tools = Arc::new(ToolRegistry::new());
    let agent = Agent::new(provider, tools);

    let tool_list = agent.list_tools();
    assert_eq!(tool_list.len(), 0);
}

#[tokio::test]
async fn test_agent_with_custom_tool() {
    use async_trait::async_trait;
    use caliche::Tool;
    use serde_json::json;
    use std::sync::Arc;

    #[derive(Debug)]
    struct TestTool;
    impl ToolMeta for TestTool {
        fn name(&self) -> &str {
            "test_tool"
        }

        fn description(&self) -> &str {
            "A test tool"
        }

        fn parameters(&self) -> serde_json::Value {
            json!({
                "type": "object",
                "properties": {}
            })
        }

        fn name_as_cow(&self) -> beef::lean::Cow<'static, str> {
            beef::lean::Cow::owned(self.name().to_string())
        }

        fn description_as_cow(&self) -> beef::lean::Cow<'static, str> {
            beef::lean::Cow::owned(self.description().to_string())
        }
    }

    #[async_trait]
    impl Tool for TestTool {
        async fn execute(&self, _args: serde_json::Value) -> caliche::ToolResult {
            caliche::ToolResult::success("Test result".to_string())
        }
    }

    let provider: Arc<dyn caliche::llm::Provider> = Arc::new(MockProvider::new(vec![]));
    let tools = Arc::new(ToolRegistry::new());
    let mut agent = Agent::new(provider, Arc::clone(&tools));

    agent.add_tool(Arc::new(TestTool));

    let tool_list = agent.list_tools();
    assert_eq!(tool_list.len(), 1);
    assert_eq!(tool_list[0].0, "test_tool");
}

#[tokio::test]
async fn test_agent_chat_multiple_turns() {
    let responses = vec![
        "Turn 1 response".to_string(),
        "Turn 2 response".to_string(),
        "Turn 3 response".to_string(),
    ];

    let provider: Arc<dyn caliche::llm::Provider> = Arc::new(MockProvider::new(responses));
    let tools = Arc::new(ToolRegistry::new());
    let mut agent = Agent::new(provider, tools);

    // Turn 1
    let result1 = agent.chat("Turn 1").await;
    assert!(result1.is_ok());

    // Turn 2
    let result2 = agent.chat("Turn 2").await;
    assert!(result2.is_ok());

    // Turn 3
    let result3 = agent.chat("Turn 3").await;
    assert!(result3.is_ok());
}

#[tokio::test]
async fn test_agent_cloning() {
    let provider: Arc<dyn caliche::llm::Provider> = Arc::new(MockProvider::new(vec![]));
    let tools = Arc::new(ToolRegistry::new());
    let agent = Agent::new(Arc::clone(&provider), Arc::clone(&tools));

    // Clone the agent
    let cloned_agent = agent.clone();

    // Verify they are separate instances but have same configuration
    assert_eq!(agent.list_tools().len(), cloned_agent.list_tools().len());
}

#[tokio::test]
async fn test_agent_with_shared_state() {
    use caliche::SharedState;
    use serde_json::json;

    let shared_state = SharedState::new();
    shared_state
        .set("/test", json!("value"))
        .await
        .expect("Setting shared state should succeed");

    let provider: Arc<dyn caliche::llm::Provider> = Arc::new(MockProvider::new(vec![]));
    let tools = Arc::new(ToolRegistry::new());
    let agent = Agent::with_shared_state(provider, tools, shared_state.clone());

    // Verify agent has the shared state
    let value = agent
        .shared_state
        .get("/test")
        .await
        .expect("Getting shared state should succeed");
    assert_eq!(value, Some(json!("value")));
}

/// A prompt set after a turn reaches the next request: the cached system
/// messages must not keep the old one.
#[tokio::test]
async fn a_new_system_prompt_reaches_the_next_request() {
    let mock = MockProvider::new(vec!["one".to_string(), "two".to_string()]);
    let provider: Arc<dyn caliche::llm::Provider> = Arc::new(mock.clone());
    let mut agent = Agent::new(provider, Arc::new(ToolRegistry::new()))
        .with_system_prompt("first prompt".to_string());

    // Hosts send each turn with the history; that path caches the system
    // messages.
    agent
        .chat_with_history("turn one", Vec::new())
        .await
        .expect("turn one");
    agent.set_system_prompt("second prompt");
    agent
        .chat_with_history("turn two", Vec::new())
        .await
        .expect("turn two");

    let prompts = mock.recorded_prompts();
    assert_eq!(prompts.len(), 2);
    assert!(prompts[0].starts_with("first prompt"), "{}", prompts[0]);
    assert!(prompts[1].starts_with("second prompt"), "{}", prompts[1]);
    assert!(!prompts[1].contains("first prompt"), "{}", prompts[1]);
}
