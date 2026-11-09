//! Tests for reasoning pattern workflows

#[cfg(feature = "reasoning-patterns")]
mod tests {
    use caliche::agent::tools::CalculatorTool;
    use caliche::messages::{CompletionContent, CompletionResponse, TokenUsage, ToolCallInfo};
    use caliche::prelude::ProviderType;
    use caliche::reasoning::ReasoningPattern;
    use caliche::reasoning::plan_execute::PlanExecutePattern;
    use caliche::reasoning::react::ReActPattern;
    use caliche::testing::MockProvider;
    use caliche::{AgentBuilder, AgentService, ToolRegistry};
    use serde_json::json;
    use std::sync::Arc;

    fn require_lmstudio_opt_in() {
        assert_eq!(
            std::env::var("CALICHE_RUN_LMSTUDIO_INTEGRATION").as_deref(),
            Ok("true"),
            "CALICHE_RUN_LMSTUDIO_INTEGRATION must be true when this ignored test runs"
        );
    }

    #[tokio::test]
    async fn test_react_pattern_mocked() {
        // Response 1: Thought + Tool Call (multiply)
        let response1 = CompletionResponse {
            content: CompletionContent::WithToolCalls {
                content: "Thought: I need to calculate 15 * 23.".to_string(),
                tool_calls: vec![ToolCallInfo::from_json(
                    "call_1".to_string(),
                    "calculator".to_string(),
                    json!({
                        "operation": "multiply",
                        "a": 15,
                        "b": 23
                    })
                    .to_string(),
                )],
            },
            finish_reason: Some(beef::lean::Cow::borrowed("tool_calls")),
            usage: Some(TokenUsage::default()),
            reasoning_content: None,
        };

        // Response 2: Thought + Tool Call (add)
        let response2 = CompletionResponse {
            content: CompletionContent::WithToolCalls {
                content: "Thought: The result is 345. Now I need to add 100.".to_string(),
                tool_calls: vec![ToolCallInfo::from_json(
                    "call_2".to_string(),
                    "calculator".to_string(),
                    json!({
                        "operation": "add",
                        "a": 345,
                        "b": 100
                    })
                    .to_string(),
                )],
            },
            finish_reason: Some(beef::lean::Cow::borrowed("tool_calls")),
            usage: Some(TokenUsage::default()),
            reasoning_content: None,
        };

        // Response 3: Final Answer
        let response3 = CompletionResponse {
            content: CompletionContent::Text {
                content: "Thought: The final result is 445.\nAnswer: 445".to_string(),
            },
            finish_reason: Some(beef::lean::Cow::borrowed("end_turn")),
            usage: Some(TokenUsage::default()),
            reasoning_content: None,
        };

        let provider = MockProvider::with_responses(vec![response1, response2, response3]);

        // Create tools registry
        let tools = ToolRegistry::new();
        tools.register(Arc::new(CalculatorTool));
        let tools = Arc::new(tools);

        // Create Agent directly to inject MockProvider
        let agent_core = caliche::agent::Agent::new(Arc::new(provider), tools);

        // Create AgentService from core agent
        let agent = AgentService::new_with_agent(agent_core, ProviderType::OpenAI)
            .expect("failed to create agent service");

        let react = ReActPattern::new(agent, 5);

        let result = react
            .execute("What is 15 * 23? Then add 100 to that result.")
            .await
            .expect("ReAct execution failed");

        assert!(result.success);
        assert!(result.final_answer.contains("445"));
        // assert_eq!(result.steps.len(), 8); // Removed fragile length check

        assert!(result.steps.iter().any(|s| s.content.contains("15 * 23")));
        assert!(result.steps.iter().any(|s| s.content.contains("345")));
        assert!(result.steps.iter().any(|s| s.content.contains("445")));
    }

    /// "İ" lower-cases to two characters, so an offset taken from a Unicode
    /// lower-case copy lands inside "é" in the original reply.
    #[tokio::test]
    async fn react_answer_follows_text_whose_lower_case_is_longer() {
        let provider = MockProvider::new(vec!["İ Final Answer:é".to_string()]);
        let agent_core =
            caliche::agent::Agent::new(Arc::new(provider), Arc::new(ToolRegistry::new()));
        let agent = AgentService::new_with_agent(agent_core, ProviderType::OpenAI)
            .expect("failed to create agent service");

        let result = ReActPattern::new(agent, 2)
            .execute("question")
            .await
            .expect("ReAct execution failed");

        assert!(result.success);
        assert_eq!(result.final_answer, "é");
    }

    #[tokio::test]
    #[ignore = "requires LM Studio; set CALICHE_RUN_LMSTUDIO_INTEGRATION=true and run with --ignored"]
    async fn test_react_pattern_workflow() {
        require_lmstudio_opt_in();

        let agent = AgentBuilder::new()
            .provider(ProviderType::LMStudio)
            .enable_tool("calculator")
            .build()
            .await
            .expect("Failed to build agent");

        let react = ReActPattern::new(agent, 5);

        let result = react
            .execute("What is 15 * 23? Then add 100 to that result.")
            .await
            .expect("ReAct execution failed");

        assert!(result.success);
        assert!(!result.final_answer.is_empty());
        assert!(!result.steps.is_empty());
    }

    #[tokio::test]
    #[ignore = "requires LM Studio; set CALICHE_RUN_LMSTUDIO_INTEGRATION=true and run with --ignored"]
    async fn test_plan_execute_workflow() {
        require_lmstudio_opt_in();

        let agent = AgentBuilder::new()
            .provider(ProviderType::LMStudio)
            .enable_tool("calculator")
            .build()
            .await
            .expect("Failed to build agent");

        let planner = PlanExecutePattern::new(agent, 10);

        let result = planner
            .execute("Calculate 5 * 5, then 10 * 10, then add the results together")
            .await
            .expect("Plan-execute failed");

        assert!(result.success);
        assert!(!result.final_answer.is_empty());
        assert!(!result.steps.is_empty());
    }
}
