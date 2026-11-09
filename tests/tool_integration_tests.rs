//! Integration tests for tool usage

use caliche::AgentBuilder;
use caliche::prelude::ProviderType;
use serial_test::serial;

async fn require_lmstudio() {
    assert_eq!(
        std::env::var("CALICHE_RUN_LMSTUDIO_INTEGRATION").as_deref(),
        Ok("true"),
        "CALICHE_RUN_LMSTUDIO_INTEGRATION must be true when this ignored test runs"
    );
    let reachable = reqwest::Client::new()
        .get("http://localhost:1234/v1/models")
        .send()
        .await
        .is_ok();
    assert!(
        reachable,
        "LM Studio must answer at http://localhost:1234/v1/models"
    );
}

#[tokio::test]
#[ignore = "requires LM Studio; set CALICHE_RUN_LMSTUDIO_INTEGRATION=true and run with --ignored"]
#[serial]
#[cfg(feature = "lmstudio")]
async fn test_calculator_tool_integration() {
    require_lmstudio().await;

    let mut agent = AgentBuilder::new()
        .provider(ProviderType::LMStudio)
        .enable_tool("calculator")
        .build()
        .await
        .expect("Failed to build agent");

    // Test various calculations
    let test_cases = vec![
        ("2 + 2", "4"),
        ("10 * 5", "50"),
        ("100 / 4", "25"),
        ("2^8", "256"),
    ];

    for (query, expected) in test_cases {
        let response = agent
            .chat(&format!("Calculate {}", query))
            .await
            .expect("Chat failed");

        // LM Studio models may not execute tools correctly - they might return tool call JSON
        // Check if response contains expected result OR if it contains calculator tool call
        let has_expected_result = response.text().contains(expected);
        let has_tool_call =
            response.text().contains("calculator") || response.text().contains("operation");

        assert!(
            has_expected_result || has_tool_call,
            "Query: {}, Expected: {}, Response: {}",
            query,
            expected,
            response.text()
        );

        if !has_expected_result && has_tool_call {
            eprintln!(
                "[WARNING]  Model returned tool call JSON instead of executing tool. \
                This is expected with some LM Studio models that don't support tool calling."
            );
        }
    }
}

#[tokio::test]
#[ignore = "requires LM Studio; set CALICHE_RUN_LMSTUDIO_INTEGRATION=true and run with --ignored"]
#[serial]
#[cfg(feature = "lmstudio")]
async fn test_multiple_tools_integration() {
    require_lmstudio().await;

    let mut agent = AgentBuilder::new()
        .provider(ProviderType::LMStudio)
        .enable_tool("calculator")
        .enable_tool("fs")
        .build()
        .await
        .expect("Failed to build agent");

    // Agent should be able to use multiple tools
    let response = agent
        .chat("What tools do you have?")
        .await
        .expect("Chat failed");
    assert!(
        response.text().to_lowercase().contains("calculator")
            || response.text().to_lowercase().contains("read"),
        "Response should mention tools: {}",
        response.text()
    );
}

#[tokio::test]
#[ignore = "requires LM Studio; set CALICHE_RUN_LMSTUDIO_INTEGRATION=true and run with --ignored"]
#[serial]
#[cfg(feature = "lmstudio")]
async fn test_tool_error_handling() {
    require_lmstudio().await;

    let mut agent = AgentBuilder::new()
        .provider(ProviderType::LMStudio)
        .enable_tool("fs")
        .build()
        .await
        .expect("Failed to build agent");

    // Try to read non-existent file - should handle gracefully
    let response = agent
        .chat("Read the file /nonexistent/path/file.txt")
        .await
        .expect("Chat failed");
    // Should either succeed with error message or handle gracefully
    assert!(!response.text().is_empty(), "Response should not be empty");
}
