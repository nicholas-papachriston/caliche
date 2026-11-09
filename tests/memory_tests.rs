//! Tests for memory management workflows

use caliche::AgentBuilder;
use caliche::memory::BufferMemoryConfig;
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
async fn test_buffer_memory_workflow() {
    require_lmstudio().await;

    let mut agent = AgentBuilder::new()
        .provider(ProviderType::LMStudio)
        .model("local-model".to_string())
        .buffer_memory(BufferMemoryConfig {
            max_tokens: 1000,
            ..Default::default()
        })
        .build()
        .await
        .expect("Failed to build agent");

    // Multiple turns should maintain context
    agent
        .chat("Remember: I like pizza")
        .await
        .expect("Chat failed");
    let response = agent.chat("What do I like?").await.expect("Chat failed");
    assert!(response.text().to_lowercase().contains("pizza"));
}

#[tokio::test]
#[ignore = "requires LM Studio; set CALICHE_RUN_LMSTUDIO_INTEGRATION=true and run with --ignored"]
#[serial]
#[cfg(feature = "lmstudio")]
async fn test_memory_context_retention() {
    require_lmstudio().await;

    let mut agent = AgentBuilder::new()
        .provider(ProviderType::LMStudio)
        .model("local-model".to_string())
        .buffer_memory(BufferMemoryConfig {
            max_tokens: 2000,
            ..Default::default()
        })
        .build()
        .await
        .expect("Failed to build agent");

    // Establish context
    agent
        .chat("My favorite color is blue")
        .await
        .expect("Chat failed");
    agent
        .chat("I work as a software engineer")
        .await
        .expect("Chat failed");

    // Verify context is retained
    let response = agent
        .chat("What's my favorite color and job?")
        .await
        .expect("Chat failed");
    assert!(response.text().to_lowercase().contains("blue"));
    assert!(
        response.text().to_lowercase().contains("software")
            || response.text().to_lowercase().contains("engineer")
    );
}
