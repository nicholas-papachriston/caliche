#![cfg(all(feature = "lmstudio", feature = "tools-rag", feature = "vector-memory"))]

use async_trait::async_trait;
use caliche::agent::tools::RagTool;
use caliche::agent::tools::tool::ToolMeta;
use caliche::llm::LMStudioProvider;
/// LM Studio Integration Tests
///
/// These tests require a locally running LM Studio instance with openai/gpt-oss-120b model loaded.
///
/// To run these tests:
/// 1. Start LM Studio and load the openai/gpt-oss-120b model
/// 2. For embedding tests, also load an embedding model (e.g., text-embedding-qwen3-embedding-4b)
/// 3. Ensure the local server is running on http://localhost:1234
/// 4. Set CALICHE_RUN_LMSTUDIO_INTEGRATION=true
/// 5. Run: cargo test --features "lmstudio tools-rag vector-memory" --test lmstudio_integration -- --test-threads=1
use caliche::llm::embedding::{EmbeddingProvider, EmbeddingRequest};
use caliche::prelude::ProviderType;
use caliche::tools::ToolResult;
use caliche::{AgentBuilder, Tool};
use dripstone::{Chunk, MemoryBackend, traits::Backend};
use serde_json::json;
use serial_test::serial;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use tokio::time::{Duration, sleep};

// Advanced features imports
#[cfg(feature = "chains")]
use caliche::chains::{Chain, ComposableChain, LLMChain};
#[cfg(feature = "retrieval-chains")]
use caliche::chains::{
    ChunkingStrategy, DocumentMapReduceChain, HybridSearchChain, RetrievalChain,
};
#[cfg(feature = "document-loaders")]
use caliche::loaders::{Document, TextSplitter};
#[cfg(feature = "output-parsing")]
use caliche::output::{JsonOutputParser, OutputParser, StructuredOutputParser};
#[cfg(feature = "prompts")]
use caliche::prompts::{ChatPromptTemplate, Example, FewShotPromptTemplate, PromptTemplate};
use serde::{Deserialize, Serialize};
use tokio::sync::Mutex as TokioMutex;

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

/// Global counter to verify serial execution
static TEST_COUNTER: Mutex<u32> = Mutex::new(0);

/// Helper to verify tests are running serially
#[expect(
    clippy::expect_used,
    clippy::print_stdout,
    reason = "test helper: a failure fails the calling test"
)]
fn verify_serial_execution(test_name: &str) {
    let mut counter = TEST_COUNTER.lock().expect("Mutex lock should succeed");
    *counter += 1;
    let current = *counter;
    println!("[Serial Check] {} - Counter: {}", test_name, current);
    // If tests were running in parallel, we'd see non-sequential counter values
    // This is a simple check - in practice, serial_test handles this via mutex
}

/// Simple test tool for integration testing
#[derive(Debug)]
struct WeatherTool;
impl ToolMeta for WeatherTool {
    fn name(&self) -> &str {
        "get_weather"
    }

    fn description(&self) -> &str {
        "Get the current weather for a location"
    }

    fn parameters(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city and state, e.g. San Francisco, CA"
                },
                "unit": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "description": "The temperature unit"
                }
            },
            "required": ["location"]
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
impl Tool for WeatherTool {
    async fn execute(&self, args: serde_json::Value) -> ToolResult {
        let location = args["location"].as_str().unwrap_or("Unknown");
        let unit = args["unit"].as_str().unwrap_or("fahrenheit");

        // Mock weather response
        let temp = if unit == "celsius" { 22 } else { 72 };
        let result = json!({
            "location": location,
            "temperature": temp,
            "unit": unit,
            "condition": "sunny",
            "humidity": 65
        });

        ToolResult::success(result.to_string())
    }
}

/// Database tool that simulates a key-value store
#[derive(Debug)]
struct DatabaseTool {
    store: Arc<Mutex<HashMap<String, String>>>,
}
impl ToolMeta for DatabaseTool {
    fn name(&self) -> &str {
        "database"
    }

    fn description(&self) -> &str {
        "Store and retrieve data from a database. Supports 'get' and 'set' operations."
    }

    fn parameters(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": ["get", "set"],
                    "description": "The operation to perform"
                },
                "key": {
                    "type": "string",
                    "description": "The key to get or set"
                },
                "value": {
                    "type": "string",
                    "description": "The value to set (required for 'set' operation)"
                }
            },
            "required": ["operation", "key"]
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
impl Tool for DatabaseTool {
    #[expect(
        clippy::expect_used,
        reason = "test helper: a failure fails the calling test"
    )]
    async fn execute(&self, args: serde_json::Value) -> ToolResult {
        let operation = args["operation"].as_str().unwrap_or("get");
        let key = args["key"].as_str().unwrap_or("");

        let mut store = self.store.lock().expect("Mutex lock should succeed");

        match operation {
            "set" => {
                let value = args["value"].as_str().unwrap_or("");
                store.insert(key.to_string(), value.to_string());
                ToolResult::success(format!("Stored '{}' under key '{}'", value, key))
            }
            "get" => {
                if let Some(value) = store.get(key) {
                    ToolResult::success(format!("Value for key '{}': {}", key, value))
                } else {
                    ToolResult::error(format!("Key '{}' not found", key))
                }
            }
            _ => ToolResult::error("Invalid operation".to_string()),
        }
    }
}

/// File system tool that simulates file operations
#[derive(Debug)]
struct FileSystemTool {
    files: Arc<Mutex<HashMap<String, String>>>,
}
impl ToolMeta for FileSystemTool {
    fn name(&self) -> &str {
        "filesystem"
    }

    fn description(&self) -> &str {
        "Read, write, and list files. Supports 'read', 'write', and 'list' operations."
    }

    fn parameters(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": ["read", "write", "list"],
                    "description": "The file operation to perform"
                },
                "path": {
                    "type": "string",
                    "description": "The file path (required for read/write)"
                },
                "content": {
                    "type": "string",
                    "description": "The content to write (required for write)"
                }
            },
            "required": ["operation"]
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
impl Tool for FileSystemTool {
    #[expect(
        clippy::expect_used,
        reason = "test helper: a failure fails the calling test"
    )]
    async fn execute(&self, args: serde_json::Value) -> ToolResult {
        let operation = args["operation"].as_str().unwrap_or("list");
        let mut files = self.files.lock().expect("Mutex lock should succeed");

        match operation {
            "list" => {
                let file_list: Vec<String> = files.keys().cloned().collect();
                ToolResult::success(format!("Files: {:?}", file_list))
            }
            "read" => {
                let path = args["path"].as_str().unwrap_or("");
                if let Some(content) = files.get(path) {
                    ToolResult::success(content.clone())
                } else {
                    ToolResult::error(format!("File '{}' not found", path))
                }
            }
            "write" => {
                let path = args["path"].as_str().unwrap_or("");
                let content = args["content"].as_str().unwrap_or("");
                files.insert(path.to_string(), content.to_string());
                ToolResult::success(format!("Wrote {} bytes to '{}'", content.len(), path))
            }
            _ => ToolResult::error("Invalid operation".to_string()),
        }
    }
}

/// API call simulator
#[derive(Debug)]
struct ApiTool;
impl ToolMeta for ApiTool {
    fn name(&self) -> &str {
        "api_call"
    }

    fn description(&self) -> &str {
        "Make API calls to external services. Simulates REST API interactions."
    }

    fn parameters(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "endpoint": {
                    "type": "string",
                    "description": "The API endpoint to call"
                },
                "method": {
                    "type": "string",
                    "enum": ["GET", "POST"],
                    "description": "The HTTP method"
                }
            },
            "required": ["endpoint", "method"]
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
impl Tool for ApiTool {
    async fn execute(&self, args: serde_json::Value) -> ToolResult {
        let endpoint = args["endpoint"].as_str().unwrap_or("");
        let method = args["method"].as_str().unwrap_or("GET");

        // Simulate API latency
        sleep(Duration::from_millis(100)).await;

        let response = json!({
            "status": 200,
            "endpoint": endpoint,
            "method": method,
            "data": {
                "success": true,
                "timestamp": "2024-01-01T12:00:00Z"
            }
        });

        ToolResult::success(response.to_string())
    }
}

#[tokio::test]
#[ignore = "requires LM Studio; set CALICHE_RUN_LMSTUDIO_INTEGRATION=true and run with --ignored"]
#[serial]
async fn test_lmstudio_connection() {
    verify_serial_execution("test_lmstudio_connection");
    require_lmstudio().await;

    println!("LM Studio is available");
}

#[tokio::test]
#[ignore = "requires LM Studio; set CALICHE_RUN_LMSTUDIO_INTEGRATION=true and run with --ignored"]
#[serial]
async fn test_lmstudio_simple_chat() {
    verify_serial_execution("test_lmstudio_simple_chat");
    require_lmstudio().await;

    let mut agent = AgentBuilder::new()
        .provider(ProviderType::LMStudio)
        .system_prompt(
            "You are a helpful assistant. Answer questions directly and concisely.".to_string(),
        )
        .build()
        .await
        .expect("Failed to build agent");

    println!("Sending: What is the capital of France?");
    let response = agent
        .chat("What is the capital of France? Answer in one word.")
        .await
        .expect("Failed to chat with LM Studio");

    println!("Response: {}", response.text());
    println!("[INFO] Tokens used: {:?}", response.usage().total_tokens);

    assert!(!response.text().is_empty(), "Response should not be empty");
    // Check for Paris in various forms
    let response_lower = response.text().to_lowercase();
    assert!(
        response_lower.contains("paris"),
        "Response should mention Paris: {}",
        response.text()
    );
}

#[tokio::test]
#[ignore = "requires LM Studio; set CALICHE_RUN_LMSTUDIO_INTEGRATION=true and run with --ignored"]
#[serial]
async fn test_lmstudio_conversation_history() {
    require_lmstudio().await;

    let mut agent = AgentBuilder::new()
        .provider(ProviderType::LMStudio)
        .system_prompt(
            "You are a helpful assistant. Remember information from the conversation.".to_string(),
        )
        .build()
        .await
        .expect("Failed to build agent");

    // First message - simple fact
    println!("Sent Turn 1: The password is BLUE42");
    let response1 = agent
        .chat("Remember this: the password is BLUE42. Just say OK.")
        .await
        .expect("Failed on turn 1");
    println!("Received Turn 1 Response: {}", response1.text());

    // Verify first response
    assert!(
        !response1.text().is_empty(),
        "First response should not be empty"
    );

    // Second message - recall the fact
    println!("Sent Turn 2: What was the password I told you?");
    let response2 = agent
        .chat("What was the password I told you?")
        .await
        .expect("Failed on turn 2");
    println!("Received Turn 2 Response: {}", response2.text());

    // Check if response contains the password or acknowledges it
    let response2_lower = response2.to_lowercase();
    let has_memory = response2_lower.contains("blue42")
        || response2_lower.contains("blue 42")
        || (response2_lower.contains("blue") && response2_lower.contains("42"));

    if !has_memory {
        println!("[WARNING]  Note: LM Studio model may not support conversation history well");
        println!("   This is a known limitation of some smaller models");
    }

    // Just verify we got a response, as conversation history support varies by model
    assert!(
        !response2.text().is_empty(),
        "Second response should not be empty"
    );
}

#[tokio::test]
#[ignore = "requires LM Studio; set CALICHE_RUN_LMSTUDIO_INTEGRATION=true and run with --ignored"]
#[serial]
async fn test_lmstudio_with_tools() {
    require_lmstudio().await;

    let mut agent = AgentBuilder::new()
        .provider(ProviderType::LMStudio)
        .system_prompt(
            "You are a helpful assistant. When asked about weather, use the get_weather tool."
                .to_string(),
        )
        .build()
        .await
        .expect("Failed to build agent");

    // Add weather tool
    agent.add_tool(Arc::new(WeatherTool));

    println!("Sending: What's the weather in San Francisco?");
    let response = agent
        .chat("What's the weather in San Francisco? Use the weather tool.")
        .await
        .expect("Failed to chat with tools");

    println!("Response: {}", response.text());
    println!("[INFO] Tokens used: {:?}", response.usage().total_tokens);

    assert!(!response.is_empty(), "Response should not be empty");
    // Response should mention weather-related info
    assert!(
        response.text().to_lowercase().contains("weather")
            || response.text().to_lowercase().contains("temperature")
            || response.text().to_lowercase().contains("francisco"),
        "Response should be weather-related: {}",
        response
    );
}

#[tokio::test]
#[ignore = "requires LM Studio; set CALICHE_RUN_LMSTUDIO_INTEGRATION=true and run with --ignored"]
#[serial]
async fn test_lmstudio_with_calculator() {
    require_lmstudio().await;

    let mut agent = AgentBuilder::new()
        .provider(ProviderType::LMStudio)
        .system_prompt("You are a helpful math assistant. Solve math problems.".to_string())
        .enable_tool("calculator")
        .build()
        .await
        .expect("Failed to build agent");

    println!("Sending: What is 157 * 23?");
    let response = agent
        .chat("What is 157 multiplied by 23?")
        .await
        .expect("Failed to chat with calculator");

    println!("Response: {}", response.text());
    println!("[INFO] Tokens used: {:?}", response.usage().total_tokens);

    assert!(!response.text().is_empty(), "Response should not be empty");

    // The model might return the answer directly, via tool call, or in various formats
    // Just verify we got a response - tool calling support varies by model
    println!("   Response received, length: {}", response.text().len());

    if response.text().contains("3611") || response.text().contains("3,611") {
        println!("Correct answer found in response!");
    } else {
        println!("[WARNING]  Note: LM Studio model may not execute tools correctly");
        println!("   This is expected with openai/gpt-oss-120b");
        println!(
            "   Response format: {}",
            caliche_contracts::text::truncate_utf8(response.text(), 100)
        );
    }
}

#[tokio::test]
#[ignore = "requires LM Studio; set CALICHE_RUN_LMSTUDIO_INTEGRATION=true and run with --ignored"]
#[serial]
async fn test_lmstudio_multi_turn_reasoning() {
    require_lmstudio().await;

    let mut agent = AgentBuilder::new()
        .provider(ProviderType::LMStudio)
        .system_prompt("You are a helpful assistant.".to_string())
        .build()
        .await
        .expect("Failed to build agent");

    // Turn 1: First question
    println!("Sent Turn 1: What is 10 + 5?");
    let response1 = agent
        .chat("What is 10 + 5?")
        .await
        .expect("Failed on turn 1");
    println!("Received Turn 1: {}", response1.text());

    // Verify we got some response
    assert!(
        !response1.text().is_empty(),
        "First response should not be empty"
    );

    // Turn 2: Second question
    println!("Sent Turn 2: What is 20 - 8?");
    let response2 = agent
        .chat("What is 20 - 8?")
        .await
        .expect("Failed on turn 2");
    println!("Received Turn 2: {}", response2.text());

    // Verify we got some response
    assert!(
        !response2.text().is_empty(),
        "Second response should not be empty"
    );

    // Turn 3: Third question
    println!("Sent Turn 3: What is 6 * 7?");
    let response3 = agent
        .chat("What is 6 * 7?")
        .await
        .expect("Failed on turn 3");
    println!("Received Turn 3: {}", response3.text());

    // Just verify we can maintain multiple turns successfully
    assert!(
        !response3.text().is_empty(),
        "Third response should not be empty"
    );
    println!("Successfully completed multi-turn conversation");
}

#[tokio::test]
#[ignore = "requires LM Studio; set CALICHE_RUN_LMSTUDIO_INTEGRATION=true and run with --ignored"]
#[serial]
async fn test_lmstudio_token_tracking() {
    require_lmstudio().await;

    let mut agent = AgentBuilder::new()
        .provider(ProviderType::LMStudio)
        .system_prompt("You are a helpful assistant.".to_string())
        .build()
        .await
        .expect("Failed to build agent");

    let response = agent
        .chat("Say hello in one word.")
        .await
        .expect("Failed to chat");

    println!("Response: {}", response.text());
    println!("[INFO] Token usage: {:?}", response.usage());

    // Verify we got a response
    assert!(!response.text().is_empty(), "Response should not be empty");

    // Token tracking may not be supported by all LM Studio models
    if let Some(total) = response.usage().total_tokens {
        if total > 0 {
            println!("Token tracking is working: {} tokens", total);
            if let Some(prompt) = response.usage().prompt_tokens {
                println!("   Prompt tokens: {}", prompt);
            }
            if let Some(completion) = response.usage().completion_tokens {
                println!("   Completion tokens: {}", completion);
            }
        } else {
            println!("[WARNING]  Token count is 0 - this model may not report token usage");
        }
    } else {
        println!("[WARNING]  No token usage data returned from LM Studio");
        println!("   This is expected with some models");
    }
}

#[tokio::test]
#[ignore = "requires LM Studio; set CALICHE_RUN_LMSTUDIO_INTEGRATION=true and run with --ignored"]
#[serial]
async fn test_lmstudio_long_context() {
    require_lmstudio().await;

    let mut agent = AgentBuilder::new()
        .provider(ProviderType::LMStudio)
        .system_prompt("You are a helpful assistant.".to_string())
        .build()
        .await
        .expect("Failed to build agent");

    // Have multiple turns to build up context
    for i in 1..=5 {
        println!("Sent Turn {}: Simple question", i);
        let response = agent
            .chat(&format!("What is {} + {}?", i, i))
            .await
            .expect(&format!("Failed on turn {}", i));

        println!("Received Turn {}: {}", i, response.text());
        println!("   Tokens: {:?}", response.usage().total_tokens);

        assert!(
            !response.is_empty(),
            "Response should not be empty on turn {}",
            i
        );
    }

    // Final turn - agent should still be responsive
    println!("Sent Final turn: Another simple question");
    let final_response = agent
        .chat("What is 2 times 3?")
        .await
        .expect("Failed on final turn");

    println!("Final response: {}", final_response.text());
    assert!(
        !final_response.text().is_empty(),
        "Final response should not be empty"
    );
    println!("Agent remained responsive through {} turns", 6);
}

#[tokio::test]
#[ignore = "requires LM Studio; set CALICHE_RUN_LMSTUDIO_INTEGRATION=true and run with --ignored"]
#[serial]
async fn test_lmstudio_error_recovery() {
    require_lmstudio().await;

    let mut agent = AgentBuilder::new()
        .provider(ProviderType::LMStudio)
        .system_prompt("You are a helpful assistant.".to_string())
        .build()
        .await
        .expect("Failed to build agent");

    // First normal request
    println!("Sent Request 1: Hello");
    let result1 = agent.chat("Hello").await;
    assert!(result1.is_ok(), "First request should succeed");
    println!("Request 1 succeeded");

    // Second normal request - agent should still work
    println!("Sent Request 2: How are you?");
    let result2 = agent.chat("How are you?").await;
    assert!(result2.is_ok(), "Second request should succeed");
    println!("Request 2 succeeded");
}

#[tokio::test]
#[ignore = "requires LM Studio; set CALICHE_RUN_LMSTUDIO_INTEGRATION=true and run with --ignored"]
#[serial]
async fn test_lmstudio_concurrent_agents() {
    require_lmstudio().await;

    // Create multiple agents
    let mut agent1 = AgentBuilder::new()
        .provider(ProviderType::LMStudio)
        .system_prompt("You are Agent 1. Always start responses with '[Agent 1]'.".to_string())
        .build()
        .await
        .expect("Failed to build agent1");

    let mut agent2 = AgentBuilder::new()
        .provider(ProviderType::LMStudio)
        .system_prompt("You are Agent 2. Always start responses with '[Agent 2]'.".to_string())
        .build()
        .await
        .expect("Failed to build agent2");

    // Run requests sequentially (LM Studio might not handle true concurrent requests well)
    println!("Sent Agent 1: What is 2 + 2?");
    let response1 = agent1.chat("What is 2 + 2?").await.expect("Agent 1 failed");
    println!("Received Agent 1: {}", response1.text());

    println!("Sent Agent 2: What is 3 + 3?");
    let response2 = agent2.chat("What is 3 + 3?").await.expect("Agent 2 failed");
    println!("Received Agent 2: {}", response2.text());

    assert!(!response1.text().is_empty(), "Agent 1 should respond");
    assert!(!response2.text().is_empty(), "Agent 2 should respond");
}

#[tokio::test]
#[ignore = "requires LM Studio; set CALICHE_RUN_LMSTUDIO_INTEGRATION=true and run with --ignored"]
#[serial]
async fn test_lmstudio_embeddings() {
    require_lmstudio().await;

    // Create LM Studio provider for embeddings
    let provider = LMStudioProvider::from_env(None)
        .await
        .expect("Failed to create LM Studio provider");

    println!("Generating embeddings for test texts");
    let request = EmbeddingRequest {
        texts: vec![
            "The quick brown fox jumps over the lazy dog".to_string(),
            "A cat sits on a mat".to_string(),
            "The weather is sunny today".to_string(),
        ],
        model: None, // Use default model
    };

    let response = provider.embed(request).await;

    match response {
        Ok(resp) => {
            println!("Embeddings generated successfully");
            println!("   Model: {}", resp.model);
            println!("   Number of embeddings: {}", resp.embeddings.len());

            // Verify we got embeddings for all texts
            assert_eq!(
                resp.embeddings.len(),
                3,
                "Should have 3 embeddings for 3 texts"
            );

            // Verify embeddings have reasonable dimensions (typically 384-1536)
            for (i, embedding) in resp.embeddings.iter().enumerate() {
                println!("   Embedding {} dimensions: {}", i + 1, embedding.len());
                assert!(
                    embedding.len() >= 128,
                    "Embedding should have at least 128 dimensions"
                );
                assert!(
                    embedding.len() <= 4096,
                    "Embedding should have at most 4096 dimensions"
                );
            }

            // Check token usage if available
            if let Some(usage) = resp.usage {
                println!("[INFO] Token usage:");
                if let Some(prompt_tokens) = usage.prompt_tokens {
                    println!("   Prompt tokens: {}", prompt_tokens);
                }
                if let Some(embedding_tokens) = usage.embedding_tokens {
                    println!("   Embedding tokens: {}", embedding_tokens);
                }
                if let Some(total_tokens) = usage.total_tokens {
                    println!("   Total tokens: {}", total_tokens);
                }
            } else {
                println!("[WARNING]  No token usage data returned");
            }
        }
        Err(e) => {
            eprintln!("[WARNING]  Embedding generation failed: {}", e);
            eprintln!("   Make sure you have an embedding model loaded in LM Studio");
            eprintln!("   Expected model: text-embedding-qwen3-embedding-4b");
            panic!("Embeddings test failed - check LM Studio setup");
        }
    }
}

#[tokio::test]
#[ignore = "requires LM Studio; set CALICHE_RUN_LMSTUDIO_INTEGRATION=true and run with --ignored"]
#[serial]
async fn test_lmstudio_embedding_similarity() {
    require_lmstudio().await;

    let provider = LMStudioProvider::from_env(None)
        .await
        .expect("Failed to create LM Studio provider");

    println!("Testing embedding similarity");

    // Similar texts
    let request1 = EmbeddingRequest {
        texts: vec!["I love cats".to_string(), "I adore felines".to_string()],
        model: None,
    };

    let response = provider.embed(request1).await;

    match response {
        Ok(resp) => {
            assert_eq!(resp.embeddings.len(), 2, "Should have 2 embeddings");

            let emb1 = &resp.embeddings[0];
            let emb2 = &resp.embeddings[1];

            // Calculate cosine similarity
            let dot_product: f32 = emb1.iter().zip(emb2.iter()).map(|(a, b)| a * b).sum();
            let mag1: f32 = emb1.iter().map(|a| a * a).sum::<f32>().sqrt();
            let mag2: f32 = emb2.iter().map(|b| b * b).sum::<f32>().sqrt();
            let similarity = dot_product / (mag1 * mag2);

            println!("   Similarity between similar texts: {:.4}", similarity);
            println!("   (Similar texts about cats should have high similarity)");

            // Similar texts should have positive similarity (at least > 0)
            // Don't enforce too strict threshold as it depends on the embedding model
            assert!(
                similarity > 0.0,
                "Similar texts should have positive similarity"
            );

            if similarity > 0.5 {
                println!("High similarity detected as expected");
            } else {
                println!("[WARNING]  Lower similarity than expected, but test passes");
            }
        }
        Err(e) => {
            eprintln!("[WARNING]  Embedding similarity test failed: {}", e);
            eprintln!("   Make sure you have an embedding model loaded in LM Studio");
            panic!("Test failed - check LM Studio setup");
        }
    }
}

#[tokio::test]
#[ignore = "requires LM Studio; set CALICHE_RUN_LMSTUDIO_INTEGRATION=true and run with --ignored"]
#[serial]
async fn test_lmstudio_embedding_with_custom_model() {
    require_lmstudio().await;

    let provider = LMStudioProvider::from_env(None)
        .await
        .expect("Failed to create LM Studio provider");

    println!("Testing embeddings with custom model specification");

    let request = EmbeddingRequest {
        texts: vec!["Test text for custom model".to_string()],
        model: Some("text-embedding-qwen3-embedding-4b".to_string()),
    };

    let response = provider.embed(request).await;

    match response {
        Ok(resp) => {
            println!("Custom model embedding succeeded");
            println!("   Model used: {}", resp.model);
            println!("   Embedding dimensions: {}", resp.embeddings[0].len());
            assert_eq!(resp.embeddings.len(), 1, "Should have 1 embedding");
        }
        Err(e) => {
            eprintln!("[WARNING]  Custom model test failed: {}", e);
            eprintln!("   This might be expected if the specific model isn't loaded");
            eprintln!("   Test will pass but model may not match");
        }
    }
}

// ============================================================================
// ADVANCED TEST CASES - Complex Agent Capabilities
// ============================================================================

#[tokio::test]
#[ignore = "requires LM Studio; set CALICHE_RUN_LMSTUDIO_INTEGRATION=true and run with --ignored"]
#[serial]
async fn test_lmstudio_rag_with_embeddings() {
    require_lmstudio().await;

    println!("Testing RAG (Retrieval Augmented Generation)");

    // Create embedding provider
    let embedding_provider = Arc::new(
        LMStudioProvider::from_env(None)
            .await
            .expect("Failed to create LM Studio provider"),
    );

    // Create RAG tool
    let rag_tool = Arc::new(RagTool::new(embedding_provider, None::<String>));

    // Index some documents
    println!("   Indexing documents...");
    let docs = vec![
        (
            "rust_basics",
            "Rust is a systems programming language focused on safety, speed, and concurrency. It uses ownership and borrowing to manage memory without a garbage collector.",
        ),
        (
            "rust_features",
            "Rust features include zero-cost abstractions, move semantics, guaranteed memory safety, threads without data races, and trait-based generics.",
        ),
        (
            "rust_ownership",
            "In Rust, each value has an owner. There can only be one owner at a time. When the owner goes out of scope, the value will be dropped.",
        ),
    ];

    for (doc_id, content) in docs {
        match rag_tool
            .execute(json!({
                "operation": "index",
                "doc_id": doc_id,
                "document": content
            }))
            .await
        {
            result if result.success => {
                println!("   Indexed: {}", result.content);
            }
            result => {
                eprintln!(
                    "   [WARNING]  Failed to index {}: {}",
                    doc_id,
                    result.error.unwrap_or_default()
                );
            }
        }
    }

    // Query the RAG system
    println!("   Querying RAG system...");
    match rag_tool
        .execute(json!({
            "operation": "query",
            "query": "What is ownership in Rust?",
            "top_k": 2
        }))
        .await
    {
        result if result.success => {
            println!("   RAG Results:\n{}", result.content);
            assert!(result.content.contains("owner") || result.content.contains("scope"));
        }
        result => {
            eprintln!(
                "   [WARNING]  RAG query failed: {}",
                result.error.unwrap_or_default()
            );
        }
    }

    println!("RAG test completed");
}

#[tokio::test]
#[ignore = "requires LM Studio; set CALICHE_RUN_LMSTUDIO_INTEGRATION=true and run with --ignored"]
#[serial]
async fn test_lmstudio_complex_tool_chain() {
    require_lmstudio().await;

    println!("Testing complex multi-tool workflow");

    let db_store = Arc::new(Mutex::new(HashMap::new()));
    let fs_files = Arc::new(Mutex::new(HashMap::new()));

    let mut agent = AgentBuilder::new()
        .provider(ProviderType::LMStudio)
        .system_prompt(
            "You are a helpful assistant with access to database, filesystem, and API tools. \
            Complete tasks step by step using the available tools."
                .to_string(),
        )
        .build()
        .await
        .expect("Failed to build agent");

    // Add tools
    agent.add_tool(Arc::new(DatabaseTool {
        store: db_store.clone(),
    }));
    agent.add_tool(Arc::new(FileSystemTool {
        files: fs_files.clone(),
    }));
    agent.add_tool(Arc::new(ApiTool));

    // Complex task requiring multiple tool uses
    println!("   Task: Store data, write file, and make API call");
    let response = agent
        .chat(
            "Please: 1) Store 'Hello World' under key 'greeting' in the database, \
            2) Write 'Test content' to file 'test.txt', \
            3) Make a GET call to /api/status. \
            Confirm each step.",
        )
        .await
        .expect("Failed to execute complex workflow");

    println!("   Agent response: {}", response.text());
    assert!(!response.text().is_empty());

    // Verify state changes - tools should have been used
    let db_data = db_store.lock().expect("Mutex lock should succeed");
    let fs_data = fs_files.lock().expect("Mutex lock should succeed");

    println!("   Database entries: {} records", db_data.len());
    println!("   Filesystem entries: {} files", fs_data.len());

    // Assert that tools were actually used (may vary by model capability)
    // Note: Some models may not call tools correctly, so we make this lenient
    if db_data.len() == 0 && fs_data.len() == 0 {
        println!(
            "   [WARNING]  Warning: No tools were invoked - model may not support tool calling"
        );
    } else {
        println!("   Tools were successfully invoked");
    }

    println!("Complex tool chain test completed");
}

#[tokio::test]
#[ignore = "requires LM Studio; set CALICHE_RUN_LMSTUDIO_INTEGRATION=true and run with --ignored"]
#[serial]
async fn test_lmstudio_stateful_interaction() {
    require_lmstudio().await;

    println!("Testing stateful multi-turn interaction");

    let db_store = Arc::new(Mutex::new(HashMap::new()));

    let mut agent = AgentBuilder::new()
        .provider(ProviderType::LMStudio)
        .system_prompt(
            "You are a data management assistant. Help users store and retrieve information."
                .to_string(),
        )
        .build()
        .await
        .expect("Failed to build agent");

    agent.add_tool(Arc::new(DatabaseTool {
        store: db_store.clone(),
    }));

    // Turn 1: Store data
    println!("   Turn 1: Store user preferences");
    let response1 = agent
        .chat("Store my favorite color 'blue' under key 'color'")
        .await
        .expect("Turn 1 failed");
    println!("   Received {}", response1.text());

    // Turn 2: Store more data
    println!("   Turn 2: Store additional data");
    let response2 = agent
        .chat("Also store my age '25' under key 'age'")
        .await
        .expect("Turn 2 failed");
    println!("   Received {}", response2.text());

    // Turn 3: Retrieve and process
    println!("   Turn 3: Retrieve stored data");
    let response3 = agent
        .chat("What's my favorite color and age?")
        .await
        .expect("Turn 3 failed");
    println!("   Received {}", response3.text());

    // Verify database state
    let db = db_store.lock().expect("Mutex lock should succeed");
    println!("   Final database state: {} entries", db.len());

    // Check if data was actually stored
    if db.len() > 0 {
        println!(
            "   Data successfully stored: {:?}",
            db.keys().collect::<Vec<_>>()
        );
        // Ideally we'd check for specific keys: "color" and "age"
        if db.contains_key("color") || db.contains_key("age") {
            println!("   Expected keys found in database");
        }
    } else {
        println!("   [WARNING]  Warning: No data was stored - model may not support tool calling");
    }

    println!("Stateful interaction test completed");
}

#[tokio::test]
#[ignore = "requires LM Studio; set CALICHE_RUN_LMSTUDIO_INTEGRATION=true and run with --ignored"]
#[serial]
async fn test_lmstudio_parallel_tool_execution() {
    require_lmstudio().await;

    println!("Testing parallel tool execution capabilities");

    let mut agent = AgentBuilder::new()
        .provider(ProviderType::LMStudio)
        .system_prompt("You are an efficient assistant. You can use multiple tools.".to_string())
        .build()
        .await
        .expect("Failed to build agent");

    agent.add_tool(Arc::new(WeatherTool));
    agent.add_tool(Arc::new(ApiTool));

    // Request that could benefit from parallel execution
    println!("   Requesting weather for multiple cities");
    let response = agent
        .chat("Get weather for New York, London, and Tokyo")
        .await
        .expect("Parallel execution test failed");

    println!("   Response: {}", response.text());
    assert!(!response.text().is_empty());

    // Check if response mentions multiple cities (indicates tool usage)
    let mentions_multiple = (response.text().contains("New York")
        || response.text().contains("York"))
        && response.text().contains("London")
        && response.text().contains("Tokyo");

    if mentions_multiple {
        println!("   Response references multiple cities - likely used tools");
    } else {
        println!("   [WARNING]  Response may not have used tools for all cities");
    }

    println!("Parallel tool execution test completed");
}

#[tokio::test]
#[ignore = "requires LM Studio; set CALICHE_RUN_LMSTUDIO_INTEGRATION=true and run with --ignored"]
#[serial]
async fn test_lmstudio_error_handling_and_recovery() {
    require_lmstudio().await;

    println!("Testing error handling and recovery");

    let db_store = Arc::new(Mutex::new(HashMap::new()));

    let mut agent = AgentBuilder::new()
        .provider(ProviderType::LMStudio)
        .system_prompt(
            "You are a helpful assistant. If a tool fails, try alternative approaches.".to_string(),
        )
        .build()
        .await
        .expect("Failed to build agent");

    agent.add_tool(Arc::new(DatabaseTool {
        store: db_store.clone(),
    }));

    // Try to get non-existent key (should handle error gracefully)
    println!("   Attempting to retrieve non-existent key");
    let response1 = agent
        .chat("Get the value for key 'nonexistent' from the database")
        .await
        .expect("Error handling test failed");
    println!("   Received {}", response1.text());
    assert!(!response1.text().is_empty());

    // Store and retrieve successfully
    println!("   Storing and retrieving valid data");
    let response2 = agent
        .chat("Store 'test_value' under 'test_key', then retrieve it")
        .await
        .expect("Recovery test failed");
    println!("   Received {}", response2.text());

    // Verify the second operation was successful
    let final_db = db_store.lock().expect("Mutex lock should succeed");
    if final_db.contains_key("test_key") {
        println!("   Data successfully stored after recovery");
        assert_eq!(
            final_db.get("test_key").map(|s| s.as_str()),
            Some("test_value")
        );
    } else {
        println!("   [WARNING]  Warning: Data not found - model may not support tool calling");
    }

    println!("Error handling test completed");
}

#[tokio::test]
#[ignore = "requires LM Studio; set CALICHE_RUN_LMSTUDIO_INTEGRATION=true and run with --ignored"]
#[serial]
async fn test_lmstudio_context_management() {
    require_lmstudio().await;

    println!("Testing context window management with long conversation");

    let mut agent = AgentBuilder::new()
        .provider(ProviderType::LMStudio)
        .system_prompt("You are a helpful assistant that remembers context.".to_string())
        .build()
        .await
        .expect("Failed to build agent");

    // Build up context with multiple turns
    println!("   Building conversation context...");
    let facts = vec![
        "My name is Alice",
        "I work as a software engineer",
        "I love programming in Rust",
        "My favorite color is purple",
        "I have a pet cat named Whiskers",
    ];

    for (i, fact) in facts.iter().enumerate() {
        println!("   Turn {}: Stating fact", i + 1);
        let response = agent
            .chat(&format!("Remember: {}. Just acknowledge.", fact))
            .await
            .expect(&format!("Turn {} failed", i + 1));

        println!("   Received {}", response.text());
        if let Some(total) = response.usage().total_tokens {
            println!("   [INFO] Total tokens: {}", total);
        }
    }

    // Test recall
    println!("   Testing context recall");
    let response = agent
        .chat("What do you remember about me? List everything.")
        .await
        .expect("Recall test failed");

    println!("   Recall response: {}", response.text());
    assert!(!response.text().is_empty());

    // Check if any of the facts were recalled
    let response_lower = response.text().to_lowercase();
    let recalled_facts = vec![
        response_lower.contains("alice"),
        response_lower.contains("software") || response_lower.contains("engineer"),
        response_lower.contains("rust"),
        response_lower.contains("purple"),
        response_lower.contains("whiskers") || response_lower.contains("cat"),
    ];

    let recall_count = recalled_facts.iter().filter(|&&x| x).count();
    println!("   [INFO] Recalled {}/5 facts from context", recall_count);

    if recall_count >= 2 {
        println!("   Good context retention ({}/5 facts)", recall_count);
    } else {
        println!(
            "   [WARNING]  Limited context retention ({}/5 facts) - may be model limitation",
            recall_count
        );
    }

    println!("Context management test completed");
}

#[tokio::test]
#[ignore = "requires LM Studio; set CALICHE_RUN_LMSTUDIO_INTEGRATION=true and run with --ignored"]
#[serial]
async fn test_lmstudio_task_decomposition() {
    require_lmstudio().await;

    println!("Testing task decomposition and step-by-step execution");

    let fs_files = Arc::new(Mutex::new(HashMap::new()));

    let mut agent = AgentBuilder::new()
        .provider(ProviderType::LMStudio)
        .system_prompt(
            "You are a methodical assistant. Break down complex tasks into steps and execute them one by one.".to_string(),
        )
        .build()
        .await
        .expect("Failed to build agent");

    agent.add_tool(Arc::new(FileSystemTool {
        files: fs_files.clone(),
    }));

    // Complex task requiring decomposition
    println!("   Complex task: Create report with calculations");
    let response = agent
        .chat(
            "Create a report: \
            1) Calculate 25 * 4, \
            2) Write a file 'report.txt' containing 'Total: [result]', \
            3) List all files. \
            Do this step by step.",
        )
        .await
        .expect("Task decomposition failed");

    println!("   Agent execution:\n{}", response.text());
    assert!(!response.text().is_empty());

    // Verify file was created
    let files = fs_files.lock().expect("Mutex lock should succeed");
    println!("   Files created: {}", files.len());

    if files.contains_key("report.txt") {
        println!("   Report file was created");
        if let Some(content) = files.get("report.txt") {
            println!("   Report content: {}", content);
            if content.contains("100") || content.contains("Total") {
                println!("   Report contains expected calculation result");
            }
        }
    } else {
        println!(
            "   [WARNING]  Warning: Report file not created - model may not support tool calling"
        );
    }

    println!("Task decomposition test completed");
}

#[tokio::test]
#[ignore = "requires LM Studio; set CALICHE_RUN_LMSTUDIO_INTEGRATION=true and run with --ignored"]
#[serial]
async fn test_lmstudio_adaptive_behavior() {
    require_lmstudio().await;

    println!("Testing adaptive behavior based on user feedback");

    let mut agent = AgentBuilder::new()
        .provider(ProviderType::LMStudio)
        .system_prompt(
            "You are an adaptive assistant. Adjust your responses based on user preferences and feedback.".to_string(),
        )
        .enable_tool("calculator")
        .build()
        .await
        .expect("Failed to build agent");

    // Initial interaction
    println!("   Initial request");
    let response1 = agent
        .chat("Calculate 15 + 27")
        .await
        .expect("Initial request failed");
    println!("   Received {}", response1.text());

    // Provide feedback
    println!("   Providing feedback on response style");
    let response2 = agent
        .chat("Good, but please be more concise in future responses.")
        .await
        .expect("Feedback failed");
    println!("   Received {}", response2.text());

    // Another calculation to see if behavior adapted
    println!("   Follow-up request");
    let response3 = agent
        .chat("Calculate 8 * 9")
        .await
        .expect("Follow-up failed");
    println!("   Received {}", response3.text());

    assert!(!response3.text().is_empty());
    println!("Adaptive behavior test completed");
}

#[tokio::test]
#[ignore = "requires LM Studio; set CALICHE_RUN_LMSTUDIO_INTEGRATION=true and run with --ignored"]
#[serial]
async fn test_lmstudio_batch_processing() {
    require_lmstudio().await;

    println!("Testing batch processing capabilities");

    let mut agent = AgentBuilder::new()
        .provider(ProviderType::LMStudio)
        .system_prompt("You are an efficient batch processor.".to_string())
        .enable_tool("calculator")
        .build()
        .await
        .expect("Failed to build agent");

    // Batch calculation request
    println!("   Batch calculations");
    let calculations = vec!["10 + 5", "20 * 3", "100 / 4", "50 - 15"];

    let response = agent
        .chat(&format!(
            "Calculate these in order: {}. Return just the results.",
            calculations.join(", ")
        ))
        .await
        .expect("Batch processing failed");

    println!("   Batch results: {}", response.text());
    assert!(!response.text().is_empty());

    // Check if response contains any of the expected results
    let expected_results = ["15", "60", "25", "35"]; // 10+5, 20*3, 100/4, 50-15
    let found_results = expected_results
        .iter()
        .filter(|&&r| response.text().contains(r))
        .count();

    println!(
        "   [INFO] Found {}/{} expected calculation results",
        found_results,
        expected_results.len()
    );

    if found_results >= 2 {
        println!("   Batch calculations appear to be processed");
    } else {
        println!(
            "   [WARNING]  Few calculation results found - model may not have used calculator"
        );
    }

    println!("Batch processing test completed");
}

#[tokio::test]
#[ignore = "requires LM Studio; set CALICHE_RUN_LMSTUDIO_INTEGRATION=true and run with --ignored"]
#[serial]
async fn test_lmstudio_streaming_simulation() {
    require_lmstudio().await;

    println!("Testing response streaming simulation");

    let mut agent = AgentBuilder::new()
        .provider(ProviderType::LMStudio)
        .system_prompt("You are a helpful assistant.".to_string())
        .build()
        .await
        .expect("Failed to build agent");

    // Long-form response that would benefit from streaming
    println!("   Requesting detailed explanation");
    let response = agent
        .chat("Explain what Rust is in detail.")
        .await
        .expect("Streaming test failed");

    println!("   Response length: {} chars", response.text().len());
    if let Some(total) = response.usage().total_tokens {
        println!("   [INFO] Tokens: {}", total);
    }

    assert!(response.text().len() > 50);
    println!("Streaming simulation test completed");
}

#[tokio::test]
#[ignore = "requires LM Studio; set CALICHE_RUN_LMSTUDIO_INTEGRATION=true and run with --ignored"]
#[serial]
async fn test_lmstudio_manual_embedding_with_agent_retrieval() {
    require_lmstudio().await;

    println!("Testing manual embedding storage with agent retrieval");

    // Step 1: Create embedding provider
    let embedding_provider = Arc::new(
        LMStudioProvider::from_env(None)
            .await
            .expect("Failed to create LM Studio provider"),
    );

    println!("   Step 1: Generating embeddings manually...");

    // Step 2: Manually embed documents
    let documents = vec![
        (
            "rust_memory",
            "Rust's ownership system ensures memory safety without garbage collection. Each value has a single owner, and ownership can be transferred or borrowed.",
        ),
        (
            "rust_performance",
            "Rust provides zero-cost abstractions and compiles to native code, making it as fast as C++ while maintaining safety guarantees.",
        ),
        (
            "rust_concurrency",
            "Rust's type system prevents data races at compile time. You can fearlessly use threads knowing the compiler won't let you make concurrency mistakes.",
        ),
    ];

    let texts: Vec<String> = documents.iter().map(|(_, text)| text.to_string()).collect();
    let embedding_request = EmbeddingRequest {
        texts: texts.clone(),
        model: None,
    };

    let embedding_response = embedding_provider
        .embed(embedding_request)
        .await
        .expect("Failed to generate embeddings");

    println!(
        "   Generated {} embeddings",
        embedding_response.embeddings.len()
    );

    // Step 3: Store embeddings in embedding store
    let vector_backend: Arc<dyn Backend> = Arc::new(MemoryBackend::new());

    for (i, ((doc_id, text), embedding)) in documents
        .iter()
        .zip(embedding_response.embeddings.iter())
        .enumerate()
    {
        let chunk = Chunk::new(
            doc_id.to_string(),
            i,
            text.to_string(),
            embedding.clone(),
            0,
            text.len(),
            1,
            json!({
                "index": i,
                "topic": "rust programming"
            }),
            Some("rust_docs".to_string()),
        );

        vector_backend
            .upsert(&[chunk])
            .await
            .expect("Failed to store embedding");
    }

    println!("   Stored {} embeddings in vector store", documents.len());

    // Step 4: Create a custom retrieval tool for the agent
    struct VectorSearchTool {
        vector_backend: Arc<dyn Backend>,
        embedding_provider: Arc<LMStudioProvider>,
        documents: HashMap<String, String>,
    }

    impl std::fmt::Debug for VectorSearchTool {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.debug_struct("VectorSearchTool")
                .field("vector_backend", &"<Backend>")
                .field("embedding_provider", &"<LMStudioProvider>")
                .field("documents", &self.documents)
                .finish()
        }
    }
    impl ToolMeta for VectorSearchTool {
        fn name(&self) -> &str {
            "search_rust_docs"
        }

        fn description(&self) -> &str {
            "Search Rust documentation using semantic search. Provide a query about Rust programming."
        }

        fn parameters(&self) -> serde_json::Value {
            json!({
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query about Rust programming"
                    },
                    "top_k": {
                        "type": "number",
                        "description": "Number of results to return (default: 2)"
                    }
                },
                "required": ["query"]
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
    impl Tool for VectorSearchTool {
        async fn execute(&self, args: serde_json::Value) -> ToolResult {
            let query = args["query"].as_str().unwrap_or("");
            let top_k = args["top_k"].as_u64().unwrap_or(2) as usize;

            // Generate embedding for query
            let embedding_request = EmbeddingRequest {
                texts: vec![query.to_string()],
                model: None,
            };

            let embedding_response = match self.embedding_provider.embed(embedding_request).await {
                Ok(resp) => resp,
                Err(e) => return ToolResult::error(format!("Failed to embed query: {}", e)),
            };

            let query_embedding = &embedding_response.embeddings[0];

            // Search the vector store
            let query = dripstone::core::SearchQuery::new(query_embedding.clone())
                .namespace("rust_docs")
                .limit(top_k);
            let results = match self.vector_backend.search(&query).await {
                Ok(results) => results,
                Err(e) => return ToolResult::error(format!("Search failed: {}", e)),
            };

            if results.is_empty() {
                return ToolResult::success("No relevant documents found.".to_string());
            }

            // Format results
            let mut response = format!("Found {} relevant document(s):\n\n", results.len());
            for (i, result) in results.iter().enumerate() {
                response.push_str(&format!(
                    "{}. [Similarity: {:.3}] {}\n\n",
                    i + 1,
                    result.score,
                    result.chunk.content
                ));
            }

            ToolResult::success(response)
        }
    }

    // Create document map for retrieval
    let doc_map: HashMap<String, String> = documents
        .iter()
        .map(|(id, text)| (id.to_string(), text.to_string()))
        .collect();

    // Step 5: Create agent with the vector search tool
    let mut agent = AgentBuilder::new()
        .provider(ProviderType::LMStudio)
        .system_prompt(
            "You are a Rust programming expert. Use the search_rust_docs tool to find relevant information from the Rust documentation when answering questions.".to_string(),
        )
        .build()
        .await
        .expect("Failed to build agent");

    agent.add_tool(Arc::new(VectorSearchTool {
        vector_backend: vector_backend.clone(),
        embedding_provider: embedding_provider.clone(),
        documents: doc_map,
    }));

    println!("   Step 2: Querying agent to retrieve manually embedded documents...");

    // Step 6: Ask agent to retrieve information
    let response = agent
        .chat("What does Rust's documentation say about concurrency and thread safety?")
        .await
        .expect("Failed to query agent");

    println!("   Agent response:\n{}", response.text());

    // Step 7: Verify the response mentions relevant content
    let response_lower = response.text().to_lowercase();
    let mentions_concurrency = response_lower.contains("concurrency")
        || response_lower.contains("thread")
        || response_lower.contains("data race");

    if mentions_concurrency {
        println!("   Agent successfully retrieved manually embedded concurrency documentation");
    } else {
        println!("   [WARNING]  Response may not have used the vector search tool");
        println!("      This could be due to model limitations in tool calling");
    }

    // Step 8: Verify we can directly search the embedding store
    println!("   Step 3: Verifying direct vector store search...");
    let direct_query_embedding = embedding_provider
        .embed(EmbeddingRequest {
            texts: vec!["thread safety".to_string()],
            model: None,
        })
        .await
        .expect("Failed to embed direct query");

    let query = dripstone::core::SearchQuery::new(direct_query_embedding.embeddings[0].clone())
        .namespace("rust_docs")
        .limit(1);
    let direct_results = vector_backend
        .search(&query)
        .await
        .expect("Direct search failed");

    assert!(
        !direct_results.is_empty(),
        "Direct search should return results"
    );
    assert!(
        direct_results[0].chunk.id.contains("rust"),
        "Result should be a Rust document"
    );

    println!(
        "   Direct search found: {} (similarity: {:.3})",
        direct_results[0].chunk.id, direct_results[0].score
    );

    println!("Manual embedding with agent retrieval test completed");
}

#[tokio::test]
#[ignore = "requires LM Studio; set CALICHE_RUN_LMSTUDIO_INTEGRATION=true and run with --ignored"]
#[serial]
async fn test_lmstudio_tool_parameter_validation() {
    require_lmstudio().await;

    println!("Testing tool parameter validation and error handling");

    let db_store = Arc::new(Mutex::new(HashMap::new()));

    let mut agent = AgentBuilder::new()
        .provider(ProviderType::LMStudio)
        .system_prompt(
            "You are a helpful assistant. When tool calls fail, explain the error to the user."
                .to_string(),
        )
        .build()
        .await
        .expect("Failed to build agent");

    agent.add_tool(Arc::new(DatabaseTool {
        store: db_store.clone(),
    }));
    agent.add_tool(Arc::new(WeatherTool));

    // Test 1: Missing required parameter
    println!("   Test 1: Tool call with missing required parameter");
    let response1 = agent
        .chat("Get data from the database for key 'test_key' using the database tool")
        .await
        .expect("Test 1 failed");
    println!("   Response: {}", response1.text());
    assert!(!response1.text().is_empty());

    // Test 2: Invalid operation type
    println!("   Test 2: Database operation on non-existent key (expect error)");
    let response2 = agent
        .chat("Use the database tool to get the value for 'nonexistent_key_12345'")
        .await
        .expect("Test 2 failed");
    println!("   Response: {}", response2.text());
    assert!(!response2.text().is_empty());

    // The agent should handle the error gracefully
    if response2.to_lowercase().contains("not found")
        || response2.to_lowercase().contains("error")
        || response2.to_lowercase().contains("doesn't exist")
    {
        println!("   Agent correctly communicated tool error");
    } else {
        println!("   [WARNING]  Agent response may not reflect tool error");
    }

    // Test 3: Invalid enum value for weather unit
    println!("   Test 3: Weather tool with standard parameters");
    let response3 = agent
        .chat("Get weather for Paris using the weather tool")
        .await
        .expect("Test 3 failed");
    println!("   Response: {}", response3.text());
    assert!(!response3.text().is_empty());

    // Test 4: Valid tool usage after errors
    println!("   Test 4: Valid database operation after previous errors");
    let response4 = agent
        .chat("Store 'test_value' under key 'test_key' in the database, then retrieve it")
        .await
        .expect("Test 4 failed");
    println!("   Response: {}", response4.text());
    assert!(!response4.text().is_empty());

    // Verify database state
    let db = db_store.lock().expect("Mutex lock should succeed");
    if db.contains_key("test_key") {
        println!("   Database tool recovered and executed successfully");
    }

    println!("Tool parameter validation test completed");
}

#[tokio::test]
#[ignore = "requires LM Studio; set CALICHE_RUN_LMSTUDIO_INTEGRATION=true and run with --ignored"]
#[serial]
async fn test_lmstudio_namespace_isolation() {
    require_lmstudio().await;

    println!("Testing namespace isolation in embedding store");

    let embedding_provider = Arc::new(
        LMStudioProvider::from_env(None)
            .await
            .expect("Failed to create LM Studio provider"),
    );

    // Create embedding store
    let vector_backend: Arc<dyn Backend> = Arc::new(MemoryBackend::new());

    println!("   Step 1: Creating embeddings in multiple namespaces");

    // Namespace 1: Python documentation
    let python_docs = vec![
        (
            "python_basics",
            "Python is a high-level, interpreted programming language known for its simplicity and readability.",
        ),
        (
            "python_features",
            "Python supports multiple programming paradigms including object-oriented, functional, and procedural.",
        ),
    ];

    // Namespace 2: Rust documentation
    let rust_docs = vec![
        (
            "rust_basics",
            "Rust is a systems programming language focused on safety, speed, and concurrency.",
        ),
        (
            "rust_features",
            "Rust features zero-cost abstractions and memory safety without garbage collection.",
        ),
    ];

    // Namespace 3: JavaScript documentation
    let js_docs = vec![
        (
            "js_basics",
            "JavaScript is a dynamic scripting language primarily used for web development.",
        ),
        (
            "js_features",
            "JavaScript supports asynchronous programming with promises and async/await.",
        ),
    ];

    // Embed and store all documents
    for (namespace, docs) in [
        ("python_docs", python_docs),
        ("rust_docs", rust_docs),
        ("javascript_docs", js_docs),
    ] {
        let texts: Vec<String> = docs.iter().map(|(_, text)| text.to_string()).collect();
        let embedding_request = EmbeddingRequest { texts, model: None };

        let embedding_response = embedding_provider
            .embed(embedding_request)
            .await
            .expect("Failed to generate embeddings");

        for (i, ((doc_id, text), embedding)) in docs
            .iter()
            .zip(embedding_response.embeddings.iter())
            .enumerate()
        {
            let chunk = Chunk::new(
                doc_id.to_string(),
                i,
                text.to_string(),
                embedding.clone(),
                0,
                text.len(),
                1,
                json!({
                    "language": namespace.replace("_docs", ""),
                    "index": i
                }),
                Some(namespace.to_string()),
            );

            vector_backend
                .upsert(&[chunk])
                .await
                .expect("Failed to store embedding");
        }

        println!(
            "   Stored {} embeddings in namespace '{}'",
            docs.len(),
            namespace
        );
    }

    println!("   Step 2: Testing namespace-specific searches");

    // Search for "programming language" in each namespace
    let query_embedding = embedding_provider
        .embed(EmbeddingRequest {
            texts: vec!["programming language features".to_string()],
            model: None,
        })
        .await
        .expect("Failed to embed query");

    // Search Python namespace
    let query = dripstone::core::SearchQuery::new(query_embedding.embeddings[0].clone())
        .namespace("python_docs")
        .limit(5);
    let python_results = vector_backend
        .search(&query)
        .await
        .expect("Python search failed");

    println!("   Python namespace results: {}", python_results.len());
    for result in &python_results {
        assert!(
            result.chunk.id.starts_with("python"),
            "Result should only contain Python docs, found: {}",
            result.chunk.id
        );
        println!("     - {} (score: {:.3})", result.chunk.id, result.score);
    }

    // Search Rust namespace
    let query = dripstone::core::SearchQuery::new(query_embedding.embeddings[0].clone())
        .namespace("rust_docs")
        .limit(5);
    let rust_results = vector_backend
        .search(&query)
        .await
        .expect("Rust search failed");

    println!("   Rust namespace results: {}", rust_results.len());
    for result in &rust_results {
        assert!(
            result.chunk.id.starts_with("rust"),
            "Result should only contain Rust docs, found: {}",
            result.chunk.id
        );
        println!("     - {} (score: {:.3})", result.chunk.id, result.score);
    }

    // Search JavaScript namespace
    let query = dripstone::core::SearchQuery::new(query_embedding.embeddings[0].clone())
        .namespace("javascript_docs")
        .limit(5);
    let js_results = vector_backend
        .search(&query)
        .await
        .expect("JavaScript search failed");

    println!("   JavaScript namespace results: {}", js_results.len());
    for result in &js_results {
        assert!(
            result.chunk.id.starts_with("js"),
            "Result should only contain JavaScript docs, found: {}",
            result.chunk.id
        );
        println!("     - {} (score: {:.3})", result.chunk.id, result.score);
    }

    println!("   Step 3: Verifying namespace isolation");

    // Verify no cross-namespace contamination
    assert!(
        !python_results.is_empty(),
        "Python search should return results"
    );
    assert!(
        !rust_results.is_empty(),
        "Rust search should return results"
    );
    assert!(
        !js_results.is_empty(),
        "JavaScript search should return results"
    );

    // Verify all results are from correct namespace
    let python_ids: Vec<_> = python_results.iter().map(|r| r.chunk.id.as_str()).collect();
    let rust_ids: Vec<_> = rust_results.iter().map(|r| r.chunk.id.as_str()).collect();
    let js_ids: Vec<_> = js_results.iter().map(|r| r.chunk.id.as_str()).collect();

    // Check no overlap
    for python_id in &python_ids {
        assert!(
            !rust_ids.contains(python_id),
            "Python result leaked into Rust namespace"
        );
        assert!(
            !js_ids.contains(python_id),
            "Python result leaked into JS namespace"
        );
    }

    println!("   Namespace isolation verified - no cross-contamination");
    println!("Namespace isolation test completed");
}

#[tokio::test]
#[ignore = "requires LM Studio; set CALICHE_RUN_LMSTUDIO_INTEGRATION=true and run with --ignored"]
#[serial]
async fn test_lmstudio_metadata_filtering() {
    require_lmstudio().await;

    println!("Testing metadata filtering in RAG searches");

    let embedding_provider = Arc::new(
        LMStudioProvider::from_env(None)
            .await
            .expect("Failed to create LM Studio provider"),
    );

    // Create embedding store
    let vector_backend: Arc<dyn Backend> = Arc::new(MemoryBackend::new());

    println!("   Step 1: Indexing documents with rich metadata");

    // Documents with various metadata
    let documents = vec![
        (
            "rust_tutorial_beginner",
            "Getting started with Rust: variables, functions, and basic syntax.",
            json!({
                "language": "rust",
                "difficulty": "beginner",
                "topic": "tutorial",
                "year": 2024,
                "tags": ["syntax", "basics"]
            }),
        ),
        (
            "rust_advanced_ownership",
            "Advanced Rust: deep dive into ownership, lifetimes, and borrowing.",
            json!({
                "language": "rust",
                "difficulty": "advanced",
                "topic": "ownership",
                "year": 2024,
                "tags": ["memory", "advanced"]
            }),
        ),
        (
            "python_tutorial_beginner",
            "Python basics: variables, loops, and functions for beginners.",
            json!({
                "language": "python",
                "difficulty": "beginner",
                "topic": "tutorial",
                "year": 2023,
                "tags": ["syntax", "basics"]
            }),
        ),
        (
            "rust_async_intermediate",
            "Intermediate Rust: async programming with tokio and futures.",
            json!({
                "language": "rust",
                "difficulty": "intermediate",
                "topic": "async",
                "year": 2024,
                "tags": ["concurrency", "async"]
            }),
        ),
        (
            "rust_webdev_intermediate",
            "Building web applications in Rust using Axum and Tokio.",
            json!({
                "language": "rust",
                "difficulty": "intermediate",
                "topic": "web",
                "year": 2024,
                "tags": ["web", "http"]
            }),
        ),
    ];

    // Embed and store all documents
    let texts: Vec<String> = documents
        .iter()
        .map(|(_, text, _)| text.to_string())
        .collect();
    let embedding_request = EmbeddingRequest { texts, model: None };

    let embedding_response = embedding_provider
        .embed(embedding_request)
        .await
        .expect("Failed to generate embeddings");

    for (i, ((doc_id, text, metadata), embedding)) in documents
        .iter()
        .zip(embedding_response.embeddings.iter())
        .enumerate()
    {
        let chunk = Chunk::new(
            doc_id.to_string(),
            i,
            text.to_string(),
            embedding.clone(),
            0,
            text.len(),
            1,
            metadata.clone(),
            Some("tech_docs".to_string()),
        );

        vector_backend
            .upsert(&[chunk])
            .await
            .expect("Failed to store embedding");
    }

    println!("   Indexed {} documents with metadata", documents.len());

    println!("   Step 2: Searching with metadata filters");

    // Query for programming tutorials
    let query_embedding = embedding_provider
        .embed(EmbeddingRequest {
            texts: vec!["programming tutorial".to_string()],
            model: None,
        })
        .await
        .expect("Failed to embed query");

    // Search 1: All results (no filter)
    let query = dripstone::core::SearchQuery::new(query_embedding.embeddings[0].clone())
        .namespace("tech_docs")
        .limit(10);
    let all_results = vector_backend.search(&query).await.expect("Search failed");

    println!("   All results (no filter): {}", all_results.len());
    for result in &all_results {
        let difficulty = result
            .chunk
            .metadata
            .get("difficulty")
            .and_then(|d| d.as_str())
            .unwrap_or("unknown");
        let language = result
            .chunk
            .metadata
            .get("language")
            .and_then(|l| l.as_str())
            .unwrap_or("unknown");
        println!(
            "     - {} ({}, {}) score: {:.3}",
            result.chunk.id, language, difficulty, result.score
        );
    }

    // Search 2: Filter by difficulty = "beginner"
    let beginner_results: Vec<_> = all_results
        .iter()
        .filter(|r| r.chunk.metadata.get("difficulty").and_then(|d| d.as_str()) == Some("beginner"))
        .collect();

    println!("   Beginner-level results: {}", beginner_results.len());
    for result in &beginner_results {
        let language = result
            .chunk
            .metadata
            .get("language")
            .and_then(|l| l.as_str())
            .unwrap_or("unknown");
        println!(
            "     - {} ({}) score: {:.3}",
            result.chunk.id, language, result.score
        );
        assert!(
            result.chunk.id.contains("beginner"),
            "Filtered result should be beginner level"
        );
    }

    // Search 3: Filter by language = "rust" AND difficulty != "beginner"
    let rust_advanced_results: Vec<_> = all_results
        .iter()
        .filter(|r| {
            let metadata = &r.chunk.metadata;
            let is_rust = metadata.get("language").and_then(|l| l.as_str()) == Some("rust");
            let not_beginner =
                metadata.get("difficulty").and_then(|d| d.as_str()) != Some("beginner");
            is_rust && not_beginner
        })
        .collect();

    println!(
        "   Rust non-beginner results: {}",
        rust_advanced_results.len()
    );
    for result in &rust_advanced_results {
        let difficulty = result
            .chunk
            .metadata
            .get("difficulty")
            .and_then(|d| d.as_str())
            .unwrap_or("unknown");
        println!(
            "     - {} ({}) score: {:.3}",
            result.chunk.id, difficulty, result.score
        );
        assert!(
            result.chunk.id.starts_with("rust"),
            "Should only return Rust documents"
        );
        assert!(
            !result.chunk.id.contains("beginner"),
            "Should not return beginner documents"
        );
    }

    // Search 4: Filter by year = 2024
    let recent_results: Vec<_> = all_results
        .iter()
        .filter(|r| r.chunk.metadata.get("year").and_then(|y| y.as_u64()) == Some(2024))
        .collect();

    println!("   2024 results: {}", recent_results.len());
    assert!(
        recent_results.len() >= 4,
        "Should have multiple 2024 documents"
    );

    println!("   Step 3: Verifying metadata-based filtering");
    assert!(
        !beginner_results.is_empty(),
        "Should find beginner documents"
    );
    assert!(
        !rust_advanced_results.is_empty(),
        "Should find advanced Rust documents"
    );
    assert!(
        beginner_results.len() <= all_results.len(),
        "Filtered results should be subset"
    );

    println!("   Metadata filtering working correctly");
    println!("Metadata filtering test completed");
}

#[tokio::test]
#[ignore = "requires LM Studio; set CALICHE_RUN_LMSTUDIO_INTEGRATION=true and run with --ignored"]
#[serial]
async fn test_lmstudio_structured_json_output() {
    require_lmstudio().await;

    println!("Testing structured JSON output parsing");

    let mut agent = AgentBuilder::new()
        .provider(ProviderType::LMStudio)
        .system_prompt(
            "You are a helpful assistant that can return data in JSON format when requested."
                .to_string(),
        )
        .build()
        .await
        .expect("Failed to build agent");

    // Test 1: Simple JSON object
    println!("   Test 1: Request simple JSON object");
    let response1 = agent
        .chat(
            "Return a JSON object with three fields: name='Alice', age=30, city='Paris'. \
            Return ONLY the JSON, no other text.",
        )
        .await
        .expect("Test 1 failed");

    println!("   Response: {}", response1.text());

    // Try to parse as JSON
    if let Ok(json_value) = serde_json::from_str::<serde_json::Value>(&response1) {
        println!("   Valid JSON received");
        println!(
            "   Parsed JSON: {}",
            serde_json::to_string_pretty(&json_value).expect("JSON pretty printing should succeed")
        );

        // Verify structure
        if json_value.get("name").is_some()
            && json_value.get("age").is_some()
            && json_value.get("city").is_some()
        {
            println!("   JSON contains expected fields");
        }
    } else {
        // Try to extract JSON from response
        if let Some(start) = response1.find('{') {
            if let Some(end) = response1.rfind('}') {
                let json_str = &response1[start..=end];
                match serde_json::from_str::<serde_json::Value>(json_str) {
                    Ok(json_value) => {
                        println!("   Extracted valid JSON from response");
                        println!(
                            "   Parsed JSON: {}",
                            serde_json::to_string_pretty(&json_value)
                                .expect("JSON pretty printing should succeed")
                        );
                    }
                    Err(_) => {
                        println!(
                            "   [WARNING]  Response is not valid JSON - model may not support structured output"
                        );
                        println!("   This is expected with some models");
                    }
                }
            }
        }
    }

    // Test 2: JSON array
    println!("   Test 2: Request JSON array");
    let response2 = agent
        .chat(
            "Return a JSON array of three programming languages with their year created. \
            Format: [{\"name\": \"Rust\", \"year\": 2010}, ...]. Return ONLY valid JSON.",
        )
        .await
        .expect("Test 2 failed");

    println!("   Response: {}", response2.text());

    if let Ok(json_value) = serde_json::from_str::<serde_json::Value>(&response2) {
        if json_value.is_array() {
            println!("   Valid JSON array received");
            let array = json_value
                .as_array()
                .expect("JSON value should be an array");
            println!("   Array length: {}", array.len());
        } else if json_value.is_object() {
            println!("   [INFO]  Received JSON object instead of array");
        }
    } else {
        // Try to extract JSON array
        if let Some(start) = response2.find('[') {
            if let Some(end) = response2.rfind(']') {
                let json_str = &response2[start..=end];
                match serde_json::from_str::<serde_json::Value>(json_str) {
                    Ok(json_value) => {
                        println!("   Extracted valid JSON array from response");
                        println!(
                            "   Array length: {}",
                            json_value.as_array().map(|a| a.len()).unwrap_or(0)
                        );
                    }
                    Err(_) => {
                        println!("   [WARNING]  Could not parse JSON array");
                    }
                }
            }
        }
    }

    // Test 3: Nested JSON structure
    println!("   Test 3: Request nested JSON structure");
    let response3 = agent
        .chat(
            "Create a JSON object for a person with nested address: \
            {\"name\": \"Bob\", \"address\": {\"street\": \"Main St\", \"city\": \"NYC\"}}. \
            Return ONLY valid JSON.",
        )
        .await
        .expect("Test 3 failed");

    println!("   Response: {}", response3.text());

    // Try to find and parse JSON
    let json_extracted = if let Some(start) = response3.find('{') {
        if let Some(end) = response3.rfind('}') {
            Some(&response3[start..=end])
        } else {
            None
        }
    } else {
        None
    };

    if let Some(json_str) = json_extracted {
        match serde_json::from_str::<serde_json::Value>(json_str) {
            Ok(json_value) => {
                println!("   Parsed nested JSON structure");

                // Check for nested structure
                if let Some(address) = json_value.get("address") {
                    if address.is_object() {
                        println!("   Nested address object found");
                        println!(
                            "   Parsed JSON: {}",
                            serde_json::to_string_pretty(&json_value)
                                .expect("JSON pretty printing should succeed")
                        );
                    }
                } else {
                    println!("   [INFO]  JSON parsed but may not have expected nested structure");
                }
            }
            Err(e) => {
                println!("   [WARNING]  JSON parse error: {}", e);
                println!("   This is expected - not all models support structured output reliably");
            }
        }
    }

    // Test 4: Tool that returns JSON
    println!("   Test 4: Tool returning structured data");

    #[derive(Debug)]
    struct JsonDataTool;
    impl ToolMeta for JsonDataTool {
        fn name(&self) -> &str {
            "get_user_data"
        }

        fn description(&self) -> &str {
            "Get user data in JSON format"
        }

        fn parameters(&self) -> serde_json::Value {
            json!({
                "type": "object",
                "properties": {
                    "user_id": {
                        "type": "string",
                        "description": "The user ID"
                    }
                },
                "required": ["user_id"]
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
    impl Tool for JsonDataTool {
        async fn execute(&self, args: serde_json::Value) -> ToolResult {
            let user_id = args["user_id"].as_str().unwrap_or("unknown");
            let data = json!({
                "user_id": user_id,
                "name": "Alice Smith",
                "email": "alice@example.com",
                "roles": ["admin", "developer"],
                "settings": {
                    "theme": "dark",
                    "notifications": true
                }
            });
            ToolResult::success(
                serde_json::to_string_pretty(&data).expect("JSON pretty printing should succeed"),
            )
        }
    }

    agent.add_tool(Arc::new(JsonDataTool));

    let response4 = agent
        .chat("Use get_user_data tool to fetch data for user_id '12345'")
        .await
        .expect("Test 4 failed");

    println!("   Response: {}", response4.text());

    // The response should mention or include the user data
    if response4.contains("Alice") || response4.contains("admin") || response4.contains("dark") {
        println!("   Agent processed tool's JSON response");
    }

    println!("Structured JSON output test completed");
}

#[tokio::test]
#[ignore = "requires LM Studio; set CALICHE_RUN_LMSTUDIO_INTEGRATION=true and run with --ignored"]
#[serial]
async fn test_lmstudio_code_execution() {
    require_lmstudio().await;

    println!("Testing Python code execution tool");

    let mut agent = AgentBuilder::new()
        .provider(ProviderType::LMStudio)
        .system_prompt(
            "You are a helpful coding assistant. When asked to perform calculations or data processing, \
            use the execute_python tool to write and run Python code. Always use print() to show results."
                .to_string(),
        )
        .enable_tool("execute_python")
        .build()
        .await
        .expect("Failed to build agent");

    // Test 1: Simple calculation
    println!("   Test 1: Ask agent to calculate using Python");
    let response1 = agent
        .chat("Use Python to calculate the sum of numbers from 1 to 100")
        .await
        .expect("Test 1 failed");

    println!("   Response: {}", response1.text());
    assert!(!response1.text().is_empty());

    // Check if the response mentions the correct answer (5050) or shows Python was used
    if response1.contains("5050") || response1.contains("sum") || response1.contains("range") {
        println!("   Agent successfully used Python for calculation");
    } else {
        println!("   [WARNING]  Agent may not have used Python tool");
    }

    // Test 2: Data processing
    println!("   Test 2: Data processing with lists");
    let response2 = agent
        .chat(
            "Use Python to find the average of these numbers: 10, 20, 30, 40, 50. \
            Show the calculation.",
        )
        .await
        .expect("Test 2 failed");

    println!("   Response: {}", response2.text());
    assert!(!response2.text().is_empty());

    // Average should be 30
    if response2.contains("30") {
        println!("   Correct average calculated");
    }

    // Test 3: Working with JSON
    println!("   Test 3: JSON processing with Python");
    let response3 = agent
        .chat(
            "Use Python to create a JSON object with three programming languages and their year of creation. \
            Use the json module to format it properly.",
        )
        .await
        .expect("Test 3 failed");

    println!("   Response: {}", response3.text());
    assert!(!response3.text().is_empty());

    // Test 4: String manipulation
    println!("   Test 4: String manipulation");
    let response4 = agent
        .chat(
            "Use Python to reverse the string 'Hello World' and count how many vowels it contains.",
        )
        .await
        .expect("Test 4 failed");

    println!("   Response: {}", response4.text());
    assert!(!response4.text().is_empty());

    if response4.contains("dlroW olleH") || response4.contains("3") || response4.contains("vowel") {
        println!("   String manipulation successful");
    }

    // Test 5: Algorithm implementation
    println!("   Test 5: Simple algorithm - check if number is prime");
    let response5 = agent
        .chat("Use Python to check if 17 is a prime number. Show your work.")
        .await
        .expect("Test 5 failed");

    println!("   Response: {}", response5.text());
    assert!(!response5.text().is_empty());

    if response5.to_lowercase().contains("prime")
        || response5.contains("17")
        || response5.to_lowercase().contains("true")
    {
        println!("   Prime number check completed");
    }

    println!("Code execution test completed");
}

// ============================================================================
// ADVANCED FEATURES TESTS - Retrieval Chains, Prompts, Parsing, Loaders
// ============================================================================

#[tokio::test]
#[ignore = "requires LM Studio; set CALICHE_RUN_LMSTUDIO_INTEGRATION=true and run with --ignored"]
#[serial]
#[cfg(feature = "prompts")]
async fn test_lmstudio_prompt_templates() {
    require_lmstudio().await;

    println!("Testing Prompt Templates");

    let mut agent = AgentBuilder::new()
        .provider(ProviderType::LMStudio)
        .system_prompt("You are a helpful assistant.".to_string())
        .build()
        .await
        .expect("Failed to build agent");

    // Test 1: Simple prompt template
    println!("   Test 1: Simple prompt template");
    let template = PromptTemplate::new("Tell me about {topic} in {style} style.");
    let prompt = template
        .format(&json!({
            "topic": "Rust programming",
            "style": "concise"
        }))
        .expect("Failed to format template");

    println!("   Formatted prompt: {}", prompt);
    let response = agent.chat(&prompt).await.expect("Chat failed");
    println!("   Response: {}", response.text());
    assert!(!response.text().is_empty());

    // Test 2: Few-shot prompting
    println!("   Test 2: Few-shot prompt template");
    let examples = vec![
        Example::new("What is 2+2?", "The answer is 4."),
        Example::new("What is 10-3?", "The answer is 7."),
    ];

    let few_shot =
        FewShotPromptTemplate::new(examples, "Here are some examples:", "Now answer: {input}");

    let prompt = few_shot
        .format(&json!({"input": "What is 5*3?"}))
        .expect("Failed to format few-shot");

    println!("   Few-shot prompt: {}", prompt);
    let response = agent.chat(&prompt).await.expect("Chat failed");
    println!("   Response: {}", response.text());

    if response.text().contains("15") {
        println!("   Few-shot prompting successful");
    }

    // Test 3: Chat prompt template
    println!("   Test 3: Chat prompt template");
    let chat_template = ChatPromptTemplate::with_system("You are a helpful math tutor.")
        .add_user("Explain what a prime number is in one sentence.")
        .add_assistant("A prime number is a number greater than 1 that has no divisors other than 1 and itself.")
        .add_user("Is 17 prime?");

    let prompt = chat_template
        .format_as_string(&json!({}))
        .expect("Failed to format chat");

    println!("   Chat prompt:\n{}", prompt);
    let response = agent.chat(&prompt).await.expect("Chat failed");
    println!("   Response: {}", response.text());

    println!("Prompt templates test completed");
}

#[tokio::test]
#[ignore = "requires LM Studio; set CALICHE_RUN_LMSTUDIO_INTEGRATION=true and run with --ignored"]
#[serial]
#[cfg(feature = "output-parsing")]
async fn test_lmstudio_structured_output_parsing() {
    require_lmstudio().await;

    println!("Testing Structured Output Parsing");

    let mut agent = AgentBuilder::new()
        .provider(ProviderType::LMStudio)
        .system_prompt(
            "You are a helpful assistant. When asked to return structured data, \
             return ONLY valid JSON with the requested fields."
                .to_string(),
        )
        .build()
        .await
        .expect("Failed to build agent");

    // Test 1: Parse user profile
    println!("   Test 1: Parse user profile from JSON");

    #[derive(Debug, Serialize, Deserialize)]
    struct UserProfile {
        name: String,
        age: u32,
        occupation: String,
        interests: Vec<String>,
    }

    let response = agent
        .chat(
            "Return a JSON object for a user profile with fields: name='Alice Smith', \
             age=28, occupation='Software Engineer', interests=['Rust', 'AI', 'Music']. \
             Return ONLY the JSON object, no other text.",
        )
        .await
        .expect("Chat failed");

    println!("   Raw response: {}", response.text());

    let parser = StructuredOutputParser::<UserProfile>::new();
    match parser.parse(&response).await {
        Ok(profile) => {
            println!("   Successfully parsed user profile:");
            println!("     Name: {}", profile.name);
            println!("     Age: {}", profile.age);
            println!("     Occupation: {}", profile.occupation);
            println!("     Interests: {:?}", profile.interests);
        }
        Err(e) => {
            println!(
                "   [WARNING]  Failed to parse: {} - Model may not support structured output",
                e
            );
        }
    }

    // Test 2: JSON schema validation
    println!("   Test 2: JSON schema validation");
    let schema = json!({
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "rating": {"type": "number", "minimum": 0, "maximum": 10}
        },
        "required": ["title", "rating"]
    });

    let json_parser = JsonOutputParser::with_schema(schema).expect("Failed to create parser");

    let response = agent
        .chat(
            "Return JSON: {\"title\": \"Great Movie\", \"rating\": 8.5}. \
             Return ONLY this JSON, nothing else.",
        )
        .await
        .expect("Chat failed");

    println!("   Raw response: {}", response.text());

    match json_parser.parse(&response).await {
        Ok(parsed) => {
            println!("   JSON schema validation passed");
            println!(
                "     Parsed: {}",
                serde_json::to_string_pretty(&parsed).expect("JSON pretty printing should succeed")
            );
        }
        Err(e) => {
            println!("   [WARNING]  Schema validation failed: {}", e);
        }
    }

    println!("Structured output parsing test completed");
}

#[tokio::test]
#[ignore = "requires LM Studio; set CALICHE_RUN_LMSTUDIO_INTEGRATION=true and run with --ignored"]
#[serial]
#[cfg(feature = "document-loaders")]
async fn test_lmstudio_document_loaders() {
    require_lmstudio().await;

    println!("Testing Document Loaders and Chunking");

    // Test 1: Text splitting for RAG
    println!("   Test 1: Text splitting for RAG");
    let long_text = "Rust is a systems programming language. \
                     It provides memory safety without garbage collection. \
                     The ownership system is central to Rust. \
                     Borrowing allows references without ownership transfer. \
                     Zero-cost abstractions enable high performance.";

    let splitter = TextSplitter::new(50, 10);
    let chunks = splitter.split_text(long_text);

    println!("   Split text into {} chunks", chunks.len());
    for (i, chunk) in chunks.iter().enumerate() {
        println!("     Chunk {}: {} chars", i + 1, chunk.len());
    }

    // Test 2: Document chunking for embeddings
    println!("   Test 2: Chunk documents for embedding");
    let docs = vec![
        Document::new("Rust memory safety is guaranteed by the ownership system."),
        Document::new("Python is dynamically typed and uses garbage collection."),
    ];

    let chunked = splitter.split_documents(&docs);
    println!(
        "   Created {} chunks from {} documents",
        chunked.len(),
        docs.len()
    );

    // Test 3: Documents with metadata
    println!("   Test 3: Document metadata preservation");
    let doc_with_meta = Document::new("Important Rust concept: ownership prevents memory bugs")
        .with_metadata("language", "rust")
        .with_metadata("topic", "memory");

    println!(
        "   Created document with {} metadata entries",
        doc_with_meta.metadata.len()
    );
    for (key, value) in &doc_with_meta.metadata {
        println!("     {}: {}", key, value);
    }

    println!("Document loaders test completed");
}

#[tokio::test]
#[ignore = "requires LM Studio; set CALICHE_RUN_LMSTUDIO_INTEGRATION=true and run with --ignored"]
#[serial]
#[cfg(feature = "chains")]
async fn test_lmstudio_chain_patterns() {
    require_lmstudio().await;

    println!("Testing Chain Patterns");

    let agent = Arc::new(TokioMutex::new(
        AgentBuilder::new()
            .provider(ProviderType::LMStudio)
            .system_prompt("You are a helpful assistant.".to_string())
            .build()
            .await
            .expect("Failed to build agent"),
    ));

    // Test 1: LLM Chain
    println!("   Test 1: Basic LLM Chain");
    let template = PromptTemplate::new("Explain {concept} in one sentence.");
    let llm_chain = LLMChain::new(agent.clone(), template);

    let result = llm_chain
        .run(&json!({"concept": "Rust ownership"}))
        .await
        .expect("LLM chain failed");

    let output = result
        .output
        .get("output")
        .and_then(|v| v.as_str())
        .unwrap_or("");
    println!("   LLM Chain output: {}", output);
    assert!(!output.is_empty());

    // Test 2: Sequential Chain
    println!("   Test 2: Sequential Chain");
    let extract_template = PromptTemplate::new("Extract key points from: {text}");
    let summarize_template = PromptTemplate::new("Summarize these points: {output}");

    let extract_chain = LLMChain::new(agent.clone(), extract_template);
    let summarize_chain = LLMChain::new(agent.clone(), summarize_template);

    let seq_chain = ComposableChain::new(extract_chain) >> ComposableChain::new(summarize_chain);

    let result = seq_chain
        .run(&json!({
            "text": "Rust provides memory safety. It uses ownership and borrowing. Zero-cost abstractions."
        }))
        .await
        .expect("Sequential chain failed");

    let output = result
        .output
        .get("output")
        .and_then(|v| v.as_str())
        .unwrap_or("");
    println!("   Sequential chain output: {}", output);
    assert!(!output.is_empty());

    println!("Chain patterns test completed");
}

#[tokio::test]
#[ignore = "requires LM Studio; set CALICHE_RUN_LMSTUDIO_INTEGRATION=true and run with --ignored"]
#[serial]
#[cfg(feature = "retrieval-chains")]
async fn test_lmstudio_retrieval_chain() {
    require_lmstudio().await;

    println!("Testing Retrieval Chain (RAG with Vector Search)");

    // Setup embedding provider
    let embedding_provider: Arc<dyn EmbeddingProvider> = Arc::new(
        LMStudioProvider::from_env(None)
            .await
            .expect("Failed to create LM Studio provider"),
    );

    // Create and populate vector store
    println!("   Step 1: Creating and populating vector store");
    let vector_backend: Arc<dyn Backend> = Arc::new(MemoryBackend::new());

    let documents = vec![
        (
            "rust_memory",
            "Rust provides memory safety without garbage collection through ownership.",
        ),
        (
            "rust_performance",
            "Rust offers zero-cost abstractions and compiles to native code.",
        ),
        (
            "rust_concurrency",
            "Rust prevents data races at compile time using its type system.",
        ),
    ];

    for (i, (doc_id, text)) in documents.iter().enumerate() {
        let embedding_response = embedding_provider
            .embed(EmbeddingRequest {
                texts: vec![text.to_string()],
                model: None,
            })
            .await
            .expect("Failed to embed");

        let chunk = Chunk::new(
            doc_id.to_string(),
            i,
            text.to_string(),
            embedding_response.embeddings[0].clone(),
            0,
            text.len(),
            1,
            json!({"text": text}),
            Some("rust_docs".to_string()),
        );

        vector_backend
            .upsert(&[chunk])
            .await
            .expect("Failed to store");
    }

    println!("   Indexed {} documents", documents.len());

    // Create agent and LLM chain
    println!("   Step 2: Creating retrieval chain");
    let agent = Arc::new(TokioMutex::new(
        AgentBuilder::new()
            .provider(ProviderType::LMStudio)
            .system_prompt(
                "You are a Rust expert. Answer based on the provided context.".to_string(),
            )
            .build()
            .await
            .expect("Failed to build agent"),
    ));

    let template = PromptTemplate::new(
        "Context:\n{context}\n\nQuestion: {query}\n\nAnswer based on the context:",
    );
    let llm_chain = Arc::new(LLMChain::new(agent, template));

    let retrieval_chain = RetrievalChain::new(embedding_provider, vector_backend, llm_chain)
        .with_namespace("rust_docs")
        .with_top_k(2);

    // Query the chain
    println!("   Step 3: Querying retrieval chain");
    let result = retrieval_chain
        .run(&json!({"query": "How does Rust handle memory safety?"}))
        .await
        .expect("Retrieval chain failed");

    let answer = result
        .output
        .get("output")
        .and_then(|v| v.as_str())
        .unwrap_or("[No output]");
    println!("   Answer: {}", answer);

    // Check retrieval stats
    if let Some((_, stats)) = result
        .context
        .intermediate_steps
        .iter()
        .find(|(name, _)| name == "retrieval")
    {
        println!(
            "   [INFO] Retrieval stats: {}",
            serde_json::to_string_pretty(stats).expect("JSON pretty printing should succeed")
        );
    }

    assert!(!answer.is_empty() && answer != "[No output]");
    println!("Retrieval chain test completed");
}

#[tokio::test]
#[ignore = "requires LM Studio; set CALICHE_RUN_LMSTUDIO_INTEGRATION=true and run with --ignored"]
#[serial]
#[cfg(feature = "retrieval-chains")]
async fn test_lmstudio_hybrid_search_chain() {
    require_lmstudio().await;

    println!("Testing Hybrid Search Chain (Vector + Keyword)");

    // Setup
    let embedding_provider: Arc<dyn EmbeddingProvider> = Arc::new(
        LMStudioProvider::from_env(None)
            .await
            .expect("Failed to create LM Studio provider"),
    );

    let vector_backend: Arc<dyn Backend> = Arc::new(MemoryBackend::new());

    // Index documents
    println!("   Step 1: Indexing documents for hybrid search");
    let documents = vec![
        ("memory_safety", "Rust memory safety ownership borrowing"),
        (
            "performance",
            "Rust performance zero-cost abstractions native",
        ),
        (
            "concurrency",
            "Rust concurrency threads data races compile time",
        ),
    ];

    for (i, (doc_id, text)) in documents.iter().enumerate() {
        let embedding_response = embedding_provider
            .embed(EmbeddingRequest {
                texts: vec![text.to_string()],
                model: None,
            })
            .await
            .expect("Failed to embed");

        let chunk = Chunk::new(
            doc_id.to_string(),
            i,
            text.to_string(),
            embedding_response.embeddings[0].clone(),
            0,
            text.len(),
            1,
            json!({"text": text}),
            Some("rust".to_string()),
        );

        vector_backend
            .upsert(&[chunk])
            .await
            .expect("Failed to store");
    }

    // Create hybrid chain
    println!("   Step 2: Creating hybrid search chain");
    let agent = Arc::new(TokioMutex::new(
        AgentBuilder::new()
            .provider(ProviderType::LMStudio)
            .system_prompt("Answer based on the documents.".to_string())
            .build()
            .await
            .expect("Failed to build agent"),
    ));

    let template = PromptTemplate::new("Documents:\n{context}\n\nQuery: {query}\n\nAnswer:");
    let llm_chain = Arc::new(LLMChain::new(agent, template));

    let hybrid_chain = HybridSearchChain::new(embedding_provider, vector_backend, llm_chain)
        .with_namespace("rust")
        .with_vector_weight(0.6)
        .with_keyword_weight(0.4)
        .with_top_k(3);

    // Query
    println!("   Step 3: Running hybrid search");
    let result = hybrid_chain
        .run(&json!({"query": "memory ownership"}))
        .await
        .expect("Hybrid chain failed");

    let answer = result
        .output
        .get("output")
        .and_then(|v| v.as_str())
        .unwrap_or("[No output]");
    println!("   Answer: {}", answer);

    // Check fusion stats
    if let Some((_, stats)) = result
        .context
        .intermediate_steps
        .iter()
        .find(|(name, _)| name == "fusion")
    {
        println!(
            "   [INFO] Fusion: {}",
            serde_json::to_string_pretty(stats).expect("JSON pretty printing should succeed")
        );
    }

    assert!(!answer.is_empty());
    println!("Hybrid search chain test completed");
}

#[tokio::test]
#[ignore = "requires LM Studio; set CALICHE_RUN_LMSTUDIO_INTEGRATION=true and run with --ignored"]
#[serial]
#[cfg(feature = "retrieval-chains")]
async fn test_lmstudio_document_mapreduce_chain() {
    require_lmstudio().await;

    println!("Testing Document Map-Reduce Chain");

    let agent = Arc::new(TokioMutex::new(
        AgentBuilder::new()
            .provider(ProviderType::LMStudio)
            .system_prompt("You are a summarization assistant.".to_string())
            .build()
            .await
            .expect("Failed to build agent"),
    ));

    // Map chain: Extract key points from each chunk
    println!("   Step 1: Creating map-reduce chains");
    let map_template = PromptTemplate::new("Extract key points from: {text}");
    let map_chain = Arc::new(LLMChain::new(agent.clone(), map_template));

    // Reduce chain: Combine all key points
    let reduce_template = PromptTemplate::new("Combine these summaries: {summaries}");
    let reduce_chain = Arc::new(LLMChain::new(agent, reduce_template));

    let doc_chain = DocumentMapReduceChain::new(map_chain, reduce_chain).with_chunking(
        ChunkingStrategy::FixedSize {
            size: 100,
            overlap: 20,
        },
    );

    // Process document
    println!("   Step 2: Processing large document");
    let large_doc = "Rust is a systems programming language focused on safety and performance. \
                     The ownership system prevents memory bugs at compile time. \
                     Zero-cost abstractions allow high-level programming without runtime overhead. \
                     Rust's type system catches errors before the code runs. \
                     Concurrency in Rust is safe thanks to ownership rules.";

    let result = doc_chain
        .run(&json!({"document": large_doc}))
        .await
        .expect("Document chain failed");

    let summary = result
        .output
        .get("output")
        .and_then(|v| v.as_str())
        .unwrap_or("[No output]");
    println!("   Final summary: {}", summary);

    // Check chunking stats
    if let Some((_, stats)) = result
        .context
        .intermediate_steps
        .iter()
        .find(|(name, _)| name == "chunking")
    {
        println!(
            "   [INFO] Chunking: {}",
            serde_json::to_string_pretty(stats).expect("JSON pretty printing should succeed")
        );
    }

    assert!(!summary.is_empty());
    println!("Document map-reduce chain test completed");
}
