//! Tools Example
//!
//! This example demonstrates:
//! - Built-in tools (calculator, echo, file operations, HTTP, etc.)
//! - Implementing custom tools using the Tool trait
//! - Using toolkits to bundle related tools
//! - Tool credentials management
//! - Python code execution tool
//! - SQL tools for database operations
//!
//! Best practices shown:
//! - Proper error handling in tool execution
//! - Clear tool descriptions and parameters
//! - Explicit tool selection for security
//! - Credential management patterns
//!
//! Run with:
//! ```bash
//! cargo run --example tools --features full
//! ```

#![expect(
    clippy::print_stdout,
    reason = "examples print their results for the reader"
)]

use anyhow::Result;
use async_trait::async_trait;
use caliche::UserContext;
use caliche::agent::tools::{FileSystemToolkit, HttpTool, ToolCredential};
use caliche::prelude::ProviderType;
use caliche::{AgentBuilder, Tool, ToolMeta, ToolResult, Toolkit};
use serde_json::json;
use std::collections::HashMap;
use std::env;
use std::sync::Arc;

// Custom tool implementations
/// A custom tool that reverses a string
#[derive(Debug)]
struct ReverseStringTool;

impl ToolMeta for ReverseStringTool {
    fn name(&self) -> &str {
        "reverse_string"
    }

    fn description(&self) -> &str {
        "Reverses a string. Takes a 'text' parameter and returns the reversed text."
    }

    fn parameters(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "The text to reverse"
                }
            },
            "required": ["text"]
        })
    }
}

#[async_trait]
impl Tool for ReverseStringTool {
    async fn execute(&self, args: serde_json::Value) -> ToolResult {
        let text = match args["text"].as_str() {
            Some(t) => t,
            None => return ToolResult::error("Missing 'text' parameter".to_string()),
        };

        let reversed: String = text.chars().rev().collect();

        let result = json!({
            "original": text,
            "reversed": reversed
        });

        ToolResult::success(result.to_string())
    }
}

/// A custom tool that counts words in text
#[derive(Debug)]
struct WordCountTool;

impl ToolMeta for WordCountTool {
    fn name(&self) -> &str {
        "count_words"
    }

    fn description(&self) -> &str {
        "Counts the number of words in a text string. Takes a 'text' parameter."
    }

    fn parameters(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "The text to count words in"
                }
            },
            "required": ["text"]
        })
    }
}

#[async_trait]
impl Tool for WordCountTool {
    async fn execute(&self, args: serde_json::Value) -> ToolResult {
        let text = match args["text"].as_str() {
            Some(t) => t,
            None => return ToolResult::error("Missing 'text' parameter".to_string()),
        };

        let word_count = text.split_whitespace().count();
        let char_count = text.chars().count();
        let line_count = text.lines().count();

        let result = json!({
            "text": text,
            "words": word_count,
            "characters": char_count,
            "lines": line_count
        });

        ToolResult::success(result.to_string())
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    dotenvy::dotenv().ok();

    println!("=== Built-in Tools Examples ===\n");

    // Example 1: Calculator Tool
    println!("Example 1: Calculator Tool");
    let mut agent = AgentBuilder::new()
        .provider(ProviderType::Bedrock)
        .model("us.anthropic.claude-haiku-4-5-20251001-v1:0".to_string())
        .enable_tool("calculator")
        .build()
        .await?;

    let response = agent.chat("What is 42 multiplied by 17?").await?;
    println!("Query: What is 42 multiplied by 17?");
    println!("Response: {}\n", response.text());

    // Example 2: Echo Tool
    println!("Example 2: Echo Tool");
    let mut agent = AgentBuilder::new()
        .provider(ProviderType::Bedrock)
        .model("us.anthropic.claude-haiku-4-5-20251001-v1:0".to_string())
        .enable_tool("fs")
        .build()
        .await?;

    let response = agent
        .chat("Echo back: Hello from the tools showcase!")
        .await?;
    println!("Query: Echo back: Hello from the tools showcase!");
    println!("Response: {}\n", response.text());

    // Example 3: Read File Tool
    println!("Example 3: Read File Tool");
    let mut agent = AgentBuilder::new()
        .provider(ProviderType::Bedrock)
        .model("us.anthropic.claude-haiku-4-5-20251001-v1:0".to_string())
        .enable_tool("fs")
        .build()
        .await?;

    let response = agent
        .chat("Read the first 5 lines of the README.md file")
        .await?;
    println!("Query: Read the first 5 lines of the README.md file");
    println!(
        "Response: {}...\n",
        caliche_contracts::text::truncate_utf8(response.text(), 150)
    );

    // Example 4: List Directory Tool
    println!("Example 4: List Directory Tool");
    let mut agent = AgentBuilder::new()
        .provider(ProviderType::Bedrock)
        .model("us.anthropic.claude-haiku-4-5-20251001-v1:0".to_string())
        .enable_tool("fs")
        .build()
        .await?;

    let response = agent
        .chat("List the files in the examples directory")
        .await?;
    println!("Query: List the files in the examples directory");
    println!(
        "Response: {}...\n",
        caliche_contracts::text::truncate_utf8(response.text(), 200)
    );

    // Example 5: Fetch URL Tool
    println!("Example 5: Fetch URL Tool");
    let mut agent = AgentBuilder::new()
        .provider(ProviderType::Bedrock)
        .model("us.anthropic.claude-haiku-4-5-20251001-v1:0".to_string())
        .enable_tool("http")
        .build()
        .await?;

    let response = agent
        .chat("Fetch the content from https://example.com and summarize it")
        .await?;

    println!("Query: Fetch content from https://example.com");
    println!(
        "Response: {}...\n",
        caliche_contracts::text::truncate_utf8(response.text(), 150)
    );

    // Example 6: HTTP Request Tool
    println!("Example 6: HTTP Request Tool");
    let mut agent = AgentBuilder::new()
        .provider(ProviderType::Bedrock)
        .model("us.anthropic.claude-haiku-4-5-20251001-v1:0".to_string())
        .enable_tool("http")
        .build()
        .await?;

    let response = agent
        .chat("Make a GET request to https://api.github.com/zen")
        .await?;

    println!("Query: GET request to https://api.github.com/zen");
    println!("Response: {}\n", response.text());

    // Example 7: Code Execution Tool
    println!("Example 7: Python Code Execution Tool");
    example_code_execution().await?;

    // Example 8: Data Manipulation Tool
    println!("Example 8: Data Manipulation Tool");
    let mut agent = AgentBuilder::new()
        .provider(ProviderType::Bedrock)
        .model("us.anthropic.claude-haiku-4-5-20251001-v1:0".to_string())
        .enable_tool("data_manipulation")
        .build()
        .await?;

    let json_data = r#"{"users": [{"name": "Alice", "age": 30}, {"name": "Bob", "age": 25}]}"#;

    let query = format!("Parse this JSON and extract all user names: {}", json_data);

    let response = agent.chat(&query).await?;

    println!("Query: Extract user names from JSON");
    println!("Response: {}\n", response.text());

    // Example 9: Process Document Tool
    println!("Example 9: Process Document Tool");
    println!("Process Document tool supports:");
    println!("  - Extract text from images (OCR)");
    println!("  - Extract text from PDFs");
    println!("  - Read image metadata");
    println!("  - Support for S3 documents\n");

    // Example 10: Multiple Tools Together
    println!("Example 10: Multiple Tools Together");
    let mut agent = AgentBuilder::new()
        .provider(ProviderType::Bedrock)
        .model("us.anthropic.claude-haiku-4-5-20251001-v1:0".to_string())
        .enable_tools(&["calculator", "http", "execute_python", "data_manipulation"])
        .build()
        .await?;

    let query =
        "Calculate the sum of 123 and 456, then use Python to check if the result is prime.";

    println!("Query: {}", query);
    let response = agent.chat(query).await?;
    println!("Response: {}\n", response.text());

    println!("=== Custom Tools Examples ===\n");

    // Example 11: Custom Tools
    let mut agent = AgentBuilder::new()
        .provider(ProviderType::Bedrock)
        .add_tool(Arc::new(ReverseStringTool))
        .add_tool(Arc::new(WordCountTool))
        .enable_tool("calculator")
        .build()
        .await?;

    let response = agent
        .chat("Please reverse this text: 'Hello, World!'")
        .await?;
    println!("Query: Please reverse this text: 'Hello, World!'");
    println!("Response: {}\n", response.text());

    let response = agent
        .chat("Count the words in this text: 'The quick brown fox jumps over the lazy dog'")
        .await?;
    println!("Query: Count the words in this text");
    println!("Response: {}\n", response.text());

    println!("=== Toolkit Examples ===\n");

    // Example 12: Using Built-in Toolkits
    println!("Example 12: Using FileSystemToolkit");
    let agent = AgentBuilder::new()
        .provider(ProviderType::Bedrock)
        .model("us.anthropic.claude-haiku-4-5-20251001-v1:0".to_string())
        .register_toolkit(Arc::new(FileSystemToolkit))
        .build()
        .await?;

    println!("Agent tools after registering FileSystemToolkit:");
    for (name, desc, _) in agent.list_tools() {
        println!("  - {}: {}", name, desc);
    }
    println!();

    // Example 12b: Creating a Custom Toolkit
    println!("Example 12b: Creating a Custom Toolkit");

    /// A custom toolkit that bundles text processing tools together
    struct TextProcessingToolkit;

    impl Toolkit for TextProcessingToolkit {
        fn get_tools(&self) -> Vec<Arc<dyn Tool>> {
            vec![Arc::new(ReverseStringTool), Arc::new(WordCountTool)]
        }

        fn name(&self) -> &str {
            "text_processing"
        }
    }

    println!("Created TextProcessingToolkit with tools:");
    let toolkit = TextProcessingToolkit;
    for tool in toolkit.get_tools() {
        println!("  - {}: {}", tool.name(), tool.description());
    }

    let mut agent = AgentBuilder::new()
        .provider(ProviderType::Bedrock)
        .model("us.anthropic.claude-haiku-4-5-20251001-v1:0".to_string())
        .register_toolkit(Arc::new(toolkit))
        .build()
        .await?;

    println!("\nAgent tools after registering TextProcessingToolkit:");
    for (name, desc, _) in agent.list_tools() {
        println!("  - {}: {}", name, desc);
    }

    let response = agent
        .chat("Use the reverse_string tool to reverse 'Hello, World!'")
        .await?;
    println!("\nQuery: Use the reverse_string tool to reverse 'Hello, World!'");
    println!("Response: {}\n", response.text());

    println!("=== Tool Credentials Examples ===\n");

    // Example 13: Tool Credentials
    println!("Example 13: Construction-time credentials (highest priority)");
    let bearer_token = ToolCredential::BearerToken("construction-time-token".to_string());
    let _http_tool = HttpTool::with_credentials(bearer_token);
    println!("   Tool created with construction-time bearer token\n");

    println!("Example 14: Runtime credentials via tool arguments");
    let _http_tool_no_cred = HttpTool::new();
    println!("   Tool created without construction-time credentials\n");

    println!("Example 15: Credentials from UserContext.custom_claims");
    let mut custom_claims = HashMap::new();
    custom_claims.insert("api_key".to_string(), json!("user-context-api-key"));

    let _user_context = UserContext {
        user_id: "user123".to_string(),
        tenant_id: Some("tenant456".to_string()),
        email: Some("user@example.com".to_string()),
        roles: vec!["user".to_string()],
        custom_claims,
    };
    println!("   UserContext created with api_key in custom_claims\n");

    println!("Credential priority order:");
    println!("   1. Construction-time credentials (highest, cannot be overridden)");
    println!("   2. Tool arguments (runtime credentials)");
    println!("   3. UserContext.custom_claims (JWT/user context)");
    println!("   4. None (fail if required)\n");

    println!("=== SQL Tools Examples ===\n");

    // Example 16: SQL Tools
    if let Ok(database_url) = std::env::var("DATABASE_URL") {
        println!("Example 16: SQL Tools");
        let mut agent = AgentBuilder::new()
            .provider(ProviderType::OpenAI)
            .enable_sql_tools_with_postgres(&database_url, "warehouse")
            .sql_tools_top_k(6)
            .sql_tools_max_rows(250)
            .build()
            .await?;

        println!("SQL tools enabled with schema namespace 'warehouse'");

        let response = agent
            .chat("Use sql_generate to create a query that lists all tables in the database")
            .await?;

        println!("\nGenerated SQL:");
        println!("{}", response.text());
        if let Some(total) = response.usage().total_tokens {
            println!("Tokens used: {}", total);
        }

        let response = agent
            .chat("Use sql_execute to run: SELECT version()")
            .await?;

        println!("\nSQL Execution Result:");
        println!("{}", response.text());
    } else {
        println!("Example 16: SQL Tools (skipped - DATABASE_URL not set)");
        println!("Set DATABASE_URL to enable SQL tools examples\n");
    }

    println!("=== Examples Complete ===");
    println!("\nTips:");
    println!("- Use explicit tool selection for security");
    println!("- Implement custom tools for domain-specific functionality");
    println!("- Use toolkits to bundle related tools together");
    println!("- Manage credentials securely using the credential priority chain");

    Ok(())
}

async fn example_code_execution() -> Result<()> {
    // Detect provider from command line or environment
    let provider = env::args()
        .position(|arg| arg == "--provider")
        .and_then(|pos| env::args().nth(pos + 1))
        .or_else(|| env::var("AGENT_PROVIDER").ok())
        .unwrap_or_else(|| {
            // Try to auto-detect based on available API keys
            if env::var("GOOGLE_API_KEY").is_ok() || env::var("GEMINI_API_KEY").is_ok() {
                "google".to_string()
            } else if env::var("ANTHROPIC_API_KEY").is_ok() {
                "anthropic".to_string()
            } else if env::var("OPENAI_API_KEY").is_ok() {
                "openai".to_string()
            } else {
                "lmstudio".to_string()
            }
        });

    let provider_type = match provider.to_lowercase().as_str() {
        "openai" => ProviderType::OpenAI,
        "anthropic" => ProviderType::Anthropic,
        "lmstudio" => ProviderType::LMStudio,
        "openrouter" => ProviderType::OpenRouter,
        "bedrock" => ProviderType::Bedrock,
        "perplexity" => ProviderType::Perplexity,
        "google" => ProviderType::Google,
        _ => ProviderType::Bedrock,
    };

    let model = match provider_type {
        ProviderType::OpenAI => "gpt-4o-mini",
        ProviderType::Anthropic => "claude-sonnet-4-5-20250929",
        ProviderType::LMStudio => "local-model",
        ProviderType::OpenRouter => "openai/gpt-4o-mini",
        ProviderType::Bedrock => "us.anthropic.claude-haiku-4-5-20251001-v1:0",
        ProviderType::Perplexity => "llama-3.1-sonar-small-128k-online",
        ProviderType::Google => "models/gemini-2.5-flash",
        _ => "us.anthropic.claude-haiku-4-5-20251001-v1:0",
    };

    let mut agent = AgentBuilder::new()
        .provider(provider_type)
        .model(model.to_string())
        .system_prompt(
            "You are a helpful coding assistant. When asked to perform calculations \
            or data processing, use the execute_python tool to write and run Python code. \
            Always use print() statements to show results. Only Python standard library \
            is available - no external packages."
                .to_string(),
        )
        .enable_tool("execute_python")
        .build()
        .await?;

    let response = agent
        .chat("Use Python to calculate the factorial of 10")
        .await?;
    println!("Query: Use Python to calculate the factorial of 10");
    println!("Response: {}\n", response.text());

    Ok(())
}
