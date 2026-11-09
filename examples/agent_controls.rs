//! Agent Lifecycle Controls Example
//!
//! This example demonstrates advanced agent control features:
//! - Lifecycle hooks (PreToolUse/PostToolUse) for intercepting tool execution
//! - Permission modes (AcceptEdits, AcceptAll, DenyAll) for automatic permission handling
//! - Allowed tools filtering to restrict which tools can be used
//! - In-process MCP servers for direct tool registration without subprocess overhead
//! - Session forking for parallel work and subagents
//!
//! Run with:
//! ```bash
//! cargo run --example agent_controls --features=full,mcp
//! ```

#![expect(
    clippy::print_stdout,
    reason = "examples print their results for the reader"
)]

use anyhow::Result;
use async_trait::async_trait;
use caliche::prelude::{Message, ProviderType};
use caliche::{
    AgentBuilder, Tool, ToolMeta, ToolResult,
    agent::hooks::permission::PermissionMode,
    agent::hooks::{HookContext, HookMatcher, HookResult, LifecycleHook},
    agent::session::AgentSession,
};
use serde_json::json;
use std::sync::Arc;

/// Example hook that blocks dangerous bash commands
#[derive(Debug)]
struct BashSecurityHook;

#[async_trait]
impl LifecycleHook for BashSecurityHook {
    async fn on_pre_tool_use(&self, ctx: &HookContext<'_>) -> HookResult {
        if ctx.tool_name == "Bash" {
            // Check if command contains dangerous patterns
            if let Some(command) = ctx.tool_input.get("command").and_then(|v| v.as_str()) {
                let dangerous_patterns = ["rm -rf", "format", "delete", "drop database"];
                for pattern in &dangerous_patterns {
                    if command.contains(pattern) {
                        return HookResult::deny(format!(
                            "Command contains dangerous pattern: {}",
                            pattern
                        ));
                    }
                }
            }
        }
        HookResult::allow()
    }

    async fn on_post_tool_use(&self, _ctx: &HookContext<'_>, result: &ToolResult) -> HookResult {
        // Log tool execution for audit purposes
        if !result.success {
            tracing::warn!("Tool execution failed: {:?}", result.error);
        }
        HookResult::allow()
    }
}

/// Example custom tool for in-process MCP server
#[derive(Debug)]
struct GreetTool;

impl ToolMeta for GreetTool {
    fn name(&self) -> &str {
        "greet"
    }

    fn description(&self) -> &str {
        "Greet a user by name"
    }

    fn parameters(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Name of the person to greet"
                }
            },
            "required": ["name"]
        })
    }
}

#[async_trait]
impl Tool for GreetTool {
    async fn execute(&self, args: serde_json::Value) -> ToolResult {
        let name = args.get("name").and_then(|v| v.as_str()).unwrap_or("World");
        ToolResult::success(format!("Hello, {}!", name))
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    dotenvy::dotenv().ok();
    tracing_subscriber::fmt::init();

    println!("=== Agent Lifecycle Controls Demo ===\n");

    // Example 1: Hooks - Security hook for bash commands
    println!("=== Example 1: Lifecycle Hooks ===");
    let _agent_with_hooks = AgentBuilder::new()
        .provider(ProviderType::Bedrock)
        .model("us.anthropic.claude-haiku-4-5-20251001-v1:0".to_string())
        .enable_tool("calculator")
        .hook(HookMatcher::tool("Bash"), Arc::new(BashSecurityHook))
        .build()
        .await?;

    println!("Agent configured with security hook for Bash tool");
    println!("(Hook will block dangerous commands)\n");

    // Example 2: Permission Modes
    println!("=== Example 2: Permission Modes ===");

    // AcceptEdits mode - auto-accept file edits
    let _agent_accept_edits = AgentBuilder::new()
        .provider(ProviderType::Bedrock)
        .model("us.anthropic.claude-haiku-4-5-20251001-v1:0".to_string())
        .permission_mode(PermissionMode::AcceptEdits)
        .enable_tool("fs")
        .build()
        .await?;
    println!("Agent configured with AcceptEdits permission mode");
    println!("(File edits will be auto-accepted)\n");

    // DenyAll mode - block all tool calls
    let _agent_deny_all = AgentBuilder::new()
        .provider(ProviderType::Bedrock)
        .model("us.anthropic.claude-haiku-4-5-20251001-v1:0".to_string())
        .permission_mode(PermissionMode::DenyAll)
        .enable_tool("calculator")
        .build()
        .await?;
    println!("Agent configured with DenyAll permission mode");
    println!("(All tool calls will be blocked)\n");

    // Example 3: Allowed Tools
    println!("=== Example 3: Allowed Tools Filtering ===");
    let allowed = vec!["calculator", "fs"];
    let agent_allowed_tools = AgentBuilder::new()
        .provider(ProviderType::Bedrock)
        .model("us.anthropic.claude-haiku-4-5-20251001-v1:0".to_string())
        .allowed_tools(&allowed)
        .enable_all_default_tools() // Enable all, but only calculator and echo will be allowed
        .build()
        .await?;

    println!("Agent configured with allowed_tools: {:?}", allowed);
    println!("Available tools (allowed tools marked with *):");
    for (name, _, _) in agent_allowed_tools.list_tools() {
        let marker = if allowed.iter().any(|a| *a == name) {
            "*"
        } else {
            " "
        };
        println!("  {} {}", marker, name);
    }
    println!();

    // Example 4: In-Process MCP Server (requires mcp feature)
    #[cfg(feature = "mcp")]
    {
        println!("=== Example 4: In-Process MCP Server ===");
        use caliche::mcp::InProcessMcpServer;

        // Create an in-process MCP server with custom tools
        let server = InProcessMcpServer::builder("my-tools")
            .version("1.0.0")
            .add_tool(GreetTool)
            .build()?;

        println!(
            "Created in-process MCP server '{}' v{}",
            server.name(),
            server.version()
        );
        println!("Tools in server:");
        for tool in server.list_tools() {
            println!("  - {}: {}", tool.name, tool.description);
        }

        // Register the MCP server with an agent
        let agent_with_mcp = AgentBuilder::new()
            .provider(ProviderType::Bedrock)
            .model("us.anthropic.claude-haiku-4-5-20251001-v1:0".to_string())
            .mcp_server("my-tools", server)
            .build()
            .await?;

        println!("\nAgent configured with in-process MCP server");
        println!("Available tools (including MCP server tools):");
        for (name, _, _) in agent_with_mcp.list_tools() {
            println!("  - {}", name);
        }
        println!();
    }

    #[cfg(not(feature = "mcp"))]
    {
        println!("=== Example 4: In-Process MCP Server ===");
        println!("(Skipped: Requires 'mcp' feature)");
        println!("Run with: cargo run --example agent_controls --features=full,mcp\n");
    }

    // Example 5: Session Forking
    println!("=== Example 5: Session Forking ===");

    // Create a main session
    let main_session = AgentSession::new("main-session");
    main_session
        .add_message(Message::user("Hello, I'm starting a task"))
        .map_err(|e| anyhow::anyhow!("Failed to add message: {}", e))?;

    println!("Created main session: {}", main_session.id);
    println!(
        "Main session history: {} messages",
        main_session
            .get_history()
            .map_err(|e| anyhow::anyhow!("Failed to get history: {}", e))?
            .len()
    );

    // Fork the session for parallel work
    let forked_session = main_session
        .fork("parallel-worker")
        .await
        .map_err(|e| anyhow::anyhow!("Failed to fork session: {}", e))?;
    println!("\nForked session: {}", forked_session.id);
    println!(
        "Forked session history: {} messages (copied from main)",
        forked_session
            .get_history()
            .map_err(|e| anyhow::anyhow!("Failed to get history: {}", e))?
            .len()
    );

    // Add messages to the forked session
    forked_session
        .add_message(Message::user("Working on parallel task"))
        .map_err(|e| anyhow::anyhow!("Failed to add message: {}", e))?;
    forked_session
        .add_message(Message::assistant("Completed parallel work"))
        .map_err(|e| anyhow::anyhow!("Failed to add message: {}", e))?;

    println!(
        "After adding messages to fork: {} messages",
        forked_session
            .get_history()
            .map_err(|e| anyhow::anyhow!("Failed to get history: {}", e))?
            .len()
    );

    // Merge the forked session back into main
    let mut main_session = main_session;
    main_session
        .merge(&forked_session)
        .await
        .map_err(|e| anyhow::anyhow!("Failed to merge session: {}", e))?;
    println!("\nMerged fork back into main session");
    println!(
        "Main session history after merge: {} messages",
        main_session
            .get_history()
            .map_err(|e| anyhow::anyhow!("Failed to get history: {}", e))?
            .len()
    );

    println!("\nSession forking enables:");
    println!("  - Parallel work with isolated state");
    println!("  - Subagents that can work independently");
    println!("  - Merging results back into main conversation");
    println!();

    // Example 6: Combined Features
    println!("=== Example 6: Combined Features ===");
    let _combined_agent = AgentBuilder::new()
        .provider(ProviderType::Bedrock)
        .model("us.anthropic.claude-haiku-4-5-20251001-v1:0".to_string())
        .allowed_tools(&["calculator", "fs"])
        .permission_mode(PermissionMode::AcceptEdits)
        .hook(HookMatcher::all(), Arc::new(BashSecurityHook))
        .build()
        .await?;

    println!("Agent configured with:");
    println!("  - Allowed tools: [calculator, echo]");
    println!("  - Permission mode: AcceptEdits");
    println!("  - Security hook for all tools");
    println!();

    println!("=== Examples Complete ===");
    println!("\nKey Features:");
    println!("1. Lifecycle Hooks: Intercept and control tool execution");
    println!("2. Permission Modes: Control tool call permissions automatically");
    println!("3. Allowed Tools: Restrict which tools can be used");
    println!("4. In-Process MCP Servers: Direct tool registration without subprocess overhead");
    println!("5. Session Forking: Enable parallel work and subagents");

    Ok(())
}
