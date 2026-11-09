//! Schema-Based Plugin Authoring Example
//!
//! This example demonstrates:
//! - Building an agent with the plugins feature and a persistent store
//! - Creating a tool plugin via the `plugin` tool (definition JSON + step DSL)
//! - Invoking the newly created tool
//! - Listing and exporting plugins
//!
//! The example defines a "greet" tool that takes a `name` argument and returns
//! "Hello, {{args.name}}!" using a single `return` step with template interpolation.
//!
//! Run with:
//! ```bash
//! cargo run --example plugins --features "plugins openai"
//! ```
//!
//! Set `OPENAI_API_KEY` if you later chat with the agent; the plugin create/invoke
//! steps do not require an LLM.

#![expect(
    clippy::print_stdout,
    reason = "examples print their results for the reader"
)]

use anyhow::{Context, Result};
use caliche::AgentBuilder;
use caliche::prelude::ProviderType;
use serde_json::json;

#[tokio::main]
async fn main() -> Result<()> {
    dotenvy::dotenv().ok();

    let store_path = std::env::temp_dir().join("caliche_plugins_example.db");
    let store_path = store_path.to_string_lossy().to_string();

    println!("Building agent with plugins store: {}", store_path);

    let service = AgentBuilder::new()
        .provider(ProviderType::OpenAI)
        .model("gpt-4o-mini".to_string())
        .with_plugins_store(store_path.as_str())
        .build()
        .await
        .context("Build agent (plugins feature + provider)")?;

    let tools = service.agent().tools.clone();

    println!("\n1. Create a 'greet' plugin (tool with one return step)");

    let create_result = tools
        .execute(
            "plugin",
            json!({
                "op": "create",
                "definition": {
                    "name": "greet",
                    "version": "1.0.0",
                    "description": "Returns a greeting for the given name.",
                    "plugin_type": "tool",
                    "spec": {
                        "type": "tool",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "name": { "type": "string", "description": "Name to greet" }
                            },
                            "required": ["name"]
                        },
                        "steps": [
                            {
                                "name": "out",
                                "type": { "type": "return", "value": "Hello, {{args.name}}!" }
                            }
                        ],
                        "is_readonly": true,
                        "capabilities": [],
                        "timeout_secs": 10
                    }
                }
            }),
        )
        .await;

    if create_result.success {
        println!("   Result: {}", create_result.content);
    } else {
        anyhow::bail!(
            "Create failed: {}",
            create_result.error.as_deref().unwrap_or("unknown")
        );
    }

    println!("\n2. Invoke the new 'greet' tool with name=\"World\"");

    let greet_result = tools.execute("greet", json!({ "name": "World" })).await;

    if greet_result.success {
        println!("   Result: {}", greet_result.content);
    } else {
        anyhow::bail!(
            "Greet failed: {}",
            greet_result.error.as_deref().unwrap_or("unknown")
        );
    }

    println!("\n3. List plugins");

    let list_result = tools.execute("plugin", json!({ "op": "list" })).await;

    if list_result.success {
        println!("   {}", list_result.content);
    } else {
        println!(
            "   List failed: {}",
            list_result.error.as_deref().unwrap_or("unknown")
        );
    }

    println!("\n4. Export the 'greet' plugin (JSON)");

    let export_result = tools
        .execute("plugin", json!({ "op": "export", "name": "greet" }))
        .await;

    if export_result.success {
        let truncated = if export_result.content.len() > 200 {
            format!(
                "{}...",
                caliche_contracts::text::truncate_utf8(&export_result.content, 200)
            )
        } else {
            export_result.content
        };
        println!("   {}", truncated);
    } else {
        println!(
            "   Export failed: {}",
            export_result.error.as_deref().unwrap_or("unknown")
        );
    }

    println!("\nDone. Plugins are persisted at {}", store_path);

    Ok(())
}
