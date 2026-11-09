//! Agent Orchestration Example
//!
//! This example demonstrates:
//! - Creating and registering multiple agents
//! - Using AgentCoordinator for load balancing
//! - Capability-based routing
//! - Agent handoffs and transfers
//! - Orchestration patterns (Pipeline, Hierarchical, Supervisor, Debate)
//! - Agent flows with AgentFlow
//!
//! Best practices shown:
//! - Specialized agents with clear capabilities
//! - Capability-based task delegation
//! - Pattern selection for different use cases
//!
//! Run with:
//! ```bash
//! cargo run --example orchestration
//! ```

#![expect(
    clippy::print_stdout,
    reason = "examples print their results for the reader"
)]

use anyhow::Result;
use caliche::orchestration::patterns::{
    DebatePattern, HierarchicalPattern, PipelinePattern, SupervisorPattern,
};
use caliche::orchestration::workflow::{ExecutionContext, Workflow, WorkflowEngine};
use caliche::orchestration::{AgentCapability, AgentFlow, AgentMetadata};
use caliche::prelude::ProviderType;
use caliche::{AgentBuilder, prelude::*};
use serde_json::json;
use std::sync::Arc;

#[tokio::main]
async fn main() -> Result<()> {
    dotenvy::dotenv().ok();

    println!("=== Agent Orchestration Examples ===\n");

    // Example 1: Agent Coordinator
    example_coordinator().await?;

    // Example 2: Agent Handoffs
    example_handoffs().await?;

    // Example 3: Orchestration Patterns
    example_orchestration_patterns().await?;

    Ok(())
}

async fn example_coordinator() -> Result<()> {
    println!("Example 1: Agent Coordinator\n");

    let coordinator = AgentCoordinator::new();

    // Create specialized agents with focused system prompts and tools
    let research_agent = AgentBuilder::new()
        .provider(ProviderType::Bedrock)
        .system_prompt("You are a research specialist. Focus on gathering information.".to_string())
        .enable_tool("http")
        .enable_tool("fs")
        .build()
        .await?;

    let research_metadata = AgentMetadata::new(
        "Research Agent".to_string(),
        "Researcher".to_string(),
        vec![AgentCapability::Research],
    );

    coordinator
        .register_agent(research_metadata, research_agent.into_agent())
        .await?;

    let writing_agent = AgentBuilder::new()
        .provider(ProviderType::Bedrock)
        .system_prompt("You are a writing specialist. Focus on clear communication.".to_string())
        .build()
        .await?;

    let writing_metadata = AgentMetadata::new(
        "Writing Agent".to_string(),
        "Writer".to_string(),
        vec![AgentCapability::Writing],
    );

    coordinator
        .register_agent(writing_metadata, writing_agent.into_agent())
        .await?;

    let coding_agent = AgentBuilder::new()
        .provider(ProviderType::Bedrock)
        .system_prompt("You are a coding expert. Focus on technical solutions.".to_string())
        .enable_tool("calculator")
        .build()
        .await?;

    let coding_metadata = AgentMetadata::new(
        "Coding Agent".to_string(),
        "Engineer".to_string(),
        vec![AgentCapability::Coding],
    );

    coordinator
        .register_agent(coding_metadata, coding_agent.into_agent())
        .await?;

    println!("Registered 3 agents: Research, Writing, Coding");

    // Delegate tasks based on capability
    let (agent_id, response, _) = coordinator
        .delegate_task(
            "What are the key features of Rust?",
            Some(&AgentCapability::Research),
        )
        .await?;
    println!("\nTask: Research about Rust");
    println!("Assigned to: {}", agent_id);
    println!(
        "Response: {}...",
        caliche_contracts::text::truncate_utf8(&response, 80)
    );

    let (agent_id, response, _) = coordinator
        .delegate_task(
            "Write a haiku about programming",
            Some(&AgentCapability::Writing),
        )
        .await?;
    println!("\nTask: Write a haiku");
    println!("Assigned to: {}", agent_id);
    println!("Response: {}", response);

    // Coordinator statistics
    let stats = coordinator.get_stats().await;
    println!("\nCoordinator statistics:");
    println!("  Total agents: {}", stats.total_agents);
    println!("  Total capacity: {}", stats.total_capacity);
    println!("  Current load: {}", stats.total_load);

    // List agents by capability
    let research_agents = coordinator
        .find_agents_by_capability(&AgentCapability::Research)
        .await;
    println!("\nResearch-capable agents: {}", research_agents.len());

    Ok(())
}

async fn example_handoffs() -> Result<()> {
    println!("\nExample 2: Agent Handoffs\n");

    // Research agent - specializes in research tasks
    let research_service = AgentBuilder::new()
        .provider(ProviderType::Bedrock)
        .model("us.anthropic.claude-haiku-4-5-20251001-v1:0".to_string())
        .system_prompt(
            "You are a research specialist. When you need to write content, transfer to the writer agent using transfer_to tool."
                .to_string(),
        )
        .enable_tool("transfer_to")
        .build()
        .await?;
    let research_agent = research_service.into_agent();

    // Writer agent - specializes in writing
    let writer_service = AgentBuilder::new()
        .provider(ProviderType::Bedrock)
        .model("us.anthropic.claude-haiku-4-5-20251001-v1:0".to_string())
        .system_prompt(
            "You are a writing specialist. Focus on creating clear, well-structured content."
                .to_string(),
        )
        .build()
        .await?;
    let writer_agent = writer_service.into_agent();

    // Build the flow
    println!("Building agent flow...");
    let flow = AgentFlow::builder()
        .add_node("researcher", research_agent)
        .add_node("writer", writer_agent)
        .set_entry_point("researcher")
        .build()?;

    println!("Flow entry point: {}", flow.entry_point());
    println!("Available nodes: {:?}\n", flow.node_names());

    // Execute the flow
    println!("Executing flow with initial input...");
    let result = flow
        .run("Research the benefits of Rust programming language and write a summary")
        .await?;

    println!("\nFlow Execution Result:");
    println!("Final Response:\n{}", result.final_response);
    println!("\nExecution Log:");
    for (i, step) in result.execution_log.iter().enumerate() {
        println!("\nStep {}: Agent '{}'", i + 1, step.agent_name);
        if let Some(ref response) = step.response {
            println!("  Response: {}", response);
        }
        if let Some(ref handoff) = step.handoff {
            println!(
                "  Handoff to: {} ({})",
                handoff.target_agent, handoff.reason
            );
        }
    }

    // Generate Mermaid diagram
    println!("\nFlow Visualization (Mermaid):");
    println!("{}\n", flow.to_mermaid());

    Ok(())
}

async fn example_orchestration_patterns() -> Result<()> {
    println!("Example 3: Orchestration Patterns\n");

    // Pipeline Pattern
    println!("Pattern 1: Pipeline Pattern");
    let pipeline = PipelinePattern::new("document_analysis")
        .with_description("Analyze a document through multiple stages")
        .add_stage("extract", "Extract key information from the document")
        .add_stage("summarize", "Create a concise summary")
        .add_stage("categorize", "Categorize the document");

    let engine = WorkflowEngine::new();
    let pipeline_arc: Arc<dyn Workflow> = Arc::new(pipeline);
    engine.register_workflow(Arc::clone(&pipeline_arc)).await?;

    let mut context = ExecutionContext::new("document_analysis".to_string());
    context.set_variable(
        "document".to_string(),
        json!({
            "title": "The Future of AI",
            "content": "Artificial Intelligence is rapidly transforming industries..."
        }),
    );

    let _result = pipeline_arc.execute(&engine, &mut context).await?;
    println!("  Pipeline pattern executed\n");

    // Hierarchical Pattern
    println!("Pattern 2: Hierarchical Pattern");
    let hierarchical = HierarchicalPattern::new("research_project")
        .with_description("Multi-level research team")
        .add_layer("senior_researchers", 2, "High-level research")
        .add_layer("specialists", 4, "Technical domains")
        .add_layer("assistants", 8, "Data gathering");

    let hierarchical_arc: Arc<dyn Workflow> = Arc::new(hierarchical);
    engine
        .register_workflow(Arc::clone(&hierarchical_arc))
        .await?;

    let mut context = ExecutionContext::new("research_project".to_string());
    context.set_variable(
        "project_goal".to_string(),
        json!("Investigate quantum computing applications"),
    );

    let _result = hierarchical_arc.execute(&engine, &mut context).await?;
    println!("  Hierarchical pattern executed\n");

    // Supervisor Pattern
    println!("Pattern 3: Supervisor Pattern");
    let supervisor = SupervisorPattern::new(
        "data_collection",
        "Coordinate data collection from multiple sources",
        5,
    )
    .with_description("Supervisor coordinates parallel tasks")
    .with_worker_capability("DataCollection");

    let supervisor_arc: Arc<dyn Workflow> = Arc::new(supervisor);
    engine
        .register_workflow(Arc::clone(&supervisor_arc))
        .await?;

    let mut context = ExecutionContext::new("data_collection".to_string());
    context.set_variable(
        "sources".to_string(),
        json!(["API endpoint 1", "Database A", "File system"]),
    );

    let _result = supervisor_arc.execute(&engine, &mut context).await?;
    println!("  Supervisor pattern executed\n");

    // Debate Pattern
    println!("Pattern 4: Debate Pattern");
    let debate = DebatePattern::new(
        "technology_decision",
        "Which technology stack should we use?",
        4,
        3,
    )
    .with_description("Multi-agent debate to reach consensus")
    .with_moderator_prompt("Synthesize all viewpoints and recommend the best technology stack");

    let debate_arc: Arc<dyn Workflow> = Arc::new(debate);
    engine.register_workflow(Arc::clone(&debate_arc)).await?;

    let mut context = ExecutionContext::new("technology_decision".to_string());
    context.set_variable(
        "options".to_string(),
        json!(["Rust + PostgreSQL", "Go + MongoDB", "Python + Redis"]),
    );

    let _result = debate_arc.execute(&engine, &mut context).await?;
    println!("  Debate pattern executed\n");

    println!("=== Examples Complete ===");
    println!("\nTips:");
    println!("- Use AgentCoordinator for load balancing and capability routing");
    println!("- Use AgentFlow for sequential handoffs between agents");
    println!("- Choose orchestration patterns based on your use case:");
    println!("  - Pipeline: Sequential data transformation");
    println!("  - Hierarchical: Multi-level agent teams");
    println!("  - Supervisor: Coordinator with workers");
    println!("  - Debate: Multi-agent consensus building");

    Ok(())
}
