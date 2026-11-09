//! Workflows Example
//!
//! This example demonstrates:
//! - Creating workflows with WorkflowBuilder
//! - Sequential and parallel agent execution
//! - Built-in workflow patterns (Supervisor, Pipeline, etc.)
//! - Workflow checkpoints and recovery
//! - Cyclic workflows with state machine execution
//! - Human-in-the-Loop (HITL) workflows
//!
//! Best practices shown:
//! - Workflow pattern selection
//! - Checkpoint-based recovery for long-running workflows
//! - Conditional transitions and loops
//!
//! Run with:
//! ```bash
//! cargo run --example workflows
//! ```

#![expect(
    clippy::print_stdout,
    reason = "examples print their results for the reader"
)]

use anyhow::Result;
use caliche::orchestration::{
    DebatePattern, HierarchicalPattern, PipelinePattern, SupervisorPattern, Workflow,
    WorkflowBuilder, WorkflowEngine,
};
use serde_json::json;
use std::sync::Arc;

#[tokio::main]
async fn main() -> Result<()> {
    dotenvy::dotenv().ok();

    println!("=== Workflows Examples ===\n");

    // Example 1: Basic Workflows
    example_basic_workflows().await?;

    // Example 2: Cyclic Workflows
    example_cyclic_workflows().await?;

    // Example 3: HITL Workflows
    example_hitl_workflows().await?;

    // Example 4: Workflow Recovery
    example_workflow_recovery().await?;

    Ok(())
}

async fn example_basic_workflows() -> Result<()> {
    println!("Example 1: Basic Workflows\n");

    let engine = WorkflowEngine::new();

    // Sequential Pipeline
    let workflow = WorkflowBuilder::new("research_pipeline")
        .with_description("Research and write workflow")
        .add_agent(
            "researcher",
            "Research the topic: Rust programming language",
        )
        .then("writer", "Write a brief summary based on the research")
        .then("reviewer", "Review the summary for accuracy")
        .build();

    engine.register_workflow(Arc::new(workflow)).await?;
    println!("Sequential workflow registered: researcher -> writer -> reviewer");

    // Supervisor Pattern
    let supervisor_pattern = SupervisorPattern::new(
        "data_processing",
        "Coordinate data processing across workers",
        3,
    )
    .with_worker_capability("DataProcessing")
    .with_description("Process large dataset in parallel");

    engine
        .register_workflow(Arc::new(supervisor_pattern))
        .await?;
    println!("Supervisor pattern registered: 1 supervisor, 3 workers");

    // Hierarchical Team Pattern
    let hierarchical = HierarchicalPattern::new("enterprise_workflow")
        .add_layer("executives", 2, "Strategic planning")
        .add_layer("managers", 4, "Tactical execution")
        .add_layer("workers", 10, "Task completion");

    engine.register_workflow(Arc::new(hierarchical)).await?;
    println!("Hierarchical workflow registered: 2 executives -> 4 managers -> 10 workers");

    // Debate Pattern
    let debate = DebatePattern::new(
        "architecture_debate",
        "Should we use microservices architecture?",
        4,
        3,
    )
    .with_moderator_prompt("Synthesize all arguments and provide final recommendation");

    engine.register_workflow(Arc::new(debate)).await?;
    println!("Debate pattern registered: 4 debaters, 3 rounds, with moderator");

    // Pipeline Pattern
    let pipeline = PipelinePattern::new("etl_pipeline")
        .add_stage("extract", "Extract data from sources")
        .add_stage("transform", "Clean and transform data")
        .add_stage("validate", "Validate data quality")
        .add_stage("load", "Load into destination");

    engine.register_workflow(Arc::new(pipeline)).await?;
    println!("Pipeline pattern registered: extract -> transform -> validate -> load");

    println!("\nRegistered workflows:");
    for workflow_id in engine.list_workflows() {
        println!("  - {}", workflow_id);
    }
    println!();

    Ok(())
}

async fn example_cyclic_workflows() -> Result<()> {
    println!("Example 2: Cyclic Workflows\n");

    let engine = WorkflowEngine::new();

    // Review/Revise Loop
    println!("Review/Revise Loop Workflow");
    let review_workflow = WorkflowBuilder::new("review_revise_loop")
        .with_description("Document review workflow that loops until approved")
        .add_agent("draft", "Draft the document")
        .add_agent("review", "Review the document and provide verdict")
        .add_agent("revise", "Revise based on feedback")
        .add_agent("approved", "Document is approved")
        .with_entry_point("draft")
        .add_edge_with_condition("draft", "review", "true")
        .add_edge_with_condition("review", "revise", "verdict == 'revise'")
        .add_edge_with_condition("review", "approved", "verdict == 'approved'")
        .add_edge_with_condition("revise", "draft", "true")
        .build();

    println!("  Structure: draft -> review -> [revise -> draft (loop) | approved (end)]");
    Workflow::validate(&review_workflow)?;
    println!("  Validation passed\n");

    // Counter Loop
    println!("Counter Loop Workflow");
    let counter_workflow = WorkflowBuilder::new("counter_loop")
        .with_description("Counter that increments until reaching threshold")
        .add_agent("init", "Initialize counter to 0")
        .add_agent("increment", "Increment counter by 1")
        .add_agent("check", "Check if counter >= 5")
        .with_entry_point("init")
        .add_edge_with_condition("init", "increment", "true")
        .add_edge_with_condition("increment", "check", "true")
        .add_edge_with_condition("check", "increment", "count < 5")
        .build();

    println!("  Structure: init -> increment -> check -> [increment (if count < 5) | end]");
    Workflow::validate(&counter_workflow)?;
    println!("  Validation passed\n");

    // Retry Loop
    println!("Retry Loop with Max Attempts");
    let retry_workflow = WorkflowBuilder::new("retry_loop")
        .with_description("Retry operation until success or max attempts")
        .add_agent("attempt", "Attempt the operation")
        .add_agent("check_success", "Check if operation succeeded")
        .add_agent("check_attempts", "Check attempt count")
        .add_agent("success", "Operation succeeded")
        .add_agent("failure", "Max attempts reached")
        .with_entry_point("attempt")
        .add_edge_with_condition("attempt", "check_success", "true")
        .add_edge_with_condition("check_success", "success", "success == true")
        .add_edge_with_condition("check_success", "check_attempts", "success == false")
        .add_edge_with_condition("check_attempts", "attempt", "attempts < 3")
        .add_edge_with_condition("check_attempts", "failure", "attempts >= 3")
        .build();

    println!(
        "  Structure: attempt -> check_success -> [success | check_attempts -> attempt (if < 3) | failure]"
    );
    Workflow::validate(&retry_workflow)?;
    println!("  Validation passed\n");

    engine.register_workflow(Arc::new(review_workflow)).await?;
    engine.register_workflow(Arc::new(counter_workflow)).await?;
    engine.register_workflow(Arc::new(retry_workflow)).await?;

    println!("Key Differences:");
    println!("• DAG Mode (no entry_point): No cycles, topological order");
    println!("• State Machine Mode (with entry_point): Cycles allowed, conditional transitions\n");

    Ok(())
}

async fn example_hitl_workflows() -> Result<()> {
    println!("Example 3: Human-in-the-Loop (HITL) Workflows\n");

    let engine = WorkflowEngine::new();

    // Document Approval Workflow
    println!("Document Approval Workflow");
    let approval_workflow = WorkflowBuilder::new("document_approval")
        .with_description("Document preparation and approval workflow")
        .add_agent("prepare", "Prepare the document for review")
        .human_interrupt(
            "approval",
            "Please review the document and provide approval decision",
            Some(vec!["approval".to_string(), "notes".to_string()]),
        )
        .add_agent("finalize", "Finalize the approved document")
        .with_entry_point("prepare")
        .add_edge_with_condition("prepare", "approval", "true")
        .add_edge_with_condition("approval", "finalize", "interrupt_approval == 'approved'")
        .build();

    println!("  Structure: prepare -> approval (HITL) -> finalize");
    Workflow::validate(&approval_workflow)?;
    println!("  Validation passed\n");

    // Payment Approval Workflow
    println!("Payment Approval Workflow");
    let payment_workflow = WorkflowBuilder::new("payment_approval")
        .with_description("Payment processing with approval gate")
        .add_agent("validate", "Validate payment details")
        .human_interrupt(
            "manager_approval",
            "Payment exceeds $1000. Manager approval required.",
            Some(vec!["approved".to_string(), "reason".to_string()]),
        )
        .add_agent("process", "Process the approved payment")
        .add_agent("notify", "Send payment confirmation")
        .with_entry_point("validate")
        .add_edge_with_condition("validate", "manager_approval", "amount > 1000")
        .add_edge_with_condition("validate", "process", "amount <= 1000")
        .add_edge_with_condition("manager_approval", "process", "interrupt_approved == 'yes'")
        .add_edge_with_condition("process", "notify", "true")
        .build();

    println!("  Structure: validate -> [manager_approval (HITL) | process] -> notify");
    Workflow::validate(&payment_workflow)?;
    println!("  Validation passed\n");

    engine
        .register_workflow(Arc::new(approval_workflow))
        .await?;
    engine.register_workflow(Arc::new(payment_workflow)).await?;

    println!("HITL Workflow Management:");
    println!("1. Workflow pauses at human_interrupt node");
    println!("2. List pending interrupts: engine.list_pending_interrupts().await");
    println!(
        "3. Resume with feedback: engine.resume_execution(interrupt_id, feedback_map).await?\n"
    );

    Ok(())
}

#[expect(
    clippy::unused_async,
    reason = "every example section has the same async shape"
)]
async fn example_workflow_recovery() -> Result<()> {
    println!("Example 4: Workflow Recovery\n");
    println!(
        "Persisted workflow state recovery has been removed from the core crate; \
         plug your own store (Postgres, SQLite, agentfs, etc.) behind \
         `WorkflowEngine` using its checkpoint hooks to integrate recovery."
    );
    // Suppress the `unused` warning on `json!` while demonstrating the builder
    // surface is unaffected by the removal.
    let _ = json!({"example": "workflow_recovery"});
    println!();
    Ok(())
}
