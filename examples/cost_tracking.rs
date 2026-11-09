//! Cost Tracking Example
//!
//! This example demonstrates:
//! - Pre-flight cost estimation before execution
//! - Real-time cost tracking during agent execution
//! - Per-tenant cost analytics and breakdowns
//! - Budget alerts and thresholds
//!
//! Best practices shown:
//! - Cost-aware agent configuration
//! - Usage tracking for billing
//! - Cost optimization strategies
//!
//! Run with:
//! ```bash
//! cargo run --example cost_tracking
//! ```

#![expect(
    clippy::print_stdout,
    reason = "examples print their results for the reader"
)]

use anyhow::Result;
use caliche::AgentBuilder;
use caliche::cost::{CostEstimator, CostTracker, UsageRecord};
use caliche::prelude::ProviderType;
use chrono::Utc;

#[tokio::main]
async fn main() -> Result<()> {
    dotenvy::dotenv().ok();

    // Example 1: Pre-flight cost estimation
    let estimator = CostEstimator::new();
    let message = "Explain quantum computing in detail";

    let estimate = estimator.estimate_chat(
        message,
        ProviderType::OpenAI,
        Some("gpt-4o"),
        0,
        &vec!["calculator".to_string()],
    );

    println!("Cost Estimation:");
    println!("  Message: {}", message);
    println!("  Provider: OpenAI");
    println!("  Model: gpt-4o");
    println!("  Estimated cost: ${:.6}", estimate.estimated_cost_usd);
    println!("  Confidence: {:?}", estimate.confidence);
    println!("  Breakdown:");
    println!("    Prompt tokens: {}", estimate.breakdown.prompt_tokens);
    println!(
        "    Estimated completion tokens: {}",
        estimate.breakdown.estimated_completion_tokens
    );
    println!(
        "    Prompt cost: ${:.6}",
        estimate.breakdown.prompt_cost_usd
    );
    println!(
        "    Completion cost: ${:.6}",
        estimate.breakdown.completion_cost_usd
    );

    // Example 2: Real-time cost tracking
    let cost_tracker = CostTracker::new();

    // Set pricing for different models
    cost_tracker.set_pricing("openai/gpt-4o", 0.005, 0.015);
    cost_tracker.set_pricing("anthropic/claude-3-5-sonnet", 0.003, 0.015);
    cost_tracker.set_pricing("openrouter/qwen-plus", 0.001, 0.003);

    // Execute agent request
    let mut agent = AgentBuilder::new()
        .provider(ProviderType::OpenAI)
        .model("gpt-4o".to_string())
        .build()
        .await?;

    let response = agent.chat("What is 2 + 2?").await?;
    let usage = response.usage();

    // Record usage
    let usage_record = UsageRecord {
        timestamp: Utc::now(),
        user_id: Some("user_123".to_string()),
        tenant_id: Some("tenant_abc".to_string()),
        agent_id: "agent_1".to_string(),
        provider: "openai".to_string(),
        model: "gpt-4o".to_string(),
        prompt_tokens: usage.prompt_tokens.unwrap_or(0) as u64,
        completion_tokens: usage.completion_tokens.unwrap_or(0) as u64,
        total_tokens: usage.total_tokens.unwrap_or(0) as u64,
        cost_usd: cost_tracker.calculate_cost(
            "openai",
            "gpt-4o",
            usage.prompt_tokens.unwrap_or(0) as u64,
            usage.completion_tokens.unwrap_or(0) as u64,
        ),
    };

    cost_tracker.record_usage(usage_record.clone());

    println!("\nCost Tracking:");
    println!("  Response: {}", response);
    println!("  Tokens used: {}", usage_record.total_tokens);
    println!("  Cost: ${:.6}", usage_record.cost_usd);

    // Example 3: Per-tenant cost report
    let report = cost_tracker.get_report(Some("tenant_abc"));
    println!("\nTenant Cost Report:");
    println!("  Total tokens: {}", report.total_tokens);
    println!("  Total cost: ${:.6}", report.total_cost_usd);
    println!("  Requests: {}", report.records_count);
    println!("  Breakdown by model:");
    for model_cost in report.breakdown_by_model {
        println!(
            "    {}: ${:.6} ({} tokens, {} requests)",
            model_cost.model,
            model_cost.cost_usd,
            model_cost.total_tokens,
            model_cost.request_count
        );
    }

    // Example 4: Budget alert simulation
    let budget_threshold = 0.01; // $0.01
    if report.total_cost_usd > budget_threshold {
        println!(
            "\n[WARNING] Budget Alert: Tenant has exceeded ${:.2} threshold",
            budget_threshold
        );
    }

    Ok(())
}
