//! Enterprise Features Example
//!
//! This example demonstrates:
//! - LLM response caching
//! - Rate limiting
//! - Cost tracking
//! - Security features (authentication, RBAC)
//! - Observability (metrics, health checks)
//!
//! Best practices shown:
//! - Production-ready agent configuration
//! - Cost management and monitoring
//! - Security and compliance
//!
//! Run with:
//! ```bash
//! cargo run --example enterprise
//! ```

#![expect(
    clippy::print_stdout,
    reason = "examples print their results for the reader"
)]

use caliche::caching::{CachePolicy, LlmCache};
use caliche::cost::{CostTracker, UsageRecord};
use caliche::health::HealthChecker;
use caliche::observability::MetricsRecorder;
use caliche::security::{
    ApiKeyValidator, AuthContext, Permission, RateLimitConfig, RateLimiter, RbacEnforcer, Role,
};
use chrono::Utc;
use std::time::Duration;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    dotenvy::dotenv().ok();

    // Example 1: LLM Response Caching
    let llm_cache = LlmCache::new(1000, 3600, CachePolicy::ExactMatch);

    let stats = llm_cache.get_stats().await?;
    println!(
        "Cache stats - hits: {}, misses: {}, size: {}",
        stats.hits, stats.misses, stats.size
    );

    // Example 2: Rate Limiting
    let rate_limiter = RateLimiter::new(RateLimitConfig {
        requests_per_window: 100,
        window_duration: Duration::from_secs(60),
        burst_size: 10,
    });

    let user_id = "user_123";
    for i in 1..=5 {
        let allowed = rate_limiter.check_rate_limit(user_id);
        println!(
            "Request {}: {}",
            i,
            if allowed { "Allowed" } else { "Rate limited" }
        );
    }

    // Example 3: Cost Tracking
    let cost_tracker = CostTracker::new();
    cost_tracker.set_pricing("openai/gpt-4o", 0.005, 0.015);
    cost_tracker.set_pricing("anthropic/claude-3-5-sonnet", 0.003, 0.015);

    let usage = UsageRecord {
        timestamp: Utc::now(),
        user_id: Some("user_123".to_string()),
        tenant_id: Some("tenant_abc".to_string()),
        agent_id: "agent_1".to_string(),
        provider: "openai".to_string(),
        model: "gpt-4o".to_string(),
        prompt_tokens: 1000,
        completion_tokens: 500,
        total_tokens: 1500,
        cost_usd: cost_tracker.calculate_cost("openai", "gpt-4o", 1000, 500),
    };

    cost_tracker.record_usage(usage);

    let report = cost_tracker.get_report(Some("tenant_abc"));
    println!("\nCost Report:");
    println!("  Total tokens: {}", report.total_tokens);
    println!("  Total cost: ${:.4}", report.total_cost_usd);
    println!("  Requests: {}", report.records_count);

    // Example 4: Authentication & Authorization
    let api_validator = ApiKeyValidator::new();
    let auth_context = AuthContext {
        user_id: "user_123".to_string(),
        tenant_id: Some("tenant_abc".to_string()),
        roles: vec!["developer".to_string()],
        permissions: vec!["read".to_string(), "write".to_string()],
    };
    api_validator.add_key("api_key_xyz".to_string(), auth_context.clone());

    match api_validator.validate("api_key_xyz") {
        Ok(ctx) => println!("\nAPI key validated for user: {}", ctx.user_id),
        Err(_) => println!("\nInvalid API key"),
    }

    let rbac = RbacEnforcer::new();
    rbac.add_role(Role {
        name: "developer".to_string(),
        permissions: vec![
            Permission {
                resource: "agents".to_string(),
                action: "read".to_string(),
            },
            Permission {
                resource: "agents".to_string(),
                action: "write".to_string(),
            },
        ],
    });

    let has_permission = rbac.check_permission(&auth_context.roles, "agents", "read");
    println!("RBAC check (read agents): {}", has_permission);

    // Example 5: Observability
    let metrics = MetricsRecorder::new()?;
    metrics.request_total.inc();
    metrics.token_usage.inc_by(1500);
    metrics.cache_hits.inc_by(10);
    metrics.cache_misses.inc_by(5);
    metrics.active_workflows.set(3);
    metrics.agent_pool_size.set(5);

    println!("\nMetrics recorded:");
    println!("  Token usage: 1500");
    println!("  Cache hit rate: 66.7%");
    println!("  Active workflows: 3");

    let health_checker = HealthChecker::new();
    let liveness = health_checker.liveness();
    let readiness = health_checker.readiness();

    println!("\nHealth checks:");
    println!("  Liveness: {:?}", liveness.status);
    println!("  Readiness: {:?}", readiness.status);

    Ok(())
}
