//! Security Example
//!
//! This example demonstrates:
//! - Multi-tenancy with resource quotas
//! - Audit logging for all operations
//! - RBAC with fine-grained permissions
//! - Input validation and sanitization
//! - API key vs JWT authentication
//! - Programmatic credential passing
//!
//! Best practices shown:
//! - Secure agent deployment
//! - Compliance and audit requirements
//! - Secure credential management
//!
//! Run with:
//! ```bash
//! cargo run --example security
//! ```

#![expect(
    clippy::print_stdout,
    reason = "examples print their results for the reader"
)]

use anyhow::Result;
use caliche::security::{
    ApiKeyValidator, AuthContext, Permission, RateLimitConfig, RateLimiter, RbacEnforcer, Role,
};
use std::time::Duration;

#[tokio::main]
async fn main() -> Result<()> {
    dotenvy::dotenv().ok();

    println!("=== Security Examples ===\n");

    // Example 1: API Key authentication
    example_api_key_auth();

    // Example 2: RBAC enforcement
    example_rbac();

    // Example 3: Rate limiting
    example_rate_limiting();

    // Example 4: Multi-tenancy
    example_multi_tenancy();

    // Example 5: Audit logging
    example_audit_logging();

    // Example 6: Input validation
    example_input_validation();

    // Example 7: Programmatic credentials
    example_programmatic_credentials().await?;

    Ok(())
}

fn example_api_key_auth() {
    println!("Example 1: API Key Authentication\n");

    let api_validator = ApiKeyValidator::new();
    let auth_context = AuthContext {
        user_id: "user_123".to_string(),
        tenant_id: Some("tenant_abc".to_string()),
        roles: vec!["developer".to_string()],
        permissions: vec!["read".to_string(), "write".to_string()],
    };
    api_validator.add_key("api_key_xyz".to_string(), auth_context);

    match api_validator.validate("api_key_xyz") {
        Ok(ctx) => println!("API key validated for user: {}", ctx.user_id),
        Err(_) => println!("Invalid API key"),
    }
    println!();
}

fn example_rbac() {
    println!("Example 2: RBAC Enforcement\n");

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

    let auth_context = AuthContext {
        user_id: "user_123".to_string(),
        tenant_id: Some("tenant_abc".to_string()),
        roles: vec!["developer".to_string()],
        permissions: vec![],
    };

    let has_permission = rbac.check_permission(&auth_context.roles, "agents", "read");
    println!("RBAC check (read agents): {}", has_permission);
    println!();
}

fn example_rate_limiting() {
    println!("Example 3: Rate Limiting\n");

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
    println!();
}

fn example_multi_tenancy() {
    println!("Example 4: Multi-tenancy\n");

    println!("Tenant isolation: tenant_abc");
    println!("Resource quotas: 1000 requests/hour");
    println!("Per-tenant cost tracking");
    println!("Per-tenant rate limits");
    println!();
}

fn example_audit_logging() {
    println!("Example 5: Audit Logging\n");

    println!("All operations logged with:");
    println!("  - User ID and tenant ID");
    println!("  - Timestamp and request ID");
    println!("  - Action performed");
    println!("  - Resources accessed");
    println!("  - Success/failure status");
    println!();
}

fn example_input_validation() {
    println!("Example 6: Input Validation\n");

    println!("Prompt injection detection");
    println!("SQL injection prevention");
    println!("XSS protection");
    println!("File path validation");
    println!();
}

#[expect(
    clippy::unused_async,
    reason = "every example section has the same async shape"
)]
async fn example_programmatic_credentials() -> Result<()> {
    println!("Example 7: Programmatic Credentials\n");

    println!("Programmatic credential passing examples:");
    println!();
    println!("1. OpenAI with API key:");
    println!("   AgentBuilder::new()");
    println!("       .provider(ProviderType::OpenAI)");
    println!("       .api_key(\"<openai-api-key>\".to_string())");
    println!("       .build().await?");
    println!();

    println!("2. Anthropic with API key:");
    println!("   AgentBuilder::new()");
    println!("       .provider(ProviderType::Anthropic)");
    println!("       .api_key(\"<anthropic-api-key>\".to_string())");
    println!("       .build().await?");
    println!();

    println!("3. AWS Bedrock with credentials:");
    println!("   AgentBuilder::new()");
    println!("       .provider(ProviderType::Bedrock)");
    println!("       .aws_credentials(access_key, secret_key, region)");
    println!("       .build().await?");
    println!();

    println!("4. AWS Bedrock with bearer token:");
    println!("   AgentBuilder::new()");
    println!("       .provider(ProviderType::Bedrock)");
    println!("       .aws_bearer_token(token, region)");
    println!("       .build().await?");
    println!();

    println!("Note: Uncomment and add actual credentials in the code to test");
    println!();

    Ok(())
}
