/**
 * Security Example
 *
 * This example demonstrates:
 * - Multi-tenancy with resource quotas
 * - Audit logging for all operations
 * - RBAC with fine-grained permissions
 * - Input validation and sanitization
 * - API key vs JWT authentication
 * - Programmatic credential passing
 *
 * Best practices shown:
 * - Secure agent deployment
 * - Compliance and audit requirements
 * - Secure credential management
 *
 * Run with:
 * ```bash
 * bun run examples/security.ts
 * */

function apiKeyAuth(): void {
  console.log("=== Example 1: API Key Authentication ===\n");

  console.log("API key validation:");
  console.log("  - Validate API keys before processing requests");
  console.log("  - Associate keys with user and tenant contexts");
  console.log("  - Store keys securely (hashed, not plaintext)");
  console.log();
}

function rbac(): void {
  console.log("=== Example 2: RBAC Enforcement ===\n");

  console.log("RBAC (Role-Based Access Control):");
  console.log("  - Define roles with specific permissions");
  console.log("  - Check permissions before allowing actions");
  console.log("  - Example roles: admin, developer, viewer");
  console.log("  - Example permissions: read, write, delete");
  console.log();
}

function rateLimiting(): void {
  console.log("=== Example 3: Rate Limiting ===\n");

  console.log("Rate limiting:");
  console.log("  - Limit requests per time window");
  console.log("  - Prevent abuse and ensure fair usage");
  console.log("  - Example: 100 requests per minute per user");
  console.log("  - Burst handling for legitimate spikes");
  console.log();
}

function multiTenancy(): void {
  console.log("=== Example 4: Multi-tenancy ===\n");

  console.log("Multi-tenancy features:");
  console.log("  - Tenant isolation: tenant_abc");
  console.log("  - Resource quotas: 1000 requests/hour");
  console.log("  - Per-tenant cost tracking");
  console.log("  - Per-tenant rate limits");
  console.log();
}

function auditLogging(): void {
  console.log("=== Example 5: Audit Logging ===\n");

  console.log("All operations logged with:");
  console.log("  - User ID and tenant ID");
  console.log("  - Timestamp and request ID");
  console.log("  - Action performed");
  console.log("  - Resources accessed");
  console.log("  - Success/failure status");
  console.log();
}

function inputValidation(): void {
  console.log("=== Example 6: Input Validation ===\n");

  console.log("Input validation:");
  console.log("  - Prompt injection detection");
  console.log("  - SQL injection prevention");
  console.log("  - XSS protection");
  console.log("  - File path validation");
  console.log();
}

function programmaticCredentials(): void {
  console.log("=== Example 7: Programmatic Credentials ===\n");

  console.log("Programmatic credential passing examples:");
  console.log();
  console.log("1. OpenAI with API key:");
  console.log("   const builder = new AgentBuilder()");
  console.log("   builder.provider('openai')");
  console.log("   builder.apiKey('<openai-api-key>')");
  console.log("   const agent = await builder.build()");
  console.log();

  console.log("2. Anthropic with API key:");
  console.log("   const builder = new AgentBuilder()");
  console.log("   builder.provider('anthropic')");
  console.log("   builder.apiKey('<anthropic-api-key>')");
  console.log("   const agent = await builder.build()");
  console.log();

  console.log("3. AWS Bedrock with credentials:");
  console.log("   const builder = new AgentBuilder()");
  console.log("   builder.provider('bedrock')");
  console.log("   builder.awsCredentials(accessKey, secretKey, region)");
  console.log("   const agent = await builder.build()");
  console.log();

  console.log("Note: Uncomment and add actual credentials in the code to test");
  console.log();
}

async function main(): Promise<void> {
  console.log("=== Security Examples ===\n");

  apiKeyAuth();
  rbac();
  rateLimiting();
  multiTenancy();
  auditLogging();
  inputValidation();
  programmaticCredentials();

  console.log("=== Examples Complete ===");
  console.log("\nTips:");
  console.log("- Use API key validation for authentication");
  console.log("- Implement RBAC for fine-grained access control");
  console.log("- Apply rate limiting to prevent abuse");
  console.log("- Enable audit logging for compliance");
  console.log("- Validate all inputs to prevent injection attacks");
  console.log("- Use programmatic credentials for secure credential management");
}

main().catch(console.error);
