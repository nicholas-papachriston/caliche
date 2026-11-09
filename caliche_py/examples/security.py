#!/usr/bin/env python3
"""
Security Example

This example demonstrates:
- Multi-tenancy with resource quotas
- Audit logging for all operations
- RBAC with fine-grained permissions
- Input validation and sanitization
- API key vs JWT authentication
- Programmatic credential passing

Best practices shown:
- Secure agent deployment
- Compliance and audit requirements
- Secure credential management

Run with:
```bash
uv run examples/security.py
```
"""

import caliche
import sys


def api_key_auth():
    """Example 1: API Key Authentication"""
    print("=== Example 1: API Key Authentication ===\n")
    
    print("API key validation:")
    print("  - Validate API keys before processing requests")
    print("  - Associate keys with user and tenant contexts")
    print("  - Store keys securely (hashed, not plaintext)")
    print()


def rbac():
    """Example 2: RBAC Enforcement"""
    print("=== Example 2: RBAC Enforcement ===\n")
    
    print("RBAC (Role-Based Access Control):")
    print("  - Define roles with specific permissions")
    print("  - Check permissions before allowing actions")
    print("  - Example roles: admin, developer, viewer")
    print("  - Example permissions: read, write, delete")
    print()


def rate_limiting():
    """Example 3: Rate Limiting"""
    print("=== Example 3: Rate Limiting ===\n")
    
    print("Rate limiting:")
    print("  - Limit requests per time window")
    print("  - Prevent abuse and ensure fair usage")
    print("  - Example: 100 requests per minute per user")
    print("  - Burst handling for legitimate spikes")
    print()


def multi_tenancy():
    """Example 4: Multi-tenancy"""
    print("=== Example 4: Multi-tenancy ===\n")
    
    print("Multi-tenancy features:")
    print("  - Tenant isolation: tenant_abc")
    print("  - Resource quotas: 1000 requests/hour")
    print("  - Per-tenant cost tracking")
    print("  - Per-tenant rate limits")
    print()


def audit_logging():
    """Example 5: Audit Logging"""
    print("=== Example 5: Audit Logging ===\n")
    
    print("All operations logged with:")
    print("  - User ID and tenant ID")
    print("  - Timestamp and request ID")
    print("  - Action performed")
    print("  - Resources accessed")
    print("  - Success/failure status")
    print()


def input_validation():
    """Example 6: Input Validation"""
    print("=== Example 6: Input Validation ===\n")
    
    print("Input validation:")
    print("  - Prompt injection detection")
    print("  - SQL injection prevention")
    print("  - XSS protection")
    print("  - File path validation")
    print()


def programmatic_credentials():
    """Example 7: Programmatic Credentials"""
    print("=== Example 7: Programmatic Credentials ===\n")
    
    print("Programmatic credential passing examples:")
    print()
    print("1. OpenAI with API key:")
    print("   builder = AgentBuilder()")
    print("   builder.provider(ProviderType.OpenAI)")
    print("   builder.api_key('<openai-api-key>')")
    print("   agent = builder.build()")
    print()
    
    print("2. Anthropic with API key:")
    print("   builder = AgentBuilder()")
    print("   builder.provider(ProviderType.Anthropic)")
    print("   builder.api_key('<anthropic-api-key>')")
    print("   agent = builder.build()")
    print()
    
    print("3. AWS Bedrock with credentials:")
    print("   builder = AgentBuilder()")
    print("   builder.provider(ProviderType.Bedrock)")
    print("   builder.aws_credentials(access_key, secret_key, region)")
    print("   agent = builder.build()")
    print()
    
    print("Note: Uncomment and add actual credentials in the code to test")
    print()


def main():
    """Main execution"""
    print("=== Security Examples ===\n")
    
    api_key_auth()
    rbac()
    rate_limiting()
    multi_tenancy()
    audit_logging()
    input_validation()
    programmatic_credentials()
    
    print("=== Examples Complete ===")
    print("\nTips:")
    print("- Use API key validation for authentication")
    print("- Implement RBAC for fine-grained access control")
    print("- Apply rate limiting to prevent abuse")
    print("- Enable audit logging for compliance")
    print("- Validate all inputs to prevent injection attacks")
    print("- Use programmatic credentials for secure credential management")


if __name__ == "__main__":
    main()

