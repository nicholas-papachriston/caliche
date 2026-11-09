//! MCP remote URL SSRF policy (delegates host blocking to core `HttpPolicy`).

use caliche::plugins::http_policy::HttpPolicy;

/// Validate remote MCP URL host is not private, loopback, or metadata.
///
/// The host check stays fail-closed for loopback. Egress then applies the
/// same MCP allowlist used by the client.
pub fn validate_mcp_remote_url(raw: &str) -> anyhow::Result<()> {
    HttpPolicy::new(None).validate_resolved_url(raw)?;
    caliche::gateway::egress::current_policy()?
        .validate_static(caliche::gateway::egress::EgressClass::Mcp, raw)?;
    Ok(())
}
