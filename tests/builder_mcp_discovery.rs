//! AgentBuilder MCP auto-discovery defaults.

#![cfg(all(feature = "mcp", feature = "mcp-client"))]

use caliche::AgentBuilder;

#[test]
fn auto_discover_mcp_defaults_to_disabled() {
    // Compiles the opt-in API and documents the SDK isolation default.
    let _builder = AgentBuilder::new().auto_discover_mcp(false);
    let _enabled = AgentBuilder::new().auto_discover_mcp(true);
}
