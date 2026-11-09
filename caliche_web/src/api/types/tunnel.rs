//! Cloudflare tunnel runtime types.

use serde::{Deserialize, Serialize};

/// Per-tunnel runtime info returned by the server.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TunnelInfo {
    /// Tunnel running status: "stopped" | "starting" | "running" | "failed"
    pub status: String,
    /// Public URL assigned by Cloudflare (populated when running)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub public_url: Option<String>,
    /// Local URL being proxied
    #[serde(skip_serializing_if = "Option::is_none")]
    pub local_url: Option<String>,
    /// Named tunnel identifier (named tunnels only)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    /// ISO-8601 timestamp when the tunnel was started
    #[serde(skip_serializing_if = "Option::is_none")]
    pub started_at: Option<String>,
    /// Error message if status is "failed"
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

/// Overall tunnel status response.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TunnelStatusResponse {
    /// Ad-hoc (quick) tunnel state
    pub adhoc: TunnelInfo,
    /// Named (authenticated) tunnel state
    pub named: TunnelInfo,
    /// Whether `cloudflared login` has been completed
    pub logged_in: bool,
    /// Cloudflare account email, if logged in
    #[serde(skip_serializing_if = "Option::is_none")]
    pub account: Option<String>,
    /// Whether the `cloudflared` binary is present on PATH
    pub cloudflared_available: bool,
}

/// Tunnel login response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TunnelLoginResponse {
    /// URL to open in the browser for OAuth (returned when not yet logged in)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub login_url: Option<String>,
    /// Whether login completed synchronously (rare)
    pub completed: bool,
    /// Status message
    pub message: String,
}
