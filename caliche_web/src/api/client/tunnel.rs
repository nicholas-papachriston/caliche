//! `tunnel` API client methods.

use anyhow::Result;

use super::ApiClient;
use crate::api::types::{TunnelLoginResponse, TunnelStatusResponse};

#[allow(
    dead_code,
    reason = "JSON-RPC mode has no tunnel transport; settings does not call these"
)]
impl ApiClient {
    // ── Cloudflare Tunnel ────────────────────────────────────────────────────

    /// Fetch the current tunnel status (both ad-hoc and named).
    pub async fn get_tunnel_status(&self) -> Result<TunnelStatusResponse> {
        anyhow::bail!("Tunnel operations are not available in JSON-RPC-only mode")
    }

    /// Start an ad-hoc (quick) tunnel for `local_url`.
    pub async fn start_adhoc_tunnel(&self, local_url: &str) -> Result<TunnelStatusResponse> {
        let _ = local_url;
        anyhow::bail!("Tunnel operations are not available in JSON-RPC-only mode")
    }

    /// Stop the running ad-hoc tunnel.
    pub async fn stop_adhoc_tunnel(&self) -> Result<TunnelStatusResponse> {
        anyhow::bail!("Tunnel operations are not available in JSON-RPC-only mode")
    }

    /// Initiate the `cloudflared login` browser OAuth flow.
    pub async fn tunnel_login(&self) -> Result<TunnelLoginResponse> {
        anyhow::bail!("Tunnel operations are not available in JSON-RPC-only mode")
    }

    /// Start a named (authenticated) Cloudflare Tunnel.
    pub async fn start_named_tunnel(
        &self,
        name: &str,
        hostname: Option<&str>,
        local_url: &str,
        credentials_file: Option<&str>,
    ) -> Result<TunnelStatusResponse> {
        let _ = (name, hostname, local_url, credentials_file);
        anyhow::bail!("Tunnel operations are not available in JSON-RPC-only mode")
    }
    /// Stop the running named tunnel.
    pub async fn stop_named_tunnel(&self) -> Result<TunnelStatusResponse> {
        anyhow::bail!("Tunnel operations are not available in JSON-RPC-only mode")
    }
}
