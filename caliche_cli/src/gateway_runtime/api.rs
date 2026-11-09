use super::process::cleanup_dead;
use super::registry::{load_registry, normalize_cwd, save_registry};
use super::types::{GatewayRuntimeEntry, GatewayRuntimeRole};
use anyhow::{Context, Result, anyhow};
use serde_json::json;
use std::path::Path;

pub(super) fn reqwest_client() -> Result<reqwest::Client> {
    reqwest::Client::builder()
        .build()
        .context("failed to build reqwest client")
}

pub(super) async fn call_gateway_api(
    method: &str,
    endpoint: &str,
    payload: Option<String>,
) -> Result<Vec<u8>> {
    let client = reqwest_client()?;
    let method_parsed = reqwest::Method::from_bytes(method.as_bytes())
        .with_context(|| format!("invalid HTTP method '{method}'"))?;
    let mut request = client.request(method_parsed, endpoint);
    if let Some(body) = payload {
        request = request
            .header(reqwest::header::CONTENT_TYPE, "application/json")
            .body(body);
    }

    let response = request
        .send()
        .await
        .context("failed to execute gateway request")?;
    let status = response.status();
    let body_bytes = response
        .bytes()
        .await
        .context("failed to read gateway response body")?;
    if !status.is_success() {
        let body_text = String::from_utf8_lossy(&body_bytes);
        let stale_hint = if endpoint.contains("/api/gateway/cluster/routes")
            && matches!(status.as_u16(), 404 | 500)
        {
            " (gateway runtime may be stale or unhealthy; restart with '/gateway restart' and check gateway logs)"
        } else {
            ""
        };
        return Err(anyhow!(
            "gateway request failed: HTTP {} {}{}",
            status,
            body_text.trim(),
            stale_hint
        ));
    }
    Ok(body_bytes.to_vec())
}

pub(super) async fn gateway_supports_cluster_routes(port: u16) -> bool {
    let endpoint = format!("http://127.0.0.1:{port}/api/gateway/cluster/routes");
    let Ok(client) = reqwest_client() else {
        // Don't kill a healthy runtime just because client construction failed.
        return true;
    };
    match client.get(&endpoint).send().await {
        Ok(response) => response.status().as_u16() != 404,
        // Transport error (connection refused, etc.) isn't enough to treat the
        // runtime as stale - the caller will recheck via subsequent calls.
        Err(_) => true,
    }
}

pub(super) async fn register_worker_with_main_gateway(entry: &GatewayRuntimeEntry) -> Result<()> {
    if entry.role != GatewayRuntimeRole::Worker.as_str() {
        return Ok(());
    }
    let main_url = entry
        .main_gateway_url
        .as_deref()
        .ok_or_else(|| anyhow!("missing main gateway url for worker runtime"))?;
    let endpoint = format!(
        "{}/api/gateway/cluster/register",
        main_url.trim_end_matches('/')
    );
    let payload = json!({
        "agent_id": entry.agent_id,
        "cwd": entry.cwd,
        "base_url": format!("http://127.0.0.1:{}", entry.port),
        "is_main": false
    });

    let client = reqwest_client()?;
    let response = client
        .post(&endpoint)
        .json(&payload)
        .send()
        .await
        .context("failed to send worker registration request")?;
    if !response.status().is_success() {
        let status = response.status();
        let body = response.text().await.unwrap_or_default();
        return Err(anyhow!(
            "registration request failed: HTTP {} {}",
            status,
            body.trim()
        ));
    }
    Ok(())
}

pub(super) async fn resolve_main_gateway_url(cwd: Option<&Path>) -> Result<String> {
    let cwd = normalize_cwd(cwd)?;
    let mut registry = load_registry().await?;
    cleanup_dead(&mut registry).await;
    save_registry(&registry).await?;

    let current = registry
        .entries
        .iter()
        .find(|e| e.cwd == cwd)
        .ok_or_else(|| anyhow!("no gateway runtime registered for directory: {cwd}"))?;

    if current
        .role
        .eq_ignore_ascii_case(GatewayRuntimeRole::Main.as_str())
    {
        return Ok(format!("http://127.0.0.1:{}", current.port));
    }

    if let Some(main_url) = &current.main_gateway_url {
        return Ok(main_url.clone());
    }

    let main = registry
        .entries
        .iter()
        .find(|e| {
            e.role
                .eq_ignore_ascii_case(GatewayRuntimeRole::Main.as_str())
        })
        .ok_or_else(|| anyhow!("unable to resolve main gateway runtime"))?;
    Ok(format!("http://127.0.0.1:{}", main.port))
}
