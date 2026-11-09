//! MCP client used by the agent to talk to external MCP servers.
//!
//! The client tries `server/discover` first. When the remote server does not
//! speak that method, the client falls back to `initialize`. Real MCP stdio
//! servers (npm packages, `uvx` scripts, compiled binaries) expect a
//! long-lived connection, so this module keeps one child per [`McpClient`]
//! and multiplexes requests over its stdin/stdout.
//!
//! HTTP (streamable) clients pin DNS per request and do not follow redirects.

use anyhow::{Context, Result, anyhow, bail};
use beef::lean::Cow;
use serde_json::{Value, json};
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, ChildStdin, Command};
use tokio::sync::{Mutex, oneshot};

use crate::mcp::protocol::{
    ClientCapabilities, ClientInfo, InitializeParams, JsonRpcRequest, JsonRpcResponse,
    LATEST_PROTOCOL_VERSION, McpPrompt, McpTool, PromptsListParams, PromptsListResult,
    ResourcesListParams, ResourcesListResult, ResourcesReadParams, ResourcesReadResult,
    ToolsCallParams, ToolsCallResult, ToolsListParams, ToolsListResult,
};
use caliche_mcp::revision::{META_PROTOCOL_VERSION, UNSUPPORTED_PROTOCOL_VERSION};

/// Transport configuration for [`McpClient`].
#[derive(Debug, Clone)]
pub enum McpClientTransportConfig {
    /// Launch a subprocess that speaks JSON-RPC over stdin/stdout. The child
    /// is kept alive for the lifetime of the client and reused across
    /// requests.
    ///
    /// The child gets the scrubbed host environment plus `env`
    /// ([`caliche_contracts::child_env::host_child_env_with`], which
    /// filters a configured `PATH` like the host's), so a server that needs
    /// a key names it in its configured `env`.
    Stdio {
        command: String,
        args: Vec<String>,
        env: Vec<(String, String)>,
    },
    /// Streamable HTTP transport (MCP 2025-03-26+). The client POSTs
    /// JSON-RPC bodies and accepts either `application/json` or
    /// `text/event-stream` responses.
    StreamableHttp { url: String },
}

/// Configuration bundle passed to [`McpClient::new`].
#[derive(Debug, Clone)]
pub struct McpClientConfig {
    pub name: String,
    pub transport: McpClientTransportConfig,
    /// Skip the stdio command allowlist. Only set via
    /// [`crate::agent::builder::AgentBuilder::dangerously_add_stdio_mcp_server`].
    pub skip_stdio_allowlist: bool,
}

/// Long-lived MCP client. Cloning reuses the same underlying session.
#[derive(Clone)]
pub struct McpClient {
    inner: Arc<Inner>,
}

impl std::fmt::Debug for McpClient {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("McpClient")
            .field("name", &self.inner.config.name)
            .field("transport", &self.inner.config.transport)
            .finish()
    }
}

struct Inner {
    config: McpClientConfig,
    next_id: AtomicU64,
    stdio: Mutex<Option<StdioSession>>,
}

struct StdioSession {
    #[expect(
        dead_code,
        reason = "owns the server process so it lives as long as the session"
    )]
    child: Child,
    stdin: ChildStdin,
    pending: Arc<Mutex<HashMap<u64, oneshot::Sender<JsonRpcResponse>>>>,
    initialized: bool,
    modern: bool,
}

/// A server returned `resultType: input_required` and needs another client turn.
#[derive(Debug)]
pub struct McpInputRequired {
    pub result: Value,
}

impl std::fmt::Display for McpInputRequired {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("MCP server requires more input before this request can finish")
    }
}

impl std::error::Error for McpInputRequired {}

impl McpClient {
    pub fn new(config: McpClientConfig) -> Self {
        Self {
            inner: Arc::new(Inner {
                config,
                next_id: AtomicU64::new(1),
                stdio: Mutex::new(None),
            }),
        }
    }

    /// List prompts from the connected server.
    pub async fn list_prompts(&self) -> Result<Vec<McpPrompt>> {
        let params = serde_json::to_value(PromptsListParams { cursor: None })?;
        let value = self.send_request("prompts/list", Some(params)).await?;
        let parsed: PromptsListResult =
            serde_json::from_value(value).context("Invalid prompts/list response payload")?;
        Ok(parsed.prompts)
    }

    pub fn name(&self) -> &str {
        &self.inner.config.name
    }

    /// Eagerly establish the transport. For stdio this spawns the child and
    /// completes the `initialize` + `notifications/initialized` handshake.
    /// For HTTP this validates the URL against the customization SSRF policy.
    /// Calling this is optional; requests will establish the session lazily if needed.
    pub async fn connect(&self) -> Result<()> {
        self.ensure_connect_allowed()?;
        match &self.inner.config.transport {
            McpClientTransportConfig::Stdio { .. } => self.ensure_stdio_session().await,
            McpClientTransportConfig::StreamableHttp { url } => {
                crate::gateway::egress::current_policy()?
                    .validate(crate::gateway::egress::EgressClass::Mcp, url)
                    .map(|_| ())
            }
        }
    }

    pub async fn list_tools(&self) -> Result<Vec<McpTool>> {
        let params = serde_json::to_value(ToolsListParams { cursor: None })?;
        let value = self.send_request("tools/list", Some(params)).await?;
        let parsed: ToolsListResult =
            serde_json::from_value(value).context("Invalid tools/list response payload")?;
        Ok(parsed.tools)
    }

    pub async fn call_tool(&self, name: &str, arguments: Value) -> Result<ToolsCallResult> {
        let tool_key = format!("{}__{name}", self.name());
        if let Some(reason) = crate::mcp::policy_gate::tool_deny_reason(&tool_key, &arguments) {
            anyhow::bail!("MCP tool denied: {reason}");
        }
        let params = ToolsCallParams {
            name: name.to_string(),
            arguments: Some(arguments),
        };
        let value = self
            .send_request("tools/call", Some(serde_json::to_value(params)?))
            .await?;
        serde_json::from_value(value).context("Invalid tools/call response payload")
    }

    pub async fn list_resources(&self) -> Result<ResourcesListResult> {
        let params = serde_json::to_value(ResourcesListParams { cursor: None })?;
        let value = self.send_request("resources/list", Some(params)).await?;
        serde_json::from_value(value).context("Invalid resources/list response payload")
    }

    pub async fn read_resource(&self, uri: impl Into<String>) -> Result<ResourcesReadResult> {
        let params = ResourcesReadParams { uri: uri.into() };
        let value = self
            .send_request("resources/read", Some(serde_json::to_value(params)?))
            .await?;
        serde_json::from_value(value).context("Invalid resources/read response payload")
    }

    fn next_id(&self) -> u64 {
        self.inner.next_id.fetch_add(1, Ordering::Relaxed)
    }

    fn ensure_connect_allowed(&self) -> Result<()> {
        if let Some(reason) = crate::mcp::policy_gate::connect_deny_reason(self.name()) {
            anyhow::bail!("MCP connect denied: {reason}");
        }
        Ok(())
    }

    async fn send_request(&self, method: &str, params: Option<Value>) -> Result<Value> {
        self.ensure_connect_allowed()?;
        match &self.inner.config.transport {
            McpClientTransportConfig::Stdio { .. } => self.send_stdio(method, params).await,
            McpClientTransportConfig::StreamableHttp { url } => {
                self.send_http(url, method, params).await
            }
        }
    }

    async fn send_stdio(&self, method: &str, params: Option<Value>) -> Result<Value> {
        let id = self.next_id();
        let rx = self.enqueue_stdio(id, method, params).await?;
        let response = rx
            .await
            .map_err(|_| anyhow!("MCP stdio server closed before responding to id {id}"))?;
        extract_result(response)
    }

    async fn enqueue_stdio(
        &self,
        id: u64,
        method: &str,
        params: Option<Value>,
    ) -> Result<oneshot::Receiver<JsonRpcResponse>> {
        let mut guard = self.inner.stdio.lock().await;
        if guard.is_none() {
            *guard = Some(self.spawn_stdio_session().await?);
        }
        let Some(session) = guard.as_mut() else {
            bail!("MCP stdio session is missing after spawn");
        };
        let params = if session.modern {
            modern_params(params)
        } else {
            params
        };

        let (tx, rx) = oneshot::channel();
        {
            let mut pending = session.pending.lock().await;
            pending.insert(id, tx);
        }

        let req = JsonRpcRequest {
            jsonrpc: Cow::borrowed(caliche_contracts::rpc::JSONRPC_VERSION),
            method: Cow::owned(method.to_string()),
            params,
            id: Some(json!(id)),
        };
        if let Err(err) = write_line(&mut session.stdin, &req).await {
            // Drop dead session so the next call reconnects.
            let mut pending = session.pending.lock().await;
            pending.remove(&id);
            drop(pending);
            *guard = None;
            return Err(err);
        }

        Ok(rx)
    }

    async fn ensure_stdio_session(&self) -> Result<()> {
        let mut guard = self.inner.stdio.lock().await;
        if guard.is_none() {
            *guard = Some(self.spawn_stdio_session().await?);
        }
        Ok(())
    }

    async fn spawn_stdio_session(&self) -> Result<StdioSession> {
        let (command, args, env) = match &self.inner.config.transport {
            McpClientTransportConfig::Stdio { command, args, env } => {
                (command.as_str(), args.as_slice(), env.as_slice())
            }
            McpClientTransportConfig::StreamableHttp { .. } => {
                return Err(anyhow!("cannot spawn stdio session for HTTP transport"));
            }
        };

        crate::mcp::validate_stdio_command_for_library(command)?;

        let mut child = Command::new(command)
            .args(args)
            .env_clear()
            .envs(caliche_contracts::child_env::host_child_env_with(
                env.iter().map(|(key, value)| (key, value)),
            ))
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::null())
            .kill_on_drop(true)
            .spawn()
            .with_context(|| format!("Failed to spawn MCP stdio server '{command}'"))?;

        let stdin = child
            .stdin
            .take()
            .ok_or_else(|| anyhow!("Failed to open stdin for MCP stdio server"))?;
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| anyhow!("Failed to open stdout for MCP stdio server"))?;

        let pending: Arc<Mutex<HashMap<u64, oneshot::Sender<JsonRpcResponse>>>> =
            Arc::new(Mutex::new(HashMap::new()));

        let reader_pending = Arc::clone(&pending);
        let server_name = self.inner.config.name.clone();
        tokio::spawn(async move {
            let mut lines = BufReader::new(stdout).lines();
            loop {
                match lines.next_line().await {
                    Ok(Some(line)) => {
                        let trimmed = line.trim();
                        if trimmed.is_empty() {
                            continue;
                        }
                        match serde_json::from_str::<JsonRpcResponse>(trimmed) {
                            Ok(response) => {
                                if let Some(id_value) = response.id.as_ref()
                                    && let Some(id) = id_value.as_u64()
                                {
                                    let mut guard = reader_pending.lock().await;
                                    if let Some(sender) = guard.remove(&id) {
                                        let _ = sender.send(response);
                                    }
                                }
                                // No id: notification from server, ignore for now.
                            }
                            Err(err) => {
                                tracing::debug!(
                                    server = %server_name,
                                    error = %err,
                                    line = %trimmed,
                                    "Unparseable JSON-RPC frame from MCP stdio server"
                                );
                            }
                        }
                    }
                    Ok(None) => break,
                    Err(err) => {
                        tracing::debug!(
                            server = %server_name,
                            error = %err,
                            "MCP stdio reader task stopping"
                        );
                        break;
                    }
                }
            }
            // Flush any remaining waiters so callers don't hang forever.
            let mut guard = reader_pending.lock().await;
            guard.clear();
        });

        let mut session = StdioSession {
            child,
            stdin,
            pending,
            initialized: false,
            modern: false,
        };
        self.establish_session(&mut session).await?;
        Ok(session)
    }

    async fn establish_session(&self, session: &mut StdioSession) -> Result<()> {
        let response = self
            .round_trip(
                session,
                self.next_id(),
                "server/discover",
                modern_params(None),
            )
            .await?;
        match plan_after_discover(&response) {
            SessionPlan::Modern => {
                session.modern = true;
                session.initialized = true;
                Ok(())
            }
            SessionPlan::Legacy => self.perform_initialize_handshake(session).await,
            SessionPlan::Unsupported => Err(anyhow!(
                "MCP server rejected protocol version {LATEST_PROTOCOL_VERSION}"
            )),
        }
    }

    async fn round_trip(
        &self,
        session: &mut StdioSession,
        id: u64,
        method: &str,
        params: Option<Value>,
    ) -> Result<JsonRpcResponse> {
        let (tx, rx) = oneshot::channel();
        {
            let mut pending = session.pending.lock().await;
            pending.insert(id, tx);
        }
        let req = JsonRpcRequest {
            jsonrpc: Cow::borrowed(caliche_contracts::rpc::JSONRPC_VERSION),
            method: Cow::owned(method.to_string()),
            params,
            id: Some(json!(id)),
        };
        if let Err(err) = write_line(&mut session.stdin, &req).await {
            let mut pending = session.pending.lock().await;
            pending.remove(&id);
            return Err(err);
        }
        rx.await
            .map_err(|_| anyhow!("MCP stdio server closed before responding to {method}"))
    }

    async fn perform_initialize_handshake(&self, session: &mut StdioSession) -> Result<()> {
        let init_id = self.next_id();
        let (tx, rx) = oneshot::channel();
        {
            let mut pending = session.pending.lock().await;
            pending.insert(init_id, tx);
        }
        let init_req = JsonRpcRequest {
            jsonrpc: Cow::borrowed(caliche_contracts::rpc::JSONRPC_VERSION),
            method: Cow::borrowed("initialize"),
            params: Some(serde_json::to_value(InitializeParams {
                protocol_version: FALLBACK_PROTOCOL_VERSION.to_string(),
                capabilities: ClientCapabilities::default(),
                client_info: Some(ClientInfo {
                    name: Cow::borrowed("caliche"),
                    version: Some(Cow::borrowed(env!("CARGO_PKG_VERSION"))),
                    title: None,
                    icons: None,
                }),
            })?),
            id: Some(json!(init_id)),
        };
        write_line(&mut session.stdin, &init_req)
            .await
            .context("failed to write MCP initialize request")?;

        let response = rx
            .await
            .map_err(|_| anyhow!("MCP stdio server closed during initialize"))?;
        let _ = extract_result(response).context("MCP initialize returned an error")?;

        let initialized = JsonRpcRequest {
            jsonrpc: Cow::borrowed(caliche_contracts::rpc::JSONRPC_VERSION),
            method: Cow::borrowed("notifications/initialized"),
            params: None,
            id: None,
        };
        write_line(&mut session.stdin, &initialized)
            .await
            .context("failed to write notifications/initialized")?;
        session.initialized = true;
        Ok(())
    }

    async fn send_http(&self, url: &str, method: &str, params: Option<Value>) -> Result<Value> {
        let policy = crate::gateway::egress::current_policy()?;
        let validated = policy.validate(crate::gateway::egress::EgressClass::Mcp, url)?;
        let client = validated.pinned_client()?;
        let parsed = validated.url;
        let id = self.next_id();
        let params = modern_params(params);
        let req = JsonRpcRequest {
            jsonrpc: Cow::borrowed(caliche_contracts::rpc::JSONRPC_VERSION),
            method: Cow::owned(method.to_string()),
            params,
            id: Some(json!(id)),
        };

        let response = client
            .post(parsed.as_str())
            .header(
                reqwest::header::ACCEPT,
                "application/json, text/event-stream",
            )
            .header("MCP-Protocol-Version", LATEST_PROTOCOL_VERSION)
            .header("Mcp-Method", method)
            .header(
                "Mcp-Name",
                caliche_mcp::revision::expected_mcp_name(method, req.params.as_ref()),
            )
            .json(&req)
            .send()
            .await
            .with_context(|| format!("Failed MCP HTTP request to {url}"))?;
        let status = response.status();
        let content_type = response
            .headers()
            .get(reqwest::header::CONTENT_TYPE)
            .and_then(|v| v.to_str().ok())
            .unwrap_or_default()
            .to_string();
        let body = response
            .text()
            .await
            .context("Failed to read MCP HTTP response body")?;
        if !status.is_success() {
            return Err(anyhow!(
                "MCP HTTP request failed with status {status}: {body}"
            ));
        }

        let frame = if content_type.contains("text/event-stream") {
            extract_sse_json(&body)
                .ok_or_else(|| anyhow!("MCP HTTP SSE response contained no data frame"))?
        } else {
            body.as_str()
        };

        let parsed: JsonRpcResponse = serde_json::from_str(frame)
            .context("Invalid JSON-RPC response from MCP HTTP endpoint")?;
        extract_result(parsed)
    }
}

async fn write_line(stdin: &mut ChildStdin, req: &JsonRpcRequest) -> Result<()> {
    let line = serde_json::to_string(req)?;
    stdin
        .write_all(line.as_bytes())
        .await
        .context("Failed writing MCP stdio request")?;
    stdin.write_all(b"\n").await?;
    stdin.flush().await?;
    Ok(())
}

fn extract_result(response: JsonRpcResponse) -> Result<Value> {
    if let Some(error) = response.error {
        return Err(anyhow!(
            "MCP request failed ({}): {}",
            error.code,
            error.message
        ));
    }
    let result = response
        .result
        .ok_or_else(|| anyhow!("MCP response missing both result and error"))?;
    if result.get("resultType").and_then(|value| value.as_str()) == Some("input_required") {
        if let Some(reason) = crate::mcp::policy_gate::rejected_input_url(&result) {
            anyhow::bail!("MCP input request URL denied: {reason}");
        }
        return Err(McpInputRequired { result }.into());
    }
    Ok(result)
}

fn modern_params(params: Option<Value>) -> Option<Value> {
    let mut obj = match params {
        Some(Value::Object(map)) => map,
        Some(other) => return Some(other),
        None => serde_json::Map::new(),
    };
    let entry = obj.entry("_meta").or_insert_with(|| json!({}));
    if let Some(meta) = entry.as_object_mut() {
        meta.entry(META_PROTOCOL_VERSION)
            .or_insert_with(|| json!(LATEST_PROTOCOL_VERSION));
        meta.entry("io.modelcontextprotocol/clientInfo")
            .or_insert_with(|| {
                json!({
                    "name": "caliche",
                    "version": env!("CARGO_PKG_VERSION"),
                })
            });
        meta.entry("io.modelcontextprotocol/clientCapabilities")
            .or_insert_with(|| json!({}));
    }
    Some(Value::Object(obj))
}

/// Revision offered to a server that only speaks `initialize`.
const FALLBACK_PROTOCOL_VERSION: &str = "2025-11-25";

enum SessionPlan {
    Modern,
    Legacy,
    Unsupported,
}

fn plan_after_discover(response: &JsonRpcResponse) -> SessionPlan {
    if response
        .error
        .as_ref()
        .is_some_and(|err| err.code == UNSUPPORTED_PROTOCOL_VERSION)
    {
        return SessionPlan::Unsupported;
    }
    if response.error.is_none()
        && response
            .result
            .as_ref()
            .is_some_and(|value| value.get("supportedVersions").is_some())
    {
        return SessionPlan::Modern;
    }
    SessionPlan::Legacy
}

/// Extract the first `data:` payload from a Server-Sent Events body.
fn extract_sse_json(body: &str) -> Option<&str> {
    for line in body.lines() {
        if let Some(rest) = line.strip_prefix("data:") {
            let trimmed = rest.trim_start();
            if !trimmed.is_empty() {
                return Some(trimmed);
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(unix)]
    #[tokio::test]
    async fn stdio_server_gets_its_configured_env_but_no_host_credentials() {
        use crate::discovery::auto::{DiscoveredMcpServer, DiscoveredMcpTransport, SourceScope};
        use crate::testing::planted_credential::{
            PLANTED_CREDENTIAL, PRINT_PLANTED, rerun_with_planted_credential,
        };
        use caliche_contracts::child_env::PATH_VAR;
        if rerun_with_planted_credential(
            "mcp::client::tests::stdio_server_gets_its_configured_env_but_no_host_credentials",
        ) {
            return;
        }
        let dir = tempfile::tempdir().expect("temp dir");
        let out = dir.path().join("env.txt");
        let script = format!(
            "{{ {PRINT_PLANTED}; printf '|%s|%s|%s' \"$CALICHE_MCP_FORWARDED_TOKEN\" \"$CALICHE_MCP_MODE\" \"$PATH\"; }} > '{0}.tmp' && mv '{0}.tmp' '{0}'",
            out.display()
        );
        let discovered = DiscoveredMcpServer {
            name: "probe".to_string(),
            transport: DiscoveredMcpTransport::Stdio {
                command: "sh".to_string(),
                args: vec!["-c".to_string(), script],
                env: vec![
                    (
                        "CALICHE_MCP_FORWARDED_TOKEN".to_string(),
                        format!("${{{PLANTED_CREDENTIAL}}}"),
                    ),
                    ("CALICHE_MCP_MODE".to_string(), "ro".to_string()),
                    (PATH_VAR.to_string(), ".:/usr/bin:/bin".to_string()),
                ],
            },
            source_path: std::path::PathBuf::from("/home/user/.cursor/mcp.json"),
            scope: SourceScope::UserHome,
        };
        let client = McpClient::new(discovered.to_mcp_client_config().expect("client config"));
        // The probe writes its environment and exits without answering the
        // handshake, so only the spawn matters here.
        let _handshake = client.ensure_stdio_session().await;
        let mut written = String::new();
        for _ in 0..100 {
            if let Ok(text) = std::fs::read_to_string(&out) {
                written = text;
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        }
        assert_eq!(written, "unset|planted|ro|/usr/bin:/bin");
    }

    #[test]
    fn extracts_first_sse_data_frame() {
        let body = "event: message\ndata: {\"jsonrpc\":\"2.0\",\"id\":1,\"result\":{}}\n\n";
        let frame = extract_sse_json(body).expect("sse frame");
        assert!(frame.contains("\"result\""));
    }

    #[test]
    fn returns_none_without_data_frame() {
        assert!(extract_sse_json("event: ping\n\n").is_none());
    }

    #[test]
    fn modern_params_keep_the_protocol_version() {
        let params = modern_params(Some(json!({"cursor": "abc"}))).expect("params");
        assert_eq!(
            params["_meta"][META_PROTOCOL_VERSION],
            LATEST_PROTOCOL_VERSION
        );
        assert_eq!(params["cursor"], "abc");
    }

    #[test]
    fn input_required_is_a_typed_error() {
        let err = extract_result(JsonRpcResponse::success(
            Some(json!(1)),
            json!({"resultType": "input_required", "inputRequests": {}}),
        ))
        .expect_err("pending");
        assert!(err.downcast_ref::<McpInputRequired>().is_some());
    }

    #[test]
    fn input_required_metadata_url_is_not_returned() {
        let err = extract_result(JsonRpcResponse::success(
            Some(json!(1)),
            json!({
                "resultType": "input_required",
                "inputRequests": {
                    "1": { "params": { "url": "http://169.254.169.254/latest" } }
                }
            }),
        ))
        .expect_err("denied");
        assert!(err.downcast_ref::<McpInputRequired>().is_none());
        assert!(err.to_string().contains("denied"));
    }

    #[test]
    fn discover_success_selects_the_modern_session() {
        let response =
            JsonRpcResponse::success(Some(json!(1)), json!({"supportedVersions": ["2026-07-28"]}));
        assert!(matches!(
            plan_after_discover(&response),
            SessionPlan::Modern
        ));
    }

    #[test]
    fn discover_method_not_found_falls_back_to_legacy() {
        let response = JsonRpcResponse::error(
            Some(json!(1)),
            crate::mcp::protocol::JsonRpcError::method_not_found("server/discover"),
        );
        assert!(matches!(
            plan_after_discover(&response),
            SessionPlan::Legacy
        ));
    }
}
