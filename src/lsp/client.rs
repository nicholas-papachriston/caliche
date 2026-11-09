//! LSP JSON-RPC client for communicating with language servers.

use crate::lsp::config::ServerConfig;
use anyhow::{Context, Result, anyhow};
use dashmap::DashMap;
use serde_json::{Value, json};
use std::path::PathBuf;
use std::process::Stdio;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::process::{Child, Command};
use tokio::sync::{mpsc, oneshot};
use tokio::task::JoinHandle;
use tracing::{debug, warn};

type RequestId = u64;

#[derive(Debug)]
enum OutboundMessage {
    Request {
        id: RequestId,
        method: String,
        params: Value,
    },
    Notification {
        method: String,
        params: Value,
    },
}

/// Notification emitted by a language server.
#[derive(Debug, Clone)]
pub struct LspNotification {
    pub method: String,
    pub params: Value,
}

/// LSP client for a single language server.
pub struct LspClient {
    /// Child process.
    process: Child,
    /// Outbound message sender.
    tx: mpsc::Sender<OutboundMessage>,
    /// Pending request responders.
    pending: Arc<DashMap<RequestId, oneshot::Sender<Result<Value, anyhow::Error>>>>,
    /// Server notification receiver.
    notifications: mpsc::Receiver<LspNotification>,
    /// Request ID counter.
    id_counter: AtomicU64,
    /// Workspace root.
    workspace_root: PathBuf,
    request_timeout: Duration,
    writer_handle: JoinHandle<()>,
    reader_handle: JoinHandle<()>,
    stderr_handle: JoinHandle<()>,
    shutdown_started: bool,
}

impl LspClient {
    /// Create a new LSP client and start the server process.
    pub fn new(
        config: &ServerConfig,
        workspace_root: PathBuf,
        request_timeout: Duration,
    ) -> Result<Self> {
        if config.command.trim().is_empty() {
            return Err(anyhow!("LSP command is empty"));
        }

        let mut process = Command::new(&config.command)
            .args(&config.args)
            .env_clear()
            .envs(caliche_contracts::child_env::host_child_env_with(
                &config.env,
            ))
            .current_dir(&workspace_root)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .context(format!("Failed to spawn LSP server: {}", config.command))?;

        let stdin = process.stdin.take().context("Failed to get stdin")?;
        let stdout = process.stdout.take().context("Failed to get stdout")?;
        let stderr = process.stderr.take().context("Failed to get stderr")?;

        let (tx, mut rx) = mpsc::channel::<OutboundMessage>(100);
        let (notification_tx, notifications) = mpsc::channel::<LspNotification>(100);
        let pending: Arc<DashMap<RequestId, oneshot::Sender<Result<Value, anyhow::Error>>>> =
            Arc::new(DashMap::new());

        let writer_handle = {
            let mut stdin = stdin;
            tokio::spawn(async move {
                while let Some(message) = rx.recv().await {
                    let value = match message {
                        OutboundMessage::Request { id, method, params } => json!({
                            "jsonrpc": caliche_contracts::rpc::JSONRPC_VERSION,
                            "id": id,
                            "method": method,
                            "params": params,
                        }),
                        OutboundMessage::Notification { method, params } => json!({
                            "jsonrpc": caliche_contracts::rpc::JSONRPC_VERSION,
                            "method": method,
                            "params": params,
                        }),
                    };
                    if let Err(error) = write_lsp_message(&mut stdin, &value).await {
                        warn!("Failed to write LSP message: {}", error);
                        break;
                    }
                }
            })
        };

        let reader_handle = {
            let pending = Arc::clone(&pending);
            tokio::spawn(async move {
                let mut stdout = stdout;
                loop {
                    match read_lsp_message(&mut stdout).await {
                        Ok(Some(message)) => {
                            if let Some(id) = message.get("id").and_then(Value::as_u64) {
                                if let Some((_, sender)) = pending.remove(&id) {
                                    let response = if let Some(error) = message.get("error") {
                                        Err(anyhow!("LSP server error: {error}"))
                                    } else {
                                        Ok(message.get("result").cloned().unwrap_or(Value::Null))
                                    };
                                    let _ = sender.send(response);
                                }
                            } else if let Some(method) =
                                message.get("method").and_then(Value::as_str)
                            {
                                let params = message.get("params").cloned().unwrap_or(Value::Null);
                                if notification_tx
                                    .send(LspNotification {
                                        method: method.to_string(),
                                        params,
                                    })
                                    .await
                                    .is_err()
                                {
                                    break;
                                }
                            }
                        }
                        Ok(None) => break,
                        Err(error) => {
                            warn!("Failed to read LSP message: {}", error);
                            break;
                        }
                    }
                }
            })
        };

        let stderr_handle = tokio::spawn(async move {
            let mut stderr = stderr;
            let mut buffer = vec![0_u8; 4096];
            loop {
                match stderr.read(&mut buffer).await {
                    Ok(0) => break,
                    Ok(bytes) => {
                        let text = String::from_utf8_lossy(&buffer[..bytes]);
                        debug!("LSP stderr: {}", text.trim_end());
                    }
                    Err(error) => {
                        warn!("Failed to read LSP stderr: {}", error);
                        break;
                    }
                }
            }
        });

        Ok(Self {
            process,
            tx,
            pending,
            notifications,
            id_counter: AtomicU64::new(1),
            workspace_root,
            request_timeout,
            writer_handle,
            reader_handle,
            stderr_handle,
            shutdown_started: false,
        })
    }

    /// Initialize the LSP server.
    pub async fn initialize(&mut self) -> Result<()> {
        let response = self
            .request(
                "initialize",
                json!({
                    "processId": std::process::id(),
                    "rootPath": self.workspace_root.to_str(),
                    "rootUri": file_uri(&self.workspace_root)?.as_str(),
                    "capabilities": {
                        "textDocument": {
                            "synchronization": {
                                "didSave": true,
                                "change": 1,
                            },
                            "completion": {
                                "completionItem": {
                                    "snippetSupport": false,
                                }
                            },
                            "hover": {},
                            "definition": {},
                            "implementation": {},
                            "references": {},
                            "documentSymbol": {},
                            "workspaceSymbol": {},
                        },
                        "workspace": {
                            "workspaceFolders": {
                                "supported": true,
                            }
                        }
                    }
                }),
            )
            .await?;

        debug!("LSP initialized: {:?}", response);

        // Send initialized notification
        self.notify("initialized", json!({})).await?;

        Ok(())
    }

    /// Send a request and wait for response.
    pub async fn request(&mut self, method: &str, params: Value) -> Result<Value> {
        let id = self.id_counter.fetch_add(1, Ordering::SeqCst);
        let (tx, rx) = oneshot::channel();
        self.pending.insert(id, tx);
        self.tx
            .send(OutboundMessage::Request {
                id,
                method: method.to_string(),
                params,
            })
            .await
            .context("failed to queue LSP request")?;

        match tokio::time::timeout(self.request_timeout, rx).await {
            Ok(Ok(Ok(value))) => Ok(value),
            Ok(Ok(Err(error))) => Err(error),
            Ok(Err(_)) => Err(anyhow!("LSP request channel closed for method: {}", method)),
            Err(_) => {
                self.pending.remove(&id);
                Err(anyhow!("LSP request timeout for method: {}", method))
            }
        }
    }

    /// Send a notification (no response expected).
    pub async fn notify(&mut self, method: &str, params: Value) -> Result<()> {
        self.tx
            .send(OutboundMessage::Notification {
                method: method.to_string(),
                params,
            })
            .await
            .context("failed to queue LSP notification")?;
        Ok(())
    }

    /// Receive a server notification, waiting up to `timeout`.
    pub async fn recv_notification(&mut self, timeout: Duration) -> Option<LspNotification> {
        tokio::time::timeout(timeout, self.notifications.recv())
            .await
            .ok()
            .flatten()
    }

    /// Shutdown the language server.
    pub async fn shutdown(&mut self) -> Result<()> {
        if self.shutdown_started {
            return Ok(());
        }
        self.shutdown_started = true;
        let _ = self.request("shutdown", Value::Null).await;
        let _ = self.notify("exit", Value::Null).await;
        if self.process.id().is_some() {
            let _ = tokio::time::timeout(Duration::from_millis(500), self.process.wait()).await;
        }
        if self.process.id().is_some() {
            let _ = self.process.kill().await;
        }
        self.writer_handle.abort();
        self.reader_handle.abort();
        self.stderr_handle.abort();
        Ok(())
    }

    /// Get workspace root.
    pub fn workspace_root(&self) -> &PathBuf {
        &self.workspace_root
    }
}

impl Drop for LspClient {
    fn drop(&mut self) {
        self.writer_handle.abort();
        self.reader_handle.abort();
        self.stderr_handle.abort();
        let _ = self.process.start_kill();
    }
}

async fn write_lsp_message<W>(writer: &mut W, value: &Value) -> Result<()>
where
    W: AsyncWriteExt + Unpin,
{
    let body = serde_json::to_vec(value)?;
    let header = format!("Content-Length: {}\r\n\r\n", body.len());
    writer.write_all(header.as_bytes()).await?;
    writer.write_all(&body).await?;
    writer.flush().await?;
    Ok(())
}

async fn read_lsp_message<R>(reader: &mut R) -> Result<Option<Value>>
where
    R: AsyncReadExt + Unpin,
{
    let mut header = Vec::new();
    let mut byte = [0_u8; 1];
    while !header.ends_with(b"\r\n\r\n") {
        let read = reader.read(&mut byte).await?;
        if read == 0 {
            return Ok(None);
        }
        header.push(byte[0]);
    }

    let header_text = std::str::from_utf8(&header).context("invalid LSP header UTF-8")?;
    let content_length = header_text
        .lines()
        .find_map(|line| line.strip_prefix("Content-Length:"))
        .ok_or_else(|| anyhow!("missing Content-Length header"))?
        .trim()
        .parse::<usize>()
        .context("invalid Content-Length header")?;
    let mut body = vec![0_u8; content_length];
    reader.read_exact(&mut body).await?;
    Ok(Some(serde_json::from_slice(&body)?))
}

pub fn file_uri(path: &std::path::Path) -> Result<url::Url> {
    let abs = if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir()?.join(path)
    };
    url::Url::from_file_path(&abs)
        .map_err(|()| anyhow!("failed to convert path to file URI: {}", abs.display()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(unix)]
    #[tokio::test]
    async fn lsp_server_gets_its_configured_env_but_no_host_credentials() {
        use crate::testing::planted_credential::{PRINT_PLANTED, rerun_with_planted_credential};
        use caliche_contracts::child_env::PATH_VAR;
        if rerun_with_planted_credential(
            "lsp::client::tests::lsp_server_gets_its_configured_env_but_no_host_credentials",
        ) {
            return;
        }
        let dir = tempfile::tempdir().expect("temp dir");
        let out = dir.path().join("env.txt");
        let config = ServerConfig {
            command: "sh".to_string(),
            args: vec![
                "-c".to_string(),
                format!(
                    "{{ {PRINT_PLANTED}; printf '|%s|%s' \"$CALICHE_LSP_CONFIGURED_KEY\" \"$PATH\"; }} > env.tmp && mv env.tmp env.txt"
                ),
            ],
            extensions: vec![".rs".to_string()],
            root_patterns: Vec::new(),
            env: std::collections::HashMap::from([
                (
                    "CALICHE_LSP_CONFIGURED_KEY".to_string(),
                    "configured".to_string(),
                ),
                (PATH_VAR.to_string(), ".:/usr/bin:/bin".to_string()),
            ]),
            enabled: true,
        };
        let _client = LspClient::new(&config, dir.path().to_path_buf(), Duration::from_secs(5))
            .expect("spawn lsp probe");
        let mut written = String::new();
        for _ in 0..100 {
            if let Ok(text) = std::fs::read_to_string(&out) {
                written = text;
                break;
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
        assert_eq!(written, "unset|configured|/usr/bin:/bin");
    }

    #[test]
    fn test_file_uri() {
        let uri = file_uri(std::path::Path::new(".")).expect("path should convert");
        assert_eq!(uri.scheme(), "file");
    }
}
