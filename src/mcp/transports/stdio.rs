//! Standard I/O transport for MCP (line-buffered JSON-RPC over stdin/stdout).

use crate::mcp::handler::McpHandler;
use crate::mcp::protocol::{JsonRpcRequest, JsonRpcResponse};
use anyhow::Result;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};

/// Run MCP server over stdio transport
pub async fn run_stdio_transport(mut handler: McpHandler) -> Result<()> {
    let stdin = tokio::io::stdin();
    let mut stdout = tokio::io::stdout();
    let mut reader = BufReader::new(stdin);
    let mut line = String::new();

    loop {
        line.clear();
        let bytes_read = reader.read_line(&mut line).await?;

        if bytes_read == 0 {
            // EOF reached
            break;
        }

        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }

        // Parse JSON-RPC request
        let request: JsonRpcRequest = match serde_json::from_str(trimmed) {
            Ok(req) => req,
            Err(e) => {
                // Send parse error response
                let error_response = JsonRpcResponse::error(
                    None,
                    crate::mcp::protocol::JsonRpcError::parse_error(Some(
                        serde_json::json!({"error": e.to_string()}),
                    )),
                );
                let response_json = serde_json::to_string(&error_response)?;
                stdout.write_all(response_json.as_bytes()).await?;
                stdout.write_all(b"\n").await?;
                stdout.flush().await?;
                continue;
            }
        };

        // Handle request
        if let Some(response) = handler.handle_request(request).await {
            // Only send response if one is returned (notifications don't get responses)
            let response_json = serde_json::to_string(&response)?;
            stdout.write_all(response_json.as_bytes()).await?;
            stdout.write_all(b"\n").await?;
            stdout.flush().await?;
        }
    }

    Ok(())
}
