//! Line loop for ACP over stdio.
//!
//! `session/prompt` runs on its own task so the next stdin line, including
//! `session/cancel` and the client's `session/request_permission` result, can
//! be handled while the turn is still producing updates.

use crate::acp::outbound::AcpFrameSink;
use crate::acp::server::AcpServer;
use anyhow::Result;
use caliche_contracts::acp::methods;
use serde_json::Value;
use std::sync::Arc;
use tokio::io::{AsyncBufRead, AsyncBufReadExt};
use tokio::task::JoinHandle;

pub async fn serve_acp(
    server: Arc<AcpServer>,
    reader: impl AsyncBufRead + Unpin,
    frames: Arc<dyn AcpFrameSink>,
) -> Result<()> {
    let mut lines = reader.lines();
    let mut prompt_task: Option<JoinHandle<Result<()>>> = None;
    while let Some(line) = lines.next_line().await? {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let value = match serde_json::from_str::<Value>(trimmed) {
            Ok(value) => value,
            Err(err) => {
                frames
                    .write_frame(&crate::acp::server::parse_error_frame(&err))
                    .map_err(|write_err| anyhow::anyhow!("write ACP parse error: {write_err}"))?;
                continue;
            }
        };
        if is_prompt_request(&value) {
            if let Some(task) = prompt_task.take() {
                finish_prompt(task).await?;
            }
            // Queue before spawn so a cancel already buffered on stdin sticks
            // to this prompt even if the task has not claimed its token yet.
            if let Some(session_id) = value.pointer("/params/sessionId").and_then(Value::as_str) {
                server.queue_prompt(session_id);
            }
            let server = Arc::clone(&server);
            let frames = Arc::clone(&frames);
            prompt_task = Some(tokio::spawn(async move {
                Box::pin(server.handle_json_value(value, &frames)).await
            }));
        } else {
            server.handle_json_value(value, &frames).await?;
        }
    }
    server.cancel_all();
    if let Some(task) = prompt_task {
        finish_prompt(task).await?;
    }
    Ok(())
}

async fn finish_prompt(task: JoinHandle<Result<()>>) -> Result<()> {
    match task.await {
        Ok(result) => result,
        Err(err) => anyhow::bail!("acp prompt task failed: {err}"),
    }
}

fn is_prompt_request(value: &Value) -> bool {
    value.get("method").and_then(Value::as_str) == Some(methods::SESSION_PROMPT)
}
