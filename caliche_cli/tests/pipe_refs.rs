//! `--pipe` reads `@` references in its prompt only with `--pipe-refs`.
//!
//! The binary runs against an in-test model server, in a temp HOME with a
//! cleared environment. The model answers with the user message it got, so
//! stdout shows whether a referenced file reached the request.

#![cfg(target_os = "macos")]

use axum::Json;
use caliche_cli::handlers::PIPE_REFS_OFF;
use serde_json::{Value, json};
use std::io::Write;
use std::path::Path;
use std::process::Stdio;
use std::time::Duration;

const WAIT: Duration = Duration::from_secs(60);

/// Text of the file the prompt references.
const NOTE: &str = "NOTE-BODY-7f3a";

/// Every reply repeats the last user message.
async fn chat_completions(Json(request): Json<Value>) -> Json<Value> {
    let messages = request["messages"].as_array().cloned().unwrap_or_default();
    let user = messages
        .iter()
        .rev()
        .find(|message| message["role"] == "user")
        .map(|message| message["content"].to_string())
        .unwrap_or_default();
    Json(json!({
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "created": 1,
        "model": "fake-model",
        "choices": [{
            "index": 0,
            "message": { "role": "assistant", "content": format!("got: {user}") },
            "finish_reason": "stop",
        }],
        "usage": { "prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2 },
    }))
}

async fn fake_model() -> anyhow::Result<String> {
    let app = axum::Router::new().route(
        "/v1/chat/completions",
        axum::routing::post(chat_completions),
    );
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await?;
    let base_url = format!("http://{}/v1", listener.local_addr()?);
    tokio::spawn(async move { axum::serve(listener, app).await });
    Ok(base_url)
}

/// `echo input | caliche <extra> --pipe "summarize @note.txt"` in
/// `workspace`. Returns the exit code, whether the note reached the model,
/// and stderr.
async fn pipe(
    home: &Path,
    workspace: &Path,
    base_url: &str,
    extra: &[&str],
) -> anyhow::Result<(i32, bool, String)> {
    let mut command = caliche::policy::program_command(env!("CARGO_BIN_EXE_caliche"));
    command
        .args(extra)
        .args(["--pipe", "summarize @note.txt"])
        .current_dir(workspace)
        .env_clear()
        .env("PATH", std::env::var_os("PATH").unwrap_or_default())
        .env("HOME", home)
        .env("AGENT_PROVIDER", "ollama")
        .env("OLLAMA_MODEL", "fake-model")
        .env("OLLAMA_BASE_URL", base_url)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    let run = tokio::task::spawn_blocking(move || {
        let mut child = command.spawn()?;
        child
            .stdin
            .take()
            .ok_or_else(|| std::io::Error::other("no stdin"))?
            .write_all(b"input\n")?;
        child.wait_with_output()
    });
    let output = tokio::time::timeout(WAIT, run).await???;
    let stdout = String::from_utf8_lossy(&output.stdout);
    Ok((
        output.status.code().unwrap_or(-1),
        stdout.contains(NOTE),
        String::from_utf8_lossy(&output.stderr).trim().to_owned(),
    ))
}

#[tokio::test]
async fn pipe_reads_at_references_only_with_pipe_refs() -> anyhow::Result<()> {
    let dir = tempfile::tempdir()?;
    let base = dir.path().canonicalize()?;
    let home = base.join("home");
    let workspace = base.join("workspace");
    std::fs::create_dir_all(&home)?;
    std::fs::create_dir_all(&workspace)?;
    std::fs::write(workspace.join("note.txt"), NOTE)?;
    let base_url = fake_model().await?;

    let (code, read, stderr) = pipe(&home, &workspace, &base_url, &[]).await?;
    assert_eq!(
        (
            code,
            read,
            stderr.ends_with(&format!("{PIPE_REFS_OFF}: @note.txt"))
        ),
        (1, false, true),
        "stderr={stderr}"
    );

    let (code, read, stderr) = pipe(&home, &workspace, &base_url, &["--pipe-refs"]).await?;
    assert_eq!((code, read), (0, true), "stderr={stderr}");
    Ok(())
}
