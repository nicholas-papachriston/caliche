//! `--pipe` runs tools in the same sandbox as the REPL and `-p`.
//!
//! The binary runs against an in-test model server, in a temp HOME with a
//! cleared environment. The model asks for one bash call, then answers
//! with the text of the tool result, so stdout shows what the tool
//! returned.

#![cfg(target_os = "macos")]

use axum::Json;
use caliche_sandbox::SANDBOX_UNAVAILABLE;
use caliche_sandbox::paths::git_refusal;
use serde_json::{Value, json};
use std::io::Write;
use std::path::Path;
use std::process::Stdio;
use std::time::Duration;

const WAIT: Duration = Duration::from_secs(60);

/// The bash call the model asks for: a marker in the workspace, and a
/// write into `.git`, which the sandbox denies and the host would allow.
const COMMAND: &str = ": > ran.txt; : > .git/escaped";

/// The first reply is a bash call for [`COMMAND`]; the reply after the
/// tool result repeats that result.
async fn chat_completions(Json(request): Json<Value>) -> Json<Value> {
    let messages = request["messages"].as_array().cloned().unwrap_or_default();
    let tool_result = messages
        .iter()
        .find(|message| message["role"] == "tool")
        .map(|message| message["content"].as_str().unwrap_or_default().to_owned());
    let (message, finish_reason) = match tool_result {
        Some(result) => (
            json!({ "role": "assistant", "content": format!("tool said: {result}") }),
            "stop",
        ),
        None => (
            json!({
                "role": "assistant",
                "content": null,
                "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "bash",
                        "arguments": json!({ "command": COMMAND }).to_string(),
                    },
                }],
            }),
            "tool_calls",
        ),
    };
    Json(json!({
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "created": 1,
        "model": "fake-model",
        "choices": [{ "index": 0, "message": message, "finish_reason": finish_reason }],
        "usage": { "prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2 },
    }))
}

/// Start the model server and return its OpenAI-compatible base URL.
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

/// `echo input | caliche --pipe "run it"` in `workspace`. Returns the exit
/// code and stdout.
async fn pipe(home: &Path, workspace: &Path, base_url: &str) -> anyhow::Result<(i32, String)> {
    let mut command = std::process::Command::new(env!("CARGO_BIN_EXE_caliche"));
    command
        .args(["--pipe", "run it"])
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
    let stdout = String::from_utf8_lossy(&output.stdout).into_owned();
    let stderr = String::from_utf8_lossy(&output.stderr);
    Ok((
        output.status.code().unwrap_or(-1),
        format!("{stdout}{stderr}"),
    ))
}

/// A loose git dir: what git discovery needs to accept `dir`.
fn git_dir_at(dir: &Path) -> anyhow::Result<()> {
    std::fs::create_dir_all(dir.join("objects"))?;
    std::fs::create_dir_all(dir.join("refs").join("heads"))?;
    std::fs::write(dir.join("HEAD"), "ref: refs/heads/main\n")?;
    Ok(())
}

/// A temp HOME and a workspace, both canonical.
fn home_and_workspace()
-> anyhow::Result<(tempfile::TempDir, std::path::PathBuf, std::path::PathBuf)> {
    let dir = tempfile::tempdir()?;
    let base = dir.path().canonicalize()?;
    let home = base.join("home");
    let workspace = base.join("workspace");
    std::fs::create_dir_all(&home)?;
    std::fs::create_dir_all(&workspace)?;
    Ok((dir, home, workspace))
}

/// `.git` is a gitfile naming `meta` inside the workspace, which the
/// sandbox could write, so the sandbox refuses the workspace. The bash
/// call returns the one-line refusal and does not run on the host.
#[tokio::test(flavor = "multi_thread")]
async fn pipe_bash_does_not_run_on_the_host_when_the_sandbox_refuses() -> anyhow::Result<()> {
    let (_dir, home, workspace) = home_and_workspace()?;
    git_dir_at(&workspace.join("meta"))?;
    std::fs::write(workspace.join(".git"), "gitdir: meta\n")?;
    let base_url = fake_model().await?;

    let (code, output) = pipe(&home, &workspace, &base_url).await?;

    let refusal = format!(
        "{SANDBOX_UNAVAILABLE}: {}",
        git_refusal::git_dir_in_workspace(&workspace.join("meta"))
    );
    assert_eq!(code, 0, "{output}");
    assert!(output.contains(&refusal), "{output}");
    assert!(!workspace.join("ran.txt").exists(), "{output}");
    Ok(())
}

/// In a workspace the sandbox accepts, the bash call runs in the sandbox:
/// the marker is written and the write into `.git` is denied.
#[tokio::test(flavor = "multi_thread")]
async fn pipe_bash_runs_in_the_sandbox_when_it_accepts_the_workspace() -> anyhow::Result<()> {
    let (_dir, home, workspace) = home_and_workspace()?;
    git_dir_at(&workspace.join(".git"))?;
    let base_url = fake_model().await?;

    let (code, output) = pipe(&home, &workspace, &base_url).await?;

    assert_eq!(code, 0, "{output}");
    assert!(workspace.join("ran.txt").is_file(), "{output}");
    assert!(!workspace.join(".git").join("escaped").exists(), "{output}");
    Ok(())
}
