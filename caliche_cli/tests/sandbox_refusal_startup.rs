//! The CLI starts and answers a prompt that needs no tool even when the
//! sandbox refuses the workspace's git layout.
//!
//! The binary runs against an in-test model server that answers every
//! request with `done`, in a temp HOME with a cleared environment.

#![cfg(target_os = "macos")]

use axum::Json;
use serde_json::{Value, json};
use std::path::Path;
use std::process::Stdio;
use std::time::Duration;
use tokio::process::Command;

const WAIT: Duration = Duration::from_secs(60);
/// What `caliche -p "say done"` prints against the fake model.
const ANSWER: &str = "Provider: ollama\nModel: fake-model\n\nUser: say done\nAgent: done\n\nToken usage - prompt: 1, completion: 1, total: 2\n";

/// The prompt that makes the fake model ask for one bash call.
const RUN_MARKER: &str = "run the marker";
/// The bash call that prompt asks for: it leaves `ran.txt` in the workspace.
const MARKER_COMMAND: &str = ": > ran.txt";

/// Answers `done`, except that the first reply to [`RUN_MARKER`] is a bash
/// call for [`MARKER_COMMAND`].
async fn chat_completions(Json(request): Json<Value>) -> Json<Value> {
    let messages = request["messages"].as_array().cloned().unwrap_or_default();
    let has_tool_result = messages.iter().any(|message| message["role"] == "tool");
    let asks_for_marker = messages
        .iter()
        .any(|message| message["role"] == "user" && message["content"] == RUN_MARKER);
    let (message, finish_reason) = if asks_for_marker && !has_tool_result {
        (
            json!({
                "role": "assistant",
                "content": null,
                "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "bash",
                        "arguments": json!({ "command": MARKER_COMMAND }).to_string(),
                    },
                }],
            }),
            "tool_calls",
        )
    } else {
        (json!({ "role": "assistant", "content": "done" }), "stop")
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

/// `caliche -p`, run in `workspace`. `--pipe` is covered in
/// `pipe_sandbox.rs`.
async fn prompt(
    home: &Path,
    workspace: &Path,
    base_url: &str,
    text: &str,
) -> anyhow::Result<(i32, String, String)> {
    let child = Command::new(env!("CARGO_BIN_EXE_caliche"))
        .args(["-p", text])
        .current_dir(workspace)
        .env_clear()
        .env("PATH", std::env::var_os("PATH").unwrap_or_default())
        .env("HOME", home)
        .env("AGENT_PROVIDER", "ollama")
        .env("OLLAMA_MODEL", "fake-model")
        .env("OLLAMA_BASE_URL", base_url)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true)
        .spawn()?;
    let output = tokio::time::timeout(WAIT, child.wait_with_output()).await??;
    Ok((
        output.status.code().unwrap_or(-1),
        String::from_utf8_lossy(&output.stdout).into_owned(),
        String::from_utf8_lossy(&output.stderr).into_owned(),
    ))
}

/// A loose git dir: what git discovery needs to accept `dir`.
fn git_dir_at(dir: &Path) -> anyhow::Result<()> {
    std::fs::create_dir_all(dir.join("objects"))?;
    std::fs::create_dir_all(dir.join("refs").join("heads"))?;
    std::fs::write(dir.join("HEAD"), "ref: refs/heads/main\n")?;
    Ok(())
}

/// A temp HOME and a workspace with a git dir of its own.
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

#[tokio::test(flavor = "multi_thread")]
async fn cli_answers_in_a_workspace_the_sandbox_accepts() -> anyhow::Result<()> {
    let (_dir, home, workspace) = home_and_workspace()?;
    git_dir_at(&workspace.join(".git"))?;
    let base_url = fake_model().await?;

    let (code, stdout, stderr) = prompt(&home, &workspace, &base_url, "say done").await?;

    assert_eq!(code, 0, "{stderr}");
    assert_eq!(stdout, ANSWER);
    Ok(())
}

/// `.git` is a gitfile naming `meta` inside the workspace, outside `.git`:
/// the sandbox could write that git dir, so it refuses the workspace. The
/// prompt needs no tool, so the CLI still answers it.
#[tokio::test(flavor = "multi_thread")]
async fn cli_answers_when_the_sandbox_refuses_the_workspace() -> anyhow::Result<()> {
    let (_dir, home, workspace) = home_and_workspace()?;
    git_dir_at(&workspace.join("meta"))?;
    std::fs::write(workspace.join(".git"), "gitdir: meta\n")?;
    let base_url = fake_model().await?;

    let (code, stdout, stderr) = prompt(&home, &workspace, &base_url, "say done").await?;

    assert_eq!(code, 0, "{stderr}");
    assert_eq!(stdout, ANSWER);
    Ok(())
}

/// The same refused workspace: a bash call the model asks for is refused
/// and does not run on the host, and the CLI still ends the turn.
#[tokio::test(flavor = "multi_thread")]
async fn bash_does_not_run_on_the_host_when_the_sandbox_refuses() -> anyhow::Result<()> {
    let (_dir, home, workspace) = home_and_workspace()?;
    git_dir_at(&workspace.join("meta"))?;
    std::fs::write(workspace.join(".git"), "gitdir: meta\n")?;
    let base_url = fake_model().await?;

    let (code, stdout, stderr) = prompt(&home, &workspace, &base_url, RUN_MARKER).await?;

    assert_eq!(code, 0, "{stderr}");
    assert!(stdout.contains("Agent: done\n"), "{stdout}");
    assert!(!workspace.join("ran.txt").exists());
    Ok(())
}

/// Control for the test above: in a workspace the sandbox accepts, the same
/// bash call runs and leaves its marker.
#[tokio::test(flavor = "multi_thread")]
async fn bash_runs_in_the_sandbox_when_it_accepts_the_workspace() -> anyhow::Result<()> {
    let (_dir, home, workspace) = home_and_workspace()?;
    git_dir_at(&workspace.join(".git"))?;
    let base_url = fake_model().await?;

    let (code, stdout, stderr) = prompt(&home, &workspace, &base_url, RUN_MARKER).await?;

    assert_eq!(code, 0, "{stderr}");
    assert!(stdout.contains("Agent: done\n"), "{stdout}");
    assert!(workspace.join("ran.txt").is_file());
    Ok(())
}
